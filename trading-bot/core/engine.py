"""Backtest engine: the core orchestrator that wires together all components.

DataFeed -> EventBus -> Strategy -> RiskManager -> Broker -> Portfolio -> Analytics

For each trading date the engine collects bars, feeds them through the
pipeline, and records everything.  At the end it computes metrics, prints a
console report, and optionally persists the run to the database.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

from analytics.metrics import compute_all_metrics
from analytics.reports import ReportGenerator
from analytics.trade_log import TradeLog
from core.event_bus import EventBus
from core.events import BarEvent, EventType, FillEvent, SignalEvent
from data.storage.models import RunRecord
from execution.sim_broker import SimBroker
from portfolio.benchmark import BenchmarkTracker
from portfolio.portfolio import Portfolio
from risk.manager import RiskManager
from strategy.base import BacktestContext


class BacktestEngine:
    """Run a full backtest over historical data stored in the database.

    Parameters
    ----------
    strategy:
        An instance of a Strategy subclass.
    database:
        Database object with ``get_daily_bars`` / ``insert_run`` methods.
    universe:
        List of ticker symbols to trade.
    start_date / end_date:
        Date range for the backtest.
    initial_capital:
        Starting cash (default $100 000).
    benchmark_symbol:
        Symbol used for buy-and-hold comparison (default ``SPY``).
    risk_rules:
        List of RiskRule instances passed to the RiskManager.
    sector_map:
        Symbol -> sector mapping for sector-concentration rules.
    slippage_pct / commission_per_share:
        Execution-cost parameters forwarded to the SimBroker.
    position_size_pct:
        Fraction of equity allocated per position.
    """

    def __init__(
        self,
        strategy,
        database,
        universe: list[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100_000.0,
        benchmark_symbol: str = "SPY",
        risk_rules: list | None = None,
        sector_map: dict | None = None,
        slippage_pct: float = 0.0001,
        commission_per_share: float = 0.005,
        position_size_pct: float = 0.06,
    ) -> None:
        self.strategy = strategy
        self.database = database
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol
        self.position_size_pct = position_size_pct

        # --- Create components ---
        self.event_bus = EventBus()
        self.portfolio = Portfolio(initial_capital, self.event_bus)
        self.broker = SimBroker(self.event_bus, slippage_pct, commission_per_share)
        self.risk_manager = RiskManager(
            risk_rules or [], self.event_bus, sector_map or {}
        )
        self.trade_log = TradeLog()
        self.benchmark = BenchmarkTracker(benchmark_symbol, initial_capital)

        # Wire up event handlers
        self.event_bus.subscribe(EventType.FILL, self._on_fill)

        # Pre-loaded bar data: symbol -> {date -> DailyBar}
        self._bar_data: dict[str, dict[date, object]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the backtest and return a metrics dict."""
        run_id = str(uuid.uuid4())[:8]

        # Load data from database
        self._load_data()

        # Build sorted date list from available data
        all_dates: set[date] = set()
        for symbol_bars in self._bar_data.values():
            all_dates.update(symbol_bars.keys())
        sorted_dates = sorted(
            d for d in all_dates if self.start_date <= d <= self.end_date
        )

        if not sorted_dates:
            # No data — return empty metrics gracefully
            return self._empty_metrics(run_id)

        # Initialise strategy
        context = BacktestContext(
            start_date=self.start_date,
            end_date=self.end_date,
            universe=self.universe,
            initial_capital=self.initial_capital,
            benchmark=self.benchmark_symbol,
        )
        self.strategy.on_start(context)

        warm_up = self.strategy.warm_up_period()
        current_prices: dict[str, float] = {}

        # ---- Main simulation loop ----
        for i, current_date in enumerate(sorted_dates):
            # Collect bars for all symbols on this date
            bars: dict[str, BarEvent] = {}
            for symbol in self.universe:
                bar_data = self._bar_data.get(symbol, {}).get(current_date)
                if bar_data:
                    bar_event = BarEvent(
                        symbol=symbol,
                        date=current_date,
                        open=bar_data.open,
                        high=bar_data.high,
                        low=bar_data.low,
                        close=bar_data.close,
                        adj_close=bar_data.adj_close,
                        volume=bar_data.volume,
                        vwap=bar_data.vwap,
                    )
                    bars[symbol] = bar_event
                    current_prices[symbol] = bar_data.close

                    # Fill pending orders for this symbol against this bar
                    self.broker.process_bar(
                        symbol,
                        bar_data.open,
                        bar_data.high,
                        bar_data.low,
                        bar_data.close,
                        bar_data.volume,
                    )

            # Update benchmark
            benchmark_bar = self._bar_data.get(
                self.benchmark_symbol, {}
            ).get(current_date)
            if benchmark_bar:
                self.benchmark.update(current_date, benchmark_bar.close)

            # Warm-up period: feed bars to strategy (for indicator priming)
            # but do NOT generate actionable signals
            if i < warm_up:
                if bars:
                    self.strategy.on_universe_bar(bars, self.portfolio)
                continue

            # Generate signals
            if bars:
                signals = self.strategy.on_universe_bar(bars, self.portfolio)

                # Evaluate each signal through the risk manager
                for signal in signals:
                    equity = self.portfolio.get_equity(current_prices)
                    price = current_prices.get(signal.symbol, 1)
                    risk_context = {
                        "equity": equity,
                        "current_prices": current_prices,
                        "position_size_pct": self.position_size_pct,
                        "default_quantity": int(
                            equity * self.position_size_pct / price
                        )
                        if price > 0
                        else 0,
                    }
                    order = self.risk_manager.evaluate_signal(
                        signal, self.portfolio, risk_context
                    )
                    if order:
                        self.broker.submit_order(order)

            # Record daily equity snapshot
            self.portfolio.record_equity(current_date, current_prices)

        # ---- End of backtest ----
        self.strategy.on_end(self.portfolio)

        # Close any remaining open trades in the trade log
        for symbol in list(self.trade_log._open_trades.keys()):
            last_price = current_prices.get(symbol, 0)
            self.trade_log.close_trade(
                symbol,
                sorted_dates[-1] if sorted_dates else self.end_date,
                last_price,
                "End of backtest",
            )

        # Compute metrics
        metrics = compute_all_metrics(
            equity_curve=self.portfolio.equity_curve,
            trades=self.trade_log.get_trade_dicts(),
            benchmark_curve=self.benchmark.equity_curve,
            risk_free_rate=0.04,
            initial_capital=self.initial_capital,
        )
        metrics["start_date"] = str(self.start_date)
        metrics["end_date"] = str(self.end_date)
        metrics["total_trades"] = len(self.trade_log.trades)

        # Console report
        report = ReportGenerator(
            run_id=run_id,
            strategy_name=type(self.strategy).__name__,
            metrics=metrics,
            trade_log=self.trade_log,
            equity_curve=self.portfolio.equity_curve,
            benchmark_curve=self.benchmark.equity_curve,
        )
        console_output = report.generate_console_report()
        print(console_output)

        # Persist run record
        self._save_run(run_id, metrics)

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load all bar data from the database into memory."""
        symbols = list(self.universe)
        if self.benchmark_symbol not in symbols:
            symbols.append(self.benchmark_symbol)

        for symbol in symbols:
            bars = self.database.get_daily_bars(
                symbol, self.start_date, self.end_date
            )
            self._bar_data[symbol] = {bar.date: bar for bar in bars}

    def _on_fill(self, fill: FillEvent) -> None:
        """Handle fill events: update portfolio and trade log."""
        current_date = (
            fill.timestamp.date()
            if hasattr(fill, "timestamp") and fill.timestamp
            else date.today()
        )

        self.portfolio.update_on_fill(fill, current_date)

        # Track in trade log
        if fill.action in ("BUY", "BTO"):
            self.trade_log.open_trade(
                fill.symbol,
                "long",
                current_date,
                fill.fill_price,
                fill.quantity,
                f"Signal filled at ${fill.fill_price:.2f}",
            )
        elif fill.action in ("SELL", "STC"):
            self.trade_log.close_trade(
                fill.symbol,
                current_date,
                fill.fill_price,
                f"Exit filled at ${fill.fill_price:.2f}",
            )
        elif fill.action in ("STO",):
            self.trade_log.open_trade(
                fill.symbol,
                "sell_put",
                current_date,
                fill.fill_price,
                fill.quantity,
                f"Premium collected at ${fill.fill_price:.2f}",
            )
        elif fill.action in ("BTC",):
            self.trade_log.close_trade(
                fill.symbol,
                current_date,
                fill.fill_price,
                f"Closed at ${fill.fill_price:.2f}",
            )

        self.strategy.on_fill(fill)

    def _save_run(self, run_id: str, metrics: dict) -> None:
        """Persist backtest results to the database."""
        run = RunRecord(
            run_id=run_id,
            mode="backtest",
            strategy_name=type(self.strategy).__name__,
            config="",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_value=metrics.get("end_value", 0),
            total_return=metrics.get("total_return", 0),
            sharpe=metrics.get("sharpe_ratio", 0),
            max_drawdown=metrics.get("max_drawdown_pct", 0),
            created_at=datetime.now(tz=timezone.utc),
            full_metrics=json.dumps(
                {k: str(v) for k, v in metrics.items()}
            ),
        )
        try:
            self.database.insert_run(run)
        except Exception:
            pass  # Don't crash on DB save failure

    def _empty_metrics(self, run_id: str) -> dict:
        """Return a minimal metrics dict when no data is available."""
        metrics: dict = {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "total_trades": 0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "end_value": self.initial_capital,
            "start_value": self.initial_capital,
        }
        self._save_run(run_id, metrics)
        return metrics
