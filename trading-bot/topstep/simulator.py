"""Topstep evaluation simulator — wraps BacktestEngine to run a single attempt."""

import threading
from datetime import date, datetime, timedelta

from core.engine import BacktestEngine
from core.event_bus import EventBus
from execution.futures_sim_broker import FuturesSimBroker
from topstep.attempt_tracker import (
    AttemptTracker,
    AttemptStatus,
    FundedAttemptTracker,
    FundedAttemptStatus,
)
from topstep.config import TopstepConfig, TradingMode
from topstep.futures_specs import FUTURES_SPECS


class TopstepEvalSimulator:
    """Run a single Topstep evaluation attempt using the BacktestEngine.

    Creates a fresh backtest over the given date range, processes the daily
    equity curve into per-day P&L, and feeds each day into an AttemptTracker
    that applies the evaluation rules (profit target, trailing drawdown,
    consistency, timeout).

    Parameters
    ----------
    strategy:
        A Strategy instance to trade with.
    database:
        Database instance with bar data loaded.
    instrument:
        Symbol to trade (e.g. "ES").
    config:
        TopstepConfig with account size, profit target, max loss, etc.
    state_machine_enabled:
        Whether the AttemptTracker should use the state machine for
        position/stop multipliers.
    timeframe:
        Bar timeframe for the BacktestEngine (``"1D"`` for daily,
        ``"5m"`` / ``"1m"`` for intraday).
    """

    def __init__(
        self,
        strategy,
        database,
        instrument: str,
        config: TopstepConfig,
        state_machine_enabled: bool = True,
        timeframe: str = "1D",
    ) -> None:
        self.strategy = strategy
        self.database = database
        self.instrument = instrument
        self.config = config
        self.timeframe = timeframe
        self.state_machine_enabled = state_machine_enabled
        self._cancel = threading.Event()

    def run_attempt(self, start_date: date) -> dict:
        """Execute a single attempt and return the tracker result.

        Dispatches to the correct internal method based on ``config.mode``.
        """
        if self.config.mode == TradingMode.FUNDED:
            return self._run_funded_attempt(start_date)
        return self._run_eval_attempt(start_date)

    # ------------------------------------------------------------------
    # Eval-mode attempt (original logic, unchanged behaviour)
    # ------------------------------------------------------------------

    def _run_eval_attempt(self, start_date: date) -> dict:
        """Execute a single evaluation attempt and return the tracker result.

        The engine runs from ``start_date`` through a generous end date
        (2x ``max_attempt_days``) so there is enough data for the attempt.
        After the engine finishes, daily equity snapshots are converted to
        P&L and fed into the AttemptTracker day by day.

        Returns
        -------
        dict
            Serialised tracker state from ``AttemptTracker.to_dict()``.
        """
        tracker = AttemptTracker(
            self.config, state_machine_enabled=self.state_machine_enabled,
        )
        end_date = start_date + timedelta(days=self.config.max_attempt_days * 2)

        engine = self._build_engine(start_date, end_date)
        engine.run()

        equity_curve = engine.portfolio.equity_curve
        if equity_curve:
            prev_equity = self.config.account_size
            for eq_date, equity in equity_curve:
                day_pnl = equity - prev_equity
                status = tracker.record_day(pnl=day_pnl, eod_balance=equity)
                prev_equity = equity
                if status != AttemptStatus.ACTIVE:
                    break

        return tracker.to_dict()

    # ------------------------------------------------------------------
    # Funded-mode attempt
    # ------------------------------------------------------------------

    def _run_funded_attempt(self, start_date: date) -> dict:
        """Execute a single funded-mode attempt and return the tracker result.

        Uses ``config.funded_sim_days`` to determine the simulation window,
        funded-specific position sizing (``position_size_pct``,
        ``max_contracts``), signal-strength sizing, and the daily-loss-limit
        rule when configured.
        """
        tracker = FundedAttemptTracker(
            self.config, state_machine_enabled=self.state_machine_enabled,
        )
        end_date = start_date + timedelta(days=int(self.config.funded_sim_days * 1.5))

        engine = self._build_engine(
            start_date,
            end_date,
            position_size_pct=self.config.position_size_pct,
            daily_loss_limit=self.config.daily_loss_limit,
            use_signal_strength_sizing=True,
            max_contracts=self.config.max_contracts,
        )
        engine.run()

        equity_curve = engine.portfolio.equity_curve
        if equity_curve:
            prev_equity = self.config.account_size
            for eq_date, equity in equity_curve:
                # Resolve calendar date for monthly bucketing
                if isinstance(eq_date, datetime):
                    cal_date = eq_date.date()
                else:
                    cal_date = eq_date

                day_pnl = equity - prev_equity
                status = tracker.record_day(
                    pnl=day_pnl, eod_balance=equity, cal_date=cal_date,
                )
                prev_equity = equity
                if status != FundedAttemptStatus.ACTIVE:
                    break

        return tracker.to_dict()

    # ------------------------------------------------------------------
    # Shared engine builder
    # ------------------------------------------------------------------

    def _build_engine(
        self,
        start_date: date,
        end_date: date,
        position_size_pct: float = 0.25,
        daily_loss_limit: float = 0,
        use_signal_strength_sizing: bool = False,
        max_contracts: int = 0,
    ) -> BacktestEngine:
        """Construct a BacktestEngine with a futures broker when applicable."""
        spec = FUTURES_SPECS.get(self.instrument)

        futures_bus = EventBus()
        broker = None
        contract_multipliers: dict[str, float] | None = None

        if spec is not None:
            if spec.contract_type == "micro":
                max_pos = self.config.max_position_micros
            else:
                max_pos = self.config.max_position_minis

            broker = FuturesSimBroker(
                event_bus=futures_bus,
                contract_spec=spec,
                slippage_pct=0.0001,
                max_position=max_pos,
            )
            contract_multipliers = {self.instrument: spec.multiplier}

        engine = BacktestEngine(
            strategy=self.strategy,
            database=self.database,
            universe=[self.instrument],
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.account_size,
            benchmark_symbol=self.instrument,
            slippage_pct=0.0001,
            commission_per_share=0.005,
            position_size_pct=position_size_pct,
            cancel_event=self._cancel,
            quiet=True,
            broker=broker,
            contract_multipliers=contract_multipliers,
            force_flat_daily=self.config.force_flat_daily,
            timeframe=self.timeframe,
            daily_loss_limit=daily_loss_limit,
            use_signal_strength_sizing=use_signal_strength_sizing,
            max_contracts=max_contracts,
        )

        if broker is not None:
            broker.event_bus = engine.event_bus

        return engine
