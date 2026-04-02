"""Topstep evaluation simulator — wraps BacktestEngine to run a single attempt."""

import threading
from datetime import date, timedelta

from core.engine import BacktestEngine
from topstep.attempt_tracker import AttemptTracker, AttemptStatus
from topstep.config import TopstepConfig


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
    """

    def __init__(
        self,
        strategy,
        database,
        instrument: str,
        config: TopstepConfig,
        state_machine_enabled: bool = True,
    ) -> None:
        self.strategy = strategy
        self.database = database
        self.instrument = instrument
        self.config = config
        self.tracker = AttemptTracker(config, state_machine_enabled=state_machine_enabled)
        self._cancel = threading.Event()

    def run_attempt(self, start_date: date) -> dict:
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
        end_date = start_date + timedelta(days=self.config.max_attempt_days * 2)

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
            position_size_pct=0.15,  # Higher than equity default to ensure >= 1 contract for futures
            cancel_event=self._cancel,
            quiet=True,
        )

        engine.run()

        # Process equity curve into daily P&L.
        # engine.portfolio.equity_curve is a list of (date, equity) tuples
        # recorded at the end of each trading day by the engine.
        equity_curve = engine.portfolio.equity_curve

        if equity_curve:
            prev_equity = self.config.account_size
            for eq_date, equity in equity_curve:
                day_pnl = equity - prev_equity
                status = self.tracker.record_day(pnl=day_pnl, eod_balance=equity)
                prev_equity = equity
                if status != AttemptStatus.ACTIVE:
                    break

        return self.tracker.to_dict()
