"""Topstep evaluation rules engine.

Tracks pass/fail state across trading days, including:
- Trailing end-of-day (EOD) drawdown floor
- Cumulative profit target
- Consistency constraint (no single day > consistency_pct of total profit)
- Timeout after max_attempt_days
"""

from enum import Enum, auto

from topstep.config import TopstepConfig


class EvalOutcome(Enum):
    ACTIVE = auto()
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()


class EvaluationRules:
    """Stateful evaluation rules engine for a single Topstep evaluation attempt."""

    def __init__(self, config: TopstepConfig) -> None:
        self.config = config
        self.eod_high_water: float = config.account_size
        self.drawdown_floor: float = config.account_size - config.max_loss
        self.total_profit: float = 0.0
        self.best_day_pnl: float = 0.0
        self.days_traded: int = 0
        self.daily_pnls: list[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate_day(self, day_pnl: float, eod_balance: float) -> EvalOutcome:
        """Call at end of each trading day.

        Args:
            day_pnl: Profit/loss for the day (negative for losses).
            eod_balance: Absolute account balance at end of day.

        Returns:
            The current EvalOutcome after processing this day's data.
        """
        # 1. Increment day counter and record pnl
        self.days_traded += 1
        self.daily_pnls.append(day_pnl)

        # 2. Update total_profit (only positive days) and best_day_pnl
        if day_pnl > 0:
            self.total_profit += day_pnl
            if day_pnl > self.best_day_pnl:
                self.best_day_pnl = day_pnl

        # 3. Update trailing drawdown: if eod_balance > eod_high_water, move both up
        if eod_balance > self.eod_high_water:
            self.eod_high_water = eod_balance
            self.drawdown_floor = eod_balance - self.config.max_loss

        # 4. FAIL if eod_balance <= drawdown_floor
        if eod_balance <= self.drawdown_floor:
            return EvalOutcome.FAIL

        # 5. PASS if cumulative_pnl >= profit_target AND consistency_satisfied
        cumulative_pnl = eod_balance - self.config.account_size
        if cumulative_pnl >= self.config.profit_target and self.consistency_satisfied:
            return EvalOutcome.PASS

        # 6. TIMEOUT if days_traded >= max_attempt_days
        if self.days_traded >= self.config.max_attempt_days:
            return EvalOutcome.TIMEOUT

        # 7. Otherwise ACTIVE
        return EvalOutcome.ACTIVE

    @property
    def consistency_satisfied(self) -> bool:
        """True when no single day's profit exceeds consistency_pct of total profit.

        Returns False when total_profit <= 0 (no profitable days yet).
        """
        if self.total_profit <= 0:
            return False
        return self.best_day_pnl <= self.config.consistency_pct * self.total_profit

    def reset(self) -> None:
        """Reset all state for a new evaluation attempt using the same config."""
        self.eod_high_water = self.config.account_size
        self.drawdown_floor = self.config.account_size - self.config.max_loss
        self.total_profit = 0.0
        self.best_day_pnl = 0.0
        self.days_traded = 0
        self.daily_pnls = []
