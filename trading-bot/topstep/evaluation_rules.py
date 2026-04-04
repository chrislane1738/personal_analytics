"""Topstep evaluation rules — pass/fail/timeout logic with trailing drawdown and consistency."""

from datetime import date
from enum import Enum, auto

from topstep.config import TopstepConfig


class EvalOutcome(Enum):
    ACTIVE = auto()
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()


class EvaluationRules:
    """Evaluates each trading day against Topstep rules.

    Tracks:
    - Trailing drawdown floor (moves up with eod_high_water, never down)
    - Consistency check: best single day <= 50% of total positive profit
    - Days traded vs max_attempt_days timeout
    - Profit target
    """

    def __init__(self, config: TopstepConfig) -> None:
        self.config = config
        self._initial_floor = config.account_size - config.max_loss
        self.eod_high_water: float = config.account_size
        self.drawdown_floor: float = self._initial_floor
        self.total_profit: float = 0.0
        self.best_day_pnl: float = 0.0
        self.days_traded: int = 0
        self.daily_pnls: list[float] = []

    @property
    def consistency_satisfied(self) -> bool:
        """True when total_profit > 0 and best_day_pnl <= consistency_pct * total_profit."""
        if self.total_profit <= 0:
            return False
        return self.best_day_pnl <= self.config.consistency_pct * self.total_profit

    def evaluate_day(self, day_pnl: float, eod_balance: float) -> EvalOutcome:
        """Record a trading day and return the resulting outcome."""
        self.days_traded += 1
        self.daily_pnls.append(day_pnl)

        # Update high-water mark and trailing drawdown floor
        if eod_balance > self.eod_high_water:
            self.eod_high_water = eod_balance
            self.drawdown_floor = self.eod_high_water - self.config.max_loss

        # Accumulate positive days only for consistency
        if day_pnl > 0:
            self.total_profit += day_pnl
            if day_pnl > self.best_day_pnl:
                self.best_day_pnl = day_pnl

        # Check fail condition first (drawdown breach)
        if eod_balance <= self.drawdown_floor:
            return EvalOutcome.FAIL

        # Check timeout
        if self.days_traded >= self.config.max_attempt_days:
            return EvalOutcome.TIMEOUT

        # Check pass condition
        cumulative = eod_balance - self.config.account_size
        if cumulative >= self.config.profit_target and self.consistency_satisfied:
            return EvalOutcome.PASS

        return EvalOutcome.ACTIVE

    def reset(self) -> None:
        """Reset all state for a new attempt."""
        self.eod_high_water = self.config.account_size
        self.drawdown_floor = self._initial_floor
        self.total_profit = 0.0
        self.best_day_pnl = 0.0
        self.days_traded = 0
        self.daily_pnls = []


# ---------------------------------------------------------------------------
# Funded-mode rules — no profit target, no timeout, no consistency.
# ---------------------------------------------------------------------------


class FundedOutcome(Enum):
    ACTIVE = auto()
    BLOWN = auto()  # trailing drawdown breached


class FundedRules:
    """Evaluate each trading day against funded-account rules.

    Funded accounts have only one failure mode: the trailing drawdown
    floor is breached.  There is no profit target, no timeout, and no
    consistency requirement.
    """

    def __init__(self, config: TopstepConfig) -> None:
        self.config = config
        self.eod_high_water: float = config.account_size
        self.drawdown_floor: float = config.account_size - config.max_loss

    def evaluate_day(
        self, day_pnl: float, eod_balance: float, cal_date: date,
    ) -> FundedOutcome:
        """Record a trading day and return the resulting outcome."""
        # Update high-water mark and trailing drawdown floor
        if eod_balance > self.eod_high_water:
            self.eod_high_water = eod_balance
            self.drawdown_floor = self.eod_high_water - self.config.max_loss

        # Check drawdown breach
        if eod_balance < self.drawdown_floor:
            return FundedOutcome.BLOWN

        return FundedOutcome.ACTIVE
