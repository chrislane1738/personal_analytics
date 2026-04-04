"""Topstep per-attempt state tracker combining EvaluationRules and StateManager."""

from datetime import date
from enum import Enum, auto

from topstep.config import TopstepConfig, TradingMode
from topstep.evaluation_rules import EvaluationRules, EvalOutcome, FundedRules, FundedOutcome
from topstep.state_manager import StateManager, EvalState


class AttemptStatus(Enum):
    ACTIVE = auto()
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()


_OUTCOME_TO_STATUS = {
    EvalOutcome.ACTIVE: AttemptStatus.ACTIVE,
    EvalOutcome.PASS: AttemptStatus.PASS,
    EvalOutcome.FAIL: AttemptStatus.FAIL,
    EvalOutcome.TIMEOUT: AttemptStatus.TIMEOUT,
}


class AttemptTracker:
    """Tracks a single Topstep evaluation attempt end-to-end.

    Combines EvaluationRules (pass/fail/timeout logic) with StateManager
    (cumulative-PnL risk states) to produce a unified per-day snapshot.
    """

    def __init__(self, config: TopstepConfig, state_machine_enabled: bool = True) -> None:
        self.config = config
        self.rules = EvaluationRules(config)
        self.state_manager = StateManager(enabled=state_machine_enabled)
        self.cumulative_pnl: float = 0.0
        self.daily_pnls: list[float] = []
        self.state_history: list[EvalState] = []
        self.status: AttemptStatus = AttemptStatus.ACTIVE

    @property
    def days_traded(self) -> int:
        return self.rules.days_traded

    def record_day(self, pnl: float, eod_balance: float) -> AttemptStatus:
        """Record end-of-day PnL and balance, returning the updated attempt status."""
        self.cumulative_pnl += pnl
        self.daily_pnls.append(pnl)
        self.state_history.append(self.state_manager.get_state(self.cumulative_pnl))
        outcome = self.rules.evaluate_day(day_pnl=pnl, eod_balance=eod_balance)
        self.status = _OUTCOME_TO_STATUS[outcome]
        return self.status

    def to_dict(self) -> dict:
        """Serialise current tracker state to a plain dictionary."""
        return {
            "cumulative_pnl": self.cumulative_pnl,
            "days_traded": self.days_traded,
            "daily_pnls": self.daily_pnls,
            "best_day_pnl": self.rules.best_day_pnl,
            "worst_day_pnl": min(self.daily_pnls) if self.daily_pnls else 0.0,
            "status": self.status.name.lower(),
            "state_history": [s.name for s in self.state_history],
        }


# ---------------------------------------------------------------------------
# Funded-mode attempt tracker
# ---------------------------------------------------------------------------


class FundedAttemptStatus(Enum):
    ACTIVE = auto()
    BLOWN = auto()


class FundedAttemptTracker:
    """Tracks a single funded-account simulation end-to-end.

    Combines FundedRules (trailing drawdown only) with StateManager
    (funded-mode conservative states) to produce a unified per-day snapshot.
    Also tracks monthly P&L bucketing, consecutive losing days, and peak
    drawdown for downstream analytics.
    """

    def __init__(
        self, config: TopstepConfig, state_machine_enabled: bool = True,
    ) -> None:
        self.config = config
        self.rules = FundedRules(config)
        self.state_manager = StateManager(
            enabled=state_machine_enabled,
            mode=TradingMode.FUNDED,
        )
        self.cumulative_pnl: float = 0.0
        self.daily_pnls: list[float] = []
        self.monthly_pnls: dict[str, float] = {}  # "YYYY-MM" -> pnl
        self.state_history: list[str] = []
        self.status: FundedAttemptStatus = FundedAttemptStatus.ACTIVE
        self.days_survived: int = 0
        self.peak_drawdown: float = 0.0
        self._consecutive_losing: int = 0
        self.max_consecutive_losing_days: int = 0

    @property
    def days_traded(self) -> int:
        return self.days_survived

    def record_day(
        self, pnl: float, eod_balance: float, cal_date: date,
    ) -> FundedAttemptStatus:
        """Record end-of-day PnL/balance and return the updated status."""
        if self.status != FundedAttemptStatus.ACTIVE:
            return self.status

        self.cumulative_pnl += pnl
        self.daily_pnls.append(pnl)
        self.days_survived += 1

        # Monthly bucketing
        month_key = cal_date.strftime("%Y-%m")
        self.monthly_pnls[month_key] = self.monthly_pnls.get(month_key, 0.0) + pnl

        # Track consecutive losing days
        if pnl < 0:
            self._consecutive_losing += 1
            self.max_consecutive_losing_days = max(
                self.max_consecutive_losing_days, self._consecutive_losing,
            )
        else:
            self._consecutive_losing = 0

        # Track peak drawdown (most negative value of cumulative vs high-water)
        drawdown = (
            self.config.account_size + self.cumulative_pnl
            - self.rules.eod_high_water
        )
        if drawdown < self.peak_drawdown:
            self.peak_drawdown = drawdown

        # State tracking
        state = self.state_manager.get_state(self.cumulative_pnl)
        self.state_history.append(state.name)

        # Evaluate rules
        outcome = self.rules.evaluate_day(pnl, eod_balance, cal_date)
        if outcome == FundedOutcome.BLOWN:
            self.status = FundedAttemptStatus.BLOWN

        return self.status

    def to_dict(self) -> dict:
        """Serialise current tracker state to a plain dictionary."""
        return {
            "status": "active" if self.status == FundedAttemptStatus.ACTIVE else "blown",
            "cumulative_pnl": self.cumulative_pnl,
            "days_traded": self.days_survived,
            "daily_pnls": self.daily_pnls,
            "monthly_pnls": self.monthly_pnls,
            "state_history": self.state_history,
            "peak_drawdown": self.peak_drawdown,
            "max_consecutive_losing_days": self.max_consecutive_losing_days,
        }
