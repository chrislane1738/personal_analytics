"""Topstep per-attempt state tracker combining EvaluationRules and StateManager."""

from enum import Enum, auto

from topstep.config import TopstepConfig
from topstep.evaluation_rules import EvaluationRules, EvalOutcome
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
