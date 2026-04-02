"""Campaign runner — executes N independent evaluation attempts with random start dates."""

import random
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

from data.storage.database import Database
from topstep.config import TopstepConfig
from topstep.simulator import TopstepEvalSimulator
from topstep.state_manager import StateManager


@dataclass
class CampaignResult:
    campaign_id: str
    strategy_name: str
    instrument: str
    state_machine_enabled: bool
    num_attempts: int
    seed: int

    # Pass-rate metrics
    pass_rate: float = 0.0
    avg_days_to_pass: float = 0.0
    avg_days_to_fail: float = 0.0
    ev_per_attempt: float = 0.0
    cost_to_funded: float = 0.0
    median_attempts_to_pass: float = 0.0
    annual_ev: float = 0.0

    # Distribution data
    attempt_outcomes: list[dict] = field(default_factory=list)
    pnl_distribution: list[float] = field(default_factory=list)
    days_distribution: list[int] = field(default_factory=list)
    state_usage: dict[str, float] = field(default_factory=dict)
    pass_by_regime: dict[str, float] = field(default_factory=dict)


class CampaignRunner:
    """Run N independent Topstep evaluation attempts with random start dates.

    Each attempt uses a fresh Strategy instance and a fresh TopstepEvalSimulator.
    Start dates are drawn uniformly at random from the available data range
    (with headroom so each attempt has enough data to run to completion).

    Parameters
    ----------
    strategy_class:
        Strategy class (not instance) to instantiate for each attempt.
    instrument:
        Symbol to trade (e.g. "ES").
    config:
        TopstepConfig with account size, profit target, max loss, etc.
    database:
        Database instance with bar data loaded.
    state_machine_enabled:
        Whether the StateManager is active during each attempt.
    num_attempts:
        Number of independent attempts to run.
    seed:
        RNG seed for reproducible start-date selection.
    """

    def __init__(
        self,
        strategy_class,
        instrument: str,
        config: TopstepConfig,
        database: Database,
        state_machine_enabled: bool = True,
        num_attempts: int = 1000,
        seed: int = 42,
    ) -> None:
        self.strategy_class = strategy_class
        self.instrument = instrument
        self.config = config
        self.database = database
        self.state_machine_enabled = state_machine_enabled
        self.num_attempts = num_attempts
        self.seed = seed

    def run(self) -> CampaignResult:
        """Execute all attempts and return aggregated results."""
        rng = random.Random(self.seed)

        # Get available date range
        date_range = self.database.get_cached_date_range(self.instrument)
        if date_range == (None, None):
            raise ValueError(f"No data for {self.instrument}")
        min_date, max_date = date_range

        # Leave enough headroom so each attempt has data to run to completion
        headroom = timedelta(days=self.config.max_attempt_days * 2)
        latest_start = max_date - headroom
        if latest_start <= min_date:
            raise ValueError("Not enough data")

        date_range_days = (latest_start - min_date).days
        start_dates = [
            min_date + timedelta(days=rng.randint(0, date_range_days))
            for _ in range(self.num_attempts)
        ]

        outcomes = []
        for start in start_dates:
            strategy = self.strategy_class()
            if self.state_machine_enabled and hasattr(strategy, "_state_manager"):
                strategy._state_manager = StateManager(enabled=True)

            sim = TopstepEvalSimulator(
                strategy=strategy,
                database=self.database,
                instrument=self.instrument,
                config=self.config,
                state_machine_enabled=self.state_machine_enabled,
            )
            result = sim.run_attempt(start_date=start)
            outcomes.append(result)

        return self._aggregate(outcomes)

    def _aggregate(self, outcomes: list[dict]) -> CampaignResult:
        """Aggregate raw attempt outcomes into a CampaignResult."""
        passes = [o for o in outcomes if o["status"] == "pass"]
        fails = [o for o in outcomes if o["status"] == "fail"]

        pass_rate = len(passes) / len(outcomes) if outcomes else 0.0
        avg_days_to_pass = (
            sum(o["days_traded"] for o in passes) / len(passes) if passes else 0.0
        )
        avg_days_to_fail = (
            sum(o["days_traded"] for o in fails) / len(fails) if fails else 0.0
        )

        gross_payout = min(
            self.config.profit_target * self.config.payout_split,
            self.config.max_payout,
        )
        ev_per_attempt = (
            pass_rate * (gross_payout - self.config.activation_fee)
            - self.config.subscription_fee
        )
        cost_to_funded = (
            self.config.subscription_fee / pass_rate if pass_rate > 0 else float("inf")
        )
        median_attempts = 1.0 / pass_rate if pass_rate > 0 else float("inf")
        annual_ev = ev_per_attempt * 4 * 12  # 4 attempts/month

        # Aggregate state history across all attempts
        all_states: list[str] = []
        for o in outcomes:
            all_states.extend(o.get("state_history", []))
        state_counts: dict[str, int] = {}
        for s in all_states:
            state_counts[s] = state_counts.get(s, 0) + 1
        total_states = len(all_states) or 1
        state_usage = {k: v / total_states for k, v in state_counts.items()}

        return CampaignResult(
            campaign_id=str(uuid.uuid4())[:8],
            strategy_name=self.strategy_class.__name__,
            instrument=self.instrument,
            state_machine_enabled=self.state_machine_enabled,
            num_attempts=len(outcomes),
            seed=self.seed,
            pass_rate=pass_rate,
            avg_days_to_pass=avg_days_to_pass,
            avg_days_to_fail=avg_days_to_fail,
            ev_per_attempt=ev_per_attempt,
            cost_to_funded=cost_to_funded,
            median_attempts_to_pass=median_attempts,
            annual_ev=annual_ev,
            attempt_outcomes=outcomes,
            pnl_distribution=[o["cumulative_pnl"] for o in outcomes],
            days_distribution=[o["days_traded"] for o in outcomes],
            state_usage=state_usage,
            pass_by_regime={},
        )
