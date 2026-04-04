"""Campaign runner — executes N independent evaluation attempts with random start dates."""

import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

from data.storage.database import Database
from topstep.config import TopstepConfig, TradingMode
from topstep.simulator import TopstepEvalSimulator
from topstep.state_manager import StateManager


@dataclass
class CampaignResult:
    campaign_id: str
    strategy_name: str
    instrument: str
    timeframe: str
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


@dataclass
class FundedCampaignResult:
    """Aggregated metrics for a funded-mode simulation campaign."""

    campaign_id: str
    strategy_name: str
    instrument: str
    timeframe: str
    state_machine_enabled: bool
    num_attempts: int
    seed: int
    sim_months: int = 12

    # Survival / income metrics
    survival_rate: float = 0.0
    avg_monthly_pnl: float = 0.0
    median_monthly_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_median: float = 0.0
    max_drawdown_95th: float = 0.0
    avg_monthly_withdrawal: float = 0.0
    expected_account_lifetime_months: float = 0.0
    annual_expected_income: float = 0.0

    # Distribution data
    attempt_outcomes: list[dict] = field(default_factory=list)
    monthly_pnl_distribution: list[float] = field(default_factory=list)
    survival_curve: list[float] = field(default_factory=list)


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
        timeframe: str = "1D",
    ) -> None:
        self.strategy_class = strategy_class
        self.instrument = instrument
        self.config = config
        self.database = database
        self.state_machine_enabled = state_machine_enabled
        self.num_attempts = num_attempts
        self.seed = seed
        self.timeframe = timeframe

    def run(self) -> CampaignResult | FundedCampaignResult:
        """Execute all attempts and return aggregated results."""
        rng = random.Random(self.seed)

        # Get available date range — use the correct table for the timeframe
        if self.timeframe == "1D":
            date_range = self.database.get_cached_date_range(self.instrument)
            if date_range == (None, None):
                raise ValueError(f"No data for {self.instrument}")
            min_date, max_date = date_range
        else:
            dt_range = self.database.get_intraday_cached_range(
                self.instrument, self.timeframe,
            )
            if dt_range == (None, None):
                raise ValueError(
                    f"No intraday {self.timeframe} data for {self.instrument}"
                )
            min_date = dt_range[0].date()
            max_date = dt_range[1].date()

        # Leave enough headroom so each attempt has data to run to completion
        if self.config.mode == TradingMode.FUNDED:
            headroom_days = int(self.config.funded_sim_days * 1.5)
        else:
            headroom_days = self.config.max_attempt_days * 2
        headroom = timedelta(days=headroom_days)
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
                strategy._state_manager = StateManager(
                    enabled=True, mode=self.config.mode,
                )

            sim = TopstepEvalSimulator(
                strategy=strategy,
                database=self.database,
                instrument=self.instrument,
                config=self.config,
                state_machine_enabled=self.state_machine_enabled,
                timeframe=self.timeframe,
            )
            result = sim.run_attempt(start_date=start)
            outcomes.append(result)

        if self.config.mode == TradingMode.FUNDED:
            return self._aggregate_funded(outcomes)
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
            timeframe=self.timeframe,
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

    def _aggregate_funded(self, outcomes: list[dict]) -> FundedCampaignResult:
        """Aggregate raw funded-attempt outcomes into a FundedCampaignResult."""
        if not outcomes:
            return FundedCampaignResult(
                campaign_id=str(uuid.uuid4())[:8],
                strategy_name=self.strategy_class.__name__,
                instrument=self.instrument,
                timeframe=self.timeframe,
                state_machine_enabled=self.state_machine_enabled,
                num_attempts=0,
                seed=self.seed,
            )

        survivors = [o for o in outcomes if o["status"] == "active"]
        survival_rate = len(survivors) / len(outcomes)

        # Collect all monthly PnLs across survivors
        all_monthly: list[float] = []
        for o in survivors:
            all_monthly.extend(o.get("monthly_pnls", {}).values())

        avg_monthly_pnl = (
            sum(all_monthly) / len(all_monthly) if all_monthly else 0.0
        )

        # Median monthly PnL
        if all_monthly:
            sorted_monthly = sorted(all_monthly)
            mid = len(sorted_monthly) // 2
            if len(sorted_monthly) % 2 == 0:
                median_monthly_pnl = (
                    sorted_monthly[mid - 1] + sorted_monthly[mid]
                ) / 2
            else:
                median_monthly_pnl = sorted_monthly[mid]
        else:
            median_monthly_pnl = 0.0

        # Sharpe ratio (annualised from monthly)
        if len(all_monthly) >= 2:
            mean_m = sum(all_monthly) / len(all_monthly)
            var_m = sum((x - mean_m) ** 2 for x in all_monthly) / (
                len(all_monthly) - 1
            )
            std_m = math.sqrt(var_m) if var_m > 0 else 0.0
            sharpe_ratio = (
                (mean_m / std_m * math.sqrt(12)) if std_m > 0 else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Peak drawdown stats
        drawdowns = [abs(o.get("peak_drawdown", 0.0)) for o in outcomes]
        sorted_dd = sorted(drawdowns)
        mid = len(sorted_dd) // 2
        max_drawdown_median = sorted_dd[mid] if sorted_dd else 0.0
        idx_95 = min(int(len(sorted_dd) * 0.95), len(sorted_dd) - 1)
        max_drawdown_95th = sorted_dd[idx_95] if sorted_dd else 0.0

        # Expected account lifetime (trading days -> months)
        trading_days_per_month = 21.7
        days_survived_list = [o.get("days_traded", 0) for o in outcomes]
        expected_lifetime = (
            sum(days_survived_list) / len(days_survived_list) / trading_days_per_month
            if days_survived_list
            else 0.0
        )

        # Monthly withdrawal = avg_monthly_pnl * payout_split (for survivors)
        avg_monthly_withdrawal = avg_monthly_pnl * self.config.payout_split

        # Annual expected income = avg_monthly_pnl * payout_split * 12 * survival_rate
        annual_expected_income = (
            avg_monthly_pnl * self.config.payout_split * 12 * survival_rate
        )

        # Survival curve: for each month 1..sim_months, fraction still alive
        sim_months = self.config.funded_sim_days // 22 or 12
        survival_curve: list[float] = []
        for month in range(1, sim_months + 1):
            min_days = int(month * trading_days_per_month)
            alive = sum(
                1 for o in outcomes if o.get("days_traded", 0) >= min_days
            )
            survival_curve.append(alive / len(outcomes))

        # Monthly PnL distribution (all monthly values across all attempts)
        all_monthly_all = []
        for o in outcomes:
            all_monthly_all.extend(o.get("monthly_pnls", {}).values())

        return FundedCampaignResult(
            campaign_id=str(uuid.uuid4())[:8],
            strategy_name=self.strategy_class.__name__,
            instrument=self.instrument,
            timeframe=self.timeframe,
            state_machine_enabled=self.state_machine_enabled,
            num_attempts=len(outcomes),
            seed=self.seed,
            sim_months=sim_months,
            survival_rate=survival_rate,
            avg_monthly_pnl=avg_monthly_pnl,
            median_monthly_pnl=median_monthly_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_median=max_drawdown_median,
            max_drawdown_95th=max_drawdown_95th,
            avg_monthly_withdrawal=avg_monthly_withdrawal,
            expected_account_lifetime_months=expected_lifetime,
            annual_expected_income=annual_expected_income,
            attempt_outcomes=outcomes,
            monthly_pnl_distribution=all_monthly_all,
            survival_curve=survival_curve,
        )
