"""Tests for funded-mode pipeline: FundedRules, FundedAttemptTracker,
simulator mode branching, FundedCampaignResult, and _aggregate_funded.
"""

import math
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from topstep.config import TopstepConfig, TradingMode, FUNDED_50K
from topstep.evaluation_rules import FundedRules, FundedOutcome
from topstep.attempt_tracker import (
    FundedAttemptTracker,
    FundedAttemptStatus,
)
from topstep.campaign_runner import FundedCampaignResult, CampaignRunner
from topstep.simulator import TopstepEvalSimulator


# ---------------------------------------------------------------------------
# Phase 5: FundedRules
# ---------------------------------------------------------------------------


class TestFundedRules:
    """FundedRules — trailing drawdown only, no profit target / timeout."""

    def test_active_when_in_profit(self):
        rules = FundedRules(TopstepConfig())
        outcome = rules.evaluate_day(500.0, 50500.0, date(2024, 1, 2))
        assert outcome == FundedOutcome.ACTIVE

    def test_blown_when_drawdown_breached(self):
        rules = FundedRules(TopstepConfig(max_loss=2000.0))
        # Account at 50000, floor at 48000.  Lose 2100 -> 47900 < 48000
        outcome = rules.evaluate_day(-2100.0, 47900.0, date(2024, 1, 2))
        assert outcome == FundedOutcome.BLOWN

    def test_not_blown_at_exact_floor(self):
        """At exactly the floor the account is still active (< is the breach)."""
        rules = FundedRules(TopstepConfig(max_loss=2000.0))
        outcome = rules.evaluate_day(-2000.0, 48000.0, date(2024, 1, 2))
        assert outcome == FundedOutcome.ACTIVE

    def test_trailing_drawdown_moves_up(self):
        rules = FundedRules(TopstepConfig(max_loss=2000.0))
        rules.evaluate_day(1000.0, 51000.0, date(2024, 1, 2))
        assert rules.eod_high_water == 51000.0
        assert rules.drawdown_floor == 49000.0

    def test_no_timeout(self):
        """Funded mode should never time out regardless of days traded."""
        rules = FundedRules(TopstepConfig(max_loss=2000.0))
        for i in range(100):
            outcome = rules.evaluate_day(10.0, 50010.0 + i * 10, date(2024, 1, 2) + timedelta(days=i))
        assert outcome == FundedOutcome.ACTIVE

    def test_no_profit_target(self):
        """Funded mode should never 'pass' — status is always ACTIVE or BLOWN."""
        rules = FundedRules(TopstepConfig(max_loss=2000.0))
        outcome = rules.evaluate_day(10000.0, 60000.0, date(2024, 1, 2))
        assert outcome == FundedOutcome.ACTIVE


# ---------------------------------------------------------------------------
# Phase 6: FundedAttemptTracker
# ---------------------------------------------------------------------------


class TestFundedAttemptTracker:
    """FundedAttemptTracker — monthly bucketing, consecutive losing, drawdown."""

    def test_monthly_bucketing(self):
        cfg = FUNDED_50K
        tracker = FundedAttemptTracker(cfg, state_machine_enabled=False)
        tracker.record_day(100.0, 50100.0, date(2024, 1, 15))
        tracker.record_day(200.0, 50300.0, date(2024, 1, 20))
        tracker.record_day(-50.0, 50250.0, date(2024, 2, 5))
        assert tracker.monthly_pnls == {"2024-01": 300.0, "2024-02": -50.0}

    def test_consecutive_losing_days(self):
        cfg = FUNDED_50K
        tracker = FundedAttemptTracker(cfg, state_machine_enabled=False)
        balance = cfg.account_size
        for i in range(5):
            balance -= 10
            tracker.record_day(-10.0, balance, date(2024, 1, 2) + timedelta(days=i))
        assert tracker.max_consecutive_losing_days == 5
        # Win day resets streak
        balance += 20
        tracker.record_day(20.0, balance, date(2024, 1, 7))
        assert tracker.max_consecutive_losing_days == 5  # max still 5
        assert tracker._consecutive_losing == 0

    def test_peak_drawdown_tracking(self):
        cfg = TopstepConfig(
            mode=TradingMode.FUNDED, account_size=50000.0, max_loss=5000.0,
        )
        tracker = FundedAttemptTracker(cfg, state_machine_enabled=False)
        tracker.record_day(500.0, 50500.0, date(2024, 1, 2))
        tracker.record_day(-800.0, 49700.0, date(2024, 1, 3))
        # cumulative = -300, high-water = 50500, balance = 49700
        # drawdown = 50000 + (-300) - 50500 = -800
        assert tracker.peak_drawdown == -800.0

    def test_to_dict_output_format(self):
        cfg = FUNDED_50K
        tracker = FundedAttemptTracker(cfg, state_machine_enabled=False)
        tracker.record_day(100.0, 50100.0, date(2024, 1, 2))
        d = tracker.to_dict()
        expected_keys = {
            "status",
            "cumulative_pnl",
            "days_traded",
            "daily_pnls",
            "monthly_pnls",
            "state_history",
            "peak_drawdown",
            "max_consecutive_losing_days",
        }
        assert expected_keys == set(d.keys())
        assert d["status"] == "active"
        assert d["days_traded"] == 1
        assert d["cumulative_pnl"] == 100.0

    def test_to_dict_blown_status(self):
        cfg = TopstepConfig(
            mode=TradingMode.FUNDED, account_size=50000.0, max_loss=2000.0,
        )
        tracker = FundedAttemptTracker(cfg, state_machine_enabled=False)
        tracker.record_day(-2100.0, 47900.0, date(2024, 1, 2))
        d = tracker.to_dict()
        assert d["status"] == "blown"

    def test_does_not_record_after_blown(self):
        cfg = TopstepConfig(
            mode=TradingMode.FUNDED, account_size=50000.0, max_loss=2000.0,
        )
        tracker = FundedAttemptTracker(cfg, state_machine_enabled=False)
        tracker.record_day(-2100.0, 47900.0, date(2024, 1, 2))
        status = tracker.record_day(500.0, 48400.0, date(2024, 1, 3))
        assert status == FundedAttemptStatus.BLOWN
        assert tracker.days_survived == 1  # second day not counted

    def test_state_history_populated(self):
        cfg = FUNDED_50K
        tracker = FundedAttemptTracker(cfg, state_machine_enabled=True)
        tracker.record_day(100.0, 50100.0, date(2024, 1, 2))
        tracker.record_day(-200.0, 49900.0, date(2024, 1, 3))
        assert len(tracker.state_history) == 2
        # First day cumulative=100 -> NORMAL; second day cumulative=-100 -> CONSERVATIVE
        assert tracker.state_history[0] == "NORMAL"
        assert tracker.state_history[1] == "CONSERVATIVE"


# ---------------------------------------------------------------------------
# Phase 7: TopstepEvalSimulator mode branching
# ---------------------------------------------------------------------------


class TestSimulatorModeBranching:
    """Verify simulator dispatches to the correct internal method."""

    @patch.object(TopstepEvalSimulator, "_run_eval_attempt", return_value={"status": "pass"})
    @patch.object(TopstepEvalSimulator, "_run_funded_attempt", return_value={"status": "active"})
    def test_eval_mode_calls_eval_attempt(self, mock_funded, mock_eval):
        cfg = TopstepConfig(mode=TradingMode.EVAL)
        sim = TopstepEvalSimulator(
            strategy=MagicMock(),
            database=MagicMock(),
            instrument="ES",
            config=cfg,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))
        mock_eval.assert_called_once()
        mock_funded.assert_not_called()
        assert result["status"] == "pass"

    @patch.object(TopstepEvalSimulator, "_run_eval_attempt", return_value={"status": "pass"})
    @patch.object(TopstepEvalSimulator, "_run_funded_attempt", return_value={"status": "active"})
    def test_funded_mode_calls_funded_attempt(self, mock_funded, mock_eval):
        cfg = TopstepConfig(mode=TradingMode.FUNDED)
        sim = TopstepEvalSimulator(
            strategy=MagicMock(),
            database=MagicMock(),
            instrument="ES",
            config=cfg,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))
        mock_funded.assert_called_once()
        mock_eval.assert_not_called()
        assert result["status"] == "active"


# ---------------------------------------------------------------------------
# Phase 8: FundedCampaignResult & _aggregate_funded
# ---------------------------------------------------------------------------


class TestFundedCampaignResult:
    """FundedCampaignResult field defaults and population."""

    def test_default_fields(self):
        r = FundedCampaignResult(
            campaign_id="test",
            strategy_name="TestStrat",
            instrument="ES",
            timeframe="5m",
            state_machine_enabled=True,
            num_attempts=100,
            seed=42,
        )
        assert r.survival_rate == 0.0
        assert r.sharpe_ratio == 0.0
        assert r.survival_curve == []
        assert r.attempt_outcomes == []
        assert r.sim_months == 12


def _make_outcome(
    status: str = "active",
    days_traded: int = 260,
    cumulative_pnl: float = 5000.0,
    peak_drawdown: float = -500.0,
    monthly_pnls: dict | None = None,
) -> dict:
    """Helper to build a fake funded-attempt outcome dict."""
    if monthly_pnls is None:
        monthly_pnls = {f"2024-{m:02d}": cumulative_pnl / 12 for m in range(1, 13)}
    return {
        "status": status,
        "cumulative_pnl": cumulative_pnl,
        "days_traded": days_traded,
        "daily_pnls": [cumulative_pnl / max(days_traded, 1)] * days_traded,
        "monthly_pnls": monthly_pnls,
        "state_history": ["NORMAL"] * days_traded,
        "peak_drawdown": peak_drawdown,
        "max_consecutive_losing_days": 2,
    }


class TestAggregateFunded:
    """CampaignRunner._aggregate_funded — survival rate, survival curve, etc."""

    def _make_runner(self, config=None) -> CampaignRunner:
        cfg = config or FUNDED_50K
        return CampaignRunner(
            strategy_class=MagicMock,
            instrument="ES",
            config=cfg,
            database=MagicMock(),
            num_attempts=10,
        )

    def test_survival_rate_all_active(self):
        runner = self._make_runner()
        outcomes = [_make_outcome(status="active") for _ in range(10)]
        result = runner._aggregate_funded(outcomes)
        assert result.survival_rate == 1.0

    def test_survival_rate_mixed(self):
        runner = self._make_runner()
        outcomes = (
            [_make_outcome(status="active") for _ in range(7)]
            + [_make_outcome(status="blown", days_traded=50) for _ in range(3)]
        )
        result = runner._aggregate_funded(outcomes)
        assert abs(result.survival_rate - 0.7) < 1e-9

    def test_survival_rate_all_blown(self):
        runner = self._make_runner()
        outcomes = [_make_outcome(status="blown", days_traded=30) for _ in range(5)]
        result = runner._aggregate_funded(outcomes)
        assert result.survival_rate == 0.0

    def test_survival_curve_shape(self):
        """Survival curve should have sim_months entries."""
        cfg = TopstepConfig(
            mode=TradingMode.FUNDED,
            account_size=50000.0,
            max_loss=2000.0,
            funded_sim_days=260,
        )
        runner = self._make_runner(config=cfg)
        outcomes = [_make_outcome(status="active", days_traded=260) for _ in range(5)]
        result = runner._aggregate_funded(outcomes)
        expected_months = 260 // 22
        assert len(result.survival_curve) == expected_months

    def test_survival_curve_decreasing_with_blowups(self):
        """Blown attempts at different points should produce decreasing curve."""
        runner = self._make_runner()
        outcomes = [
            _make_outcome(status="active", days_traded=260),
            _make_outcome(status="blown", days_traded=10),   # blown month 1
            _make_outcome(status="blown", days_traded=65),   # blown month 3
        ]
        result = runner._aggregate_funded(outcomes)
        # First month: all 3 alive at day 0 but blown at day 10 means
        # by month 1 (21.7 days) only 2 alive
        # Curve should be non-increasing
        for i in range(1, len(result.survival_curve)):
            assert result.survival_curve[i] <= result.survival_curve[i - 1] or \
                result.survival_curve[i] == result.survival_curve[i - 1]

    def test_avg_monthly_pnl_survivors_only(self):
        runner = self._make_runner()
        outcomes = [
            _make_outcome(status="active", monthly_pnls={"2024-01": 1000.0}),
            _make_outcome(status="blown", days_traded=10, monthly_pnls={"2024-01": -500.0}),
        ]
        result = runner._aggregate_funded(outcomes)
        # Only the survivor's monthly PnLs are counted
        assert result.avg_monthly_pnl == 1000.0

    def test_sharpe_ratio_positive(self):
        runner = self._make_runner()
        outcomes = [
            _make_outcome(
                status="active",
                monthly_pnls={f"2024-{m:02d}": 400.0 + m * 10 for m in range(1, 13)},
            )
            for _ in range(5)
        ]
        result = runner._aggregate_funded(outcomes)
        assert result.sharpe_ratio > 0

    def test_annual_expected_income(self):
        runner = self._make_runner()
        outcomes = [
            _make_outcome(status="active", monthly_pnls={"2024-01": 500.0}),
        ]
        result = runner._aggregate_funded(outcomes)
        # avg_monthly_pnl=500, payout_split=0.90, survival_rate=1.0
        expected = 500.0 * 0.90 * 12 * 1.0
        assert abs(result.annual_expected_income - expected) < 1e-6

    def test_empty_outcomes(self):
        runner = self._make_runner()
        result = runner._aggregate_funded([])
        assert result.num_attempts == 0
        assert result.survival_rate == 0.0
        assert result.survival_curve == []

    def test_result_type_is_funded(self):
        runner = self._make_runner()
        outcomes = [_make_outcome() for _ in range(3)]
        result = runner._aggregate_funded(outcomes)
        assert isinstance(result, FundedCampaignResult)

    def test_max_drawdown_stats(self):
        runner = self._make_runner()
        outcomes = [
            _make_outcome(peak_drawdown=-100.0),
            _make_outcome(peak_drawdown=-300.0),
            _make_outcome(peak_drawdown=-500.0),
        ]
        result = runner._aggregate_funded(outcomes)
        # Median of [100, 300, 500] = 300
        assert result.max_drawdown_median == 300.0
