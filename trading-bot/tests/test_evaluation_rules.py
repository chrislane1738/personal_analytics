"""Tests for topstep.evaluation_rules — pass/fail logic, trailing drawdown, consistency."""

from topstep.config import TopstepConfig
from topstep.evaluation_rules import EvaluationRules, EvalOutcome


def test_pass_when_target_reached_and_consistent():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=1600.0, eod_balance=51600.0)  # ACTIVE
    rules.evaluate_day(day_pnl=1500.0, eod_balance=53100.0)  # consistency violated (1600/3100=51.6%)
    result = rules.evaluate_day(day_pnl=500.0, eod_balance=53600.0)  # 1600/3600=44.4% ✓
    assert result == EvalOutcome.PASS


def test_fail_when_drawdown_breached():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=500.0, eod_balance=50500.0)
    result = rules.evaluate_day(day_pnl=-2100.0, eod_balance=48400.0)  # below floor 48500
    assert result == EvalOutcome.FAIL


def test_fail_at_exact_drawdown_limit():
    rules = EvaluationRules(TopstepConfig())
    result = rules.evaluate_day(day_pnl=-2000.0, eod_balance=48000.0)  # exactly at floor
    assert result == EvalOutcome.FAIL


def test_trailing_drawdown_moves_up():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=1000.0, eod_balance=51000.0)
    assert rules.eod_high_water == 51000.0
    assert rules.drawdown_floor == 49000.0
    result = rules.evaluate_day(day_pnl=-500.0, eod_balance=50500.0)
    assert result == EvalOutcome.ACTIVE
    assert rules.eod_high_water == 51000.0  # didn't move down


def test_consistency_check():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=2800.0, eod_balance=52800.0)
    rules.evaluate_day(day_pnl=300.0, eod_balance=53100.0)
    assert rules.consistency_satisfied is False
    assert rules.total_profit == 3100.0
    assert rules.best_day_pnl == 2800.0


def test_timeout():
    config = TopstepConfig(max_attempt_days=3)
    rules = EvaluationRules(config)
    rules.evaluate_day(day_pnl=100.0, eod_balance=50100.0)
    rules.evaluate_day(day_pnl=100.0, eod_balance=50200.0)
    result = rules.evaluate_day(day_pnl=100.0, eod_balance=50300.0)
    assert result == EvalOutcome.TIMEOUT


def test_reset_clears_state():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=1000.0, eod_balance=51000.0)
    rules.reset()
    assert rules.eod_high_water == 50000.0
    assert rules.total_profit == 0.0
    assert rules.days_traded == 0


def test_active_while_below_target():
    """Days traded but cumulative pnl has not yet hit the profit target."""
    rules = EvaluationRules(TopstepConfig())
    result = rules.evaluate_day(day_pnl=500.0, eod_balance=50500.0)
    assert result == EvalOutcome.ACTIVE


def test_pass_requires_consistency():
    """Reaching profit target alone is not enough — consistency must also hold."""
    rules = EvaluationRules(TopstepConfig())
    # Single day bang: 3500 profit but 100% of total — consistency violated
    result = rules.evaluate_day(day_pnl=3500.0, eod_balance=53500.0)
    assert result == EvalOutcome.ACTIVE


def test_total_profit_only_counts_positive_days():
    """Losing days should not reduce total_profit used for consistency calculation."""
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=2000.0, eod_balance=52000.0)
    rules.evaluate_day(day_pnl=-500.0, eod_balance=51500.0)
    # total_profit should only be 2000, not 1500
    assert rules.total_profit == 2000.0


def test_no_pass_when_total_profit_zero():
    """consistency_satisfied must be False when total_profit <= 0."""
    rules = EvaluationRules(TopstepConfig())
    assert rules.consistency_satisfied is False


def test_reset_resets_drawdown_floor():
    """After a reset the drawdown floor should return to initial value."""
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=5000.0, eod_balance=55000.0)
    assert rules.drawdown_floor == 53000.0  # trailed up
    rules.reset()
    assert rules.drawdown_floor == 48000.0  # account_size - max_loss


def test_days_traded_increments():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=100.0, eod_balance=50100.0)
    rules.evaluate_day(day_pnl=100.0, eod_balance=50200.0)
    assert rules.days_traded == 2


def test_daily_pnls_recorded():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=200.0, eod_balance=50200.0)
    rules.evaluate_day(day_pnl=-100.0, eod_balance=50100.0)
    assert rules.daily_pnls == [200.0, -100.0]
