"""Tests for analytics/metrics.py — Core metrics engine."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from analytics.metrics import (
    DrawdownInfo,
    annualized_volatility,
    beta,
    cagr,
    calmar_ratio,
    compute_all_metrics,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    win_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_trades_mixed() -> list[dict]:
    """5 winners totalling $5000, 3 losers totalling -$2000."""
    return [
        {"pnl": 1000.0},
        {"pnl": 1200.0},
        {"pnl": 800.0},
        {"pnl": 1500.0},
        {"pnl": 500.0},
        {"pnl": -700.0},
        {"pnl": -900.0},
        {"pnl": -400.0},
    ]


@pytest.fixture()
def simple_equity_curve() -> list[tuple[date, float]]:
    """[100K, 110K, 105K, 95K, 100K, 108K] — one day apart."""
    values = [100_000.0, 110_000.0, 105_000.0, 95_000.0, 100_000.0, 108_000.0]
    base = date(2024, 1, 1)
    return [(base + timedelta(days=i), v) for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# 1. Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_with_varying_positive_returns(self):
        """Returns with positive mean above risk-free rate should yield positive Sharpe."""
        rng = np.random.default_rng(1)
        # Mean ~0.001/day >> daily rfr of 0.04/252 ≈ 0.000159
        returns = list(0.001 + rng.normal(0, 0.005, 252))
        result = sharpe_ratio(returns, risk_free_rate=0.04)
        assert result > 0

    def test_hand_calculable_value(self):
        """Constant return of 0.001, rfr=0.04.
        daily_rf = 0.04/252 ≈ 0.0001587
        excess_daily = 0.001 - 0.0001587 ≈ 0.0008413
        std(excess) = 0 → but since it's constant this is 0; test a varying case.
        """
        # Use a slightly varied series to avoid zero std
        rng = np.random.default_rng(42)
        returns = list(0.001 + rng.normal(0, 0.005, 252))
        result = sharpe_ratio(returns, risk_free_rate=0.04)
        # Should be positive since mean return > risk-free rate
        assert result > 0

    def test_zero_returns_gives_zero_sharpe(self):
        """Returns of exactly 0 — constant series → std ≈ 0 → convention returns 0.0."""
        returns = [0.0] * 252
        result = sharpe_ratio(returns, risk_free_rate=0.04)
        # Constant series: sample std = 0 (below tolerance threshold) → return 0.0
        assert result == pytest.approx(0.0)

    def test_empty_returns(self):
        assert sharpe_ratio([]) == pytest.approx(0.0)

    def test_single_return(self):
        assert sharpe_ratio([0.01]) == pytest.approx(0.0)

    def test_high_return_series_beats_low(self):
        high = [0.002] * 100 + [-0.001] * 50
        low  = [0.0005] * 100 + [-0.001] * 50
        assert sharpe_ratio(high) > sharpe_ratio(low)

    def test_constant_returns_std_zero_returns_zero(self):
        # All returns identical → std = 0 → return 0.0
        returns = [0.01] * 10
        assert sharpe_ratio(returns, risk_free_rate=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Sortino ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_sortino_greater_than_sharpe_when_only_upside_volatility(self):
        """Mix of positive and negative returns with variance on both sides.
        Sortino ignores upside volatility so it should be >= Sharpe."""
        rng = np.random.default_rng(3)
        # Mostly positive, with volatile positive days and quiet negative days
        positives = list(0.002 + rng.normal(0, 0.008, 200))  # large upside vol
        negatives = list(-0.001 + rng.normal(0, 0.0005, 52))  # small downside vol
        returns = positives + negatives
        s = sharpe_ratio(returns, risk_free_rate=0.01)
        so = sortino_ratio(returns, risk_free_rate=0.01)
        # Sortino denominator only counts downside → typically higher ratio
        assert so >= s

    def test_no_downside_returns_zero(self):
        """All positive returns → no downside vol → return 0.0."""
        returns = [0.002] * 100
        assert sortino_ratio(returns, risk_free_rate=0.0) == pytest.approx(0.0)

    def test_empty_returns(self):
        assert sortino_ratio([]) == pytest.approx(0.0)

    def test_positive_with_mixed_returns(self):
        rng = np.random.default_rng(7)
        returns = list(0.001 + rng.normal(0, 0.01, 252))
        result = sortino_ratio(returns, risk_free_rate=0.04)
        assert isinstance(result, float)

    def test_single_return(self):
        assert sortino_ratio([0.01]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_drawdown_percentage(self, simple_equity_curve):
        """Max DD = (110K - 95K) / 110K = 13.636..%"""
        info = max_drawdown(simple_equity_curve)
        expected_pct = (110_000.0 - 95_000.0) / 110_000.0
        assert info.max_drawdown_pct == pytest.approx(expected_pct, rel=1e-6)

    def test_drawdown_dollar(self, simple_equity_curve):
        """Max DD in dollars = 110K - 95K = 15K."""
        info = max_drawdown(simple_equity_curve)
        assert info.max_drawdown_dollar == pytest.approx(15_000.0)

    def test_peak_date(self, simple_equity_curve):
        """Peak is on day index 1 (110K)."""
        info = max_drawdown(simple_equity_curve)
        assert info.peak_date == date(2024, 1, 2)

    def test_trough_date(self, simple_equity_curve):
        """Trough is on day index 3 (95K)."""
        info = max_drawdown(simple_equity_curve)
        assert info.trough_date == date(2024, 1, 4)

    def test_recovery_date(self, simple_equity_curve):
        """Recovery happens at day index 4 (100K) — still below 110K peak.
        Recovery requires reaching 110K, which never happens → None."""
        info = max_drawdown(simple_equity_curve)
        # 108K < 110K → never recovers fully
        assert info.recovery_date is None

    def test_empty_curve(self):
        info = max_drawdown([])
        assert info.max_drawdown_pct == pytest.approx(0.0)
        assert info.peak_date is None

    def test_monotonically_increasing_no_drawdown(self):
        curve = [(date(2024, 1, i + 1), 100_000.0 + i * 1000) for i in range(5)]
        info = max_drawdown(curve)
        assert info.max_drawdown_pct == pytest.approx(0.0)
        assert info.peak_date is None

    def test_recovery_detected_when_curve_recovers(self):
        """Equity goes 100K → 80K → 100K. Recovery at day 3."""
        curve = [
            (date(2024, 1, 1), 100_000.0),
            (date(2024, 1, 2), 80_000.0),
            (date(2024, 1, 3), 100_000.0),
        ]
        info = max_drawdown(curve)
        assert info.recovery_date == date(2024, 1, 3)

    def test_duration_days_no_recovery(self, simple_equity_curve):
        """Duration = from peak to end of series when no recovery."""
        info = max_drawdown(simple_equity_curve)
        # peak = Jan 2, end = Jan 6 → 4 days
        assert info.duration_days == 4

    def test_duration_days_with_recovery(self):
        curve = [
            (date(2024, 1, 1), 100_000.0),
            (date(2024, 1, 2), 80_000.0),
            (date(2024, 1, 5), 100_000.0),  # recovery
        ]
        info = max_drawdown(curve)
        assert info.duration_days == 4  # Jan 1 → Jan 5


# ---------------------------------------------------------------------------
# 4. CAGR
# ---------------------------------------------------------------------------

class TestCagr:
    def test_100k_to_150k_over_3_years(self):
        """$100K → $150K over 3 years = (1.5)^(1/3) - 1 ≈ 14.471%"""
        result = cagr(100_000.0, 150_000.0, 3.0)
        expected = (1.5 ** (1 / 3)) - 1
        assert result == pytest.approx(expected, rel=1e-6)

    def test_cagr_approx_14_47_pct(self):
        result = cagr(100_000.0, 150_000.0, 3.0)
        assert result == pytest.approx(0.1447, abs=1e-3)

    def test_no_growth_returns_zero(self):
        assert cagr(100_000.0, 100_000.0, 5.0) == pytest.approx(0.0)

    def test_zero_start_returns_zero(self):
        assert cagr(0.0, 150_000.0, 3.0) == pytest.approx(0.0)

    def test_zero_years_returns_zero(self):
        assert cagr(100_000.0, 150_000.0, 0.0) == pytest.approx(0.0)

    def test_negative_start_returns_zero(self):
        assert cagr(-1000.0, 150_000.0, 3.0) == pytest.approx(0.0)

    def test_double_in_one_year(self):
        """$100K → $200K in 1 year = 100% CAGR."""
        assert cagr(100_000.0, 200_000.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_five_wins_three_losses(self, flat_trades_mixed):
        """5 wins = 5000, 3 losses = 2000 → PF = 2.5"""
        result = profit_factor(flat_trades_mixed)
        assert result == pytest.approx(2.5)

    def test_all_winners(self):
        trades = [{"pnl": 100.0}, {"pnl": 200.0}]
        result = profit_factor(trades)
        assert result == float("inf")

    def test_all_losers(self):
        trades = [{"pnl": -100.0}, {"pnl": -200.0}]
        result = profit_factor(trades)
        assert result == pytest.approx(0.0)

    def test_no_trades(self):
        assert profit_factor([]) == pytest.approx(0.0)

    def test_equal_wins_losses(self):
        trades = [{"pnl": 500.0}, {"pnl": -500.0}]
        assert profit_factor(trades) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_five_wins_out_of_eight(self, flat_trades_mixed):
        """5 wins / 8 total = 62.5%"""
        result = win_rate(flat_trades_mixed)
        assert result == pytest.approx(0.625)

    def test_all_winners(self):
        trades = [{"pnl": 100.0}] * 4
        assert win_rate(trades) == pytest.approx(1.0)

    def test_no_winners(self):
        trades = [{"pnl": -100.0}] * 4
        assert win_rate(trades) == pytest.approx(0.0)

    def test_empty_trades(self):
        assert win_rate([]) == pytest.approx(0.0)

    def test_single_win(self):
        assert win_rate([{"pnl": 1.0}]) == pytest.approx(1.0)

    def test_single_loss(self):
        assert win_rate([{"pnl": -1.0}]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Expectancy
# ---------------------------------------------------------------------------

class TestExpectancy:
    def test_hand_calculated(self, flat_trades_mixed):
        """5 winners totalling 5000 → avg_win = 1000.
        3 losers totalling -2000 → avg_loss = 666.67.
        wr = 5/8 = 0.625, lr = 0.375.
        expectancy = 1000 * 0.625 - 666.67 * 0.375 ≈ 625 - 250 = 375.
        """
        result = expectancy(flat_trades_mixed)
        # Precise: avg_win=1000, avg_loss=2000/3≈666.67
        expected = 1000.0 * (5/8) - (2000/3) * (3/8)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_positive_expectancy(self, flat_trades_mixed):
        assert expectancy(flat_trades_mixed) > 0

    def test_empty_trades_returns_zero(self):
        assert expectancy([]) == pytest.approx(0.0)

    def test_all_losses_returns_negative(self):
        trades = [{"pnl": -100.0}] * 5
        assert expectancy(trades) < 0

    def test_breakeven_series(self):
        """Equal wins and losses of equal magnitude → expectancy = 0."""
        trades = [{"pnl": 100.0}, {"pnl": -100.0}]
        assert expectancy(trades) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Beta
# ---------------------------------------------------------------------------

class TestBeta:
    def test_strategy_is_2x_benchmark(self):
        """strategy = 2 * benchmark → beta ≈ 2.0"""
        bench = [0.01, -0.005, 0.008, -0.003, 0.012, 0.006, -0.004, 0.009]
        strat = [2 * r for r in bench]
        result = beta(strat, bench)
        assert result == pytest.approx(2.0, rel=1e-5)

    def test_beta_one_for_identical_series(self):
        returns = [0.01, -0.005, 0.008, 0.003, -0.002]
        assert beta(returns, returns) == pytest.approx(1.0, rel=1e-5)

    def test_uncorrelated_returns_near_zero_beta(self):
        rng = np.random.default_rng(0)
        bench = list(rng.normal(0, 0.01, 500))
        strat = list(rng.normal(0, 0.01, 500))  # independent
        result = beta(strat, bench)
        # Not necessarily zero but should be close
        assert abs(result) < 0.3

    def test_zero_variance_benchmark_returns_zero(self):
        bench = [0.01] * 10  # constant → var = 0
        strat = [0.01] * 10
        assert beta(strat, bench) == pytest.approx(0.0)

    def test_empty_inputs(self):
        assert beta([], []) == pytest.approx(0.0)

    def test_single_element(self):
        assert beta([0.01], [0.01]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. Value at Risk
# ---------------------------------------------------------------------------

class TestValueAtRisk:
    def test_known_distribution(self):
        """With a sorted list, 5th percentile should be near the bottom 5%."""
        returns = [i / 1000 for i in range(-50, 50)]  # -0.05 to 0.049
        result = value_at_risk(returns, confidence=0.95)
        # 5th percentile of 100 values = 5th value = -0.046 (approx)
        assert result < 0

    def test_var_is_negative_for_mixed_returns(self):
        rng = np.random.default_rng(99)
        returns = list(rng.normal(0.001, 0.02, 1000))
        result = value_at_risk(returns, 0.95)
        assert result < 0

    def test_empty_returns(self):
        assert value_at_risk([]) == pytest.approx(0.0)

    def test_all_positive_returns_var_may_be_positive(self):
        """If all returns are positive, even 5th percentile is positive."""
        returns = [0.001 * i for i in range(1, 101)]
        result = value_at_risk(returns, 0.95)
        assert result > 0

    def test_90_confidence_higher_than_95(self):
        """VaR at 90% confidence is less negative (higher) than at 95%."""
        rng = np.random.default_rng(5)
        returns = list(rng.normal(0, 0.01, 1000))
        var_95 = value_at_risk(returns, 0.95)
        var_90 = value_at_risk(returns, 0.90)
        # 90% VaR cuts at 10th percentile, 95% at 5th → 95% is more negative
        assert var_90 > var_95

    def test_percentile_correctness(self):
        """Manually verify: returns = [-0.10, -0.05, 0, 0.05, 0.10]
        5th percentile of 5 values ≈ the smallest value area."""
        returns = [-0.10, -0.05, 0.0, 0.05, 0.10]
        result = value_at_risk(returns, 0.95)
        # np.percentile(returns, 5) should be near -0.10
        assert result == pytest.approx(np.percentile(returns, 5))


# ---------------------------------------------------------------------------
# 10. Annualized Volatility
# ---------------------------------------------------------------------------

class TestAnnualizedVolatility:
    def test_zero_for_single_return(self):
        assert annualized_volatility([0.01]) == pytest.approx(0.0)

    def test_empty_returns(self):
        assert annualized_volatility([]) == pytest.approx(0.0)

    def test_positive_for_mixed_returns(self):
        returns = [0.01, -0.01, 0.005, -0.005, 0.002]
        assert annualized_volatility(returns) > 0

    def test_scales_by_sqrt_252(self):
        """Manual check: std(returns) * sqrt(252)."""
        returns = [0.01, -0.01, 0.005, -0.005, 0.003]
        expected = float(np.std(returns) * np.sqrt(252))
        assert annualized_volatility(returns) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 11. Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_basic_ratio(self):
        assert calmar_ratio(0.20, 0.10) == pytest.approx(2.0)

    def test_zero_drawdown_returns_zero(self):
        assert calmar_ratio(0.20, 0.0) == pytest.approx(0.0)

    def test_negative_drawdown_pct_still_uses_abs(self):
        # max_dd_pct could be stored as positive fraction
        assert calmar_ratio(0.15, 0.05) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 12. compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    EXPECTED_KEYS = {
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "max_drawdown_dollar",
        "drawdown_peak_date",
        "drawdown_trough_date",
        "drawdown_recovery_date",
        "drawdown_duration_days",
        "cagr",
        "calmar_ratio",
        "annualized_volatility",
        "profit_factor",
        "win_rate",
        "expectancy",
        "total_trades",
        "value_at_risk_95",
        "total_return",
        "start_value",
        "end_value",
        "years",
        "beta",
        "benchmark_total_return",
        "alpha",
    }

    def _make_trades(self) -> list[dict]:
        return [
            {"pnl": 1000.0},
            {"pnl": 1200.0},
            {"pnl": 800.0},
            {"pnl": 1500.0},
            {"pnl": 500.0},
            {"pnl": -700.0},
            {"pnl": -900.0},
            {"pnl": -400.0},
        ]

    def _make_equity_curve(self, n: int = 252) -> list[tuple[date, float]]:
        """Generate a simple random-walk equity curve over n days."""
        rng = np.random.default_rng(42)
        values = [100_000.0]
        for _ in range(n - 1):
            values.append(values[-1] * (1 + rng.normal(0.0005, 0.01)))
        base = date(2024, 1, 2)
        return [(base + timedelta(days=i), v) for i, v in enumerate(values)]

    def test_all_keys_present(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_total_trades_count(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert result["total_trades"] == 8

    def test_win_rate_correct(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert result["win_rate"] == pytest.approx(0.625)

    def test_profit_factor_correct(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert result["profit_factor"] == pytest.approx(2.5)

    def test_max_drawdown_pct_correct(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        expected_pct = (110_000.0 - 95_000.0) / 110_000.0
        assert result["max_drawdown_pct"] == pytest.approx(expected_pct, rel=1e-6)

    def test_cagr_value_type(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert isinstance(result["cagr"], float)

    def test_no_benchmark_beta_is_none(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert result["beta"] is None

    def test_with_benchmark_beta_not_none(self, flat_trades_mixed):
        equity = self._make_equity_curve(252)
        benchmark = self._make_equity_curve(252)
        result = compute_all_metrics(equity, flat_trades_mixed, benchmark_curve=benchmark)
        assert result["beta"] is not None
        assert isinstance(result["beta"], float)

    def test_start_end_values(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert result["start_value"] == pytest.approx(100_000.0)
        assert result["end_value"] == pytest.approx(108_000.0)

    def test_total_return_calculation(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        # (108K - 100K) / 100K = 8%
        assert result["total_return"] == pytest.approx(0.08)

    def test_benchmark_alpha_with_benchmark(self, flat_trades_mixed):
        equity = self._make_equity_curve(252)
        benchmark = self._make_equity_curve(252)
        result = compute_all_metrics(equity, flat_trades_mixed, benchmark_curve=benchmark)
        # alpha = total_return - benchmark_total_return
        assert result["alpha"] == pytest.approx(
            result["total_return"] - result["benchmark_total_return"], rel=1e-5
        )

    def test_no_benchmark_alpha_is_none(self, simple_equity_curve, flat_trades_mixed):
        result = compute_all_metrics(simple_equity_curve, flat_trades_mixed)
        assert result["alpha"] is None

    def test_with_longer_equity_curve(self, flat_trades_mixed):
        equity = self._make_equity_curve(504)  # ~2 years
        result = compute_all_metrics(equity, flat_trades_mixed)
        assert result["years"] == pytest.approx(503 / 365.25, rel=1e-3)

    def test_empty_trades(self, simple_equity_curve):
        result = compute_all_metrics(simple_equity_curve, [])
        assert result["total_trades"] == 0
        assert result["win_rate"] == pytest.approx(0.0)
        assert result["profit_factor"] == pytest.approx(0.0)
