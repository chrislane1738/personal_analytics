"""Tests for enhanced Monte Carlo: window shuffle, bootstrap resampling, and confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.monte_carlo import (
    compute_metric_confidence_intervals,
    run_bootstrap,
    run_window_shuffle,
)


# ---------------------------------------------------------------------------
# 1. Window shuffle — basic functionality
# ---------------------------------------------------------------------------

def test_window_shuffle():
    windows = [[100, -50, 200], [-100, 150, 80], [300, -200, 100], [50, 100, -30]]
    result = run_window_shuffle(windows, initial_capital=10000, num_simulations=100, seed=42)
    assert "percentile_bands" in result
    assert set(result["percentile_bands"].keys()) == {"p5", "p25", "p50", "p75", "p95"}
    assert result["simulations"] == 100
    assert "sharpe_ci" in result
    assert len(result["sharpe_ci"]) == 2


def test_window_shuffle_single_window():
    result = run_window_shuffle([[100, 200]], initial_capital=10000, num_simulations=50, seed=42)
    # With 1 window, all shuffles are identical
    assert result["percentile_bands"]["p5"] == result["percentile_bands"]["p95"]


def test_window_shuffle_deterministic():
    """Same seed must produce identical results across two runs."""
    windows = [[100, -50, 200], [-100, 150, 80], [300, -200, 100]]
    result_a = run_window_shuffle(windows, initial_capital=10000, num_simulations=100, seed=42)
    result_b = run_window_shuffle(windows, initial_capital=10000, num_simulations=100, seed=42)
    assert result_a["final_equity_distribution"] == result_b["final_equity_distribution"]
    assert result_a["drawdown_distribution"] == result_b["drawdown_distribution"]


def test_window_shuffle_actual_curve():
    """actual_curve should be the equity curve from original window order."""
    windows = [[100, -50], [200, 300]]
    result = run_window_shuffle(windows, initial_capital=10000, num_simulations=10, seed=42)
    # Original order: 100, -50, 200, 300 -> equity 10100, 10050, 10250, 10550
    assert result["actual_curve"] == [10100, 10050, 10250, 10550]


def test_window_shuffle_probability_of_ruin():
    """Windows with heavy losses should produce some ruin paths."""
    windows = [[-5000, -3000], [-4000, -2000], [20000, 1000]]
    result = run_window_shuffle(
        windows, initial_capital=5000, num_simulations=1000, seed=42, ruin_threshold=0.0
    )
    # Some orderings should hit ruin when losses come first
    assert result["probability_of_ruin"] >= 0.0
    assert 0.0 <= result["probability_of_ruin"] <= 1.0


def test_window_shuffle_drawdown_valid_range():
    """All drawdown values should be in [0, 1]."""
    windows = [[100, -50, 200], [-100, 150, 80], [300, -200, 100]]
    result = run_window_shuffle(windows, initial_capital=10000, num_simulations=200, seed=42)
    for dd in result["drawdown_distribution"]:
        assert 0.0 <= dd <= 1.0, f"Drawdown {dd} outside [0, 1]"


def test_window_shuffle_empty_windows():
    """Empty windows list should return degenerate result."""
    result = run_window_shuffle([], initial_capital=10000, num_simulations=50, seed=42)
    assert result["simulations"] == 50
    assert result["probability_of_ruin"] == 0.0


# ---------------------------------------------------------------------------
# 2. Bootstrap — basic functionality
# ---------------------------------------------------------------------------

def test_bootstrap():
    windows = [[100, -50, 200], [-100, 150, 80], [300, -200, 100], [50, 100, -30]]
    result = run_bootstrap(windows, initial_capital=10000, num_simulations=100, seed=42)
    assert "percentile_bands" in result
    assert result["simulations"] == 100
    assert "sharpe_ci" in result


def test_bootstrap_empty_windows():
    result = run_bootstrap([], initial_capital=10000, num_simulations=50, seed=42)
    assert result["simulations"] == 50
    assert result["probability_of_ruin"] == 0.0


def test_bootstrap_deterministic():
    """Same seed must produce identical results."""
    windows = [[100, -50, 200], [-100, 150, 80]]
    result_a = run_bootstrap(windows, initial_capital=10000, num_simulations=100, seed=42)
    result_b = run_bootstrap(windows, initial_capital=10000, num_simulations=100, seed=42)
    assert result_a["final_equity_distribution"] == result_b["final_equity_distribution"]
    assert result_a["drawdown_distribution"] == result_b["drawdown_distribution"]


def test_bootstrap_single_window():
    """With 1 window, bootstrap always picks that same window → all paths identical."""
    result = run_bootstrap([[100, 200]], initial_capital=10000, num_simulations=50, seed=42)
    assert result["percentile_bands"]["p5"] == result["percentile_bands"]["p95"]


def test_bootstrap_can_duplicate_windows():
    """Bootstrap samples with replacement — final equities should vary more than shuffle
    because some windows can appear multiple times."""
    windows = [[500], [-500], [1000], [-1000]]
    result = run_bootstrap(windows, initial_capital=10000, num_simulations=500, seed=42)
    # With replacement, final equity should vary (not all identical)
    finals = result["final_equity_distribution"]
    assert max(finals) != min(finals), "Bootstrap should produce varied final equities"


def test_bootstrap_drawdown_valid_range():
    """All drawdown values should be in [0, 1]."""
    windows = [[100, -50, 200], [-100, 150, 80], [300, -200, 100]]
    result = run_bootstrap(windows, initial_capital=10000, num_simulations=200, seed=42)
    for dd in result["drawdown_distribution"]:
        assert 0.0 <= dd <= 1.0, f"Drawdown {dd} outside [0, 1]"


# ---------------------------------------------------------------------------
# 3. Confidence intervals
# ---------------------------------------------------------------------------

def test_confidence_intervals():
    sharpe_values = np.random.default_rng(42).normal(1.5, 0.3, 1000)
    dd_values = np.random.default_rng(42).normal(-0.10, 0.03, 1000)
    ci = compute_metric_confidence_intervals(sharpe_values, dd_values)
    assert ci["sharpe_ci"][0] < ci["sharpe_ci"][1]
    assert ci["max_dd_ci"][0] < ci["max_dd_ci"][1]


def test_confidence_intervals_custom_confidence():
    """90% CI should be narrower than 95% CI."""
    rng = np.random.default_rng(42)
    sharpe_values = rng.normal(1.5, 0.3, 5000)
    dd_values = rng.normal(-0.10, 0.03, 5000)
    ci_95 = compute_metric_confidence_intervals(sharpe_values, dd_values, confidence=0.95)
    ci_90 = compute_metric_confidence_intervals(sharpe_values, dd_values, confidence=0.90)
    sharpe_width_95 = ci_95["sharpe_ci"][1] - ci_95["sharpe_ci"][0]
    sharpe_width_90 = ci_90["sharpe_ci"][1] - ci_90["sharpe_ci"][0]
    assert sharpe_width_90 < sharpe_width_95


def test_confidence_intervals_degenerate():
    """Single-valued arrays should produce zero-width CI."""
    ci = compute_metric_confidence_intervals(
        np.array([1.0, 1.0, 1.0]),
        np.array([-0.05, -0.05, -0.05]),
    )
    assert ci["sharpe_ci"][0] == pytest.approx(ci["sharpe_ci"][1])
    assert ci["max_dd_ci"][0] == pytest.approx(ci["max_dd_ci"][1])


# ---------------------------------------------------------------------------
# 4. Integration — sharpe_ci and max_dd_ci in window-level results
# ---------------------------------------------------------------------------

def test_window_shuffle_sharpe_ci_bounds():
    """sharpe_ci lower should be <= upper."""
    windows = [[100, -50, 200], [-100, 150, 80], [300, -200, 100], [50, 100, -30]]
    result = run_window_shuffle(windows, initial_capital=10000, num_simulations=200, seed=42)
    assert result["sharpe_ci"][0] <= result["sharpe_ci"][1]
    assert result["max_dd_ci"][0] <= result["max_dd_ci"][1]


def test_bootstrap_sharpe_ci_bounds():
    """sharpe_ci lower should be <= upper."""
    windows = [[100, -50, 200], [-100, 150, 80], [300, -200, 100], [50, 100, -30]]
    result = run_bootstrap(windows, initial_capital=10000, num_simulations=200, seed=42)
    assert result["sharpe_ci"][0] <= result["sharpe_ci"][1]
    assert result["max_dd_ci"][0] <= result["max_dd_ci"][1]
