"""Tests for Monte Carlo percentile bands (Task 0.4).

Validates that run_monte_carlo returns per-time-step percentile bands
and an actual equity curve, enabling fan-chart visualization.
"""

from __future__ import annotations

import pytest

from analytics.monte_carlo import MonteCarloResult, run_monte_carlo


# ---------------------------------------------------------------------------
# 1. New fields exist and have correct types
# ---------------------------------------------------------------------------

class TestPercentileBandsStructure:
    """Verify the new fields are present, correctly typed, and sized."""

    @pytest.fixture
    def result(self):
        trades = [500.0, -200.0, 300.0, -100.0, 150.0]
        return run_monte_carlo(trades, initial_capital=100_000.0,
                               num_simulations=500, seed=42)

    def test_percentile_bands_is_dict(self, result):
        assert isinstance(result.percentile_bands, dict)

    def test_percentile_bands_has_required_keys(self, result):
        expected_keys = {"p5", "p25", "p50", "p75", "p95"}
        assert set(result.percentile_bands.keys()) == expected_keys

    def test_percentile_bands_values_are_lists_of_floats(self, result):
        for key, values in result.percentile_bands.items():
            assert isinstance(values, list), f"{key} is not a list"
            assert all(isinstance(v, float) for v in values), f"{key} contains non-float"

    def test_percentile_bands_length_equals_num_trades(self, result):
        """Each band should have one value per trade step."""
        for key, values in result.percentile_bands.items():
            assert len(values) == 5, f"{key} has {len(values)} values, expected 5"

    def test_actual_curve_is_list_of_floats(self, result):
        assert isinstance(result.actual_curve, list)
        assert all(isinstance(v, float) for v in result.actual_curve)

    def test_actual_curve_length_equals_num_trades(self, result):
        assert len(result.actual_curve) == 5


# ---------------------------------------------------------------------------
# 2. Percentile ordering per time step
# ---------------------------------------------------------------------------

class TestPercentileOrdering:
    """p5 <= p25 <= p50 <= p75 <= p95 at every time step."""

    def test_monotonic_ordering_at_each_step(self):
        trades = [1000.0, -500.0, 800.0, -300.0, 600.0, -200.0, 400.0]
        result = run_monte_carlo(trades, initial_capital=50_000.0,
                                 num_simulations=2000, seed=7)

        bands = result.percentile_bands
        for i in range(len(trades)):
            assert bands["p5"][i] <= bands["p25"][i], f"p5 > p25 at step {i}"
            assert bands["p25"][i] <= bands["p50"][i], f"p25 > p50 at step {i}"
            assert bands["p50"][i] <= bands["p75"][i], f"p50 > p75 at step {i}"
            assert bands["p75"][i] <= bands["p95"][i], f"p75 > p95 at step {i}"


# ---------------------------------------------------------------------------
# 3. Identical trades — all bands converge
# ---------------------------------------------------------------------------

class TestIdenticalTrades:
    """When all trades are the same value, all shuffles are identical."""

    def test_uniform_trades_all_bands_equal(self):
        """10 trades of +$1000: all permutations produce the same equity curve."""
        trades = [1000.0] * 10
        result = run_monte_carlo(trades, initial_capital=100_000.0,
                                 num_simulations=500, seed=42)

        bands = result.percentile_bands
        for i in range(len(trades)):
            expected = 100_000.0 + 1000.0 * (i + 1)
            for key in ("p5", "p25", "p50", "p75", "p95"):
                assert bands[key][i] == pytest.approx(expected), \
                    f"{key} at step {i}: {bands[key][i]} != {expected}"

    def test_actual_curve_matches_bands_for_uniform(self):
        trades = [1000.0] * 5
        result = run_monte_carlo(trades, initial_capital=100_000.0,
                                 num_simulations=100, seed=42)

        for i, val in enumerate(result.actual_curve):
            assert val == pytest.approx(result.percentile_bands["p50"][i])


# ---------------------------------------------------------------------------
# 4. Actual curve correctness
# ---------------------------------------------------------------------------

class TestActualCurve:
    """The actual_curve should be the cumulative equity from the original trade order."""

    def test_actual_curve_values(self):
        trades = [500.0, -200.0, 300.0]
        result = run_monte_carlo(trades, initial_capital=100_000.0,
                                 num_simulations=100, seed=42)

        expected = [100_500.0, 100_300.0, 100_600.0]
        assert result.actual_curve == pytest.approx(expected)

    def test_actual_curve_final_matches_actual_final_equity(self):
        trades = [500.0, -200.0, 300.0, -100.0, 150.0]
        result = run_monte_carlo(trades, initial_capital=100_000.0,
                                 num_simulations=100, seed=42)

        assert result.actual_curve[-1] == pytest.approx(result.actual_final_equity)


# ---------------------------------------------------------------------------
# 5. Empty trades edge case
# ---------------------------------------------------------------------------

class TestEmptyTradesBands:
    """Empty trade list should produce empty bands and empty actual_curve."""

    def test_empty_trades_empty_bands(self):
        result = run_monte_carlo([], initial_capital=100_000.0,
                                 num_simulations=100, seed=42)

        for key in ("p5", "p25", "p50", "p75", "p95"):
            assert result.percentile_bands[key] == []

    def test_empty_trades_empty_actual_curve(self):
        result = run_monte_carlo([], initial_capital=100_000.0,
                                 num_simulations=100, seed=42)

        assert result.actual_curve == []


# ---------------------------------------------------------------------------
# 6. Reproducibility — same seed produces same bands
# ---------------------------------------------------------------------------

class TestBandsReproducibility:
    def test_same_seed_same_bands(self):
        trades = [500.0, -200.0, 300.0, -100.0, 150.0]
        result_a = run_monte_carlo(trades, seed=42, num_simulations=500)
        result_b = run_monte_carlo(trades, seed=42, num_simulations=500)

        assert result_a.percentile_bands == result_b.percentile_bands
        assert result_a.actual_curve == result_b.actual_curve


# ---------------------------------------------------------------------------
# 7. Final step of p50 band ≈ median_final_equity
# ---------------------------------------------------------------------------

class TestBandFinalStepConsistency:
    def test_p50_final_step_equals_median_final_equity(self):
        """The last value of the p50 band should equal the median final equity."""
        trades = [500.0, -200.0, 300.0, -100.0, 150.0, 400.0, -300.0]
        result = run_monte_carlo(trades, initial_capital=100_000.0,
                                 num_simulations=5000, seed=42)

        assert result.percentile_bands["p50"][-1] == pytest.approx(
            result.median_final_equity, rel=1e-6
        )

    def test_p5_final_step_equals_percentile_5(self):
        trades = [500.0, -200.0, 300.0, -100.0, 150.0, 400.0, -300.0]
        result = run_monte_carlo(trades, initial_capital=100_000.0,
                                 num_simulations=5000, seed=42)

        assert result.percentile_bands["p5"][-1] == pytest.approx(
            result.percentile_5, rel=1e-6
        )

    def test_p95_final_step_equals_percentile_95(self):
        trades = [500.0, -200.0, 300.0, -100.0, 150.0, 400.0, -300.0]
        result = run_monte_carlo(trades, initial_capital=100_000.0,
                                 num_simulations=5000, seed=42)

        assert result.percentile_bands["p95"][-1] == pytest.approx(
            result.percentile_95, rel=1e-6
        )
