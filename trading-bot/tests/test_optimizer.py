"""Tests for core/optimizer.py — ParameterOptimizer (grid search + walk-forward).

Test cases:
1. Grid search basic: 2x2 param grid returns 4 results ranked by objective.
2. Edge parameter warning: best params at grid boundary -> edge_warning=True.
3. Edge parameter no warning: best params in middle -> edge_warning=False.
4. Objective metric: results are ranked by the specified objective.
5. Walk-forward basic: at least one window produced; WalkForwardResult populated.
6. Single combo grid: degenerate grid with one combination still works.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from core.optimizer import (
    OptimizationResult,
    ParameterOptimizer,
    WalkForwardResult,
    _add_months,
    _apply_params,
    _deep_copy_config,
)
from data.storage.database import Database
from tests.fixtures.sample_bars import generate_spy_bars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRATEGY_YAML = Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml"


@pytest.fixture
def tmp_db():
    """Temporary SQLite DB with SPY bars pre-loaded."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    db.create_tables()
    bars = generate_spy_bars("SPY")
    db.insert_daily_bars(bars)
    yield db
    db.close()


@pytest.fixture
def spy_dates():
    """First and last date from the SPY fixture bars."""
    bars = generate_spy_bars("SPY")
    return bars[0].date, bars[-1].date


@pytest.fixture
def optimizer(tmp_db):
    """ParameterOptimizer wired to the temp DB + SPY universe."""
    return ParameterOptimizer(
        strategy_config_path=str(STRATEGY_YAML),
        database=tmp_db,
        universe=["SPY"],
        initial_capital=100_000.0,
        objective="sharpe_ratio",
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_optimizer_with_mock_grid_search(
    tmp_db,
    window_results: list[dict],
) -> ParameterOptimizer:
    """Return an optimizer whose grid_search is replaced by a mock.

    *window_results* is a list of OptimizationResult objects or plain dicts
    that will be returned in sequence for successive grid_search calls.
    """
    opt = ParameterOptimizer(
        strategy_config_path=str(STRATEGY_YAML),
        database=tmp_db,
        universe=["SPY"],
        initial_capital=100_000.0,
        objective="sharpe_ratio",
    )
    opt.grid_search = MagicMock(side_effect=window_results)
    return opt


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_deep_copy_is_independent(self):
        original = {"indicators": {"rsi_14": {"type": "RSI", "period": 14}}}
        copy_ = _deep_copy_config(original)
        copy_["indicators"]["rsi_14"]["period"] = 99
        assert original["indicators"]["rsi_14"]["period"] == 14

    def test_apply_params_rsi_period(self):
        config = {"indicators": {"rsi_14": {"type": "RSI", "period": 14}}}
        _apply_params(config, {"rsi_period": 10})
        assert config["indicators"]["rsi_14"]["period"] == 10

    def test_apply_params_sma_period(self):
        config = {
            "indicators": {
                "sma_20": {"type": "SMA", "period": 20},
                "sma_50": {"type": "SMA", "period": 50},
            }
        }
        _apply_params(config, {"sma_period": 15})
        # Should update the first indicator whose name starts with "sma"
        assert config["indicators"]["sma_20"]["period"] == 15
        # sma_50 should be unchanged
        assert config["indicators"]["sma_50"]["period"] == 50

    def test_apply_params_unknown_prefix_ignored(self):
        config = {"indicators": {"rsi_14": {"type": "RSI", "period": 14}}}
        _apply_params(config, {"zzz_foo": 999})
        assert config["indicators"]["rsi_14"]["period"] == 14

    def test_apply_params_no_underscore_ignored(self):
        config = {"indicators": {"rsi_14": {"type": "RSI", "period": 14}}}
        _apply_params(config, {"nounderscore": 5})
        assert config["indicators"]["rsi_14"]["period"] == 14

    def test_add_months_simple(self):
        assert _add_months(date(2024, 1, 1), 3) == date(2024, 4, 1)

    def test_add_months_year_rollover(self):
        assert _add_months(date(2024, 11, 1), 3) == date(2025, 2, 1)

    def test_add_months_clamps_to_month_end(self):
        # Jan 31 + 1 month = Feb 28 (2024 is a leap year → Feb 29)
        result = _add_months(date(2024, 1, 31), 1)
        assert result == date(2024, 2, 29)


# ---------------------------------------------------------------------------
# Grid search tests
# ---------------------------------------------------------------------------

class TestGridSearch:
    """Tests for ParameterOptimizer.grid_search."""

    def test_two_by_two_returns_four_results(self, optimizer, spy_dates):
        """A 2x2 param grid should produce exactly 4 combinations."""
        start, end = spy_dates
        param_grid = {"rsi_period": [10, 14], "sma_period": [15, 20]}

        result = optimizer.grid_search(param_grid, start, end)

        assert isinstance(result, OptimizationResult)
        assert len(result.param_combinations) == 4
        assert len(result.metrics) == 4
        assert len(result.ranked_results) == 4

    def test_result_has_best_params(self, optimizer, spy_dates):
        """best_params should be a non-empty dict after a successful run."""
        start, end = spy_dates
        param_grid = {"rsi_period": [10, 14]}

        result = optimizer.grid_search(param_grid, start, end)

        assert isinstance(result.best_params, dict)
        assert "rsi_period" in result.best_params
        assert result.best_params["rsi_period"] in [10, 14]

    def test_ranked_by_objective_descending(self, optimizer, spy_dates):
        """ranked_results must be sorted by the objective metric, best first."""
        start, end = spy_dates
        param_grid = {"rsi_period": [10, 14], "sma_period": [15, 20]}

        result = optimizer.grid_search(param_grid, start, end)

        objective_values = [
            r["metrics"].get(optimizer.objective, float("-inf"))
            for r in result.ranked_results
        ]
        assert objective_values == sorted(objective_values, reverse=True), (
            f"ranked_results not sorted descending: {objective_values}"
        )

    def test_best_metric_value_matches_top_ranked(self, optimizer, spy_dates):
        """best_metric_value should equal the top entry in ranked_results."""
        start, end = spy_dates
        param_grid = {"rsi_period": [10, 14]}

        result = optimizer.grid_search(param_grid, start, end)

        top_metric = result.ranked_results[0]["metrics"].get(optimizer.objective, 0)
        assert result.best_metric_value == top_metric

    def test_objective_stored_in_result(self, optimizer, spy_dates):
        """OptimizationResult.objective should match the optimizer's objective."""
        start, end = spy_dates
        result = optimizer.grid_search({"rsi_period": [10, 14]}, start, end)
        assert result.objective == "sharpe_ratio"

    def test_single_combination(self, optimizer, spy_dates):
        """A single-value grid (1 combination) should still complete."""
        start, end = spy_dates
        result = optimizer.grid_search({"rsi_period": [14]}, start, end)

        assert len(result.param_combinations) == 1
        assert len(result.metrics) == 1
        assert result.best_params == {"rsi_period": 14}


# ---------------------------------------------------------------------------
# Edge-of-grid warning tests
# ---------------------------------------------------------------------------

class TestEdgeWarning:
    """Tests for the edge_warning flag in OptimizationResult."""

    def test_edge_warning_true_when_best_at_boundary(self, tmp_db):
        """When the best combo is at the min/max of any axis, edge_warning=True."""
        # Mock grid_search internals: patch _backtest_with_params so the combo
        # at the grid edge (rsi_period=10, sma_period=15) gets the best score.
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        # rsi_period values: [10, 14, 20] — 10 is the minimum (edge)
        # We want 10 to win; patch _backtest_with_params to return controlled metrics.
        call_count = [0]

        def fake_backtest(base_config, params, database, universe, start, end, capital):
            call_count[0] += 1
            # Best score for rsi_period=10
            if params.get("rsi_period") == 10:
                return {"sharpe_ratio": 2.5, "total_return": 0.10}
            return {"sharpe_ratio": 0.5, "total_return": 0.02}

        with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
            result = opt.grid_search(
                {"rsi_period": [10, 14, 20]},
                date(2024, 1, 2),
                date(2024, 5, 31),
            )

        assert result.best_params["rsi_period"] == 10
        assert result.edge_warning is True

    def test_edge_warning_false_when_best_in_middle(self, tmp_db):
        """When the best combo is an interior point of the grid, edge_warning=False."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        # rsi_period values: [10, 14, 20] — 14 is the middle (not an edge)
        def fake_backtest(base_config, params, database, universe, start, end, capital):
            if params.get("rsi_period") == 14:
                return {"sharpe_ratio": 2.5, "total_return": 0.10}
            return {"sharpe_ratio": 0.5, "total_return": 0.02}

        with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
            result = opt.grid_search(
                {"rsi_period": [10, 14, 20]},
                date(2024, 1, 2),
                date(2024, 5, 31),
            )

        assert result.best_params["rsi_period"] == 14
        assert result.edge_warning is False

    def test_edge_warning_false_for_single_value_param(self, tmp_db):
        """A single-value param has no real edges; edge_warning should stay False."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        def fake_backtest(base_config, params, database, universe, start, end, capital):
            return {"sharpe_ratio": 1.5}

        with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
            result = opt.grid_search(
                {"rsi_period": [14]},
                date(2024, 1, 2),
                date(2024, 5, 31),
            )

        # Single-value grid should not trigger edge_warning
        assert result.edge_warning is False


# ---------------------------------------------------------------------------
# Objective metric tests
# ---------------------------------------------------------------------------

class TestObjectiveMetric:
    """Verify ranking respects the chosen objective."""

    def test_total_return_objective(self, tmp_db):
        """With objective='total_return', ranked_results sorted by total_return."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="total_return",
        )
        start, end = generate_spy_bars("SPY")[0].date, generate_spy_bars("SPY")[-1].date

        call_index = [0]
        scores = [0.15, 0.05, 0.20, 0.08]

        def fake_backtest(base_config, params, database, universe, s, e, cap):
            idx = call_index[0]
            call_index[0] += 1
            return {"sharpe_ratio": 0.5, "total_return": scores[idx]}

        with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
            result = opt.grid_search(
                {"rsi_period": [10, 14], "sma_period": [15, 20]},
                start,
                end,
            )

        assert result.objective == "total_return"
        tr_values = [r["metrics"]["total_return"] for r in result.ranked_results]
        assert tr_values == sorted(tr_values, reverse=True)
        assert result.best_metric_value == max(scores)

    def test_sortino_ratio_objective(self, tmp_db):
        """With objective='sortino_ratio', best_params selected by sortino."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sortino_ratio",
        )

        def fake_backtest(base_config, params, database, universe, s, e, cap):
            score = 3.0 if params.get("rsi_period") == 14 else 1.0
            return {"sortino_ratio": score, "sharpe_ratio": 0.5}

        with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
            result = opt.grid_search(
                {"rsi_period": [10, 14, 20]},
                date(2024, 1, 2),
                date(2024, 5, 31),
            )

        assert result.best_params["rsi_period"] == 14
        assert result.best_metric_value == 3.0


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------

class TestWalkForward:
    """Tests for ParameterOptimizer.walk_forward."""

    def _make_opt_result(self, best_params: dict, best_value: float) -> OptimizationResult:
        return OptimizationResult(
            param_combinations=[best_params],
            metrics=[{"sharpe_ratio": best_value}],
            best_params=best_params,
            best_metric_value=best_value,
            objective="sharpe_ratio",
            edge_warning=False,
            ranked_results=[{"params": best_params, "metrics": {"sharpe_ratio": best_value}}],
        )

    def test_walk_forward_produces_windows(self, tmp_db):
        """Walk-forward with mocked grid_search should produce at least one window."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        # Provide enough mock results for two grid_search calls (two windows)
        mock_results = [
            self._make_opt_result({"rsi_period": 14}, 1.5),
            self._make_opt_result({"rsi_period": 10}, 1.2),
        ]

        def fake_backtest(base_config, params, database, universe, s, e, cap):
            return {"sharpe_ratio": 0.8, "total_return": 0.05}

        with patch.object(opt, "grid_search", side_effect=mock_results):
            with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
                result = opt.walk_forward(
                    param_grid={"rsi_period": [10, 14]},
                    start_date=date(2020, 1, 1),
                    end_date=date(2022, 6, 1),
                    train_months=12,
                    test_months=6,
                )

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) >= 1
        assert result.objective == "sharpe_ratio"

    def test_walk_forward_window_fields(self, tmp_db):
        """Each window dict must contain the required keys."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        mock_results = [self._make_opt_result({"rsi_period": 14}, 1.5)]

        def fake_backtest(base_config, params, database, universe, s, e, cap):
            return {"sharpe_ratio": 0.9}

        with patch.object(opt, "grid_search", side_effect=mock_results):
            with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
                result = opt.walk_forward(
                    param_grid={"rsi_period": [10, 14]},
                    start_date=date(2020, 1, 1),
                    end_date=date(2021, 6, 1),
                    train_months=12,
                    test_months=6,
                )

        required_keys = {
            "train_start", "train_end", "test_start", "test_end",
            "best_params", "in_sample", "out_of_sample",
        }
        for window in result.windows:
            assert required_keys.issubset(window.keys()), (
                f"Window missing keys: {required_keys - window.keys()}"
            )

    def test_walk_forward_degradation_computed(self, tmp_db):
        """degradation_pct = (in - out) / |in| * 100 must be calculated correctly."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        # in_sample=2.0 for both windows, out_of_sample=1.0
        # avg_in=2.0, avg_out=1.0 -> degradation=(2-1)/2*100=50%
        mock_results = [
            self._make_opt_result({"rsi_period": 14}, 2.0),
            self._make_opt_result({"rsi_period": 14}, 2.0),
        ]

        call_count = [0]

        def fake_backtest(base_config, params, database, universe, s, e, cap):
            call_count[0] += 1
            return {"sharpe_ratio": 1.0}

        with patch.object(opt, "grid_search", side_effect=mock_results):
            with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
                result = opt.walk_forward(
                    param_grid={"rsi_period": [10, 14]},
                    start_date=date(2020, 1, 1),
                    end_date=date(2022, 6, 1),
                    train_months=12,
                    test_months=6,
                )

        assert result.avg_in_sample_metric == pytest.approx(2.0)
        assert result.avg_out_of_sample_metric == pytest.approx(1.0)
        assert result.degradation_pct == pytest.approx(50.0)

    def test_walk_forward_overfitting_detected(self, tmp_db):
        """overfitting_detected=True when degradation > threshold (50%)."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        # in=2.0, out=0.5 -> degradation=75% > 50% threshold
        mock_results = [self._make_opt_result({"rsi_period": 14}, 2.0)]

        def fake_backtest(base_config, params, database, universe, s, e, cap):
            return {"sharpe_ratio": 0.5}

        with patch.object(opt, "grid_search", side_effect=mock_results):
            with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
                result = opt.walk_forward(
                    param_grid={"rsi_period": [10, 14]},
                    start_date=date(2020, 1, 1),
                    end_date=date(2021, 9, 1),  # wide enough for 12+6 = 18 months
                    train_months=12,
                    test_months=6,
                    overfitting_threshold=50.0,
                )

        assert result.overfitting_detected is True

    def test_walk_forward_no_overfitting(self, tmp_db):
        """overfitting_detected=False when degradation <= threshold."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        # in=2.0, out=1.9 -> degradation=5% < 50% threshold
        mock_results = [self._make_opt_result({"rsi_period": 14}, 2.0)]

        def fake_backtest(base_config, params, database, universe, s, e, cap):
            return {"sharpe_ratio": 1.9}

        with patch.object(opt, "grid_search", side_effect=mock_results):
            with patch("core.optimizer._backtest_with_params", side_effect=fake_backtest):
                result = opt.walk_forward(
                    param_grid={"rsi_period": [10, 14]},
                    start_date=date(2020, 1, 1),
                    end_date=date(2021, 6, 1),
                    train_months=12,
                    test_months=6,
                    overfitting_threshold=50.0,
                )

        assert result.overfitting_detected is False

    def test_walk_forward_no_data_period(self, tmp_db):
        """When the date range is too short for even one window, return empty result."""
        opt = ParameterOptimizer(
            strategy_config_path=str(STRATEGY_YAML),
            database=tmp_db,
            universe=["SPY"],
            initial_capital=100_000.0,
            objective="sharpe_ratio",
        )

        result = opt.walk_forward(
            param_grid={"rsi_period": [10, 14]},
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),  # only 1 month — not enough for 12+6
            train_months=12,
            test_months=6,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.windows == []
        assert result.degradation_pct == 0.0
        assert result.overfitting_detected is False


# ---------------------------------------------------------------------------
# Integration smoke test (real backtest, no mocks)
# ---------------------------------------------------------------------------

class TestGridSearchIntegration:
    """Light integration test that runs actual backtests end-to-end."""

    def test_grid_search_real_run_2x1(self, optimizer, spy_dates):
        """2x1 grid runs two real backtests against the sample DB."""
        start, end = spy_dates
        result = optimizer.grid_search(
            {"rsi_period": [10, 14]},
            start,
            end,
        )

        assert len(result.param_combinations) == 2
        assert len(result.ranked_results) == 2
        # Verify metrics dict contains expected keys
        for metrics in result.metrics:
            assert "sharpe_ratio" in metrics
            assert "total_return" in metrics
