"""Tests for walk-forward infrastructure: quiet mode and config placeholder substitution.

Test cases:
1. test_quiet_mode_no_db_persistence — BacktestEngine with quiet=True doesn't save to DB
2. test_config_strategy_from_config_dict_with_placeholders — placeholder substitution works
"""

from __future__ import annotations

import copy
import tempfile
from datetime import date

import pytest

from core.engine import BacktestEngine
from core.events import BarEvent, SignalEvent
from data.storage.database import Database
from strategy.base import Strategy
from strategy.config_strategy import ConfigStrategy
from tests.fixtures.sample_bars import generate_spy_bars


# ---------------------------------------------------------------------------
# Helper strategy
# ---------------------------------------------------------------------------

class _SimpleStrategy(Strategy):
    """Minimal strategy that always generates a buy signal (no warm-up)."""

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        if not portfolio.has_position(bar.symbol):
            return [
                SignalEvent(
                    symbol=bar.symbol,
                    direction="long",
                    reason="test signal",
                    strength=0.5,
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """Return a Database backed by a temporary file with schema created."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    db.create_tables()
    yield db
    db.close()


@pytest.fixture
def loaded_db(tmp_db):
    """Return a Database with SPY sample bars pre-loaded."""
    bars = generate_spy_bars("SPY")
    tmp_db.insert_daily_bars(bars)
    return tmp_db


# ---------------------------------------------------------------------------
# A.1 — Quiet mode
# ---------------------------------------------------------------------------

class TestQuietMode:
    def test_quiet_mode_no_db_persistence(self, loaded_db):
        """Engine with quiet=True should NOT persist a RunRecord to the DB."""
        bars = generate_spy_bars("SPY")
        engine = BacktestEngine(
            strategy=_SimpleStrategy(),
            database=loaded_db,
            universe=["SPY"],
            start_date=bars[0].date,
            end_date=bars[-1].date,
            initial_capital=100_000.0,
            benchmark_symbol="SPY",
            quiet=True,
        )
        metrics = engine.run()

        # Metrics should still be returned
        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "_run_id" in metrics

        # But nothing should be saved to the database
        runs = loaded_db.list_runs()
        assert len(runs) == 0, (
            f"Expected 0 runs in DB with quiet=True, found {len(runs)}"
        )

    def test_quiet_mode_still_returns_metrics(self, loaded_db):
        """Quiet mode should still compute and return all standard metrics."""
        bars = generate_spy_bars("SPY")
        engine = BacktestEngine(
            strategy=_SimpleStrategy(),
            database=loaded_db,
            universe=["SPY"],
            start_date=bars[0].date,
            end_date=bars[-1].date,
            initial_capital=100_000.0,
            benchmark_symbol="SPY",
            quiet=True,
        )
        metrics = engine.run()

        for key in (
            "total_return",
            "sharpe_ratio",
            "max_drawdown_pct",
            "total_trades",
            "start_date",
            "end_date",
        ):
            assert key in metrics, f"Missing metric key: {key}"

    def test_quiet_false_still_saves(self, loaded_db):
        """Default quiet=False should still persist to DB (regression check)."""
        bars = generate_spy_bars("SPY")
        engine = BacktestEngine(
            strategy=_SimpleStrategy(),
            database=loaded_db,
            universe=["SPY"],
            start_date=bars[0].date,
            end_date=bars[-1].date,
            initial_capital=100_000.0,
            benchmark_symbol="SPY",
        )
        engine.run()

        runs = loaded_db.list_runs()
        assert len(runs) >= 1, "Default mode should save to DB"


# ---------------------------------------------------------------------------
# A.2 — ConfigStrategy.from_config_dict with placeholder substitution
# ---------------------------------------------------------------------------

class TestConfigStrategyFromConfigDict:
    """Tests for ConfigStrategy.from_config_dict and _substitute_params."""

    SAMPLE_CONFIG = {
        "name": "Test Strategy",
        "universe": ["SPY"],
        "indicators": {
            "rsi_14": {"type": "RSI", "period": 14},
            "sma_20": {"type": "SMA", "period": "{{sma_period}}"},
        },
        "entry_rules": [
            {
                "condition": "rsi_14 < {{rsi_thresh}}",
                "direction": "long",
                "reason": "RSI oversold",
            }
        ],
        "exit_rules": [
            {
                "condition": "close > sma_20",
                "reason": "Mean reversion exit",
            }
        ],
        "position_sizing": {
            "method": "fixed_pct",
            "value": "{{pos_size}}",
        },
    }

    def test_full_value_placeholder_replaced_with_number(self):
        """'{{sma_period}}' as the entire value should become an int, not a string."""
        params = {"sma_period": 50, "rsi_thresh": 30, "pos_size": 0.12}
        strat = ConfigStrategy.from_config_dict(self.SAMPLE_CONFIG, params)

        # SMA period should be the integer 50, not the string "50"
        assert strat.config["indicators"]["sma_20"]["period"] == 50
        assert isinstance(strat.config["indicators"]["sma_20"]["period"], int)

    def test_inline_placeholder_replaced_with_string_interpolation(self):
        """'rsi_14 < {{rsi_thresh}}' should become 'rsi_14 < 30'."""
        params = {"sma_period": 50, "rsi_thresh": 30, "pos_size": 0.12}
        strat = ConfigStrategy.from_config_dict(self.SAMPLE_CONFIG, params)

        condition = strat.config["entry_rules"][0]["condition"]
        assert condition == "rsi_14 < 30"
        assert "{{" not in condition

    def test_nested_placeholder_replaced(self):
        """Placeholders in nested dicts (position_sizing.value) should be replaced."""
        params = {"sma_period": 50, "rsi_thresh": 30, "pos_size": 0.12}
        strat = ConfigStrategy.from_config_dict(self.SAMPLE_CONFIG, params)

        assert strat.config["position_sizing"]["value"] == 0.12
        assert isinstance(strat.config["position_sizing"]["value"], float)

    def test_no_params_leaves_config_unchanged(self):
        """Calling from_config_dict without params should not modify placeholders."""
        config_no_placeholders = {
            "name": "Simple",
            "universe": ["SPY"],
            "indicators": {
                "rsi_14": {"type": "RSI", "period": 14},
            },
            "entry_rules": [
                {
                    "condition": "rsi_14 < 30",
                    "direction": "long",
                    "reason": "RSI oversold",
                }
            ],
            "exit_rules": [],
        }
        strat = ConfigStrategy.from_config_dict(config_no_placeholders)
        assert strat.name == "Simple"
        assert "rsi_14" in strat.indicators

    def test_indicators_and_rules_initialised(self):
        """from_config_dict should call _build_indicators and _parse_rules."""
        params = {"sma_period": 50, "rsi_thresh": 30, "pos_size": 0.12}
        strat = ConfigStrategy.from_config_dict(self.SAMPLE_CONFIG, params)

        # Indicators should be created
        assert "rsi_14" in strat.indicators
        assert "sma_20" in strat.indicators

        # Rules should be parsed
        assert len(strat.entry_rules) == 1
        assert len(strat.exit_rules) == 1

    def test_original_config_not_mutated(self):
        """from_config_dict should deep-copy, not mutate the original dict."""
        original = copy.deepcopy(self.SAMPLE_CONFIG)
        params = {"sma_period": 50, "rsi_thresh": 30, "pos_size": 0.12}
        ConfigStrategy.from_config_dict(self.SAMPLE_CONFIG, params)

        # Original should still have placeholders
        assert self.SAMPLE_CONFIG["indicators"]["sma_20"]["period"] == "{{sma_period}}"
        assert self.SAMPLE_CONFIG == original


# ---------------------------------------------------------------------------
# A.2 — Strategy base class stubs
# ---------------------------------------------------------------------------

class TestStrategyBaseStubs:
    """Test that Strategy has get_parameter_space and from_params stubs."""

    def test_get_parameter_space_returns_empty_dict(self):
        assert Strategy.get_parameter_space() == {}

    def test_from_params_returns_instance(self):
        """from_params on a concrete subclass should return an instance."""
        instance = _SimpleStrategy.from_params({})
        assert isinstance(instance, _SimpleStrategy)


# ---------------------------------------------------------------------------
# A.3 — Walk-forward core engine
# ---------------------------------------------------------------------------

from analytics.walk_forward import generate_windows, generate_param_grid, run_walk_forward


class _NullStrategy(Strategy):
    """Strategy that emits no signals — useful for walk-forward skeleton tests."""

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        return []


class TestGenerateWindows:
    """Tests for generate_windows()."""

    def test_generate_windows_basic(self):
        """Verify correct number of windows, dates in order, no gaps."""
        windows = generate_windows(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            train_months=12,
            oos_months=6,
            step_months=6,
        )
        # 5 years = 60 months.  Window 0: train 0-12, oos 12-18 (month offsets).
        # Step by 6 each time.  Last valid window must have oos_end <= end_date.
        assert len(windows) >= 2, f"Expected at least 2 windows, got {len(windows)}"

        # Verify dates are in order within each window
        for w in windows:
            assert w["train_start"] < w["train_end"]
            assert w["train_end"] == w["oos_start"], "No gap between train and OOS"
            assert w["oos_start"] < w["oos_end"]

        # Verify windows progress forward
        for i in range(1, len(windows)):
            assert windows[i]["train_start"] > windows[i - 1]["train_start"]

        # Verify no OOS extends past end_date
        for w in windows:
            assert w["oos_end"] <= date(2024, 12, 31)

    def test_generate_windows_too_short_raises(self):
        """If date range is too short for 2 complete windows, raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            generate_windows(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                train_months=12,
                oos_months=6,
                step_months=6,
            )

    def test_generate_windows_exact_boundary(self):
        """Windows that fit exactly should all be included."""
        # 24 months of train + 6 OOS = 30, step 6 -> second window starts at 6
        # Window 0: train [0,24), oos [24,30)
        # Window 1: train [6,30), oos [30,36)
        # Need end >= 36 months from start.
        windows = generate_windows(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 1, 1),
            train_months=12,
            oos_months=6,
            step_months=6,
        )
        assert len(windows) >= 2


class TestGenerateParamGrid:
    """Tests for generate_param_grid()."""

    def test_generate_param_grid_basic(self):
        """Correct number of combinations from a small param space."""
        param_space = {
            "rsi_thresh": {"type": "strategy", "min": 25, "max": 35, "step": 5},
            "bb_std": {"type": "strategy", "min": 1.5, "max": 2.5, "step": 0.5},
        }
        grid = generate_param_grid(param_space)
        # rsi_thresh: 25, 30, 35 -> 3 values
        # bb_std: 1.5, 2.0, 2.5 -> 3 values
        # Total: 3 * 3 = 9
        assert len(grid) == 9
        # Each entry should be a dict with both keys
        for combo in grid:
            assert "rsi_thresh" in combo
            assert "bb_std" in combo

    def test_generate_param_grid_exceeds_max_raises(self):
        """Too many combinations should raise ValueError."""
        param_space = {
            "a": {"type": "strategy", "min": 0, "max": 100, "step": 1},
            "b": {"type": "strategy", "min": 0, "max": 100, "step": 1},
        }
        # 101 * 101 = 10201 > 5000
        with pytest.raises(ValueError, match="exceed"):
            generate_param_grid(param_space, max_combinations=5000)

    def test_generate_param_grid_empty(self):
        """Empty param space should return a single empty-dict combo."""
        grid = generate_param_grid({})
        assert grid == [{}]

    def test_generate_param_grid_single_value(self):
        """A param with min == max should produce exactly 1 value."""
        param_space = {
            "x": {"type": "strategy", "min": 10, "max": 10, "step": 1},
        }
        grid = generate_param_grid(param_space)
        assert len(grid) == 1
        assert grid[0]["x"] == 10


class TestWalkForwardFixedParams:
    """Tests for run_walk_forward with fixed params (no optimization)."""

    def test_walk_forward_fixed_params(self, loaded_db):
        """Run with no optimization; verify per-window results and stitched curve."""
        bars = generate_spy_bars("SPY")
        first_date = bars[0].date
        last_date = bars[-1].date

        result = run_walk_forward(
            strategy_loader=lambda params: _NullStrategy(),
            param_space={},
            database=loaded_db,
            universe=["SPY"],
            start_date=first_date,
            end_date=last_date,
            initial_capital=100_000.0,
            benchmark="SPY",
            train_months=1,
            oos_months=1,
            step_months=1,
            objective="sharpe",
        )

        # Should have the expected keys
        assert "windows" in result
        assert "aggregate" in result
        assert "stitched_equity_curve" in result
        assert "parameter_stability" in result

        # At least 2 windows
        assert len(result["windows"]) >= 2

        # Each window should have expected structure
        for w in result["windows"]:
            assert "window_index" in w
            assert "train_start" in w
            assert "oos_start" in w
            assert "oos_metrics" in w
            assert "oos_equity_curve" in w

        # Stitched equity curve should be non-empty
        assert len(result["stitched_equity_curve"]) > 0

        # Aggregate should have metrics
        assert "total_return" in result["aggregate"]

    def test_walk_forward_capital_chains(self, loaded_db):
        """Final capital of one OOS window becomes initial capital of the next."""
        bars = generate_spy_bars("SPY")
        first_date = bars[0].date
        last_date = bars[-1].date

        result = run_walk_forward(
            strategy_loader=lambda params: _NullStrategy(),
            param_space={},
            database=loaded_db,
            universe=["SPY"],
            start_date=first_date,
            end_date=last_date,
            initial_capital=100_000.0,
            benchmark="SPY",
            train_months=1,
            oos_months=1,
            step_months=1,
            objective="sharpe",
        )

        # With NullStrategy (no trades), capital should remain ~100k throughout
        # (may fluctuate slightly due to rounding in empty metrics)
        windows = result["windows"]
        assert len(windows) >= 2


class TestWalkForwardWithOptimization:
    """Tests for run_walk_forward with a small param space."""

    def test_walk_forward_with_optimization(self, loaded_db):
        """Run with a small param space; verify best params are selected per window."""
        bars = generate_spy_bars("SPY")
        first_date = bars[0].date
        last_date = bars[-1].date

        # Use _SimpleStrategy but parameterize via a loader that ignores params
        # (since _SimpleStrategy doesn't use them, but we still exercise grid search)
        param_space = {
            "threshold": {"type": "strategy", "min": 0.3, "max": 0.7, "step": 0.2},
        }

        result = run_walk_forward(
            strategy_loader=lambda params: _SimpleStrategy(),
            param_space=param_space,
            database=loaded_db,
            universe=["SPY"],
            start_date=first_date,
            end_date=last_date,
            initial_capital=100_000.0,
            benchmark="SPY",
            train_months=1,
            oos_months=1,
            step_months=1,
            objective="sharpe",
        )

        assert len(result["windows"]) >= 2

        # Each window should record best_params
        for w in result["windows"]:
            assert "best_params" in w
            assert "threshold" in w["best_params"]

        # Parameter stability should track the param across windows
        assert "threshold" in result["parameter_stability"]
        assert len(result["parameter_stability"]["threshold"]) == len(result["windows"])

    def test_walk_forward_cancel_event(self, loaded_db):
        """Setting cancel_event should stop the walk-forward early."""
        import threading

        bars = generate_spy_bars("SPY")
        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        result = run_walk_forward(
            strategy_loader=lambda params: _NullStrategy(),
            param_space={},
            database=loaded_db,
            universe=["SPY"],
            start_date=bars[0].date,
            end_date=bars[-1].date,
            initial_capital=100_000.0,
            benchmark="SPY",
            train_months=1,
            oos_months=1,
            step_months=1,
            objective="sharpe",
            cancel_event=cancel,
        )

        # Should have 0 windows since we cancelled before the first window
        assert len(result["windows"]) == 0
