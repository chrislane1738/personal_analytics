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
