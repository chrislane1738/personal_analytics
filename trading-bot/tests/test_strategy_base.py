"""Tests for strategy/base.py and strategy/registry.py."""

from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.events import BarEvent, FillEvent, SignalEvent
from strategy.base import BacktestContext, Strategy
from strategy.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(symbol: str = "AAPL") -> BarEvent:
    return BarEvent(symbol=symbol, date=date(2024, 1, 2), close=150.0)


def _make_portfolio():
    return MagicMock()


# ---------------------------------------------------------------------------
# Concrete test strategy (defined at module level for importlib tests)
# ---------------------------------------------------------------------------

class SimpleTestStrategy(Strategy):
    """Returns a single hardcoded BUY signal for any bar."""

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        return [SignalEvent(symbol=bar.symbol, direction="long", strength=1.0)]


# ---------------------------------------------------------------------------
# 1. ABC enforcement
# ---------------------------------------------------------------------------

class TestStrategyABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_subclass_without_generate_signals_raises(self):
        class Incomplete(Strategy):
            pass  # does not implement generate_signals

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# 2. Concrete subclass works
# ---------------------------------------------------------------------------

class TestConcreteStrategy:
    def test_instantiation(self):
        s = SimpleTestStrategy()
        assert isinstance(s, Strategy)

    def test_generate_signals_returns_signal(self):
        s = SimpleTestStrategy()
        bar = _make_bar("TSLA")
        portfolio = _make_portfolio()
        signals = s.generate_signals(bar, portfolio)

        assert len(signals) == 1
        assert signals[0].symbol == "TSLA"
        assert signals[0].direction == "long"

    def test_indicators_dict_initialized(self):
        s = SimpleTestStrategy()
        assert isinstance(s.indicators, dict)
        assert len(s.indicators) == 0


# ---------------------------------------------------------------------------
# 3. Default on_universe_bar delegates per-symbol
# ---------------------------------------------------------------------------

class TestOnUniverseBar:
    def test_delegates_to_generate_signals_for_each_symbol(self):
        s = SimpleTestStrategy()
        portfolio = _make_portfolio()
        bars = {
            "AAPL": _make_bar("AAPL"),
            "MSFT": _make_bar("MSFT"),
        }

        signals = s.on_universe_bar(bars, portfolio)

        assert len(signals) == 2
        symbols = {sig.symbol for sig in signals}
        assert symbols == {"AAPL", "MSFT"}

    def test_empty_bars_returns_empty_list(self):
        s = SimpleTestStrategy()
        signals = s.on_universe_bar({}, _make_portfolio())
        assert signals == []


# ---------------------------------------------------------------------------
# 4. on_end default returns empty list
# ---------------------------------------------------------------------------

class TestOnEnd:
    def test_returns_empty_list(self):
        s = SimpleTestStrategy()
        result = s.on_end(_make_portfolio())
        assert result == []
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 5. warm_up_period default returns 0
# ---------------------------------------------------------------------------

class TestWarmUpPeriod:
    def test_default_is_zero(self):
        s = SimpleTestStrategy()
        assert s.warm_up_period() == 0

    def test_subclass_can_override(self):
        class SlowStrategy(Strategy):
            def generate_signals(self, bar, portfolio):
                return []

            def warm_up_period(self) -> int:
                return 50

        s = SlowStrategy()
        assert s.warm_up_period() == 50


# ---------------------------------------------------------------------------
# 6. BacktestContext dataclass
# ---------------------------------------------------------------------------

class TestBacktestContext:
    def test_required_fields(self):
        ctx = BacktestContext(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            universe=["AAPL", "MSFT"],
            initial_capital=100_000.0,
        )
        assert ctx.start_date == date(2023, 1, 1)
        assert ctx.end_date == date(2023, 12, 31)
        assert ctx.universe == ["AAPL", "MSFT"]
        assert ctx.initial_capital == 100_000.0
        assert ctx.benchmark == "SPY"  # default

    def test_custom_benchmark(self):
        ctx = BacktestContext(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            universe=["QQQ"],
            initial_capital=50_000.0,
            benchmark="QQQ",
        )
        assert ctx.benchmark == "QQQ"

    def test_on_start_receives_context(self):
        calls = []

        class TrackingStrategy(Strategy):
            def generate_signals(self, bar, portfolio):
                return []

            def on_start(self, context: BacktestContext) -> None:
                calls.append(context)

        ctx = BacktestContext(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 30),
            universe=["SPY"],
            initial_capital=10_000.0,
        )
        s = TrackingStrategy()
        s.on_start(ctx)
        assert calls == [ctx]


# ---------------------------------------------------------------------------
# 7. Registry — Python module path
# ---------------------------------------------------------------------------

class TestRegistryPythonPath:
    def test_resolves_strategy_from_module_path(self):
        # Inject this test module's SimpleTestStrategy into a synthetic module
        # so StrategyRegistry can import it via a dotted path.
        module_name = "tests._strategy_fixture"
        mod = types.ModuleType(module_name)
        mod.SimpleTestStrategy = SimpleTestStrategy  # type: ignore[attr-defined]
        sys.modules[module_name] = mod

        try:
            result = StrategyRegistry.resolve(f"{module_name}.SimpleTestStrategy")
            assert isinstance(result, SimpleTestStrategy)
        finally:
            del sys.modules[module_name]

    def test_resolve_returns_instance_not_class(self):
        module_name = "tests._strategy_fixture2"
        mod = types.ModuleType(module_name)
        mod.SimpleTestStrategy = SimpleTestStrategy  # type: ignore[attr-defined]
        sys.modules[module_name] = mod

        try:
            result = StrategyRegistry.resolve(f"{module_name}.SimpleTestStrategy")
            assert isinstance(result, Strategy)
        finally:
            del sys.modules[module_name]


# ---------------------------------------------------------------------------
# 8. Registry — YAML path exists (returns None placeholder)
# ---------------------------------------------------------------------------

class TestRegistryYamlExists:
    def test_existing_yaml_returns_none(self):
        yaml_path = Path(__file__).parent / "fixtures" / "sample_strategy.yaml"
        assert yaml_path.exists(), f"Fixture missing: {yaml_path}"

        result = StrategyRegistry.resolve(str(yaml_path))
        assert result is None  # Task 9 placeholder

    def test_yml_extension_also_accepted(self, tmp_path):
        yml_file = tmp_path / "strat.yml"
        yml_file.write_text("name: test\n")

        result = StrategyRegistry.resolve(str(yml_file))
        assert result is None


# ---------------------------------------------------------------------------
# 9. Registry — YAML not found raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestRegistryYamlNotFound:
    def test_missing_yaml_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Strategy config not found"):
            StrategyRegistry.resolve("/nonexistent/path/strategy.yaml")

    def test_missing_yml_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            StrategyRegistry.resolve("/nonexistent/path/strategy.yml")


# ---------------------------------------------------------------------------
# 10. Registry — invalid / unresolvable paths raise ValueError
# ---------------------------------------------------------------------------

class TestRegistryInvalidPath:
    def test_nonexistent_module_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot resolve strategy"):
            StrategyRegistry.resolve("does.not.exist.SomeStrategy")

    def test_no_dot_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid strategy reference"):
            StrategyRegistry.resolve("NoDotHere")

    def test_non_strategy_class_raises_type_error(self):
        module_name = "tests._not_a_strategy"
        mod = types.ModuleType(module_name)

        class NotAStrategy:
            pass

        mod.NotAStrategy = NotAStrategy  # type: ignore[attr-defined]
        sys.modules[module_name] = mod

        try:
            with pytest.raises(TypeError, match="is not a Strategy subclass"):
                StrategyRegistry.resolve(f"{module_name}.NotAStrategy")
        finally:
            del sys.modules[module_name]
