"""Tests for ORBStrategy (Opening Range Breakout)."""

from __future__ import annotations

from datetime import date

import pytest

from core.events import BarEvent
from topstep.strategies.orb_strategy import ORBStrategy
from topstep.state_manager import StateManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(
    symbol: str = "ES",
    d: date = date(2024, 6, 1),
    o: float = 5000.0,
    h: float = 5050.0,
    l: float = 4960.0,
    c: float = 5030.0,
    v: int = 1_000_000,
) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        date=d,
        open=o,
        high=h,
        low=l,
        close=c,
        adj_close=c,
        volume=v,
    )


class MockPortfolio:
    def has_position(self, symbol: str) -> bool:
        return False

    def get_equity(self, prices=None) -> float:
        return 50_000.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_orb_warm_up_period():
    assert ORBStrategy().warm_up_period() >= 5


def test_orb_no_signal_during_warm_up():
    strategy = ORBStrategy()
    signals = strategy.generate_signals(_make_bar(), MockPortfolio())
    assert signals == []


def test_orb_bullish_signal_after_warm_up():
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    # Feed warm-up bars
    for i in range(strategy.warm_up_period()):
        strategy.generate_signals(
            _make_bar(d=date(2024, 1, 2 + i), o=5000 + i, h=5050 + i, l=4960 + i, c=5030 + i),
            portfolio,
        )
    # Strong bullish bar: close >> open
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 2, 1), o=5000, h=5080, l=4990, c=5070),
        portfolio,
    )
    assert any(s.direction == "long" for s in signals)


def test_orb_bearish_signal():
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    # Feed warm-up bars (bearish trend so ATR is populated)
    for i in range(strategy.warm_up_period()):
        strategy.generate_signals(
            _make_bar(d=date(2024, 1, 2 + i), o=5000 - i, h=5050 - i, l=4960 - i, c=4970 - i),
            portfolio,
        )
    # Strong bearish bar: close << open
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 2, 1), o=5000, h=5010, l=4920, c=4930),
        portfolio,
    )
    assert any(s.direction == "short" for s in signals)


def test_orb_no_signal_on_doji():
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        strategy.generate_signals(
            _make_bar(d=date(2024, 1, 2 + i), o=5000, h=5050, l=4950, c=5000 + i),
            portfolio,
        )
    # Doji: close ≈ open — body_ratio will be tiny
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 2, 1), o=5000, h=5020, l=4980, c=5001),
        portfolio,
    )
    assert signals == []


def test_orb_state_manager_integration():
    sm = StateManager(enabled=True)
    strategy = ORBStrategy(state_manager=sm)
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        strategy.generate_signals(
            _make_bar(d=date(2024, 1, 2 + i), o=5000 + i, h=5050 + i, l=4960 + i, c=5030 + i),
            portfolio,
        )
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 2, 1), o=5000, h=5080, l=4990, c=5070),
        portfolio,
    )
    assert len(signals) >= 1
    assert signals[0].strength > 0


def test_orb_signal_has_symbol():
    """Signal symbol should match bar symbol."""
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        strategy.generate_signals(
            _make_bar(symbol="NQ", d=date(2024, 1, 2 + i), o=18000 + i, h=18050 + i, l=17960 + i, c=18030 + i),
            portfolio,
        )
    signals = strategy.generate_signals(
        _make_bar(symbol="NQ", d=date(2024, 2, 1), o=18000, h=18080, l=17990, c=18070),
        portfolio,
    )
    long_signals = [s for s in signals if s.direction == "long"]
    assert len(long_signals) >= 1
    assert long_signals[0].symbol == "NQ"


def test_orb_strength_in_range():
    """Signal strength must be in [0, 1]."""
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        strategy.generate_signals(
            _make_bar(d=date(2024, 1, 2 + i), o=5000 + i, h=5050 + i, l=4960 + i, c=5030 + i),
            portfolio,
        )
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 2, 1), o=5000, h=5080, l=4990, c=5070),
        portfolio,
    )
    for s in signals:
        assert 0.0 <= s.strength <= 1.0, f"strength {s.strength} out of range"


def test_orb_get_parameter_space():
    space = ORBStrategy.get_parameter_space()
    assert "atr_period" in space
    assert "sma_period" in space
    assert "body_threshold" in space


def test_orb_from_params():
    params = {"atr_period": 10, "sma_period": 3, "body_threshold": 0.2}
    strategy = ORBStrategy.from_params(params)
    assert isinstance(strategy, ORBStrategy)
    assert strategy.warm_up_period() >= 10


def test_orb_custom_threshold_filters_weak_bar():
    """High threshold (0.9) should filter out a moderately bullish bar."""
    strategy = ORBStrategy(body_threshold=0.9)
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        strategy.generate_signals(
            _make_bar(d=date(2024, 1, 2 + i), o=5000, h=5050, l=4950, c=5010 + i),
            portfolio,
        )
    # Mild bullish bar — body is small relative to ATR
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 2, 1), o=5000, h=5050, l=4980, c=5005),
        portfolio,
    )
    assert signals == []
