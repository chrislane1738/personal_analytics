"""Tests for VWAPReversionStrategy.

Tests are written before implementation (TDD).
"""
from datetime import date

import pytest

from core.events import BarEvent
from topstep.strategies.vwap_reversion_strategy import VWAPReversionStrategy
from topstep.state_manager import StateManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(
    symbol="ES",
    d=date(2024, 6, 1),
    o=5000.0,
    h=5050.0,
    l=4960.0,
    c=5030.0,
    v=1_000_000,
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
# Warm-up
# ---------------------------------------------------------------------------

def test_vwap_warm_up_period():
    """warm_up_period() must be at least as large as the default sma_period (5)."""
    assert VWAPReversionStrategy().warm_up_period() >= 5


def test_vwap_warm_up_period_reflects_max_of_periods():
    """warm_up_period() equals max(sma_period, atr_period)."""
    s = VWAPReversionStrategy(sma_period=3, atr_period=20)
    assert s.warm_up_period() == 20

    s2 = VWAPReversionStrategy(sma_period=10, atr_period=5)
    assert s2.warm_up_period() == 10


def test_vwap_no_signal_during_warm_up():
    """No signal is returned on the very first bar (warm-up not complete)."""
    strategy = VWAPReversionStrategy()
    assert strategy.generate_signals(_make_bar(), MockPortfolio()) == []


# ---------------------------------------------------------------------------
# Long (oversold) signal
# ---------------------------------------------------------------------------

def _prime_strategy(strategy, n=5, base_close=5000.0, base_date_offset=2):
    """Feed n bars at stable price to build SMA / ATR history."""
    portfolio = MockPortfolio()
    for i in range(n):
        strategy.generate_signals(
            _make_bar(
                d=date(2024, 1, base_date_offset + i),
                o=base_close,
                h=base_close + 20,
                l=base_close - 20,
                c=base_close,
            ),
            portfolio,
        )
    return portfolio


def test_vwap_long_signal_on_oversold():
    """A bar priced well below SMA triggers a long signal."""
    strategy = VWAPReversionStrategy(
        sma_period=5, atr_period=5, deviation_threshold=1.0
    )
    portfolio = _prime_strategy(strategy, n=5)

    # Price drops sharply — SMA ~5000, close=4920 → deviation deeply negative
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 1, 8), o=4940, h=4950, l=4910, c=4920),
        portfolio,
    )
    assert any(s.direction == "long" for s in signals)


def test_vwap_long_signal_strength_in_range():
    """Strength of a long signal must be in (0, 1]."""
    strategy = VWAPReversionStrategy(
        sma_period=5, atr_period=5, deviation_threshold=1.0
    )
    portfolio = _prime_strategy(strategy, n=5)
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 1, 8), o=4940, h=4950, l=4910, c=4920),
        portfolio,
    )
    long_signals = [s for s in signals if s.direction == "long"]
    assert long_signals, "Expected at least one long signal"
    for sig in long_signals:
        assert 0.0 < sig.strength <= 1.0, f"strength out of range: {sig.strength}"


# ---------------------------------------------------------------------------
# Short (overbought) signal
# ---------------------------------------------------------------------------

def test_vwap_short_signal_on_overbought():
    """A bar priced well above SMA triggers a short signal."""
    strategy = VWAPReversionStrategy(
        sma_period=5, atr_period=5, deviation_threshold=1.0
    )
    portfolio = _prime_strategy(strategy, n=5)

    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 1, 8), o=5060, h=5090, l=5050, c=5080),
        portfolio,
    )
    assert any(s.direction == "short" for s in signals)


def test_vwap_short_signal_strength_in_range():
    """Strength of a short signal must be in (0, 1]."""
    strategy = VWAPReversionStrategy(
        sma_period=5, atr_period=5, deviation_threshold=1.0
    )
    portfolio = _prime_strategy(strategy, n=5)
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 1, 8), o=5060, h=5090, l=5050, c=5080),
        portfolio,
    )
    short_signals = [s for s in signals if s.direction == "short"]
    assert short_signals, "Expected at least one short signal"
    for sig in short_signals:
        assert 0.0 < sig.strength <= 1.0, f"strength out of range: {sig.strength}"


# ---------------------------------------------------------------------------
# No signal near fair value
# ---------------------------------------------------------------------------

def test_vwap_no_signal_near_fair_value():
    """A bar priced within threshold of SMA produces no signal."""
    strategy = VWAPReversionStrategy(
        sma_period=5, atr_period=5, deviation_threshold=1.0
    )
    portfolio = _prime_strategy(strategy, n=5)

    # Price barely above SMA — deviation well within threshold
    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 1, 8), o=4998, h=5010, l=4990, c=5002),
        portfolio,
    )
    assert signals == []


# ---------------------------------------------------------------------------
# StateManager integration
# ---------------------------------------------------------------------------

def test_vwap_state_manager_integration():
    """Strategy with StateManager still produces a signal when oversold."""
    sm = StateManager(enabled=True)
    strategy = VWAPReversionStrategy(
        sma_period=5, atr_period=5, state_manager=sm
    )
    portfolio = _prime_strategy(strategy, n=5)

    signals = strategy.generate_signals(
        _make_bar(d=date(2024, 1, 8), o=4940, h=4950, l=4910, c=4920),
        portfolio,
    )
    assert len(signals) >= 1


def test_vwap_state_manager_disabled_same_signal():
    """Disabled StateManager (multiplier=1.0) gives same direction as no manager."""
    sm_off = StateManager(enabled=False)
    strat_with = VWAPReversionStrategy(
        sma_period=5, atr_period=5, state_manager=sm_off
    )
    strat_without = VWAPReversionStrategy(sma_period=5, atr_period=5)

    for strat in (strat_with, strat_without):
        _prime_strategy(strat, n=5)

    bar = _make_bar(d=date(2024, 1, 8), o=4940, h=4950, l=4910, c=4920)
    portfolio = MockPortfolio()
    sigs_with = strat_with.generate_signals(bar, portfolio)
    sigs_without = strat_without.generate_signals(bar, portfolio)

    dirs_with = {s.direction for s in sigs_with}
    dirs_without = {s.direction for s in sigs_without}
    assert dirs_with == dirs_without


# ---------------------------------------------------------------------------
# Parameter space / from_params
# ---------------------------------------------------------------------------

def test_vwap_parameter_space():
    """get_parameter_space() returns all required keys with non-empty lists."""
    space = VWAPReversionStrategy.get_parameter_space()
    for key in ("sma_period", "atr_period", "deviation_threshold"):
        assert key in space, f"Missing key: {key}"
        assert len(space[key]) > 0


def test_vwap_from_params_creates_valid_instance():
    """from_params() returns a functional VWAPReversionStrategy."""
    params = {"sma_period": 3, "atr_period": 10, "deviation_threshold": 0.8}
    strategy = VWAPReversionStrategy.from_params(params)
    assert isinstance(strategy, VWAPReversionStrategy)
    assert strategy.warm_up_period() == 10


def test_vwap_signal_has_symbol():
    """SignalEvent carries the correct symbol from the BarEvent."""
    strategy = VWAPReversionStrategy(
        sma_period=5, atr_period=5, deviation_threshold=1.0
    )
    portfolio = _prime_strategy(strategy, n=5)
    signals = strategy.generate_signals(
        _make_bar(symbol="MES", d=date(2024, 1, 8), o=4940, h=4950, l=4910, c=4920),
        portfolio,
    )
    long_signals = [s for s in signals if s.direction == "long"]
    assert long_signals
    assert long_signals[0].symbol == "MES"
