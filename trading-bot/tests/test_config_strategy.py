"""Tests for ConfigStrategy — config-driven strategy loaded from YAML."""

import os
from datetime import date
from pathlib import Path

import pytest

from core.events import BarEvent, SignalEvent
from strategy.config_strategy import ConfigStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

YAML_DIR = Path(__file__).parent.parent / "config" / "strategies"
MEAN_REVERSION_YAML = str(YAML_DIR / "mean_reversion.yaml")


class MockPosition:
    """Minimal position duck type for portfolio context."""
    def __init__(self, avg_cost: float, quantity: int):
        self.avg_cost = avg_cost
        self.quantity = quantity


class MockPortfolio:
    """Minimal portfolio duck type matching portfolio.portfolio.Portfolio interface."""
    def __init__(self, positions: dict | None = None):
        self.positions = positions or {}

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions


def make_bar(close: float, **kwargs) -> BarEvent:
    """Helper to create a BarEvent with sane defaults."""
    defaults = dict(
        symbol="SPY",
        date=date(2024, 6, 1),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        adj_close=close,
        volume=1_000_000,
        vwap=close,
    )
    defaults.update(kwargs)
    return BarEvent(**defaults)


# ---------------------------------------------------------------------------
# Loading & initialisation
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_yaml(self):
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        assert strat.name == "Mean Reversion SPY"
        assert strat.universe == ["SPY"]

    def test_indicators_created(self):
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        assert "sma_20" in strat.indicators
        assert "sma_50" in strat.indicators
        assert "rsi_14" in strat.indicators
        assert "bb_20" in strat.indicators

    def test_warm_up_period(self):
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        # sma_20=20, sma_50=50, rsi_14=15, bb_20=20 → max = 50
        assert strat.warm_up_period() == 50

    def test_entry_rules_parsed(self):
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        assert len(strat.entry_rules) == 1
        assert strat.entry_rules[0]["direction"] == "long"

    def test_exit_rules_parsed(self):
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        assert len(strat.exit_rules) == 2


# ---------------------------------------------------------------------------
# Warm-up behaviour
# ---------------------------------------------------------------------------

class TestWarmUp:
    def test_no_signals_during_warmup(self):
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        portfolio = MockPortfolio()

        # Feed 25 bars (less than warm-up of 50) — should produce no signals
        for i in range(25):
            bar = make_bar(close=100.0 + i * 0.1)
            signals = strat.generate_signals(bar, portfolio)
            assert signals == [], f"Expected no signals on bar {i}, got {signals}"


# ---------------------------------------------------------------------------
# Entry signal generation
# ---------------------------------------------------------------------------

class TestEntrySignals:
    def test_entry_signal_triggered(self):
        """After warm-up, feed a bar with close < BB lower and RSI < 30
        to trigger a long entry signal."""
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        portfolio = MockPortfolio()

        # Feed enough bars for warm-up (50 bars at a stable price).
        # Then a small decline to push RSI down but keep BB bands tight,
        # followed by a sudden sharp drop below the lower band.
        for i in range(50):
            bar = make_bar(close=100.0)
            strat.generate_signals(bar, portfolio)

        # Small gradual decline (11 bars: from 98 down to 78).
        # This pushes RSI below 30 while BB window still includes
        # mostly 100-level prices, so lower band stays relatively high.
        all_signals = []
        for i in range(11):
            p = 100.0 - (i + 1) * 2.0
            bar = make_bar(close=p)
            sigs = strat.generate_signals(bar, portfolio)
            all_signals.extend(sigs)

        # After the first decline bar (close=98), RSI is 0 and close < bb_lower
        # because bb_lower ~ 99 when the window is mostly 100s and one 98.
        entry_signals = [s for s in all_signals if s.direction == "long"]
        assert len(entry_signals) > 0, "Expected at least one long entry signal"
        assert "BB" in entry_signals[0].reason or "RSI" in entry_signals[0].reason

    def test_no_entry_when_position_exists(self):
        """If portfolio already has a position, entry rules should not fire."""
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        portfolio = MockPortfolio(positions={
            "SPY": MockPosition(avg_cost=80.0, quantity=100),
        })

        # Warm up
        for i in range(50):
            bar = make_bar(close=100.0)
            strat.generate_signals(bar, portfolio)

        # Decline to trigger entry conditions
        for i in range(20):
            bar = make_bar(close=100.0 - (i + 1) * 2.0)
            strat.generate_signals(bar, portfolio)

        # These bars would normally trigger entry but shouldn't because position exists
        all_signals = []
        for i in range(5):
            bar = make_bar(close=55.0 - i * 1.0)
            sigs = strat.generate_signals(bar, portfolio)
            all_signals.extend(sigs)

        entry_signals = [s for s in all_signals if s.direction == "long"]
        assert len(entry_signals) == 0, "Should not generate entry signals with existing position"


# ---------------------------------------------------------------------------
# Exit signal generation
# ---------------------------------------------------------------------------

class TestExitSignals:
    def test_exit_signal_mean_reversion(self):
        """With a position, close > sma_20 should trigger an exit."""
        strat = ConfigStrategy(MEAN_REVERSION_YAML)

        # Warm up with declining prices ending low, so SMA_20 is low
        prices = [100.0] * 30 + [80.0] * 20
        no_pos_portfolio = MockPortfolio()
        for p in prices:
            bar = make_bar(close=p)
            strat.generate_signals(bar, no_pos_portfolio)

        # Now simulate having a position
        portfolio = MockPortfolio(positions={
            "SPY": MockPosition(avg_cost=75.0, quantity=100),
        })

        # The SMA_20 is around 80.0 (last 20 bars were 80.0).
        # Feed a bar with close clearly above SMA_20 to trigger exit
        all_signals = []
        for _ in range(3):
            bar = make_bar(close=90.0)
            sigs = strat.generate_signals(bar, portfolio)
            all_signals.extend(sigs)

        exit_signals = [s for s in all_signals if s.direction == "short"]
        assert len(exit_signals) > 0, "Expected exit signal when close > sma_20"
        assert any("mean" in s.reason.lower() or "revert" in s.reason.lower()
                    for s in exit_signals)

    def test_exit_signal_stop_loss(self):
        """With a position and pnl_pct < -0.05, stop loss should trigger."""
        strat = ConfigStrategy(MEAN_REVERSION_YAML)

        # Warm up
        no_pos = MockPortfolio()
        for i in range(55):
            bar = make_bar(close=100.0)
            strat.generate_signals(bar, no_pos)

        # Position with avg_cost = 100, and close = 90 => pnl_pct = -0.10
        portfolio = MockPortfolio(positions={
            "SPY": MockPosition(avg_cost=100.0, quantity=100),
        })

        bar = make_bar(close=90.0)
        signals = strat.generate_signals(bar, portfolio)

        stop_signals = [s for s in signals if "stop" in s.reason.lower() or "loss" in s.reason.lower()]
        assert len(stop_signals) > 0, "Expected stop loss exit signal"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_portfolio_no_crash(self):
        """Passing None for portfolio should not crash."""
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        bar = make_bar(close=100.0)
        signals = strat.generate_signals(bar, None)
        assert isinstance(signals, list)

    def test_signal_has_correct_symbol(self):
        strat = ConfigStrategy(MEAN_REVERSION_YAML)
        portfolio = MockPortfolio()

        # Warm up
        for i in range(50):
            bar = make_bar(close=100.0)
            strat.generate_signals(bar, portfolio)

        # Decline
        for i in range(20):
            bar = make_bar(close=100.0 - (i + 1) * 2.0)
            strat.generate_signals(bar, portfolio)

        all_signals = []
        for i in range(5):
            bar = make_bar(close=55.0 - i)
            sigs = strat.generate_signals(bar, portfolio)
            all_signals.extend(sigs)

        for sig in all_signals:
            assert sig.symbol == "SPY"
