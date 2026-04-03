"""Tests for the DailyLossLimitRule and engine day_pnl tracking.

Covers:
  - DailyLossLimitRule blocks when day_pnl exceeds limit
  - DailyLossLimitRule passes when under limit
  - Context without day_pnl defaults to 0 (passes)
  - Engine tracks day_pnl correctly across bars
  - Daily P&L resets at new calendar day
"""

from __future__ import annotations

import tempfile
from datetime import date

import pytest

from core.engine import BacktestEngine
from core.event_bus import EventBus
from core.events import BarEvent, SignalEvent
from data.storage.database import Database
from data.storage.models import DailyBar
from portfolio.portfolio import Portfolio
from risk.rules import DailyLossLimitRule
from strategy.base import Strategy


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def make_signal(symbol: str = "AAPL", direction: str = "long") -> SignalEvent:
    return SignalEvent(symbol=symbol, direction=direction)


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def portfolio(bus: EventBus) -> Portfolio:
    return Portfolio(initial_capital=100_000.0, event_bus=bus)


# ---------------------------------------------------------------------------
# 1. DailyLossLimitRule — unit tests
# ---------------------------------------------------------------------------


class TestDailyLossLimitRule:
    """Unit tests for DailyLossLimitRule in isolation."""

    def test_blocks_when_day_pnl_exceeds_limit(self, portfolio: Portfolio):
        """Day P&L of -$2000 with a $2000 limit should block."""
        rule = DailyLossLimitRule(daily_limit=2000)
        ctx = {"day_pnl": -2000.0}
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert not passes
        assert "Daily loss limit" in reason

    def test_blocks_when_day_pnl_exceeds_limit_further(self, portfolio: Portfolio):
        """Day P&L of -$3000 with a $2000 limit should also block."""
        rule = DailyLossLimitRule(daily_limit=2000)
        ctx = {"day_pnl": -3000.0}
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert not passes

    def test_passes_when_under_limit(self, portfolio: Portfolio):
        """Day P&L of -$500 with a $2000 limit should pass."""
        rule = DailyLossLimitRule(daily_limit=2000)
        ctx = {"day_pnl": -500.0}
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert passes
        assert reason == ""

    def test_passes_when_positive_pnl(self, portfolio: Portfolio):
        """Positive day P&L should always pass."""
        rule = DailyLossLimitRule(daily_limit=2000)
        ctx = {"day_pnl": 1500.0}
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert passes
        assert reason == ""

    def test_passes_when_zero_pnl(self, portfolio: Portfolio):
        """Zero day P&L should pass (not at or beyond negative limit)."""
        rule = DailyLossLimitRule(daily_limit=2000)
        ctx = {"day_pnl": 0.0}
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert passes

    def test_context_without_day_pnl_defaults_to_zero(self, portfolio: Portfolio):
        """When day_pnl is missing from context, default to 0 (passes)."""
        rule = DailyLossLimitRule(daily_limit=2000)
        passes, reason = rule.check(make_signal(), portfolio, {})
        assert passes
        assert reason == ""

    def test_exactly_at_limit_blocks(self, portfolio: Portfolio):
        """Day P&L exactly at the negative limit should block (<= check)."""
        rule = DailyLossLimitRule(daily_limit=1000)
        ctx = {"day_pnl": -1000.0}
        passes, _ = rule.check(make_signal(), portfolio, ctx)
        assert not passes

    def test_just_above_limit_passes(self, portfolio: Portfolio):
        """Day P&L just above the negative limit should pass."""
        rule = DailyLossLimitRule(daily_limit=1000)
        ctx = {"day_pnl": -999.99}
        passes, _ = rule.check(make_signal(), portfolio, ctx)
        assert passes

    def test_reason_contains_dollar_amounts(self, portfolio: Portfolio):
        """Reason string should mention the limit and current P&L."""
        rule = DailyLossLimitRule(daily_limit=2000)
        ctx = {"day_pnl": -2500.0}
        _, reason = rule.check(make_signal(), portfolio, ctx)
        assert "$2,000" in reason
        assert "$-2,500" in reason or "-$2,500" in reason


# ---------------------------------------------------------------------------
# 2. Engine integration — day_pnl tracking
# ---------------------------------------------------------------------------


class _AlwaysBuyStrategy(Strategy):
    """Emits a long signal for the first symbol on every bar."""

    def __init__(self):
        super().__init__()

    def warm_up_period(self) -> int:
        return 0

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        if not portfolio.has_position(bar.symbol):
            return [
                SignalEvent(
                    symbol=bar.symbol,
                    direction="long",
                    reason="always buy",
                    strength=0.5,
                )
            ]
        return []


def _make_daily_bars(symbol: str, prices: list[tuple[date, float]]) -> list[DailyBar]:
    """Create DailyBar objects from (date, close) pairs."""
    bars = []
    for d, close in prices:
        bars.append(DailyBar(
            symbol=symbol,
            date=d,
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            adj_close=close,
            volume=1_000_000,
            vwap=close,
        ))
    return bars


@pytest.fixture
def tmp_db():
    """Return a Database backed by a temporary file with schema created."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    db.create_tables()
    yield db
    db.close()


class TestEngineDayPnlTracking:
    """Integration tests for the engine's daily P&L tracking."""

    def test_daily_loss_limit_zero_does_not_inject_rule(self, tmp_db):
        """Default daily_loss_limit=0 should NOT add DailyLossLimitRule."""
        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            database=tmp_db,
            universe=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            initial_capital=100_000.0,
            daily_loss_limit=0,
            quiet=True,
        )
        rule_names = [type(r).__name__ for r in engine.risk_manager.rules]
        assert "DailyLossLimitRule" not in rule_names

    def test_daily_loss_limit_positive_injects_rule(self, tmp_db):
        """A positive daily_loss_limit should auto-inject DailyLossLimitRule."""
        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            database=tmp_db,
            universe=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            initial_capital=100_000.0,
            daily_loss_limit=2000,
            quiet=True,
        )
        rule_names = [type(r).__name__ for r in engine.risk_manager.rules]
        assert "DailyLossLimitRule" in rule_names

    def test_day_pnl_resets_on_new_day(self, tmp_db):
        """day_start_equity should reset when the calendar day changes."""
        # Two days of data: day 1 flat, day 2 flat
        bars = _make_daily_bars("SPY", [
            (date(2024, 1, 2), 500.0),
            (date(2024, 1, 3), 500.0),
        ])
        tmp_db.insert_daily_bars(bars)

        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            database=tmp_db,
            universe=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            initial_capital=100_000.0,
            daily_loss_limit=500,
            quiet=True,
        )
        engine.run()

        # After two days, _prev_cal_date should be the last day
        assert engine._prev_cal_date == date(2024, 1, 3)
        # _day_start_equity should have been set (not None)
        assert engine._day_start_equity is not None

    def test_engine_runs_normally_without_daily_loss_limit(self, tmp_db):
        """Existing behaviour is preserved when daily_loss_limit=0."""
        from tests.fixtures.sample_bars import generate_spy_bars

        bars = generate_spy_bars("SPY")
        tmp_db.insert_daily_bars(bars)

        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            database=tmp_db,
            universe=["SPY"],
            start_date=bars[0].date,
            end_date=bars[-1].date,
            initial_capital=100_000.0,
            daily_loss_limit=0,
            quiet=True,
        )
        metrics = engine.run()
        assert isinstance(metrics, dict)
        assert "total_return" in metrics
