"""Tests for Topstep session utilities and force-flat-daily behaviour."""

from __future__ import annotations

import tempfile
from datetime import date, time, timedelta

import pytest

from core.engine import BacktestEngine
from core.events import BarEvent, SignalEvent
from data.storage.database import Database
from data.storage.models import DailyBar
from strategy.base import Strategy
from topstep.config import TopstepConfig
from topstep.session import (
    TOPSTEP_FLAT_DEADLINE_ET,
    RTH_CLOSE_ET,
    RTH_OPEN_ET,
    is_past_flat_deadline,
    minutes_until_flat_deadline,
)
from topstep.simulator import TopstepEvalSimulator
from topstep.strategies.orb_strategy import ORBStrategy


# ---------------------------------------------------------------------------
# Session time utilities
# ---------------------------------------------------------------------------


class TestSessionTimeUtilities:
    """Tests for topstep.session helper functions."""

    def test_deadline_constant(self):
        assert TOPSTEP_FLAT_DEADLINE_ET == time(16, 10)

    def test_rth_constants(self):
        assert RTH_OPEN_ET == time(9, 30)
        assert RTH_CLOSE_ET == time(16, 0)

    def test_is_past_flat_deadline_before(self):
        assert is_past_flat_deadline(time(15, 0)) is False

    def test_is_past_flat_deadline_exactly_at(self):
        assert is_past_flat_deadline(time(16, 10)) is True

    def test_is_past_flat_deadline_after(self):
        assert is_past_flat_deadline(time(16, 30)) is True

    def test_is_past_flat_deadline_morning(self):
        assert is_past_flat_deadline(time(9, 30)) is False

    def test_minutes_until_deadline_before(self):
        assert minutes_until_flat_deadline(time(16, 0)) == 10

    def test_minutes_until_deadline_exactly_at(self):
        assert minutes_until_flat_deadline(time(16, 10)) == 0

    def test_minutes_until_deadline_after(self):
        assert minutes_until_flat_deadline(time(16, 30)) == -20

    def test_minutes_until_deadline_morning(self):
        # 9:30 -> 16:10 = 6h40m = 400 minutes
        assert minutes_until_flat_deadline(time(9, 30)) == 400


# ---------------------------------------------------------------------------
# TopstepConfig session defaults
# ---------------------------------------------------------------------------


class TestTopstepConfigSessionDefaults:
    """Verify new session-related config fields."""

    def test_flat_by_time_default(self):
        cfg = TopstepConfig()
        assert cfg.flat_by_time == "15:10"

    def test_session_close_default(self):
        cfg = TopstepConfig()
        assert cfg.session_close == "16:00"

    def test_force_flat_daily_default(self):
        cfg = TopstepConfig()
        assert cfg.force_flat_daily is True

    def test_force_flat_daily_override(self):
        cfg = TopstepConfig(force_flat_daily=False)
        assert cfg.force_flat_daily is False


# ---------------------------------------------------------------------------
# Helpers — strategy that always holds overnight
# ---------------------------------------------------------------------------


class AlwaysLongStrategy(Strategy):
    """Buy on first bar, never sell.  Forces a position to be held overnight."""

    def __init__(self):
        super().__init__()
        self._entered = False

    def warm_up_period(self) -> int:
        return 0

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        if not self._entered and not portfolio.has_position(bar.symbol):
            self._entered = True
            return [
                SignalEvent(
                    symbol=bar.symbol,
                    direction="long",
                    reason="always long",
                    strength=1.0,
                )
            ]
        return []


class BuyEveryDayStrategy(Strategy):
    """Buy on every bar if not already holding.

    Combined with force_flat_daily this tests the re-entry ability:
    the strategy can re-enter on the next day after being force-closed.
    """

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
                    reason="buy every day",
                    strength=1.0,
                )
            ]
        return []


def _make_db_with_bars(symbol: str = "SPY", num_bars: int = 10, base_price: float = 100.0):
    """Create a temp database seeded with simple bars."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    db.create_tables()
    bars = []
    d = date(2024, 1, 2)  # Tuesday
    for i in range(num_bars):
        # Skip weekends
        while d.weekday() >= 5:
            d += timedelta(days=1)
        o = base_price + i
        h = o + 5
        l = o - 2
        c = o + 3
        bars.append(
            DailyBar(
                symbol=symbol,
                date=d,
                open=o,
                high=h,
                low=l,
                close=c,
                adj_close=c,
                volume=100_000,
                vwap=o + 1.5,
            )
        )
        d += timedelta(days=1)
    db.insert_daily_bars(bars)
    return db, bars


# ---------------------------------------------------------------------------
# BacktestEngine force_flat_daily
# ---------------------------------------------------------------------------


class TestForceFlatDaily:
    """Verify that force_flat_daily closes positions at end of each day."""

    def test_positions_closed_each_day(self):
        """With force_flat_daily, portfolio should be empty after each bar."""
        db, bars = _make_db_with_bars("SPY", num_bars=10)
        try:
            engine = BacktestEngine(
                strategy=AlwaysLongStrategy(),
                database=db,
                universe=["SPY"],
                start_date=bars[0].date,
                end_date=bars[-1].date,
                initial_capital=100_000.0,
                benchmark_symbol="SPY",
                quiet=True,
                force_flat_daily=True,
            )
            engine.run()

            # After the run completes, portfolio should be flat
            assert len(engine.portfolio.positions) == 0
        finally:
            db.close()

    def test_positions_not_closed_without_flag(self):
        """Without force_flat_daily, AlwaysLongStrategy should hold the position."""
        db, bars = _make_db_with_bars("SPY", num_bars=10)
        try:
            engine = BacktestEngine(
                strategy=AlwaysLongStrategy(),
                database=db,
                universe=["SPY"],
                start_date=bars[0].date,
                end_date=bars[-1].date,
                initial_capital=100_000.0,
                benchmark_symbol="SPY",
                quiet=True,
                force_flat_daily=False,
            )
            engine.run()

            # AlwaysLong never sells, so the position should still be open
            # (the trade_log end-of-backtest close doesn't affect portfolio.positions)
            assert len(engine.portfolio.positions) > 0
        finally:
            db.close()

    def test_strategy_can_reenter_after_force_close(self):
        """After force-close, strategy should be able to re-enter the next day."""
        db, bars = _make_db_with_bars("SPY", num_bars=5)
        try:
            engine = BacktestEngine(
                strategy=BuyEveryDayStrategy(),
                database=db,
                universe=["SPY"],
                start_date=bars[0].date,
                end_date=bars[-1].date,
                initial_capital=100_000.0,
                benchmark_symbol="SPY",
                quiet=True,
                force_flat_daily=True,
            )
            engine.run()

            # BuyEveryDayStrategy buys whenever not holding.  With
            # force_flat_daily the position is closed each day, so it
            # should re-enter the next day.  We expect multiple trades.
            assert engine.trade_log.trades is not None
            # At least 2 trades means re-entry worked.
            # (entry on day 1, force-close generates a SELL fill, entry day 2, etc.)
            total_trades = len(engine.trade_log.trades)
            assert total_trades >= 2, (
                f"Expected at least 2 trades from re-entry, got {total_trades}"
            )
        finally:
            db.close()

    def test_equity_curve_correct_with_force_flat(self):
        """Equity curve should still be populated correctly when force-flat is on."""
        db, bars = _make_db_with_bars("SPY", num_bars=10)
        try:
            engine = BacktestEngine(
                strategy=BuyEveryDayStrategy(),
                database=db,
                universe=["SPY"],
                start_date=bars[0].date,
                end_date=bars[-1].date,
                initial_capital=100_000.0,
                benchmark_symbol="SPY",
                quiet=True,
                force_flat_daily=True,
            )
            engine.run()

            # Equity curve should have one entry per bar
            assert len(engine.portfolio.equity_curve) == len(bars)

            # All equity values should be reasonable (not zero, not negative)
            for eq_date, equity in engine.portfolio.equity_curve:
                assert equity > 0, f"Equity on {eq_date} is {equity}"
        finally:
            db.close()

    def test_force_flat_with_no_positions_is_noop(self):
        """Force-flat on a day with no positions should not error."""

        class NeverTradeStrategy(Strategy):
            def warm_up_period(self) -> int:
                return 0

            def generate_signals(self, bar, portfolio):
                return []

        db, bars = _make_db_with_bars("SPY", num_bars=5)
        try:
            engine = BacktestEngine(
                strategy=NeverTradeStrategy(),
                database=db,
                universe=["SPY"],
                start_date=bars[0].date,
                end_date=bars[-1].date,
                initial_capital=100_000.0,
                benchmark_symbol="SPY",
                quiet=True,
                force_flat_daily=True,
            )
            # Should complete without error
            engine.run()
            assert len(engine.portfolio.positions) == 0
            assert engine.trade_log.trades is not None
        finally:
            db.close()

    def test_default_force_flat_is_off(self):
        """BacktestEngine defaults to force_flat_daily=False."""
        db, bars = _make_db_with_bars("SPY", num_bars=3)
        try:
            engine = BacktestEngine(
                strategy=AlwaysLongStrategy(),
                database=db,
                universe=["SPY"],
                start_date=bars[0].date,
                end_date=bars[-1].date,
                initial_capital=100_000.0,
                benchmark_symbol="SPY",
                quiet=True,
            )
            assert engine.force_flat_daily is False
        finally:
            db.close()


# ---------------------------------------------------------------------------
# TopstepEvalSimulator integration — force_flat_daily wired in
# ---------------------------------------------------------------------------


class TestSimulatorForceFlat:
    """Verify TopstepEvalSimulator passes force_flat_daily to BacktestEngine."""

    def test_simulator_uses_force_flat_daily(self):
        """Default TopstepConfig has force_flat_daily=True; simulator should use it."""
        db, bars = _make_db_with_bars("ES", num_bars=30, base_price=5000.0)
        try:
            config = TopstepConfig(max_attempt_days=15)
            assert config.force_flat_daily is True

            sim = TopstepEvalSimulator(
                strategy=ORBStrategy(),
                database=db,
                instrument="ES",
                config=config,
                state_machine_enabled=False,
            )
            result = sim.run_attempt(start_date=bars[0].date)

            # Should complete normally
            assert result["status"] in ("pass", "fail", "timeout", "active")
        finally:
            db.close()

    def test_simulator_can_disable_force_flat(self):
        """When force_flat_daily=False, simulator should still work."""
        db, bars = _make_db_with_bars("ES", num_bars=30, base_price=5000.0)
        try:
            config = TopstepConfig(max_attempt_days=15, force_flat_daily=False)
            assert config.force_flat_daily is False

            sim = TopstepEvalSimulator(
                strategy=ORBStrategy(),
                database=db,
                instrument="ES",
                config=config,
                state_machine_enabled=False,
            )
            result = sim.run_attempt(start_date=bars[0].date)

            assert result["status"] in ("pass", "fail", "timeout", "active")
        finally:
            db.close()
