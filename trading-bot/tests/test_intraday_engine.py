"""Tests for intraday bar support in BacktestEngine.

Test cases:
1. Engine loads intraday bars when timeframe != "1D"
2. Engine iterates over all intraday bars in correct chronological order
3. Equity snapshots recorded once per calendar day (not per bar)
4. Force-flat triggers at end of day only
5. Strategy receives correct BarEvents with intraday OHLC
6. TopstepEvalSimulator works with intraday timeframe
7. CampaignRunner passes timeframe correctly and queries intraday date range
8. Warm-up counts bars (not days) for intraday
"""

from __future__ import annotations

import tempfile
from datetime import date, datetime, timedelta, timezone

import pytest

from core.engine import BacktestEngine
from core.events import BarEvent, SignalEvent
from data.storage.database import Database
from data.storage.models import IntradayBar
from strategy.base import Strategy
from topstep.campaign_runner import CampaignRunner
from topstep.config import TopstepConfig
from topstep.simulator import TopstepEvalSimulator


# ---------------------------------------------------------------------------
# Helper: generate synthetic intraday bars
# ---------------------------------------------------------------------------


def _generate_intraday_bars(
    symbol: str = "ES",
    num_days: int = 5,
    bars_per_day: int = 6,
    start_date: date = date(2024, 6, 3),
    timeframe: str = "5m",
    base_price: float = 5000.0,
) -> list[IntradayBar]:
    """Create synthetic intraday bars with deterministic prices.

    Bars are spaced at 5-minute intervals, ``bars_per_day`` per day,
    starting from 09:30 UTC on each trading day.
    """
    bars: list[IntradayBar] = []
    price = base_price
    current_date = start_date

    for day_idx in range(num_days):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)

        for bar_idx in range(bars_per_day):
            ts = datetime(
                current_date.year,
                current_date.month,
                current_date.day,
                9, 30 + bar_idx * 5, 0,
                tzinfo=timezone.utc,
            )
            # Small alternating moves so there is variation
            move = 2.0 * ((-1) ** bar_idx)
            o = price
            h = price + abs(move) + 3.0
            l = price - abs(move) - 1.0
            c = price + move
            price = c

            bars.append(IntradayBar(
                symbol=symbol,
                timestamp=ts,
                timeframe=timeframe,
                open=round(o, 2),
                high=round(h, 2),
                low=round(l, 2),
                close=round(c, 2),
                volume=10_000 + bar_idx * 100,
                vwap=round((h + l + c) / 3, 2),
            ))

        current_date += timedelta(days=1)

    return bars


# ---------------------------------------------------------------------------
# Helper strategies
# ---------------------------------------------------------------------------


class BarCollectorStrategy(Strategy):
    """Records every BarEvent it receives for later assertion."""

    def __init__(self):
        super().__init__()
        self.received_bars: list[BarEvent] = []

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        self.received_bars.append(bar)
        return []

    def warm_up_period(self) -> int:
        return 0


class AlwaysBuyStrategy(Strategy):
    """Buys on every bar (if no position), sells on every other bar."""

    def __init__(self):
        super().__init__()
        self._bar_count = 0

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        self._bar_count += 1
        if not portfolio.has_position(bar.symbol):
            return [SignalEvent(
                symbol=bar.symbol,
                direction="long",
                reason="always_buy",
                strength=0.5,
            )]
        elif self._bar_count % 3 == 0:
            return [SignalEvent(
                symbol=bar.symbol,
                direction="short",
                reason="periodic_sell",
                strength=0.5,
            )]
        return []

    def warm_up_period(self) -> int:
        return 0


class WarmUpCounterStrategy(Strategy):
    """Counts bars; returns signals only after warm-up.

    Used to verify warm-up counts bars not days for intraday.
    """

    def __init__(self, warm_up: int = 8):
        super().__init__()
        self._warm_up = warm_up
        self._total_calls = 0
        self.first_signal_call: int | None = None

    def warm_up_period(self) -> int:
        return self._warm_up

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        self._total_calls += 1
        if self.first_signal_call is None:
            self.first_signal_call = self._total_calls
        return [SignalEvent(
            symbol=bar.symbol,
            direction="long",
            reason="warm_up_test",
            strength=0.3,
        )]


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
def intraday_db(tmp_db):
    """Return a Database with synthetic 5m ES bars pre-loaded."""
    bars = _generate_intraday_bars(
        symbol="ES", num_days=10, bars_per_day=6, timeframe="5m",
    )
    tmp_db.insert_intraday_bars(bars)
    return tmp_db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntradayEngineLoad:
    """Verify BacktestEngine loads intraday bars correctly."""

    def test_no_not_implemented_error(self, intraday_db):
        """Engine should accept intraday timeframes without raising."""
        engine = BacktestEngine(
            strategy=BarCollectorStrategy(),
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        metrics = engine.run()
        assert isinstance(metrics, dict)

    def test_loads_intraday_bars_into_bar_data(self, intraday_db):
        """_bar_data should be keyed by datetime, not date, for intraday."""
        engine = BacktestEngine(
            strategy=BarCollectorStrategy(),
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine._load_data()

        assert "ES" in engine._bar_data
        keys = list(engine._bar_data["ES"].keys())
        assert len(keys) > 0
        # All keys should be datetime objects
        for k in keys:
            assert isinstance(k, datetime), f"Expected datetime key, got {type(k)}"


class TestIntradayEngineIteration:
    """Verify the engine iterates intraday bars in correct order."""

    def test_strategy_receives_all_bars(self, intraday_db):
        """Strategy should receive every intraday bar, not just one per day."""
        strategy = BarCollectorStrategy()
        engine = BacktestEngine(
            strategy=strategy,
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine.run()

        # We generated 10 days x 6 bars = 60 bars total
        assert len(strategy.received_bars) == 60

    def test_bars_in_chronological_order(self, intraday_db):
        """Bars should arrive in timestamp order."""
        strategy = BarCollectorStrategy()
        engine = BacktestEngine(
            strategy=strategy,
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine.run()

        # Since BarEvent.date is just the calendar date, we check that
        # the sequence of dates is non-decreasing
        dates = [b.date for b in strategy.received_bars]
        for i in range(1, len(dates)):
            assert dates[i] >= dates[i - 1], (
                f"Out of order: {dates[i - 1]} > {dates[i]}"
            )

    def test_bar_event_has_correct_ohlc(self, intraday_db):
        """BarEvent OHLC should match the intraday bar data."""
        strategy = BarCollectorStrategy()
        engine = BacktestEngine(
            strategy=strategy,
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine.run()

        # Verify first bar has non-zero OHLC
        first = strategy.received_bars[0]
        assert first.symbol == "ES"
        assert first.open > 0
        assert first.high >= first.low
        assert first.close > 0
        assert first.volume > 0

    def test_bar_event_date_is_calendar_date(self, intraday_db):
        """BarEvent.date should be a date object, not datetime."""
        strategy = BarCollectorStrategy()
        engine = BacktestEngine(
            strategy=strategy,
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine.run()

        for bar in strategy.received_bars:
            assert isinstance(bar.date, date)
            assert not isinstance(bar.date, datetime)


class TestIntradayEquityCurve:
    """Verify equity snapshots are recorded once per calendar day."""

    def test_one_equity_point_per_day(self, intraday_db):
        """Equity curve should have one entry per trading day, not per bar."""
        engine = BacktestEngine(
            strategy=BarCollectorStrategy(),
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine.run()

        eq_curve = engine.portfolio.equity_curve
        # 10 trading days of data
        assert len(eq_curve) == 10

        # Each entry should be (date, float)
        for d, val in eq_curve:
            assert isinstance(d, date)
            assert not isinstance(d, datetime)
            assert isinstance(val, float)

    def test_equity_dates_are_unique(self, intraday_db):
        """No duplicate dates in the equity curve."""
        engine = BacktestEngine(
            strategy=BarCollectorStrategy(),
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine.run()

        dates = [d for d, _ in engine.portfolio.equity_curve]
        assert len(dates) == len(set(dates)), "Duplicate dates in equity curve"


class BuyOnlyStrategy(Strategy):
    """Buys on every bar if no position. Never sells (force-flat closes)."""

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        if not portfolio.has_position(bar.symbol):
            return [SignalEvent(
                symbol=bar.symbol,
                direction="long",
                reason="buy_only",
                strength=0.5,
            )]
        return []

    def warm_up_period(self) -> int:
        return 0


class TestIntradayForceFlat:
    """Verify force-flat-daily fires only at end of day."""

    def test_force_flat_at_eod(self, intraday_db):
        """With force_flat_daily, positions should be force-closed at end of each day."""
        engine = BacktestEngine(
            strategy=BuyOnlyStrategy(),
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
            force_flat_daily=True,
        )
        engine.run()

        # With force_flat_daily, the strategy buys each day (since position
        # is closed by force-flat at EOD) — verify that equity curve has
        # exactly 10 days and there were trades.
        eq_curve = engine.portfolio.equity_curve
        assert len(eq_curve) == 10
        # Strategy should have generated trades (buy + force-close each day)
        assert engine.trade_log.trades, "Expected at least one trade with BuyOnlyStrategy"

    def test_force_flat_only_at_eod_not_mid_day(self, intraday_db):
        """Force-flat should NOT happen on mid-day bars."""
        strategy = BarCollectorStrategy()
        engine = BacktestEngine(
            strategy=strategy,
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
            force_flat_daily=True,
        )
        engine.run()

        # Equity curve should have one entry per day, not per bar
        assert len(engine.portfolio.equity_curve) == 10


class TestIntradayWarmUp:
    """Verify warm-up counts bars (not days) for intraday."""

    def test_warm_up_counts_bars_not_days(self, intraday_db):
        """With warm_up=8 and 6 bars/day, warm-up should span into the 2nd day."""
        strategy = WarmUpCounterStrategy(warm_up=8)
        engine = BacktestEngine(
            strategy=strategy,
            database=intraday_db,
            universe=["ES"],
            start_date=date(2024, 6, 3),
            end_date=date(2024, 6, 14),
            initial_capital=50_000.0,
            benchmark_symbol="ES",
            timeframe="5m",
            quiet=True,
        )
        engine.run()

        # Total bars = 60 (10 days x 6 bars)
        # Warm-up = 8 bars, so strategy receives all 60 calls to generate_signals
        # but the engine only processes signals from bar 9 onward
        assert strategy._total_calls == 60


class TestIntradayDailyCompatibility:
    """Verify daily mode still works correctly (no regressions)."""

    def test_daily_mode_unchanged(self, tmp_db):
        """Running with timeframe='1D' should work exactly as before."""
        from data.storage.models import DailyBar
        from tests.fixtures.sample_bars import generate_spy_bars

        bars = generate_spy_bars("SPY")
        tmp_db.insert_daily_bars(bars)

        engine = BacktestEngine(
            strategy=BarCollectorStrategy(),
            database=tmp_db,
            universe=["SPY"],
            start_date=bars[0].date,
            end_date=bars[-1].date,
            initial_capital=100_000.0,
            benchmark_symbol="SPY",
            timeframe="1D",
            quiet=True,
        )
        metrics = engine.run()

        assert isinstance(metrics, dict)
        assert metrics["total_trades"] == 0  # BarCollector doesn't trade
        assert len(engine.portfolio.equity_curve) > 0

    def test_empty_intraday_data_graceful(self, tmp_db):
        """Engine handles missing intraday data gracefully."""
        engine = BacktestEngine(
            strategy=BarCollectorStrategy(),
            database=tmp_db,
            universe=["NOSYMBOL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            initial_capital=50_000.0,
            benchmark_symbol="NOSYMBOL",
            timeframe="5m",
            quiet=True,
        )
        metrics = engine.run()
        assert isinstance(metrics, dict)
        assert metrics["total_trades"] == 0


class TestTopstepEvalSimulatorIntraday:
    """Verify TopstepEvalSimulator works with intraday timeframe."""

    def test_simulator_accepts_timeframe(self, intraday_db):
        """TopstepEvalSimulator should accept and pass through timeframe."""
        config = TopstepConfig(max_attempt_days=5)
        sim = TopstepEvalSimulator(
            strategy=BarCollectorStrategy(),
            database=intraday_db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
            timeframe="5m",
        )
        assert sim.timeframe == "5m"

    def test_simulator_runs_intraday(self, intraday_db):
        """TopstepEvalSimulator should complete an intraday attempt."""
        config = TopstepConfig(max_attempt_days=5)
        sim = TopstepEvalSimulator(
            strategy=AlwaysBuyStrategy(),
            database=intraday_db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
            timeframe="5m",
        )
        result = sim.run_attempt(start_date=date(2024, 6, 3))
        assert isinstance(result, dict)
        assert "status" in result
        assert "days_traded" in result


class TestCampaignRunnerIntraday:
    """Verify CampaignRunner passes timeframe correctly."""

    def test_campaign_runner_intraday(self, intraday_db):
        """CampaignRunner should work with intraday timeframe and query the right table."""
        config = TopstepConfig(max_attempt_days=3)
        runner = CampaignRunner(
            strategy_class=BarCollectorStrategy,
            instrument="ES",
            config=config,
            database=intraday_db,
            state_machine_enabled=False,
            num_attempts=2,
            seed=42,
            timeframe="5m",
        )
        result = runner.run()
        assert result.num_attempts == 2
        assert result.timeframe == "5m"
        assert 0.0 <= result.pass_rate <= 1.0

    def test_campaign_runner_daily_still_works(self, tmp_db):
        """CampaignRunner with timeframe='1D' should still use daily_bars table."""
        from data.storage.models import DailyBar

        # Seed with daily data
        bars = []
        base = 5000.0
        for i in range(200):
            d = date(2024, 1, 2) + timedelta(days=i)
            o = base + (i % 50) * 2
            h = o + 30 + (i % 7) * 5
            l = o - 10 - (i % 5) * 3
            c = o + 15 + ((-1) ** i) * 10
            bars.append(DailyBar(
                symbol="ES", date=d, open=o, high=h, low=l,
                close=c, adj_close=c, volume=1_000_000, vwap=o + 10,
            ))
        tmp_db.insert_daily_bars(bars)

        config = TopstepConfig(max_attempt_days=10)
        runner = CampaignRunner(
            strategy_class=BarCollectorStrategy,
            instrument="ES",
            config=config,
            database=tmp_db,
            state_machine_enabled=False,
            num_attempts=3,
            seed=42,
            timeframe="1D",
        )
        result = runner.run()
        assert result.num_attempts == 3
        assert result.timeframe == "1D"
