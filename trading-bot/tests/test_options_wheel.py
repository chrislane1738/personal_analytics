"""Tests for OptionsWheelStrategy."""
from __future__ import annotations

import pytest
from datetime import date

from strategy.library.options_wheel import OptionsWheelStrategy
from core.events import BarEvent, SignalEvent
from core.engine import BacktestEngine
from data.storage.database import Database
from tests.fixtures.sample_bars import generate_sample_bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(symbol: str, day_offset: int, price: float) -> BarEvent:
    """Build a minimal BarEvent for testing."""
    return BarEvent(
        symbol=symbol,
        date=date(2024, 1, 2 + day_offset),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        adj_close=price,
        volume=1_000_000,
        vwap=price,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptionsWheelStrategy:

    def test_no_signals_during_warmup(self):
        """RSI needs warm-up period before generating signals."""
        strategy = OptionsWheelStrategy(rsi_period=14)
        bar = BarEvent(
            symbol="AAPL",
            date=date(2024, 1, 2),
            open=185,
            high=187,
            low=184,
            close=186,
            adj_close=185.5,
            volume=16_000_000,
            vwap=185.8,
        )
        signals = strategy.generate_signals(bar, None)
        assert signals == []

    def test_warm_up_period(self):
        """warm_up_period() should equal rsi_period + 1."""
        strategy = OptionsWheelStrategy(rsi_period=14)
        assert strategy.warm_up_period() == 15

    def test_sell_put_when_oversold(self):
        """After warm-up, when RSI drops below threshold, should emit sell_put signal."""
        strategy = OptionsWheelStrategy(rsi_period=5, rsi_threshold=40)
        # Rising prices to build up gains, then sharp drops to push RSI low
        prices = [100 + i for i in range(6)] + [106 - i * 3 for i in range(5)]

        mock_portfolio = type("MockPortfolio", (), {"has_position": lambda self, s: False})()

        last_signals: list[SignalEvent] = []
        for i, price in enumerate(prices):
            bar = _make_bar("AAPL", i, price)
            last_signals = strategy.generate_signals(bar, mock_portfolio)

        # After the sharp decline, RSI should be low → sell_put signal
        put_signals = [s for s in last_signals if s.direction == "sell_put"]
        assert len(put_signals) > 0
        assert put_signals[0].option_type == "put"
        assert put_signals[0].strike < prices[-1]  # OTM put is below current price

    def test_sell_call_when_holding(self):
        """When portfolio has a position, should emit sell_call signal."""
        strategy = OptionsWheelStrategy(rsi_period=5)
        mock_portfolio = type("MockPortfolio", (), {"has_position": lambda self, s: True})()

        # Warm up with rising prices (enough bars to satisfy rsi_period + 1)
        for i in range(7):
            bar = _make_bar("AAPL", i, 100 + i)
            strategy.generate_signals(bar, mock_portfolio)

        # Now test: holding stock → covered call
        bar = BarEvent(
            symbol="AAPL",
            date=date(2024, 1, 10),
            open=107,
            high=108,
            low=106,
            close=107,
            adj_close=107,
            volume=1_000_000,
            vwap=107,
        )
        signals = strategy.generate_signals(bar, mock_portfolio)

        call_signals = [s for s in signals if s.direction == "sell_call"]
        assert len(call_signals) > 0
        assert call_signals[0].option_type == "call"
        assert call_signals[0].strike > 107  # OTM call is above current price

    def test_monthly_expiry_calculation(self):
        """_next_monthly_expiry should return the 3rd Friday of the next month."""
        # Jan 15, 2024 → Feb 2024 third Friday = Feb 16 (a Friday)
        expiry = OptionsWheelStrategy._next_monthly_expiry(date(2024, 1, 15))
        assert expiry == date(2024, 2, 16)
        assert expiry.weekday() == 4  # Friday

    def test_monthly_expiry_december_rollover(self):
        """December should roll over to January of the following year."""
        expiry = OptionsWheelStrategy._next_monthly_expiry(date(2024, 12, 1))
        assert expiry.year == 2025
        assert expiry.month == 1
        assert expiry.weekday() == 4  # Friday

    def test_expiry_is_always_friday(self):
        """Expiry must be a Friday for every month of 2024."""
        for month in range(1, 13):
            expiry = OptionsWheelStrategy._next_monthly_expiry(date(2024, month, 1))
            assert expiry.weekday() == 4, (
                f"Expiry for month {month} is not a Friday: {expiry}"
            )

    def test_otm_strike_calculation(self):
        """Put strike should be otm_pct below the current close price."""
        strategy = OptionsWheelStrategy(rsi_period=5, otm_pct=0.05)
        mock_portfolio = type("MockPortfolio", (), {"has_position": lambda self, s: False})()

        # Drive RSI below 40 with sharply declining prices after warm-up
        prices = [100 + i for i in range(6)] + [106 - i * 3 for i in range(5)]
        last_signals: list[SignalEvent] = []
        for i, price in enumerate(prices):
            bar = _make_bar("AAPL", i, price)
            last_signals = strategy.generate_signals(bar, mock_portfolio)

        put_signals = [s for s in last_signals if s.direction == "sell_put"]
        if put_signals:
            last_price = prices[-1]
            expected_strike = round(last_price * 0.95, 2)
            assert put_signals[0].strike == expected_strike

    def test_no_signal_when_rsi_above_threshold_no_position(self):
        """No put signal emitted when RSI is above the threshold and no position held."""
        strategy = OptionsWheelStrategy(rsi_period=5, rsi_threshold=40)
        mock_portfolio = type("MockPortfolio", (), {"has_position": lambda self, s: False})()

        # Steadily rising prices → RSI stays high
        prices = [100 + i * 2 for i in range(12)]
        last_signals: list[SignalEvent] = []
        for i, price in enumerate(prices):
            bar = _make_bar("AAPL", i, price)
            last_signals = strategy.generate_signals(bar, mock_portfolio)

        # No signals: RSI is high and we don't hold the stock
        put_signals = [s for s in last_signals if s.direction == "sell_put"]
        assert put_signals == []

    def test_signal_contains_expiration(self):
        """All emitted signals must carry a valid expiration date."""
        strategy = OptionsWheelStrategy(rsi_period=5, rsi_threshold=40)
        mock_portfolio = type("MockPortfolio", (), {"has_position": lambda self, s: False})()

        prices = [100 + i for i in range(6)] + [106 - i * 3 for i in range(5)]
        all_signals: list[SignalEvent] = []
        for i, price in enumerate(prices):
            bar = _make_bar("AAPL", i, price)
            sigs = strategy.generate_signals(bar, mock_portfolio)
            all_signals.extend(sigs)

        for sig in all_signals:
            assert sig.expiration is not None
            assert isinstance(sig.expiration, date)

    def test_end_to_end_backtest(self, tmp_path):
        """Full backtest with options wheel strategy completes without error."""
        db = Database(str(tmp_path / "test.db"))
        db.create_tables()
        bars = generate_sample_bars()
        db.insert_daily_bars(bars)

        strategy = OptionsWheelStrategy(rsi_period=10, rsi_threshold=40)
        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy,
            database=db,
            universe=["SPY"],
            start_date=start_date,
            end_date=end_date,
            initial_capital=100_000.0,
            benchmark_symbol="SPY",
        )
        metrics = engine.run()

        assert metrics is not None
        assert "total_return" in metrics
        db.close()
