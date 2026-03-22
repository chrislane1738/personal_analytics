"""Tests for analytics/regime.py."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from analytics.regime import MarketRegime, compute_regime_stats, detect_regimes
from analytics.trade_log import TradeLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_range(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _uptrend_prices(n: int = 300, start: float = 100.0, daily_gain: float = 0.002) -> list[tuple[date, float]]:
    """Steadily rising prices."""
    dates = _date_range(date(2020, 1, 2), n)
    prices = [start * (1 + daily_gain) ** i for i in range(n)]
    return list(zip(dates, prices))


def _downtrend_prices(n: int = 300, start: float = 200.0, daily_loss: float = 0.002) -> list[tuple[date, float]]:
    """Steadily falling prices."""
    dates = _date_range(date(2020, 1, 2), n)
    prices = [start * (1 - daily_loss) ** i for i in range(n)]
    return list(zip(dates, prices))


def _flat_prices(n: int = 300, base: float = 100.0, noise: float = 0.001) -> list[tuple[date, float]]:
    """Nearly flat prices (tiny oscillation)."""
    import math
    dates = _date_range(date(2020, 1, 2), n)
    prices = [base + noise * math.sin(i / 5.0) for i in range(n)]
    return list(zip(dates, prices))


# ---------------------------------------------------------------------------
# 1. Uptrend → mostly BULL
# ---------------------------------------------------------------------------

class TestBullRegime:
    def test_majority_bull_in_uptrend(self):
        prices = _uptrend_prices(n=300)
        regimes = detect_regimes(prices)
        assert len(regimes) == 300
        bull_count = sum(1 for _, r in regimes if r == MarketRegime.BULL)
        # After SMA-200 warm-up (200 bars) we expect most to be BULL
        # At minimum >50% of the total series
        assert bull_count > len(regimes) * 0.50

    def test_no_bear_in_strong_uptrend(self):
        prices = _uptrend_prices(n=300)
        regimes = detect_regimes(prices)
        bear_count = sum(1 for _, r in regimes if r == MarketRegime.BEAR)
        # Strong uptrend should have very few (if any) bear days
        assert bear_count < len(regimes) * 0.05

    def test_returns_correct_length(self):
        prices = _uptrend_prices(n=250)
        regimes = detect_regimes(prices)
        assert len(regimes) == 250

    def test_dates_match_input(self):
        prices = _uptrend_prices(n=50)
        regimes = detect_regimes(prices)
        input_dates = [d for d, _ in prices]
        output_dates = [d for d, _ in regimes]
        assert input_dates == output_dates


# ---------------------------------------------------------------------------
# 2. Downtrend → mostly BEAR
# ---------------------------------------------------------------------------

class TestBearRegime:
    def test_majority_bear_in_downtrend(self):
        prices = _downtrend_prices(n=300)
        regimes = detect_regimes(prices)
        bear_count = sum(1 for _, r in regimes if r == MarketRegime.BEAR)
        assert bear_count > len(regimes) * 0.50

    def test_no_bull_in_strong_downtrend(self):
        prices = _downtrend_prices(n=300)
        regimes = detect_regimes(prices)
        bull_count = sum(1 for _, r in regimes if r == MarketRegime.BULL)
        assert bull_count < len(regimes) * 0.05


# ---------------------------------------------------------------------------
# 3. Flat prices → mostly SIDEWAYS
# ---------------------------------------------------------------------------

class TestSidewaysRegime:
    def test_majority_sideways_in_flat_market(self):
        prices = _flat_prices(n=300)
        regimes = detect_regimes(prices)
        sideways_count = sum(1 for _, r in regimes if r == MarketRegime.SIDEWAYS)
        assert sideways_count > len(regimes) * 0.50

    def test_no_bull_or_bear_in_flat(self):
        prices = _flat_prices(n=300)
        regimes = detect_regimes(prices)
        directional = sum(
            1 for _, r in regimes
            if r in (MarketRegime.BULL, MarketRegime.BEAR)
        )
        assert directional < len(regimes) * 0.05


# ---------------------------------------------------------------------------
# 4. High VIX overrides to HIGH_VOL
# ---------------------------------------------------------------------------

class TestHighVolRegime:
    def _make_rising_prices(self, n: int = 50) -> list[tuple[date, float]]:
        """Short uptrend series."""
        dates = _date_range(date(2022, 1, 3), n)
        prices = [100.0 + i * 0.5 for i in range(n)]
        return list(zip(dates, prices))

    def test_high_vix_overrides_to_high_vol(self):
        prices = self._make_rising_prices(50)
        # All VIX values above 25
        vix = [(d, 30.0) for d, _ in prices]
        regimes = detect_regimes(prices, vix_prices=vix)
        assert all(r == MarketRegime.HIGH_VOL for _, r in regimes)

    def test_low_vix_does_not_trigger_high_vol(self):
        prices = _uptrend_prices(n=300)
        vix = [(d, 15.0) for d, _ in prices]
        regimes = detect_regimes(prices, vix_prices=vix)
        high_vol_count = sum(1 for _, r in regimes if r == MarketRegime.HIGH_VOL)
        assert high_vol_count == 0

    def test_partial_high_vix_affects_subset(self):
        prices = _uptrend_prices(n=300)
        all_dates = [d for d, _ in prices]
        # VIX > 25 only for first 30 dates
        vix = [(d, 30.0 if i < 30 else 15.0) for i, d in enumerate(all_dates)]
        regimes = detect_regimes(prices, vix_prices=vix)
        high_vol_count = sum(1 for _, r in regimes if r == MarketRegime.HIGH_VOL)
        assert high_vol_count == 30

    def test_no_vix_no_high_vol_in_flat(self):
        prices = _flat_prices(n=300)
        regimes = detect_regimes(prices)
        high_vol_count = sum(1 for _, r in regimes if r == MarketRegime.HIGH_VOL)
        assert high_vol_count == 0

    def test_empty_prices_returns_empty(self):
        assert detect_regimes([]) == []


# ---------------------------------------------------------------------------
# 5. compute_regime_stats
# ---------------------------------------------------------------------------

class TestComputeRegimeStats:
    def _make_regime_map(self, date_regime_pairs: list[tuple[date, MarketRegime]]) -> list[tuple[date, MarketRegime]]:
        return date_regime_pairs

    def test_bull_trades_grouped_correctly(self):
        bull_date = date(2024, 6, 1)
        bear_date = date(2024, 6, 2)
        regimes = [
            (bull_date, MarketRegime.BULL),
            (bear_date, MarketRegime.BEAR),
        ]
        trades = [
            {"entry_date": bull_date, "pnl": 200.0},
            {"entry_date": bull_date, "pnl": 300.0},
            {"entry_date": bear_date, "pnl": -100.0},
        ]
        stats = compute_regime_stats(trades, regimes)
        assert stats["bull"]["trades"] == 2
        assert stats["bear"]["trades"] == 1

    def test_win_rate_calculation(self):
        d1, d2, d3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
        regimes = [
            (d1, MarketRegime.BULL),
            (d2, MarketRegime.BULL),
            (d3, MarketRegime.BULL),
        ]
        trades = [
            {"entry_date": d1, "pnl": 100.0},
            {"entry_date": d2, "pnl": -50.0},
            {"entry_date": d3, "pnl": 200.0},
        ]
        stats = compute_regime_stats(trades, regimes)
        assert stats["bull"]["win_rate"] == pytest.approx(2 / 3)

    def test_avg_pnl_calculation(self):
        d1, d2 = date(2024, 2, 1), date(2024, 2, 2)
        regimes = [(d1, MarketRegime.BEAR), (d2, MarketRegime.BEAR)]
        trades = [
            {"entry_date": d1, "pnl": -100.0},
            {"entry_date": d2, "pnl": -200.0},
        ]
        stats = compute_regime_stats(trades, regimes)
        assert stats["bear"]["avg_pnl"] == pytest.approx(-150.0)

    def test_empty_regime_has_zero_trades(self):
        bull_date = date(2024, 3, 1)
        regimes = [(bull_date, MarketRegime.BULL)]
        trades = [{"entry_date": bull_date, "pnl": 100.0}]
        stats = compute_regime_stats(trades, regimes)
        assert stats["bear"]["trades"] == 0
        assert stats["sideways"]["trades"] == 0
        assert stats["high_vol"]["trades"] == 0

    def test_works_with_completed_trade_objects(self):
        """compute_regime_stats should accept CompletedTrade objects too."""
        log = TradeLog()
        trade_date = date(2024, 4, 1)
        log.open_trade(
            symbol="SPY",
            direction="long",
            entry_date=trade_date,
            entry_price=400.0,
            quantity=10,
            entry_reason="test",
        )
        trade = log.close_trade("SPY", date(2024, 4, 10), 420.0, "target")
        regimes = [(trade_date, MarketRegime.BULL)]
        stats = compute_regime_stats([trade], regimes)
        assert stats["bull"]["trades"] == 1
        assert stats["bull"]["avg_pnl"] == pytest.approx(200.0)

    def test_total_pnl_field_present(self):
        d = date(2024, 5, 1)
        regimes = [(d, MarketRegime.SIDEWAYS)]
        trades = [{"entry_date": d, "pnl": 75.0}]
        stats = compute_regime_stats(trades, regimes)
        assert stats["sideways"]["total_pnl"] == pytest.approx(75.0)

    def test_trade_date_not_in_regimes_is_skipped(self):
        """Trades whose entry_date doesn't appear in regimes are ignored."""
        regimes = [(date(2024, 1, 1), MarketRegime.BULL)]
        trades = [{"entry_date": date(2030, 1, 1), "pnl": 999.0}]
        stats = compute_regime_stats(trades, regimes)
        assert all(s.get("trades", 0) == 0 for s in stats.values())
