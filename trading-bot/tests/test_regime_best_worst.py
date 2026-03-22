"""Tests for regime stats best_trade / worst_trade fields (Task 0.5).

Validates that compute_regime_stats includes best_trade and worst_trade
in the per-regime result dictionaries.
"""

from __future__ import annotations

from datetime import date

import pytest

from analytics.regime import MarketRegime, compute_regime_stats


# ---------------------------------------------------------------------------
# 1. best_trade and worst_trade are present
# ---------------------------------------------------------------------------

class TestBestWorstTradeFields:
    def test_fields_present_for_regime_with_trades(self):
        d1, d2, d3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
        regimes = [
            (d1, MarketRegime.BULL),
            (d2, MarketRegime.BULL),
            (d3, MarketRegime.BULL),
        ]
        trades = [
            {"entry_date": d1, "pnl": 200.0},
            {"entry_date": d2, "pnl": -50.0},
            {"entry_date": d3, "pnl": 500.0},
        ]
        stats = compute_regime_stats(trades, regimes)
        assert "best_trade" in stats["bull"]
        assert "worst_trade" in stats["bull"]

    def test_fields_absent_for_empty_regime(self):
        """A regime with 0 trades should NOT have best_trade / worst_trade."""
        d1 = date(2024, 1, 1)
        regimes = [(d1, MarketRegime.BULL)]
        trades = [{"entry_date": d1, "pnl": 100.0}]
        stats = compute_regime_stats(trades, regimes)
        # bear has 0 trades
        assert "best_trade" not in stats["bear"]
        assert "worst_trade" not in stats["bear"]


# ---------------------------------------------------------------------------
# 2. Correct values
# ---------------------------------------------------------------------------

class TestBestWorstTradeValues:
    def test_best_and_worst_correct(self):
        d1, d2, d3 = date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 3)
        regimes = [
            (d1, MarketRegime.BEAR),
            (d2, MarketRegime.BEAR),
            (d3, MarketRegime.BEAR),
        ]
        trades = [
            {"entry_date": d1, "pnl": -300.0},
            {"entry_date": d2, "pnl": -100.0},
            {"entry_date": d3, "pnl": 50.0},
        ]
        stats = compute_regime_stats(trades, regimes)
        assert stats["bear"]["best_trade"] == pytest.approx(50.0)
        assert stats["bear"]["worst_trade"] == pytest.approx(-300.0)

    def test_single_trade_best_equals_worst(self):
        d = date(2024, 3, 1)
        regimes = [(d, MarketRegime.SIDEWAYS)]
        trades = [{"entry_date": d, "pnl": 75.0}]
        stats = compute_regime_stats(trades, regimes)
        assert stats["sideways"]["best_trade"] == pytest.approx(75.0)
        assert stats["sideways"]["worst_trade"] == pytest.approx(75.0)

    def test_all_negative_trades(self):
        d1, d2 = date(2024, 4, 1), date(2024, 4, 2)
        regimes = [(d1, MarketRegime.HIGH_VOL), (d2, MarketRegime.HIGH_VOL)]
        trades = [
            {"entry_date": d1, "pnl": -500.0},
            {"entry_date": d2, "pnl": -100.0},
        ]
        stats = compute_regime_stats(trades, regimes)
        assert stats["high_vol"]["best_trade"] == pytest.approx(-100.0)
        assert stats["high_vol"]["worst_trade"] == pytest.approx(-500.0)

    def test_all_positive_trades(self):
        d1, d2, d3 = date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3)
        regimes = [
            (d1, MarketRegime.BULL),
            (d2, MarketRegime.BULL),
            (d3, MarketRegime.BULL),
        ]
        trades = [
            {"entry_date": d1, "pnl": 100.0},
            {"entry_date": d2, "pnl": 400.0},
            {"entry_date": d3, "pnl": 200.0},
        ]
        stats = compute_regime_stats(trades, regimes)
        assert stats["bull"]["best_trade"] == pytest.approx(400.0)
        assert stats["bull"]["worst_trade"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 3. Multiple regimes
# ---------------------------------------------------------------------------

class TestBestWorstAcrossRegimes:
    def test_each_regime_tracks_independently(self):
        d_bull = date(2024, 6, 1)
        d_bear = date(2024, 6, 2)
        regimes = [
            (d_bull, MarketRegime.BULL),
            (d_bear, MarketRegime.BEAR),
        ]
        trades = [
            {"entry_date": d_bull, "pnl": 1000.0},
            {"entry_date": d_bull, "pnl": -200.0},
            {"entry_date": d_bear, "pnl": -500.0},
            {"entry_date": d_bear, "pnl": 50.0},
        ]
        stats = compute_regime_stats(trades, regimes)

        assert stats["bull"]["best_trade"] == pytest.approx(1000.0)
        assert stats["bull"]["worst_trade"] == pytest.approx(-200.0)
        assert stats["bear"]["best_trade"] == pytest.approx(50.0)
        assert stats["bear"]["worst_trade"] == pytest.approx(-500.0)
