"""Tests for analytics/trade_log.py."""

from __future__ import annotations

from datetime import date

import pytest

from analytics.trade_log import CompletedTrade, TradeLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open(
    log: TradeLog,
    symbol: str = "AAPL",
    direction: str = "long",
    entry_date: date = date(2024, 1, 2),
    entry_price: float = 50.0,
    quantity: int = 100,
    entry_reason: str = "signal",
) -> None:
    log.open_trade(
        symbol=symbol,
        direction=direction,
        entry_date=entry_date,
        entry_price=entry_price,
        quantity=quantity,
        entry_reason=entry_reason,
    )


def _close(
    log: TradeLog,
    symbol: str = "AAPL",
    exit_date: date = date(2024, 1, 10),
    exit_price: float = 55.0,
    exit_reason: str = "target",
) -> CompletedTrade | None:
    return log.close_trade(
        symbol=symbol,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# 1. Open and close a trade – verify CompletedTrade fields
# ---------------------------------------------------------------------------

class TestOpenCloseTrade:
    def test_returns_completed_trade(self):
        log = TradeLog()
        _open(log)
        trade = _close(log)
        assert isinstance(trade, CompletedTrade)

    def test_symbol_preserved(self):
        log = TradeLog()
        _open(log, symbol="MSFT")
        trade = _close(log, symbol="MSFT")
        assert trade.symbol == "MSFT"

    def test_direction_preserved(self):
        log = TradeLog()
        _open(log, direction="long")
        trade = _close(log)
        assert trade.direction == "long"

    def test_entry_date_preserved(self):
        log = TradeLog()
        _open(log, entry_date=date(2024, 3, 1))
        trade = _close(log, exit_date=date(2024, 3, 10))
        assert trade.entry_date == date(2024, 3, 1)

    def test_exit_date_preserved(self):
        log = TradeLog()
        _open(log)
        trade = _close(log, exit_date=date(2024, 1, 15))
        assert trade.exit_date == date(2024, 1, 15)

    def test_entry_price_preserved(self):
        log = TradeLog()
        _open(log, entry_price=123.45)
        trade = _close(log)
        assert trade.entry_price == pytest.approx(123.45)

    def test_exit_price_preserved(self):
        log = TradeLog()
        _open(log)
        trade = _close(log, exit_price=99.0)
        assert trade.exit_price == pytest.approx(99.0)

    def test_entry_reason_preserved(self):
        log = TradeLog()
        _open(log, entry_reason="breakout")
        trade = _close(log)
        assert trade.entry_reason == "breakout"

    def test_exit_reason_preserved(self):
        log = TradeLog()
        _open(log)
        trade = _close(log, exit_reason="stop_loss")
        assert trade.exit_reason == "stop_loss"

    def test_trade_added_to_completed_list(self):
        log = TradeLog()
        _open(log)
        _close(log)
        assert len(log.trades) == 1

    def test_open_trade_removed_from_open(self):
        log = TradeLog()
        _open(log)
        _close(log)
        assert "AAPL" not in log._open_trades


# ---------------------------------------------------------------------------
# 2. Long P&L: buy $50, sell $55, qty 100 → pnl = $500
# ---------------------------------------------------------------------------

class TestLongPnL:
    def test_pnl_is_500(self):
        log = TradeLog()
        _open(log, entry_price=50.0, quantity=100)
        trade = _close(log, exit_price=55.0)
        assert trade.pnl == pytest.approx(500.0)

    def test_pnl_pct_is_10_percent(self):
        log = TradeLog()
        _open(log, entry_price=50.0, quantity=100)
        trade = _close(log, exit_price=55.0)
        assert trade.pnl_pct == pytest.approx(0.10)

    def test_pnl_negative_on_loss(self):
        log = TradeLog()
        _open(log, entry_price=50.0, quantity=100)
        trade = _close(log, exit_price=45.0)
        assert trade.pnl == pytest.approx(-500.0)


# ---------------------------------------------------------------------------
# 3. Short P&L: short $55, cover $50, qty 100 → pnl = $500
# ---------------------------------------------------------------------------

class TestShortPnL:
    def test_short_profit_when_price_falls(self):
        log = TradeLog()
        _open(log, direction="short", entry_price=55.0, quantity=100)
        trade = _close(log, exit_price=50.0)
        assert trade.pnl == pytest.approx(500.0)

    def test_short_loss_when_price_rises(self):
        log = TradeLog()
        _open(log, direction="short", entry_price=50.0, quantity=100)
        trade = _close(log, exit_price=55.0)
        assert trade.pnl == pytest.approx(-500.0)


# ---------------------------------------------------------------------------
# 4. Holding days
# ---------------------------------------------------------------------------

class TestHoldingDays:
    def test_holding_days_8(self):
        log = TradeLog()
        _open(log, entry_date=date(2024, 1, 2))
        trade = _close(log, exit_date=date(2024, 1, 10))
        assert trade.holding_days == 8

    def test_holding_days_zero_same_day(self):
        log = TradeLog()
        d = date(2024, 1, 5)
        _open(log, entry_date=d)
        trade = _close(log, exit_date=d)
        assert trade.holding_days == 0


# ---------------------------------------------------------------------------
# 5. Multiple trades; query by symbol
# ---------------------------------------------------------------------------

class TestMultipleTrades:
    def test_two_symbols_returns_two_trades(self):
        log = TradeLog()
        _open(log, symbol="AAPL")
        _close(log, symbol="AAPL")
        _open(log, symbol="GOOG")
        _close(log, symbol="GOOG")
        assert len(log.get_trades()) == 2

    def test_filter_by_symbol(self):
        log = TradeLog()
        _open(log, symbol="AAPL")
        _close(log, symbol="AAPL")
        _open(log, symbol="GOOG")
        _close(log, symbol="GOOG")
        aapl = log.get_trades("AAPL")
        assert len(aapl) == 1
        assert aapl[0].symbol == "AAPL"

    def test_filter_unknown_symbol_returns_empty(self):
        log = TradeLog()
        _open(log)
        _close(log)
        assert log.get_trades("XYZ") == []

    def test_close_nonexistent_returns_none(self):
        log = TradeLog()
        result = log.close_trade("FAKE", date(2024, 1, 1), 100.0, "test")
        assert result is None


# ---------------------------------------------------------------------------
# 6. get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty_log_returns_zero_total(self):
        log = TradeLog()
        stats = log.get_stats()
        assert stats["total_trades"] == 0

    def test_total_trades_count(self):
        log = TradeLog()
        for i in range(3):
            _open(log, symbol=f"SYM{i}")
            _close(log, symbol=f"SYM{i}")
        assert log.get_stats()["total_trades"] == 3

    def test_winners_and_losers(self):
        log = TradeLog()
        # Win: +500
        _open(log, symbol="W1", entry_price=50.0, quantity=100)
        _close(log, symbol="W1", exit_price=55.0)
        # Loss: -500
        _open(log, symbol="L1", entry_price=50.0, quantity=100)
        _close(log, symbol="L1", exit_price=45.0)
        stats = log.get_stats()
        assert stats["winners"] == 1
        assert stats["losers"] == 1

    def test_avg_holding_days(self):
        log = TradeLog()
        _open(log, symbol="A", entry_date=date(2024, 1, 1))
        _close(log, symbol="A", exit_date=date(2024, 1, 11))   # 10 days
        _open(log, symbol="B", entry_date=date(2024, 1, 1))
        _close(log, symbol="B", exit_date=date(2024, 1, 21))   # 20 days
        assert log.get_stats()["avg_holding_days"] == pytest.approx(15.0)

    def test_largest_win(self):
        log = TradeLog()
        _open(log, symbol="A", entry_price=50.0, quantity=100)
        _close(log, symbol="A", exit_price=55.0)    # +500
        _open(log, symbol="B", entry_price=50.0, quantity=200)
        _close(log, symbol="B", exit_price=60.0)    # +2000
        assert log.get_stats()["largest_win"] == pytest.approx(2000.0)

    def test_largest_loss(self):
        log = TradeLog()
        _open(log, symbol="A", entry_price=50.0, quantity=100)
        _close(log, symbol="A", exit_price=45.0)    # -500
        _open(log, symbol="B", entry_price=50.0, quantity=200)
        _close(log, symbol="B", exit_price=40.0)    # -2000
        assert log.get_stats()["largest_loss"] == pytest.approx(-2000.0)


# ---------------------------------------------------------------------------
# 7. Consecutive streaks: W W W L L W → max_win=3, max_loss=2
# ---------------------------------------------------------------------------

class TestStreaks:
    def _build_log(self, outcomes: list[float]) -> TradeLog:
        """outcomes[i] > 0 means win, <= 0 means loss."""
        log = TradeLog()
        for i, pnl_target in enumerate(outcomes):
            sym = f"S{i}"
            # entry_price=100, qty=1; choose exit to match desired sign
            if pnl_target > 0:
                _open(log, symbol=sym, entry_price=100.0, quantity=1)
                _close(log, symbol=sym, exit_price=101.0)
            else:
                _open(log, symbol=sym, entry_price=100.0, quantity=1)
                _close(log, symbol=sym, exit_price=99.0)
        return log

    def test_max_win_streak_three(self):
        log = self._build_log([1, 1, 1, -1, -1, 1])
        assert log.get_stats()["max_win_streak"] == 3

    def test_max_loss_streak_two(self):
        log = self._build_log([1, 1, 1, -1, -1, 1])
        assert log.get_stats()["max_loss_streak"] == 2

    def test_all_wins_streak(self):
        log = self._build_log([1, 1, 1, 1])
        assert log.get_stats()["max_win_streak"] == 4
        assert log.get_stats()["max_loss_streak"] == 0

    def test_all_losses_streak(self):
        log = self._build_log([-1, -1, -1])
        assert log.get_stats()["max_loss_streak"] == 3
        assert log.get_stats()["max_win_streak"] == 0


# ---------------------------------------------------------------------------
# 8. get_trade_dicts
# ---------------------------------------------------------------------------

class TestGetTradeDicts:
    def test_returns_list_of_dicts(self):
        log = TradeLog()
        _open(log)
        _close(log)
        dicts = log.get_trade_dicts()
        assert isinstance(dicts, list)
        assert isinstance(dicts[0], dict)

    def test_has_pnl_key(self):
        log = TradeLog()
        _open(log, entry_price=50.0, quantity=100)
        _close(log, exit_price=55.0)
        assert "pnl" in log.get_trade_dicts()[0]

    def test_pnl_value_correct(self):
        log = TradeLog()
        _open(log, entry_price=50.0, quantity=100)
        _close(log, exit_price=55.0)
        assert log.get_trade_dicts()[0]["pnl"] == pytest.approx(500.0)

    def test_has_symbol_key(self):
        log = TradeLog()
        _open(log, symbol="TSLA")
        _close(log, symbol="TSLA")
        assert log.get_trade_dicts()[0]["symbol"] == "TSLA"

    def test_has_holding_days_key(self):
        log = TradeLog()
        _open(log, entry_date=date(2024, 1, 1))
        _close(log, exit_date=date(2024, 1, 6))
        assert log.get_trade_dicts()[0]["holding_days"] == 5

    def test_has_direction_key(self):
        log = TradeLog()
        _open(log, direction="short")
        _close(log)
        assert log.get_trade_dicts()[0]["direction"] == "short"
