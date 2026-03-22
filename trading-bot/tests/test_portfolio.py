"""Tests for portfolio/, accounting/, and benchmark modules."""

from __future__ import annotations

from datetime import date

import pytest

from core.event_bus import EventBus
from core.events import EventType, FillEvent, PortfolioUpdateEvent
from portfolio.accounting import AccountingLedger, Transaction
from portfolio.benchmark import BenchmarkTracker
from portfolio.portfolio import Portfolio, Position


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_fill(
    symbol: str = "AAPL",
    action: str = "BUY",
    quantity: int = 100,
    fill_price: float = 50.0,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> FillEvent:
    return FillEvent(
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=fill_price,
        commission=commission,
        slippage=slippage,
    )


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def portfolio(bus: EventBus) -> Portfolio:
    return Portfolio(initial_capital=100_000.0, event_bus=bus)


# ---------------------------------------------------------------------------
# 1. Buy test
# ---------------------------------------------------------------------------

class TestBuy:
    def test_cash_decreases_by_cost_plus_commission(self, portfolio: Portfolio):
        fill = make_fill(quantity=100, fill_price=50.0, commission=5.0)
        portfolio.update_on_fill(fill, date(2024, 1, 2))
        # 100 * 50 + 5 commission = 5005
        assert portfolio.cash == pytest.approx(100_000.0 - 5_005.0)

    def test_position_created(self, portfolio: Portfolio):
        fill = make_fill(quantity=100, fill_price=50.0)
        portfolio.update_on_fill(fill, date(2024, 1, 2))
        assert portfolio.has_position("AAPL")

    def test_position_quantity(self, portfolio: Portfolio):
        fill = make_fill(quantity=100, fill_price=50.0)
        portfolio.update_on_fill(fill, date(2024, 1, 2))
        assert portfolio.positions["AAPL"].quantity == 100

    def test_position_avg_cost(self, portfolio: Portfolio):
        fill = make_fill(quantity=100, fill_price=50.0)
        portfolio.update_on_fill(fill, date(2024, 1, 2))
        assert portfolio.positions["AAPL"].avg_cost == pytest.approx(50.0)

    def test_cash_no_commission(self, portfolio: Portfolio):
        fill = make_fill(quantity=100, fill_price=50.0, commission=0.0)
        portfolio.update_on_fill(fill, date(2024, 1, 2))
        assert portfolio.cash == pytest.approx(95_000.0)

    def test_second_buy_averages_cost(self, portfolio: Portfolio):
        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0), date(2024, 1, 2))
        portfolio.update_on_fill(make_fill(quantity=100, fill_price=60.0), date(2024, 1, 3))
        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 200
        assert pos.avg_cost == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# 2. Sell test
# ---------------------------------------------------------------------------

class TestSell:
    def _buy_then_sell(
        self,
        portfolio: Portfolio,
        buy_price: float = 50.0,
        sell_price: float = 55.0,
        commission: float = 0.0,
    ) -> None:
        portfolio.update_on_fill(
            make_fill(action="BUY", quantity=100, fill_price=buy_price, commission=commission),
            date(2024, 1, 2),
        )
        portfolio.update_on_fill(
            make_fill(action="SELL", quantity=100, fill_price=sell_price, commission=commission),
            date(2024, 1, 3),
        )

    def test_position_removed_after_full_sell(self, portfolio: Portfolio):
        self._buy_then_sell(portfolio)
        assert not portfolio.has_position("AAPL")

    def test_cash_increases_on_sell(self, portfolio: Portfolio):
        self._buy_then_sell(portfolio, buy_price=50.0, sell_price=55.0, commission=0.0)
        # Started 100k, bought 100@50 = -5000, sold 100@55 = +5500 → 100500
        assert portfolio.cash == pytest.approx(100_500.0)

    def test_realized_pnl_after_sell(self, portfolio: Portfolio):
        self._buy_then_sell(portfolio, buy_price=50.0, sell_price=55.0, commission=0.0)
        assert portfolio._realized_pnl == pytest.approx(500.0)

    def test_realized_pnl_accounts_for_commissions(self, portfolio: Portfolio):
        # Buy commission=5, sell commission=5: P&L = (55-50)*100 - 5 = 495
        self._buy_then_sell(portfolio, buy_price=50.0, sell_price=55.0, commission=5.0)
        assert portfolio._realized_pnl == pytest.approx(495.0)

    def test_partial_sell_reduces_quantity(self, portfolio: Portfolio):
        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0), date(2024, 1, 2))
        portfolio.update_on_fill(
            make_fill(action="SELL", quantity=40, fill_price=55.0), date(2024, 1, 3)
        )
        assert portfolio.positions["AAPL"].quantity == 60


# ---------------------------------------------------------------------------
# 3. Unrealized P&L test
# ---------------------------------------------------------------------------

class TestUnrealizedPnL:
    def test_unrealized_pnl_via_get_equity(self, portfolio: Portfolio):
        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0), date(2024, 1, 2))
        equity = portfolio.get_equity({"AAPL": 55.0})
        # cash = 95000, position value = 100 * 55 = 5500 → total = 100500
        assert equity == pytest.approx(100_500.0)
        # Unrealized gain = 500
        pos = portfolio.positions["AAPL"]
        pos.update_unrealized_pnl(55.0)
        assert pos.unrealized_pnl == pytest.approx(500.0)

    def test_unrealized_pnl_loss(self, portfolio: Portfolio):
        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0), date(2024, 1, 2))
        pos = portfolio.positions["AAPL"]
        pos.update_unrealized_pnl(45.0)
        assert pos.unrealized_pnl == pytest.approx(-500.0)


# ---------------------------------------------------------------------------
# 4. Drawdown test
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_no_drawdown_at_start(self, portfolio: Portfolio):
        assert portfolio.get_drawdown() == pytest.approx(0.0)

    def test_drawdown_after_equity_drops(self, portfolio: Portfolio):
        # Buy to consume cash, then check drawdown with lower prices
        portfolio.update_on_fill(
            make_fill(quantity=1000, fill_price=100.0, commission=0.0), date(2024, 1, 2)
        )
        # high_water_mark was 100k at start; after buy cash = 0, position @100 = equity 100k
        # Now price falls: equity = 0 + 1000 * 90 = 90000
        drawdown = portfolio.get_drawdown()
        # After the buy, avg_cost=100, hwm updated to 100k, drawdown=0
        assert drawdown == pytest.approx(0.0)

    def test_drawdown_negative_percentage(self, portfolio: Portfolio):
        # Manually set high_water_mark above current equity
        portfolio.high_water_mark = 100_000.0
        # current equity (all cash, no positions) = 90000
        portfolio.cash = 90_000.0
        drawdown = portfolio.get_drawdown()
        assert drawdown == pytest.approx(-0.10)

    def test_drawdown_capped_at_zero(self, portfolio: Portfolio):
        portfolio.cash = 110_000.0  # equity above hwm
        portfolio.high_water_mark = 100_000.0
        assert portfolio.get_drawdown() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. PortfolioUpdateEvent emission test
# ---------------------------------------------------------------------------

class TestPortfolioUpdateEvent:
    def test_event_emitted_on_buy(self, portfolio: Portfolio, bus: EventBus):
        received: list[PortfolioUpdateEvent] = []
        bus.subscribe(EventType.PORTFOLIO_UPDATE, received.append)

        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0), date(2024, 1, 2))

        assert len(received) == 1
        assert isinstance(received[0], PortfolioUpdateEvent)

    def test_event_emitted_on_sell(self, portfolio: Portfolio, bus: EventBus):
        received: list[PortfolioUpdateEvent] = []
        bus.subscribe(EventType.PORTFOLIO_UPDATE, received.append)

        portfolio.update_on_fill(make_fill(action="BUY", quantity=100, fill_price=50.0), date(2024, 1, 2))
        portfolio.update_on_fill(make_fill(action="SELL", quantity=100, fill_price=55.0), date(2024, 1, 3))

        assert len(received) == 2

    def test_event_contains_correct_cash(self, portfolio: Portfolio, bus: EventBus):
        received: list[PortfolioUpdateEvent] = []
        bus.subscribe(EventType.PORTFOLIO_UPDATE, received.append)

        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0, commission=0.0), date(2024, 1, 2))

        assert received[0].cash == pytest.approx(95_000.0)

    def test_event_equity_type(self, portfolio: Portfolio, bus: EventBus):
        received: list[PortfolioUpdateEvent] = []
        bus.subscribe(EventType.PORTFOLIO_UPDATE, received.append)

        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0), date(2024, 1, 2))

        assert isinstance(received[0].equity, float)

    def test_event_drawdown_non_positive(self, portfolio: Portfolio, bus: EventBus):
        received: list[PortfolioUpdateEvent] = []
        bus.subscribe(EventType.PORTFOLIO_UPDATE, received.append)

        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0), date(2024, 1, 2))

        assert received[0].drawdown_pct <= 0.0


# ---------------------------------------------------------------------------
# 6. Equity curve test
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_three_entries_recorded(self, portfolio: Portfolio):
        portfolio.record_equity(date(2024, 1, 1), {})
        portfolio.record_equity(date(2024, 1, 2), {})
        portfolio.record_equity(date(2024, 1, 3), {})
        assert len(portfolio.equity_curve) == 3

    def test_dates_are_stored(self, portfolio: Portfolio):
        d1, d2 = date(2024, 1, 1), date(2024, 1, 2)
        portfolio.record_equity(d1, {})
        portfolio.record_equity(d2, {})
        assert portfolio.equity_curve[0][0] == d1
        assert portfolio.equity_curve[1][0] == d2

    def test_equity_reflects_positions(self, portfolio: Portfolio):
        portfolio.update_on_fill(make_fill(quantity=100, fill_price=50.0, commission=0.0), date(2024, 1, 2))
        portfolio.record_equity(date(2024, 1, 2), {"AAPL": 60.0})
        # cash=95000, position=100*60=6000 → 101000
        assert portfolio.equity_curve[0][1] == pytest.approx(101_000.0)

    def test_hwm_updated_via_record_equity(self, portfolio: Portfolio):
        portfolio.high_water_mark = 100_000.0
        portfolio.cash = 110_000.0
        portfolio.record_equity(date(2024, 1, 1), {})
        assert portfolio.high_water_mark == pytest.approx(110_000.0)


# ---------------------------------------------------------------------------
# 7. Benchmark test
# ---------------------------------------------------------------------------

class TestBenchmarkTracker:
    def test_equity_curve_length(self):
        bt = BenchmarkTracker(initial_capital=100_000.0)
        prices = [100.0, 102.0, 101.0, 104.0, 103.0]
        for i, price in enumerate(prices):
            bt.update(date(2024, 1, i + 1), price)
        assert len(bt.equity_curve) == 5

    def test_first_equity_equals_initial_capital(self):
        bt = BenchmarkTracker(initial_capital=100_000.0)
        bt.update(date(2024, 1, 1), 100.0)
        assert bt.equity_curve[0][1] == pytest.approx(100_000.0)

    def test_equity_scales_correctly(self):
        bt = BenchmarkTracker(initial_capital=100_000.0)
        bt.update(date(2024, 1, 1), 100.0)  # buys 1000 shares
        bt.update(date(2024, 1, 2), 110.0)
        assert bt.equity_curve[1][1] == pytest.approx(110_000.0)

    def test_start_price_is_first_price(self):
        bt = BenchmarkTracker(initial_capital=50_000.0)
        bt.update(date(2024, 1, 1), 200.0)  # buys 250 shares
        bt.update(date(2024, 1, 2), 250.0)
        assert bt.equity_curve[1][1] == pytest.approx(62_500.0)

    def test_returns_length_matches_equity_curve(self):
        bt = BenchmarkTracker()
        for i in range(5):
            bt.update(date(2024, 1, i + 1), 100.0 + i)
        returns = bt.get_returns()
        assert len(returns) == 5

    def test_first_return_is_zero(self):
        bt = BenchmarkTracker()
        bt.update(date(2024, 1, 1), 100.0)
        assert bt.get_returns()[0] == pytest.approx(0.0)

    def test_daily_return_calculation(self):
        bt = BenchmarkTracker(initial_capital=100_000.0)
        bt.update(date(2024, 1, 1), 100.0)
        bt.update(date(2024, 1, 2), 105.0)
        returns = bt.get_returns()
        assert returns[1] == pytest.approx(0.05)

    def test_empty_tracker_returns_empty(self):
        bt = BenchmarkTracker()
        assert bt.get_returns() == []


# ---------------------------------------------------------------------------
# 8. Accounting ledger test
# ---------------------------------------------------------------------------

class TestAccountingLedger:
    def _make_txn(
        self,
        symbol: str = "AAPL",
        action: str = "BUY",
        quantity: int = 100,
        price: float = 50.0,
        commission: float = 0.0,
        pnl: float = 0.0,
        txn_date: date = date(2024, 1, 2),
    ) -> Transaction:
        return Transaction(
            date=txn_date,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            commission=commission,
            pnl=pnl,
        )

    def test_record_three_transactions(self):
        ledger = AccountingLedger()
        for i in range(3):
            ledger.record_transaction(self._make_txn(txn_date=date(2024, 1, i + 1)))
        assert len(ledger.transactions) == 3

    def test_get_transactions_all(self):
        ledger = AccountingLedger()
        ledger.record_transaction(self._make_txn(symbol="AAPL"))
        ledger.record_transaction(self._make_txn(symbol="GOOG"))
        ledger.record_transaction(self._make_txn(symbol="AAPL"))
        assert len(ledger.get_transactions()) == 3

    def test_get_transactions_filtered_by_symbol(self):
        ledger = AccountingLedger()
        ledger.record_transaction(self._make_txn(symbol="AAPL"))
        ledger.record_transaction(self._make_txn(symbol="GOOG"))
        ledger.record_transaction(self._make_txn(symbol="AAPL"))
        aapl_txns = ledger.get_transactions("AAPL")
        assert len(aapl_txns) == 2
        assert all(t.symbol == "AAPL" for t in aapl_txns)

    def test_realized_pnl_accumulates(self):
        ledger = AccountingLedger()
        ledger.record_transaction(self._make_txn(pnl=200.0))
        ledger.record_transaction(self._make_txn(pnl=300.0))
        ledger.record_transaction(self._make_txn(pnl=-50.0))
        assert ledger.get_realized_pnl() == pytest.approx(450.0)

    def test_realized_pnl_starts_at_zero(self):
        ledger = AccountingLedger()
        assert ledger.get_realized_pnl() == 0.0

    def test_get_transactions_unknown_symbol_returns_empty(self):
        ledger = AccountingLedger()
        ledger.record_transaction(self._make_txn(symbol="AAPL"))
        assert ledger.get_transactions("XYZ") == []

    def test_transaction_fields_preserved(self):
        ledger = AccountingLedger()
        txn = self._make_txn(
            symbol="MSFT",
            action="SELL",
            quantity=50,
            price=300.0,
            commission=2.5,
            pnl=750.0,
        )
        ledger.record_transaction(txn)
        stored = ledger.transactions[0]
        assert stored.symbol == "MSFT"
        assert stored.action == "SELL"
        assert stored.quantity == 50
        assert stored.price == pytest.approx(300.0)
        assert stored.commission == pytest.approx(2.5)
        assert stored.pnl == pytest.approx(750.0)


# ---------------------------------------------------------------------------
# Position dataclass tests
# ---------------------------------------------------------------------------

class TestPosition:
    def test_market_value(self):
        pos = Position(symbol="AAPL", quantity=100, avg_cost=50.0)
        assert pos.market_value == pytest.approx(5_000.0)

    def test_market_value_uses_abs_quantity(self):
        pos = Position(symbol="AAPL", quantity=-50, avg_cost=50.0)
        assert pos.market_value == pytest.approx(2_500.0)

    def test_update_unrealized_pnl_gain(self):
        pos = Position(symbol="AAPL", quantity=100, avg_cost=50.0)
        pos.update_unrealized_pnl(55.0)
        assert pos.unrealized_pnl == pytest.approx(500.0)

    def test_update_unrealized_pnl_loss(self):
        pos = Position(symbol="AAPL", quantity=100, avg_cost=50.0)
        pos.update_unrealized_pnl(45.0)
        assert pos.unrealized_pnl == pytest.approx(-500.0)
