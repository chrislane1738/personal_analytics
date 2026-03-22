"""Tests for SimBroker: slippage, commission, fill logic, and cancellation."""

from __future__ import annotations

import pytest

from core.event_bus import EventBus
from core.events import EventType, FillEvent, OrderEvent
from execution.sim_broker import SimBroker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_broker(slippage_pct: float = 0.0001, commission_per_share: float = 0.005):
    bus = EventBus()
    broker = SimBroker(bus, slippage_pct=slippage_pct, commission_per_share=commission_per_share)
    return bus, broker


def collected_fills(bus: EventBus) -> list[FillEvent]:
    return bus.get_history(EventType.FILL)


def market_order(symbol: str, action: str, quantity: int) -> OrderEvent:
    return OrderEvent(symbol=symbol, action=action, order_type="market", quantity=quantity)


def limit_order(symbol: str, action: str, quantity: int, price: float) -> OrderEvent:
    return OrderEvent(symbol=symbol, action=action, order_type="limit", quantity=quantity, price=price)


def stop_order(symbol: str, action: str, quantity: int, price: float) -> OrderEvent:
    return OrderEvent(symbol=symbol, action=action, order_type="stop", quantity=quantity, price=price)


# ---------------------------------------------------------------------------
# 1. Market buy — fill price includes positive slippage
# ---------------------------------------------------------------------------

def test_market_buy_fill():
    bus, broker = make_broker()
    order = market_order("AAPL", "BUY", 100)
    broker.submit_order(order)

    broker.process_bar("AAPL", open_price=150.0, high=152.0, low=149.0, close=151.0, volume=1_000_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    fill: FillEvent = fills[0]

    expected_price = 150.0 * 1.0001  # slippage increases buy price
    assert fill.fill_price == pytest.approx(expected_price, rel=1e-6)
    assert fill.commission == pytest.approx(100 * 0.005, rel=1e-6)
    assert fill.quantity == 100
    assert fill.symbol == "AAPL"
    assert fill.action == "BUY"
    assert str(order.event_id) == fill.order_ref


def test_market_buy_removes_order_from_pending():
    bus, broker = make_broker()
    broker.submit_order(market_order("AAPL", "BUY", 100))
    broker.process_bar("AAPL", open_price=150.0, high=152.0, low=149.0, close=151.0, volume=1_000_000)
    assert broker.get_pending_orders() == []


# ---------------------------------------------------------------------------
# 2. Market sell — slippage reduces fill price
# ---------------------------------------------------------------------------

def test_market_sell_fill():
    bus, broker = make_broker()
    order = market_order("AAPL", "SELL", 50)
    broker.submit_order(order)

    broker.process_bar("AAPL", open_price=150.0, high=152.0, low=149.0, close=151.0, volume=1_000_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    fill = fills[0]

    expected_price = 150.0 * (1.0 - 0.0001)  # slippage reduces sell price
    assert fill.fill_price == pytest.approx(expected_price, rel=1e-6)
    assert fill.commission == pytest.approx(50 * 0.005, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Limit buy — fills when bar's low touches the limit
# ---------------------------------------------------------------------------

def test_limit_buy_fills_when_low_touches_limit():
    bus, broker = make_broker()
    order = limit_order("AAPL", "BUY", 100, price=145.0)
    broker.submit_order(order)

    # low=144 dips below limit → should fill at min(open=148, limit=145) = 145
    broker.process_bar("AAPL", open_price=148.0, high=150.0, low=144.0, close=149.0, volume=500_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(145.0, rel=1e-6)


def test_limit_buy_fills_when_low_equals_limit():
    bus, broker = make_broker()
    order = limit_order("AAPL", "BUY", 10, price=145.0)
    broker.submit_order(order)

    # low exactly equals limit price
    broker.process_bar("AAPL", open_price=148.0, high=150.0, low=145.0, close=149.0, volume=500_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(145.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Limit buy NO fill — low stays above limit price
# ---------------------------------------------------------------------------

def test_limit_buy_no_fill_when_low_above_limit():
    bus, broker = make_broker()
    order = limit_order("AAPL", "BUY", 100, price=145.0)
    broker.submit_order(order)

    # low=146 — never reaches $145 limit
    broker.process_bar("AAPL", open_price=148.0, high=150.0, low=146.0, close=149.0, volume=500_000)

    assert collected_fills(bus) == []
    assert len(broker.get_pending_orders()) == 1


# ---------------------------------------------------------------------------
# 5. Stop sell — triggers when bar's low drops to or below stop price
# ---------------------------------------------------------------------------

def test_stop_sell_fills_when_low_hits_stop():
    bus, broker = make_broker()
    order = stop_order("AAPL", "SELL", 200, price=148.0)
    broker.submit_order(order)

    # low=147 drops below stop=$148 → fill at min(open=150, stop=148) with sell slippage
    broker.process_bar("AAPL", open_price=150.0, high=151.0, low=147.0, close=148.5, volume=800_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    fill = fills[0]

    expected_raw = min(150.0, 148.0)   # = 148.0
    expected_price = expected_raw * (1.0 - 0.0001)
    assert fill.fill_price == pytest.approx(expected_price, rel=1e-6)


def test_stop_sell_no_fill_when_low_above_stop():
    bus, broker = make_broker()
    order = stop_order("AAPL", "SELL", 200, price=148.0)
    broker.submit_order(order)

    # low=149 — never reaches stop
    broker.process_bar("AAPL", open_price=150.0, high=151.0, low=149.0, close=150.0, volume=800_000)

    assert collected_fills(bus) == []
    assert len(broker.get_pending_orders()) == 1


# ---------------------------------------------------------------------------
# 6. Stop buy — triggers when bar's high rises to or above stop price
# ---------------------------------------------------------------------------

def test_stop_buy_fills_when_high_hits_stop():
    bus, broker = make_broker()
    order = stop_order("AAPL", "BUY", 100, price=152.0)
    broker.submit_order(order)

    # high=153 exceeds stop=$152 → fill at max(open=150, stop=152) + slippage
    broker.process_bar("AAPL", open_price=150.0, high=153.0, low=149.0, close=152.5, volume=600_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    fill = fills[0]

    expected_raw = max(150.0, 152.0)   # = 152.0
    expected_price = expected_raw * (1.0 + 0.0001)
    assert fill.fill_price == pytest.approx(expected_price, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Cancel — cancelled order produces no fill
# ---------------------------------------------------------------------------

def test_cancel_prevents_fill():
    bus, broker = make_broker()
    order = market_order("AAPL", "BUY", 100)
    broker.submit_order(order)

    # Cancel before the bar is processed.
    cancelled = broker.cancel_order(str(order.event_id))

    assert cancelled is True
    assert broker.get_pending_orders() == []

    broker.process_bar("AAPL", open_price=150.0, high=152.0, low=149.0, close=151.0, volume=1_000_000)
    assert collected_fills(bus) == []


def test_cancel_nonexistent_order_returns_false():
    _, broker = make_broker()
    import uuid
    assert broker.cancel_order(str(uuid.uuid4())) is False


# ---------------------------------------------------------------------------
# 8. Multiple orders for different symbols — only matching symbol fills
# ---------------------------------------------------------------------------

def test_multiple_orders_different_symbols():
    bus, broker = make_broker()

    order_aapl = market_order("AAPL", "BUY", 100)
    order_msft = market_order("MSFT", "BUY", 200)
    order_tsla = market_order("TSLA", "SELL", 50)

    for o in (order_aapl, order_msft, order_tsla):
        broker.submit_order(o)

    # Process only AAPL bar — only AAPL order should fill.
    broker.process_bar("AAPL", open_price=150.0, high=152.0, low=149.0, close=151.0, volume=1_000_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    assert fills[0].symbol == "AAPL"
    assert len(broker.get_pending_orders()) == 2  # MSFT and TSLA still pending

    # Now process MSFT.
    broker.process_bar("MSFT", open_price=300.0, high=305.0, low=298.0, close=302.0, volume=500_000)
    fills = collected_fills(bus)
    assert len(fills) == 2
    assert fills[1].symbol == "MSFT"
    assert len(broker.get_pending_orders()) == 1  # Only TSLA remains

    # Now process TSLA.
    broker.process_bar("TSLA", open_price=250.0, high=255.0, low=248.0, close=252.0, volume=700_000)
    fills = collected_fills(bus)
    assert len(fills) == 3
    assert fills[2].symbol == "TSLA"
    assert broker.get_pending_orders() == []


# ---------------------------------------------------------------------------
# 9. Limit sell — fills when bar's high reaches or exceeds limit
# ---------------------------------------------------------------------------

def test_limit_sell_fills_when_high_reaches_limit():
    bus, broker = make_broker()
    order = limit_order("AAPL", "SELL", 75, price=153.0)
    broker.submit_order(order)

    # high=154 exceeds limit=$153 → fill at max(open=150, limit=153) = 153
    broker.process_bar("AAPL", open_price=150.0, high=154.0, low=149.0, close=153.0, volume=600_000)

    fills = collected_fills(bus)
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(153.0, rel=1e-6)


def test_limit_sell_no_fill_when_high_below_limit():
    bus, broker = make_broker()
    order = limit_order("AAPL", "SELL", 75, price=153.0)
    broker.submit_order(order)

    # high=152 — never reaches $153 limit
    broker.process_bar("AAPL", open_price=150.0, high=152.0, low=149.0, close=151.0, volume=600_000)

    assert collected_fills(bus) == []
    assert len(broker.get_pending_orders()) == 1


# ---------------------------------------------------------------------------
# 10. FillEvent is emitted on the bus (subscription test)
# ---------------------------------------------------------------------------

def test_fill_event_emitted_to_subscribers():
    bus, broker = make_broker()
    received: list[FillEvent] = []
    bus.subscribe(EventType.FILL, lambda e: received.append(e))

    order = market_order("AAPL", "BUY", 10)
    broker.submit_order(order)
    broker.process_bar("AAPL", open_price=100.0, high=101.0, low=99.0, close=100.5, volume=100_000)

    assert len(received) == 1
    assert isinstance(received[0], FillEvent)
