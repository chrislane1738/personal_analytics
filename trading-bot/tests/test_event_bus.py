"""Tests for core/event_bus.py — pub/sub event bus."""

from datetime import date

import pytest

from core.event_bus import EventBus
from core.events import (
    BarEvent,
    EventType,
    FillEvent,
    PortfolioUpdateEvent,
    SignalEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bar(symbol="AAPL") -> BarEvent:
    return BarEvent(
        symbol=symbol,
        date=date(2024, 1, 15),
        open=180.0,
        high=185.0,
        low=179.5,
        close=184.0,
        adj_close=184.0,
        volume=50_000_000,
        vwap=182.5,
    )


def make_signal(symbol="AAPL") -> SignalEvent:
    return SignalEvent(
        symbol=symbol,
        direction="long",
        reason="test",
        strength=0.8,
    )


def make_portfolio_update() -> PortfolioUpdateEvent:
    return PortfolioUpdateEvent(
        equity=100_000.0,
        cash=50_000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        drawdown_pct=0.0,
    )


# ---------------------------------------------------------------------------
# Basic subscribe + emit
# ---------------------------------------------------------------------------

class TestSubscribeAndEmit:
    def test_handler_called_on_matching_event(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.BAR, lambda e: received.append(e))
        evt = make_bar()
        bus.emit(evt)
        assert len(received) == 1
        assert received[0] is evt

    def test_handler_not_called_for_different_event_type(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.SIGNAL, lambda e: received.append(e))
        bus.emit(make_bar())
        assert len(received) == 0

    def test_multiple_subscribers_both_receive(self):
        bus = EventBus()
        received_a = []
        received_b = []
        bus.subscribe(EventType.BAR, lambda e: received_a.append(e))
        bus.subscribe(EventType.BAR, lambda e: received_b.append(e))
        bus.emit(make_bar())
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_emit_multiple_events_in_order(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.BAR, lambda e: received.append(e))
        evt1 = make_bar("AAPL")
        evt2 = make_bar("MSFT")
        bus.emit(evt1)
        bus.emit(evt2)
        assert received[0].symbol == "AAPL"
        assert received[1].symbol == "MSFT"


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------

class TestUnsubscribe:
    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        received = []

        def handler(e):
            received.append(e)

        bus.subscribe(EventType.BAR, handler)
        bus.emit(make_bar())
        assert len(received) == 1

        bus.unsubscribe(EventType.BAR, handler)
        bus.emit(make_bar())
        assert len(received) == 1  # no new deliveries

    def test_unsubscribe_nonexistent_handler_does_not_raise(self):
        bus = EventBus()
        # Should not raise
        bus.unsubscribe(EventType.BAR, lambda e: None)

    def test_unsubscribe_only_removes_specified_handler(self):
        bus = EventBus()
        received_a = []
        received_b = []

        def handler_a(e):
            received_a.append(e)

        def handler_b(e):
            received_b.append(e)

        bus.subscribe(EventType.BAR, handler_a)
        bus.subscribe(EventType.BAR, handler_b)
        bus.unsubscribe(EventType.BAR, handler_a)
        bus.emit(make_bar())
        assert len(received_a) == 0
        assert len(received_b) == 1


# ---------------------------------------------------------------------------
# subscribe_all
# ---------------------------------------------------------------------------

class TestSubscribeAll:
    def test_subscribe_all_receives_all_event_types(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: received.append(e))
        bus.emit(make_bar())
        bus.emit(make_signal())
        bus.emit(make_portfolio_update())
        assert len(received) == 3

    def test_subscribe_all_and_type_specific_both_called(self):
        bus = EventBus()
        all_received = []
        bar_received = []
        bus.subscribe_all(lambda e: all_received.append(e))
        bus.subscribe(EventType.BAR, lambda e: bar_received.append(e))
        bus.emit(make_bar())
        assert len(all_received) == 1
        assert len(bar_received) == 1

    def test_subscribe_all_signal_only_arrives_once_per_emit(self):
        """subscribe_all should deliver each event exactly once."""
        bus = EventBus()
        count = []
        bus.subscribe_all(lambda e: count.append(1))
        bus.subscribe_all(lambda e: count.append(1))
        bus.emit(make_bar())
        # Two all-subscribers → 2 calls
        assert len(count) == 2


# ---------------------------------------------------------------------------
# Event history
# ---------------------------------------------------------------------------

class TestEventHistory:
    def test_history_records_emitted_events(self):
        bus = EventBus()
        evt = make_bar()
        bus.emit(evt)
        history = bus.get_history(EventType.BAR)
        assert evt in history

    def test_history_is_per_event_type(self):
        bus = EventBus()
        bus.emit(make_bar())
        bus.emit(make_signal())
        assert len(bus.get_history(EventType.BAR)) == 1
        assert len(bus.get_history(EventType.SIGNAL)) == 1

    def test_history_accumulates_multiple_emits(self):
        bus = EventBus()
        bus.emit(make_bar("AAPL"))
        bus.emit(make_bar("MSFT"))
        bus.emit(make_bar("GOOG"))
        assert len(bus.get_history(EventType.BAR)) == 3

    def test_get_history_empty_for_unemitted_type(self):
        bus = EventBus()
        assert bus.get_history(EventType.FILL) == []

    def test_clear_history_removes_all(self):
        bus = EventBus()
        bus.emit(make_bar())
        bus.emit(make_signal())
        bus.clear_history()
        assert bus.get_history(EventType.BAR) == []
        assert bus.get_history(EventType.SIGNAL) == []

    def test_clear_history_does_not_affect_subscribers(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.BAR, lambda e: received.append(e))
        bus.emit(make_bar())
        bus.clear_history()
        bus.emit(make_bar())
        # subscriber still active after clear
        assert len(received) == 2

    def test_get_history_returns_copy_or_safe_reference(self):
        """Mutating returned history should not corrupt internal state."""
        bus = EventBus()
        bus.emit(make_bar())
        h = bus.get_history(EventType.BAR)
        original_len = len(h)
        h.clear()  # mutate the returned list
        # Internal history should still have the event
        assert len(bus.get_history(EventType.BAR)) == original_len


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_emit_with_no_subscribers_does_not_raise(self):
        bus = EventBus()
        bus.emit(make_bar())  # should not raise

    def test_fresh_bus_has_empty_history(self):
        bus = EventBus()
        for et in EventType:
            assert bus.get_history(et) == []

    def test_multiple_buses_are_independent(self):
        bus1 = EventBus()
        bus2 = EventBus()
        received1 = []
        received2 = []
        bus1.subscribe(EventType.BAR, lambda e: received1.append(e))
        bus2.subscribe(EventType.BAR, lambda e: received2.append(e))
        bus1.emit(make_bar())
        assert len(received1) == 1
        assert len(received2) == 0
