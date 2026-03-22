"""Synchronous pub/sub event bus for the trading bot backtesting framework.

Components subscribe to specific EventTypes (or all events) and receive
callbacks when events are emitted.  History is kept per EventType and can
be queried or cleared.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, List

from core.events import BaseEvent, EventType

Handler = Callable[[BaseEvent], None]


class EventBus:
    """Central message broker.

    Usage
    -----
    bus = EventBus()
    bus.subscribe(EventType.BAR, my_strategy.on_bar)
    bus.emit(bar_event)
    """

    def __init__(self) -> None:
        # type-specific subscribers: EventType -> list of handlers
        self._subscribers: dict[EventType, list[Handler]] = defaultdict(list)
        # catch-all subscribers that receive every event
        self._all_subscribers: list[Handler] = []
        # history per EventType
        self._history: dict[EventType, list[BaseEvent]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register *handler* to be called for events of *event_type*."""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove *handler* from *event_type* subscribers.

        Silently does nothing if the handler is not registered.
        """
        try:
            self._subscribers[event_type].remove(handler)
        except ValueError:
            pass

    def subscribe_all(self, handler: Handler) -> None:
        """Register *handler* to receive every event regardless of type."""
        self._all_subscribers.append(handler)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: BaseEvent) -> None:
        """Deliver *event* to all relevant subscribers and record in history.

        Delivery order:
          1. Type-specific subscribers (in subscription order)
          2. All-subscribers (in subscription order)
        """
        # Record history first so handlers can query it if needed
        self._history[event.event_type].append(event)

        # Deliver to type-specific subscribers
        for handler in list(self._subscribers[event.event_type]):
            handler(event)

        # Deliver to catch-all subscribers
        for handler in list(self._all_subscribers):
            handler(event)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, event_type: EventType) -> list[BaseEvent]:
        """Return a *copy* of the history list for *event_type*."""
        return list(self._history[event_type])

    def clear_history(self) -> None:
        """Discard all recorded history for every event type."""
        self._history.clear()
