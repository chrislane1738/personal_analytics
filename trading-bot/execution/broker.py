"""Abstract base class defining the broker interface."""

from abc import ABC, abstractmethod

from core.events import OrderEvent


class Broker(ABC):
    """Abstract broker interface.

    All broker implementations — simulated or live — must satisfy this
    contract so that strategies and the backtesting engine are decoupled
    from the specific execution venue.
    """

    @abstractmethod
    def submit_order(self, order: OrderEvent) -> None:
        """Submit an order for execution."""

    @abstractmethod
    def process_bar(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Process a new bar — check if any pending orders should fill."""

    @abstractmethod
    def get_pending_orders(self) -> list[OrderEvent]:
        """Return unfilled orders."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by its event_id string.

        Returns True if the order was found and removed, False otherwise.
        """
