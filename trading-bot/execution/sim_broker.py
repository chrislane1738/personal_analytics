"""Simulated broker for backtesting.

Fills orders against OHLCV bars with configurable slippage and
per-share commission.  Every fill is published as a FillEvent on the
EventBus so the portfolio and any analytics subscribers are notified
immediately.
"""

from __future__ import annotations

from core.event_bus import EventBus
from core.events import FillEvent, OrderEvent
from execution.broker import Broker

# Buy-side actions — slippage increases the fill price.
_BUY_ACTIONS = {"BUY", "BTO", "BTC"}
# Sell-side actions — slippage decreases the fill price.
_SELL_ACTIONS = {"SELL", "STO", "STC"}


class SimBroker(Broker):
    """Paper-trading broker that matches orders against historical bars.

    Parameters
    ----------
    event_bus:
        The shared EventBus on which FillEvents will be published.
    slippage_pct:
        Fractional slippage applied to every fill (default 0.01 %, i.e.
        0.0001).  Buys pay more; sells receive less.
    commission_per_share:
        Flat per-share (or per-contract) commission charged on every fill
        (default $0.005).
    """

    def __init__(
        self,
        event_bus: EventBus,
        slippage_pct: float = 0.0001,
        commission_per_share: float = 0.005,
    ) -> None:
        self.event_bus = event_bus
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share
        self.pending_orders: list[OrderEvent] = []

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    def submit_order(self, order: OrderEvent) -> None:
        """Enqueue *order* for processing on the next bar."""
        self.pending_orders.append(order)

    def get_pending_orders(self) -> list[OrderEvent]:
        """Return a copy of the current pending-order queue."""
        return list(self.pending_orders)

    def cancel_order(self, order_id: str) -> bool:
        """Remove the order whose ``event_id`` matches *order_id*.

        Returns True if found and removed, False if not found.
        """
        for i, order in enumerate(self.pending_orders):
            if str(order.event_id) == order_id:
                self.pending_orders.pop(i)
                return True
        return False

    def process_bar(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Attempt to fill every pending order whose symbol matches *symbol*.

        Fill logic
        ----------
        MARKET
            Always fills; execution price = open_price adjusted for
            slippage.

        LIMIT buy
            Fills if ``low <= limit_price``.  Execution price =
            ``min(open_price, limit_price)`` (no additional slippage —
            limit orders guarantee the named price or better).

        LIMIT sell
            Fills if ``high >= limit_price``.  Execution price =
            ``max(open_price, limit_price)``.

        STOP buy
            Fills if ``high >= stop_price``.  Execution price =
            ``max(open_price, stop_price)`` + slippage.

        STOP sell
            Fills if ``low <= stop_price``.  Execution price =
            ``min(open_price, stop_price)`` − slippage.
        """
        filled: list[OrderEvent] = []

        for order in list(self.pending_orders):
            if order.symbol != symbol:
                continue

            fill_price = self._compute_fill_price(
                order, open_price, high, low
            )
            if fill_price is None:
                # Conditions not met — order remains pending.
                continue

            commission = order.quantity * self.commission_per_share
            slippage_amount = self._slippage_amount(order, fill_price)

            fill = FillEvent(
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage_amount,
                order_ref=str(order.event_id),
            )
            self.event_bus.emit(fill)
            filled.append(order)

        for order in filled:
            self.pending_orders.remove(order)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_buy_side(self, order: OrderEvent) -> bool:
        return order.action in _BUY_ACTIONS

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Return *price* adjusted for slippage."""
        if is_buy:
            return price * (1.0 + self.slippage_pct)
        return price * (1.0 - self.slippage_pct)

    def _slippage_amount(self, order: OrderEvent, fill_price: float) -> float:
        """Signed slippage dollar amount (positive = cost to trader)."""
        # Back-calculate the raw price before slippage was applied.
        is_buy = self._is_buy_side(order)
        if is_buy:
            raw = fill_price / (1.0 + self.slippage_pct)
            return fill_price - raw
        else:
            raw = fill_price / (1.0 - self.slippage_pct)
            return raw - fill_price

    def _compute_fill_price(
        self,
        order: OrderEvent,
        open_price: float,
        high: float,
        low: float,
    ) -> float | None:
        """Return the fill price for *order* given the bar OHLC, or None.

        Returns None when the order's conditions are not satisfied.
        """
        otype = order.order_type.lower()
        is_buy = self._is_buy_side(order)
        limit_price = order.price  # used for LIMIT and STOP_LIMIT
        stop_price = order.price   # used for STOP

        if otype == "market":
            return self._apply_slippage(open_price, is_buy)

        if otype == "limit":
            if is_buy:
                # Fill if price dipped to or below our limit.
                if low <= limit_price:
                    return min(open_price, limit_price)
            else:
                # Fill if price rose to or above our limit.
                if high >= limit_price:
                    return max(open_price, limit_price)
            return None

        if otype == "stop":
            if is_buy:
                # Triggered when price rises to or above the stop.
                if high >= stop_price:
                    raw = max(open_price, stop_price)
                    return self._apply_slippage(raw, is_buy)
            else:
                # Triggered when price falls to or below the stop.
                if low <= stop_price:
                    raw = min(open_price, stop_price)
                    return self._apply_slippage(raw, is_buy)
            return None

        # STOP_LIMIT: trigger = stop_price, fill cap = limit_price.
        # order.price is interpreted as the stop; for stop_limit orders
        # that need two prices, callers should put stop in price and
        # limit in a separate field — for now we treat price as stop
        # and the same value as the limit cap.
        if otype == "stop_limit":
            if is_buy:
                if high >= stop_price:
                    raw = max(open_price, stop_price)
                    fill = self._apply_slippage(raw, is_buy)
                    # Honour limit cap.
                    return fill if fill <= limit_price else None
            else:
                if low <= stop_price:
                    raw = min(open_price, stop_price)
                    fill = self._apply_slippage(raw, is_buy)
                    return fill if fill >= limit_price else None
            return None

        # Unknown order type — leave pending.
        return None
