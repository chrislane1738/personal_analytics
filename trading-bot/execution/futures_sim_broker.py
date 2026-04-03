"""Futures-aware simulated broker for backtesting.

Extends :class:`SimBroker` with tick-size rounding and per-contract
commission logic driven by a :class:`FuturesContractSpec`.
"""

from __future__ import annotations

from core.event_bus import EventBus
from core.events import FillEvent, OrderEvent
from execution.sim_broker import SimBroker
from topstep.futures_specs import FuturesContractSpec


class FuturesSimBroker(SimBroker):
    """Paper-trading broker for futures contracts.

    Inherits all fill logic from :class:`SimBroker` (market, limit, stop,
    stop-limit) and adds:

    * **Tick rounding** -- every computed fill price is rounded to the
      nearest valid tick for the contract.
    * **Per-contract commission** -- commission is ``quantity * spec.commission``
      per fill (one side), replacing the per-share model.

    Parameters
    ----------
    event_bus:
        Shared event bus for publishing :class:`FillEvent` instances.
    contract_spec:
        The :class:`FuturesContractSpec` governing this broker's instrument.
    slippage_pct:
        Fractional slippage (same semantics as :class:`SimBroker`).
    """

    def __init__(
        self,
        event_bus: EventBus,
        contract_spec: FuturesContractSpec,
        slippage_pct: float = 0.0001,
    ) -> None:
        # Initialise the parent with commission_per_share=0 because we
        # override commission calculation entirely.
        super().__init__(
            event_bus=event_bus,
            slippage_pct=slippage_pct,
            commission_per_share=0.0,
        )
        self.contract_spec = contract_spec

    # ------------------------------------------------------------------
    # Tick rounding
    # ------------------------------------------------------------------

    @staticmethod
    def round_to_tick(price: float, tick_size: float) -> float:
        """Round *price* to the nearest multiple of *tick_size*."""
        if tick_size <= 0:
            return price
        return round(round(price / tick_size) * tick_size, 10)

    # ------------------------------------------------------------------
    # Overridden fill pipeline
    # ------------------------------------------------------------------

    def process_bar(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Fill pending orders with tick-rounded prices and per-contract commission."""
        tick = self.contract_spec.tick_size
        filled: list[OrderEvent] = []

        for order in list(self.pending_orders):
            if order.symbol != symbol:
                continue

            fill_price = self._compute_fill_price(order, open_price, high, low)
            if fill_price is None:
                continue

            # Round to nearest valid tick
            fill_price = self.round_to_tick(fill_price, tick)

            # Per-contract commission (one side)
            commission = order.quantity * self.contract_spec.commission

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
