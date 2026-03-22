"""Risk manager: evaluates signals against a list of rules and converts
approved signals into OrderEvents.

If any rule blocks the signal a RiskEvent is emitted on the bus and None is
returned.  If all rules pass an OrderEvent is constructed from the signal and
returned (but NOT emitted — that is the caller's responsibility).
"""

from __future__ import annotations

from core.event_bus import EventBus
from core.events import OrderEvent, RiskEvent, SignalEvent
from portfolio.portfolio import Portfolio
from risk.rules import RiskRule


class RiskManager:
    """Evaluates signals against a list of :class:`RiskRule` instances.

    Parameters
    ----------
    rules:
        Ordered list of rules.  Evaluation stops at the first failure.
    event_bus:
        Bus used to emit :class:`RiskEvent` when a signal is blocked.
    sector_map:
        Optional symbol → sector mapping forwarded to rules via context.
    """

    def __init__(
        self,
        rules: list[RiskRule],
        event_bus: EventBus,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self.rules = rules
        self.event_bus = event_bus
        self.sector_map: dict[str, str] = sector_map or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_signal(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> OrderEvent | None:
        """Check *signal* against all rules.

        Returns an :class:`OrderEvent` when all rules pass, or ``None`` when
        at least one rule blocks the signal (a :class:`RiskEvent` is emitted).
        """
        for rule in self.rules:
            passes, reason = rule.check(signal, portfolio, context)
            if not passes:
                self.event_bus.emit(
                    RiskEvent(
                        rule_name=type(rule).__name__,
                        signal_blocked_id=str(signal.event_id),
                        reason=reason,
                    )
                )
                return None

        # All rules passed — build the order
        action = self._signal_to_action(signal)
        quantity = self._compute_quantity(signal, portfolio, context)

        return OrderEvent(
            symbol=signal.symbol,
            action=action,
            order_type="market",
            quantity=quantity,
            price=None,
            option_type=signal.option_type,
            strike=signal.strike,
            expiration=signal.expiration,
            signal_ref=str(signal.event_id),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _signal_to_action(self, signal: SignalEvent) -> str:
        """Map signal *direction* to an order action string."""
        mapping: dict[str, str] = {
            "long": "BUY",
            "short": "SELL",
            "sell_put": "STO",
            "sell_call": "STO",
            "buy_to_close": "BTC",
            "sell_to_close": "STC",
        }
        return mapping.get(signal.direction, "BUY")

    def _compute_quantity(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> int:
        """Compute share count based on *position_size_pct* of equity."""
        equity = context.get("equity", portfolio.initial_capital)
        price = context.get("current_prices", {}).get(signal.symbol, 0)
        position_size_pct = context.get("position_size_pct", 0.06)

        if price <= 0:
            return 0

        max_value = equity * position_size_pct
        quantity = int(max_value / price)
        return max(quantity, 0)
