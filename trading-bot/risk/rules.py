"""Risk rules for the trading bot framework.

Each rule implements the RiskRule ABC and exposes a single ``check`` method
that returns ``(passes: bool, reason: str)``.  When ``passes`` is ``True`` the
signal is allowed through; when ``False`` it is blocked and ``reason``
describes why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.events import SignalEvent
from portfolio.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RiskRule(ABC):
    """Abstract base class for all risk rules."""

    @abstractmethod
    def check(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> tuple[bool, str]:
        """Evaluate a signal against this rule.

        Parameters
        ----------
        signal:
            The incoming trade signal to evaluate.
        portfolio:
            Current portfolio state.
        context:
            Arbitrary key/value bag supplied by the caller.  Common keys:

            - ``equity``          – current portfolio equity (float)
            - ``current_prices``  – dict[symbol, price]
            - ``default_quantity``– default share count to use in estimates
            - ``position_size_pct``– fraction of equity per position

        Returns
        -------
        (passes, reason)
            ``passes=True`` means the signal is allowed.
            ``passes=False`` means the signal is blocked; ``reason`` explains why.
        """


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------


class MaxPositionSizeRule(RiskRule):
    """Block signals whose estimated order value exceeds *max_pct* of equity."""

    def __init__(self, max_pct: float = 0.06) -> None:
        self.max_pct = max_pct

    def check(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> tuple[bool, str]:
        equity = context.get("equity", portfolio.initial_capital)
        max_value = equity * self.max_pct

        price = context.get("current_prices", {}).get(signal.symbol, 0)
        if price <= 0:
            # Cannot estimate value — allow through
            return True, ""

        quantity = context.get("default_quantity", 0)
        position_value = price * quantity

        if position_value > max_value:
            return (
                False,
                f"Position value ${position_value:.0f} exceeds "
                f"{self.max_pct * 100:.0f}% limit (${max_value:.0f})",
            )
        return True, ""


class MaxOpenPositionsRule(RiskRule):
    """Block new signals when the portfolio already holds *max_positions*."""

    def __init__(self, max_positions: int = 20) -> None:
        self.max_positions = max_positions

    def check(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> tuple[bool, str]:
        current = len(portfolio.positions)
        if current >= self.max_positions:
            return (
                False,
                f"Already at max {self.max_positions} open positions",
            )
        return True, ""


class MaxSectorConcentrationRule(RiskRule):
    """Block signals that would push a sector beyond *max_pct* of equity."""

    def __init__(
        self,
        max_pct: float = 0.25,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self.max_pct = max_pct
        self.sector_map: dict[str, str] = sector_map or {}

    def check(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> tuple[bool, str]:
        sector = self.sector_map.get(signal.symbol, "Unknown")
        equity = context.get("equity", portfolio.initial_capital)
        if equity <= 0:
            return True, ""

        current_prices = context.get("current_prices", {})

        # Sum current exposure in the same sector
        sector_value = 0.0
        for sym, pos in portfolio.positions.items():
            if self.sector_map.get(sym, "Unknown") == sector:
                price = current_prices.get(sym, pos.avg_cost)
                sector_value += abs(pos.quantity) * price

        # Add the proposed position's estimated value
        price = current_prices.get(signal.symbol, 0)
        quantity = context.get("default_quantity", 0)
        proposed = price * quantity

        if (sector_value + proposed) / equity > self.max_pct:
            return (
                False,
                f"Sector '{sector}' would exceed {self.max_pct * 100:.0f}% concentration",
            )
        return True, ""


class DrawdownCircuitBreakerRule(RiskRule):
    """Block all signals when drawdown is at or below *max_drawdown*."""

    def __init__(self, max_drawdown: float = -0.15) -> None:
        self.max_drawdown = max_drawdown

    def check(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> tuple[bool, str]:
        dd = portfolio.get_drawdown()
        if dd <= self.max_drawdown:
            return (
                False,
                f"Circuit breaker: drawdown {dd * 100:.1f}% exceeds limit "
                f"{self.max_drawdown * 100:.1f}%",
            )
        return True, ""


class MaxOptionsNotionalRule(RiskRule):
    """Block options signals that would push total options notional above *max_pct* of equity.

    Currently a lightweight placeholder: it passes non-options trades immediately
    and approves options trades.  A full implementation would track options
    notional across the portfolio.
    """

    def __init__(self, max_pct: float = 0.30) -> None:
        self.max_pct = max_pct

    def check(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        context: dict,
    ) -> tuple[bool, str]:
        if signal.option_type is None:
            return True, ""  # Not an options trade — always pass
        # Placeholder: options notional tracking not yet implemented
        return True, ""
