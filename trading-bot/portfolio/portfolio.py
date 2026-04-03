"""Portfolio tracker: manages holdings, cash, P&L, and equity curve.

Processes FillEvents to maintain real-time position state, tracks the
high-water mark for drawdown calculation, and emits PortfolioUpdateEvents
after every fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from core.event_bus import EventBus
from core.events import EventType, FillEvent, PortfolioUpdateEvent


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents an open position in a single symbol."""

    symbol: str
    quantity: int        # positive = long, negative = short
    avg_cost: float
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Gross market value using avg_cost as a proxy when no live price."""
        return abs(self.quantity) * self.avg_cost

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Recalculate unrealized P&L given a current market price."""
        self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

_BUY_ACTIONS = {"BUY", "BTO", "BTC"}   # actions that increase / open long
_SELL_ACTIONS = {"SELL", "STC", "STO"}  # actions that reduce / close long


class Portfolio:
    """Tracks cash, positions, equity curve, and high-water-mark drawdown."""

    def __init__(
        self,
        initial_capital: float,
        event_bus: EventBus,
        contract_multipliers: dict[str, float] | None = None,
    ) -> None:
        self.cash: float = initial_capital
        self.initial_capital: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.event_bus: EventBus = event_bus
        self.high_water_mark: float = initial_capital
        self.equity_curve: list[tuple[date, float]] = []

        # Optional futures multipliers: symbol -> point multiplier.
        # When provided, buy/sell cash flows are replaced with P&L-only
        # accounting and get_equity uses multiplier-based valuation.
        self.contract_multipliers: dict[str, float] = contract_multipliers or {}

        # Running totals kept here so PortfolioUpdateEvent can include them
        self._realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def update_on_fill(self, fill: FillEvent, current_date: date) -> None:
        """Process a FillEvent: update cash, positions, and emit an update.

        BUY / BTO / BTC — opens or increases a long position.
        SELL / STC / STO — reduces or closes a long position, realises P&L.
        """
        total_cost = fill.fill_price * fill.quantity + fill.commission

        if fill.action in _BUY_ACTIONS:
            self._process_buy(fill, total_cost)
        elif fill.action in _SELL_ACTIONS:
            self._process_sell(fill)
        else:
            raise ValueError(f"Unknown fill action: {fill.action!r}")

        # Keep high-water mark current (use avg_cost as best available price)
        current_prices = {
            sym: pos.avg_cost for sym, pos in self.positions.items()
        }
        equity = self.get_equity(current_prices)
        if equity > self.high_water_mark:
            self.high_water_mark = equity

        # Emit portfolio state
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        event = PortfolioUpdateEvent(
            equity=equity,
            cash=self.cash,
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
            drawdown_pct=self.get_drawdown(),
        )
        self.event_bus.emit(event)

    def _process_buy(self, fill: FillEvent, total_cost: float) -> None:
        sym = fill.symbol
        if sym in self.contract_multipliers:
            # Futures: no notional cash outlay — only deduct commission.
            self.cash -= fill.commission
        else:
            self.cash -= total_cost

        if sym in self.positions:
            pos = self.positions[sym]
            # Weighted-average cost
            total_qty = pos.quantity + fill.quantity
            if total_qty == 0:
                return
            pos.avg_cost = (
                (pos.avg_cost * pos.quantity + fill.fill_price * fill.quantity)
                / total_qty
            )
            pos.quantity = total_qty
        else:
            self.positions[sym] = Position(
                symbol=sym,
                quantity=fill.quantity,
                avg_cost=fill.fill_price,
            )

    def _process_sell(self, fill: FillEvent) -> None:
        sym = fill.symbol
        multiplier = self.contract_multipliers.get(sym)

        if multiplier is not None:
            # Futures: credit realised P&L (multiplied) minus commission.
            if sym in self.positions:
                pos = self.positions[sym]
                realized = (
                    (fill.fill_price - pos.avg_cost)
                    * fill.quantity
                    * multiplier
                    - fill.commission
                )
                self._realized_pnl += realized
                self.cash += realized
                pos.quantity -= fill.quantity
                if pos.quantity <= 0:
                    del self.positions[sym]
            else:
                # Short sale — create negative-quantity position; deduct commission only.
                self.cash -= fill.commission
                self.positions[sym] = Position(
                    symbol=sym,
                    quantity=-fill.quantity,
                    avg_cost=fill.fill_price,
                )
        else:
            # Equity path (unchanged).
            proceeds = fill.fill_price * fill.quantity - fill.commission
            self.cash += proceeds

            if sym in self.positions:
                pos = self.positions[sym]
                realized = (fill.fill_price - pos.avg_cost) * fill.quantity - fill.commission
                self._realized_pnl += realized
                pos.quantity -= fill.quantity
                if pos.quantity <= 0:
                    del self.positions[sym]
            else:
                # Short sale — create negative-quantity position
                self.positions[sym] = Position(
                    symbol=sym,
                    quantity=-fill.quantity,
                    avg_cost=fill.fill_price,
                )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_equity(self, current_prices: dict[str, float]) -> float:
        """Total equity = cash + sum of all position values at current prices.

        For futures symbols (those with a contract multiplier), position
        value is the unrealised P&L:
        ``(price - avg_cost) * quantity * multiplier``.

        For equity symbols the original formula is used:
        ``price * quantity``.
        """
        position_value = 0.0
        for sym, pos in self.positions.items():
            price = current_prices.get(sym, pos.avg_cost)
            multiplier = self.contract_multipliers.get(sym)
            if multiplier is not None:
                # Futures: unrealised P&L
                position_value += (price - pos.avg_cost) * pos.quantity * multiplier
            else:
                # Equity: full market value
                position_value += price * pos.quantity
        return self.cash + position_value

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_positions(self) -> dict[str, Position]:
        return dict(self.positions)

    def get_drawdown(self) -> float:
        """Current drawdown from high-water mark as a negative fraction.

        Returns 0.0 when equity equals the high-water mark.
        """
        if self.high_water_mark == 0.0:
            return 0.0
        current_prices = {
            sym: pos.avg_cost for sym, pos in self.positions.items()
        }
        equity = self.get_equity(current_prices)
        drawdown = (equity - self.high_water_mark) / self.high_water_mark
        return min(drawdown, 0.0)  # never positive

    def record_equity(
        self,
        current_date: date,
        current_prices: dict[str, float],
    ) -> None:
        """Append a (date, equity) snapshot to the equity curve."""
        equity = self.get_equity(current_prices)
        self.equity_curve.append((current_date, equity))
        # Maintain high-water mark
        if equity > self.high_water_mark:
            self.high_water_mark = equity
