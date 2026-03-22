"""Trade log: open/close trade lifecycle with P&L tracking and summary stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# CompletedTrade
# ---------------------------------------------------------------------------

@dataclass
class CompletedTrade:
    """An immutable record of a fully closed trade."""

    symbol: str
    direction: str          # "long" / "short"
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    entry_reason: str
    exit_reason: str
    holding_days: int
    option_type: Optional[str] = None   # "call" / "put" / None
    strike: Optional[float] = None
    expiration: Optional[date] = None


# ---------------------------------------------------------------------------
# TradeLog
# ---------------------------------------------------------------------------

class TradeLog:
    """Manages open and closed trade records with statistics."""

    def __init__(self) -> None:
        self.trades: list[CompletedTrade] = []
        self._open_trades: dict[str, dict] = {}   # symbol -> entry info

    # ------------------------------------------------------------------
    # Entry / exit
    # ------------------------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry_date: date,
        entry_price: float,
        quantity: int,
        entry_reason: str,
        option_type: Optional[str] = None,
        strike: Optional[float] = None,
        expiration: Optional[date] = None,
    ) -> None:
        """Record a new trade entry."""
        self._open_trades[symbol] = {
            "direction": direction,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "quantity": quantity,
            "entry_reason": entry_reason,
            "option_type": option_type,
            "strike": strike,
            "expiration": expiration,
        }

    def close_trade(
        self,
        symbol: str,
        exit_date: date,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[CompletedTrade]:
        """Close an open trade, compute P&L, and add to completed trades.

        Returns the completed trade, or None if no open trade exists for *symbol*.
        """
        if symbol not in self._open_trades:
            return None

        entry = self._open_trades.pop(symbol)

        # Raw P&L = price_diff * quantity; flip sign for shorts
        pnl = (exit_price - entry["entry_price"]) * entry["quantity"]
        if entry["direction"] == "short":
            pnl = -pnl

        cost_basis = entry["entry_price"] * entry["quantity"]
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0.0

        holding_days = (exit_date - entry["entry_date"]).days

        trade = CompletedTrade(
            symbol=symbol,
            direction=entry["direction"],
            entry_date=entry["entry_date"],
            exit_date=exit_date,
            entry_price=entry["entry_price"],
            exit_price=exit_price,
            quantity=entry["quantity"],
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_reason=entry["entry_reason"],
            exit_reason=exit_reason,
            holding_days=holding_days,
            option_type=entry["option_type"],
            strike=entry["strike"],
            expiration=entry["expiration"],
        )
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trades(self, symbol: Optional[str] = None) -> list[CompletedTrade]:
        """Return all completed trades, optionally filtered by *symbol*."""
        if symbol:
            return [t for t in self.trades if t.symbol == symbol]
        return list(self.trades)

    def get_trade_dicts(self) -> list[dict]:
        """Return trades as dicts with keys expected by metrics functions."""
        return [
            {
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "symbol": t.symbol,
                "holding_days": t.holding_days,
                "direction": t.direction,
            }
            for t in self.trades
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return summary statistics across all completed trades."""
        if not self.trades:
            return {"total_trades": 0}

        holding_days = [t.holding_days for t in self.trades]
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        # Consecutive win / loss streaks
        max_win_streak = max_loss_streak = current_streak = 0
        is_winning: Optional[bool] = None

        for t in self.trades:
            if t.pnl > 0:
                if is_winning:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning = True
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if not is_winning and is_winning is not None:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning = False
                max_loss_streak = max(max_loss_streak, current_streak)

        return {
            "total_trades": len(self.trades),
            "winners": len(wins),
            "losers": len(losses),
            "avg_holding_days": sum(holding_days) / len(holding_days),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "largest_win": max(t.pnl for t in self.trades),
            "largest_loss": min(t.pnl for t in self.trades),
        }
