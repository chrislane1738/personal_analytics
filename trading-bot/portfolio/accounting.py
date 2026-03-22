"""Accounting ledger: immutable transaction log with realized P&L tracking.

Every executed trade is recorded as a Transaction.  The ledger is append-only
and supports filtering by symbol and querying aggregate realized P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """An immutable record of a single executed trade."""

    date: date
    symbol: str
    action: str          # BUY / SELL / BTO / STO / BTC / STC
    quantity: int
    price: float
    commission: float
    pnl: float           # realized P&L; 0.0 for opening transactions


# ---------------------------------------------------------------------------
# AccountingLedger
# ---------------------------------------------------------------------------

class AccountingLedger:
    """Append-only ledger that tracks all transactions and cumulative P&L."""

    def __init__(self) -> None:
        self.transactions: list[Transaction] = []
        self.realized_pnl: float = 0.0

    def record_transaction(self, txn: Transaction) -> None:
        """Append *txn* and accumulate its P&L."""
        self.transactions.append(txn)
        self.realized_pnl += txn.pnl

    def get_realized_pnl(self) -> float:
        """Return cumulative realized P&L across all transactions."""
        return self.realized_pnl

    def get_transactions(
        self,
        symbol: Optional[str] = None,
    ) -> list[Transaction]:
        """Return all transactions, optionally filtered to a single symbol."""
        if symbol is None:
            return list(self.transactions)
        return [t for t in self.transactions if t.symbol == symbol]
