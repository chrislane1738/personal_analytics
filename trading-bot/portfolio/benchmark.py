"""Benchmark tracker: buy-and-hold reference for performance comparison.

The first price fed to BenchmarkTracker establishes a notional 'purchase'
of as many shares as the initial capital can buy.  Subsequent prices scale
that share count to produce an equity curve directly comparable to the
strategy's equity curve.
"""

from __future__ import annotations

from datetime import date
from typing import Optional


class BenchmarkTracker:
    """Tracks a buy-and-hold benchmark (default: SPY)."""

    def __init__(
        self,
        symbol: str = "SPY",
        initial_capital: float = 100_000.0,
    ) -> None:
        self.symbol: str = symbol
        self.initial_capital: float = initial_capital
        self.equity_curve: list[tuple[date, float]] = []
        self._start_price: Optional[float] = None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, bar_date: date, price: float) -> None:
        """Record benchmark equity for *bar_date* at *price*.

        The first call sets the start price (i.e., the 'buy' price).
        All subsequent calls scale the fixed share count by the new price.
        """
        if self._start_price is None:
            self._start_price = price

        shares = self.initial_capital / self._start_price
        equity = shares * price
        self.equity_curve.append((bar_date, equity))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_returns(self) -> list[float]:
        """Daily percentage returns as a list of floats.

        The first entry is 0.0 (no prior day to compare against).
        Subsequent entries are (today - yesterday) / yesterday.
        """
        if not self.equity_curve:
            return []

        returns: list[float] = [0.0]  # first day has no prior reference
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1][1]
            curr = self.equity_curve[i][1]
            daily_return = (curr - prev) / prev if prev != 0.0 else 0.0
            returns.append(daily_return)

        return returns
