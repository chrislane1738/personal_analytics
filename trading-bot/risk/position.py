"""Position summary dataclass for risk-layer checks.

Provides an aggregated view of a position that the risk rules can inspect
without needing direct access to the portfolio's internal Position objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PositionSummary:
    """Aggregated position info for risk checks."""

    symbol: str
    quantity: int
    market_value: float
    sector: str
    is_option: bool = False
    option_notional: float = 0.0  # notional value for options positions
