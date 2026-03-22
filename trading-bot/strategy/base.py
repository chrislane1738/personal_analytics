"""Base class for all trading strategies.

Every strategy must subclass Strategy and implement generate_signals().
Optional lifecycle hooks: on_start, on_end, on_fill, on_universe_bar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Optional

from core.events import BarEvent, FillEvent, SignalEvent

if TYPE_CHECKING:
    pass  # Portfolio imported as duck type to avoid circular deps


# ---------------------------------------------------------------------------
# BacktestContext
# ---------------------------------------------------------------------------

@dataclass
class BacktestContext:
    start_date: date
    end_date: date
    universe: list[str]
    initial_capital: float
    benchmark: str = "SPY"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Base class for all strategies."""

    def __init__(self):
        self.indicators: dict = {}  # name -> Indicator instance

    @abstractmethod
    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        """Given a bar for a single symbol and portfolio state, return signals."""

    def on_universe_bar(self, bars: dict[str, BarEvent], portfolio) -> list[SignalEvent]:
        """Called once per date with ALL symbols' bars. Default delegates per-symbol."""
        signals = []
        for symbol, bar in bars.items():
            signals.extend(self.generate_signals(bar, portfolio))
        return signals

    def on_fill(self, fill: FillEvent) -> None:
        """Optional — react to fills."""
        pass

    def on_start(self, context: BacktestContext) -> None:
        """Optional — initialize before backtest."""
        pass

    def on_end(self, portfolio) -> list[SignalEvent]:
        """Optional — emit closing signals at end."""
        return []

    def warm_up_period(self) -> int:
        """Bars needed before signals can be generated."""
        return 0
