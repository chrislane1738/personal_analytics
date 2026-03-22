"""Abstract base class for market data feeds."""
from abc import ABC, abstractmethod
from datetime import date

from data.storage.models import DailyBar


class DataFeed(ABC):
    """Interface that every concrete data feed must implement.

    All methods are synchronous for v1 of the framework.  Async support can
    be layered on top later without changing the interface contract.
    """

    @abstractmethod
    def get_daily_bars(
        self, symbol: str, start: date, end: date
    ) -> list[DailyBar]:
        """Fetch daily OHLCV bars for *symbol* between *start* and *end*.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``'AAPL'``).
        start:
            First date of the range, inclusive.
        end:
            Last date of the range, inclusive.

        Returns
        -------
        list[DailyBar]
            Bars sorted by date ascending.  May be empty if no data is
            available for the requested range.
        """

    @abstractmethod
    def get_symbols(self, universe: str) -> list[str]:
        """Return the list of ticker symbols belonging to *universe*.

        Parameters
        ----------
        universe:
            Named universe identifier, e.g. ``'sp500'``, ``'nasdaq100'``.

        Returns
        -------
        list[str]
            Ticker symbols as uppercase strings.
        """
