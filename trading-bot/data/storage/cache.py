"""Smart caching layer that sits between DataFeed and the SQLite database.

The cache follows this logic for ``get_daily_bars(symbol, start, end)``:

1. Query the DB for the min/max cached date range of *symbol*.
2. If the full requested range is already cached → return from DB.
3. If partially cached → fetch only the missing prefix/suffix from the feed,
   store them, then return the full range from DB.
4. If nothing is cached → fetch the full range, store it, return it.

``ensure_cached`` pre-fetches a list of symbols for offline/batch use.
"""
from __future__ import annotations

from datetime import date
from typing import Sequence

from data.feeds.base import DataFeed
from data.storage.database import Database
from data.storage.models import DailyBar


class DataCache:
    """Transparent caching layer that merges DB and live feed data.

    Parameters
    ----------
    database:
        Open :class:`~data.storage.database.Database` instance (tables must
        already exist).
    feed:
        A concrete :class:`~data.feeds.base.DataFeed` implementation used to
        fetch data that is not yet in the DB.
    """

    def __init__(self, database: Database, feed: DataFeed) -> None:
        self._db = database
        self._feed = feed

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def get_daily_bars(
        self, symbol: str, start: date, end: date
    ) -> list[DailyBar]:
        """Return daily bars for *symbol* over [*start*, *end*], using cache.

        Missing date ranges are fetched from the underlying feed, stored in
        the database, and then the complete set is returned from the DB so
        that the return value is always consistent with what is persisted.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        start:
            First date of the range, inclusive.
        end:
            Last date of the range, inclusive.
        """
        cached_min, cached_max = self._db.get_cached_date_range(symbol)

        if cached_min is None:
            # Nothing cached at all — fetch full range
            self._fetch_and_store(symbol, start, end)
        else:
            # Fetch any missing prefix (data before cached_min)
            if start < cached_min:
                # Fetch up to one day before cached_min to avoid duplicates
                from datetime import timedelta
                fetch_end = cached_min - timedelta(days=1)
                if start <= fetch_end:
                    self._fetch_and_store(symbol, start, fetch_end)

            # Fetch any missing suffix (data after cached_max)
            if end > cached_max:
                from datetime import timedelta
                fetch_start = cached_max + timedelta(days=1)
                if fetch_start <= end:
                    self._fetch_and_store(symbol, fetch_start, end)

        return self._db.get_daily_bars(symbol, start, end)

    def ensure_cached(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> None:
        """Pre-fetch and cache *symbols* for the given date range.

        This is a convenience wrapper around :meth:`get_daily_bars` for
        batch/offline scenarios.  It calls ``get_daily_bars`` for each symbol
        but discards the return value — the intent is only to populate the DB.

        Parameters
        ----------
        symbols:
            Iterable of ticker symbols to warm up.
        start:
            First date of the range, inclusive.
        end:
            Last date of the range, inclusive.
        """
        for symbol in symbols:
            self.get_daily_bars(symbol, start, end)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_store(
        self, symbol: str, start: date, end: date
    ) -> None:
        """Fetch bars from the feed and insert them into the DB.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        start:
            Fetch range start (inclusive).
        end:
            Fetch range end (inclusive).
        """
        bars = self._feed.get_daily_bars(symbol, start, end)
        if bars:
            self._db.insert_daily_bars(bars)
