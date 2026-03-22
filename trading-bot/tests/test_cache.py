"""Tests for data/storage/cache.py.

Uses an in-memory SQLite database and a mock DataFeed to verify cache-miss,
cache-hit, and partial-cache logic without touching the real FMP API.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence
from unittest.mock import MagicMock, call

import pytest

from data.feeds.base import DataFeed
from data.storage.cache import DataCache
from data.storage.database import Database
from data.storage.models import DailyBar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    """In-memory SQLite database with schema created."""
    database = Database(":memory:")
    database.create_tables()
    return database


def _make_bar(symbol: str = "AAPL", d: date = date(2024, 1, 2), close: float = 185.0) -> DailyBar:
    return DailyBar(
        symbol=symbol,
        date=d,
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        adj_close=close,
        volume=10_000_000,
        vwap=close,
    )


def _bars_for_range(symbol: str, start: date, end: date) -> list[DailyBar]:
    """Generate one bar per calendar day between start and end inclusive."""
    bars = []
    d = start
    while d <= end:
        bars.append(_make_bar(symbol=symbol, d=d))
        d += timedelta(days=1)
    return bars


class MockFeed(DataFeed):
    """A DataFeed whose get_daily_bars() returns canned data."""

    def __init__(self, bars_by_symbol: dict[str, list[DailyBar]] | None = None) -> None:
        # Map from symbol -> ALL available bars (the feed filters by date)
        self._all: dict[str, list[DailyBar]] = bars_by_symbol or {}
        self.call_log: list[tuple[str, date, date]] = []

    def get_daily_bars(self, symbol: str, start: date, end: date) -> list[DailyBar]:
        self.call_log.append((symbol, start, end))
        all_bars = self._all.get(symbol, [])
        return [b for b in all_bars if start <= b.date <= end]

    def get_symbols(self, universe: str) -> list[str]:
        return list(self._all.keys())


# ---------------------------------------------------------------------------
# Cache miss — full fetch
# ---------------------------------------------------------------------------

class TestCacheMiss:
    def test_full_miss_calls_feed_once(self, db):
        feed = MockFeed({"AAPL": _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))})
        cache = DataCache(database=db, feed=feed)

        cache.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))

        assert len(feed.call_log) == 1
        assert feed.call_log[0] == ("AAPL", date(2024, 1, 2), date(2024, 1, 5))

    def test_full_miss_stores_bars_in_db(self, db):
        bars_to_store = _bars_for_range("AAPL", date(2024, 1, 2), date(2024, 1, 5))
        feed = MockFeed({"AAPL": bars_to_store})
        cache = DataCache(database=db, feed=feed)

        cache.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))

        stored = db.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))
        assert len(stored) == len(bars_to_store)

    def test_full_miss_returns_correct_bars(self, db):
        bars = _bars_for_range("AAPL", date(2024, 1, 2), date(2024, 1, 5))
        feed = MockFeed({"AAPL": bars})
        cache = DataCache(database=db, feed=feed)

        result = cache.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))

        assert [b.date for b in result] == [b.date for b in bars]


# ---------------------------------------------------------------------------
# Cache hit — no feed call
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_full_hit_does_not_call_feed(self, db):
        # Pre-populate DB
        bars = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        db.insert_daily_bars(bars)

        feed = MockFeed()  # empty — any call would return []
        cache = DataCache(database=db, feed=feed)

        cache.get_daily_bars("AAPL", date(2024, 1, 5), date(2024, 1, 10))

        assert feed.call_log == [], "Feed should not be called when data is cached"

    def test_full_hit_returns_correct_bars(self, db):
        bars = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        db.insert_daily_bars(bars)
        feed = MockFeed()
        cache = DataCache(database=db, feed=feed)

        result = cache.get_daily_bars("AAPL", date(2024, 1, 5), date(2024, 1, 10))

        expected_dates = [date(2024, 1, d) for d in range(5, 11)]
        assert [b.date for b in result] == expected_dates


# ---------------------------------------------------------------------------
# Partial cache — only missing ranges are fetched
# ---------------------------------------------------------------------------

class TestPartialCache:
    def test_prefix_missing_fetches_only_prefix(self, db):
        """Cache has Jan 10–31; request is Jan 5–20 → only Jan 5–9 fetched."""
        cached = _bars_for_range("AAPL", date(2024, 1, 10), date(2024, 1, 31))
        db.insert_daily_bars(cached)

        all_bars = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        feed = MockFeed({"AAPL": all_bars})
        cache = DataCache(database=db, feed=feed)

        cache.get_daily_bars("AAPL", date(2024, 1, 5), date(2024, 1, 20))

        assert len(feed.call_log) == 1
        fetch_start, fetch_end = feed.call_log[0][1], feed.call_log[0][2]
        # Must not overlap with already-cached range (Jan 10 onward)
        assert fetch_start == date(2024, 1, 5)
        assert fetch_end < date(2024, 1, 10)

    def test_suffix_missing_fetches_only_suffix(self, db):
        """Cache has Jan 1–15; request is Jan 10–20 → only Jan 16–20 fetched."""
        cached = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        db.insert_daily_bars(cached)

        all_bars = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        feed = MockFeed({"AAPL": all_bars})
        cache = DataCache(database=db, feed=feed)

        cache.get_daily_bars("AAPL", date(2024, 1, 10), date(2024, 1, 20))

        assert len(feed.call_log) == 1
        fetch_start, fetch_end = feed.call_log[0][1], feed.call_log[0][2]
        assert fetch_start > date(2024, 1, 15)
        assert fetch_end == date(2024, 1, 20)

    def test_both_sides_missing_fetches_two_ranges(self, db):
        """Cache has Jan 10–15; request is Jan 5–20 → prefix AND suffix fetched."""
        cached = _bars_for_range("AAPL", date(2024, 1, 10), date(2024, 1, 15))
        db.insert_daily_bars(cached)

        all_bars = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        feed = MockFeed({"AAPL": all_bars})
        cache = DataCache(database=db, feed=feed)

        cache.get_daily_bars("AAPL", date(2024, 1, 5), date(2024, 1, 20))

        assert len(feed.call_log) == 2

    def test_partial_result_contains_all_dates(self, db):
        """After partial fetch, the returned list must cover the full range."""
        cached = _bars_for_range("AAPL", date(2024, 1, 10), date(2024, 1, 15))
        db.insert_daily_bars(cached)

        all_bars = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        feed = MockFeed({"AAPL": all_bars})
        cache = DataCache(database=db, feed=feed)

        result = cache.get_daily_bars("AAPL", date(2024, 1, 5), date(2024, 1, 20))

        result_dates = {b.date for b in result}
        expected_dates = {
            date(2024, 1, d) for d in range(5, 21)
        }
        assert expected_dates == result_dates


# ---------------------------------------------------------------------------
# ensure_cached
# ---------------------------------------------------------------------------

class TestEnsureCached:
    def test_ensure_cached_warms_all_symbols(self, db):
        symbols = ["AAPL", "MSFT", "GOOGL"]
        all_bars = {
            s: _bars_for_range(s, date(2024, 1, 1), date(2024, 1, 31))
            for s in symbols
        }
        feed = MockFeed(all_bars)
        cache = DataCache(database=db, feed=feed)

        cache.ensure_cached(symbols, date(2024, 1, 5), date(2024, 1, 10))

        fetched_symbols = [log[0] for log in feed.call_log]
        for s in symbols:
            assert s in fetched_symbols

    def test_ensure_cached_stores_data_for_each_symbol(self, db):
        symbols = ["AAPL", "MSFT"]
        all_bars = {
            s: _bars_for_range(s, date(2024, 1, 1), date(2024, 1, 31))
            for s in symbols
        }
        feed = MockFeed(all_bars)
        cache = DataCache(database=db, feed=feed)

        cache.ensure_cached(symbols, date(2024, 1, 5), date(2024, 1, 10))

        for s in symbols:
            stored = db.get_daily_bars(s, date(2024, 1, 5), date(2024, 1, 10))
            assert len(stored) > 0, f"No bars stored for {s}"

    def test_ensure_cached_skips_already_cached_symbols(self, db):
        """Pre-populated symbols should not trigger another feed call."""
        bars = _bars_for_range("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        db.insert_daily_bars(bars)

        feed = MockFeed({"AAPL": bars})
        cache = DataCache(database=db, feed=feed)

        cache.ensure_cached(["AAPL"], date(2024, 1, 5), date(2024, 1, 10))

        assert feed.call_log == [], "Feed should not be called for fully cached symbol"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_feed_response_returns_empty_list(self, db):
        feed = MockFeed({})  # returns [] for any symbol
        cache = DataCache(database=db, feed=feed)

        result = cache.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))

        assert result == []

    def test_single_day_range(self, db):
        bars = _bars_for_range("AAPL", date(2024, 1, 5), date(2024, 1, 5))
        feed = MockFeed({"AAPL": bars})
        cache = DataCache(database=db, feed=feed)

        result = cache.get_daily_bars("AAPL", date(2024, 1, 5), date(2024, 1, 5))

        assert len(result) == 1
        assert result[0].date == date(2024, 1, 5)

    def test_second_call_uses_cache(self, db):
        """Second identical call should not hit the feed at all."""
        bars = _bars_for_range("AAPL", date(2024, 1, 2), date(2024, 1, 5))
        feed = MockFeed({"AAPL": bars})
        cache = DataCache(database=db, feed=feed)

        cache.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))
        feed.call_log.clear()  # reset

        cache.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))

        assert feed.call_log == [], "Second call should be served from cache"
