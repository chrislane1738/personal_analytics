"""Tests for data/feeds/fmp_feed.py.

All tests use pre-fabricated JSON fixtures and mock httpx — the real FMP API
is never called unless the ``integration`` mark is explicitly requested.
"""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from data.feeds.fmp_feed import FMPFeed, StockSplitEvent
from data.storage.models import DailyBar, SymbolMetadata
from utils.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Fixtures — pre-fabricated FMP JSON payloads
# ---------------------------------------------------------------------------

AAPL_HISTORICAL_RESPONSE = {
    "symbol": "AAPL",
    "historical": [
        {
            "date": "2024-01-05",
            "open": 182.0,
            "high": 183.5,
            "low": 181.0,
            "close": 182.8,
            "adjClose": 182.5,
            "volume": 55_000_000,
            "vwap": 182.4,
            "changePercent": 0.2,
        },
        {
            "date": "2024-01-04",
            "open": 181.5,
            "high": 183.0,
            "low": 180.5,
            "close": 182.3,
            "adjClose": 182.0,
            "volume": 48_000_000,
            "vwap": 181.8,
            "changePercent": 0.4,
        },
        {
            "date": "2024-01-03",
            "open": 184.0,
            "high": 185.2,
            "low": 180.0,
            "close": 181.0,
            "adjClose": 180.7,
            "volume": 60_000_000,
            "vwap": 182.1,
            "changePercent": -1.6,
        },
    ],
}

AAPL_PROFILE_RESPONSE = [
    {
        "symbol": "AAPL",
        "companyName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "exchangeShortName": "NASDAQ",
        "mktCap": 3_000_000_000_000,
        "description": "Apple designs consumer electronics.",
    }
]

AAPL_SPLITS_RESPONSE = {
    "symbol": "AAPL",
    "historical": [
        {"date": "2020-08-31", "numerator": 4.0, "denominator": 1.0, "label": "4:1"},
        {"date": "2014-06-09", "numerator": 7.0, "denominator": 1.0, "label": "7:1"},
    ],
}

SP500_CONSTITUENTS_RESPONSE = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "GOOGL", "name": "Alphabet"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feed(api_key: str = "test-key") -> FMPFeed:
    """Return an FMPFeed backed by a no-op rate limiter."""
    # Use a very high limit so tests never block
    rl = RateLimiter(max_requests_per_minute=10_000)
    return FMPFeed(api_key=api_key, rate_limiter=rl)


def _mock_response(json_data: object, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response that returns *json_data*."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# get_daily_bars
# ---------------------------------------------------------------------------

class TestGetDailyBars:
    def test_returns_correct_number_of_bars(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_HISTORICAL_RESPONSE)):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 5))
        assert len(bars) == 3

    def test_bars_are_sorted_ascending_by_date(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_HISTORICAL_RESPONSE)):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 5))
        dates = [b.date for b in bars]
        assert dates == sorted(dates)

    def test_bar_fields_are_correctly_parsed(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_HISTORICAL_RESPONSE)):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 5))
        # Oldest bar
        first = bars[0]
        assert first.symbol == "AAPL"
        assert first.date == date(2024, 1, 3)
        assert first.open == 184.0
        assert first.high == 185.2
        assert first.low == 180.0
        assert first.close == 181.0
        assert first.adj_close == 180.7
        assert first.volume == 60_000_000
        assert first.vwap == 182.1

    def test_returns_dailybar_instances(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_HISTORICAL_RESPONSE)):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 5))
        assert all(isinstance(b, DailyBar) for b in bars)

    def test_empty_historical_returns_empty_list(self):
        feed = _make_feed()
        empty_resp = {"symbol": "AAPL", "historical": []}
        with patch.object(feed._client, "get", return_value=_mock_response(empty_resp)):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert bars == []

    def test_missing_historical_key_returns_empty_list(self):
        """If FMP returns a response without 'historical', return empty list."""
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response({})):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert bars == []

    def test_malformed_records_are_skipped(self):
        """Records with missing required fields should be silently dropped."""
        feed = _make_feed()
        bad_response = {
            "symbol": "AAPL",
            "historical": [
                # Good record
                {
                    "date": "2024-01-05",
                    "open": 182.0,
                    "high": 183.5,
                    "low": 181.0,
                    "close": 182.8,
                    "adjClose": 182.5,
                    "volume": 55_000_000,
                    "vwap": 182.4,
                },
                # Malformed — missing "close"
                {"date": "2024-01-04", "open": 181.5},
            ],
        }
        with patch.object(feed._client, "get", return_value=_mock_response(bad_response)):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 4), date(2024, 1, 5))
        assert len(bars) == 1
        assert bars[0].date == date(2024, 1, 5)

    def test_rate_limiter_is_called(self):
        """acquire() must be called exactly once per get_daily_bars() call."""
        rl = MagicMock(spec=RateLimiter)
        feed = FMPFeed(api_key="test-key", rate_limiter=rl)
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_HISTORICAL_RESPONSE)):
            feed.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 5))
        rl.acquire.assert_called_once()

    def test_correct_url_is_called(self):
        """The request URL must include the symbol and date parameters."""
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_HISTORICAL_RESPONSE)) as mock_get:
            feed.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 5))
        call_kwargs = mock_get.call_args
        url = call_kwargs[0][0]
        params = call_kwargs[1]["params"]
        assert "AAPL" in url
        assert params["from"] == "2024-01-03"
        assert params["to"] == "2024-01-05"
        assert params["apikey"] == "test-key"

    def test_adj_close_falls_back_to_close_when_missing(self):
        """If adjClose is absent, adj_close should equal close."""
        feed = _make_feed()
        response = {
            "symbol": "AAPL",
            "historical": [
                {
                    "date": "2024-01-03",
                    "open": 184.0,
                    "high": 185.2,
                    "low": 180.0,
                    "close": 181.0,
                    # no adjClose
                    "volume": 60_000_000,
                    "vwap": 182.1,
                }
            ],
        }
        with patch.object(feed._client, "get", return_value=_mock_response(response)):
            bars = feed.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 3))
        assert bars[0].adj_close == bars[0].close


# ---------------------------------------------------------------------------
# get_symbol_profile
# ---------------------------------------------------------------------------

class TestGetSymbolProfile:
    def test_returns_symbol_metadata(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_PROFILE_RESPONSE)):
            meta = feed.get_symbol_profile("AAPL")
        assert isinstance(meta, SymbolMetadata)

    def test_profile_fields_are_correctly_parsed(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_PROFILE_RESPONSE)):
            meta = feed.get_symbol_profile("AAPL")
        assert meta.symbol == "AAPL"
        assert meta.company_name == "Apple Inc."
        assert meta.sector == "Technology"
        assert meta.industry == "Consumer Electronics"
        assert meta.exchange == "NASDAQ"
        assert meta.market_cap == 3_000_000_000_000

    def test_updated_at_is_recent_datetime(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_PROFILE_RESPONSE)):
            meta = feed.get_symbol_profile("AAPL")
        assert isinstance(meta.updated_at, datetime)

    def test_empty_profile_raises_value_error(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response([])):
            with pytest.raises(ValueError, match="Empty profile"):
                feed.get_symbol_profile("AAPL")

    def test_rate_limiter_is_called(self):
        rl = MagicMock(spec=RateLimiter)
        feed = FMPFeed(api_key="test-key", rate_limiter=rl)
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_PROFILE_RESPONSE)):
            feed.get_symbol_profile("AAPL")
        rl.acquire.assert_called_once()


# ---------------------------------------------------------------------------
# get_stock_splits
# ---------------------------------------------------------------------------

class TestGetStockSplits:
    def test_returns_list_of_split_events(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_SPLITS_RESPONSE)):
            splits = feed.get_stock_splits("AAPL")
        assert len(splits) == 2
        assert all(isinstance(s, StockSplitEvent) for s in splits)

    def test_split_fields_are_correctly_parsed(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_SPLITS_RESPONSE)):
            splits = feed.get_stock_splits("AAPL")
        first = splits[0]
        assert first.symbol == "AAPL"
        assert first.date == date(2020, 8, 31)
        assert first.numerator == 4.0
        assert first.denominator == 1.0

    def test_empty_historical_returns_empty_list(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response({"symbol": "AAPL", "historical": []})):
            splits = feed.get_stock_splits("AAPL")
        assert splits == []

    def test_rate_limiter_is_called(self):
        rl = MagicMock(spec=RateLimiter)
        feed = FMPFeed(api_key="test-key", rate_limiter=rl)
        with patch.object(feed._client, "get", return_value=_mock_response(AAPL_SPLITS_RESPONSE)):
            feed.get_stock_splits("AAPL")
        rl.acquire.assert_called_once()


# ---------------------------------------------------------------------------
# get_symbols
# ---------------------------------------------------------------------------

class TestGetSymbols:
    def test_returns_list_of_symbols(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response(SP500_CONSTITUENTS_RESPONSE)):
            symbols = feed.get_symbols("sp500")
        assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_symbols_are_uppercase(self):
        feed = _make_feed()
        lowercase = [{"symbol": "aapl"}, {"symbol": "msft"}]
        with patch.object(feed._client, "get", return_value=_mock_response(lowercase)):
            symbols = feed.get_symbols("sp500")
        assert all(s == s.upper() for s in symbols)

    def test_rate_limiter_is_called(self):
        rl = MagicMock(spec=RateLimiter)
        feed = FMPFeed(api_key="test-key", rate_limiter=rl)
        with patch.object(feed._client, "get", return_value=_mock_response(SP500_CONSTITUENTS_RESPONSE)):
            feed.get_symbols("sp500")
        rl.acquire.assert_called_once()

    def test_sp500_endpoint(self):
        """get_symbols('sp500') must call the sp500_constituent endpoint."""
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response([])) as mock_get:
            feed.get_symbols("sp500")
        url = mock_get.call_args[0][0]
        assert "sp500_constituent" in url

    def test_nasdaq100_endpoint(self):
        feed = _make_feed()
        with patch.object(feed._client, "get", return_value=_mock_response([])) as mock_get:
            feed.get_symbols("nasdaq100")
        url = mock_get.call_args[0][0]
        assert "nasdaq_constituent" in url


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_closes_client():
    """FMPFeed should work as a context manager and close the httpx client."""
    with _make_feed() as feed:
        assert feed._client is not None
    # After __exit__ the client should be closed (httpx marks it as closed)
    assert feed._client.is_closed


# ---------------------------------------------------------------------------
# Integration test (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_fetch_aapl_bars():
    """Fetch real AAPL data from FMP (requires FMP_API_KEY in environment)."""
    import os
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        pytest.skip("FMP_API_KEY not set")

    feed = FMPFeed(api_key=api_key)
    bars = feed.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 5))
    assert len(bars) > 0
    assert all(isinstance(b, DailyBar) for b in bars)
    assert bars[0].date <= bars[-1].date
