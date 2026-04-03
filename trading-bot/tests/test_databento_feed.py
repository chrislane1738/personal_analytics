"""Tests for Databento feed, IntradayBar model, intraday DB methods, and cache."""

import os
import tempfile
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from data.storage.models import DailyBar, IntradayBar
from data.storage.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Provide a temporary database with all tables created."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    database.create_tables()
    yield database
    database.close()
    os.unlink(db_path)


def _make_intraday_bar(
    symbol: str = "ES",
    timestamp: datetime | None = None,
    timeframe: str = "1m",
    close: float = 5000.25,
) -> IntradayBar:
    """Helper to create a sample IntradayBar."""
    if timestamp is None:
        timestamp = datetime(2024, 6, 3, 14, 30, 0, tzinfo=timezone.utc)
    return IntradayBar(
        symbol=symbol,
        timestamp=timestamp,
        timeframe=timeframe,
        open=5000.0,
        high=5001.0,
        low=4999.5,
        close=close,
        volume=1500,
        vwap=0.0,
        data_quality_score=1.0,
    )


# ---------------------------------------------------------------------------
# IntradayBar model tests
# ---------------------------------------------------------------------------

class TestIntradayBarModel:
    """Tests for the IntradayBar dataclass."""

    def test_creation_with_defaults(self):
        bar = IntradayBar(
            symbol="ES",
            timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            timeframe="1m",
            open=5000.0,
            high=5001.0,
            low=4999.5,
            close=5000.25,
            volume=1500,
        )
        assert bar.symbol == "ES"
        assert bar.timeframe == "1m"
        assert bar.vwap == 0.0
        assert bar.data_quality_score == 1.0

    def test_creation_with_explicit_values(self):
        bar = IntradayBar(
            symbol="NQ",
            timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            timeframe="5m",
            open=18000.0,
            high=18050.0,
            low=17990.0,
            close=18020.0,
            volume=3000,
            vwap=18010.0,
            data_quality_score=0.95,
        )
        assert bar.symbol == "NQ"
        assert bar.vwap == 18010.0
        assert bar.data_quality_score == 0.95

    def test_timestamp_is_full_datetime(self):
        ts = datetime(2024, 6, 3, 9, 31, 0, tzinfo=timezone.utc)
        bar = _make_intraday_bar(timestamp=ts)
        assert bar.timestamp.hour == 9
        assert bar.timestamp.minute == 31


# ---------------------------------------------------------------------------
# Database intraday_bars tests
# ---------------------------------------------------------------------------

class TestDatabaseIntradayBars:
    """Tests for the intraday_bars table and associated DB methods."""

    def test_intraday_bars_table_exists(self, db):
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'", ()
        ).fetchall()
        table_names = {row[0] for row in rows}
        assert "intraday_bars" in table_names

    def test_insert_and_query_intraday_bars(self, db):
        bars = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
                close=5000.0,
            ),
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 31, tzinfo=timezone.utc),
                close=5001.0,
            ),
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 32, tzinfo=timezone.utc),
                close=5002.0,
            ),
        ]
        db.insert_intraday_bars(bars)

        result = db.get_intraday_bars(
            "ES",
            datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 14, 32, tzinfo=timezone.utc),
            "1m",
        )
        assert len(result) == 3
        assert result[0].close == 5000.0
        assert result[2].close == 5002.0

    def test_intraday_bars_ordered_by_timestamp(self, db):
        bars = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 32, tzinfo=timezone.utc),
            ),
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            ),
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 31, tzinfo=timezone.utc),
            ),
        ]
        db.insert_intraday_bars(bars)

        result = db.get_intraday_bars(
            "ES",
            datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc),
            "1m",
        )
        timestamps = [r.timestamp for r in result]
        assert timestamps == sorted(timestamps)

    def test_upsert_intraday_bars(self, db):
        bar = _make_intraday_bar(close=5000.0)
        db.insert_intraday_bars([bar])

        updated = _make_intraday_bar(close=5999.0)
        db.insert_intraday_bars([updated])

        result = db.get_intraday_bars(
            "ES",
            datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            "1m",
        )
        assert len(result) == 1
        assert result[0].close == 5999.0

    def test_query_filters_by_timeframe(self, db):
        bar_1m = _make_intraday_bar(timeframe="1m")
        bar_5m = _make_intraday_bar(timeframe="5m")
        db.insert_intraday_bars([bar_1m, bar_5m])

        result_1m = db.get_intraday_bars(
            "ES",
            datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc),
            "1m",
        )
        result_5m = db.get_intraday_bars(
            "ES",
            datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc),
            "5m",
        )
        assert len(result_1m) == 1
        assert len(result_5m) == 1
        assert result_1m[0].timeframe == "1m"
        assert result_5m[0].timeframe == "5m"

    def test_query_filters_by_symbol(self, db):
        bar_es = _make_intraday_bar(symbol="ES")
        bar_nq = _make_intraday_bar(symbol="NQ")
        db.insert_intraday_bars([bar_es, bar_nq])

        result = db.get_intraday_bars(
            "ES",
            datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc),
            "1m",
        )
        assert len(result) == 1
        assert result[0].symbol == "ES"

    def test_get_intraday_cached_range(self, db):
        bars = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 9, 30, tzinfo=timezone.utc),
            ),
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 16, 0, tzinfo=timezone.utc),
            ),
        ]
        db.insert_intraday_bars(bars)

        min_ts, max_ts = db.get_intraday_cached_range("ES", "1m")
        assert min_ts is not None
        assert max_ts is not None
        assert min_ts.hour == 9
        assert max_ts.hour == 16

    def test_get_intraday_cached_range_empty(self, db):
        min_ts, max_ts = db.get_intraday_cached_range("ES", "1m")
        assert min_ts is None
        assert max_ts is None


# ---------------------------------------------------------------------------
# DatabentFeed tests (mocked -- no real API calls)
# ---------------------------------------------------------------------------

def _make_mock_df(rows: list[dict], index_name: str = "ts_event"):
    """Build a mock pandas-like DataFrame from a list of dicts.

    Each dict must include a 'ts_event' key (used as index) and OHLCV cols.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.pop(index_name), utc=True)
    return df


@pytest.fixture
def mock_databento_client():
    """Return a mock databento.Historical client."""
    client = MagicMock()
    return client


class TestDatabentFeedSymbolMapping:
    """Test symbol and timeframe mapping without importing databento."""

    def test_symbol_map_contains_es(self):
        from data.feeds.databento_feed import DatabentFeed

        assert "ES" in DatabentFeed.SYMBOL_MAP
        assert DatabentFeed.SYMBOL_MAP["ES"] == "ES.FUT"

    def test_symbol_map_contains_all_expected(self):
        from data.feeds.databento_feed import DatabentFeed

        expected = {"ES", "MES", "NQ", "MNQ"}
        assert set(DatabentFeed.SYMBOL_MAP.keys()) == expected

    def test_timeframe_to_schema(self):
        from data.feeds.databento_feed import DatabentFeed

        assert DatabentFeed.TIMEFRAME_TO_SCHEMA["1m"] == "ohlcv-1m"
        assert DatabentFeed.TIMEFRAME_TO_SCHEMA["5m"] == "ohlcv-5m"
        assert DatabentFeed.TIMEFRAME_TO_SCHEMA["1D"] == "ohlcv-1d"

    def test_dataset_is_cme_globex(self):
        from data.feeds.databento_feed import DatabentFeed

        assert DatabentFeed.DATASET == "GLBX.MDP3"


class TestDatabentFeedIntradayBars:
    """Test get_intraday_bars with mocked Databento client."""

    def test_unknown_symbol_raises_value_error(self, mock_databento_client):
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            with pytest.raises(ValueError, match="Unknown symbol"):
                feed.get_intraday_bars("INVALID", "1m", date(2024, 1, 1), date(2024, 1, 2))

    def test_unknown_timeframe_raises_value_error(self, mock_databento_client):
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            with pytest.raises(ValueError, match="Unknown timeframe"):
                feed.get_intraday_bars("ES", "99m", date(2024, 1, 1), date(2024, 1, 2))

    def test_parses_bars_from_dataframe(self, mock_databento_client):
        """Verify bars are correctly parsed from a DataFrame (normal prices)."""
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            mock_df = _make_mock_df([
                {
                    "ts_event": "2024-06-03T14:30:00+00:00",
                    "open": 5000.0,
                    "high": 5001.0,
                    "low": 4999.5,
                    "close": 5000.25,
                    "volume": 1500,
                },
                {
                    "ts_event": "2024-06-03T14:31:00+00:00",
                    "open": 5000.25,
                    "high": 5002.0,
                    "low": 5000.0,
                    "close": 5001.5,
                    "volume": 1200,
                },
            ])

            mock_store = MagicMock()
            mock_store.to_df.return_value = mock_df
            mock_databento_client.timeseries.get_range.return_value = mock_store

            bars = feed.get_intraday_bars("ES", "1m", date(2024, 6, 3), date(2024, 6, 4))

            assert len(bars) == 2
            assert bars[0].symbol == "ES"
            assert bars[0].timeframe == "1m"
            assert bars[0].close == 5000.25
            assert bars[1].close == 5001.5
            assert bars[0].volume == 1500

    def test_handles_fixed_point_prices(self, mock_databento_client):
        """Verify that fixed-point (billionths) prices are auto-detected and converted."""
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            # Prices in billionths: 5000.25 * 1e9 = 5000250000000
            mock_df = _make_mock_df([
                {
                    "ts_event": "2024-06-03T14:30:00+00:00",
                    "open": 5000250000000.0,
                    "high": 5001000000000.0,
                    "low": 4999500000000.0,
                    "close": 5000250000000.0,
                    "volume": 1500,
                },
            ])

            mock_store = MagicMock()
            mock_store.to_df.return_value = mock_df
            mock_databento_client.timeseries.get_range.return_value = mock_store

            bars = feed.get_intraday_bars("ES", "1m", date(2024, 6, 3), date(2024, 6, 4))

            assert len(bars) == 1
            assert abs(bars[0].open - 5000.25) < 0.01
            assert abs(bars[0].close - 5000.25) < 0.01

    def test_empty_dataframe_returns_empty_list(self, mock_databento_client):
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            import pandas as pd
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            mock_store = MagicMock()
            mock_store.to_df.return_value = pd.DataFrame()
            mock_databento_client.timeseries.get_range.return_value = mock_store

            bars = feed.get_intraday_bars("ES", "1m", date(2024, 6, 3), date(2024, 6, 4))
            assert bars == []

    def test_calls_databento_with_correct_params(self, mock_databento_client):
        """Verify the Databento API is called with correct dataset, symbol, schema."""
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            import pandas as pd
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            mock_store = MagicMock()
            mock_store.to_df.return_value = pd.DataFrame()
            mock_databento_client.timeseries.get_range.return_value = mock_store

            feed.get_intraday_bars("MES", "5m", date(2024, 6, 3), date(2024, 6, 4))

            mock_databento_client.timeseries.get_range.assert_called_once_with(
                dataset="GLBX.MDP3",
                symbols=["MES.FUT"],
                schema="ohlcv-5m",
                start="2024-06-03",
                end="2024-06-04",
            )


class TestDatabentFeedDailyBars:
    """Test get_daily_bars with mocked Databento client."""

    def test_daily_bars_parses_correctly(self, mock_databento_client):
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            mock_df = _make_mock_df([
                {
                    "ts_event": "2024-06-03T00:00:00+00:00",
                    "open": 5000.0,
                    "high": 5050.0,
                    "low": 4980.0,
                    "close": 5030.0,
                    "volume": 1_500_000,
                },
            ])

            mock_store = MagicMock()
            mock_store.to_df.return_value = mock_df
            mock_databento_client.timeseries.get_range.return_value = mock_store

            bars = feed.get_daily_bars("ES", date(2024, 6, 3), date(2024, 6, 4))

            assert len(bars) == 1
            assert isinstance(bars[0], DailyBar)
            assert bars[0].symbol == "ES"
            assert bars[0].close == 5030.0
            assert bars[0].date == date(2024, 6, 3)

    def test_daily_bars_unknown_symbol(self, mock_databento_client):
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            feed._client = mock_databento_client

            with pytest.raises(ValueError, match="Unknown symbol"):
                feed.get_daily_bars("AAPL", date(2024, 1, 1), date(2024, 1, 2))


class TestDatabentFeedGetSymbols:
    """Test get_symbols method."""

    def test_returns_all_mapped_symbols(self):
        with patch("data.feeds.databento_feed.DatabentFeed.__init__", return_value=None):
            from data.feeds.databento_feed import DatabentFeed

            feed = DatabentFeed.__new__(DatabentFeed)
            symbols = feed.get_symbols("any_universe")
            assert set(symbols) == {"ES", "MES", "NQ", "MNQ"}


# ---------------------------------------------------------------------------
# IntradayCache tests
# ---------------------------------------------------------------------------

class TestIntradayCache:
    """Test the IntradayCache layer."""

    def test_fetches_from_feed_when_cache_empty(self, db):
        from data.storage.cache import IntradayCache

        mock_feed = MagicMock()
        bars = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            ),
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 31, tzinfo=timezone.utc),
            ),
        ]
        mock_feed.get_intraday_bars.return_value = bars

        cache = IntradayCache(db, mock_feed)
        result = cache.get_intraday_bars(
            "ES", "1m", date(2024, 6, 3), date(2024, 6, 3)
        )

        mock_feed.get_intraday_bars.assert_called_once()
        assert len(result) == 2

    def test_uses_cached_data_on_second_call(self, db):
        from data.storage.cache import IntradayCache

        mock_feed = MagicMock()
        bars = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            ),
        ]
        mock_feed.get_intraday_bars.return_value = bars

        cache = IntradayCache(db, mock_feed)

        # First call fetches from feed
        cache.get_intraday_bars("ES", "1m", date(2024, 6, 3), date(2024, 6, 3))
        assert mock_feed.get_intraday_bars.call_count == 1

        # Second call for same range should not re-fetch
        result = cache.get_intraday_bars("ES", "1m", date(2024, 6, 3), date(2024, 6, 3))
        assert mock_feed.get_intraday_bars.call_count == 1
        assert len(result) == 1

    def test_fetches_missing_suffix(self, db):
        from data.storage.cache import IntradayCache

        mock_feed = MagicMock()

        # Pre-populate cache with day 3 data
        day3_bars = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            ),
        ]
        db.insert_intraday_bars(day3_bars)

        # Feed will return day 4 data
        day4_bars = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 4, 14, 30, tzinfo=timezone.utc),
            ),
        ]
        mock_feed.get_intraday_bars.return_value = day4_bars

        cache = IntradayCache(db, mock_feed)
        result = cache.get_intraday_bars(
            "ES", "1m", date(2024, 6, 3), date(2024, 6, 4)
        )

        # Should have fetched the suffix
        mock_feed.get_intraday_bars.assert_called_once()
        assert len(result) == 2

    def test_ensure_cached(self, db):
        from data.storage.cache import IntradayCache

        mock_feed = MagicMock()
        mock_feed.get_intraday_bars.return_value = [
            _make_intraday_bar(
                timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
            ),
        ]

        cache = IntradayCache(db, mock_feed)
        cache.ensure_cached(["ES", "NQ"], "1m", date(2024, 6, 3), date(2024, 6, 3))

        # Feed should have been called once per symbol
        assert mock_feed.get_intraday_bars.call_count == 2


# ---------------------------------------------------------------------------
# BacktestEngine timeframe guard
# ---------------------------------------------------------------------------

class TestBacktestEngineTimeframe:
    """Test that BacktestEngine rejects intraday timeframes for now."""

    def test_daily_timeframe_accepted(self, db):
        from core.engine import BacktestEngine

        strategy = MagicMock()
        # Should not raise for "1D"
        engine = BacktestEngine(
            strategy=strategy,
            database=db,
            universe=["ES"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            timeframe="1D",
        )
        assert engine.timeframe == "1D"

    def test_intraday_timeframe_raises(self, db):
        from core.engine import BacktestEngine

        strategy = MagicMock()
        with pytest.raises(NotImplementedError, match="Intraday timeframe"):
            BacktestEngine(
                strategy=strategy,
                database=db,
                universe=["ES"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                timeframe="1m",
            )

    def test_5m_timeframe_raises(self, db):
        from core.engine import BacktestEngine

        strategy = MagicMock()
        with pytest.raises(NotImplementedError, match="Intraday timeframe"):
            BacktestEngine(
                strategy=strategy,
                database=db,
                universe=["ES"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                timeframe="5m",
            )
