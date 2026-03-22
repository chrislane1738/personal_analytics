"""Financial Modeling Prep (FMP) data feed implementation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx

from data.feeds.base import DataFeed
from data.storage.models import DailyBar, SymbolMetadata
from utils.rate_limiter import RateLimiter


@dataclass
class StockSplitEvent:
    """A single historical stock-split event."""

    symbol: str
    date: date
    numerator: float    # shares after
    denominator: float  # shares before


class FMPFeed(DataFeed):
    """Synchronous FMP API client that respects the per-minute rate limit.

    Parameters
    ----------
    api_key:
        FMP API key string.
    base_url:
        Base URL of the FMP API, defaults to the public endpoint.
    rate_limiter:
        Optional pre-configured :class:`RateLimiter`.  A default 700 req/min
        limiter is created when *None* is passed.
    """

    DEFAULT_BASE_URL = "https://financialmodelingprep.com/api"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._rate_limiter = rate_limiter or RateLimiter()
        self._client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------
    # DataFeed interface
    # ------------------------------------------------------------------

    def get_daily_bars(
        self, symbol: str, start: date, end: date
    ) -> list[DailyBar]:
        """Fetch daily OHLCV bars from ``/api/v3/historical-price-full/{symbol}``.

        Parameters
        ----------
        symbol:
            Ticker, e.g. ``'AAPL'``.
        start:
            First date of the range, inclusive (``YYYY-MM-DD``).
        end:
            Last date of the range, inclusive (``YYYY-MM-DD``).

        Returns
        -------
        list[DailyBar]
            Bars sorted by date ascending.
        """
        self._rate_limiter.acquire()
        url = f"{self._base_url}/v3/historical-price-full/{symbol}"
        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "apikey": self._api_key,
        }
        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return self._parse_daily_bars(symbol, data)

    def get_symbols(self, universe: str) -> list[str]:
        """Fetch the component tickers of a named *universe*.

        Supported universes:
        * ``'sp500'`` — calls ``/api/v3/sp500_constituent``
        * ``'nasdaq100'`` — calls ``/api/v3/nasdaq_constituent``
        * ``'dowjones'`` — calls ``/api/v3/dowjones_constituent``

        For any other value the endpoint is constructed as
        ``/api/v3/{universe}_constituent``.

        Returns
        -------
        list[str]
            Uppercase ticker symbols.
        """
        self._rate_limiter.acquire()
        endpoint_map = {
            "sp500": "sp500_constituent",
            "nasdaq100": "nasdaq_constituent",
            "dowjones": "dowjones_constituent",
        }
        endpoint = endpoint_map.get(universe.lower(), f"{universe}_constituent")
        url = f"{self._base_url}/v3/{endpoint}"
        params = {"apikey": self._api_key}
        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return [item["symbol"].upper() for item in data if "symbol" in item]

    # ------------------------------------------------------------------
    # Additional API methods
    # ------------------------------------------------------------------

    def get_symbol_profile(self, symbol: str) -> SymbolMetadata:
        """Fetch company profile from ``/api/v3/profile/{symbol}``.

        Parameters
        ----------
        symbol:
            Ticker, e.g. ``'AAPL'``.

        Returns
        -------
        SymbolMetadata
            Populated metadata record with ``updated_at`` set to now.
        """
        self._rate_limiter.acquire()
        url = f"{self._base_url}/v3/profile/{symbol}"
        params = {"apikey": self._api_key}
        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return self._parse_symbol_profile(symbol, data)

    def get_stock_splits(self, symbol: str) -> list[StockSplitEvent]:
        """Fetch historical stock splits from ``/api/v3/historical-stock-split/{symbol}``.

        Parameters
        ----------
        symbol:
            Ticker, e.g. ``'AAPL'``.

        Returns
        -------
        list[StockSplitEvent]
            Split events sorted by date descending (newest first) as returned
            by FMP.
        """
        self._rate_limiter.acquire()
        url = f"{self._base_url}/v3/historical-stock-split/{symbol}"
        params = {"apikey": self._api_key}
        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return self._parse_stock_splits(symbol, data)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_daily_bars(symbol: str, data: dict[str, Any]) -> list[DailyBar]:
        """Convert the FMP historical-price-full JSON payload to DailyBar list.

        The FMP response looks like::

            {
                "symbol": "AAPL",
                "historical": [
                    {
                        "date": "2024-01-02",
                        "open": 185.5,
                        "high": 187.2,
                        "low": 184.8,
                        "close": 186.9,
                        "adjClose": 186.45,
                        "volume": 16900000,
                        "vwap": 186.3,
                        ...
                    },
                    ...
                ]
            }

        Bars are returned sorted by date ascending (oldest first).
        """
        historical = data.get("historical", [])
        bars: list[DailyBar] = []
        for item in historical:
            try:
                bar = DailyBar(
                    symbol=symbol,
                    date=date.fromisoformat(item["date"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    adj_close=float(item.get("adjClose", item["close"])),
                    volume=int(item.get("volume", 0)),
                    vwap=float(item.get("vwap", item["close"])),
                )
                bars.append(bar)
            except (KeyError, ValueError, TypeError):
                # Skip malformed records rather than blowing up the whole fetch
                continue
        # FMP returns newest-first; reverse to oldest-first
        bars.sort(key=lambda b: b.date)
        return bars

    @staticmethod
    def _parse_symbol_profile(
        symbol: str, data: list[dict[str, Any]]
    ) -> SymbolMetadata:
        """Convert the FMP profile JSON payload to a SymbolMetadata record."""
        if not data:
            raise ValueError(f"Empty profile response for {symbol}")
        profile = data[0]
        return SymbolMetadata(
            symbol=symbol,
            company_name=profile.get("companyName", ""),
            sector=profile.get("sector", ""),
            industry=profile.get("industry", ""),
            exchange=profile.get("exchangeShortName", ""),
            market_cap=float(profile.get("mktCap", 0) or 0),
            updated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _parse_stock_splits(
        symbol: str, data: dict[str, Any]
    ) -> list[StockSplitEvent]:
        """Convert the FMP historical-stock-split JSON payload to a list of events."""
        historical = data.get("historical", [])
        events: list[StockSplitEvent] = []
        for item in historical:
            try:
                events.append(
                    StockSplitEvent(
                        symbol=symbol,
                        date=date.fromisoformat(item["date"]),
                        numerator=float(item.get("numerator", 1)),
                        denominator=float(item.get("denominator", 1)),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        return events

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "FMPFeed":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
