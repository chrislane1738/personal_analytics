"""Databento data feed for CME Globex futures (ES, MES, NQ, MNQ).

Provides intraday OHLCV bars (1m, 5m) and daily bars via the Databento
Historical API.  The ``databento`` package is imported lazily so the rest
of the framework works even when it is not installed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from data.feeds.base import DataFeed
from data.storage.models import DailyBar, IntradayBar

logger = logging.getLogger(__name__)


class DatabentFeed(DataFeed):
    """Databento data feed for futures intraday and daily bars.

    Parameters
    ----------
    api_key:
        Databento API key string.
    """

    DATASET = "GLBX.MDP3"  # CME Globex

    # Map our internal symbols to Databento continuous front-month symbols
    SYMBOL_MAP: dict[str, str] = {
        "ES": "ES.FUT",
        "MES": "MES.FUT",
        "NQ": "NQ.FUT",
        "MNQ": "MNQ.FUT",
    }

    TIMEFRAME_TO_SCHEMA: dict[str, str] = {
        "1m": "ohlcv-1m",
        "5m": "ohlcv-5m",
        "1D": "ohlcv-1d",
    }

    # Databento fixed-point prices are stored as billionths (1e-9).
    # Reasonable ES price range used to auto-detect whether conversion is
    # needed (in case a future SDK version normalises prices automatically).
    _PRICE_SANITY_THRESHOLD = 1_000_000.0

    def __init__(self, api_key: str) -> None:
        try:
            import databento as db  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The 'databento' package is required for DatabentFeed. "
                "Install it with:  pip install databento"
            ) from exc

        self._client = db.Historical(key=api_key)

    # ------------------------------------------------------------------
    # DataFeed interface
    # ------------------------------------------------------------------

    def get_daily_bars(
        self, symbol: str, start: date, end: date
    ) -> list[DailyBar]:
        """Fetch daily OHLCV bars from Databento.

        Parameters
        ----------
        symbol:
            Internal symbol (e.g. ``'ES'``).
        start:
            First date of the range, inclusive.
        end:
            Last date of the range, inclusive.

        Returns
        -------
        list[DailyBar]
            Bars sorted by date ascending.
        """
        dbn_symbol = self.SYMBOL_MAP.get(symbol)
        if not dbn_symbol:
            raise ValueError(
                f"Unknown symbol: {symbol!r}. "
                f"Supported: {sorted(self.SYMBOL_MAP)}"
            )

        data = self._client.timeseries.get_range(
            dataset=self.DATASET,
            symbols=[dbn_symbol],
            schema="ohlcv-1d",
            start=start.isoformat(),
            end=end.isoformat(),
        )

        df = data.to_df()
        if df.empty:
            return []

        bars: list[DailyBar] = []
        for idx, row in df.iterrows():
            o, h, l, c = (
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
            )
            # Auto-detect fixed-point pricing and convert if needed
            if abs(o) > self._PRICE_SANITY_THRESHOLD:
                o /= 1e9
                h /= 1e9
                l /= 1e9
                c /= 1e9

            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            bar_date = ts.date() if isinstance(ts, datetime) else ts

            bars.append(
                DailyBar(
                    symbol=symbol,
                    date=bar_date,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    adj_close=c,  # Futures have no corporate actions
                    volume=int(row["volume"]),
                    vwap=c,  # VWAP not provided in OHLCV schema; default to close
                )
            )

        bars.sort(key=lambda b: b.date)
        return bars

    def get_symbols(self, universe: str) -> list[str]:
        """Return supported futures symbols.

        The *universe* argument is accepted for interface compatibility but
        is ignored — we always return all mapped CME Globex symbols.
        """
        return sorted(self.SYMBOL_MAP.keys())

    # ------------------------------------------------------------------
    # Intraday bars (not part of the DataFeed ABC)
    # ------------------------------------------------------------------

    def get_intraday_bars(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> list[IntradayBar]:
        """Fetch intraday bars from Databento.

        Parameters
        ----------
        symbol:
            Internal symbol (e.g. ``'ES'``).
        timeframe:
            Bar interval — ``'1m'`` or ``'5m'``.
        start:
            First date of the range, inclusive.
        end:
            Last date of the range, inclusive.

        Returns
        -------
        list[IntradayBar]
            Bars sorted by timestamp ascending.
        """
        dbn_symbol = self.SYMBOL_MAP.get(symbol)
        if not dbn_symbol:
            raise ValueError(
                f"Unknown symbol: {symbol!r}. "
                f"Supported: {sorted(self.SYMBOL_MAP)}"
            )

        schema = self.TIMEFRAME_TO_SCHEMA.get(timeframe)
        if not schema:
            raise ValueError(
                f"Unknown timeframe: {timeframe!r}. "
                f"Supported: {sorted(self.TIMEFRAME_TO_SCHEMA)}"
            )

        data = self._client.timeseries.get_range(
            dataset=self.DATASET,
            symbols=[dbn_symbol],
            schema=schema,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        df = data.to_df()
        if df.empty:
            return []

        bars: list[IntradayBar] = []
        for idx, row in df.iterrows():
            o, h, l, c = (
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
            )
            # Auto-detect fixed-point pricing and convert if needed
            if abs(o) > self._PRICE_SANITY_THRESHOLD:
                o /= 1e9
                h /= 1e9
                l /= 1e9
                c /= 1e9

            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            if not isinstance(ts, datetime):
                ts = datetime.combine(ts, datetime.min.time(), tzinfo=timezone.utc)

            bars.append(
                IntradayBar(
                    symbol=symbol,
                    timestamp=ts,
                    timeframe=timeframe,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=int(row["volume"]),
                    vwap=0.0,  # OHLCV schema does not include VWAP
                )
            )

        bars.sort(key=lambda b: b.timestamp)
        return bars
