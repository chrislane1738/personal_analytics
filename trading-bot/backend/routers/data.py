"""Data management endpoints: symbols list, fetch, validate, quality."""

from __future__ import annotations

import logging
import os
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.dependencies import get_database
from backend.schemas import (
    DataFetchRequest,
    DataFetchResponse,
    DataQualityLogEntry,
    DataQualityResponse,
    DataValidateRequest,
    DataValidateResponse,
    SymbolInfoResponse,
    SymbolListResponse,
)
from data.storage.database import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


# ---------------------------------------------------------------------------
# GET /api/data/symbols — list all cached symbols with bar counts + ranges
# ---------------------------------------------------------------------------


@router.get("/symbols", response_model=SymbolListResponse)
def list_symbols(
    db: Database = Depends(get_database),
) -> SymbolListResponse:
    """List all symbols from symbol_metadata with bar counts and date ranges."""
    sql = """
    SELECT
        sm.symbol,
        sm.company_name,
        sm.sector,
        sm.industry,
        sm.exchange,
        sm.market_cap,
        sm.updated_at,
        COALESCE(bar_stats.bar_count, 0)  AS bar_count,
        bar_stats.min_date,
        bar_stats.max_date
    FROM symbol_metadata sm
    LEFT JOIN (
        SELECT
            symbol,
            COUNT(*)   AS bar_count,
            MIN(date)  AS min_date,
            MAX(date)  AS max_date
        FROM daily_bars
        GROUP BY symbol
    ) bar_stats ON sm.symbol = bar_stats.symbol
    ORDER BY sm.symbol ASC
    """
    rows = db.execute(sql).fetchall()

    symbols = []
    total_bars = 0
    for row in rows:
        bar_count = row["bar_count"]
        total_bars += bar_count

        min_date = None
        max_date = None
        if row["min_date"]:
            try:
                min_date = date.fromisoformat(row["min_date"]) if isinstance(row["min_date"], str) else row["min_date"]
            except (ValueError, TypeError):
                pass
        if row["max_date"]:
            try:
                max_date = date.fromisoformat(row["max_date"]) if isinstance(row["max_date"], str) else row["max_date"]
            except (ValueError, TypeError):
                pass

        updated_at = None
        if row["updated_at"]:
            from datetime import datetime
            try:
                updated_at = datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"]
            except (ValueError, TypeError):
                pass

        symbols.append(
            SymbolInfoResponse(
                symbol=row["symbol"],
                company_name=row["company_name"],
                sector=row["sector"],
                industry=row["industry"],
                exchange=row["exchange"],
                market_cap=row["market_cap"],
                bar_count=bar_count,
                min_date=min_date,
                max_date=max_date,
                updated_at=updated_at,
            )
        )

    # Also include symbols that have bars but no metadata entry
    bars_only_sql = """
    SELECT
        db.symbol,
        COUNT(*)   AS bar_count,
        MIN(db.date) AS min_date,
        MAX(db.date) AS max_date
    FROM daily_bars db
    LEFT JOIN symbol_metadata sm ON db.symbol = sm.symbol
    WHERE sm.symbol IS NULL
    GROUP BY db.symbol
    ORDER BY db.symbol ASC
    """
    bars_only_rows = db.execute(bars_only_sql).fetchall()
    for row in bars_only_rows:
        bar_count = row["bar_count"]
        total_bars += bar_count

        min_date = None
        max_date = None
        if row["min_date"]:
            try:
                min_date = date.fromisoformat(row["min_date"]) if isinstance(row["min_date"], str) else row["min_date"]
            except (ValueError, TypeError):
                pass
        if row["max_date"]:
            try:
                max_date = date.fromisoformat(row["max_date"]) if isinstance(row["max_date"], str) else row["max_date"]
            except (ValueError, TypeError):
                pass

        symbols.append(
            SymbolInfoResponse(
                symbol=row["symbol"],
                bar_count=bar_count,
                min_date=min_date,
                max_date=max_date,
            )
        )

    return SymbolListResponse(
        symbols=symbols,
        total_symbols=len(symbols),
        total_bars=total_bars,
    )


# ---------------------------------------------------------------------------
# POST /api/data/fetch — fetch and cache bars for a symbol
# ---------------------------------------------------------------------------


@router.post("/fetch", response_model=DataFetchResponse)
def fetch_data(
    req: DataFetchRequest,
    db: Database = Depends(get_database),
) -> DataFetchResponse:
    """Fetch daily bars from FMP and cache them in the database."""
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="FMP_API_KEY environment variable is not set. "
            "Please configure it to enable data fetching.",
        )

    from data.feeds.fmp_feed import FMPFeed
    from data.storage.cache import DataCache

    try:
        feed = FMPFeed(api_key=api_key)
        cache = DataCache(database=db, feed=feed)

        bars = cache.get_daily_bars(req.symbol, req.start_date, req.end_date)

        # Also try to fetch and store metadata if not present
        existing_meta = db.get_symbol_metadata(req.symbol)
        if existing_meta is None:
            try:
                meta = feed.get_symbol_profile(req.symbol)
                db.insert_symbol_metadata(meta)
            except Exception:
                logger.warning(
                    "Failed to fetch metadata for %s, continuing without it",
                    req.symbol,
                )

        feed.close()

        return DataFetchResponse(
            symbol=req.symbol,
            bars_fetched=len(bars),
            message=f"Successfully fetched {len(bars)} bars for {req.symbol}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching data for %s", req.symbol)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /api/data/validate — validate symbols
# ---------------------------------------------------------------------------


@router.post("/validate", response_model=DataValidateResponse)
def validate_data(
    req: DataValidateRequest,
    db: Database = Depends(get_database),
) -> DataValidateResponse:
    """Validate data quality for the given symbols.

    Checks for missing bars, stale data, and quality issues.
    """
    results: dict[str, str] = {}

    for symbol in req.symbols:
        cached_min, cached_max = db.get_cached_date_range(symbol)
        if cached_min is None:
            results[symbol] = "no_data"
            continue

        # Check for quality issues logged
        quality_sql = """
        SELECT COUNT(*) FROM data_quality_log
        WHERE symbol = ? AND resolved = FALSE
        """
        unresolved = db.execute(quality_sql, (symbol,)).fetchone()[0]

        # Count total bars
        bar_sql = "SELECT COUNT(*) FROM daily_bars WHERE symbol = ?"
        bar_count = db.execute(bar_sql, (symbol,)).fetchone()[0]

        if unresolved > 0:
            results[symbol] = f"issues:{unresolved}"
        elif bar_count > 0:
            results[symbol] = "ok"
        else:
            results[symbol] = "no_data"

    return DataValidateResponse(results=results)


# ---------------------------------------------------------------------------
# GET /api/data/quality — query data quality log for a symbol
# ---------------------------------------------------------------------------


@router.get("/quality", response_model=DataQualityResponse)
def get_data_quality(
    symbol: str = Query(..., description="Symbol to check quality for"),
    db: Database = Depends(get_database),
) -> DataQualityResponse:
    """Query the data_quality_log for a specific symbol."""
    sql = """
    SELECT symbol, date, issue_type, severity, details, resolved
    FROM data_quality_log
    WHERE symbol = ?
    ORDER BY date DESC
    """
    rows = db.execute(sql, (symbol,)).fetchall()

    entries = []
    for row in rows:
        dt = row["date"]
        if isinstance(dt, str):
            try:
                dt = date.fromisoformat(dt)
            except (ValueError, TypeError):
                pass

        entries.append(
            DataQualityLogEntry(
                symbol=row["symbol"],
                date=dt,
                issue_type=row["issue_type"],
                severity=row["severity"],
                details=row["details"],
                resolved=bool(row["resolved"]),
            )
        )

    return DataQualityResponse(entries=entries, total=len(entries))
