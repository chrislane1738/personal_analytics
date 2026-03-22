"""Tests for SQLite database layer."""
import os
import tempfile
from datetime import date, datetime

import pytest

from data.storage.models import (
    DailyBar,
    OptionsChain,
    RunRecord,
    SymbolMetadata,
    TradeRecord,
)
from data.storage.database import Database


@pytest.fixture
def db():
    """Provide a temporary database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    database.create_tables()
    yield database
    database.close()
    os.unlink(db_path)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_create_tables_creates_all_8_tables(db):
    """All 8 expected tables must exist after create_tables()."""
    expected = {
        "daily_bars",
        "options_chains",
        "fundamentals",
        "indicators_cache",
        "symbol_metadata",
        "runs",
        "trades",
        "data_quality_log",
    }
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'", ()
    ).fetchall()
    actual = {row[0] for row in rows}
    assert expected.issubset(actual), f"Missing tables: {expected - actual}"


# ---------------------------------------------------------------------------
# daily_bars tests
# ---------------------------------------------------------------------------

def _make_bar(symbol="AAPL", date_val=date(2024, 1, 2), close=185.0) -> DailyBar:
    return DailyBar(
        symbol=symbol,
        date=date_val,
        open=182.0,
        high=186.0,
        low=181.0,
        close=close,
        adj_close=close,
        volume=50_000_000,
        vwap=183.5,
        data_quality_score=1.0,
    )


def test_insert_and_query_daily_bars(db):
    bars = [
        _make_bar(date_val=date(2024, 1, 2), close=185.0),
        _make_bar(date_val=date(2024, 1, 3), close=186.0),
        _make_bar(date_val=date(2024, 1, 4), close=184.0),
    ]
    db.insert_daily_bars(bars)

    result = db.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 4))
    assert len(result) == 3
    assert result[0].date == date(2024, 1, 2)
    assert result[0].close == 185.0
    assert result[2].date == date(2024, 1, 4)


def test_daily_bars_ordered_by_date(db):
    bars = [
        _make_bar(date_val=date(2024, 1, 4)),
        _make_bar(date_val=date(2024, 1, 2)),
        _make_bar(date_val=date(2024, 1, 3)),
    ]
    db.insert_daily_bars(bars)
    result = db.get_daily_bars("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    dates = [r.date for r in result]
    assert dates == sorted(dates)


def test_upsert_daily_bars(db):
    """INSERT OR REPLACE should overwrite an existing (symbol, date) row."""
    original = _make_bar(close=185.0)
    db.insert_daily_bars([original])

    updated = _make_bar(close=999.0)
    db.insert_daily_bars([updated])

    result = db.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert len(result) == 1
    assert result[0].close == 999.0


def test_get_daily_bars_returns_only_requested_symbol(db):
    db.insert_daily_bars([_make_bar(symbol="AAPL")])
    db.insert_daily_bars([_make_bar(symbol="MSFT")])

    result = db.get_daily_bars("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert all(r.symbol == "AAPL" for r in result)
    assert len(result) == 1


def test_get_daily_bars_respects_date_range(db):
    bars = [_make_bar(date_val=date(2024, 1, d)) for d in range(2, 8)]
    db.insert_daily_bars(bars)

    result = db.get_daily_bars("AAPL", date(2024, 1, 3), date(2024, 1, 5))
    assert len(result) == 3
    assert result[0].date == date(2024, 1, 3)
    assert result[-1].date == date(2024, 1, 5)


# ---------------------------------------------------------------------------
# get_cached_date_range tests
# ---------------------------------------------------------------------------

def test_get_cached_date_range_returns_min_max(db):
    bars = [
        _make_bar(date_val=date(2024, 1, 2)),
        _make_bar(date_val=date(2024, 3, 15)),
        _make_bar(date_val=date(2024, 6, 1)),
    ]
    db.insert_daily_bars(bars)
    min_d, max_d = db.get_cached_date_range("AAPL")
    assert min_d == date(2024, 1, 2)
    assert max_d == date(2024, 6, 1)


def test_get_cached_date_range_empty_returns_none_none(db):
    min_d, max_d = db.get_cached_date_range("AAPL")
    assert min_d is None
    assert max_d is None


# ---------------------------------------------------------------------------
# RunRecord tests
# ---------------------------------------------------------------------------

def _make_run(run_id="run-001") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        mode="backtest",
        strategy_name="MomentumStrategy",
        config='{"lookback": 20}',
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=100_000.0,
        final_value=115_000.0,
        total_return=0.15,
        sharpe=1.42,
        max_drawdown=0.08,
        created_at=datetime(2024, 1, 15, 10, 30, 0),
        full_metrics='{"alpha": 0.05}',
    )


def test_insert_and_get_run(db):
    run = _make_run()
    db.insert_run(run)

    retrieved = db.get_run("run-001")
    assert retrieved is not None
    assert retrieved.run_id == "run-001"
    assert retrieved.strategy_name == "MomentumStrategy"
    assert retrieved.total_return == 0.15
    assert retrieved.sharpe == 1.42


def test_get_run_nonexistent_returns_none(db):
    assert db.get_run("does-not-exist") is None


def test_list_runs_ordered_by_created_at_desc(db):
    run1 = _make_run("run-001")
    run1.created_at = datetime(2024, 1, 10)
    run2 = _make_run("run-002")
    run2.created_at = datetime(2024, 1, 20)
    run3 = _make_run("run-003")
    run3.created_at = datetime(2024, 1, 15)

    db.insert_run(run1)
    db.insert_run(run2)
    db.insert_run(run3)

    runs = db.list_runs()
    assert len(runs) == 3
    assert runs[0].run_id == "run-002"
    assert runs[1].run_id == "run-003"
    assert runs[2].run_id == "run-001"


# ---------------------------------------------------------------------------
# SymbolMetadata tests
# ---------------------------------------------------------------------------

def _make_meta(symbol="AAPL") -> SymbolMetadata:
    return SymbolMetadata(
        symbol=symbol,
        company_name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        exchange="NASDAQ",
        market_cap=3_000_000_000_000.0,
        updated_at=datetime(2024, 1, 15, 12, 0, 0),
    )


def test_insert_and_get_symbol_metadata(db):
    meta = _make_meta()
    db.insert_symbol_metadata(meta)

    retrieved = db.get_symbol_metadata("AAPL")
    assert retrieved is not None
    assert retrieved.symbol == "AAPL"
    assert retrieved.company_name == "Apple Inc."
    assert retrieved.sector == "Technology"
    assert retrieved.market_cap == 3_000_000_000_000.0


def test_get_symbol_metadata_nonexistent_returns_none(db):
    assert db.get_symbol_metadata("ZZZZZ") is None


def test_upsert_symbol_metadata(db):
    meta = _make_meta()
    db.insert_symbol_metadata(meta)

    updated = _make_meta()
    updated.company_name = "Apple Incorporated"
    updated.market_cap = 3_100_000_000_000.0
    db.insert_symbol_metadata(updated)

    retrieved = db.get_symbol_metadata("AAPL")
    assert retrieved.company_name == "Apple Incorporated"
    assert retrieved.market_cap == 3_100_000_000_000.0


# ---------------------------------------------------------------------------
# TradeRecord tests
# ---------------------------------------------------------------------------

def test_insert_trades(db):
    run = _make_run()
    db.insert_run(run)

    trades = [
        TradeRecord(
            trade_id=f"trade-{i:03d}",
            run_id="run-001",
            symbol="AAPL",
            direction="long",
            entry_date=date(2023, 1, 5),
            exit_date=date(2023, 1, 10),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=0.033,
            entry_reason="momentum signal",
            exit_reason="target hit",
        )
        for i in range(3)
    ]
    # Give each trade a unique id
    trades[0].trade_id = "trade-001"
    trades[1].trade_id = "trade-002"
    trades[2].trade_id = "trade-003"

    db.insert_trades(trades)

    rows = db.execute(
        "SELECT COUNT(*) FROM trades WHERE run_id = ?", ("run-001",)
    ).fetchone()
    assert rows[0] == 3
