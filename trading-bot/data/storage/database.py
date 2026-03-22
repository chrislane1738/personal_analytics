"""SQLite database management for the trading bot backtesting framework."""
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from data.storage.models import DailyBar, RunRecord, SymbolMetadata, TradeRecord


class Database:
    """Manages the SQLite database for all trading bot persistence needs.

    Uses WAL journal mode and enforces foreign key constraints.
    """

    def __init__(self, db_path: str) -> None:
        """Initialise the database connection.

        Creates all parent directories if they do not exist, then opens (or
        creates) the SQLite file and enables WAL mode and foreign-key support.
        """
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Do not use detect_types: the built-in TIMESTAMP converter was
        # deprecated in Python 3.12 and no longer works in 3.14 for ISO-8601
        # strings that use a 'T' separator.  We handle type conversion
        # explicitly in _to_date / _to_datetime instead.
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create all 8 schema tables and their indexes if they do not exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol              TEXT    NOT NULL,
            date                DATE    NOT NULL,
            open                REAL    NOT NULL,
            high                REAL    NOT NULL,
            low                 REAL    NOT NULL,
            close               REAL    NOT NULL,
            adj_close           REAL    NOT NULL,
            volume              INTEGER NOT NULL,
            vwap                REAL    NOT NULL,
            data_quality_score  REAL    NOT NULL DEFAULT 1.0,
            PRIMARY KEY (symbol, date)
        );

        CREATE TABLE IF NOT EXISTS options_chains (
            symbol              TEXT    NOT NULL,
            date                DATE    NOT NULL,
            expiration          DATE    NOT NULL,
            strike              REAL    NOT NULL,
            option_type         TEXT    NOT NULL,
            last_price          REAL,
            bid                 REAL,
            ask                 REAL,
            volume              INTEGER,
            open_interest       INTEGER,
            implied_volatility  REAL,
            delta               REAL,
            gamma               REAL,
            theta               REAL,
            vega                REAL,
            PRIMARY KEY (symbol, date, expiration, strike, option_type)
        );

        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol          TEXT    NOT NULL,
            date            DATE    NOT NULL,
            period          TEXT    NOT NULL,
            revenue         REAL,
            net_income      REAL,
            eps             REAL,
            pe_ratio        REAL,
            debt_to_equity  REAL,
            free_cash_flow  REAL,
            roe             REAL,
            raw_json        TEXT,
            PRIMARY KEY (symbol, date, period)
        );

        CREATE TABLE IF NOT EXISTS indicators_cache (
            symbol          TEXT    NOT NULL,
            date            DATE    NOT NULL,
            indicator_name  TEXT    NOT NULL,
            value           REAL,
            params          TEXT    NOT NULL DEFAULT '',
            PRIMARY KEY (symbol, date, indicator_name, params)
        );

        CREATE TABLE IF NOT EXISTS symbol_metadata (
            symbol          TEXT    PRIMARY KEY,
            company_name    TEXT,
            sector          TEXT,
            industry        TEXT,
            exchange        TEXT,
            market_cap      REAL,
            updated_at      TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id          TEXT    PRIMARY KEY,
            mode            TEXT    NOT NULL,
            strategy_name   TEXT    NOT NULL,
            config          TEXT,
            start_date      DATE,
            end_date        DATE,
            initial_capital REAL,
            final_value     REAL    DEFAULT 0.0,
            total_return    REAL    DEFAULT 0.0,
            sharpe          REAL    DEFAULT 0.0,
            max_drawdown    REAL    DEFAULT 0.0,
            created_at      TIMESTAMP,
            full_metrics    TEXT    DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS trades (
            trade_id        TEXT    PRIMARY KEY,
            run_id          TEXT    REFERENCES runs(run_id),
            symbol          TEXT,
            direction       TEXT,
            entry_date      DATE,
            exit_date       DATE,
            entry_price     REAL,
            exit_price      REAL,
            quantity        INTEGER,
            pnl             REAL    DEFAULT 0.0,
            pnl_pct         REAL    DEFAULT 0.0,
            entry_reason    TEXT    DEFAULT '',
            exit_reason     TEXT    DEFAULT '',
            option_type     TEXT,
            strike          REAL,
            expiration      DATE
        );

        CREATE TABLE IF NOT EXISTS data_quality_log (
            symbol      TEXT    NOT NULL,
            date        DATE    NOT NULL,
            issue_type  TEXT    NOT NULL,
            severity    TEXT,
            details     TEXT,
            resolved    BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY (symbol, date, issue_type)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_bars_symbol
            ON daily_bars (symbol);

        CREATE INDEX IF NOT EXISTS idx_daily_bars_date
            ON daily_bars (date);

        CREATE INDEX IF NOT EXISTS idx_trades_run_id
            ON trades (run_id);
        """
        self._conn.executescript(ddl)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_date(value) -> Optional[date]:
        """Convert a stored value to a date object."""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        # SQLite may return the value as a string "YYYY-MM-DD"
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @staticmethod
    def _to_datetime(value) -> Optional[datetime]:
        """Convert a stored value to a datetime object."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    # ------------------------------------------------------------------
    # daily_bars
    # ------------------------------------------------------------------

    def insert_daily_bars(self, bars: list[DailyBar]) -> None:
        """Bulk INSERT OR REPLACE daily bar records."""
        sql = """
        INSERT OR REPLACE INTO daily_bars
            (symbol, date, open, high, low, close, adj_close, volume, vwap,
             data_quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                b.symbol,
                b.date.isoformat(),
                b.open,
                b.high,
                b.low,
                b.close,
                b.adj_close,
                b.volume,
                b.vwap,
                b.data_quality_score,
            )
            for b in bars
        ]
        self._conn.executemany(sql, params)
        self._conn.commit()

    def get_daily_bars(
        self, symbol: str, start: date, end: date
    ) -> list[DailyBar]:
        """Return daily bars for *symbol* between *start* and *end* inclusive."""
        sql = """
        SELECT symbol, date, open, high, low, close, adj_close, volume, vwap,
               data_quality_score
        FROM   daily_bars
        WHERE  symbol = ?
          AND  date BETWEEN ? AND ?
        ORDER  BY date ASC
        """
        rows = self._conn.execute(
            sql, (symbol, start.isoformat(), end.isoformat())
        ).fetchall()
        return [
            DailyBar(
                symbol=row["symbol"],
                date=self._to_date(row["date"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                adj_close=row["adj_close"],
                volume=row["volume"],
                vwap=row["vwap"],
                data_quality_score=row["data_quality_score"],
            )
            for row in rows
        ]

    def get_cached_date_range(
        self, symbol: str
    ) -> tuple[Optional[date], Optional[date]]:
        """Return the (min_date, max_date) of cached bars for *symbol*.

        Returns (None, None) if no data is stored for the symbol.
        """
        sql = "SELECT MIN(date), MAX(date) FROM daily_bars WHERE symbol = ?"
        row = self._conn.execute(sql, (symbol,)).fetchone()
        if row is None or row[0] is None:
            return (None, None)
        return (self._to_date(row[0]), self._to_date(row[1]))

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------

    def insert_run(self, run: RunRecord) -> None:
        """Insert a run record."""
        sql = """
        INSERT OR REPLACE INTO runs
            (run_id, mode, strategy_name, config, start_date, end_date,
             initial_capital, final_value, total_return, sharpe, max_drawdown,
             created_at, full_metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        created_at = run.created_at.isoformat() if run.created_at else None
        self._conn.execute(
            sql,
            (
                run.run_id,
                run.mode,
                run.strategy_name,
                run.config,
                run.start_date.isoformat() if run.start_date else None,
                run.end_date.isoformat() if run.end_date else None,
                run.initial_capital,
                run.final_value,
                run.total_return,
                run.sharpe,
                run.max_drawdown,
                created_at,
                run.full_metrics,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Return a RunRecord by its ID, or None if not found."""
        sql = """
        SELECT run_id, mode, strategy_name, config, start_date, end_date,
               initial_capital, final_value, total_return, sharpe,
               max_drawdown, created_at, full_metrics
        FROM   runs
        WHERE  run_id = ?
        """
        row = self._conn.execute(sql, (run_id,)).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            mode=row["mode"],
            strategy_name=row["strategy_name"],
            config=row["config"],
            start_date=self._to_date(row["start_date"]),
            end_date=self._to_date(row["end_date"]),
            initial_capital=row["initial_capital"],
            final_value=row["final_value"],
            total_return=row["total_return"],
            sharpe=row["sharpe"],
            max_drawdown=row["max_drawdown"],
            created_at=self._to_datetime(row["created_at"]),
            full_metrics=row["full_metrics"] or "",
        )

    def list_runs(self) -> list[RunRecord]:
        """Return all run records ordered by created_at descending."""
        sql = """
        SELECT run_id, mode, strategy_name, config, start_date, end_date,
               initial_capital, final_value, total_return, sharpe,
               max_drawdown, created_at, full_metrics
        FROM   runs
        ORDER  BY created_at DESC
        """
        rows = self._conn.execute(sql).fetchall()
        return [
            RunRecord(
                run_id=row["run_id"],
                mode=row["mode"],
                strategy_name=row["strategy_name"],
                config=row["config"],
                start_date=self._to_date(row["start_date"]),
                end_date=self._to_date(row["end_date"]),
                initial_capital=row["initial_capital"],
                final_value=row["final_value"],
                total_return=row["total_return"],
                sharpe=row["sharpe"],
                max_drawdown=row["max_drawdown"],
                created_at=self._to_datetime(row["created_at"]),
                full_metrics=row["full_metrics"] or "",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # trades
    # ------------------------------------------------------------------

    def insert_trades(self, trades: list[TradeRecord]) -> None:
        """Bulk insert trade records."""
        sql = """
        INSERT OR REPLACE INTO trades
            (trade_id, run_id, symbol, direction, entry_date, exit_date,
             entry_price, exit_price, quantity, pnl, pnl_pct,
             entry_reason, exit_reason, option_type, strike, expiration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                t.trade_id,
                t.run_id,
                t.symbol,
                t.direction,
                t.entry_date.isoformat() if t.entry_date else None,
                t.exit_date.isoformat() if t.exit_date else None,
                t.entry_price,
                t.exit_price,
                t.quantity,
                t.pnl,
                t.pnl_pct,
                t.entry_reason,
                t.exit_reason,
                t.option_type,
                t.strike,
                t.expiration.isoformat() if t.expiration else None,
            )
            for t in trades
        ]
        self._conn.executemany(sql, params)
        self._conn.commit()

    # ------------------------------------------------------------------
    # symbol_metadata
    # ------------------------------------------------------------------

    def insert_symbol_metadata(self, meta: SymbolMetadata) -> None:
        """INSERT OR REPLACE a symbol metadata record."""
        sql = """
        INSERT OR REPLACE INTO symbol_metadata
            (symbol, company_name, sector, industry, exchange, market_cap,
             updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self._conn.execute(
            sql,
            (
                meta.symbol,
                meta.company_name,
                meta.sector,
                meta.industry,
                meta.exchange,
                meta.market_cap,
                meta.updated_at.isoformat() if meta.updated_at else None,
            ),
        )
        self._conn.commit()

    def get_symbol_metadata(self, symbol: str) -> Optional[SymbolMetadata]:
        """Return a SymbolMetadata record, or None if not found."""
        sql = """
        SELECT symbol, company_name, sector, industry, exchange, market_cap,
               updated_at
        FROM   symbol_metadata
        WHERE  symbol = ?
        """
        row = self._conn.execute(sql, (symbol,)).fetchone()
        if row is None:
            return None
        return SymbolMetadata(
            symbol=row["symbol"],
            company_name=row["company_name"],
            sector=row["sector"],
            industry=row["industry"],
            exchange=row["exchange"],
            market_cap=row["market_cap"],
            updated_at=self._to_datetime(row["updated_at"]),
        )

    # ------------------------------------------------------------------
    # Generic / housekeeping
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()):
        """Execute arbitrary SQL and return the cursor (for test use)."""
        return self._conn.execute(sql, params)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
