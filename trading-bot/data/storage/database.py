"""SQLite database management for the trading bot backtesting framework."""
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from data.storage.models import (
    DailyBar,
    EquityCurvePoint,
    EvalCampaignRecord,
    RunRecord,
    SymbolMetadata,
    TradeRecord,
    WalkForwardStudy,
)


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
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create all 9 schema tables and their indexes if they do not exist."""
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

        CREATE TABLE IF NOT EXISTS equity_curves (
            run_id          TEXT    NOT NULL REFERENCES runs(run_id),
            date            DATE    NOT NULL,
            strategy_value  REAL    NOT NULL,
            benchmark_value REAL    NOT NULL,
            PRIMARY KEY (run_id, date)
        );

        CREATE INDEX IF NOT EXISTS idx_equity_curves_run_id
            ON equity_curves (run_id);

        CREATE TABLE IF NOT EXISTS walk_forward_studies (
            study_id        TEXT    PRIMARY KEY,
            strategy_name   TEXT    NOT NULL,
            config          TEXT    NOT NULL,
            start_date      DATE    NOT NULL,
            end_date        DATE    NOT NULL,
            initial_capital REAL    NOT NULL DEFAULT 100000.0,
            train_months    INTEGER NOT NULL DEFAULT 24,
            oos_months      INTEGER NOT NULL DEFAULT 6,
            step_months     INTEGER NOT NULL DEFAULT 3,
            objective       TEXT    NOT NULL DEFAULT 'sharpe',
            status          TEXT    NOT NULL DEFAULT 'running',
            results         TEXT    DEFAULT '',
            monte_carlo     TEXT    DEFAULT '',
            created_at      TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS eval_campaigns (
            campaign_id      TEXT PRIMARY KEY,
            strategy_name    TEXT NOT NULL,
            instrument       TEXT NOT NULL,
            state_machine    BOOLEAN NOT NULL DEFAULT 1,
            topstep_config   TEXT NOT NULL,
            num_attempts     INTEGER NOT NULL,
            seed             INTEGER,
            pass_rate        REAL,
            ev_per_attempt   REAL,
            cost_to_funded   REAL,
            avg_days_to_pass REAL,
            annual_ev        REAL,
            created_at       TIMESTAMP,
            full_results     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_eval_campaigns_strategy
            ON eval_campaigns (strategy_name);
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

    _VALID_SORT_COLUMNS = frozenset({
        "created_at", "total_return", "sharpe", "max_drawdown",
        "strategy_name", "start_date", "end_date",
    })

    def list_runs(
        self,
        sort: str = "created_at",
        order: str = "desc",
        strategy_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunRecord]:
        """Return run records with sorting, filtering, and pagination.

        Args:
            sort: Column to sort by. Must be one of: created_at, total_return,
                sharpe, max_drawdown, strategy_name, start_date, end_date.
            order: Sort direction, "asc" or "desc" (case-insensitive).
            strategy_filter: If provided, only return runs with this strategy_name.
            limit: Maximum number of records to return (default 50).
            offset: Number of records to skip (default 0).

        Raises:
            ValueError: If *sort* is not a whitelisted column name.
        """
        if sort not in self._VALID_SORT_COLUMNS:
            raise ValueError(
                f"Invalid sort column: {sort!r}. "
                f"Must be one of {sorted(self._VALID_SORT_COLUMNS)}"
            )
        order = "DESC" if order.lower() == "desc" else "ASC"

        sql = """
        SELECT run_id, mode, strategy_name, config, start_date, end_date,
               initial_capital, final_value, total_return, sharpe,
               max_drawdown, created_at, full_metrics
        FROM   runs
        """
        params: list = []
        if strategy_filter is not None:
            sql += " WHERE strategy_name = ?"
            params.append(strategy_filter)

        # sort is validated against _VALID_SORT_COLUMNS whitelist above
        sql += f" ORDER BY {sort} {order}"
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
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

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated trades and equity curve points."""
        self._conn.execute("DELETE FROM equity_curves WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()

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

    def get_trades(self, run_id: str) -> list[TradeRecord]:
        """Return all trades for a given run_id."""
        sql = """
        SELECT trade_id, run_id, symbol, direction, entry_date, exit_date,
               entry_price, exit_price, quantity, pnl, pnl_pct,
               entry_reason, exit_reason, option_type, strike, expiration
        FROM   trades
        WHERE  run_id = ?
        ORDER  BY entry_date ASC
        """
        rows = self._conn.execute(sql, (run_id,)).fetchall()
        return [
            TradeRecord(
                trade_id=row["trade_id"],
                run_id=row["run_id"],
                symbol=row["symbol"],
                direction=row["direction"],
                entry_date=self._to_date(row["entry_date"]),
                exit_date=self._to_date(row["exit_date"]),
                entry_price=row["entry_price"],
                exit_price=row["exit_price"],
                quantity=row["quantity"],
                pnl=row["pnl"],
                pnl_pct=row["pnl_pct"],
                entry_reason=row["entry_reason"] or "",
                exit_reason=row["exit_reason"] or "",
                option_type=row["option_type"],
                strike=row["strike"],
                expiration=self._to_date(row["expiration"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # equity_curves
    # ------------------------------------------------------------------

    def insert_equity_curve(self, points: list[EquityCurvePoint]) -> None:
        """Bulk insert equity curve points."""
        sql = """
        INSERT OR REPLACE INTO equity_curves
            (run_id, date, strategy_value, benchmark_value)
        VALUES (?, ?, ?, ?)
        """
        params = [
            (
                p.run_id,
                p.date.isoformat(),
                p.strategy_value,
                p.benchmark_value,
            )
            for p in points
        ]
        self._conn.executemany(sql, params)
        self._conn.commit()

    def get_equity_curve(self, run_id: str) -> list[EquityCurvePoint]:
        """Return equity curve points for a given run_id, ordered by date."""
        sql = """
        SELECT run_id, date, strategy_value, benchmark_value
        FROM   equity_curves
        WHERE  run_id = ?
        ORDER  BY date ASC
        """
        rows = self._conn.execute(sql, (run_id,)).fetchall()
        return [
            EquityCurvePoint(
                run_id=row["run_id"],
                date=self._to_date(row["date"]),
                strategy_value=row["strategy_value"],
                benchmark_value=row["benchmark_value"],
            )
            for row in rows
        ]

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
    # walk_forward_studies
    # ------------------------------------------------------------------

    def insert_walk_forward_study(self, study: WalkForwardStudy) -> None:
        """Insert a walk-forward study record."""
        sql = """
        INSERT OR REPLACE INTO walk_forward_studies
            (study_id, strategy_name, config, start_date, end_date,
             initial_capital, train_months, oos_months, step_months,
             objective, status, results, monte_carlo, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._conn.execute(
            sql,
            (
                study.study_id,
                study.strategy_name,
                study.config,
                study.start_date.isoformat() if study.start_date else None,
                study.end_date.isoformat() if study.end_date else None,
                study.initial_capital,
                study.train_months,
                study.oos_months,
                study.step_months,
                study.objective,
                study.status,
                study.results,
                study.monte_carlo,
                study.created_at.isoformat() if study.created_at else None,
            ),
        )
        self._conn.commit()

    def get_walk_forward_study(self, study_id: str) -> Optional[WalkForwardStudy]:
        """Return a WalkForwardStudy by its ID, or None if not found."""
        sql = """
        SELECT study_id, strategy_name, config, start_date, end_date,
               initial_capital, train_months, oos_months, step_months,
               objective, status, results, monte_carlo, created_at
        FROM   walk_forward_studies
        WHERE  study_id = ?
        """
        row = self._conn.execute(sql, (study_id,)).fetchone()
        if row is None:
            return None
        return WalkForwardStudy(
            study_id=row["study_id"],
            strategy_name=row["strategy_name"],
            config=row["config"],
            start_date=self._to_date(row["start_date"]),
            end_date=self._to_date(row["end_date"]),
            initial_capital=row["initial_capital"],
            train_months=row["train_months"],
            oos_months=row["oos_months"],
            step_months=row["step_months"],
            objective=row["objective"],
            status=row["status"],
            results=row["results"] or "",
            monte_carlo=row["monte_carlo"] or "",
            created_at=self._to_datetime(row["created_at"]),
        )

    _VALID_WF_SORT_COLUMNS = frozenset({
        "created_at", "strategy_name", "start_date", "end_date", "status",
    })

    def list_walk_forward_studies(
        self,
        sort: str = "created_at",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> list[WalkForwardStudy]:
        """Return walk-forward studies with sorting and pagination.

        Args:
            sort: Column to sort by. Must be one of: created_at, strategy_name,
                start_date, end_date, status.
            order: Sort direction, "asc" or "desc" (case-insensitive).
            limit: Maximum number of records to return (default 50).
            offset: Number of records to skip (default 0).

        Raises:
            ValueError: If *sort* is not a whitelisted column name.
        """
        if sort not in self._VALID_WF_SORT_COLUMNS:
            raise ValueError(
                f"Invalid sort column: {sort!r}. "
                f"Must be one of {sorted(self._VALID_WF_SORT_COLUMNS)}"
            )
        order = "DESC" if order.lower() == "desc" else "ASC"

        sql = """
        SELECT study_id, strategy_name, config, start_date, end_date,
               initial_capital, train_months, oos_months, step_months,
               objective, status, results, monte_carlo, created_at
        FROM   walk_forward_studies
        """
        # sort is validated against _VALID_WF_SORT_COLUMNS whitelist above
        sql += f" ORDER BY {sort} {order}"
        sql += " LIMIT ? OFFSET ?"
        params: list = [limit, offset]

        rows = self._conn.execute(sql, params).fetchall()
        return [
            WalkForwardStudy(
                study_id=row["study_id"],
                strategy_name=row["strategy_name"],
                config=row["config"],
                start_date=self._to_date(row["start_date"]),
                end_date=self._to_date(row["end_date"]),
                initial_capital=row["initial_capital"],
                train_months=row["train_months"],
                oos_months=row["oos_months"],
                step_months=row["step_months"],
                objective=row["objective"],
                status=row["status"],
                results=row["results"] or "",
                monte_carlo=row["monte_carlo"] or "",
                created_at=self._to_datetime(row["created_at"]),
            )
            for row in rows
        ]

    def count_walk_forward_studies(self) -> int:
        """Return the total number of walk-forward studies."""
        sql = "SELECT COUNT(*) FROM walk_forward_studies"
        row = self._conn.execute(sql).fetchone()
        return row[0] if row else 0

    def update_walk_forward_study(self, study_id: str, **kwargs) -> None:
        """Update specific fields of a walk-forward study.

        Allowed fields: status, results, monte_carlo.
        """
        allowed = {"status", "results", "monte_carlo"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return

        set_clause = ", ".join(f"{col} = ?" for col in updates)
        values = list(updates.values()) + [study_id]

        sql = f"UPDATE walk_forward_studies SET {set_clause} WHERE study_id = ?"
        self._conn.execute(sql, values)
        self._conn.commit()

    def delete_walk_forward_study(self, study_id: str) -> None:
        """Delete a walk-forward study."""
        self._conn.execute(
            "DELETE FROM walk_forward_studies WHERE study_id = ?", (study_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # eval_campaigns
    # ------------------------------------------------------------------

    def insert_eval_campaign(self, record: EvalCampaignRecord) -> None:
        """INSERT OR REPLACE an eval campaign record."""
        sql = """
        INSERT OR REPLACE INTO eval_campaigns
            (campaign_id, strategy_name, instrument, state_machine,
             topstep_config, num_attempts, seed, pass_rate, ev_per_attempt,
             cost_to_funded, avg_days_to_pass, annual_ev, created_at,
             full_results)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._conn.execute(
            sql,
            (
                record.campaign_id,
                record.strategy_name,
                record.instrument,
                int(record.state_machine),
                record.topstep_config,
                record.num_attempts,
                record.seed,
                record.pass_rate,
                record.ev_per_attempt,
                record.cost_to_funded,
                record.avg_days_to_pass,
                record.annual_ev,
                record.created_at.isoformat() if record.created_at else None,
                record.full_results,
            ),
        )
        self._conn.commit()

    def get_eval_campaign(self, campaign_id: str) -> Optional[EvalCampaignRecord]:
        """Return an EvalCampaignRecord by its ID, or None if not found."""
        sql = """
        SELECT campaign_id, strategy_name, instrument, state_machine,
               topstep_config, num_attempts, seed, pass_rate, ev_per_attempt,
               cost_to_funded, avg_days_to_pass, annual_ev, created_at,
               full_results
        FROM   eval_campaigns
        WHERE  campaign_id = ?
        """
        row = self._conn.execute(sql, (campaign_id,)).fetchone()
        if row is None:
            return None
        return EvalCampaignRecord(
            campaign_id=row["campaign_id"],
            strategy_name=row["strategy_name"],
            instrument=row["instrument"],
            state_machine=bool(row["state_machine"]),
            topstep_config=row["topstep_config"],
            num_attempts=row["num_attempts"],
            seed=row["seed"],
            pass_rate=row["pass_rate"] or 0.0,
            ev_per_attempt=row["ev_per_attempt"] or 0.0,
            cost_to_funded=row["cost_to_funded"] or 0.0,
            avg_days_to_pass=row["avg_days_to_pass"] or 0.0,
            annual_ev=row["annual_ev"] or 0.0,
            created_at=self._to_datetime(row["created_at"]),
            full_results=row["full_results"] or "",
        )

    _VALID_EVAL_SORT_COLUMNS = frozenset({
        "created_at", "pass_rate", "ev_per_attempt", "strategy_name", "annual_ev",
    })

    def list_eval_campaigns(
        self,
        sort: str = "created_at",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> list[EvalCampaignRecord]:
        """Return eval campaign records with sorting and pagination.

        Args:
            sort: Column to sort by. Must be one of: created_at, pass_rate,
                ev_per_attempt, strategy_name, annual_ev.
            order: Sort direction, "asc" or "desc" (case-insensitive).
            limit: Maximum number of records to return (default 50).
            offset: Number of records to skip (default 0).

        Raises:
            ValueError: If *sort* is not a whitelisted column name.
        """
        if sort not in self._VALID_EVAL_SORT_COLUMNS:
            raise ValueError(
                f"Invalid sort column: {sort!r}. "
                f"Must be one of {sorted(self._VALID_EVAL_SORT_COLUMNS)}"
            )
        order = "DESC" if order.lower() == "desc" else "ASC"

        sql = """
        SELECT campaign_id, strategy_name, instrument, state_machine,
               topstep_config, num_attempts, seed, pass_rate, ev_per_attempt,
               cost_to_funded, avg_days_to_pass, annual_ev, created_at,
               full_results
        FROM   eval_campaigns
        """
        # sort is validated against _VALID_EVAL_SORT_COLUMNS whitelist above
        sql += f" ORDER BY {sort} {order}"
        sql += " LIMIT ? OFFSET ?"
        params: list = [limit, offset]

        rows = self._conn.execute(sql, params).fetchall()
        return [
            EvalCampaignRecord(
                campaign_id=row["campaign_id"],
                strategy_name=row["strategy_name"],
                instrument=row["instrument"],
                state_machine=bool(row["state_machine"]),
                topstep_config=row["topstep_config"],
                num_attempts=row["num_attempts"],
                seed=row["seed"],
                pass_rate=row["pass_rate"] or 0.0,
                ev_per_attempt=row["ev_per_attempt"] or 0.0,
                cost_to_funded=row["cost_to_funded"] or 0.0,
                avg_days_to_pass=row["avg_days_to_pass"] or 0.0,
                annual_ev=row["annual_ev"] or 0.0,
                created_at=self._to_datetime(row["created_at"]),
                full_results=row["full_results"] or "",
            )
            for row in rows
        ]

    def delete_eval_campaign(self, campaign_id: str) -> None:
        """Delete an eval campaign record."""
        self._conn.execute(
            "DELETE FROM eval_campaigns WHERE campaign_id = ?", (campaign_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Generic / housekeeping
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()):
        """Execute arbitrary SQL and return the cursor (for test use)."""
        return self._conn.execute(sql, params)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
