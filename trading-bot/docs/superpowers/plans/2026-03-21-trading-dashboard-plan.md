# Trading Bot Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-featured Next.js dashboard for the trading bot backtesting framework — 11 pages, FastAPI backend, real-time WebSocket monitoring.

**Architecture:** FastAPI backend (thin wrapper over existing Python modules) + Next.js frontend (App Router, shadcn/ui, Recharts, TanStack Query). SQLite database. WebSocket for live event streaming.

**Tech Stack:** Python 3.12+, FastAPI, uvicorn, Next.js 16, TypeScript, shadcn/ui, Tailwind CSS, Recharts, TanStack Query, TanStack Table, Monaco Editor, Geist fonts.

**Spec:** `docs/superpowers/specs/2026-03-21-trading-dashboard-design.md`

---

## File Structure

### Phase 0: Persistence Prerequisites (modify existing files)

```
core/engine.py                    — MODIFY: persist trades + equity curve, add cancel token, fix metric serialization
data/storage/database.py          — MODIFY: add equity_curves table, delete_run, list_runs pagination, get_trades
data/storage/models.py            — MODIFY: add EquityCurvePoint dataclass
analytics/monte_carlo.py          — MODIFY: return percentile bands per time step
analytics/regime.py               — MODIFY: add best/worst trade to regime stats
analytics/trade_log.py            — MODIFY: add entry_date to get_trade_dicts()
tests/test_persistence.py         — CREATE: tests for all Phase 0 changes
```

### Phase 1-4: New Files

```
backend/
├── __init__.py
├── main.py                       — FastAPI app, CORS, lifespan, router includes
├── dependencies.py               — Database singleton, shared state
├── schemas.py                    — Pydantic request/response models
├── routers/
│   ├── __init__.py
│   ├── runs.py                   — GET/DELETE /api/runs
│   ├── trades.py                 — GET /api/trades
│   ├── analytics.py              — GET /api/analytics/{id}/*
│   ├── strategies.py             — CRUD /api/strategies
│   ├── backtest.py               — POST /api/backtest/run, status, stop
│   └── data.py                   — GET/POST /api/data/*
├── services/
│   ├── __init__.py
│   ├── backtest_service.py       — Background thread runner + cancel token
│   └── ws_manager.py             — WebSocket connection + EventBus bridge
frontend/
├── package.json
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── app/
│   ├── layout.tsx                — Root layout: sidebar + event ticker
│   ├── page.tsx                  — Run Browser (home)
│   ├── runs/[runId]/page.tsx     — Run Detail
│   ├── compare/page.tsx          — Run Comparison
│   ├── strategies/page.tsx       — Strategy Editor
│   ├── launch/page.tsx           — Backtest Launcher
│   ├── monitor/page.tsx          — Live Event Monitor
│   ├── data/page.tsx             — Data Manager
│   └── paper/page.tsx            — Paper Trading placeholder
├── components/
│   ├── ui/                       — shadcn/ui components (auto-generated)
│   ├── layout/
│   │   ├── sidebar.tsx           — Collapsible sidebar navigation
│   │   └── event-ticker.tsx      — Bottom event ticker bar
│   ├── charts/
│   │   ├── equity-curve.tsx      — Recharts equity curve with benchmark
│   │   ├── drawdown-chart.tsx    — Recharts inverted area chart
│   │   ├── monthly-heatmap.tsx   — Year x Month return heatmap
│   │   ├── fan-chart.tsx         — Monte Carlo percentile bands
│   │   ├── regime-timeline.tsx   — Horizontal regime bar chart
│   │   └── chart-theme.ts        — Shared colors and styles
│   ├── tables/
│   │   ├── runs-table.tsx        — DataTable for run browser
│   │   ├── trades-table.tsx      — DataTable for trade log
│   │   └── symbols-table.tsx     — DataTable for data manager
│   ├── metrics-strip.tsx         — Reusable KPI card row
│   ├── regime-cards.tsx          — Per-regime stat cards
│   └── monaco-editor.tsx         — YAML strategy editor wrapper
├── hooks/
│   ├── use-runs.ts               — TanStack Query for runs API
│   ├── use-trades.ts             — TanStack Query for trades API
│   ├── use-analytics.ts          — TanStack Query for analytics API
│   ├── use-strategies.ts         — TanStack Query for strategies API
│   ├── use-data.ts               — TanStack Query for data API
│   ├── use-backtest.ts           — TanStack Query for backtest launch
│   └── use-websocket.ts          — WebSocket hook for live monitor
├── lib/
│   ├── api.ts                    — Fetch client for localhost:8000
│   ├── types.ts                  — TypeScript types matching backend schemas
│   ├── utils.ts                  — Formatters (currency, percent, date, color)
│   └── query-client.tsx          — TanStack QueryClientProvider wrapper
```

---

## Phase 0: Persistence Prerequisites

These tasks patch the existing Python codebase so the database has all data the dashboard needs. **Must complete before Phase 1.** Run the full 714-test suite after each task to prevent regressions.

---

### Task 0.1: Add EquityCurvePoint model and equity_curves table

**Files:**
- Modify: `data/storage/models.py:85` (append after TradeRecord)
- Modify: `data/storage/database.py:38-163` (add table to create_tables DDL)
- Modify: `data/storage/database.py:355` (append new methods)
- Test: `tests/test_persistence.py` (create)

- [ ] **Step 1: Write test for equity curve storage**

```python
# tests/test_persistence.py
import uuid
from datetime import date
from data.storage.database import Database
from data.storage.models import EquityCurvePoint

def test_insert_and_get_equity_curve(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    run_id = str(uuid.uuid4())[:8]

    points = [
        EquityCurvePoint(run_id=run_id, date=date(2024, 1, 2), strategy_value=100000.0, benchmark_value=100000.0),
        EquityCurvePoint(run_id=run_id, date=date(2024, 1, 3), strategy_value=100500.0, benchmark_value=100200.0),
    ]
    db.insert_equity_curve(points)
    result = db.get_equity_curve(run_id)

    assert len(result) == 2
    assert result[0].strategy_value == 100000.0
    assert result[1].benchmark_value == 100200.0
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_insert_and_get_equity_curve -v`
Expected: FAIL — `EquityCurvePoint` not defined

- [ ] **Step 3: Add EquityCurvePoint dataclass**

In `data/storage/models.py`, append at the end of the file (after the `TradeRecord` class):

```python
@dataclass
class EquityCurvePoint:
    run_id: str
    date: date
    strategy_value: float
    benchmark_value: float
```

- [ ] **Step 4: Add equity_curves table to DDL**

In `data/storage/database.py`, inside the `create_tables` DDL string (before the closing `"""`), add:

```sql
CREATE TABLE IF NOT EXISTS equity_curves (
    run_id          TEXT    NOT NULL REFERENCES runs(run_id),
    date            DATE    NOT NULL,
    strategy_value  REAL    NOT NULL,
    benchmark_value REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (run_id, date)
);

CREATE INDEX IF NOT EXISTS idx_equity_curves_run_id
    ON equity_curves (run_id);
```

- [ ] **Step 5: Add insert_equity_curve and get_equity_curve methods**

In `data/storage/database.py`, add after the `insert_trades` method (after line 392):

```python
# ------------------------------------------------------------------
# equity_curves
# ------------------------------------------------------------------

def insert_equity_curve(self, points: list) -> None:
    """Bulk insert equity curve data points."""
    from data.storage.models import EquityCurvePoint
    sql = """
    INSERT OR REPLACE INTO equity_curves
        (run_id, date, strategy_value, benchmark_value)
    VALUES (?, ?, ?, ?)
    """
    params = [
        (p.run_id, p.date.isoformat(), p.strategy_value, p.benchmark_value)
        for p in points
    ]
    self._conn.executemany(sql, params)
    self._conn.commit()

def get_equity_curve(self, run_id: str) -> list:
    """Return equity curve points for a run, ordered by date."""
    from data.storage.models import EquityCurvePoint
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_insert_and_get_equity_curve -v`
Expected: PASS

- [ ] **Step 7: Run full test suite for regressions**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/ -v --tb=short`
Expected: All 714+ tests pass

- [ ] **Step 8: Commit**

```bash
git add data/storage/models.py data/storage/database.py tests/test_persistence.py
git commit -m "feat: add equity_curves table and EquityCurvePoint model"
```

---

### Task 0.2: Add delete_run, list_runs with pagination, get_trades to Database

**Files:**
- Modify: `data/storage/database.py:328-355` (replace list_runs, add new methods)
- Test: `tests/test_persistence.py` (append)

- [ ] **Step 1: Write tests for delete_run, list_runs pagination, get_trades**

Append to `tests/test_persistence.py`:

```python
from data.storage.models import RunRecord, TradeRecord
from datetime import datetime, timezone

def _make_run(run_id, strategy="TestStrategy", total_return=0.1, sharpe=1.5):
    return RunRecord(
        run_id=run_id, mode="backtest", strategy_name=strategy,
        config="", start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        initial_capital=100000.0, final_value=110000.0,
        total_return=total_return, sharpe=sharpe, max_drawdown=-0.05,
        created_at=datetime.now(tz=timezone.utc), full_metrics="{}",
    )

def _make_trade(trade_id, run_id, pnl=100.0):
    return TradeRecord(
        trade_id=trade_id, run_id=run_id, symbol="SPY", direction="long",
        entry_date=date(2024, 3, 1), exit_date=date(2024, 3, 15),
        entry_price=450.0, exit_price=455.0, quantity=100,
        pnl=pnl, pnl_pct=0.011, entry_reason="test", exit_reason="test",
    )

def test_get_trades_by_run_id(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    db.insert_run(_make_run("run1"))
    db.insert_run(_make_run("run2"))
    db.insert_trades([_make_trade("t1", "run1"), _make_trade("t2", "run1"), _make_trade("t3", "run2")])

    trades = db.get_trades(run_id="run1")
    assert len(trades) == 2
    assert all(t.run_id == "run1" for t in trades)
    db.close()

def test_delete_run_cascades(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    db.insert_run(_make_run("del1"))
    db.insert_trades([_make_trade("t1", "del1")])
    from data.storage.models import EquityCurvePoint
    db.insert_equity_curve([
        EquityCurvePoint(run_id="del1", date=date(2024, 1, 2), strategy_value=100000.0, benchmark_value=100000.0),
    ])

    db.delete_run("del1")

    assert db.get_run("del1") is None
    assert db.get_trades(run_id="del1") == []
    assert db.get_equity_curve("del1") == []
    db.close()

def test_list_runs_with_pagination(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    for i in range(5):
        run = _make_run(f"r{i}", total_return=i * 0.1, sharpe=i * 0.5)
        db.insert_run(run)

    # Default: all runs
    all_runs = db.list_runs()
    assert len(all_runs) == 5

    # Pagination
    page1 = db.list_runs(limit=2, offset=0)
    assert len(page1) == 2

    page2 = db.list_runs(limit=2, offset=2)
    assert len(page2) == 2

    # Sort by sharpe desc
    sorted_runs = db.list_runs(sort="sharpe", order="desc")
    assert sorted_runs[0].sharpe >= sorted_runs[-1].sharpe

    # Filter by strategy
    db.insert_run(_make_run("special", strategy="MeanReversion"))
    filtered = db.list_runs(strategy_filter="MeanReversion")
    assert len(filtered) == 1
    assert filtered[0].strategy_name == "MeanReversion"
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py -k "test_get_trades or test_delete_run or test_list_runs_with" -v`
Expected: FAIL — methods don't exist

- [ ] **Step 3: Implement get_trades method**

In `data/storage/database.py`, after `insert_trades` method (after line 392), add:

```python
def get_trades(self, run_id: str) -> list[TradeRecord]:
    """Return all trade records for a given run_id."""
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
```

- [ ] **Step 4: Implement delete_run method**

In `data/storage/database.py`, add after `get_trades`:

```python
def delete_run(self, run_id: str) -> None:
    """Delete a run and all associated trades and equity curve data."""
    self._conn.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
    self._conn.execute("DELETE FROM equity_curves WHERE run_id = ?", (run_id,))
    self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    self._conn.commit()
```

- [ ] **Step 5: Replace list_runs with paginated version**

Replace the existing `list_runs` method at line 328-355 with:

```python
def list_runs(
    self,
    sort: str = "created_at",
    order: str = "desc",
    strategy_filter: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[RunRecord]:
    """Return run records with optional sorting, filtering, and pagination."""
    allowed_sort = {"created_at", "total_return", "sharpe", "max_drawdown", "strategy_name", "start_date", "end_date"}
    if sort not in allowed_sort:
        sort = "created_at"
    order = "DESC" if order.lower() == "desc" else "ASC"

    sql = """
    SELECT run_id, mode, strategy_name, config, start_date, end_date,
           initial_capital, final_value, total_return, sharpe,
           max_drawdown, created_at, full_metrics
    FROM   runs
    """
    params: list = []
    if strategy_filter:
        sql += " WHERE strategy_name = ?"
        params.append(strategy_filter)

    sql += f" ORDER BY {sort} {order}"

    if limit is not None:
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py -k "test_get_trades or test_delete_run or test_list_runs_with" -v`
Expected: All 3 PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add data/storage/database.py tests/test_persistence.py
git commit -m "feat: add get_trades, delete_run, list_runs pagination to Database"
```

---

### Task 0.3: Patch engine to persist trades and equity curves

**Files:**
- Modify: `core/engine.py:308-330` (_save_run method)
- Modify: `core/engine.py:200-203` (record benchmark alongside equity)
- Modify: `analytics/trade_log.py:130-141` (add entry_date to get_trade_dicts)
- Test: `tests/test_persistence.py` (append)

- [ ] **Step 1: Write test for trade persistence through engine**

Append to `tests/test_persistence.py`:

```python
def test_engine_persists_trades_and_equity_curve(tmp_path):
    """After a backtest run, trades and equity curve should be in the DB."""
    from data.storage.database import Database
    from core.engine import BacktestEngine
    from strategy.base import Strategy, BacktestContext
    from core.events import BarEvent, SignalEvent

    # Minimal strategy that generates one buy signal
    class BuyOnceStrategy(Strategy):
        def __init__(self):
            self._bought = False
        def generate_signals(self, bar, portfolio):
            if not self._bought:
                self._bought = True
                return [SignalEvent(symbol=bar.symbol, direction="long", strength=0.9, reason="test")]
            return []
        def warm_up_period(self):
            return 0

    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    # Insert minimal bar data
    from data.storage.models import DailyBar
    bars = []
    for i in range(5):
        d = date(2024, 1, 2 + i)
        bars.append(DailyBar(symbol="TEST", date=d, open=100+i, high=101+i, low=99+i, close=100+i, adj_close=100+i, volume=1000000, vwap=100+i))
    db.insert_daily_bars(bars)

    engine = BacktestEngine(
        strategy=BuyOnceStrategy(), database=db, universe=["TEST"],
        start_date=date(2024, 1, 2), end_date=date(2024, 1, 6),
        initial_capital=100000.0, benchmark_symbol="TEST",
    )
    metrics = engine.run()

    # Verify trades persisted
    trades = db.get_trades(run_id=metrics.get("_run_id", ""))
    # Engine should have set _run_id in metrics; if not we query all
    all_runs = db.list_runs()
    assert len(all_runs) >= 1
    run_id = all_runs[0].run_id
    trades = db.get_trades(run_id=run_id)
    assert len(trades) >= 1

    # Verify equity curve persisted
    curve = db.get_equity_curve(run_id)
    assert len(curve) >= 1
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_engine_persists_trades_and_equity_curve -v`
Expected: FAIL — trades and equity curve not persisted

- [ ] **Step 3: Add entry_date to get_trade_dicts**

In `analytics/trade_log.py`, modify `get_trade_dicts()` (line 130-141) to include `entry_date`:

```python
def get_trade_dicts(self) -> list[dict]:
    """Return trades as dicts with keys expected by metrics functions."""
    return [
        {
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "symbol": t.symbol,
            "holding_days": t.holding_days,
            "direction": t.direction,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
            "option_type": t.option_type,
            "strike": t.strike,
            "expiration": t.expiration,
        }
        for t in self.trades
    ]
```

- [ ] **Step 4: Patch _save_run to persist trades and equity curve**

Replace `_save_run` in `core/engine.py` (lines 308-330) with:

```python
def _save_run(self, run_id: str, metrics: dict) -> None:
    """Persist backtest results to the database."""
    import uuid as _uuid
    from data.storage.models import EquityCurvePoint, TradeRecord

    # Store config including benchmark_symbol for later retrieval
    config_data = json.dumps({
        "universe": self.universe,
        "benchmark_symbol": self.benchmark_symbol,
        "position_size_pct": self.position_size_pct,
        "slippage_pct": self.broker.slippage_pct,
        "commission_per_share": self.broker.commission_per_share,
        "strategy_class": type(self.strategy).__name__,
    })

    run = RunRecord(
        run_id=run_id,
        mode="backtest",
        strategy_name=type(self.strategy).__name__,
        config=config_data,
        start_date=self.start_date,
        end_date=self.end_date,
        initial_capital=self.initial_capital,
        final_value=metrics.get("end_value", 0),
        total_return=metrics.get("total_return", 0),
        sharpe=metrics.get("sharpe_ratio", 0),
        max_drawdown=metrics.get("max_drawdown_pct", 0),
        created_at=datetime.now(tz=timezone.utc),
        full_metrics=json.dumps(
            {k: _safe_serialize(v) for k, v in metrics.items()}
        ),
    )
    try:
        self.database.insert_run(run)
    except Exception:
        return  # Don't crash on DB save failure

    # Persist trades
    try:
        trade_records = []
        for t in self.trade_log.trades:
            trade_records.append(TradeRecord(
                trade_id=str(_uuid.uuid4())[:8],
                run_id=run_id,
                symbol=t.symbol,
                direction=t.direction,
                entry_date=t.entry_date,
                exit_date=t.exit_date,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                quantity=t.quantity,
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                entry_reason=t.entry_reason,
                exit_reason=t.exit_reason,
                option_type=t.option_type,
                strike=t.strike,
                expiration=t.expiration,
            ))
        if trade_records:
            self.database.insert_trades(trade_records)
    except Exception:
        pass

    # Persist equity curve
    try:
        benchmark_map = {d: v for d, v in self.benchmark.equity_curve}
        curve_points = []
        for d, val in self.portfolio.equity_curve:
            curve_points.append(EquityCurvePoint(
                run_id=run_id,
                date=d,
                strategy_value=val,
                benchmark_value=benchmark_map.get(d, 0.0),
            ))
        if curve_points:
            self.database.insert_equity_curve(curve_points)
    except Exception:
        pass
```

Also add a helper function at module level (before the class definition, after imports):

```python
def _safe_serialize(v):
    """Serialize a metric value preserving its type (not str-coercing)."""
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)
```

- [ ] **Step 5: Expose run_id in returned metrics**

In `core/engine.py` `run()` method, after line 227 (`metrics["total_trades"] = ...`), add:

```python
metrics["_run_id"] = run_id
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_engine_persists_trades_and_equity_curve -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add core/engine.py analytics/trade_log.py tests/test_persistence.py
git commit -m "feat: persist trades and equity curve from engine, fix metric serialization"
```

---

### Task 0.4: Extend Monte Carlo to return percentile bands per time step

**Files:**
- Modify: `analytics/monte_carlo.py:9-21` (extend MonteCarloResult)
- Modify: `analytics/monte_carlo.py:24-97` (extend run_monte_carlo)
- Test: `tests/test_persistence.py` (append)

- [ ] **Step 1: Write test for percentile bands**

Append to `tests/test_persistence.py`:

```python
from analytics.monte_carlo import run_monte_carlo

def test_monte_carlo_returns_percentile_bands():
    trade_pnls = [100, -50, 200, -30, 150, -80, 120, 50, -40, 180]
    result = run_monte_carlo(trade_pnls, initial_capital=10000, num_simulations=100, seed=42)

    # Should have percentile_bands with keys p5, p25, p50, p75, p95
    assert hasattr(result, "percentile_bands")
    assert set(result.percentile_bands.keys()) == {"p5", "p25", "p50", "p75", "p95"}

    # Each band should have len(trade_pnls) values (one per time step)
    for key, values in result.percentile_bands.items():
        assert len(values) == len(trade_pnls), f"{key} has {len(values)} values, expected {len(trade_pnls)}"

    # p5 should be <= p50 <= p95 at each step
    for i in range(len(trade_pnls)):
        assert result.percentile_bands["p5"][i] <= result.percentile_bands["p50"][i]
        assert result.percentile_bands["p50"][i] <= result.percentile_bands["p95"][i]

    # Should also have actual_curve
    assert hasattr(result, "actual_curve")
    assert len(result.actual_curve) == len(trade_pnls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_monte_carlo_returns_percentile_bands -v`
Expected: FAIL — `percentile_bands` attribute doesn't exist

- [ ] **Step 3: Extend MonteCarloResult**

In `analytics/monte_carlo.py`, add two fields to the `MonteCarloResult` dataclass (after line 21):

```python
percentile_bands: dict[str, list[float]]  # {"p5": [...], "p25": [...], ...} per time step
actual_curve: list[float]  # actual equity at each trade step
```

- [ ] **Step 4: Update run_monte_carlo to compute bands**

In `analytics/monte_carlo.py`, modify the function body. Replace lines 47-97 with:

```python
    rng = np.random.default_rng(seed)

    pnls = np.array(trade_pnls)
    num_trades = len(pnls)

    if num_trades == 0:
        return MonteCarloResult(
            simulations=num_simulations,
            median_final_equity=initial_capital,
            percentile_5=initial_capital,
            percentile_95=initial_capital,
            actual_final_equity=initial_capital,
            probability_of_ruin=0.0,
            max_drawdown_median=0.0,
            max_drawdown_95=0.0,
            is_outlier=False,
            equity_distribution=[initial_capital] * num_simulations,
            drawdown_distribution=[0.0] * num_simulations,
            percentile_bands={"p5": [], "p25": [], "p50": [], "p75": [], "p95": []},
            actual_curve=[],
        )

    # Actual equity curve for overlay
    actual_curve = (initial_capital + np.cumsum(pnls)).tolist()

    # Matrix to store all simulation equity paths: (num_simulations, num_trades)
    all_paths = np.empty((num_simulations, num_trades))
    final_equities = np.empty(num_simulations)
    max_drawdowns = np.empty(num_simulations)
    ruin_count = 0

    for sim in range(num_simulations):
        shuffled = rng.permutation(pnls)
        equity_curve = initial_capital + np.cumsum(shuffled)
        all_paths[sim] = equity_curve
        final_equities[sim] = equity_curve[-1]

        # Max drawdown
        running_max = np.maximum.accumulate(
            np.concatenate([[initial_capital], equity_curve])
        )
        drawdowns = (running_max[1:] - equity_curve) / running_max[1:]
        max_drawdowns[sim] = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        if np.any(equity_curve <= ruin_threshold):
            ruin_count += 1

    # Compute percentile bands per time step
    percentile_bands = {
        "p5": np.percentile(all_paths, 5, axis=0).tolist(),
        "p25": np.percentile(all_paths, 25, axis=0).tolist(),
        "p50": np.percentile(all_paths, 50, axis=0).tolist(),
        "p75": np.percentile(all_paths, 75, axis=0).tolist(),
        "p95": np.percentile(all_paths, 95, axis=0).tolist(),
    }

    p5 = float(np.percentile(final_equities, 5))
    p95 = float(np.percentile(final_equities, 95))
    actual_final = actual_curve[-1]

    return MonteCarloResult(
        simulations=num_simulations,
        median_final_equity=float(np.median(final_equities)),
        percentile_5=p5,
        percentile_95=p95,
        actual_final_equity=actual_final,
        probability_of_ruin=ruin_count / num_simulations,
        max_drawdown_median=float(np.median(max_drawdowns)),
        max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
        is_outlier=bool(actual_final < p5 or actual_final > p95),
        equity_distribution=final_equities.tolist(),
        drawdown_distribution=max_drawdowns.tolist(),
        percentile_bands=percentile_bands,
        actual_curve=actual_curve,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_monte_carlo_returns_percentile_bands -v`
Expected: PASS

- [ ] **Step 6: Update existing Monte Carlo tests**

The existing tests in `tests/test_monte_carlo.py` may reference `MonteCarloResult` without the new `percentile_bands` and `actual_curve` fields. Run the existing MC tests first:

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_monte_carlo.py -v --tb=short`

If any tests fail because of the new required fields, update them to validate the new fields are present and correctly typed (dict with 5 keys for percentile_bands, list for actual_curve).

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add analytics/monte_carlo.py tests/test_persistence.py tests/test_monte_carlo.py
git commit -m "feat: extend Monte Carlo to return percentile bands per time step"
```

---

### Task 0.5: Extend regime stats with best_trade and worst_trade

**Files:**
- Modify: `analytics/regime.py:199-212` (extend compute_regime_stats)
- Test: `tests/test_persistence.py` (append)

- [ ] **Step 1: Write test for best/worst trade in regime stats**

Append to `tests/test_persistence.py`:

```python
from analytics.regime import compute_regime_stats, MarketRegime

def test_regime_stats_includes_best_worst_trade():
    regimes = [
        (date(2024, 1, 2), MarketRegime.BULL),
        (date(2024, 1, 3), MarketRegime.BULL),
        (date(2024, 1, 4), MarketRegime.BEAR),
    ]
    trades = [
        {"entry_date": date(2024, 1, 2), "pnl": 500.0},
        {"entry_date": date(2024, 1, 3), "pnl": -200.0},
        {"entry_date": date(2024, 1, 4), "pnl": 300.0},
    ]
    result = compute_regime_stats(trades, regimes)

    assert result["bull"]["best_trade"] == 500.0
    assert result["bull"]["worst_trade"] == -200.0
    assert result["bear"]["best_trade"] == 300.0
    assert result["bear"]["worst_trade"] == 300.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_regime_stats_includes_best_worst_trade -v`
Expected: FAIL — `best_trade` key not in result

- [ ] **Step 3: Add best_trade and worst_trade to compute_regime_stats**

In `analytics/regime.py`, replace lines 199-212 (the result-building loop and return statement) with:

```python
    result: dict[str, dict] = {}
    for regime_name, pnls in groups.items():
        if not pnls:
            result[regime_name] = {"trades": 0}
            continue
        wins = [p for p in pnls if p > 0]
        result[regime_name] = {
            "trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "total_pnl": sum(pnls),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
        }

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_regime_stats_includes_best_worst_trade -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add analytics/regime.py tests/test_persistence.py
git commit -m "feat: add best_trade and worst_trade to regime stats"
```

---

### Task 0.6: Add cancellation token to BacktestEngine

**Files:**
- Modify: `core/engine.py:56-70` (add cancel_event to __init__)
- Modify: `core/engine.py:133` (check cancel on each date)
- Test: `tests/test_persistence.py` (append)

- [ ] **Step 1: Write test for cancellation**

Append to `tests/test_persistence.py`:

```python
import threading

def test_engine_cancellation(tmp_path):
    from core.engine import BacktestEngine
    from strategy.base import Strategy
    from core.events import SignalEvent

    class SlowStrategy(Strategy):
        def generate_signals(self, bar, portfolio):
            return []
        def warm_up_period(self):
            return 0

    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    # Insert 100 bars
    from data.storage.models import DailyBar
    bars = []
    for i in range(100):
        from datetime import timedelta
        d = date(2024, 1, 2) + timedelta(days=i)
        bars.append(DailyBar(symbol="TEST", date=d, open=100, high=101, low=99, close=100, adj_close=100, volume=1000000, vwap=100))
    db.insert_daily_bars(bars)

    cancel = threading.Event()
    engine = BacktestEngine(
        strategy=SlowStrategy(), database=db, universe=["TEST"],
        start_date=date(2024, 1, 2), end_date=date(2024, 4, 10),
        initial_capital=100000.0, benchmark_symbol="TEST",
        cancel_event=cancel,
    )

    # Cancel after a short delay
    cancel.set()
    metrics = engine.run()

    # Should have completed early — fewer equity curve points than 100
    curve = db.get_equity_curve(metrics.get("_run_id", db.list_runs()[0].run_id))
    assert len(curve) < 90  # Should have stopped well before 100 bars
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_engine_cancellation -v`
Expected: FAIL — `cancel_event` not accepted

- [ ] **Step 3: Add cancel_event parameter to __init__**

In `core/engine.py`, add `cancel_event` to the `__init__` signature (after `position_size_pct` parameter):

```python
cancel_event: threading.Event | None = None,
```

Add import at top of file: `import threading`

And in the body, after `self.benchmark = ...` (around line 88):

```python
self.cancel_event = cancel_event
```

- [ ] **Step 4: Check cancel_event in the main loop**

In `core/engine.py`, inside the main loop (at line 133, the `for i, current_date in enumerate(sorted_dates):` block), add at the very beginning of the loop body:

```python
# Check for cancellation
if self.cancel_event and self.cancel_event.is_set():
    break
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py::test_engine_cancellation -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/ -v --tb=short`
Expected: All tests pass (cancel_event defaults to None so existing tests unaffected)

- [ ] **Step 7: Commit**

```bash
git add core/engine.py tests/test_persistence.py
git commit -m "feat: add cancellation token to BacktestEngine"
```

---

## Phase 1: Foundation

Backend API + Frontend Shell + Run Browser + Run Detail. Four parallel workstreams.

---

### Task 1.1: FastAPI backend app setup

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/main.py`
- Create: `backend/dependencies.py`
- Create: `backend/schemas.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/routers/runs.py`
- Create: `backend/routers/trades.py`
- Create: `backend/routers/analytics.py`
- Create: `backend/requirements.txt`

- [ ] **Step 1: Create backend/requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0.0
```

- [ ] **Step 2: Install dependencies**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && pip install fastapi uvicorn[standard] pydantic`

- [ ] **Step 3: Create backend/__init__.py (empty)**

- [ ] **Step 4: Create backend/dependencies.py**

```python
"""Shared dependencies: database singleton, active backtest state."""
import os
from data.storage.database import Database

_db: Database | None = None

def get_database() -> Database:
    global _db
    if _db is None:
        db_path = os.environ.get("TRADING_BOT_DB", "db/trading_bot.db")
        _db = Database(db_path)
        _db.create_tables()
    return _db
```

- [ ] **Step 5: Create backend/schemas.py**

```python
"""Pydantic response/request models for the API."""
from __future__ import annotations
from datetime import date, datetime
from pydantic import BaseModel

class RunResponse(BaseModel):
    run_id: str
    mode: str
    strategy_name: str
    config: str
    start_date: date | None
    end_date: date | None
    initial_capital: float
    final_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    created_at: datetime | None
    full_metrics: dict | None = None

class RunListResponse(BaseModel):
    runs: list[RunResponse]
    total: int

class TradeResponse(BaseModel):
    trade_id: str
    run_id: str
    symbol: str
    direction: str
    entry_date: date | None
    exit_date: date | None
    entry_price: float
    exit_price: float | None
    quantity: int
    pnl: float
    pnl_pct: float
    entry_reason: str
    exit_reason: str
    option_type: str | None = None
    strike: float | None = None
    expiration: date | None = None

class EquityCurvePointResponse(BaseModel):
    date: date
    strategy_value: float
    benchmark_value: float

class EquityCurveResponse(BaseModel):
    points: list[EquityCurvePointResponse]

class RegimeStatResponse(BaseModel):
    regime: str
    trades: int
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

class MonteCarloResponse(BaseModel):
    simulations: int
    median_final_equity: float
    percentile_5: float
    percentile_95: float
    actual_final_equity: float
    probability_of_ruin: float
    max_drawdown_median: float
    max_drawdown_95: float
    is_outlier: bool
    percentile_bands: dict[str, list[float]]
    actual_curve: list[float]
    drawdown_distribution: list[float]

class OptionsAnalyticsResponse(BaseModel):
    total_premium_collected: float
    total_premium_paid: float
    net_premium: float
    assignment_count: int
    total_short_options: int
    assignment_rate: float
    win_rate_by_dte: dict[str, float]
    avg_pnl_by_dte: dict[str, float]
    greeks_timeline: list[dict] = []

class ErrorResponse(BaseModel):
    detail: str
    code: str
```

- [ ] **Step 6: Create backend/routers/__init__.py (empty)**

- [ ] **Step 7: Create backend/routers/runs.py**

```python
"""Runs API: list, get, delete backtest runs."""
from fastapi import APIRouter, HTTPException, Query
from backend.dependencies import get_database
from backend.schemas import RunResponse, RunListResponse, ErrorResponse
import json

router = APIRouter(prefix="/api/runs", tags=["runs"])

def _run_to_response(run) -> RunResponse:
    metrics = None
    if run.full_metrics:
        try:
            metrics = json.loads(run.full_metrics)
        except json.JSONDecodeError:
            metrics = {}
    return RunResponse(
        run_id=run.run_id, mode=run.mode, strategy_name=run.strategy_name,
        config=run.config, start_date=run.start_date, end_date=run.end_date,
        initial_capital=run.initial_capital, final_value=run.final_value,
        total_return=run.total_return, sharpe=run.sharpe,
        max_drawdown=run.max_drawdown, created_at=run.created_at,
        full_metrics=metrics,
    )

@router.get("", response_model=RunListResponse)
def list_runs(
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    strategy: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    db = get_database()
    runs = db.list_runs(sort=sort, order=order, strategy_filter=strategy, limit=limit, offset=offset)
    total_runs = db.list_runs(strategy_filter=strategy)
    return RunListResponse(
        runs=[_run_to_response(r) for r in runs],
        total=len(total_runs),
    )

@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str):
    db = get_database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_response(run)

@router.delete("/{run_id}")
def delete_run(run_id: str):
    db = get_database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    db.delete_run(run_id)
    return {"deleted": True}
```

- [ ] **Step 8: Create backend/routers/trades.py**

```python
"""Trades API: query trades by run_id."""
from fastapi import APIRouter, Query
from backend.dependencies import get_database
from backend.schemas import TradeResponse

router = APIRouter(prefix="/api/trades", tags=["trades"])

@router.get("", response_model=list[TradeResponse])
def list_trades(run_id: str = Query(...)):
    db = get_database()
    trades = db.get_trades(run_id=run_id)
    return [
        TradeResponse(
            trade_id=t.trade_id, run_id=t.run_id, symbol=t.symbol,
            direction=t.direction, entry_date=t.entry_date, exit_date=t.exit_date,
            entry_price=t.entry_price, exit_price=t.exit_price, quantity=t.quantity,
            pnl=t.pnl, pnl_pct=t.pnl_pct, entry_reason=t.entry_reason,
            exit_reason=t.exit_reason, option_type=t.option_type,
            strike=t.strike, expiration=t.expiration,
        )
        for t in trades
    ]
```

- [ ] **Step 9: Create backend/routers/analytics.py**

```python
"""Analytics API: equity curve, regime stats, Monte Carlo, options analytics."""
import json
from fastapi import APIRouter, HTTPException, Query
from backend.dependencies import get_database
from backend.schemas import (
    EquityCurveResponse, EquityCurvePointResponse,
    RegimeStatResponse, MonteCarloResponse, OptionsAnalyticsResponse,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/{run_id}/equity-curve", response_model=EquityCurveResponse)
def get_equity_curve(run_id: str):
    db = get_database()
    points = db.get_equity_curve(run_id)
    if not points:
        raise HTTPException(status_code=404, detail="Equity curve not found")
    return EquityCurveResponse(
        points=[EquityCurvePointResponse(date=p.date, strategy_value=p.strategy_value, benchmark_value=p.benchmark_value) for p in points]
    )

@router.get("/{run_id}/regime", response_model=list[RegimeStatResponse])
def get_regime_stats(run_id: str):
    db = get_database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Get benchmark prices for regime detection
    config = json.loads(run.config) if run.config else {}
    benchmark = config.get("benchmark_symbol", "SPY")
    benchmark_bars = db.get_daily_bars(benchmark, run.start_date, run.end_date)
    prices = [(b.date, b.close) for b in benchmark_bars]

    from analytics.regime import detect_regimes, compute_regime_stats
    regimes = detect_regimes(prices)

    trades = db.get_trades(run_id)
    trade_dicts = [{"entry_date": t.entry_date, "pnl": t.pnl} for t in trades]
    stats = compute_regime_stats(trade_dicts, regimes)

    return [
        RegimeStatResponse(
            regime=regime_name,
            trades=data.get("trades", 0),
            win_rate=data.get("win_rate", 0.0),
            avg_pnl=data.get("avg_pnl", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            best_trade=data.get("best_trade", 0.0),
            worst_trade=data.get("worst_trade", 0.0),
        )
        for regime_name, data in stats.items()
    ]

@router.get("/{run_id}/monte-carlo", response_model=MonteCarloResponse)
def get_monte_carlo(run_id: str, simulations: int = Query(10000, ge=100, le=100000)):
    db = get_database()
    trades = db.get_trades(run_id)
    if not trades:
        raise HTTPException(status_code=404, detail="No trades found for run")

    run = db.get_run(run_id)
    initial_capital = run.initial_capital if run else 100000.0
    trade_pnls = [t.pnl for t in trades]

    from analytics.monte_carlo import run_monte_carlo
    result = run_monte_carlo(trade_pnls, initial_capital=initial_capital, num_simulations=simulations, seed=42)

    return MonteCarloResponse(
        simulations=result.simulations,
        median_final_equity=result.median_final_equity,
        percentile_5=result.percentile_5,
        percentile_95=result.percentile_95,
        actual_final_equity=result.actual_final_equity,
        probability_of_ruin=result.probability_of_ruin,
        max_drawdown_median=result.max_drawdown_median,
        max_drawdown_95=result.max_drawdown_95,
        is_outlier=result.is_outlier,
        percentile_bands=result.percentile_bands,
        actual_curve=result.actual_curve,
        drawdown_distribution=result.drawdown_distribution,
    )

@router.get("/{run_id}/options", response_model=OptionsAnalyticsResponse)
def get_options_analytics(run_id: str):
    db = get_database()
    trades = db.get_trades(run_id)
    options_trades = [t for t in trades if t.option_type is not None]
    if not options_trades:
        raise HTTPException(status_code=404, detail="No options trades found")

    from analytics.options_analytics import compute_options_analytics
    trade_dicts = [
        {
            "option_type": t.option_type, "direction": t.direction,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "pnl": t.pnl, "quantity": t.quantity,
            "entry_date": t.entry_date, "exit_date": t.exit_date,
            "expiration": t.expiration, "exit_reason": t.exit_reason,
        }
        for t in options_trades
    ]
    result = compute_options_analytics(trade_dicts)

    return OptionsAnalyticsResponse(
        total_premium_collected=result.total_premium_collected,
        total_premium_paid=result.total_premium_paid,
        net_premium=result.net_premium,
        assignment_count=result.assignment_count,
        total_short_options=result.total_short_options,
        assignment_rate=result.assignment_rate,
        win_rate_by_dte=result.win_rate_by_dte,
        avg_pnl_by_dte=result.avg_pnl_by_dte,
        greeks_timeline=result.greeks_timeline,
    )
```

- [ ] **Step 10: Create backend/main.py**

```python
"""Trading Bot Dashboard — FastAPI backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import runs, trades, analytics

app = FastAPI(title="Trading Bot Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(trades.router)
app.include_router(analytics.router)

@app.get("/api/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 11: Test the API server starts**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m uvicorn backend.main:app --port 8000 &`
Then: `curl http://localhost:8000/api/health`
Expected: `{"status":"ok"}`
Then: `curl http://localhost:8000/api/runs`
Expected: JSON response with runs (may be empty list)
Kill server after testing.

- [ ] **Step 12: Commit**

```bash
git add backend/
git commit -m "feat: add FastAPI backend with runs, trades, analytics APIs"
```

---

### Task 1.2: Next.js frontend shell

**Files:**
- Create: `frontend/` (full Next.js project with shadcn/ui)

- [ ] **Step 1: Initialize Next.js project**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --use-npm`

- [ ] **Step 2: Initialize shadcn/ui**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot/frontend && npx shadcn@latest init -d`

Select: zinc base color, CSS variables for theming.

- [ ] **Step 3: Install dependencies**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot/frontend && npm install @tanstack/react-query @tanstack/react-table recharts lucide-react`

- [ ] **Step 4: Add shadcn components needed for Phase 1**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot/frontend && npx shadcn@latest add table button badge card input select tabs tooltip separator`

- [ ] **Step 5: Install Geist fonts**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot/frontend && npm install geist`

- [ ] **Step 6: Configure dark theme and Geist fonts in layout.tsx**

Replace `frontend/app/layout.tsx` with:

```tsx
import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { QueryProvider } from "@/lib/query-client";
import { Sidebar } from "@/components/layout/sidebar";

export const metadata: Metadata = {
  title: "Trading Bot Dashboard",
  description: "Backtest analysis and strategy management",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${GeistSans.variable} ${GeistMono.variable} font-sans bg-[#09090b] text-zinc-50 antialiased`}>
        <QueryProvider>
          <div className="flex h-screen">
            <Sidebar />
            <main className="flex-1 overflow-auto">{children}</main>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 7: Create lib/query-client.tsx**

```tsx
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient({
    defaultOptions: { queries: { staleTime: 30_000, refetchOnWindowFocus: false } },
  }));
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
```

- [ ] **Step 8: Create lib/api.ts**

```ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}
```

- [ ] **Step 9: Create lib/types.ts**

```ts
export interface Run {
  run_id: string;
  mode: string;
  strategy_name: string;
  config: string;
  start_date: string | null;
  end_date: string | null;
  initial_capital: number;
  final_value: number;
  total_return: number;
  sharpe: number;
  max_drawdown: number;
  created_at: string | null;
  full_metrics: Record<string, unknown> | null;
}

export interface RunListResponse {
  runs: Run[];
  total: number;
}

export interface Trade {
  trade_id: string;
  run_id: string;
  symbol: string;
  direction: string;
  entry_date: string | null;
  exit_date: string | null;
  entry_price: number;
  exit_price: number | null;
  quantity: number;
  pnl: number;
  pnl_pct: number;
  entry_reason: string;
  exit_reason: string;
  option_type: string | null;
  strike: number | null;
  expiration: string | null;
}

export interface EquityCurvePoint {
  date: string;
  strategy_value: number;
  benchmark_value: number;
}

export interface RegimeStat {
  regime: string;
  trades: number;
  win_rate: number;
  avg_pnl: number;
  total_pnl: number;
  best_trade: number;
  worst_trade: number;
}

export interface MonteCarloResult {
  simulations: number;
  median_final_equity: number;
  percentile_5: number;
  percentile_95: number;
  actual_final_equity: number;
  probability_of_ruin: number;
  max_drawdown_median: number;
  max_drawdown_95: number;
  is_outlier: boolean;
  percentile_bands: Record<string, number[]>;
  actual_curve: number[];
  drawdown_distribution: number[];
}
```

- [ ] **Step 10: Create lib/utils.ts**

```ts
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(value);
}

export function formatPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

export function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function pnlColor(value: number): string {
  if (value > 0) return "text-green-500";
  if (value < 0) return "text-red-500";
  return "text-zinc-400";
}
```

- [ ] **Step 11: Create components/layout/sidebar.tsx**

```tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, GitCompare, Settings2, Play, Activity, Database, DollarSign } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Runs", icon: BarChart3 },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/strategies", label: "Strategies", icon: Settings2 },
  { href: "/launch", label: "Launch", icon: Play },
  { href: "/monitor", label: "Monitor", icon: Activity },
  { href: "/data", label: "Data", icon: Database },
  { href: "/paper", label: "Paper", icon: DollarSign },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-14 hover:w-48 transition-all duration-200 bg-[#09090b] border-r border-[#1a1a1a] flex flex-col overflow-hidden group">
      <div className="p-3 border-b border-[#1a1a1a]">
        <span className="font-mono text-orange-500 font-bold text-sm">TB</span>
      </div>
      <nav className="flex-1 py-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2 text-sm transition-colors ${
                active
                  ? "text-zinc-50 border-l-2 border-orange-500 bg-zinc-900/50"
                  : "text-zinc-500 hover:text-zinc-300 border-l-2 border-transparent"
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">{label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
```

- [ ] **Step 12: Create hooks/use-runs.ts**

```ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Run, RunListResponse } from "@/lib/types";

export function useRuns(params?: { sort?: string; order?: string; strategy?: string; limit?: number; offset?: number }) {
  return useQuery({
    queryKey: ["runs", params],
    queryFn: () => {
      const searchParams = new URLSearchParams();
      if (params?.sort) searchParams.set("sort", params.sort);
      if (params?.order) searchParams.set("order", params.order);
      if (params?.strategy) searchParams.set("strategy", params.strategy);
      if (params?.limit) searchParams.set("limit", String(params.limit));
      if (params?.offset) searchParams.set("offset", String(params.offset));
      return apiFetch<RunListResponse>(`/api/runs?${searchParams}`);
    },
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: ["runs", runId],
    queryFn: () => apiFetch<Run>(`/api/runs/${runId}`),
    enabled: !!runId,
  });
}

export function useDeleteRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) => apiFetch(`/api/runs/${runId}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });
}
```

- [ ] **Step 13: Create a placeholder home page**

Replace `frontend/app/page.tsx`:

```tsx
export default function HomePage() {
  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold text-zinc-50">Run Browser</h1>
      <p className="text-sm text-zinc-500 mt-1">Loading...</p>
    </div>
  );
}
```

- [ ] **Step 14: Verify frontend starts**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot/frontend && npm run dev`
Expected: Compiles and serves at http://localhost:3000. Sidebar visible. Dark theme.

- [ ] **Step 15: Commit**

```bash
git add frontend/
git commit -m "feat: initialize Next.js frontend with shadcn/ui, dark theme, sidebar"
```

---

### Task 1.3: Run Browser page

**Files:**
- Create: `frontend/components/tables/runs-table.tsx`
- Modify: `frontend/app/page.tsx`
- Create: `frontend/hooks/use-trades.ts`

This task depends on Task 1.2 (frontend shell) being complete.

- [ ] **Step 1: Create hooks/use-trades.ts**

```ts
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Trade } from "@/lib/types";

export function useTrades(runId: string) {
  return useQuery({
    queryKey: ["trades", runId],
    queryFn: () => apiFetch<Trade[]>(`/api/trades?run_id=${runId}`),
    enabled: !!runId,
  });
}
```

- [ ] **Step 2: Create components/tables/runs-table.tsx**

Build a DataTable using TanStack Table with columns: run_id, strategy_name, date range, total_return, sharpe, max_drawdown, trade_count (from full_metrics), created_at. Color-code return/sharpe. Click row navigates to `/runs/[runId]`. Sortable headers. Use `@/components/ui/table` from shadcn.

The component should accept `data: Run[]` and `onDelete: (runId: string) => void` props.

Key column definitions:
- run_id: monospace, truncated, clickable link
- total_return: formatPercent, green/red
- sharpe: green if >1, red if <0
- max_drawdown: red intensity
- created_at: relative time

- [ ] **Step 3: Update app/page.tsx to wire RunBrowser**

```tsx
"use client";
import { useRuns, useDeleteRun } from "@/hooks/use-runs";
import { RunsTable } from "@/components/tables/runs-table";

export default function RunBrowserPage() {
  const { data, isLoading, error } = useRuns({ limit: 50 });
  const deleteRun = useDeleteRun();

  if (isLoading) return <div className="p-6 text-zinc-500">Loading runs...</div>;
  if (error) return <div className="p-6 text-red-500">Error: {error.message}</div>;

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h1 className="text-lg font-semibold">Run Browser</h1>
          <p className="text-sm text-zinc-500">{data?.total ?? 0} backtest runs</p>
        </div>
      </div>
      <RunsTable data={data?.runs ?? []} onDelete={(id) => deleteRun.mutate(id)} />
    </div>
  );
}
```

- [ ] **Step 4: Verify with backend running**

Start backend: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m uvicorn backend.main:app --port 8000 --reload &`
Start frontend: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot/frontend && npm run dev &`
Open http://localhost:3000 — should show runs table (may be empty if no backtests have been run with the patched engine yet).

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: add Run Browser page with sortable DataTable"
```

---

### Task 1.4: Run Detail page

**Files:**
- Create: `frontend/app/runs/[runId]/page.tsx`
- Create: `frontend/components/metrics-strip.tsx`
- Create: `frontend/components/charts/equity-curve.tsx`
- Create: `frontend/components/charts/drawdown-chart.tsx`
- Create: `frontend/components/charts/chart-theme.ts`
- Create: `frontend/components/tables/trades-table.tsx`
- Create: `frontend/components/regime-cards.tsx`
- Create: `frontend/hooks/use-analytics.ts`

This task depends on Task 1.2 (frontend shell) being complete.

- [ ] **Step 1: Create hooks/use-analytics.ts**

```ts
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { EquityCurvePoint, RegimeStat, MonteCarloResult } from "@/lib/types";

export function useEquityCurve(runId: string) {
  return useQuery({
    queryKey: ["analytics", runId, "equity-curve"],
    queryFn: () => apiFetch<{ points: EquityCurvePoint[] }>(`/api/analytics/${runId}/equity-curve`),
    enabled: !!runId,
  });
}

export function useRegimeStats(runId: string) {
  return useQuery({
    queryKey: ["analytics", runId, "regime"],
    queryFn: () => apiFetch<RegimeStat[]>(`/api/analytics/${runId}/regime`),
    enabled: !!runId,
  });
}

export function useMonteCarlo(runId: string, simulations = 10000) {
  return useQuery({
    queryKey: ["analytics", runId, "monte-carlo", simulations],
    queryFn: () => apiFetch<MonteCarloResult>(`/api/analytics/${runId}/monte-carlo?simulations=${simulations}`),
    enabled: !!runId,
  });
}
```

- [ ] **Step 2: Create components/charts/chart-theme.ts**

```ts
export const CHART_COLORS = {
  strategy: "#22c55e",
  benchmark: "#3b82f6",
  drawdown: "#ef4444",
  orange: "#f97316",
  purple: "#a855f7",
  yellow: "#eab308",
  grid: "#1a1a1a",
  text: "#71717a",
  bg: "#0f0f0f",
} as const;

export const REGIME_COLORS: Record<string, string> = {
  bull: "#22c55e",
  bear: "#ef4444",
  sideways: "#eab308",
  high_vol: "#a855f7",
};
```

- [ ] **Step 3: Create components/charts/equity-curve.tsx**

Recharts AreaChart with two Line series (strategy green, benchmark blue). Gradient fill under strategy. Dark theme axes. Tooltip shows date + both values in monospace.

- [ ] **Step 4: Create components/charts/drawdown-chart.tsx**

Recharts AreaChart inverted (red gradient fill). Shows drawdown % below zero. Tooltip with date + drawdown %.

- [ ] **Step 5: Create components/metrics-strip.tsx**

Grid of 8 KPI cards. Each card: uppercase label (Geist Sans 9px), large monospace value, color-coded by rule (green for positive returns, red for negative drawdown, etc.).

- [ ] **Step 6: Create components/tables/trades-table.tsx**

DataTable using TanStack Table. Columns: date, direction (colored), symbol, entry_price, qty, exit_price, P&L ($, colored), P&L (%, colored), holding_days, entry_reason, exit_reason. Sortable. Option columns shown conditionally.

- [ ] **Step 7: Create components/regime-cards.tsx**

4 cards with regime-colored left border. Each shows: regime name + dot, trade count, win rate, avg P&L, best/worst trade. Uses REGIME_COLORS from chart-theme.

- [ ] **Step 8: Create app/runs/[runId]/page.tsx**

Wire all components together matching the approved mockup layout:
- MetricsStrip at top (8 KPIs from run.full_metrics)
- 2-column grid: EquityCurve (left) + RegimeCards (right)
- 2-column grid: TradesTable (left) + DrawdownChart + risk metrics (right)
- Tabs below for Monthly Heatmap / Monte Carlo / Options (Phase 2)

- [ ] **Step 9: Verify with backend**

Navigate to http://localhost:3000/runs/{some-run-id} — should render the full detail page.

- [ ] **Step 10: Commit**

```bash
git add frontend/
git commit -m "feat: add Run Detail page with equity curve, trades, metrics, drawdown"
```

---

## Phase 2: Deep Analytics

Four independent pages — can be built in parallel. Each depends on Phase 1 being complete.

---

### Task 2.1: Run Comparison page

**Files:**
- Create: `frontend/app/compare/page.tsx`

Build the comparison page with multi-select dropdown (populated from useRuns), overlaid equity curves (normalized to % return), and side-by-side metrics table. Reference spec section 4.3.

- [ ] Steps: Create multi-select component, normalized equity chart, metrics comparison table, wire to API.
- [ ] Commit: `git commit -m "feat: add Run Comparison page"`

---

### Task 2.2: Monthly Returns Heatmap tab

**Files:**
- Create: `frontend/components/charts/monthly-heatmap.tsx`
- Modify: `frontend/app/runs/[runId]/page.tsx` (add tab)

Build Year x Month grid with cells colored by monthly return %. Use Recharts or custom div grid. Color scale: red (negative) → white (zero) → green (positive). Reference spec section 4.2 (Monthly Returns Heatmap tab).

- [ ] Steps: Compute monthly returns from equity curve data, render heatmap grid, add as tab in Run Detail.
- [ ] Commit: `git commit -m "feat: add monthly returns heatmap tab"`

---

### Task 2.3: Monte Carlo tab

**Files:**
- Create: `frontend/components/charts/fan-chart.tsx`
- Modify: `frontend/app/runs/[runId]/page.tsx` (add tab)

Build fan chart with stacked area bands (P5-P25, P25-P50, P50-P75, P75-P95). Overlay actual equity curve. Stats panel. Drawdown distribution histogram. Reference spec section 4.5.

- [ ] Steps: Create FanChart component, stats panel, drawdown histogram, add as tab.
- [ ] Commit: `git commit -m "feat: add Monte Carlo fan chart tab"`

---

### Task 2.4: Options Analytics tab

**Files:**
- Create: `frontend/hooks/use-options-analytics.ts`
- Modify: `frontend/app/runs/[runId]/page.tsx` (add conditional tab)

Build premium summary cards, assignment metrics, win rate by DTE bar chart, Greeks timeline (if data available). Only show tab when options trades exist. Reference spec section 4.6.

- [ ] Steps: Create options analytics hook, premium cards, DTE chart, Greeks timeline, conditional tab.
- [ ] Commit: `git commit -m "feat: add Options Analytics tab"`

---

## Phase 3: Control Center

Three workstreams with some backend dependencies.

---

### Task 3.1: Strategy Config Editor

**Files:**
- Create: `backend/routers/strategies.py`
- Modify: `backend/main.py` (include router)
- Create: `frontend/app/strategies/page.tsx`
- Create: `frontend/components/monaco-editor.tsx`
- Create: `frontend/hooks/use-strategies.ts`

Backend: CRUD for YAML files in `config/strategies/`. Frontend: Monaco editor with YAML highlighting, file list, save/delete/create. Reference spec section 4.7.

- [ ] Steps: Backend routes, frontend Monaco wrapper, strategy list panel, template picker, live validation.
- [ ] Commit: `git commit -m "feat: add Strategy Config Editor with Monaco YAML editing"`

---

### Task 3.2: Backtest Launcher

**Files:**
- Create: `backend/routers/backtest.py`
- Create: `backend/services/backtest_service.py`
- Modify: `backend/main.py` (include router)
- Create: `frontend/app/launch/page.tsx`
- Create: `frontend/hooks/use-backtest.ts`

Backend: POST /api/backtest/run runs engine in background thread with cancel_event. Returns run_id. GET status endpoint. Frontend: form with strategy picker, universe multi-select, date range, capital, etc. Reference spec section 4.8.

- [ ] Steps: Backend service with threading, routes, frontend form, launch flow with redirect to monitor.
- [ ] Commit: `git commit -m "feat: add Backtest Launcher with background execution"`

---

### Task 3.3: Live Event Monitor

**Files:**
- Create: `backend/services/ws_manager.py`
- Modify: `backend/main.py` (add WebSocket route)
- Create: `frontend/app/monitor/page.tsx`
- Create: `frontend/hooks/use-websocket.ts`

Backend: WebSocket endpoint that bridges EventBus → JSON messages. Heartbeat, throttling, subscribe/stop protocol. Frontend: live equity chart, portfolio state panel, scrolling event feed, progress bar. Reference spec section 4.9.

- [ ] Steps: WS manager with EventBus bridge, frontend WebSocket hook, live chart, event feed, portfolio panel.
- [ ] Commit: `git commit -m "feat: add Live Event Monitor with WebSocket streaming"`

---

## Phase 4: Data & Polish

Two workstreams.

---

### Task 4.1: Data Manager

**Files:**
- Create: `backend/routers/data.py`
- Modify: `backend/main.py` (include router)
- Create: `frontend/app/data/page.tsx`
- Create: `frontend/components/tables/symbols-table.tsx`
- Create: `frontend/hooks/use-data.ts`

Backend: symbol listing with bar counts/date ranges, FMP fetch wrapper, validation runner, quality log. Frontend: symbols table, fetch panel, validation panel, storage stats. Reference spec section 4.10.

- [ ] Steps: Backend routes, frontend table, fetch form, validation panel, storage stats card.
- [ ] Commit: `git commit -m "feat: add Data Manager page"`

---

### Task 4.2: Paper Trading placeholder + Event Ticker + Polish

**Files:**
- Create: `frontend/app/paper/page.tsx`
- Create: `frontend/components/layout/event-ticker.tsx`
- Modify: `frontend/app/layout.tsx` (add event ticker)

Paper trading placeholder page per spec 4.11. Event ticker bar at bottom of layout. Loading skeletons, error boundaries for all pages. Reference spec sections 4.11 and 6.

- [ ] Steps: Paper page, event ticker component, loading states, error boundaries.
- [ ] Commit: `git commit -m "feat: add paper trading placeholder, event ticker, polish"`

---

## Subagent Deployment Strategy

**Phase 0:** 3 parallel agents (Tasks 0.1+0.2 together as DB agent, Task 0.3 as engine agent, Tasks 0.4+0.5+0.6 as analytics agent). Run full test suite after each agent completes.

**Phase 1:** 4 parallel agents (Task 1.1 backend, Task 1.2 frontend shell, then Tasks 1.3 and 1.4 in parallel after 1.2 completes). Frontend agents work in worktrees.

**Phase 2:** 4 parallel agents (Tasks 2.1-2.4, all independent). Each builds one page/tab.

**Phase 3:** 3 parallel agents (Tasks 3.1-3.3). Strategy editor is fully independent. Launcher and Monitor share the backtest_service but can be built in parallel with integration after.

**Phase 4:** 2 parallel agents (Tasks 4.1-4.2).
