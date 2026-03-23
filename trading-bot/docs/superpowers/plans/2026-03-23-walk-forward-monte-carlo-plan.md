# Walk-Forward Monte Carlo Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add walk-forward optimization with Monte Carlo stress testing — rolling window parameter optimization, three simulation modes, and dashboard pages for launching/viewing studies.

**Architecture:** `WalkForwardService` orchestrates rolling windows, calling `BacktestEngine` (in quiet mode) for grid search, then stitches OOS results. Results stored in new `walk_forward_studies` table. Frontend: `/walk-forward` launch page + `/walk-forward/[studyId]` detail page.

**Tech Stack:** Python (BacktestEngine, grid search, Monte Carlo), FastAPI (API routes), Next.js (dashboard pages), Recharts (charts), TanStack Query (data fetching).

**Spec:** `docs/superpowers/specs/2026-03-23-walk-forward-monte-carlo-design.md`

---

## File Structure

### Phase A: Walk-Forward Engine (backend)

```
core/engine.py                          — MODIFY: add quiet parameter
strategy/base.py                        — MODIFY: add get_parameter_space/from_params stubs
strategy/config_strategy.py             — MODIFY: add from_config_dict class method, {{placeholder}} substitution
analytics/walk_forward.py               — CREATE: window generation, grid search orchestration, OOS stitching
data/storage/database.py                — MODIFY: add walk_forward_studies table + CRUD methods
data/storage/models.py                  — MODIFY: add WalkForwardStudy dataclass
backend/routers/walk_forward.py         — CREATE: API endpoints
backend/services/walk_forward_service.py — CREATE: background thread runner
backend/schemas.py                      — MODIFY: add walk-forward schemas
backend/main.py                         — MODIFY: include walk-forward router
tests/test_walk_forward.py              — CREATE: tests
```

### Phase B: Enhanced Monte Carlo (backend)

```
analytics/monte_carlo.py                — MODIFY: add window_shuffle, bootstrap, confidence intervals
tests/test_monte_carlo_enhanced.py      — CREATE: tests for new modes
```

### Phase C: Dashboard Pages (frontend)

```
frontend/app/walk-forward/page.tsx      — CREATE: launch + list page
frontend/app/walk-forward/[studyId]/page.tsx — CREATE: study detail page
frontend/components/charts/parameter-stability.tsx — CREATE: param stability chart
frontend/hooks/use-walk-forward.ts      — CREATE: TanStack Query hooks
frontend/lib/types.ts                   — MODIFY: add walk-forward types
frontend/app/runs/[runId]/page.tsx      — MODIFY: add Walk-Forward tab
frontend/components/layout/sidebar.tsx  — MODIFY: add Walk-Forward nav item
```

---

## Phase A: Walk-Forward Engine

---

### Task A.1: Add quiet mode to BacktestEngine

**Files:**
- Modify: `core/engine.py:58-73` (__init__ signature)
- Modify: `core/engine.py:242-256` (console report + persistence)
- Test: `tests/test_walk_forward.py` (create)

- [ ] **Step 1: Write test for quiet mode**

```python
# tests/test_walk_forward.py
from datetime import date
from data.storage.database import Database
from data.storage.models import DailyBar
from core.engine import BacktestEngine
from strategy.base import Strategy
from core.events import SignalEvent

class NullStrategy(Strategy):
    def generate_signals(self, bar, portfolio):
        return []
    def warm_up_period(self):
        return 0

def _seed_bars(db, symbol="TEST", days=20):
    bars = []
    for i in range(days):
        from datetime import timedelta
        d = date(2024, 1, 2) + timedelta(days=i)
        bars.append(DailyBar(symbol=symbol, date=d, open=100+i, high=101+i, low=99+i, close=100+i, adj_close=100+i, volume=1000000, vwap=100+i))
    db.insert_daily_bars(bars)

def test_quiet_mode_no_db_persistence(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    _seed_bars(db)

    engine = BacktestEngine(
        strategy=NullStrategy(), database=db, universe=["TEST"],
        start_date=date(2024, 1, 2), end_date=date(2024, 1, 21),
        initial_capital=100000.0, benchmark_symbol="TEST",
        quiet=True,
    )
    metrics = engine.run()

    # Should return metrics but NOT persist to DB
    assert "total_return" in metrics
    assert len(db.list_runs()) == 0  # no run saved
    assert len(db.get_trades(run_id=metrics.get("_run_id", ""))) == 0
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_walk_forward.py::test_quiet_mode_no_db_persistence -v`
Expected: FAIL — `quiet` parameter not accepted

- [ ] **Step 3: Implement quiet mode**

In `core/engine.py`, add `quiet: bool = False` parameter to `__init__` (after `cancel_event`). Store as `self.quiet`.

In `run()` method, wrap the console report and persistence sections (lines 242-256):

```python
        if not self.quiet:
            # Console report
            report = ReportGenerator(...)
            console_output = report.generate_console_report()
            print(console_output)

            # Persist run record
            self._save_run(run_id, metrics)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_walk_forward.py::test_quiet_mode_no_db_persistence -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: All pass (quiet defaults to False so existing behavior unchanged)

- [ ] **Step 6: Commit**

```bash
git add core/engine.py tests/test_walk_forward.py
git commit -m "feat: add quiet mode to BacktestEngine — skip console output and DB persistence"
```

---

### Task A.2: Add {{placeholder}} substitution to ConfigStrategy

**Files:**
- Modify: `strategy/config_strategy.py:58-76` (add from_config_dict class method)
- Modify: `strategy/base.py:37-40` (add get_parameter_space/from_params stubs)
- Test: `tests/test_walk_forward.py` (append)

- [ ] **Step 1: Write test for placeholder substitution**

```python
# append to tests/test_walk_forward.py
import yaml
from strategy.config_strategy import ConfigStrategy

def test_config_strategy_from_config_dict_with_placeholders(tmp_path):
    config = {
        "name": "Test Strategy",
        "universe": ["SPY"],
        "timeframe": "daily",
        "indicators": {
            "rsi_14": {"type": "RSI", "period": 14},
            "bb_20": {"type": "BollingerBands", "period": 20, "std_dev": "{{bb_std}}"},
        },
        "entry_rules": [
            {"condition": "rsi_14 < {{rsi_thresh}}", "direction": "long", "reason": "test"}
        ],
        "exit_rules": [
            {"condition": "close > sma_20", "reason": "test"}
        ],
        "position_sizing": {"method": "fixed_pct", "value": "{{pos_size}}"},
    }

    params = {"bb_std": 2.0, "rsi_thresh": 35, "pos_size": 0.10}
    strategy = ConfigStrategy.from_config_dict(config, params)

    # Verify substitution happened
    assert strategy.config["indicators"]["bb_20"]["std_dev"] == 2.0
    assert "35" in str(strategy.config["entry_rules"][0]["condition"])
    assert strategy.config["position_sizing"]["value"] == 0.10
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement from_config_dict**

In `strategy/config_strategy.py`, add a class method to `ConfigStrategy`:

```python
import copy
import re

@classmethod
def from_config_dict(cls, config: dict, params: dict | None = None) -> "ConfigStrategy":
    """Create a ConfigStrategy from a config dict (not a file), with optional parameter substitution.

    Any {{placeholder}} in string values is replaced with the corresponding value from params.
    """
    config = copy.deepcopy(config)
    if params:
        config = cls._substitute_params(config, params)

    instance = object.__new__(cls)
    Strategy.__init__(instance)
    instance.config = config
    instance.name = config["name"]
    instance.universe = config.get("universe", [])
    instance._build_indicators()
    instance._parse_rules()
    return instance

@staticmethod
def _substitute_params(obj, params: dict):
    """Recursively replace {{key}} placeholders in a config structure."""
    if isinstance(obj, str):
        # Check for full-value placeholder like "{{pos_size}}"
        match = re.fullmatch(r"\{\{(\w+)\}\}", obj.strip())
        if match:
            key = match.group(1)
            return params.get(key, obj)
        # Inline replacement for placeholders within strings like "rsi_14 < {{thresh}}"
        def replacer(m):
            key = m.group(1)
            return str(params.get(key, m.group(0)))
        return re.sub(r"\{\{(\w+)\}\}", replacer, obj)
    elif isinstance(obj, dict):
        return {k: ConfigStrategy._substitute_params(v, params) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ConfigStrategy._substitute_params(item, params) for item in obj]
    return obj
```

In `strategy/base.py`, add default stubs to the `Strategy` class:

```python
@classmethod
def get_parameter_space(cls) -> dict:
    """Return optimizable parameter ranges. Override in subclasses."""
    return {}

@classmethod
def from_params(cls, params: dict) -> "Strategy":
    """Create a fresh instance with given parameters. Override in subclasses."""
    return cls()
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Run full test suite**

- [ ] **Step 6: Commit**

```bash
git add strategy/config_strategy.py strategy/base.py tests/test_walk_forward.py
git commit -m "feat: add {{placeholder}} substitution and from_config_dict to ConfigStrategy"
```

---

### Task A.3: Walk-forward core engine (window generation + grid search + OOS stitching)

**Files:**
- Create: `analytics/walk_forward.py`
- Test: `tests/test_walk_forward.py` (append)

This is the core orchestration logic. No database or API — pure computation.

- [ ] **Step 1: Write test for window generation**

```python
# append to tests/test_walk_forward.py
from analytics.walk_forward import generate_windows

def test_generate_windows_basic():
    windows = generate_windows(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 1, 1),
        train_months=24,
        oos_months=6,
        step_months=3,
    )
    assert len(windows) >= 2
    for w in windows:
        assert w["train_start"] < w["train_end"]
        assert w["oos_start"] == w["train_end"]  # OOS starts right after train
        assert w["oos_end"] > w["oos_start"]
        assert w["oos_end"] <= date(2025, 1, 1)

def test_generate_windows_too_short_raises():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        generate_windows(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            train_months=24,
            oos_months=6,
            step_months=3,
        )
```

- [ ] **Step 2: Write test for grid search**

```python
from analytics.walk_forward import generate_param_grid

def test_generate_param_grid():
    space = {
        "rsi_thresh": {"type": "strategy", "min": 30, "max": 50, "step": 10},
        "pos_size": {"type": "engine", "min": 0.06, "max": 0.12, "step": 0.06},
    }
    grid = generate_param_grid(space, max_combinations=5000)
    # rsi: 30, 40, 50 = 3 values. pos_size: 0.06, 0.12 = 2 values. Total = 6
    assert len(grid) == 6
    assert all("rsi_thresh" in combo and "pos_size" in combo for combo in grid)

def test_generate_param_grid_exceeds_max_raises():
    import pytest
    space = {f"p{i}": {"type": "strategy", "min": 0, "max": 100, "step": 1} for i in range(3)}
    with pytest.raises(ValueError, match="exceeds max_combinations"):
        generate_param_grid(space, max_combinations=100)  # 101^3 > 100
```

- [ ] **Step 3: Write test for full walk-forward run**

```python
from analytics.walk_forward import run_walk_forward

def test_walk_forward_fixed_params(tmp_path):
    """Walk-forward with no optimization (fixed params) should produce per-window results."""
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    # Seed 5 years of daily bars for TEST symbol
    _seed_bars(db, symbol="TEST", days=1260)  # ~5 years

    result = run_walk_forward(
        strategy_name="NullStrategy",
        strategy_loader=lambda params: NullStrategy(),
        param_space={},  # no optimization
        database=db,
        universe=["TEST"],
        start_date=date(2024, 1, 2),
        end_date=date(2027, 6, 1),
        initial_capital=100000.0,
        benchmark="TEST",
        train_months=12,
        oos_months=6,
        step_months=6,
        objective="sharpe",
    )

    assert len(result["windows"]) >= 2
    assert "aggregate" in result
    assert "stitched_equity_curve" in result
    assert "parameter_stability" in result
    db.close()
```

- [ ] **Step 4: Implement analytics/walk_forward.py**

Key functions:

```python
def generate_windows(start_date, end_date, train_months, oos_months, step_months) -> list[dict]:
    """Generate rolling window date ranges. Returns list of {train_start, train_end, oos_start, oos_end}."""

def generate_param_grid(param_space: dict, max_combinations: int = 5000) -> list[dict]:
    """Generate all parameter combinations from space. Raises if exceeds max."""

def run_walk_forward(
    strategy_name, strategy_loader, param_space, database, universe,
    start_date, end_date, initial_capital, benchmark,
    train_months, oos_months, step_months, objective,
    cancel_event=None, progress_callback=None,
) -> dict:
    """Run full walk-forward optimization. Returns results dict matching spec."""
```

`run_walk_forward` is the main orchestrator:
1. Generate windows
2. For each window: grid search (engine in quiet mode) → best params → OOS run
3. Stitch OOS curves (compounding — each window starts with previous ending equity)
4. Compute aggregate metrics
5. Extract parameter stability
6. Return results dict

`strategy_loader` is a callable `(params: dict) -> Strategy` — created by the service layer. For YAML strategies: `lambda params: ConfigStrategy.from_config_dict(config, params)`. For Python strategies: `lambda params: cls.from_params(params)`.

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

```bash
git add analytics/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: add walk-forward core engine — window generation, grid search, OOS stitching"
```

---

### Task A.4: Database schema + CRUD for walk-forward studies

**Files:**
- Modify: `data/storage/models.py` (add WalkForwardStudy dataclass)
- Modify: `data/storage/database.py` (add table DDL + methods)
- Test: `tests/test_walk_forward.py` (append)

- [ ] **Step 1: Write tests for CRUD**

```python
from data.storage.models import WalkForwardStudy
from datetime import datetime, timezone

def test_walk_forward_study_crud(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    study = WalkForwardStudy(
        study_id="wf_test1", strategy_name="MeanReversion",
        config='{"universe":["SPY"]}', start_date=date(2020,1,1),
        end_date=date(2025,1,1), initial_capital=100000.0,
        train_months=24, oos_months=6, step_months=3,
        objective="sharpe", status="running", results="",
        monte_carlo="", created_at=datetime.now(tz=timezone.utc),
    )
    db.insert_walk_forward_study(study)

    # Get
    fetched = db.get_walk_forward_study("wf_test1")
    assert fetched is not None
    assert fetched.strategy_name == "MeanReversion"

    # List
    studies = db.list_walk_forward_studies()
    assert len(studies) == 1

    # Update
    db.update_walk_forward_study("wf_test1", status="completed", results='{"windows":[]}')
    updated = db.get_walk_forward_study("wf_test1")
    assert updated.status == "completed"

    # Delete
    db.delete_walk_forward_study("wf_test1")
    assert db.get_walk_forward_study("wf_test1") is None
    db.close()
```

- [ ] **Step 2: Implement model + DB methods**

Add `WalkForwardStudy` dataclass to `data/storage/models.py`.
Add `walk_forward_studies` table to `create_tables()` DDL in `database.py`.
Add methods: `insert_walk_forward_study`, `get_walk_forward_study`, `list_walk_forward_studies`, `update_walk_forward_study`, `delete_walk_forward_study`.

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add data/storage/models.py data/storage/database.py tests/test_walk_forward.py
git commit -m "feat: add walk_forward_studies table and CRUD methods"
```

---

### Task A.5: WalkForwardService + API routes

**Files:**
- Create: `backend/services/walk_forward_service.py`
- Create: `backend/routers/walk_forward.py`
- Modify: `backend/schemas.py` (add schemas)
- Modify: `backend/main.py` (include router)

- [ ] **Step 1: Create WalkForwardService**

Background thread runner (same pattern as `BacktestRunner`):
- `start_study(config)` — spawns thread, returns study_id
- `get_status(study_id)` — returns status + progress (windows_completed/total)
- `stop_study(study_id)` — sets cancel_event

The thread:
1. Creates fresh DB connection
2. Loads strategy config (YAML file or Python class)
3. Builds `strategy_loader` callable
4. Extracts param space from YAML `optimization` block or `get_parameter_space()`
5. Calls `analytics.walk_forward.run_walk_forward()`
6. Runs Monte Carlo on stitched OOS trades (selected modes)
7. Persists results to DB

- [ ] **Step 2: Create API routes**

```python
# backend/routers/walk_forward.py
POST /api/walk-forward/run     — launch study
GET  /api/walk-forward          — list studies
GET  /api/walk-forward/{id}     — get study results
GET  /api/walk-forward/{id}/status — poll status + progress
POST /api/walk-forward/{id}/stop — cancel study
DELETE /api/walk-forward/{id}   — delete study
```

- [ ] **Step 3: Add Pydantic schemas**

```python
# Add to backend/schemas.py
WalkForwardRunRequest: strategy, universe, start_date, end_date, initial_capital, benchmark, train_months, oos_months, step_months, objective, monte_carlo_modes, simulations
WalkForwardRunResponse: study_id, status
WalkForwardStatusResponse: study_id, status, windows_completed, windows_total, current_phase
WalkForwardStudyResponse: full study data including windows, aggregate, stitched curve, monte carlo, param stability
WalkForwardListResponse: studies list + total count
```

- [ ] **Step 4: Register router in main.py**

- [ ] **Step 5: Verify server starts**

Run: `python3 -m uvicorn backend.main:app --port 8001`
Verify: `curl http://localhost:8001/api/walk-forward` returns `{"studies":[],"total":0}`

- [ ] **Step 6: Commit**

```bash
git add backend/services/walk_forward_service.py backend/routers/walk_forward.py backend/schemas.py backend/main.py
git commit -m "feat: add WalkForwardService and API routes"
```

---

## Phase B: Enhanced Monte Carlo

---

### Task B.1: Add window shuffle, bootstrap, and confidence intervals

**Files:**
- Modify: `analytics/monte_carlo.py` (add new functions)
- Test: `tests/test_monte_carlo_enhanced.py` (create)

- [ ] **Step 1: Write tests**

```python
# tests/test_monte_carlo_enhanced.py
from analytics.monte_carlo import run_window_shuffle, run_bootstrap, compute_metric_confidence_intervals

def test_window_shuffle():
    # 4 windows, each with a list of trade P&Ls
    windows = [
        [100, -50, 200],
        [-100, 150, 80],
        [300, -200, 100],
        [50, 100, -30],
    ]
    result = run_window_shuffle(windows, initial_capital=10000, num_simulations=100, seed=42)
    assert "percentile_bands" in result
    assert set(result["percentile_bands"].keys()) == {"p5", "p25", "p50", "p75", "p95"}
    assert result["simulations"] == 100
    assert "sharpe_ci" in result
    assert len(result["sharpe_ci"]) == 2  # lower, upper

def test_bootstrap():
    windows = [
        [100, -50, 200],
        [-100, 150, 80],
        [300, -200, 100],
        [50, 100, -30],
    ]
    result = run_bootstrap(windows, initial_capital=10000, num_simulations=100, seed=42)
    assert "percentile_bands" in result
    # Bootstrap samples K=4 windows with replacement
    assert result["simulations"] == 100

def test_confidence_intervals():
    import numpy as np
    sharpe_values = np.random.normal(1.5, 0.3, 1000)
    dd_values = np.random.normal(-0.10, 0.03, 1000)
    ci = compute_metric_confidence_intervals(sharpe_values, dd_values)
    assert ci["sharpe_ci"][0] < ci["sharpe_ci"][1]
    assert ci["max_dd_ci"][0] < ci["max_dd_ci"][1]
```

- [ ] **Step 2: Implement new Monte Carlo functions**

Add to `analytics/monte_carlo.py`:

```python
def run_window_shuffle(windows, initial_capital, num_simulations, seed=None, ruin_threshold=0.0):
    """Shuffle OOS window order, keeping internal trade sequence. Returns result dict."""

def run_bootstrap(windows, initial_capital, num_simulations, seed=None, ruin_threshold=0.0):
    """Sample K windows with replacement (K = len(windows)). Returns result dict."""

def compute_metric_confidence_intervals(sharpe_values, max_dd_values, confidence=0.95):
    """Compute confidence intervals on Sharpe and max drawdown distributions."""
```

Both `run_window_shuffle` and `run_bootstrap`:
1. For each simulation: reorder/resample windows → concatenate trade P&Ls → replay equity curve
2. Collect percentile bands per time step, final equity distribution, drawdown distribution
3. Compute Sharpe and max DD for each simulation → confidence intervals
4. Return result dict matching the spec JSON structure

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Run full test suite**

- [ ] **Step 5: Commit**

```bash
git add analytics/monte_carlo.py tests/test_monte_carlo_enhanced.py
git commit -m "feat: add window shuffle, bootstrap, and confidence intervals to Monte Carlo"
```

---

## Phase C: Dashboard Pages

---

### Task C.1: Walk-Forward types + hooks + sidebar nav

**Files:**
- Modify: `frontend/lib/types.ts` (add walk-forward types)
- Create: `frontend/hooks/use-walk-forward.ts`
- Modify: `frontend/components/layout/sidebar.tsx` (add nav item)

- [ ] **Step 1: Add TypeScript types**

```typescript
// Add to frontend/lib/types.ts
export interface WalkForwardWindow {
  window_index: number;
  train_start: string;
  train_end: string;
  oos_start: string;
  oos_end: string;
  best_params: Record<string, number>;
  train_metrics: Record<string, number>;
  oos_metrics: Record<string, number>;
  oos_trades: Trade[];
}

export interface WalkForwardStudy {
  study_id: string;
  strategy_name: string;
  config: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  train_months: number;
  oos_months: number;
  step_months: number;
  objective: string;
  status: string;
  created_at: string;
  windows: WalkForwardWindow[];
  aggregate: Record<string, number>;
  stitched_equity_curve: EquityCurvePoint[];
  parameter_stability: Record<string, number[]>;
  monte_carlo: Record<string, MonteCarloResult>;
}

export interface WalkForwardListResponse {
  studies: WalkForwardStudy[];
  total: number;
}

export interface WalkForwardStatusResponse {
  study_id: string;
  status: string;
  windows_completed: number;
  windows_total: number;
  current_phase: string;
}
```

- [ ] **Step 2: Create hooks**

```typescript
// frontend/hooks/use-walk-forward.ts
useLaunchWalkForward()  — POST /api/walk-forward/run
useWalkForwardStudies() — GET /api/walk-forward
useWalkForwardStudy(id) — GET /api/walk-forward/{id}
useWalkForwardStatus(id) — GET /api/walk-forward/{id}/status (poll while running)
useStopWalkForward()    — POST /api/walk-forward/{id}/stop
useDeleteWalkForward()  — DELETE /api/walk-forward/{id}
```

- [ ] **Step 3: Add sidebar nav item**

Add `{ href: "/walk-forward", label: "Walk-Forward", icon: TrendingUp }` to sidebar nav between "Monitor" and "Data".

- [ ] **Step 4: Verify build**

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/types.ts frontend/hooks/use-walk-forward.ts frontend/components/layout/sidebar.tsx
git commit -m "feat: add walk-forward types, hooks, and sidebar nav"
```

---

### Task C.2: Walk-Forward Launch + List page

**Files:**
- Create: `frontend/app/walk-forward/page.tsx`

- [ ] **Step 1: Build the page**

**Top: Configuration form** matching the spec:
- Strategy picker (reuse useStrategies hook)
- Universe, date range, initial capital, benchmark inputs
- Train/OOS/Step month inputs (default 24/6/3)
- Objective dropdown
- Monte Carlo mode checkboxes (trade shuffle + bootstrap checked by default)
- Simulations input (default 10,000)
- "Run Study" button → POST /api/walk-forward/run → show status with progress polling

**Bottom: Previous studies table**
- DataTable with columns: study_id, strategy, date range, windows, OOS Sharpe, OOS Return, status, created_at
- Click row → `/walk-forward/{studyId}`
- Delete button per row

Use same styling patterns as the Backtest Launcher page.

- [ ] **Step 2: Verify build**

- [ ] **Step 3: Commit**

```bash
git add frontend/app/walk-forward/
git commit -m "feat: add Walk-Forward launch and list page"
```

---

### Task C.3: Walk-Forward Study Detail page

**Files:**
- Create: `frontend/app/walk-forward/[studyId]/page.tsx`
- Create: `frontend/components/charts/parameter-stability.tsx`

- [ ] **Step 1: Build parameter stability chart**

Recharts BarChart (or grouped bar chart) showing the optimizer's chosen value for each parameter across windows. One color per parameter. X-axis = window index. Y-axis = parameter value. If values cluster tightly = stable. If they scatter = overfit.

- [ ] **Step 2: Build study detail page**

Layout matching the spec:
- Header: strategy, date range, window config
- Aggregate OOS MetricsStrip
- Main grid: Stitched OOS Equity Curve (left 2/3) + Parameter Stability Chart (right 1/3)
- Per-Window Results Table (expandable rows)
- Tabs: Monte Carlo modes (one tab per mode — fan chart + stats + drawdown histogram)

Reuse existing components: MetricsStrip, EquityCurve chart (pass stitched curve), FanChart (pass per-mode MC data).

- [ ] **Step 3: Verify build**

- [ ] **Step 4: Commit**

```bash
git add frontend/app/walk-forward/ frontend/components/charts/parameter-stability.tsx
git commit -m "feat: add Walk-Forward study detail page with parameter stability chart"
```

---

### Task C.4: Add Walk-Forward tab to Run Detail page

**Files:**
- Modify: `frontend/app/runs/[runId]/page.tsx` (add tab)

- [ ] **Step 1: Add "Walk-Forward" tab**

Add a new tab alongside Monthly Returns, Monte Carlo, Options:
- Content: "Run a walk-forward study for this strategy" description
- Button: "Launch Walk-Forward Study" → navigates to `/walk-forward` with query params pre-filling the form (strategy, universe from run config, date range, capital)

- [ ] **Step 2: Verify build**

- [ ] **Step 3: Commit**

```bash
git add frontend/app/runs/
git commit -m "feat: add Walk-Forward tab to Run Detail page"
```

---

## Subagent Deployment Strategy

**Phase A (3 agents, partially parallel):**
1. Tasks A.1 + A.2 together (engine quiet mode + ConfigStrategy substitution) — touches different files, can be one agent
2. Task A.3 (walk-forward core engine) — depends on A.1 + A.2
3. Tasks A.4 + A.5 together (DB schema + service + API) — can start after A.3

**Phase B (1 agent):**
1. Task B.1 (Monte Carlo extensions) — independent of Phase A API work

**Phase C (3 agents, partially parallel):**
1. Task C.1 (types + hooks + sidebar) — first, as other tasks depend on it
2. Tasks C.2 + C.3 in parallel (launch page + detail page) — after C.1
3. Task C.4 (Run Detail tab) — quick, can be done with C.2 or after

Total: ~7 agent deployments across 3 phases.
