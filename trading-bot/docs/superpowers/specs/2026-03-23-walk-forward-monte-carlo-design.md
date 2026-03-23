# Walk-Forward Monte Carlo Testing — Design Spec

**Date:** 2026-03-23
**Status:** Approved
**Scope:** Walk-forward optimization with Monte Carlo stress testing, integrated into the dashboard

---

## 1. Purpose

Add walk-forward validation and enhanced Monte Carlo stress testing to the trading bot framework. This answers two critical questions:

1. **Would this strategy have worked in real time?** (Walk-forward validation — no future knowledge)
2. **How fragile are the results?** (Monte Carlo — stress test against different trade sequences and market conditions)

The system is fully generalized — works with any strategy implementing the Strategy interface (YAML config or Python class).

---

## 2. Walk-Forward Engine

### Window Structure

Rolling walk-forward with configurable window sizes:

```
|----Train 1 (2yr)----|--OOS 1 (6mo)--|
      |----Train 2 (2yr)----|--OOS 2 (6mo)--|
            |----Train 3 (2yr)----|--OOS 3 (6mo)--|
                  |----Train 4 (2yr)----|--OOS 4 (6mo)--|
```

Defaults: 2-year train, 6-month OOS, 3-month step. All configurable per study.

### Parameter Taxonomy

Optimizable parameters fall into two categories:

1. **Strategy parameters** — live inside the strategy config (indicator settings, condition thresholds, exit rules). These are passed to the strategy constructor.
2. **Engine parameters** — live on `BacktestEngine.__init__` (position_size_pct, slippage_pct, commission_per_share). These are passed to the engine constructor.

The walk-forward orchestrator must route each parameter to the correct constructor.

### Parameter Space Definition

**YAML strategies** — use dotted paths into the YAML structure and `{{placeholder}}` syntax in condition strings:

```yaml
name: "Mean Reversion SPY"
universe: ["SPY"]
timeframe: daily

indicators:
  sma_20: { type: SMA, period: 20 }
  rsi_14: { type: RSI, period: 14 }
  bb_20:  { type: BollingerBands, period: 20, std_dev: "{{bb_std_dev}}" }

entry_rules:
  - condition: "close < bb_20.lower AND rsi_14 < {{rsi_threshold}}"
    direction: long
    reason: "Price below lower BB with weakening RSI"

exit_rules:
  - condition: "close > sma_20"
    reason: "Price reverted to mean"
  - condition: "pnl_pct < -0.05"
    reason: "Stop loss at -5%"

position_sizing:
  method: fixed_pct
  value: "{{position_size_pct}}"

optimization:
  parameters:
    rsi_threshold:
      type: strategy  # substituted into YAML via {{placeholder}}
      min: 25
      max: 50
      step: 5
    bb_std_dev:
      type: strategy
      min: 1.5
      max: 3.0
      step: 0.5
    position_size_pct:
      type: engine  # passed to BacktestEngine constructor
      min: 0.06
      max: 0.20
      step: 0.02
  objective: sharpe
  max_combinations: 5000  # fail-fast if grid exceeds this
```

**Substitution mechanism:** Before each grid search iteration, the orchestrator:
1. Deep-copies the YAML config dict
2. Replaces all `{{placeholder}}` strings with the current parameter values (both in indicator params and condition strings)
3. Routes `type: engine` parameters to `BacktestEngine.__init__` kwargs
4. Creates a **fresh** `ConfigStrategy` instance from the modified config (new indicators, clean state)

**Python strategies** — implement optional `get_parameter_space()` and `from_params()` class methods:

```python
class MyStrategy(Strategy):
    @classmethod
    def get_parameter_space(cls) -> dict:
        return {
            "rsi_period": {"type": "strategy", "min": 10, "max": 30, "step": 5},
            "threshold": {"type": "strategy", "min": 0.01, "max": 0.05, "step": 0.01},
        }

    @classmethod
    def from_params(cls, params: dict) -> "MyStrategy":
        """Create a fresh instance with the given parameters."""
        return cls(rsi_period=params["rsi_period"], threshold=params["threshold"])
```

**No optimization defined** — falls back to fixed-parameter window validation. The strategy runs unchanged across all windows. Still valuable for testing time-period robustness.

### Optimization

Grid search within each training window:

1. Generate all parameter combinations from the defined space
2. **Guard:** If total combinations exceed `max_combinations` (default 5,000), abort with an error suggesting the user increase step sizes or reduce parameter count
3. For each combination: create a **fresh** strategy instance (new `ConfigStrategy` or `from_params()`) + fresh engine → run on training window → collect objective metric
4. Rank by objective function (Sharpe, Sortino, Calmar, Total Return, or Profit Factor)
5. Select best parameters
6. Run best parameters on the OOS window → collect OOS metrics + trades + equity curve

**Engine side effects during grid search:** The existing `BacktestEngine.run()` prints console reports and persists runs/trades/equity curves to the database. For grid search iterations, the engine must run in **quiet mode** — a new `quiet=True` parameter on `BacktestEngine.__init__` that suppresses console output and skips database persistence. Only the final OOS run for each window persists results. This is a required modification to `core/engine.py`.

**Fresh instantiation per grid point:** Each parameter combination gets a brand-new strategy instance (new indicators, clean state) and a new engine instance. The YAML file content is cached in memory to avoid re-reading from disk, but parsing and indicator creation happen fresh each time. This prevents indicator state contamination between iterations.

### Window Generation Rules

- Minimum: at least 2 complete windows required, otherwise abort with error
- Incomplete trailing window (OOS extends beyond end_date): dropped
- Windows with fewer than 20 trading days in either train or OOS: skipped with warning
- Window dates snap to actual trading days (skip weekends/holidays)

### Equity Continuity Between OOS Windows

**Compounding mode (default):** Each OOS window starts with the ending equity of the previous OOS window. Window 1 OOS starts at `initial_capital`. If Window 1 ends at $112K, Window 2 OOS starts at $112K. This mirrors real-time trading and produces a realistic stitched equity curve.

The `initial_capital` for each OOS BacktestEngine run is set to the previous window's final equity.

### Output

Per walk-forward study:

- **Per-window results:** training metrics, best parameters chosen, OOS metrics, OOS trades, OOS equity curve segment
- **Stitched OOS equity curve:** concatenating all out-of-sample equity curve segments in chronological order
- **Parameter stability:** what parameters the optimizer chose for each window
- **Aggregate metrics:** Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor — all computed on the stitched OOS curve

---

## 3. Monte Carlo Stress Testing (Enhanced)

Three simulation modes, all operating on walk-forward OOS data:

### Mode 1: Trade Shuffle (existing, enhanced)

Randomly reorder the OOS trades and replay the equity curve. Tests whether success depends on lucky trade sequencing.

- Input: list of trade P&Ls from stitched OOS results
- Process: shuffle trade order N times, replay equity curve each time
- Output: percentile bands per time step, final equity distribution, probability of ruin

### Mode 2: Window Shuffle (new)

Randomly reorder the walk-forward OOS windows. Each window keeps its internal trade sequence intact, but the order of windows is randomized.

- Input: list of OOS window results (each containing trades in original order)
- Process: shuffle window order N times, concatenate equity curves
- Output: same as Mode 1 but tests regime-sequence sensitivity

### Mode 3: Bootstrap Resampling (new)

Sample OOS windows with replacement. Some windows appear multiple times, others are skipped.

- Input: list of OOS window results
- Process: for each simulation, randomly sample K windows (with replacement) from the pool of actual windows, concatenate. K = actual number of windows (so the resampled study has the same length as the original, just different composition)
- Output: same format — tests sensitivity to which time periods were experienced

### All modes produce:

- Percentile bands (P5/P25/P50/P75/P95) for fan chart
- Probability of ruin (equity hitting zero or configurable threshold)
- Final equity distribution (histogram)
- Confidence intervals on Sharpe, Sortino, max drawdown
- Drawdown distribution (histogram of max drawdowns across simulations)

### Implementation notes:

- Window shuffle and bootstrap modes are new functions in `analytics/monte_carlo.py` (alongside existing `run_monte_carlo`)
- Confidence intervals (Sharpe CI, max DD CI) require computing Sharpe/max DD for each simulation, then taking P2.5/P97.5. New fields on `MonteCarloResult`: `sharpe_ci: tuple[float, float]`, `max_dd_ci: tuple[float, float]`
- All three modes return the same result structure for consistent frontend rendering

### User configuration:

- Select mode(s) via checkboxes (default: trade shuffle + bootstrap)
- Simulation count (default 10,000)
- Ruin threshold (default 0 = total loss)

---

## 4. Data Model

### New table: `walk_forward_studies`

```sql
CREATE TABLE IF NOT EXISTS walk_forward_studies (
    study_id        TEXT    PRIMARY KEY,
    strategy_name   TEXT    NOT NULL,
    config          TEXT    NOT NULL,    -- JSON: universe, benchmark, param space, window config
    start_date      DATE    NOT NULL,
    end_date        DATE    NOT NULL,
    initial_capital REAL    NOT NULL DEFAULT 100000.0,
    train_months    INTEGER NOT NULL DEFAULT 24,
    oos_months      INTEGER NOT NULL DEFAULT 6,
    step_months     INTEGER NOT NULL DEFAULT 3,
    objective       TEXT    NOT NULL DEFAULT 'sharpe',
    status          TEXT    NOT NULL DEFAULT 'running',
    results         TEXT    DEFAULT '', -- JSON: per-window results, aggregate metrics, param choices
    monte_carlo     TEXT    DEFAULT '', -- JSON: simulation results per mode
    created_at      TIMESTAMP
);
```

The `results` JSON structure:

```json
{
  "windows": [
    {
      "window_index": 0,
      "train_start": "2020-01-01",
      "train_end": "2021-12-31",
      "oos_start": "2022-01-01",
      "oos_end": "2022-06-30",
      "best_params": {"rsi_threshold": 35, "bb_std_dev": 2.0},
      "train_metrics": {"sharpe": 1.8, "total_return": 0.15, ...},
      "oos_metrics": {"sharpe": 1.2, "total_return": 0.08, ...},
      "oos_trades": [...],
      "oos_equity_curve": [{"date": "...", "value": ...}, ...]
    },
    ...
  ],
  "aggregate": {
    "sharpe": 1.1,
    "total_return": 0.45,
    "max_drawdown": -0.12,
    "win_rate": 0.72,
    "profit_factor": 1.8,
    "total_trades": 48
  },
  "stitched_equity_curve": [...],
  "parameter_stability": {
    "rsi_threshold": [35, 40, 35, 30, 35, 40],
    "bb_std_dev": [2.0, 2.0, 1.5, 2.0, 2.5, 2.0]
  }
}
```

The `monte_carlo` JSON structure:

```json
{
  "trade_shuffle": {
    "simulations": 10000,
    "percentile_bands": {"p5": [...], "p25": [...], ...},
    "actual_curve": [...],
    "final_equity_distribution": [...],
    "drawdown_distribution": [...],
    "probability_of_ruin": 0.02,
    "sharpe_ci": [0.8, 1.4],
    "max_dd_ci": [-0.18, -0.06]
  },
  "bootstrap": { ... same structure ... },
  "window_shuffle": { ... same structure ... }
}
```

---

## 5. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/walk-forward/run | Launch study (background thread). Body: strategy, universe, dates, capital, benchmark, train_months, oos_months, step_months, objective, monte_carlo_modes[], simulations |
| GET | /api/walk-forward | List all studies (paginated, sortable) |
| GET | /api/walk-forward/{id} | Full study results |
| GET | /api/walk-forward/{id}/status | Poll status with progress: `{ status, windows_completed, windows_total, current_phase }` |
| POST | /api/walk-forward/{id}/stop | Cancel a running study |
| DELETE | /api/walk-forward/{id} | Delete study |

### Backend service: `WalkForwardService`

Runs in a background thread with cancellation support (`threading.Event`, same pattern as `BacktestRunner`):

1. Parse strategy and parameter space
2. **Guard:** compute total grid combinations × windows. If exceeds `max_combinations`, abort immediately.
3. Generate rolling window date ranges. Validate: at least 2 complete windows, skip windows with < 20 trading days.
4. For each window (update progress after each):
   a. Generate parameter grid from param space
   b. For each param combination: create fresh strategy + fresh engine (**quiet mode**) → run on train dates → collect objective metric
   c. Select best params by objective
   d. Run best params on OOS dates (initial_capital = previous window's final equity) → collect metrics, trades, equity curve
   e. Check cancel_event — break if set
5. Stitch all OOS equity curves chronologically (compounding)
6. Compute aggregate metrics on stitched curve
7. Extract parameter stability data
8. Run Monte Carlo simulations on stitched OOS trades (selected modes)
9. Persist everything to `walk_forward_studies` table

Uses a fresh Database connection per thread (same pattern as BacktestRunner). All grid-search engine runs within a single thread share that connection. Study IDs use `str(uuid.uuid4())[:8]` consistent with run IDs.

### Required modification to BacktestEngine

Add `quiet: bool = False` parameter to `BacktestEngine.__init__`. When `quiet=True`:
- Skip console report printing (`ReportGenerator`)
- Skip database persistence (`_save_run` — no RunRecord, trades, or equity curve written to DB)
- Return metrics dict as normal

This prevents grid search from cluttering the database with thousands of intermediate runs.

---

## 6. Dashboard Pages

### `/walk-forward` — Launch & List Page

**Configuration form (top):**
- Strategy picker (dropdown of YAML files + Python paths)
- Universe (comma-separated symbols)
- Date range (start/end)
- Initial capital (default $100,000)
- Benchmark (default SPY)
- Train window: number input in months (default 24)
- OOS window: number input in months (default 6)
- Step size: number input in months (default 3)
- Objective: dropdown (Sharpe, Sortino, Calmar, Total Return, Profit Factor)
- Monte Carlo modes: checkboxes (Trade Shuffle, Window Shuffle, Bootstrap — default: first + third checked)
- Simulations: number input (default 10,000)
- "Run Study" button

**Previous studies table (bottom):**
- Columns: study_id, strategy, date range, windows, OOS Sharpe, OOS Return, status, created_at
- Click row → `/walk-forward/{studyId}`

### `/walk-forward/[studyId]` — Study Detail Page

**Header:** Strategy name, full date range, window config (24mo train / 6mo OOS / 3mo step), objective

**Aggregate OOS Metrics Strip:** Same MetricsStrip component — Sharpe, Sortino, Return, Max DD, Win Rate, Profit Factor, Total Trades. All computed from stitched OOS curve only.

**Main grid (2 columns):**
- Left (2/3): Stitched OOS equity curve (green) + benchmark (blue). Only shows OOS periods — the "would have happened in real time" view.
- Right (1/3): Parameter stability chart. One subplot per optimized parameter, showing the chosen value per window. Stable = good, erratic = overfit.

**Per-Window Results Table:**
- Columns: Window #, Train Period, OOS Period, Best Params (JSON), OOS Return, OOS Sharpe, OOS Trades
- Click row to expand → shows that window's individual trades

**Monte Carlo Tab(s):**
- One tab per selected simulation mode
- Each tab: fan chart (P5-P95 bands + actual), stats panel (median, P5, P95, P(ruin)), drawdown distribution histogram
- Confidence intervals on Sharpe and max drawdown shown in stats panel

### Existing Run Detail — New Tab

Add "Walk-Forward" tab with a button: "Run walk-forward study for this strategy." Pre-fills the `/walk-forward` form with the current run's strategy, universe, date range, and parameters.

---

## 7. Build Phases

### Phase A: Walk-Forward Engine (backend only)
- Add `walk_forward_studies` table to database schema
- Create `WalkForwardService` with rolling window generation, grid search, OOS execution
- Create `backend/routers/walk_forward.py` with all API endpoints
- Add optimization block parsing to ConfigStrategy
- Tests: window generation, parameter grid, single-window optimization, full walk-forward run

### Phase B: Enhanced Monte Carlo (backend)
- Add window shuffle and bootstrap resampling modes to `analytics/monte_carlo.py`
- Add Sharpe/max DD confidence interval computation
- Integrate into WalkForwardService (run after OOS stitching)
- Tests: window shuffle produces correct bands, bootstrap resamples correctly

### Phase C: Dashboard Pages (frontend)
- `/walk-forward` launch + list page
- `/walk-forward/[studyId]` detail page with stitched equity curve, parameter stability chart, per-window table
- Monte Carlo fan chart tabs per simulation mode
- "Walk-Forward" tab in Run Detail page
- Hooks: useWalkForwardStudies, useWalkForwardStudy, useLaunchWalkForward

### Subagent strategy:
- Phase A: 2 agents (DB schema + service, router + API)
- Phase B: 1 agent (Monte Carlo extensions)
- Phase C: 2 agents (launch/list page, detail page)
