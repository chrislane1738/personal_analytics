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

### Parameter Space Definition

**YAML strategies** — optional `optimization` block in the YAML config:

```yaml
name: "Mean Reversion SPY"
universe: ["SPY"]
timeframe: daily

indicators:
  sma_20: { type: SMA, period: 20 }
  rsi_14: { type: RSI, period: 14 }
  bb_20:  { type: BollingerBands, period: 20, std_dev: 2 }

entry_rules:
  - condition: "close < bb_20.lower AND rsi_14 < 45"
    direction: long
    reason: "Price below lower BB with weakening RSI"

exit_rules:
  - condition: "close > sma_20"
    reason: "Price reverted to mean"
  - condition: "pnl_pct < -0.05"
    reason: "Stop loss at -5%"

position_sizing:
  method: fixed_pct
  value: 0.12

optimization:
  parameters:
    rsi_threshold: { min: 25, max: 50, step: 5 }
    bb_std_dev: { min: 1.5, max: 3.0, step: 0.5 }
    position_size_pct: { min: 0.06, max: 0.20, step: 0.02 }
  objective: sharpe
```

The `optimization.parameters` keys map to strategy config values that get substituted during grid search. The engine generates all parameter combinations, runs each as a mini-backtest on the training window, and picks the best by objective.

**Python strategies** — implement optional `get_parameter_space()` method:

```python
class MyStrategy(Strategy):
    def get_parameter_space(self) -> dict:
        return {
            "rsi_period": {"min": 10, "max": 30, "step": 5},
            "threshold": {"min": 0.01, "max": 0.05, "step": 0.01},
        }
```

**No optimization defined** — falls back to fixed-parameter window validation. The strategy runs unchanged across all windows. Still valuable for testing time-period robustness.

### Optimization

Grid search within each training window:

1. Generate all parameter combinations from the defined space
2. For each combination: instantiate strategy with those params → run BacktestEngine on the training window → collect objective metric
3. Rank by objective function (Sharpe, Sortino, Calmar, Total Return, or Profit Factor)
4. Select best parameters
5. Run best parameters on the OOS window → collect OOS metrics + trades + equity curve

The existing BacktestEngine is reused for every run — no new engine code needed. The walk-forward orchestrator just calls it repeatedly.

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
- Process: for each simulation, randomly sample K windows (with replacement) from the pool of actual windows, concatenate
- Output: same format — tests sensitivity to which time periods were experienced

### All modes produce:

- Percentile bands (P5/P25/P50/P75/P95) for fan chart
- Probability of ruin (equity hitting zero or configurable threshold)
- Final equity distribution (histogram)
- Confidence intervals on Sharpe, Sortino, max drawdown
- Drawdown distribution (histogram of max drawdowns across simulations)

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
| GET | /api/walk-forward/{id}/status | Poll status (running/completed/failed) |
| DELETE | /api/walk-forward/{id} | Delete study |

### Backend service: `WalkForwardService`

Runs in a background thread (same pattern as `BacktestRunner`):

1. Parse strategy and parameter space
2. Generate rolling window date ranges from start_date/end_date using train_months/oos_months/step_months
3. For each window:
   a. Generate parameter grid from param space
   b. For each param combination: instantiate strategy → run BacktestEngine on train dates → collect objective metric
   c. Select best params by objective
   d. Run best params on OOS dates → collect metrics, trades, equity curve
4. Stitch all OOS equity curves chronologically
5. Compute aggregate metrics on stitched curve
6. Extract parameter stability data
7. Run Monte Carlo simulations on stitched OOS trades (selected modes)
8. Persist everything to `walk_forward_studies` table

Uses a fresh Database connection per thread (same pattern as BacktestRunner).

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
