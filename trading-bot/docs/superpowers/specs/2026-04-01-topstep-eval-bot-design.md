# Topstep Evaluation Bot — Design Spec

**Date:** 2026-04-01
**Status:** Approved
**Author:** Chris Lane + Claude

---

## 1. Overview

A Topstep prop firm evaluation simulator integrated into the trading-bot dashboard. Runs 1,000+ independent simulated evaluation attempts using configurable strategies, measures pass rate and expected value, and displays results in a new `/eval` dashboard page.

### The Premise

Topstep's evaluation structure creates an asymmetric payoff: $49/month subscription risk for a chance at $5,000+ funded account payouts. Even with zero trading edge, the probability math favors the trader — a random walk hits +$3K before -$2K roughly 40% of the time. With a disciplined strategy optimized for the evaluation rules (not for "good trading"), the expected value per attempt is meaningfully positive.

This tool lets us empirically measure that EV by simulating thousands of evaluation attempts across historical market data.

### Goals

- **Measure pass rate** for different strategy configurations against historical data
- **Quantify expected value** per evaluation attempt, including subscription and activation costs
- **Compare strategies** — ORB vs VWAP reversion, with and without the adaptive state machine
- **Identify regime sensitivity** — which market conditions favor which strategies
- **Dashboard integration** — full visualization in the existing trading-bot dashboard

### Non-Goals (for now)

- TopstepX API integration (live bot execution) — future scope C
- Intraday bar data ingestion — architecture supports it, not implemented in v1
- Multi-account copy trading simulation — later phase
- Schwab live data streaming — future
- Express Funded Account (XFA) simulation — only the Trading Combine evaluation

---

## 2. Architecture

### Evaluation Engine Wrapper (Approach B)

The `TopstepEvalSimulator` wraps the existing `BacktestEngine`. Evaluation-specific logic (trailing drawdown, consistency rule, pass/fail detection, attempt lifecycle) lives in the wrapper. Signal generation delegates to standard `Strategy` subclasses. The existing engine, broker, portfolio, and analytics stay untouched.

```
TopstepEvalSimulator (NEW — orchestrator)
├── EvaluationRules        — pass/fail logic, trailing drawdown, consistency
├── StateManager           — 6-state adaptive position sizing
├── AttemptTracker         — per-attempt P&L tracking, daily accounting
├── Strategy               — ORBStrategy / VWAPReversionStrategy (swappable)
└── BacktestEngine         — EXISTING, unmodified
    ├── SimBroker
    ├── Portfolio
    └── EventBus

CampaignRunner (NEW — outer loop)
├── Runs N=1,000+ independent attempts
├── Random start date selection
├── Aggregates pass/fail statistics
├── Computes EV, cost-to-funded
└── Saves results to DB
```

**How the wrapper works:** The simulator subscribes to `PortfolioUpdateEvent` on the EventBus to track evaluation state in real-time. After each trading day, it runs the evaluation rules to check for pass/fail. It feeds account state to the `StateManager`, which adjusts strategy parameters (position size multiplier, stop distance) for the next day. If the evaluation ends (pass or fail), the simulator sets the engine's `cancel_event` to stop execution early.

---

## 3. Topstep Evaluation Rules

### TopstepConfig

Configurable per account tier:

```python
@dataclass
class TopstepConfig:
    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    max_loss: float = 2_000.0            # trailing EOD drawdown
    consistency_pct: float = 0.50         # best day ≤ 50% of total profit
    subscription_fee: float = 49.0        # real cost per attempt
    activation_fee: float = 149.0         # paid on pass
    max_payout: float = 5_000.0           # XFA payout cap
    payout_split: float = 0.90            # 90/10 trader keeps 90%
    max_position_minis: int = 5           # 50K tier
    max_position_micros: int = 50         # 50K tier
    max_attempt_days: int = 60            # timeout after 60 trading days
```

Preset tiers:

| Tier | Account | Target | Max Loss | Max Minis | Max Micros | Sub Fee |
|------|---------|--------|----------|-----------|------------|---------|
| 50K  | $50,000 | $3,000 | $2,000   | 5         | 50         | $49/mo  |
| 100K | $100,000| $6,000 | $3,000   | 10        | 100        | $99/mo  |
| 150K | $150,000| $9,000 | $4,500   | 15        | 150        | $149/mo |

### EvaluationRules

End-of-day evaluation flow:

1. Record day's realized P&L
2. Update EOD balance high-water mark (only moves up)
3. Compute trailing drawdown floor = `high_water - max_loss`
4. **FAIL** if EOD balance ≤ drawdown floor
5. Check consistency: `best_day_pnl > consistency_pct × total_profit` → consistency violation (not instant fail — means more profitable days needed to dilute the best day's share)
6. **PASS** if `total_profit ≥ profit_target` AND consistency satisfied
7. **TIMEOUT** if days traded ≥ `max_attempt_days` without pass or fail

### AttemptTracker

Tracks per-attempt state:

- `daily_pnl: list[float]` — P&L per trading day
- `cumulative_pnl: float` — running total
- `eod_high_water: float` — highest end-of-day balance seen
- `drawdown_floor: float` — `eod_high_water - max_loss`
- `best_day_pnl: float` — largest single-day P&L
- `days_traded: int` — number of trading days
- `status: ACTIVE | PASS | FAIL | TIMEOUT`
- `state_history: list[EvalState]` — state machine transitions for analytics

---

## 4. State Machine

The `StateManager` maps account P&L state to strategy behavior. Six states with distinct position sizing and risk profiles:

| State | P&L Range | Position Multiplier | Stop Multiplier | Behavior |
|-------|-----------|--------------------:|----------------:|----------|
| NORMAL | $0 to +$500 | 1.0x | 1.0x | Standard signals, standard stops |
| CAREFUL | +$500 to +$1,500 | 0.7x | 0.7x | Tighter stops, protect gains |
| REPEAT | +$1,500 to +$3,000 | 0.8x | 0.9x | Same daily target, one more good day needed |
| AGGRESSIVE | -$500 to -$1,000 | 1.3x | 1.2x | Wider targets, accept more risk |
| YOLO | -$1,000 to -$1,500 | 1.8x | 1.5x | Large positions, binary outcome |
| HAIL_MARY | -$1,500 to -$2,000 | 2.5x | 2.0x | Max size, one shot — $49 reset anyway |

**Interface:**
- `get_state(cumulative_pnl: float) -> EvalState` — returns current state enum
- `get_position_multiplier(cumulative_pnl: float) -> float` — strategy reads this before sizing
- `get_stop_multiplier(cumulative_pnl: float) -> float` — strategy reads this for stop distance

State transitions happen at end-of-day based on cumulative P&L. The strategy itself doesn't know about Topstep rules — it just asks "how big should I trade?" and the state manager answers.

**Hardcoded thresholds in v1.** The simulator will empirically find optimal values — premature configurability adds complexity without value.

---

## 5. Signal Generators

Both strategies subclass the existing `Strategy` ABC. Since v1 uses daily bars, both approximate intraday behavior using daily OHLC.

### ORBStrategy (Opening Range Breakout)

- **Concept:** Uses daily bar structure as a proxy for opening range breakout behavior
- **Entry logic:** If `close > open` by a threshold (bullish day structure), go long. If `close < open` (bearish), go short. Threshold scaled by ATR to filter noise.
- **Position sizing:** Base size × `state_manager.get_position_multiplier()`, capped by Topstep position limits
- **Stop:** Opposite end of day's range × `state_manager.get_stop_multiplier()`
- **Daily target:** $1,500 at base sizing (50% of $3K profit target — consistency sweet spot)
- **Warm-up period:** 5 bars (need ATR for volatility-based sizing)
- **Indicators:** ATR(14), SMA(5) for trend filter

### VWAPReversionStrategy (Mean Reversion)

- **Concept:** Fade deviations from fair value, using 5-day SMA as VWAP proxy on daily bars
- **Entry logic:** When price deviates > 1 ATR from 5-SMA, enter toward the mean
- **Position sizing:** Base size × `state_manager.get_position_multiplier()`, capped by limits
- **Stop:** 1.5 ATR from entry × `state_manager.get_stop_multiplier()`
- **Daily target:** $300-500 per trade, multiple trades per day modeled via daily range analysis
- **Warm-up period:** 20 bars (need SMA + ATR history)
- **Indicators:** SMA(5), SMA(20), ATR(14), Bollinger Bands(20, 2)

**Both strategies accept the StateManager** via constructor injection. The state manager is optional — passing `None` disables adaptive sizing (for A/B comparison of state machine on vs off).

---

## 6. Campaign Runner

### CampaignRunner

The outer loop that runs mass evaluation attempts:

**Input:**
- `strategy_class`: ORBStrategy or VWAPReversionStrategy
- `instrument`: "MES" or "MNQ"
- `state_machine_enabled`: bool
- `topstep_config`: account tier configuration
- `num_attempts`: 1,000 (default)
- `data_range`: full available history for the instrument
- `seed`: random seed for reproducibility

**Process:**
For each attempt:
1. Pick random start date (must have ≥ `max_attempt_days` trading days of data ahead)
2. Instantiate fresh `TopstepEvalSimulator` with clean state
3. Run day-by-day until PASS, FAIL, or TIMEOUT
4. Record: outcome, days_traded, final_pnl, best_day, worst_day, state_transitions, daily_equity_curve

**Attempt model:** Independent random-start. Each attempt picks a random date from available history. Attempts are statistically independent — no carry-over between attempts. This measures "if I start an account on a random day, what's my pass rate?"

### CampaignResult

```python
@dataclass
class CampaignResult:
    campaign_id: str
    strategy_name: str
    instrument: str
    state_machine_enabled: bool
    topstep_config: TopstepConfig
    num_attempts: int
    seed: int

    # Pass-rate metrics (front and center)
    pass_rate: float                    # passes / attempts
    avg_days_to_pass: float             # among passes
    avg_days_to_fail: float             # among failures
    ev_per_attempt: float               # expected $ value per $49 attempt
    cost_to_funded: float               # avg subscription spend before first pass
    median_attempts_to_pass: float      # median attempts needed
    annual_ev: float                    # projected at 4 attempts/month

    # Distribution data (drill-down)
    attempt_outcomes: list[AttemptOutcome]  # per-attempt detail
    pnl_distribution: list[float]       # final P&L per attempt
    days_distribution: list[int]        # days per attempt
    state_usage: dict[str, float]       # % time in each state
    pass_by_regime: dict[str, float]    # pass rate per market regime
```

### EV Calculation

```
gross_payout_per_pass = min(profit_target × payout_split, max_payout)
cost_per_attempt = subscription_fee
cost_per_pass = activation_fee

ev_per_attempt = (pass_rate × gross_payout_per_pass) - cost_per_attempt - (pass_rate × cost_per_pass)

cost_to_funded = subscription_fee / pass_rate
attempts_per_month = 4  # assumption: one new account per week
annual_ev = ev_per_attempt × attempts_per_month × 12
```

---

## 7. Data Layer

### Instruments

v1 targets MES (Micro E-mini S&P 500) and MNQ (Micro E-mini Nasdaq 100).

**Data source for v1:** FMP API. ES and NQ continuous contract daily bars via `/api/v3/historical-price-full/ES` and `/api/v3/historical-price-full/NQ`. Stored in the existing `daily_bars` table.

**Futures contract specifics:**
- MES: $5 per point per contract (micro). ES: $50 per point.
- MNQ: $2 per point per contract (micro). NQ: $20 per point.
- The strategy sizes in micros by default (matching Topstep's micro-focused position limits).

**Intraday-ready architecture:** The `TopstepEvalSimulator` operates on bars from the existing `DataFeed` interface. Swapping FMP daily bars for Databento/Polygon intraday bars requires only a new `DataFeed` implementation — no changes to the simulator, strategies, or evaluation logic.

### Database Schema

One new table added to `data/storage/database.py`:

```sql
CREATE TABLE IF NOT EXISTS eval_campaigns (
    campaign_id     TEXT PRIMARY KEY,
    strategy_name   TEXT NOT NULL,
    instrument      TEXT NOT NULL,
    state_machine   BOOLEAN NOT NULL DEFAULT 1,
    topstep_config  TEXT NOT NULL,        -- JSON
    num_attempts    INTEGER NOT NULL,
    seed            INTEGER,
    pass_rate       REAL,
    ev_per_attempt  REAL,
    cost_to_funded  REAL,
    avg_days_to_pass REAL,
    annual_ev       REAL,
    created_at      TIMESTAMP,
    full_results    TEXT                  -- JSON blob of CampaignResult
);

CREATE INDEX IF NOT EXISTS idx_eval_campaigns_strategy
    ON eval_campaigns (strategy_name);
```

Follows the same pattern as the existing `runs` table — summary fields for quick querying, JSON blob for full detail.

---

## 8. Backend API

New router: `backend/routers/eval.py`

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/eval/campaigns | List all campaigns (paginated, sortable) |
| GET | /api/eval/campaigns/{id} | Single campaign with full results |
| DELETE | /api/eval/campaigns/{id} | Delete campaign |
| POST | /api/eval/campaigns/run | Launch campaign (background thread) |
| GET | /api/eval/campaigns/{id}/status | Check running campaign progress |
| POST | /api/eval/campaigns/{id}/stop | Cancel running campaign |

**POST /api/eval/campaigns/run** request body:

```json
{
    "strategy": "orb",
    "instrument": "MES",
    "state_machine_enabled": true,
    "account_tier": "50k",
    "num_attempts": 1000,
    "seed": 42
}
```

Returns `{ "campaign_id": "abc123", "status": "running" }` immediately. Campaign executes in a background thread — same pattern as the existing backtest launcher in `backend/routers/backtest.py`.

**Response schemas** follow the existing `backend/schemas.py` pattern:
- `CampaignResponse` — summary for list view
- `CampaignDetailResponse` — full results with distributions
- `CampaignStatusResponse` — progress during execution

---

## 9. Dashboard Pages

### Campaign Browser (`/eval`)

**Metrics strip (top):** 6 KPIs across all campaigns:
- Best Pass Rate (strategy name below)
- Best EV/Attempt (strategy name below)
- Cost to Funded (attempt count below)
- Avg Days to Pass (among passes)
- Annual EV (at 4 attempts/mo)
- Campaigns Run (configs tested)

**Campaign table:** Sortable DataTable with columns: campaign_id, strategy, state_machine (ON/OFF), attempts, pass_rate, ev_per_attempt, cost_to_funded, avg_days, created_at. Click row → drill-down. Color coding: pass rate green >30%, yellow 20-30%, red <20%. EV green if positive, red if negative.

**Run Campaign button:** Opens form modal with strategy selector, instrument, state machine toggle, account tier, num_attempts, seed.

### Campaign Detail (`/eval/[campaignId]`)

**Metrics strip:** 4 large KPIs:
- Pass Rate (X / N attempts below)
- EV per Attempt (cost breakdown below)
- Median Attempts to Pass (cost-to-funded below)
- Projected Annual EV (at 4 attempts/mo)

**Charts (2×2 grid):**
1. **Attempt Equity Fan Chart** — All attempt equity curves as percentile bands (P5/P25/P50/P75/P95). Horizontal lines at +$3K (PASS, green) and -$2K (FAIL, red). Reuses existing fan chart component pattern.
2. **Pass Rate by Market Regime** — Horizontal bar chart: BULL (green), BEAR (red), SIDEWAYS (yellow), HIGH_VOL (purple). Uses existing regime detection from `analytics/regime.py`.
3. **Days to Resolution Distribution** — Histogram. Green bars = passes (clustered early, days 2-5). Red bars = failures (spread wider). Shows how quickly the strategy resolves.
4. **State Machine Usage** — Horizontal bar chart showing % of total trading time spent in each of the 6 states. Reveals whether the bot spends most time in NORMAL or frequently enters YOLO/HAIL_MARY.

### Design Language

Follows the existing dashboard design spec exactly:
- Dark theme (#09090b background, #0f0f0f panels, #18181b cards)
- Geist Mono for all numbers, Geist Sans for labels
- 1px solid borders (#1a1a1a)
- Color system: green (#22c55e) for positive, red (#ef4444) for negative, orange (#f97316) for accent, yellow (#eab308) for warnings, purple (#a855f7) for high volatility
- Sidebar: new "Eval" item between "Data" and "Paper"

---

## 10. Project Structure

```
trading-bot/
├── topstep/                              # NEW — all evaluation bot code
│   ├── __init__.py
│   ├── config.py                         # TopstepConfig dataclass, preset tiers
│   ├── evaluation_rules.py               # Pass/fail logic, trailing drawdown, consistency
│   ├── state_manager.py                  # 6-state machine, multipliers
│   ├── attempt_tracker.py                # Per-attempt P&L tracking, daily accounting
│   ├── simulator.py                      # TopstepEvalSimulator (wraps BacktestEngine)
│   ├── campaign_runner.py                # Runs N attempts, aggregates stats
│   └── strategies/
│       ├── __init__.py
│       ├── orb_strategy.py               # Opening Range Breakout
│       └── vwap_reversion_strategy.py    # VWAP Mean Reversion
├── backend/
│   └── routers/
│       └── eval.py                       # NEW — campaign API endpoints
├── frontend/
│   └── app/
│       ├── eval/
│       │   └── page.tsx                  # NEW — Campaign Browser
│       └── eval/[campaignId]/
│           └── page.tsx                  # NEW — Campaign Detail
├── tests/
│   ├── test_evaluation_rules.py          # NEW
│   ├── test_state_manager.py             # NEW
│   ├── test_simulator.py                 # NEW
│   ├── test_campaign_runner.py           # NEW
│   ├── test_orb_strategy.py             # NEW
│   └── test_vwap_strategy.py            # NEW
├── data/storage/
│   └── database.py                       # MODIFY — add eval_campaigns table
├── core/                                 # EXISTING — untouched
├── strategy/                             # EXISTING — untouched
├── execution/                            # EXISTING — untouched
├── portfolio/                            # EXISTING — untouched
├── analytics/                            # EXISTING — untouched
└── indicators/                           # EXISTING — untouched
```

---

## 11. Testing Strategy

| Layer | Test Type | What's Tested |
|-------|----------|---------------|
| EvaluationRules | Unit | Trailing drawdown calculation, consistency check, pass/fail detection, edge cases (exactly at limit) |
| StateManager | Unit | State transitions for all 6 states, multiplier values, boundary conditions |
| AttemptTracker | Unit | Daily P&L accumulation, high-water mark updates, status transitions |
| TopstepEvalSimulator | Integration | Full evaluation attempt against fixture data — verify pass when target hit, fail when drawdown breached |
| ORBStrategy | Unit | Signal generation on known bar patterns, position sizing with state manager |
| VWAPReversionStrategy | Unit | Mean reversion signals on known deviation patterns |
| CampaignRunner | Integration | Run 100 attempts on fixture data, verify pass rate calculation, EV math, reproducibility with seed |
| API endpoints | Integration | Campaign CRUD, launch, status polling |

Test fixtures: deterministic bar data for ES/NQ (trending days for ORB, ranging days for VWAP) stored in `tests/fixtures/`.

---

## 12. Build Phases

### Phase 0: Data Prerequisites
- Fetch ES and NQ daily bars via FMP into existing database
- Add `eval_campaigns` table to database schema

### Phase 1: Core Evaluation Logic (3 parallel agents)
1. EvaluationRules + AttemptTracker + tests
2. StateManager + tests
3. TopstepEvalSimulator (wraps engine) + tests

### Phase 2: Strategies (2 parallel agents)
1. ORBStrategy + tests
2. VWAPReversionStrategy + tests

### Phase 3: Campaign Runner + Backend
1. CampaignRunner + aggregation logic + tests
2. Backend router + schemas

### Phase 4: Dashboard
1. Campaign Browser page (`/eval`)
2. Campaign Detail page (`/eval/[campaignId]`)

Total: ~10 agent deployments across 5 phases.
