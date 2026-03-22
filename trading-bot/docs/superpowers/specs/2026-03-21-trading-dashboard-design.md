# Trading Bot Dashboard — Design Spec

**Date:** 2026-03-21
**Status:** Approved
**Scope:** Next.js dashboard for the trading bot backtesting framework

---

## 1. Purpose

Build a full control center dashboard for the trading bot framework. The dashboard replaces the CLI + static HTML reports as the primary interface for:

- Browsing and analyzing backtest results
- Launching new backtests with configurable parameters
- Editing and creating trading strategies (YAML)
- Monitoring backtests in real-time via event streaming
- Managing market data (fetch, validate, browse)
- Placeholder for future Schwab paper trading integration

Personal tool — no authentication, runs locally.

---

## 2. Architecture

```
┌─────────────────────────────────────┐
│          Next.js Frontend           │
│  (App Router, shadcn/ui, Recharts)  │
│  Port 3000                          │
├─────────────────────────────────────┤
│  TanStack Query  │  WebSocket Client│
└────────┬─────────┴────────┬─────────┘
         │ REST              │ WS
         ▼                   ▼
┌─────────────────────────────────────┐
│          FastAPI Backend            │
│  Port 8000                          │
├─────────────────────────────────────┤
│  /api/runs       - CRUD on runs     │
│  /api/trades     - Trade queries    │
│  /api/strategies - List/CRUD YAML   │
│  /api/data       - Symbol mgmt     │
│  /api/backtest   - Launch runs      │
│  /api/analytics  - Metrics/MC/etc   │
│  /ws/monitor     - Live event stream│
├─────────────────────────────────────┤
│  Existing Python Modules            │
│  (Engine, Analytics, DB, Data Feed) │
└────────────────┬────────────────────┘
                 │
         ┌───────┴───────┐
         │   SQLite DB   │
         │ trading_bot.db│
         └───────────────┘
```

### Key decisions

- **FastAPI is a thin wrapper** — imports and calls existing modules directly (BacktestEngine, Database, FMPFeed, analytics). No logic rewrite.
- **WebSocket endpoint** hooks into the EventBus during live backtest runs, streaming events as JSON.
- **Next.js uses TanStack Query** for all REST calls (auto-caching, refetching) and a raw WebSocket client for the live monitor.
- **No reverse proxy** — direct port access for personal use.
- **SQLite stays** — the existing database with 8 tables (daily_bars, options_chains, fundamentals, indicators_cache, symbol_metadata, runs, trades, data_quality_log).

### Project structure

```
trading-bot/
├── backend/                  # NEW — FastAPI wrapper
│   ├── main.py               # FastAPI app, CORS config, router includes
│   ├── routers/
│   │   ├── runs.py           # GET /api/runs, GET /api/runs/{id}, DELETE
│   │   ├── trades.py         # GET /api/trades?run_id=X
│   │   ├── strategies.py     # GET/PUT/POST/DELETE /api/strategies
│   │   ├── data.py           # GET /api/data/symbols, POST /api/data/fetch
│   │   ├── backtest.py       # POST /api/backtest/run
│   │   └── analytics.py      # GET /api/analytics/{run_id}/monte-carlo, etc.
│   ├── schemas.py            # Pydantic response/request models
│   ├── services.py           # Thin wrappers around existing modules
│   └── ws_manager.py         # WebSocket connection manager + EventBus bridge
├── frontend/                 # NEW — Next.js dashboard
│   ├── app/
│   │   ├── layout.tsx        # Root layout with sidebar nav + event ticker
│   │   ├── page.tsx          # Run Browser (home page)
│   │   ├── runs/
│   │   │   └── [runId]/
│   │   │       └── page.tsx  # Run Detail
│   │   ├── compare/
│   │   │   └── page.tsx      # Run Comparison
│   │   ├── strategies/
│   │   │   └── page.tsx      # Strategy Config Editor
│   │   ├── launch/
│   │   │   └── page.tsx      # Backtest Launcher
│   │   ├── monitor/
│   │   │   └── page.tsx      # Live Event Monitor
│   │   ├── data/
│   │   │   └── page.tsx      # Data Manager
│   │   └── paper/
│   │       └── page.tsx      # Paper Trading (placeholder)
│   ├── components/
│   │   ├── ui/               # shadcn/ui components
│   │   ├── charts/           # Recharts wrappers (EquityCurve, Drawdown, Heatmap, etc.)
│   │   ├── layout/           # Sidebar, EventTicker, MetricsStrip
│   │   └── tables/           # DataTable configurations for runs, trades, symbols
│   ├── hooks/
│   │   ├── use-runs.ts       # TanStack Query hooks for runs API
│   │   ├── use-trades.ts     # TanStack Query hooks for trades API
│   │   ├── use-analytics.ts  # TanStack Query hooks for analytics API
│   │   ├── use-strategies.ts # TanStack Query hooks for strategies API
│   │   ├── use-data.ts       # TanStack Query hooks for data API
│   │   └── use-websocket.ts  # WebSocket connection hook for live monitor
│   ├── lib/
│   │   ├── api.ts            # Axios/fetch client configured for localhost:8000
│   │   ├── types.ts          # TypeScript types mirroring backend schemas
│   │   ├── utils.ts          # Formatters (currency, percent, date)
│   │   └── theme.ts          # Chart color constants, theme config
│   └── ...
├── analytics/                # EXISTING — untouched
├── core/                     # EXISTING — untouched
├── config/                   # EXISTING — strategies read/written by backend
├── data/                     # EXISTING — untouched
├── db/                       # EXISTING — untouched
├── execution/                # EXISTING — untouched
├── indicators/               # EXISTING — untouched
├── portfolio/                # EXISTING — untouched
├── risk/                     # EXISTING — untouched
├── strategy/                 # EXISTING — untouched
├── reports/                  # EXISTING — untouched
├── tests/                    # EXISTING — untouched
├── utils/                    # EXISTING — untouched
└── main.py                   # EXISTING CLI — untouched
```

---

## 3. Design Language

**Theme:** Dense Bloomberg-meets-modern. Dark background (#09090b), monospace for all numbers/data (Geist Mono), sans-serif for labels/headings (Geist Sans). 1px solid borders (#1a1a1a) separating panels. Minimal border-radius (4-6px on cards).

**Color system:**
- Background: #09090b (page), #0f0f0f (panels), #18181b (elevated cards)
- Borders: #1a1a1a (subtle), #27272a (emphasis)
- Text: #fafafa (primary), #a1a1aa (secondary), #71717a (muted), #525252 (dim)
- Green: #22c55e (profit, positive metrics, BUY signals)
- Red: #ef4444 (loss, negative metrics, risk blocks)
- Blue: #3b82f6 (benchmark, info, ORDER events)
- Orange: #f97316 (brand accent, FILL events, active nav)
- Yellow: #eab308 (warnings, SIDEWAYS regime)
- Purple: #a855f7 (HIGH_VOL regime, PORTFOLIO events)

**Typography:**
- Labels/headings: Geist Sans, 9-12px, uppercase with letter-spacing for section headers
- Data/numbers: Geist Mono, 10-18px, tabular-nums
- Event ticker: Geist Mono, 10px

**Layout:** Grid-based with 1px gap borders (not card-with-margin). Panels fill space. Event ticker fixed at page bottom across all pages.

---

## 4. Page Designs

### 4.1 Run Browser (`/` — Home Page)

Full-width DataTable of all backtest runs.

**Columns:**
| Column | Type | Notes |
|--------|------|-------|
| run_id | string | Truncated to 8 chars, monospace, click to navigate |
| strategy_name | string | Filterable dropdown |
| start_date | date | — |
| end_date | date | — |
| initial_capital | currency | Monospace |
| final_value | currency | Monospace |
| total_return | percent | Color-coded green/red |
| sharpe | number | Color-coded (>1 green, <0 red) |
| max_drawdown | percent | Color-coded red intensity |
| trade_count | number | — |
| created_at | datetime | Relative time ("2h ago") |

**Features:**
- Sort on any column (default: created_at desc)
- Filter by strategy name (dropdown) and date range (date pickers)
- Bulk select + delete
- Export selected/all as CSV
- Click row → `/runs/{runId}`

**API:** `GET /api/runs?sort=created_at&order=desc&strategy=X&limit=50&offset=0`

### 4.2 Run Detail (`/runs/[runId]`)

The densest page. Mirrors the approved mockup.

**Layout:**
```
┌─────────────────────────────────────────────────┐
│ Run Header: strategy | dates | capital | run_id  │
├────┬────┬────┬────┬────┬────┬────┬──────────────┤
│Ret │Shrp│Sort│Calm│MaxD│WinR│PF  │Expectancy    │
├──────────────────────┬──────────────────────────┤
│  Equity Curve        │  Regime Performance      │
│  (strategy+benchmark)│  (per-regime stats)      │
├──────────────────────┼──────────────────────────┤
│  Trade Log Table     │  Drawdown Chart          │
│  (sortable, filtered)│  + Risk Metrics panel    │
├──────────────────────┴──────────────────────────┤
│  Tabs: Monthly Heatmap | Monte Carlo | Options  │
└─────────────────────────────────────────────────┘
```

**Metrics strip:** 8 KPIs in a grid row — total_return, sharpe, sortino, calmar, max_drawdown, win_rate, profit_factor, expectancy. All from `runs.full_metrics` JSON.

**Equity curve:** Recharts AreaChart with two Line series (strategy green, benchmark blue). Gradient fill under strategy line. Interactive tooltip showing date + both values. X-axis: dates. Y-axis: dollar value.

**Regime performance:** Colored dot per regime (BULL green, BEAR red, SIDEWAYS yellow, HIGH_VOL purple). Per-regime: trade count, win rate, avg P&L, best trade, worst trade.

**Trade log:** DataTable with columns: date, direction (BUY green / SELL red), symbol, entry_price, quantity, exit_price, P&L ($), P&L (%), holding_days, entry_reason, exit_reason. Sortable, filterable. Expandable rows for full reason text. Option type/strike/expiration columns shown if options trades present.

**Drawdown chart:** Recharts AreaChart, inverted (red gradient fill down from 0%). Tooltip shows date + drawdown %. Peak drawdown annotated.

**Risk metrics panel:** VaR (95%), Calmar, Annualized Volatility, Beta, Alpha (if benchmark).

**Tabs:**
- Monthly Returns Heatmap: grid of Year (rows) x Month (columns), cells colored RdYlGn by monthly return %
- Monte Carlo: fan chart + stats (see 4.5)
- Options Analytics: premium/Greeks/DTE (see 4.6, only shown for options runs)

**APIs:**
- `GET /api/runs/{id}` — run metadata + full_metrics JSON
- `GET /api/trades?run_id={id}` — all trades for this run
- `GET /api/analytics/{id}/regime` — regime statistics
- `GET /api/analytics/{id}/equity-curve` — date + strategy_value + benchmark_value arrays

### 4.3 Run Comparison (`/compare`)

Side-by-side analysis of 2-4 selected runs.

**Controls:** Multi-select dropdown populated from runs list. "Compare" button.

**Overlaid equity curves:** Single Recharts chart with one Line series per selected run (different colors). Normalized to % return (not absolute dollar) so different capital amounts are comparable. Legend shows run_id + strategy + color.

**Metrics comparison table:** Rows = metrics (return, sharpe, sortino, calmar, max_dd, win_rate, profit_factor, expectancy, trade_count). Columns = selected runs. Best value per row highlighted green, worst highlighted red.

**Trade distribution:** Bar chart showing P&L distribution per run (histogram buckets).

**API:** Same as Run Detail APIs, called for each selected run.

### 4.4 Regime Analysis (Tab within Run Detail, also `/runs/[runId]/regime`)

**Regime timeline:** Horizontal stacked bar spanning the backtest date range. Each segment colored by regime (BULL/BEAR/SIDEWAYS/HIGH_VOL). Hover shows regime + date range + duration.

**Per-regime stat cards:** 4 cards (one per regime), each showing: trade_count, win_rate, avg_pnl, best_trade, worst_trade. Card border-left colored by regime.

**Regime-filtered trade table:** Click a regime card to filter the trade log to only trades executed during that regime.

**API:** `GET /api/analytics/{id}/regime` — returns regime classifications + per-regime statistics.

### 4.5 Monte Carlo (Tab within Run Detail, also `/runs/[runId]/monte-carlo`)

**Fan chart:** Recharts AreaChart with stacked bands for P5-P25 (light red), P25-P50 (light), P50-P75 (light green), P75-P95 (green). Actual equity curve overlaid as a solid line. If actual result is outside P5-P95, show an "OUTLIER" badge.

**Stats panel:**
- Simulations: 10,000
- Median final equity: $X
- 5th percentile: $X
- 95th percentile: $X
- Probability of ruin: X%
- Outlier: Yes/No

**Drawdown distribution:** Histogram of max drawdowns across all simulations. Vertical line showing actual max drawdown.

**API:** `GET /api/analytics/{id}/monte-carlo?simulations=10000` — runs on demand, result cached in memory. Returns:
- `percentile_bands`: `{ "p5": [...], "p25": [...], "p50": [...], "p75": [...], "p95": [...] }` — equity value at each time step for each percentile (powers the fan chart)
- `actual_curve`: `[...]` — the real equity values for overlay
- `final_equity_distribution`: `[...]` — histogram of final equity values across simulations
- `drawdown_distribution`: `[...]` — histogram of max drawdowns
- `summary`: `{ median, p5, p95, probability_of_ruin, is_outlier, simulations }`

Note: The existing `run_monte_carlo()` only returns final equities. Phase 0 extends it to track percentile bands per time step.

### 4.6 Options Analytics (Tab within Run Detail, conditional)

Only rendered if the run contains options trades (check trades for non-null option_type).

**Premium summary cards:** 3 cards — Total Collected, Total Paid, Net Premium. Green/red color coding.

**Assignment metrics:** Total short options, assignments, assignment rate (%). Progress bar visual.

**Win rate by DTE bucket:** Recharts BarChart with 4 bars (0-7d, 7-30d, 30-60d, 60+d). Each bar shows win rate %. Color intensity scales with win rate.

**Greeks timeline:** Recharts LineChart with 4 series (delta, gamma, theta, vega) over time. Toggle individual Greeks on/off.

**API:** `GET /api/analytics/{id}/options` — returns OptionsAnalyticsResult data.

### 4.7 Strategy Config Editor (`/strategies`)

**Left panel (40%):** List of all YAML strategy files from `config/strategies/`. Click to select. "New Strategy" button at top.

**Right panel (60%):** Monaco editor (react-monaco-editor) with YAML syntax highlighting. Dark theme matching dashboard.

**Features:**
- Live YAML validation — red squiggles on parse errors
- Save button → `PUT /api/strategies/{name}` — writes file to disk
- Delete button → `DELETE /api/strategies/{name}` — with confirmation dialog
- New strategy → template picker modal (blank, mean reversion, momentum, options wheel) → generates starter YAML with comments
- Side reference panel (collapsible): available indicator types + params, condition expression syntax, position sizing methods

**API:**
- `GET /api/strategies` — list all strategy files (name + content)
- `GET /api/strategies/{name}` — single file content
- `PUT /api/strategies/{name}` — update file
- `POST /api/strategies` — create new file
- `DELETE /api/strategies/{name}` — delete file

### 4.8 Backtest Launcher (`/launch`)

Form-based page for launching new backtests.

**Form fields:**
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| Strategy | Select | — | Dropdown: YAML files + Python paths |
| Universe | Multi-select | — | Symbols from symbol_metadata, with search |
| Start Date | Date picker | 1yr ago | Constrained to available data range |
| End Date | Date picker | today | Constrained to available data range |
| Initial Capital | Number | 100,000 | Step: 1000 |
| Benchmark | Select | SPY | Dropdown of cached symbols |
| Position Size % | Slider + number | 6% | Range 1-25% |
| Slippage % | Number | 0.01% | Advanced toggle |
| Commission/share | Number | $0.005 | Advanced toggle |

**Launch flow:**
1. User fills form, clicks "Run Backtest"
2. POST `/api/backtest/run` with config → returns `{ run_id, status: "running" }`
3. Auto-redirect to Live Monitor page with run_id
4. If user doesn't want to watch: "Run in Background" option, shows toast when complete

**Recent launches table** below the form: last 5 launches with status (running/completed/failed), clickable.

**API:** `POST /api/backtest/run` — body: strategy, universe, start_date, end_date, initial_capital, benchmark, position_size_pct, slippage_pct, commission_per_share. Runs BacktestEngine in a background thread. Returns run_id immediately.

### 4.9 Live Event Monitor (`/monitor`)

Real-time dashboard during active backtest execution. Connects via WebSocket.

**Layout:**
```
┌─────────────────────────────┬───────────────────────┐
│                             │  Portfolio State       │
│  Live Equity Curve          │  Cash: $X              │
│  (updates per bar event)    │  Equity: $X            │
│                             │  Positions table       │
│                             │  Unrealized P&L        │
├─────────────────────────────┼───────────────────────┤
│  Status: RUNNING            │  Event Feed            │
│  Progress: 2024-06-15       │  (scrolling log)       │
│  [=========>    ] 65%       │  SIGNAL BUY SPY...     │
│  Elapsed: 2.3s              │  ORDER MKT BUY...      │
│                             │  FILL SPY 450.28...    │
│  [Stop] [View Results]      │  RISK blocked TSLA...  │
└─────────────────────────────┴───────────────────────┘
```

**Live equity curve:** Recharts LineChart that appends data points as PortfolioUpdateEvents arrive. Starts empty, grows as the backtest progresses.

**Portfolio state panel:** Updated on every PortfolioUpdateEvent. Shows current cash, total equity, and a small table of open positions (symbol, quantity, avg_cost, unrealized_pnl, current_price).

**Event feed:** Scrolling log, newest at bottom (auto-scroll). Each event is one line with:
- Colored tag: SIGNAL (green), ORDER (blue), FILL (orange), RISK (red), PORT (purple)
- Timestamp (the simulated date)
- Event details (symbol, action, price, quantity, reason, etc.)
- Filter toggles to show/hide event types

**Status bar:** Running/Completed/Error state. Progress bar based on (current_date - start_date) / (end_date - start_date). Elapsed wall-clock time. Stop button (cancels the backtest). "View Full Results" button appears on completion.

**WebSocket protocol:**
```json
// Client → Server
{ "action": "subscribe", "run_id": "abc123" }
{ "action": "stop", "run_id": "abc123" }

// Server → Client
{ "type": "bar", "data": { "symbol": "SPY", "date": "2024-06-15", ... } }
{ "type": "signal", "data": { "symbol": "SPY", "direction": "long", ... } }
{ "type": "order", "data": { "symbol": "SPY", "action": "BUY", ... } }
{ "type": "fill", "data": { "symbol": "SPY", "fill_price": 450.28, ... } }
{ "type": "risk", "data": { "rule_name": "MaxDrawdown", "reason": "...", ... } }
{ "type": "portfolio", "data": { "equity": 124710, "cash": 42180, ... } }
{ "type": "progress", "data": { "current_date": "2024-06-15", "pct": 0.65 } }
{ "type": "complete", "data": { "run_id": "abc123" } }
{ "type": "error", "data": { "message": "..." } }
```

**WebSocket resilience:**
- Server sends `{ "type": "ping" }` every 15 seconds; client responds with `{ "type": "pong" }`. If no pong in 30s, server closes connection.
- Client auto-reconnects with exponential backoff (1s, 2s, 4s, max 10s). On reconnect, sends `{ "action": "subscribe", "run_id": "..." }` to resume.
- No event replay on reconnect — client shows a "Reconnected, some events may have been missed" toast. The equity curve and portfolio state are re-synced via a full state snapshot sent on subscribe.
- **Throttling:** Backend batches events and sends at most 1 message per 50ms (20 events/sec max). BarEvents are summarized (only latest per symbol), while Signal/Fill/Risk events are always forwarded individually.

**API:** `WS /ws/monitor` — bidirectional WebSocket. Backend subscribes to EventBus, serializes events to JSON, forwards to all connected clients for the active run.

### 4.10 Data Manager (`/data`)

**Cached symbols table:** DataTable with columns: symbol, company_name, sector, industry, exchange, market_cap, date range (min-max of daily_bars), bar_count, quality_score, last_updated. Sortable, searchable.

**Fetch panel:**
- Symbol input (with autocomplete from known symbols)
- Date range pickers
- "Fetch" button → POST `/api/data/fetch` → shows progress toast → refreshes table
- Respects FMP rate limits (700 req/min)

**Validation panel:**
- Select symbol(s) from table
- "Run Validation" button → POST `/api/data/validate`
- Shows data_quality_log entries: issue_type, severity (warning/error), details, date
- Issues: missing bars, price outliers, stale data, split adjustments

**Storage stats card:** Total bars, total symbols, DB file size, last fetch timestamp.

**APIs:**
- `GET /api/data/symbols` — all symbol_metadata with bar counts
- `POST /api/data/fetch` — { symbol, start_date, end_date } → fetches from FMP, caches to DB
- `POST /api/data/validate` — { symbols } → runs validation pipeline
- `GET /api/data/quality?symbol=X` — quality log entries

### 4.11 Paper Trading Panel (`/paper`) — v2 Placeholder

Simple page with:
- Header: "Paper Trading — Coming Soon"
- Brief description: "Live paper trading with Schwab API integration"
- Strategy selector (dropdown, disabled)
- Start/Stop controls (disabled, with tooltip "Schwab integration not yet configured")
- Mock status display showing what the live view will look like
- Link to CLI `trading-bot paper` commands (which also show "not implemented")

No backend work needed — purely a frontend placeholder page.

---

## 5. Backend API Summary

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/runs | List all runs (paginated, sortable, filterable) |
| GET | /api/runs/{id} | Single run with full_metrics |
| DELETE | /api/runs/{id} | Delete run + associated trades |
| GET | /api/trades | Query trades (filter by run_id, symbol, date) |
| GET | /api/analytics/{id}/equity-curve | Equity + benchmark arrays |
| GET | /api/analytics/{id}/regime | Regime classifications + stats |
| GET | /api/analytics/{id}/monte-carlo | Run Monte Carlo simulation |
| GET | /api/analytics/{id}/options | Options analytics |
| GET | /api/strategies | List all YAML strategy files |
| GET | /api/strategies/{name} | Get strategy file content |
| PUT | /api/strategies/{name} | Update strategy file |
| POST | /api/strategies | Create new strategy file |
| DELETE | /api/strategies/{name} | Delete strategy file |
| POST | /api/backtest/run | Launch backtest (returns run_id) |
| GET | /api/backtest/status/{id} | Check run status |
| POST | /api/backtest/stop/{id} | Cancel running backtest |
| GET | /api/data/symbols | List cached symbols with metadata |
| POST | /api/data/fetch | Fetch data from FMP |
| POST | /api/data/validate | Run validation on symbols |
| GET | /api/data/quality | Query data quality log |

### WebSocket

| Path | Protocol | Description |
|------|----------|-------------|
| /ws/monitor | JSON messages | Live event streaming during backtest |

### Backend implementation notes

- FastAPI routers import directly from existing modules: `from data.storage.database import Database`, `from core.engine import BacktestEngine`, etc.
- Backtest runs execute in a background thread (or asyncio task) so the API stays responsive.
- **Cancellation mechanism:** Add a `threading.Event` cancellation token to `BacktestEngine`. The engine checks `cancel_event.is_set()` on each date iteration. `POST /api/backtest/stop/{id}` sets the event. On cancellation, the engine saves partial results before exiting.
- EventBus bridge: register a subscriber on the engine's EventBus that serializes events and pushes to WebSocket connections.
- Monte Carlo results cached in a dict keyed by run_id (recomputed if params change).
- Strategy files read/written directly from `config/strategies/` directory using pathlib.
- CORS configured with `allow_origins=["http://localhost:3000"]`, `allow_methods=["*"]`, `allow_headers=["*"]`.
- **Error response schema:** All error responses use `{ "detail": "message", "code": "ERROR_CODE" }` format for consistent frontend handling.

---

## 6. Frontend Implementation Notes

### Shared components

- **Sidebar:** Fixed left, collapsible. Icons + labels for each section. Active state = orange left border.
- **EventTicker:** Fixed bottom bar across all pages. Shows last 5-10 events from the active WebSocket connection (live monitor). When no backtest is running, shows the last run's summary metrics instead. No separate API endpoint — state comes from WebSocket hook or TanStack Query cache.
- **MetricsStrip:** Reusable grid of KPI cards. Takes array of { label, value, format, color_rule }.
- **Chart wrappers:** EquityCurveChart, DrawdownChart, MonthlyHeatmap, FanChart, RegimeTimeline — each wraps Recharts with consistent dark theme styling.

### State management

- **TanStack Query** for all server data. Query keys: `['runs']`, `['runs', id]`, `['trades', { run_id }]`, `['analytics', id, 'regime']`, etc.
- **No global client state store needed** — URL params + TanStack Query cache cover all cases.
- **WebSocket state** managed by a custom `useWebSocket` hook that maintains connection, handles reconnect, and exposes event stream + status.

### Theming

- shadcn/ui initialized with `zinc` base color, dark mode default.
- Tailwind config extended with the color system from Section 3.
- CSS variables for chart colors so Recharts components reference them consistently.
- Geist Sans via `next/font` for body, Geist Mono via `next/font` for data.

---

## 7. Build Phases

### Phase 0: Persistence Prerequisites
Patch existing Python modules so the database contains all data the dashboard needs. **Must complete before Phase 1.**

**Engine patches (core/engine.py):**
- Call `database.insert_trades(run_id, trade_log.trades)` in `_save_run()` to persist all trades
- Persist equity curve data: add `equity_curves` table to schema (run_id, date, strategy_value, benchmark_value) and save in `_save_run()`
- Store benchmark_symbol in `RunRecord.config` JSON (already partially there, ensure consistent)
- Add `threading.Event` cancellation token — check `cancel_event.is_set()` each date iteration
- Fix `full_metrics` serialization: store numeric values as numbers, not strings (`json.dumps({k: str(v) ...})` → proper types)

**Database patches (data/storage/database.py):**
- Add `equity_curves` table: `(run_id TEXT, date TEXT, strategy_value REAL, benchmark_value REAL, PRIMARY KEY (run_id, date))`
- Add `insert_equity_curve(run_id, curve_data)` method
- Add `get_equity_curve(run_id)` method
- Add `delete_run(run_id)` method — deletes from `runs`, `trades`, and `equity_curves` (cascade)
- Add `list_runs(sort, order, strategy_filter, limit, offset)` with pagination/filtering
- Add `get_trades(run_id)` query method
- Add `get_aggregate_quality_score(symbol)` method (avg of per-bar quality scores)

**Analytics patches:**
- `monte_carlo.py`: Extend `run_monte_carlo()` to return percentile bands per time step (`percentile_bands: dict[str, list[float]]` for P5/P25/P50/P75/P95) so the fan chart can be rendered, not just final equity distribution
- `regime.py`: Extend `compute_regime_stats()` to include `best_trade` and `worst_trade` per regime

**Metric naming:** Standardize metric keys across the codebase. The canonical names used by the API and frontend will be: `total_return`, `sharpe`, `sortino`, `calmar`, `max_drawdown`, `win_rate`, `profit_factor`, `expectancy`, `volatility`, `var_95`, `alpha`, `beta`. The backend schemas map from whatever the engine produces.

**Run existing tests** after all patches to ensure no regressions across the 714 test suite.

### Phase 1: Foundation
Backend API + Frontend Shell + Run Browser + Run Detail.

**Backend tasks:**
- FastAPI app setup (main.py, CORS, router structure)
- Runs router (list, get, delete)
- Trades router (query by run_id)
- Analytics router (equity curve, basic metrics)
- Pydantic schemas for all response types

**Frontend tasks:**
- Next.js app init + shadcn/ui + Tailwind + dark theme + Geist fonts
- Sidebar navigation component
- Event ticker component (static for now)
- Run Browser page with DataTable
- Run Detail page (metrics strip, equity curve, trade log, drawdown)

**Outcome:** Browse all existing backtests, drill into full interactive detail view.

### Phase 2: Deep Analytics
Run Comparison + Regime + Monte Carlo + Options Analytics.

**Backend tasks:**
- Monte Carlo endpoint (run_monte_carlo wrapper)
- Regime endpoint (detect_regimes + regime_statistics wrapper)
- Options analytics endpoint (compute_options_analytics wrapper)

**Frontend tasks:**
- Run Comparison page (multi-select, overlaid curves, metrics table)
- Regime analysis tab (timeline, stat cards, filtered trade table)
- Monte Carlo tab (fan chart, stats panel, drawdown histogram)
- Options analytics tab (premium cards, DTE chart, Greeks timeline)

**Outcome:** Full analytical depth on every backtest run.

### Phase 3: Control Center
Strategy Editor + Backtest Launcher + Live Monitor.

**Backend tasks:**
- Strategies router (CRUD for YAML files)
- Backtest run endpoint (background thread execution)
- Backtest status/stop endpoints
- WebSocket manager (EventBus bridge, connection handling)

**Frontend tasks:**
- Strategy editor page (Monaco, validation, templates)
- Backtest launcher page (form, validation, launch flow)
- Live monitor page (WebSocket client, live equity chart, event feed, portfolio state)

**Outcome:** Full control center — edit strategies, launch backtests, watch them execute live.

### Phase 4: Data & Polish
Data Manager + Paper Trading Stub + UX polish.

**Backend tasks:**
- Data router (symbol listing, FMP fetch wrapper, validation runner)
- Data quality log queries

**Frontend tasks:**
- Data manager page (symbol table, fetch panel, validation, storage stats)
- Paper trading placeholder page
- Loading skeletons for all pages
- Error boundaries and empty states
- Responsive adjustments (though primarily desktop)

**Outcome:** Complete dashboard with all 11 sections.

---

## 8. Subagent Strategy

Each phase will be executed with aggressive parallelization:

**Phase 0 (3 parallel agents):**
1. Engine persistence agent — patch `_save_run()` to persist trades + equity curves, add cancellation token, fix metric serialization
2. Database schema agent — new `equity_curves` table, `delete_run`, `list_runs` with pagination, `get_trades`, quality score aggregation
3. Analytics extension agent — Monte Carlo percentile bands, regime best/worst trade

**Phase 1 (4 parallel agents):**
1. Backend API agent — FastAPI app + runs/trades/analytics routers + schemas
2. Frontend shell agent — Next.js init + shadcn + theme + sidebar + ticker
3. Run Browser agent — DataTable page with all features
4. Run Detail agent — Full detail page with all panels

**Phase 2 (4 parallel agents):**
1. Run Comparison page agent
2. Regime Analysis page agent
3. Monte Carlo page agent
4. Options Analytics page agent

**Phase 3 (3 parallel agents):**
1. Strategy Editor agent (backend CRUD + Monaco frontend)
2. Backtest Launcher agent (backend run endpoint + frontend form)
3. Live Monitor agent (WebSocket backend + frontend streaming UI)

**Phase 4 (2 parallel agents):**
1. Data Manager agent (backend + frontend)
2. Polish agent (loading states, error boundaries, paper trading placeholder)

Total: ~16 agent deployments across 5 phases.
