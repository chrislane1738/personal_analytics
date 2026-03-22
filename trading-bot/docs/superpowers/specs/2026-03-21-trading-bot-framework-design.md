# Trading Bot Backtesting & Execution Framework — Design Spec

**Date:** 2026-03-21
**Status:** Draft
**Author:** Chris Lane + Claude

---

## 1. Overview

A proprietary trading bot framework for backtesting and executing swing trading strategies on US equities and options. Built around an event-driven architecture where the same strategy code runs identically in backtest, paper, and live modes. Data sourced from FMP (Premium tier, 750 req/min, 30yr daily history). Local SQLite for storage. Full analytics suite with regime analysis and Monte Carlo simulation.

### Goals

- **Backtest with confidence** — deep data validation pipeline ensures no bad data contaminates results
- **Strategy flexibility** — config-driven YAML for simple strategies, full Python classes for complex logic (both plug into the same engine)
- **Complete transparency** — every trade, every signal, every risk decision is logged and reportable
- **Live-ready architecture** — same strategy code transitions from backtest → paper → live without modification
- **Anti-overfitting** — walk-forward validation and Monte Carlo simulation built into the optimization pipeline

### Non-Goals (for now)

- Web dashboard / UI (CLI-first)
- High-frequency / sub-daily trading (swing trading on daily bars)
- Multi-asset beyond equities + options (no crypto/forex in v1)
- Distributed computing / cloud deployment

---

## 2. Architecture

### Event-Driven Core

An event bus sits at the center. Every component communicates through typed events. The backtest loop fires historical bars as events; live mode fires real-time bars as the same events.

```
DataFeed (FMP/Schwab) → BarEvent → Strategy → SignalEvent → RiskManager → OrderEvent → Broker → FillEvent → Portfolio
                                                                                         ↑
                                                                        (Simulated OR Live)
```

**Event Types:**

| Event | Payload | Emitted By | Consumed By |
|-------|---------|------------|-------------|
| `BarEvent` | symbol, date, open, high, low, close, adj_close, volume, vwap | DataFeed | Strategy, Analytics |
| `SignalEvent` | symbol, direction (long/short/sell_put/sell_call/buy_to_close/sell_to_close), reason, strength (0.0-1.0), option_type (call/put/None), strike, expiration | Strategy | RiskManager |
| `OrderEvent` | symbol, action (BUY/SELL/BTO/STO/BTC/STC), order_type (market/limit/stop/stop_limit), quantity, price, option_type, strike, expiration | RiskManager | Broker |
| `FillEvent` | symbol, action, quantity, fill_price, commission, slippage, timestamp | Broker | Portfolio, Analytics |
| `RiskEvent` | rule_name, signal_blocked (ref to SignalEvent), reason | RiskManager | Analytics |
| `PortfolioUpdateEvent` | equity, cash, unrealized_pnl, realized_pnl, drawdown_pct, timestamp | Portfolio | Analytics, RiskManager (circuit breaker) |

**Key distinctions:**
- `BarEvent` carries raw OHLCV only. Indicators are computed by the strategy's indicator pipeline and accessed via `self.indicators`, not attached to the bar.
- `OrderEvent` separates **action** (BUY, SELL, BTO, STO, BTC, STC) from **order_type** (market, limit, stop, stop_limit). These are orthogonal — an options order has both.
- `PortfolioUpdateEvent` is emitted after each `FillEvent` is processed. This provides point-in-time equity snapshots for the analytics layer and feeds the drawdown circuit breaker.

All events are timestamped and serializable. The Analytics logger subscribes to ALL event types for a complete audit trail.

### Engine Modes

| Mode | Clock | DataFeed | Broker | Persistence |
|------|-------|----------|--------|-------------|
| **BACKTEST** | Replays historical dates | SQLite (pre-cached) | SimBroker | Run saved to `runs` |
| **PAPER** | Real wall time | FMP polling (daily after close) | SimBroker at real prices | Portfolio persisted in DB |
| **LIVE** | Real wall time | Schwab/WebSocket stream | LiveBroker (Schwab API) | Portfolio persisted, kill switch |

---

## 3. Project Structure

```
trading-bot/
├── config/
│   ├── settings.py              # Global config (FMP key, DB path, defaults)
│   └── strategies/              # YAML strategy configs
│       └── mean_reversion.yaml
├── core/
│   ├── engine.py                # Main backtest/live engine loop
│   ├── events.py                # Event types: BarEvent, SignalEvent, OrderEvent, FillEvent
│   ├── event_bus.py             # Pub/sub event bus
│   └── clock.py                 # Unified time source (historical replay or real-time)
├── data/
│   ├── feeds/
│   │   ├── base.py              # DataFeed interface
│   │   ├── fmp_feed.py          # FMP historical data feed
│   │   ├── live_feed.py         # Live market data feed (Schwab/WebSocket)
│   │   └── csv_feed.py          # Load from local CSV
│   ├── storage/
│   │   ├── database.py          # SQLite schema, connection management
│   │   ├── cache.py             # Smart caching — only fetch missing ranges
│   │   └── models.py            # Data models (DailyBar, OptionsChain, etc.)
│   └── validation/
│       ├── validators.py        # Gap detection, zero-volume, split verification
│       ├── cleaners.py          # Data repair (forward-fill, interpolation flagging)
│       └── quality_report.py    # Per-symbol data quality scoring
├── strategy/
│   ├── base.py                  # Strategy ABC
│   ├── config_strategy.py       # YAML-driven strategy loader
│   ├── registry.py              # Strategy discovery and registration
│   └── library/                 # Built-in strategies
│       ├── mean_reversion.py
│       ├── momentum.py
│       └── options_spread.py
├── execution/
│   ├── broker.py                # Broker interface
│   ├── sim_broker.py            # Simulated fills (slippage, commission models)
│   ├── live_broker.py           # Live broker adapter (Schwab API stub)
│   └── order_types.py           # Market, Limit, Stop, StopLimit, options orders
├── risk/
│   ├── manager.py               # Position sizing, exposure limits, circuit breaker
│   ├── position.py              # Position tracking (equity + options greeks)
│   └── rules.py                 # Configurable risk rules
├── portfolio/
│   ├── portfolio.py             # Holdings, cash, P&L tracking
│   ├── accounting.py            # Transaction log, cost basis, realized/unrealized P&L
│   └── benchmark.py             # Benchmark comparison (SPY default)
├── analytics/
│   ├── metrics.py               # Sharpe, Sortino, max drawdown, Calmar, etc.
│   ├── trade_log.py             # Full trade history with entry/exit reasons
│   ├── reports.py               # HTML/console report generation
│   ├── regime.py                # Bull/bear/sideways regime detection
│   ├── monte_carlo.py           # Trade-order randomization stress testing
│   └── options_analytics.py     # Greeks exposure tracking, premium analysis
├── indicators/
│   ├── base.py                  # Indicator interface
│   ├── technical.py             # SMA, EMA, RSI, MACD, Bollinger (local calc)
│   ├── fmp_indicators.py        # Pull pre-computed indicators from FMP
│   └── custom.py                # User-defined indicator template
├── utils/
│   ├── logging.py               # Structured logging for all events
│   ├── rate_limiter.py          # FMP API rate limiter (700 req/min with headroom)
│   └── helpers.py               # Date utils, symbol normalization
├── tests/
│   ├── test_engine.py
│   ├── test_data_validation.py
│   ├── test_strategies.py
│   └── fixtures/                # Sample data for deterministic tests
├── db/
│   └── trading_bot.db           # SQLite database (gitignored)
├── reports/                     # Generated HTML reports (gitignored)
├── .env                         # FMP_API_KEY + config (gitignored)
├── .env.example                 # Template for env vars
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── main.py                      # CLI entry point
```

---

## 4. Data Layer

### 4.1 SQLite Schema

```sql
-- Core price data
daily_bars (
    symbol TEXT,
    date DATE,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    vwap REAL,
    data_quality_score REAL,  -- 0.0-1.0, computed on ingest
    PRIMARY KEY (symbol, date)
)

-- Options data
options_chains (
    symbol TEXT,
    date DATE,
    expiration DATE,
    strike REAL,
    option_type TEXT,  -- 'call' or 'put'
    last_price REAL,
    bid REAL,
    ask REAL,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    PRIMARY KEY (symbol, date, expiration, strike, option_type)
)

-- Fundamentals
fundamentals (
    symbol TEXT,
    date DATE,
    period TEXT,  -- 'annual' or 'quarterly'
    revenue REAL,
    net_income REAL,
    eps REAL,
    pe_ratio REAL,
    debt_to_equity REAL,
    free_cash_flow REAL,
    roe REAL,
    raw_json TEXT,  -- full FMP response preserved
    PRIMARY KEY (symbol, date, period)
)

-- Locally computed indicator cache
indicators_cache (
    symbol TEXT,
    date DATE,
    indicator_name TEXT,
    value REAL,
    params TEXT,  -- Canonical JSON: sorted keys, no whitespace e.g. {"period":20}
    PRIMARY KEY (symbol, date, indicator_name, params)
)

-- Symbol metadata (sector, industry, exchange)
symbol_metadata (
    symbol TEXT PRIMARY KEY,
    company_name TEXT,
    sector TEXT,
    industry TEXT,
    exchange TEXT,
    market_cap REAL,
    updated_at TIMESTAMP
    -- Populated via FMP /api/v3/profile/{symbol} endpoint
)

-- Runs (backtest, paper, and live)
runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT,  -- 'backtest', 'paper', 'live'
    strategy_name TEXT,
    config TEXT,  -- JSON of full strategy config
    start_date DATE,
    end_date DATE,
    initial_capital REAL,
    final_value REAL,
    total_return REAL,
    sharpe REAL,
    max_drawdown REAL,
    created_at TIMESTAMP,
    full_metrics TEXT  -- JSON blob of all analytics
)

-- Trade log
trades (
    trade_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs,
    symbol TEXT,
    direction TEXT,
    entry_date DATE,
    exit_date DATE,
    entry_price REAL,
    exit_price REAL,
    quantity INTEGER,  -- whole shares for equities; contracts for options
    pnl REAL,
    pnl_pct REAL,
    entry_reason TEXT,
    exit_reason TEXT,
    option_type TEXT,  -- NULL for equities
    strike REAL,
    expiration DATE
)

-- Data quality audit trail
data_quality_log (
    symbol TEXT,
    date DATE,
    issue_type TEXT,  -- 'gap', 'zero_volume', 'spike', 'split_mismatch'
    severity TEXT,    -- 'warning', 'error'
    details TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (symbol, date, issue_type)
)
```

### 4.2 Data Validation Pipeline

Every bar passes through these checks on ingest, before database insertion:

1. **Schema Validation** — correct fields, types, no nulls in required fields
2. **Range Checks** — price > 0, volume >= 0, high >= low, high >= open/close
3. **Gap Detection** — missing trading days flagged (US market holiday calendar excluded)
4. **Spike Detection** — daily move > 50% flagged for review (could be legit or bad data)
5. **Split Verification** — close-to-close ratio compared against FMP stock split history endpoint
6. **Volume Anomalies** — zero volume on trading days, volume > 10x 20-day avg
7. **Adjusted Price Check** — verify adj_close aligns with known dividend history
8. **Quality Score** — 0.0-1.0 composite score stored with each bar
9. **Write to SQLite** — clean data stored, quality issues logged to `data_quality_log`

Strategies can filter on quality score (e.g., "only backtest bars with quality > 0.8").

### 4.3 Smart Caching

```
Request: "Get AAPL daily bars 2020-01-01 to 2025-12-31"

Check SQLite: "What date ranges exist for AAPL?"
├── Full range cached       → return from DB (zero API calls)
├── Partial gap found       → fetch only the missing date range from FMP
└── No data for this symbol → full fetch, validate, store
```

Data is fetched once and cached permanently. Subsequent backtests for the same symbol/dates are instant with no API cost.

### 4.4 FMP Rate Limiter

Token-bucket rate limiter capped at 700 req/min (50 req/min headroom below the 750 limit). Batch quote endpoints (`/api/v3/quote/AAPL,GOOG,MSFT`) used where possible to minimize calls. Bandwidth consumption tracked against the 50GB/month cap.

### 4.5 FMP Tier Constraints (Premium, ~$79/mo)

| Capability | Available | Constraint |
|-----------|-----------|-----------|
| Daily OHLCV | Yes, 30 years | — |
| Intraday | 5min, 15min, 30min, 1hr | No 1-minute bars (Ultimate only) |
| Technical indicators (API) | Yes | We compute locally to save API calls |
| Financial statements | Annual + quarterly | — |
| Options chains | Yes | — |
| Stock screener | Yes | — |
| Earnings/economic calendar | Yes | No earnings call transcripts |
| Batch quotes | Yes, comma-separated | Bulk EOD: 1 req / 10 sec |
| 13F / ETF holdings | No | Ultimate only |
| WebSocket real-time | Unconfirmed at Premium tier | Live data feed will use Schwab API as primary source; FMP WebSocket as optional fallback if available |

---

## 5. Strategy Layer

### 5.1 Strategy Interface

```python
class Strategy(ABC):
    """Base class for all strategies."""

    @abstractmethod
    def generate_signals(self, bar: BarEvent, portfolio: Portfolio) -> list[SignalEvent]:
        """Given a new bar for a single symbol and current portfolio state, return trade signals.

        The engine calls this once per symbol per date. For multi-symbol context
        (e.g., relative strength across the universe), use on_universe_bar() instead.
        """

    def on_universe_bar(self, bars: dict[str, BarEvent], portfolio: Portfolio) -> list[SignalEvent]:
        """Optional — called once per date with ALL symbols' bars for that date.

        Override this instead of generate_signals() when the strategy needs
        cross-symbol context (pairs trading, relative strength, sector rotation).
        Default implementation delegates to generate_signals() per symbol.
        """
        signals = []
        for symbol, bar in bars.items():
            signals.extend(self.generate_signals(bar, portfolio))
        return signals

    def on_fill(self, fill: FillEvent) -> None:
        """Optional — react to fills (e.g., set stop-losses after entry)."""

    def on_start(self, context: BacktestContext) -> None:
        """Optional — initialize indicators, warm-up periods."""

    def on_end(self, portfolio: Portfolio) -> list[SignalEvent]:
        """Optional — called at end of backtest/session. Use to force-close
        open positions, flush state, or emit closing signals."""
        return []

    def warm_up_period(self) -> int:
        """Bars needed before strategy can generate signals."""
        return 0
```

**Engine iteration contract:** For each trading date, the engine collects all bars for the universe, then calls `on_universe_bar(bars, portfolio)`. The default implementation iterates symbols and delegates to `generate_signals()` per symbol. Strategies that need cross-symbol context override `on_universe_bar()` directly.

Both config-driven and Python strategies implement this interface. The engine doesn't know which type it's running — both produce `SignalEvent` objects.

### 5.2 Config-Driven Strategy (YAML)

```yaml
name: "Mean Reversion SPY"
universe: ["SPY"]
timeframe: daily

indicators:
  sma_20: { type: SMA, period: 20 }
  sma_50: { type: SMA, period: 50 }
  rsi_14: { type: RSI, period: 14 }
  bb_20:  { type: BollingerBands, period: 20, std_dev: 2 }

entry_rules:
  - condition: "close < bb_20.lower AND rsi_14 < 30"
    direction: long
    reason: "Price below lower BB with oversold RSI"

exit_rules:
  - condition: "close > sma_20"
    reason: "Price reverted to mean"
  - condition: "pnl_pct < -0.05"
    reason: "Stop loss at -5%"

position_sizing:
  method: fixed_pct
  value: 0.06  # 6% of portfolio per trade
```

The `ConfigStrategy` class parses this YAML, builds the indicator pipeline, and evaluates conditions using a safe expression parser (no `eval()`).

**Expression Parser Specification:**

Supported operators: `<`, `>`, `<=`, `>=`, `==`, `!=`, `AND`, `OR`, `NOT`

Available variables in conditions:
- **Bar fields:** `open`, `high`, `low`, `close`, `adj_close`, `volume`, `vwap`
- **Indicator values:** referenced by name (e.g., `rsi_14`, `sma_20`). Sub-properties via dot notation (e.g., `bb_20.lower`, `bb_20.upper`, `bb_20.middle`)
- **Portfolio fields:** `pnl_pct` (unrealized P&L % for current position), `holding_days` (days since entry), `position_size` (current position value)
- **Numeric literals and arithmetic:** `close * 0.95`, `sma_20 + 2.0`

Error handling: If a condition references an undefined indicator or variable, the condition evaluates to `False` and a warning is logged. Parse errors at strategy load time raise an exception immediately (fail fast).

### 5.3 Python Strategy (for complex logic)

```python
class OptionsWheelStrategy(Strategy):
    """Sell cash-secured puts, take assignment, sell covered calls."""

    def generate_signals(self, bar, portfolio):
        signals = []
        if not portfolio.has_position(bar.symbol):
            if self.indicators['rsi_14'].value < 40:
                signals.append(SignalEvent(
                    symbol=bar.symbol, direction='sell_put',
                    strike=bar.close * 0.95,
                    expiration=self.next_monthly_expiry(bar.date),
                    reason="Oversold, sell CSP at 95% strike"
                ))
        else:
            signals.append(SignalEvent(
                symbol=bar.symbol, direction='sell_call',
                strike=bar.close * 1.05,
                expiration=self.next_monthly_expiry(bar.date),
                reason="Sell CC at 105% strike"
            ))
        return signals
```

### 5.4 Strategy Registry

Strategies are registered by name. The CLI resolves strategy references:

- `--strategy config/strategies/mean_reversion.yaml` → loads YAML config
- `--strategy strategy.library.options_wheel` → loads Python class by module path

---

## 6. Execution Layer

### 6.1 Broker Interface

```python
class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: OrderEvent) -> None:
        """Submit an order for execution."""

    @abstractmethod
    def get_pending_orders(self) -> list[OrderEvent]:
        """Return unfilled orders."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
```

### 6.2 SimBroker (Backtesting)

- **Fill model:** Fill at next bar's open price (realistic default) or current bar's close (optimistic, configurable)
- **Slippage model:** Fixed percentage (default 0.01%) or volume-based (larger orders get more slippage)
- **Commission model:** Per-share (default $0.005) or per-trade (configurable)
- **Options simulation model:**
  - **Pricing:** Mark options positions to market daily using stored `options_chains` data when available. When historical chain data is unavailable for a given date, fall back to Black-Scholes pricing using the underlying's price and historical implied volatility.
  - **Bid-ask spread:** Options fills include a configurable spread cost (default: 50% of mid-to-ask distance for sells, 50% of bid-to-mid for buys). Options spreads are significantly wider than equities.
  - **Assignment:** Model assignment at expiration for ITM options. Early exercise modeled for deep ITM calls approaching ex-dividend dates.
  - **Greeks:** Updated daily from chain data or Black-Scholes. Portfolio-level Greeks exposure tracked for risk management.
  - **Historical options data caveat:** FMP historical options chain coverage may be limited in depth (strikes/expirations) for dates far in the past. The framework logs when it falls back to Black-Scholes synthetic pricing so the user knows which periods use modeled vs. real chain data.
- Emits `FillEvent` with actual fill price including all costs

### 6.3 LiveBroker (Future — Schwab API)

Stub interface ready for implementation:

- Submit orders via Schwab API
- Poll for fill confirmations
- Handle partial fills (emit multiple FillEvents)
- Kill switch: flatten all positions on command

### 6.4 Order Model

Orders have two orthogonal dimensions:

**Order Action** (what you're doing):

| Action | Code | Description |
|--------|------|-------------|
| Buy | BUY | Buy equity shares |
| Sell | SELL | Sell equity shares |
| Buy to Open | BTO | Open a long options position |
| Sell to Open | STO | Open a short options position (write) |
| Buy to Close | BTC | Close a short options position |
| Sell to Close | STC | Close a long options position |

**Order Type** (how it's priced):

| Type | Equities | Options |
|------|----------|---------|
| Market | Yes | Yes |
| Limit | Yes | Yes |
| Stop | Yes | Yes |
| Stop-Limit | Yes | Yes |

An `OrderEvent` carries both: e.g., action=STO + type=Limit = "sell to open a put at a limit price."

---

## 7. Risk Management

The RiskManager sits between Strategy → Broker. Every signal must pass all active rules before becoming an order.

### Default Risk Rules

| Rule | Default Value | Description |
|------|--------------|-------------|
| Max position size | 6% of portfolio | Maximum allocation to any single position |
| Max sector concentration | 25% of portfolio | Maximum allocation to any single sector |
| Max open positions | 20 | Cap on concurrent trades |
| Drawdown circuit breaker | -15% from peak | Halt all new entries if portfolio drops 15% from high-water mark |
| Max options notional | 30% of portfolio | Cap on total notional options exposure |
| Max delta exposure | configurable | Net portfolio delta limit |
| Correlation guard | warn at r > 0.8 | Alert when adding highly correlated positions |

All rules are configurable per strategy via YAML or constructor args. When a signal is blocked, a `RiskEvent` is emitted with the specific rule and reason — visible in the analytics log.

---

## 8. Analytics & Reporting

### 8.1 Core Metrics

**Performance:**
- Total return, annualized return, CAGR
- Sharpe ratio (risk-free rate = 10yr Treasury, configurable)
- Sortino ratio (downside deviation only)
- Calmar ratio (return / max drawdown)
- Profit factor (gross wins / gross losses)

**Risk:**
- Max drawdown (% and $), drawdown duration, time to recovery
- Annualized volatility
- Beta vs benchmark (SPY default)
- Value at Risk (95th and 99th percentile)
- Ulcer Index

**Trade Statistics:**
- Total trades, win rate, avg win %, avg loss %
- Expectancy per trade
- Largest single win / loss
- Average holding period (days)
- Consecutive wins/losses streaks

**Position-Level:**
- P&L per symbol, per sector
- Holding period distribution
- Best/worst performing tickers
- Long vs short split

**Options-Specific:**
- Total premium collected vs paid
- Assignment rate
- Greeks exposure over time (delta, gamma, theta, vega curves)
- Win rate by DTE bucket
- Theta decay P&L vs directional P&L

### 8.2 Regime Analysis

Market regime detection applied to benchmark (SPY):

| Regime | Definition |
|--------|-----------|
| Bull | SMA-50 > SMA-200, positive slope |
| Bear | SMA-50 < SMA-200, negative slope |
| Sideways | SMA-50 ≈ SMA-200 (within 2%), low slope |
| High Volatility | VIX > 25 or realized vol > 20% |

**VIX data source:** VIX index data fetched from FMP via `/api/v3/historical-price-full/^VIX` and stored in `daily_bars` like any other symbol. If VIX data is unavailable for a date range, the regime detector falls back to realized volatility only (20-day rolling std dev of SPY returns, annualized).

Strategy performance is reported per regime: trades, win rate, Sharpe, max drawdown. This reveals *when* a strategy works and when it should sit out.

### 8.3 Monte Carlo Simulation

Stress-tests edge robustness:

1. Take actual trade results from backtest
2. Randomly shuffle trade order 10,000 times (configurable)
3. For each shuffle, replay the equity curve
4. Output: median final equity, 5th/95th percentile confidence bands, probability of ruin, max drawdown distribution

If the actual equity curve is an outlier vs random shuffles, the returns may depend on lucky sequencing rather than a real edge.

### 8.4 Report Formats

**Console:** Quick text summary after every run — return, Sharpe, max DD, win rate, profit factor.

**HTML:** Full interactive report (Plotly charts):
- Equity curve with drawdown overlay
- Trade markers on price chart (entry/exit points)
- Monthly returns heatmap
- Rolling Sharpe over time
- Regime performance table
- Monte Carlo confidence bands
- Complete sortable/filterable trade log
- Options Greeks exposure timeline

Reports saved to `reports/` and linked from `runs` table.

### 8.5 Benchmark Comparison

Every metric computed for both the strategy AND a buy-and-hold benchmark (SPY default). The report always answers: **"Did this strategy beat just holding the index?"**

---

## 9. Parameter Optimization & Walk-Forward Validation

### 9.1 Grid Search

```bash
python main.py backtest optimize --strategy mean_reversion \
                                 --param rsi_period:10,14,20 \
                                 --param sma_period:15,20,30 \
                                 --start 2015-01-01
```

Runs all parameter combinations, records metrics for each, outputs a ranked table. Flags if optimal params are at grid edges (overfitting signal).

### 9.2 Walk-Forward Validation

Prevents overfitting by splitting the test period into rolling train/test windows:

```
Window 1: [2015───train───2019] [2019─test─2020]
Window 2: [2016───train───2020] [2020─test─2021]
Window 3: [2017───train───2021] [2021─test─2022]
Window 4: [2018───train───2022] [2022─test─2023]
Window 5: [2019───train───2023] [2023─test─2024]
```

For each window: optimize on train period, validate on test period. Reports in-sample vs out-of-sample performance for each window. Significant degradation = overfitting detected.

**Optimization objective:** Default objective is Sharpe ratio (maximize risk-adjusted returns). Configurable via `--objective` flag: `sharpe` (default), `sortino`, `calmar`, `total_return`, `min_drawdown`. The objective function is the same metric used to rank parameter combinations in grid search.

---

## 10. CLI Interface

```bash
# Data Management
python main.py data fetch AAPL --start 2015-01-01 --end 2025-12-31
python main.py data fetch-universe sp500 --start 2015-01-01
python main.py data validate AAPL
python main.py data validate --all --report
python main.py data status

# Backtesting
python main.py backtest --strategy config/strategies/mean_reversion.yaml \
                        --start 2015-01-01 --end 2025-12-31 --capital 100000
python main.py backtest --strategy strategy.library.options_wheel \
                        --universe AAPL,MSFT,GOOGL --start 2020-01-01
python main.py backtest compare --strategies mean_reversion,momentum \
                                --start 2015-01-01 --end 2025-12-31
python main.py backtest optimize --strategy mean_reversion \
                                 --param rsi_period:10,14,20 --param sma_period:15,20,30

# Reporting
python main.py report last
python main.py report --run-id abc123 --format html
python main.py report list
python main.py report monte-carlo --run-id abc123 --simulations 10000

# Paper Trading
python main.py paper start --strategy mean_reversion --capital 100000
python main.py paper status
python main.py paper stop
```

### Configuration Hierarchy

```
Defaults (settings.py) → .env (environment variables) → Strategy config (YAML) → CLI flags
```

Each layer overrides the previous. CLI flags always win — an explicit command-line argument is never silently overridden by a forgotten `.env` value. Everything has a sensible default.

---

## 11. Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.11+ | Ecosystem (pandas, numpy), your existing expertise |
| Database | SQLite | Zero-config, sufficient for single-user backtesting |
| Data source | FMP API (Premium) | 30yr history, 750 req/min, options + fundamentals |
| Data processing | pandas, numpy | Industry standard for financial data |
| Charts (HTML reports) | Plotly | Interactive, no server needed |
| CLI framework | Typer | Clean subcommand structure, type hints, auto-generated help |
| HTTP client | httpx (async) | Rate limiting, connection pooling |
| Config parsing | PyYAML + Pydantic | Validated configs with type safety |
| Expression parser | Custom (no eval) | Safe condition evaluation for YAML strategies |
| Testing | pytest | Standard, fixtures for deterministic tests |

### Dependencies (requirements.txt)

```
pandas>=2.0
numpy>=1.24
scipy>=1.11          # Black-Scholes pricing, statistical functions
plotly>=5.18
httpx>=0.25
pyyaml>=6.0
pydantic>=2.5
typer>=0.9
python-dotenv>=1.0
pytest>=7.4
```

---

## 12. Testing Strategy

| Layer | Test Type | What's Tested |
|-------|----------|---------------|
| Data validation | Unit | Each validator (gaps, spikes, splits) with known-bad fixtures |
| Indicators | Unit | SMA, RSI, MACD output against hand-calculated values |
| Strategy signals | Unit | Known bar sequences → expected signals |
| Risk manager | Unit | Position limit enforcement, circuit breaker triggering |
| SimBroker | Unit | Fill logic, slippage, commission calculations |
| Event bus | Integration | Events flow correctly from DataFeed → Portfolio |
| Full backtest | Integration | Known dataset → deterministic expected result |
| YAML parser | Unit | Config-driven strategy loading + condition evaluation |
| Data caching | Integration | Fetch, cache, re-fetch returns cached data |
| Walk-forward validation | Integration | Known dataset where optimal in-sample params degrade out-of-sample; verify framework detects overfitting |
| Optimizer | Integration | Grid search produces correct parameter rankings; edge-parameter warning fires |

Test fixtures: small, deterministic datasets (e.g., 100 bars for AAPL with known outcomes) stored in `tests/fixtures/`.

---

## 13. Dividend Handling

Dividends are reflected through adjusted close prices only — they are **not** credited as separate cash payments to the portfolio. This is the standard approach for backtesting:

- `adj_close` accounts for both splits and dividends, providing a continuous return series
- P&L calculations use `adj_close` for accurate total return including dividends
- The `close` field (split-adjusted only) is available for strategies that need to reason about actual trading prices (e.g., options strike selection)

This means the backtest return implicitly includes dividend returns. If a future version needs to model dividend capture strategies explicitly, a `dividends` table can be added using FMP's `/api/v3/historical-stock-dividend/{symbol}` endpoint.

---

## 14. v1 Definition of Done

The minimum viable system for v1 is complete when:

1. **Data pipeline works end-to-end:** Fetch daily bars from FMP for any US equity, validate, cache in SQLite. Data quality report runs and flags issues.
2. **One YAML strategy backtests successfully:** Mean reversion (or similar) config-driven strategy runs against SPY with correct signals, fills, and P&L.
3. **One Python strategy backtests successfully:** Options wheel (or similar) class-based strategy runs with simulated options pricing.
4. **Console report:** After every backtest — return, Sharpe, Sortino, max drawdown, win rate, profit factor, benchmark comparison.
5. **HTML report:** Full interactive Plotly report with equity curve, trade markers, monthly heatmap, trade log.
6. **Risk manager enforces rules:** Position sizing (6%), sector limits, drawdown circuit breaker all functional and logged.
7. **Monte Carlo simulation:** Runs on completed backtest, produces confidence bands and probability of ruin.
8. **Regime analysis:** Segments backtest into bull/bear/sideways/high-vol and reports per-regime stats.
9. **Parameter optimization:** Grid search across 2+ params with walk-forward validation detecting overfitting.
10. **All tests pass:** Unit and integration tests for each layer.

Paper trading and live trading are **not** required for v1 but the architecture must support them without redesign.

---

## 15. Future Roadmap (out of scope for v1)

- **Schwab live broker integration** — plug into LiveBroker interface
- **Multi-timeframe strategies** — combine daily + weekly signals
- **Pairs trading / market-neutral strategies** — long/short with correlation logic
- **Web dashboard** — Next.js frontend for visualizing backtests (leverage Viking Fund Dashboard patterns)
- **Alerting** — email/SMS on paper trading signals
- **Supabase sync** — persist backtest results to cloud for cross-device access
