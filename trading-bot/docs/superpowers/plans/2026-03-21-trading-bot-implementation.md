# Trading Bot Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an event-driven backtesting framework for swing trading US equities + options with FMP data, SQLite storage, and full analytics.

**Architecture:** Event bus at center — BarEvent → SignalEvent → OrderEvent → FillEvent → PortfolioUpdateEvent. Hybrid strategy system (YAML config + Python classes). Three modes: backtest, paper, live (v1 = backtest only, architecture supports all three).

**Tech Stack:** Python 3.11+, SQLite, pandas, numpy, scipy, Plotly, httpx, Typer, Pydantic, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-03-21-trading-bot-framework-design.md`

**Parallelization Guide:** Tasks are grouped into phases. Tasks within a phase have NO dependencies on each other and SHOULD be executed in parallel by separate agents. Tasks in later phases depend on earlier phases completing.

---

## Phase 0: Project Scaffolding (sequential — do first)

### Task 0: Project setup and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `core/__init__.py`
- Create: `data/__init__.py`, `data/feeds/__init__.py`, `data/storage/__init__.py`, `data/validation/__init__.py`
- Create: `strategy/__init__.py`, `strategy/library/__init__.py`
- Create: `execution/__init__.py`
- Create: `risk/__init__.py`
- Create: `portfolio/__init__.py`
- Create: `analytics/__init__.py`
- Create: `indicators/__init__.py`
- Create: `utils/__init__.py`
- Create: `tests/__init__.py`, `tests/fixtures/`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "trading-bot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "plotly>=5.18",
    "httpx>=0.25",
    "pyyaml>=6.0",
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "typer>=0.9",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-asyncio>=0.23"]

[project.scripts]
trading-bot = "main:app"
```

- [ ] **Step 2: Create requirements.txt**

```
pandas>=2.0
numpy>=1.24
scipy>=1.11
plotly>=5.18
httpx>=0.25
pyyaml>=6.0
pydantic>=2.5
pydantic-settings>=2.1
typer>=0.9
python-dotenv>=1.0
pytest>=7.4
pytest-asyncio>=0.23
```

- [ ] **Step 3: Create config/settings.py**

```python
"""Global configuration loaded from .env and defaults."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FMP
    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com/api"
    fmp_rate_limit: int = 700  # req/min, headroom below 750

    # Database
    db_path: str = "db/trading_bot.db"

    # Defaults
    default_benchmark: str = "SPY"
    default_capital: float = 100_000.0
    default_position_size: float = 0.06  # 6%
    default_max_sector_pct: float = 0.25
    default_max_positions: int = 20
    default_drawdown_limit: float = -0.15
    default_commission_per_share: float = 0.005
    default_slippage_pct: float = 0.0001  # 0.01%
    risk_free_rate: float = 0.04  # 4% default

    # Broker (future)
    schwab_client_id: str = ""
    schwab_client_secret: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 4: Create all __init__.py files and directory structure**

Create empty `__init__.py` in each package directory. Create `db/` and `reports/` and `tests/fixtures/` directories.

```bash
cd /Users/chrislane/Desktop/Claude_Code/trading-bot
mkdir -p config core data/feeds data/storage data/validation strategy/library execution risk portfolio analytics indicators utils tests/fixtures db reports config/strategies
touch config/__init__.py core/__init__.py data/__init__.py data/feeds/__init__.py data/storage/__init__.py data/validation/__init__.py strategy/__init__.py strategy/library/__init__.py execution/__init__.py risk/__init__.py portfolio/__init__.py analytics/__init__.py indicators/__init__.py utils/__init__.py tests/__init__.py
```

- [ ] **Step 5: Install dependencies**

```bash
cd /Users/chrislane/Desktop/Claude_Code/trading-bot
python -m pip install -r requirements.txt
```

- [ ] **Step 6: Commit**

```bash
git add trading-bot/pyproject.toml trading-bot/requirements.txt trading-bot/config/ trading-bot/core/ trading-bot/data/ trading-bot/strategy/ trading-bot/execution/ trading-bot/risk/ trading-bot/portfolio/ trading-bot/analytics/ trading-bot/indicators/ trading-bot/utils/ trading-bot/tests/
git commit -m "feat: scaffold trading-bot project structure and dependencies"
```

---

## Phase 1: Foundation Layer (sequential — event system + DB must exist before all else)

### Task 1: Event types and event bus

**Files:**
- Create: `core/events.py`
- Create: `core/event_bus.py`
- Create: `tests/test_events.py`
- Create: `tests/test_event_bus.py`

- [ ] **Step 1: Write failing tests for event types**

```python
# tests/test_events.py
from datetime import date, datetime
from core.events import (
    BarEvent, SignalEvent, OrderEvent, FillEvent,
    RiskEvent, PortfolioUpdateEvent, EventType
)


def test_bar_event_creation():
    bar = BarEvent(
        symbol="AAPL", date=date(2024, 1, 2),
        open=185.0, high=187.0, low=184.0,
        close=186.0, adj_close=185.5, volume=16_000_000, vwap=185.8
    )
    assert bar.symbol == "AAPL"
    assert bar.event_type == EventType.BAR
    assert bar.timestamp is not None


def test_signal_event_creation():
    sig = SignalEvent(
        symbol="AAPL", direction="long", reason="RSI oversold",
        strength=0.8, option_type=None, strike=None, expiration=None
    )
    assert sig.direction == "long"
    assert sig.event_type == EventType.SIGNAL


def test_order_event_creation():
    order = OrderEvent(
        symbol="AAPL", action="BUY", order_type="market",
        quantity=100, price=None, option_type=None,
        strike=None, expiration=None
    )
    assert order.action == "BUY"
    assert order.event_type == EventType.ORDER


def test_fill_event_creation():
    fill = FillEvent(
        symbol="AAPL", action="BUY", quantity=100,
        fill_price=186.02, commission=0.50, slippage=0.02
    )
    assert fill.fill_price == 186.02
    assert fill.event_type == EventType.FILL


def test_risk_event_creation():
    risk = RiskEvent(
        rule_name="max_position_size", signal_blocked_id=None,
        reason="Position would exceed 6% limit"
    )
    assert risk.event_type == EventType.RISK


def test_portfolio_update_event_creation():
    update = PortfolioUpdateEvent(
        equity=105_000.0, cash=50_000.0,
        unrealized_pnl=5_000.0, realized_pnl=0.0,
        drawdown_pct=-0.02
    )
    assert update.event_type == EventType.PORTFOLIO_UPDATE


def test_events_are_serializable():
    bar = BarEvent(
        symbol="AAPL", date=date(2024, 1, 2),
        open=185.0, high=187.0, low=184.0,
        close=186.0, adj_close=185.5, volume=16_000_000, vwap=185.8
    )
    d = bar.to_dict()
    assert d["symbol"] == "AAPL"
    assert "timestamp" in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.events'`

- [ ] **Step 3: Implement event types**

```python
# core/events.py
"""Typed event definitions for the trading bot event bus."""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
import uuid


class EventType(Enum):
    BAR = "bar"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    PORTFOLIO_UPDATE = "portfolio_update"


@dataclass
class BaseEvent:
    event_type: EventType = field(init=False)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        # Serialize all date/datetime fields recursively
        for key, val in d.items():
            if isinstance(val, datetime):
                d[key] = val.isoformat()
            elif isinstance(val, date):
                d[key] = val.isoformat()
        return d


@dataclass
class BarEvent(BaseEvent):
    symbol: str = ""
    date: date = field(default_factory=date.today)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    adj_close: float = 0.0
    volume: int = 0
    vwap: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.BAR


@dataclass
class SignalEvent(BaseEvent):
    symbol: str = ""
    direction: str = ""  # long/short/sell_put/sell_call/buy_to_close/sell_to_close
    reason: str = ""
    strength: float = 1.0  # 0.0 - 1.0
    option_type: str | None = None  # call/put/None
    strike: float | None = None
    expiration: date | None = None

    def __post_init__(self):
        self.event_type = EventType.SIGNAL


@dataclass
class OrderEvent(BaseEvent):
    symbol: str = ""
    action: str = ""  # BUY/SELL/BTO/STO/BTC/STC
    order_type: str = "market"  # market/limit/stop/stop_limit
    quantity: int = 0
    price: float | None = None
    option_type: str | None = None
    strike: float | None = None
    expiration: date | None = None
    signal_ref: str | None = None  # event_id of originating signal

    def __post_init__(self):
        self.event_type = EventType.ORDER


@dataclass
class FillEvent(BaseEvent):
    symbol: str = ""
    action: str = ""
    quantity: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    order_ref: str | None = None  # event_id of originating order

    def __post_init__(self):
        self.event_type = EventType.FILL


@dataclass
class RiskEvent(BaseEvent):
    rule_name: str = ""
    signal_blocked_id: str | None = None  # event_id of blocked SignalEvent
    reason: str = ""

    def __post_init__(self):
        self.event_type = EventType.RISK


@dataclass
class PortfolioUpdateEvent(BaseEvent):
    equity: float = 0.0
    cash: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    drawdown_pct: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.PORTFOLIO_UPDATE
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_events.py -v`
Expected: ALL PASS

- [ ] **Step 5: Write failing tests for event bus**

```python
# tests/test_event_bus.py
from core.events import BarEvent, EventType
from core.event_bus import EventBus
from datetime import date


def test_subscribe_and_emit():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.BAR, lambda e: received.append(e))

    bar = BarEvent(symbol="AAPL", date=date(2024, 1, 2),
                   open=185.0, high=187.0, low=184.0,
                   close=186.0, adj_close=185.5, volume=16_000_000, vwap=185.8)
    bus.emit(bar)

    assert len(received) == 1
    assert received[0].symbol == "AAPL"


def test_multiple_subscribers():
    bus = EventBus()
    r1, r2 = [], []
    bus.subscribe(EventType.BAR, lambda e: r1.append(e))
    bus.subscribe(EventType.BAR, lambda e: r2.append(e))

    bar = BarEvent(symbol="AAPL", date=date(2024, 1, 2),
                   open=185.0, high=187.0, low=184.0,
                   close=186.0, adj_close=185.5, volume=16_000_000, vwap=185.8)
    bus.emit(bar)

    assert len(r1) == 1
    assert len(r2) == 1


def test_unsubscribe():
    bus = EventBus()
    received = []
    handler = lambda e: received.append(e)
    bus.subscribe(EventType.BAR, handler)
    bus.unsubscribe(EventType.BAR, handler)

    bar = BarEvent(symbol="AAPL", date=date(2024, 1, 2),
                   open=185.0, high=187.0, low=184.0,
                   close=186.0, adj_close=185.5, volume=16_000_000, vwap=185.8)
    bus.emit(bar)

    assert len(received) == 0


def test_subscribe_all():
    bus = EventBus()
    received = []
    bus.subscribe_all(lambda e: received.append(e))

    bar = BarEvent(symbol="AAPL", date=date(2024, 1, 2),
                   open=185.0, high=187.0, low=184.0,
                   close=186.0, adj_close=185.5, volume=16_000_000, vwap=185.8)
    bus.emit(bar)

    assert len(received) == 1


def test_event_history():
    bus = EventBus()
    bar = BarEvent(symbol="AAPL", date=date(2024, 1, 2),
                   open=185.0, high=187.0, low=184.0,
                   close=186.0, adj_close=185.5, volume=16_000_000, vwap=185.8)
    bus.emit(bar)

    history = bus.get_history(EventType.BAR)
    assert len(history) == 1
```

- [ ] **Step 6: Implement event bus**

```python
# core/event_bus.py
"""Pub/sub event bus for routing typed events between components."""

from collections import defaultdict
from typing import Callable

from core.events import BaseEvent, EventType


class EventBus:
    def __init__(self):
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)
        self._all_subscribers: list[Callable] = []
        self._history: dict[EventType, list[BaseEvent]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    def subscribe_all(self, handler: Callable) -> None:
        self._all_subscribers.append(handler)

    def emit(self, event: BaseEvent) -> None:
        self._history[event.event_type].append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)
        for handler in self._all_subscribers:
            handler(event)

    def get_history(self, event_type: EventType) -> list[BaseEvent]:
        return list(self._history.get(event_type, []))

    def clear_history(self) -> None:
        self._history.clear()
```

- [ ] **Step 7: Run all tests**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_events.py tests/test_event_bus.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add trading-bot/core/events.py trading-bot/core/event_bus.py trading-bot/tests/test_events.py trading-bot/tests/test_event_bus.py
git commit -m "feat: add typed event system and pub/sub event bus"
```

---

### Task 2: SQLite database and data models

**Files:**
- Create: `data/storage/database.py`
- Create: `data/storage/models.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write failing tests for database and models**

```python
# tests/test_database.py
import os
import tempfile
from datetime import date
from data.storage.database import Database
from data.storage.models import DailyBar


def test_create_tables(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "daily_bars" in table_names
    assert "runs" in table_names
    assert "trades" in table_names
    assert "data_quality_log" in table_names
    assert "symbol_metadata" in table_names
    assert "options_chains" in table_names
    assert "fundamentals" in table_names
    assert "indicators_cache" in table_names
    db.close()


def test_insert_and_query_daily_bar(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    bar = DailyBar(
        symbol="AAPL", date=date(2024, 1, 2),
        open=185.0, high=187.0, low=184.0,
        close=186.0, adj_close=185.5, volume=16_000_000,
        vwap=185.8, data_quality_score=0.95
    )
    db.insert_daily_bars([bar])
    rows = db.get_daily_bars("AAPL", date(2024, 1, 1), date(2024, 1, 3))
    assert len(rows) == 1
    assert rows[0].symbol == "AAPL"
    assert rows[0].close == 186.0
    db.close()


def test_upsert_daily_bar(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    bar1 = DailyBar(symbol="AAPL", date=date(2024, 1, 2),
                    open=185.0, high=187.0, low=184.0,
                    close=186.0, adj_close=185.5, volume=16_000_000,
                    vwap=185.8, data_quality_score=0.90)
    bar2 = DailyBar(symbol="AAPL", date=date(2024, 1, 2),
                    open=185.0, high=187.0, low=184.0,
                    close=186.5, adj_close=186.0, volume=16_000_000,
                    vwap=185.8, data_quality_score=0.95)
    db.insert_daily_bars([bar1])
    db.insert_daily_bars([bar2])  # should upsert
    rows = db.get_daily_bars("AAPL", date(2024, 1, 1), date(2024, 1, 3))
    assert len(rows) == 1
    assert rows[0].close == 186.5  # updated
    db.close()


def test_get_cached_date_range(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    bars = [
        DailyBar(symbol="AAPL", date=date(2024, 1, d),
                 open=185.0, high=187.0, low=184.0,
                 close=186.0, adj_close=185.5, volume=16_000_000,
                 vwap=185.8, data_quality_score=0.95)
        for d in range(2, 6)  # Jan 2-5
    ]
    db.insert_daily_bars(bars)
    min_date, max_date = db.get_cached_date_range("AAPL")
    assert min_date == date(2024, 1, 2)
    assert max_date == date(2024, 1, 5)
    db.close()


def test_get_cached_date_range_empty(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    result = db.get_cached_date_range("AAPL")
    assert result == (None, None)
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_database.py -v`
Expected: FAIL

- [ ] **Step 3: Implement data models**

```python
# data/storage/models.py
"""Data models for SQLite storage."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class DailyBar:
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    vwap: float
    data_quality_score: float = 1.0


@dataclass
class OptionsChain:
    symbol: str
    date: date
    expiration: date
    strike: float
    option_type: str  # 'call' or 'put'
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class SymbolMetadata:
    symbol: str
    company_name: str
    sector: str
    industry: str
    exchange: str
    market_cap: float
    updated_at: datetime


@dataclass
class RunRecord:
    run_id: str
    mode: str  # 'backtest', 'paper', 'live'
    strategy_name: str
    config: str  # JSON
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    created_at: datetime = None
    full_metrics: str = ""  # JSON


@dataclass
class TradeRecord:
    trade_id: str
    run_id: str
    symbol: str
    direction: str
    entry_date: date
    exit_date: Optional[date]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_reason: str = ""
    exit_reason: str = ""
    option_type: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[date] = None
```

- [ ] **Step 4: Implement database**

```python
# data/storage/database.py
"""SQLite database management with schema creation and query helpers."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from data.storage.models import DailyBar


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                adj_close REAL NOT NULL,
                volume INTEGER NOT NULL,
                vwap REAL NOT NULL,
                data_quality_score REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS options_chains (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                expiration DATE NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                last_price REAL, bid REAL, ask REAL,
                volume INTEGER, open_interest INTEGER,
                implied_volatility REAL,
                delta REAL, gamma REAL, theta REAL, vega REAL,
                PRIMARY KEY (symbol, date, expiration, strike, option_type)
            );

            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                period TEXT NOT NULL,
                revenue REAL, net_income REAL, eps REAL, pe_ratio REAL,
                debt_to_equity REAL, free_cash_flow REAL, roe REAL,
                raw_json TEXT,
                PRIMARY KEY (symbol, date, period)
            );

            CREATE TABLE IF NOT EXISTS indicators_cache (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                indicator_name TEXT NOT NULL,
                value REAL NOT NULL,
                params TEXT NOT NULL,
                PRIMARY KEY (symbol, date, indicator_name, params)
            );

            CREATE TABLE IF NOT EXISTS symbol_metadata (
                symbol TEXT PRIMARY KEY,
                company_name TEXT, sector TEXT, industry TEXT,
                exchange TEXT, market_cap REAL, updated_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                config TEXT,
                start_date DATE, end_date DATE,
                initial_capital REAL,
                final_value REAL, total_return REAL,
                sharpe REAL, max_drawdown REAL,
                created_at TIMESTAMP,
                full_metrics TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                run_id TEXT REFERENCES runs(run_id),
                symbol TEXT NOT NULL, direction TEXT NOT NULL,
                entry_date DATE, exit_date DATE,
                entry_price REAL, exit_price REAL,
                quantity INTEGER,
                pnl REAL, pnl_pct REAL,
                entry_reason TEXT, exit_reason TEXT,
                option_type TEXT, strike REAL, expiration DATE
            );

            CREATE TABLE IF NOT EXISTS data_quality_log (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (symbol, date, issue_type)
            );

            CREATE INDEX IF NOT EXISTS idx_daily_bars_symbol ON daily_bars(symbol);
            CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date);
            CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id);
        """)
        self.conn.commit()

    def insert_daily_bars(self, bars: list[DailyBar]) -> None:
        self.conn.executemany(
            """INSERT OR REPLACE INTO daily_bars
               (symbol, date, open, high, low, close, adj_close, volume, vwap, data_quality_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(b.symbol, b.date.isoformat(), b.open, b.high, b.low,
              b.close, b.adj_close, b.volume, b.vwap, b.data_quality_score)
             for b in bars]
        )
        self.conn.commit()

    def get_daily_bars(self, symbol: str, start: date, end: date) -> list[DailyBar]:
        rows = self.conn.execute(
            """SELECT * FROM daily_bars
               WHERE symbol = ? AND date >= ? AND date <= ?
               ORDER BY date""",
            (symbol, start.isoformat(), end.isoformat())
        ).fetchall()
        return [
            DailyBar(
                symbol=r["symbol"], date=date.fromisoformat(r["date"]),
                open=r["open"], high=r["high"], low=r["low"],
                close=r["close"], adj_close=r["adj_close"],
                volume=r["volume"], vwap=r["vwap"],
                data_quality_score=r["data_quality_score"]
            )
            for r in rows
        ]

    def get_cached_date_range(self, symbol: str) -> tuple[Optional[date], Optional[date]]:
        row = self.conn.execute(
            "SELECT MIN(date) as min_d, MAX(date) as max_d FROM daily_bars WHERE symbol = ?",
            (symbol,)
        ).fetchone()
        if row["min_d"] is None:
            return (None, None)
        return (date.fromisoformat(row["min_d"]), date.fromisoformat(row["max_d"]))

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_database.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add trading-bot/data/storage/database.py trading-bot/data/storage/models.py trading-bot/tests/test_database.py
git commit -m "feat: add SQLite database layer with schema and daily bar CRUD"
```

---

## Phase 2: Parallel Layer — Data, Indicators, Portfolio, Execution, Risk (ALL PARALLEL)

> **These 5 tasks have NO dependencies on each other.** Dispatch one agent per task.

### Task 3: FMP client, rate limiter, and data caching

**Files:**
- Create: `utils/rate_limiter.py`
- Create: `utils/helpers.py`
- Create: `data/feeds/base.py`
- Create: `data/feeds/fmp_feed.py`
- Create: `data/storage/cache.py`
- Create: `tests/test_fmp_feed.py`
- Create: `tests/test_rate_limiter.py`
- Create: `tests/test_cache.py`

**What to build:**
- Token-bucket rate limiter (700 req/min)
- FMP API client: fetch daily bars, symbol profile, stock splits. Parse JSON into DailyBar models.
- Smart cache: check DB for existing data, only fetch missing date ranges from FMP.
- US market holiday calendar for gap detection (use `pandas.tseries.holiday.USFederalHolidayCalendar` or hardcode major market closures).
- Batch quote support for multi-symbol fetches.

**Key FMP endpoints:**
- Daily bars: `GET /api/v3/historical-price-full/{SYMBOL}?from={start}&to={end}&apikey={key}`
- Profile: `GET /api/v3/profile/{SYMBOL}?apikey={key}`
- Splits: `GET /api/v3/historical-stock-split/{SYMBOL}?apikey={key}`

**Tests:** Use fixture data (hardcoded JSON responses) to test parsing without hitting the real API. One optional integration test (marked `@pytest.mark.integration`) that hits FMP with the real key for a single symbol.

- [ ] Steps: TDD — write failing tests for rate limiter, FMP response parsing, and cache logic. Implement each. Run tests. Commit.

---

### Task 4: Data validation pipeline

**Files:**
- Create: `data/validation/validators.py`
- Create: `data/validation/cleaners.py`
- Create: `data/validation/quality_report.py`
- Create: `tests/test_data_validation.py`
- Create: `tests/fixtures/bad_data.py` (fixture with known-bad bars)

**What to build:**
- Individual validators: schema, range, gap, spike, volume anomaly
- Quality score calculator (composite 0.0-1.0)
- Data quality report generator (per-symbol summary)
- Cleaner: forward-fill small gaps (1-2 days), flag larger gaps

**Tests:** Create fixture data with specific known issues (zero volume, price spike, missing days, negative prices) and verify each validator catches the right issue. Test quality score calculation.

- [ ] Steps: TDD for each validator individually. Then integration test running the full pipeline on fixture data. Commit.

---

### Task 5: Technical indicators

**Files:**
- Create: `indicators/base.py`
- Create: `indicators/technical.py`
- Create: `tests/test_indicators.py`

**What to build:**
- Indicator interface: `Indicator` ABC with `update(bar)` and `value` property
- SMA, EMA, RSI, MACD, Bollinger Bands implementations
- Each indicator is stateful — maintains a rolling window of values
- Sub-properties for multi-value indicators (e.g., `BollingerBands.upper`, `.lower`, `.middle`)

**Tests:** Hand-calculate expected values for small datasets (5-20 bars) and assert indicators match. Test warm-up period behavior (indicator returns None before enough data).

- [ ] Steps: TDD — one test + implementation per indicator. Commit after each indicator passes.

---

### Task 6: Portfolio and accounting

**Depends on:** Task 1 (events + event bus)

**Files:**
- Create: `portfolio/portfolio.py`
- Create: `portfolio/accounting.py`
- Create: `portfolio/benchmark.py`
- Create: `tests/test_portfolio.py`

**What to build:**
- Portfolio: tracks cash, holdings (symbol → quantity, avg cost), equity value. Accepts an EventBus in constructor.
- Methods: `update_on_fill(fill)`, `has_position(symbol)`, `get_equity(current_prices)`, `get_positions()`, `get_drawdown()`
- After each `update_on_fill()`, emit `PortfolioUpdateEvent` on the event bus with current equity, cash, unrealized/realized P&L, drawdown %.
- Accounting: transaction log, cost basis tracking (FIFO), realized/unrealized P&L split
- Benchmark: `BenchmarkTracker` — takes benchmark symbol (default SPY), computes buy-and-hold equity curve from daily bars, provides `get_benchmark_returns()` for metrics comparison. The engine passes benchmark bars into this module during backtest.

**Tests:** Simulate a sequence of buys/sells, verify cash, holdings, P&L, drawdown at each step. Verify PortfolioUpdateEvent emitted. Verify benchmark equity curve matches manual calculation.

- [ ] Steps: TDD — test buy, test sell, test P&L calculation, test drawdown tracking, test PortfolioUpdateEvent emission, test benchmark. Commit.

---

### Task 7: Broker interface and SimBroker (equities + options)

**Depends on:** Task 1 (events + event bus)

**Files:**
- Create: `execution/broker.py`
- Create: `execution/sim_broker.py`
- Create: `execution/order_types.py`
- Create: `execution/options_pricing.py`
- Create: `tests/test_sim_broker.py`
- Create: `tests/test_options_pricing.py`

**What to build:**
- Broker ABC: `submit_order()`, `get_pending_orders()`, `cancel_order()`
- Order action constants: `BUY`, `SELL`, `BTO`, `STO`, `BTC`, `STC` — separate from order type
- Order type constants: `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT`
- SimBroker for equities: fill at next bar's open (default) or current close. Slippage (fixed 0.01% default or volume-based). Commission ($0.005/share default).
- SimBroker for options:
  - `options_pricing.py`: Black-Scholes pricing function using scipy.stats.norm. Inputs: spot, strike, time_to_expiry, risk_free_rate, volatility. Outputs: call/put price, delta, gamma, theta, vega.
  - Mark-to-market daily: use stored `options_chains` data if available, otherwise fall back to Black-Scholes with historical IV. Log which method was used.
  - Bid-ask spread cost: configurable, default 50% of mid-to-edge distance for options fills.
  - Assignment at expiration: ITM options auto-assigned. Early exercise for deep ITM calls near ex-dividend.
  - Track portfolio-level Greeks (sum across all options positions).
- Emits `FillEvent` via event bus

**Tests:**
- Equity: market buy fill at next bar open + slippage, commission deduction, limit order unfilled, cancel
- Options: Black-Scholes price vs known values (use online BS calculator to generate expected), STO fill with spread cost, ITM assignment at expiration, mark-to-market repricing

- [ ] Steps: TDD — test equity fills first (market, limit, stop, cancel, slippage, commission). Then Black-Scholes pricing. Then options fills with spread. Then assignment logic. Commit after each.

---

## Phase 3: Strategy and Risk (depends on Phase 2)

> **These 3 tasks can run in parallel.**

### Task 8: Strategy base class and registry

**Files:**
- Create: `strategy/base.py`
- Create: `strategy/registry.py`
- Create: `tests/test_strategy_base.py`

**What to build:**
- Strategy ABC per spec: `generate_signals()`, `on_universe_bar()`, `on_fill()`, `on_start()`, `on_end()`, `warm_up_period()`
- Default `on_universe_bar()` delegates to `generate_signals()` per symbol
- Strategy registry: resolve strategy by YAML path or Python module path
- `BacktestContext` dataclass (start_date, end_date, universe, settings)

**Tests:** Create a simple test strategy, register it, resolve it, verify interface contract.

- [ ] Steps: TDD — test ABC enforcement, test default delegation, test registry. Commit.

---

### Task 9: Config-driven strategy + expression parser

**Depends on:** Task 5 (indicators must be implemented for YAML strategies to use them), Task 8 (Strategy ABC)

**Files:**
- Create: `strategy/expression_parser.py`
- Create: `strategy/config_strategy.py`
- Create: `config/strategies/mean_reversion.yaml`
- Create: `tests/test_expression_parser.py`
- Create: `tests/test_config_strategy.py`

**What to build:**

**Expression Parser** (`strategy/expression_parser.py`) — this is the hardest component. Build a safe recursive-descent parser. NO `eval()`.

Grammar (operator precedence, lowest to highest):
```
expression  → or_expr
or_expr     → and_expr ( "AND" and_expr )*
and_expr    → not_expr ( "AND" not_expr )*
not_expr    → "NOT" not_expr | comparison
comparison  → arithmetic ( ( "<" | ">" | "<=" | ">=" | "==" | "!=" ) arithmetic )?
arithmetic  → term ( ( "+" | "-" ) term )*
term        → factor ( ( "*" | "/" ) factor )*
factor      → NUMBER | VARIABLE | "(" expression ")"
VARIABLE    → IDENTIFIER ( "." IDENTIFIER )*   -- e.g., bb_20.lower
NUMBER      → FLOAT_LITERAL                    -- e.g., 0.95, 30
```

AST Node types:
```python
@dataclass
class NumberNode:
    value: float

@dataclass
class VariableNode:
    parts: list[str]  # ["bb_20", "lower"] for bb_20.lower

@dataclass
class BinaryOpNode:
    op: str  # "+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!="
    left: ASTNode
    right: ASTNode

@dataclass
class BooleanOpNode:
    op: str  # "AND", "OR"
    left: ASTNode
    right: ASTNode

@dataclass
class NotNode:
    operand: ASTNode
```

Evaluator: takes AST + context dict (bar fields, indicator values, portfolio fields), walks tree, returns value. Variables resolved by looking up `context[parts[0]]` or `context[parts[0]].parts[1]` for dot notation.

**Tests for expression parser:**
```python
def test_simple_comparison():
    assert evaluate("close < 100", {"close": 95.0}) == True

def test_boolean_and():
    assert evaluate("close < 100 AND rsi_14 < 30", {"close": 95.0, "rsi_14": 25.0}) == True

def test_dot_notation():
    ctx = {"bb_20": type("", (), {"lower": 90.0})()}
    assert evaluate("close < bb_20.lower", {**ctx, "close": 85.0}) == True

def test_arithmetic():
    assert evaluate("close * 0.95 < sma_20", {"close": 100.0, "sma_20": 96.0}) == True

def test_complex_expression():
    assert evaluate("close < bb_20.lower AND rsi_14 < 30", {...}) == True

def test_undefined_variable_returns_false():
    assert evaluate("undefined_var < 30", {}) == False  # with warning logged

def test_parse_error_raises():
    with pytest.raises(ExpressionParseError):
        parse("close < < 30")  # invalid syntax
```

**ConfigStrategy:** Parse YAML, build indicator instances, wrap in Strategy interface. `generate_signals()` evaluates entry/exit rule conditions against current bar + indicator values + portfolio state.

**YAML config:** Use the mean_reversion.yaml from the spec (Section 5.2).

- [ ] Steps: TDD — expression parser tokenizer first, then recursive-descent parser, then evaluator, then YAML loading, then ConfigStrategy signal generation. Commit after each major piece.

---

### Task 10: Risk manager

**Depends on:** Task 1 (events), Task 6 (portfolio)

**Files:**
- Create: `risk/manager.py`
- Create: `risk/position.py`
- Create: `risk/rules.py`
- Create: `tests/test_risk_manager.py`

**What to build:**
- RiskRule interface: `check(signal, portfolio, context) -> (bool, str)` — returns (passes, reason). Context includes sector map.
- Individual rules: max position size (6%), max sector concentration (25%), max open positions (20), drawdown circuit breaker (-15%), max options notional (30%)
- Sector lookup: RiskManager accepts a `sector_map: dict[str, str]` (symbol → sector) at construction. Populated from `symbol_metadata` table at engine startup. For testing, pass a hardcoded dict.
- RiskManager: takes list of rules + sector_map, evaluates signal against all, emits RiskEvent (with `signal_blocked_id`) for blocked signals, converts passing signals to OrderEvents
- Position tracker: aggregates positions for sector/exposure/options notional checks

**Tests:** Test each rule individually with edge cases. Test sector concentration with a 3-symbol portfolio across 2 sectors. Test manager with multiple rules. Test that blocked signals produce RiskEvents on the event bus.

- [ ] Steps: TDD — one test per rule, then manager integration, then RiskEvent emission. Commit.

---

## Phase 4a: Analytics Core (depends on Phase 2-3)

> **These 4 tasks can run in parallel** — they produce independent modules with no cross-dependencies.

### Task 11: Core metrics engine

**Files:**
- Create: `analytics/metrics.py`
- Create: `tests/test_metrics.py`

**What to build:**
- Functions that take an equity curve (list of daily values) and trade list, return all metrics:
  - `sharpe_ratio(returns, risk_free_rate)` → float
  - `sortino_ratio(returns, risk_free_rate)` → float
  - `max_drawdown(equity_curve)` → (pct, peak_date, trough_date, recovery_date)
  - `calmar_ratio(cagr, max_drawdown)` → float
  - `profit_factor(trades)` → float
  - `win_rate(trades)` → float
  - `expectancy(trades)` → float
  - `cagr(start_value, end_value, years)` → float
  - `beta(strategy_returns, benchmark_returns)` → float
  - `value_at_risk(returns, confidence)` → float
- `compute_all_metrics(equity_curve, trades, benchmark_curve, risk_free_rate)` → dict

**Tests:** Hand-calculate expected metric values for a small known equity curve and trade list.

- [ ] Steps: TDD — one test per metric function. Commit.

---

### Task 12: Trade log and regime analysis

**Files:**
- Create: `analytics/trade_log.py`
- Create: `analytics/regime.py`
- Create: `tests/test_trade_log.py`
- Create: `tests/test_regime.py`

**What to build:**
- TradeLog: accumulates trades during backtest, provides queries (by symbol, by sector, by date range)
- Trade statistics: avg holding period, consecutive streaks, per-symbol breakdown
- Regime detector: classify each date as bull/bear/sideways/high-vol using SMA-50/SMA-200 crossover + VIX/realized vol
- Per-regime performance: filter trades by regime, compute metrics per regime

**Tests:** Test trade log queries. Test regime detection on a known price series where regimes are obvious.

- [ ] Steps: TDD — trade log first, then regime detection, then per-regime metrics. Commit.

---

### Task 13: Monte Carlo simulation

**Files:**
- Create: `analytics/monte_carlo.py`
- Create: `tests/test_monte_carlo.py`

**What to build:**
- Take list of trade P&Ls, initial capital
- Shuffle trade order N times (default 10,000), replay equity curve each time
- Output: median/5th/95th percentile final equity, probability of ruin, max drawdown distribution
- Detect if actual result is an outlier (>95th or <5th percentile)

**Tests:** Use a deterministic seed. Test with known trades that should produce a narrow confidence band (consistent small wins) vs wide band (few large wins).

- [ ] Steps: TDD — test shuffle replay, test statistics, test outlier detection. Commit.

---

### Task 14: Options analytics

**Files:**
- Create: `analytics/options_analytics.py`
- Create: `tests/test_options_analytics.py`

**What to build:**
- Total premium collected vs paid (from options trades in trade log)
- Assignment rate: % of short options that were assigned vs expired worthless
- Greeks exposure over time: given a list of `PortfolioUpdateEvent`s and options positions, compute portfolio-level delta, gamma, theta, vega per date
- Win rate by DTE bucket: group options trades by days-to-expiration at entry (0-7, 7-30, 30-60, 60+), compute win rate per bucket
- Theta decay P&L vs directional P&L: decompose each options trade's P&L into component from time decay vs underlying price movement

**Tests:** Create fixture options trades with known outcomes. Verify premium calculation, assignment rate, DTE bucket grouping.

- [ ] Steps: TDD — premium calc, assignment rate, DTE bucketing, Greeks aggregation. Commit.

---

## Phase 4b: Report Generation (depends on Phase 4a — Tasks 11-14)

### Task 15: Report generation (console + HTML)

**Depends on:** Tasks 11, 12, 13, 14 (metrics, trade log, Monte Carlo, options analytics)

**Files:**
- Create: `analytics/reports.py`
- Create: `tests/test_reports.py`

**What to build:**
- Console report: formatted text table with key metrics, benchmark comparison (strategy vs SPY buy-and-hold)
- HTML report: Plotly charts — equity curve with drawdown overlay, trade markers on price chart, monthly returns heatmap, rolling Sharpe, regime performance table, Monte Carlo confidence bands, complete trade log, options Greeks timeline
- Report saved to `reports/{run_id}.html`
- Benchmark comparison: every metric shown side-by-side for strategy vs benchmark. Header answers "Did this strategy beat just holding the index?"
- Imports and uses: `metrics.compute_all_metrics()`, `TradeLog`, `MonteCarloResult`, `OptionsAnalytics`

**Tests:** Test console report string contains expected sections (return, Sharpe, benchmark comparison). Test HTML report file is created, is valid HTML, and contains expected Plotly div IDs.

- [ ] Steps: TDD — console report first (simpler), then HTML equity curve chart, then add each Plotly chart one at a time. Commit after each chart works.

---

## Phase 5: Engine and CLI (depends on all above)

### Task 16: Backtest engine

**Files:**
- Create: `core/engine.py`
- Create: `core/clock.py`
- Create: `tests/test_engine.py`
- Create: `tests/fixtures/sample_bars.py`

**What to build:**
- Clock: iterates over trading dates (historical replay)
- Engine: wires together DataFeed → EventBus → Strategy → RiskManager → Broker → Portfolio → Analytics
- Backtest loop: for each date, collect bars, call strategy, process signals through risk → broker → portfolio
- After loop: call `strategy.on_end()`, compute metrics, generate reports, save run to DB
- Create `tests/fixtures/sample_bars.py` with 100 deterministic bars for SPY

**Tests:** Full integration test — run a simple moving average crossover strategy on fixture data, verify final equity, trade count, and metrics match hand-calculated expectations.

- [ ] Steps: TDD — test clock iteration, test engine wiring, test full backtest. Commit.

---

### Task 17: CLI with Typer

**Files:**
- Create: `main.py`
- Create: `utils/logging_config.py`
- Create: `tests/test_cli.py`

**What to build:**
- `utils/logging_config.py`: structured logging setup — log all events to file + console at configurable level
- Typer app with subcommands:
  - `data fetch SYMBOL --start --end` — fetch and cache from FMP
  - `data fetch-universe {sp500|custom_file} --start` — bulk-fetch an entire universe
  - `data validate SYMBOL` / `data validate --all --report` — run validation
  - `data status` — show cached symbols and date ranges
  - `backtest --strategy --start --end --capital --universe` — run backtest
  - `backtest compare --strategies s1,s2 --start --end` — run multiple strategies, compare results
  - `backtest optimize --strategy --param key:v1,v2,v3 --objective sharpe` — grid search (wires to Task 20)
  - `report last` — show last backtest console report
  - `report --run-id --format {console|html}` — generate specific report
  - `report list` — list all historical runs
  - `report monte-carlo --run-id --simulations N` — run Monte Carlo on a past run
  - `paper start --strategy --capital` — stub (raises "not yet implemented" message)
  - `paper status` / `paper stop` — stubs
- Load settings from config, resolve strategy, create engine, run

**Tests:** Test CLI commands execute without error using fixture data (no FMP calls). Test help text renders for each subcommand.

- [ ] Steps: Implement logging config first. Then data subcommands, then backtest, then report, then paper stubs. Commit after each group.

---

## Phase 6: Built-in Strategies + Integration (depends on Phase 5)

> **These 2 tasks can run in parallel.**

### Task 18: Mean reversion strategy (YAML) — v1 DoD item #2

**Files:**
- Create: `config/strategies/mean_reversion.yaml` (from spec Section 5.2)
- Create: `tests/test_mean_reversion.py`

**What to build:**
- The YAML config from the spec (Bollinger Bands + RSI entry, SMA-20 reversion exit, -5% stop)
- End-to-end test: run backtest on SPY fixture data, verify it completes with signals, trades, and metrics

**Tests:** Run full backtest on fixture data, verify signals fire when RSI < 30 AND close < BB lower, trades execute with correct position sizing (6%), console report outputs.

- [ ] Steps: Create YAML config, write integration test, run, verify. Commit.

---

### Task 19: Options wheel strategy (Python class) — v1 DoD item #3

**Files:**
- Create: `strategy/library/options_wheel.py`
- Create: `tests/test_options_wheel.py`

**What to build:**
- Options wheel strategy per spec Section 5.3: sell cash-secured puts when RSI < 40, take assignment, sell covered calls at 5% OTM
- Uses the SimBroker options simulation model (Black-Scholes pricing, assignment, spread costs)
- Verify it runs end-to-end through the engine with options-specific analytics

**Tests:** Run backtest on fixture data with simulated options chains. Verify: CSP signals fire, assignment occurs for ITM puts at expiration, covered calls are sold after assignment, options analytics (premium collected, assignment rate) are computed.

- [ ] Steps: Write strategy class, create fixture with options chain data, write integration test, run, verify. Commit.

---

## Phase 7: Optimization + Final Integration (sequential)

### Task 20: Parameter optimizer and walk-forward validation

**Files:**
- Create: `core/optimizer.py`
- Create: `tests/test_optimizer.py`

**What to build:**
- Grid search: take strategy + param grid, run backtest for each combo, rank by objective metric
- Walk-forward: rolling train/test windows, optimize on train, validate on test
- Edge parameter detection warning
- Overfitting detection: compare in-sample vs out-of-sample Sharpe degradation

**Tests:**
- Grid search: 2x2 param grid on fixture data, verify it produces 4 results ranked by Sharpe. Test edge-parameter warning fires when best param is at grid boundary.
- Walk-forward overfitting detection: create a contrived dataset where RSI period=5 produces a perfect pattern in the first half (train) that completely inverts in the second half (test). Walk-forward should detect the Sharpe degradation (in-sample Sharpe >> out-of-sample Sharpe) and flag it.

- [ ] Steps: TDD — grid search first, then walk-forward with overfitting dataset, then wire optimizer into CLI backtest optimize command. Commit.

---

### Task 21: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

**What to build:**
- Full pipeline test: create DB → fetch data (fixture) → run mean reversion backtest → verify console report → verify HTML report → verify trades in DB → verify Monte Carlo runs → verify regime analysis
- This is the "v1 Definition of Done" acceptance test

**Tests:** Single test that exercises the entire framework from data ingest to report output.

- [ ] Steps: Write the integration test. Run it. Fix any issues. Commit.

---

## Dependency Graph Summary

```
Phase 0: Scaffolding (Task 0)
    ↓
Phase 1: Events + Database (Tasks 1-2, sequential)
    ↓
Phase 2: [FMP Client(3)] [Validators(4)] [Indicators(5)] [Portfolio(6)] [SimBroker+Options(7)]  ← 5 PARALLEL
    ↓
Phase 3: [Strategy Base(8)] [Config Strategy+Parser(9)] [Risk Manager(10)]  ← 3 PARALLEL
    ↓                                                                          (9 depends on 5+8)
Phase 4a: [Metrics(11)] [TradeLog+Regime(12)] [Monte Carlo(13)] [Options Analytics(14)]  ← 4 PARALLEL
    ↓
Phase 4b: Reports(15)  ← depends on 11-14
    ↓
Phase 5: Engine(16) + CLI(17) (sequential)
    ↓
Phase 6: [Mean Reversion YAML(18)] [Options Wheel Python(19)]  ← 2 PARALLEL
    ↓
Phase 7: Optimizer(20) + Integration Test(21) (sequential)
```

**Total: 22 tasks across 9 phases. Maximum parallelism: 5 agents in Phase 2.**
