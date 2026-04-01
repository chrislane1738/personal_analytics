# Topstep Evaluation Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Topstep prop firm evaluation simulator that runs 1,000+ independent attempts using ORB and VWAP reversion strategies with an adaptive state machine, measures pass rate and EV, and displays results in a new `/eval` dashboard page.

**Architecture:** `TopstepEvalSimulator` wraps the existing `BacktestEngine`. New modules: `EvaluationRules`, `StateManager`, `AttemptTracker`, `CampaignRunner`. Two signal generators: `ORBStrategy` and `VWAPReversionStrategy`. All new code in a `topstep/` package. Dashboard gets a new `/eval` page with campaign browser and detail views.

**Tech Stack:** Python 3.11+, existing trading-bot framework (BacktestEngine, SimBroker, Portfolio, EventBus), FastAPI, Next.js, shadcn/ui, Recharts, TanStack Query

**Spec:** `docs/superpowers/specs/2026-04-01-topstep-eval-bot-design.md`

**Parallelization Guide:** Tasks are grouped into phases. Tasks within a phase have NO dependencies on each other and SHOULD be executed in parallel by separate agents. Tasks in later phases depend on earlier phases completing.

---

## File Structure

### Phase 0: Prerequisites (modify existing + scaffold)

```
indicators/technical.py              — MODIFY: add ATR indicator class
tests/test_indicators.py             — MODIFY: add ATR tests
data/storage/database.py             — MODIFY: add eval_campaigns table DDL + CRUD methods
data/storage/models.py               — MODIFY: add EvalCampaignRecord dataclass
tests/test_persistence.py            — MODIFY: add eval_campaigns tests
topstep/__init__.py                  — CREATE: package init
topstep/config.py                    — CREATE: TopstepConfig dataclass, preset tiers
tests/test_topstep_config.py         — CREATE: config tests
```

### Phase 1–5: New Files

```
topstep/
├── __init__.py
├── config.py                         — TopstepConfig, account tier presets
├── evaluation_rules.py               — EvaluationRules class (pass/fail/consistency)
├── state_manager.py                  — EvalState enum, StateManager class
├── attempt_tracker.py                — AttemptTracker (daily P&L, high water, status)
├── simulator.py                      — TopstepEvalSimulator (wraps BacktestEngine)
├── campaign_runner.py                — CampaignRunner + CampaignResult
└── strategies/
    ├── __init__.py
    ├── orb_strategy.py               — ORBStrategy (opening range breakout)
    └── vwap_reversion_strategy.py    — VWAPReversionStrategy (mean reversion)

backend/
├── routers/
│   └── eval.py                       — NEW: campaign CRUD + launch endpoints
└── schemas.py                        — MODIFY: add campaign response schemas

frontend/app/
├── eval/
│   └── page.tsx                      — NEW: Campaign Browser
└── eval/[campaignId]/
    └── page.tsx                      — NEW: Campaign Detail

tests/
├── test_topstep_config.py            — NEW
├── test_evaluation_rules.py          — NEW
├── test_state_manager.py             — NEW
├── test_attempt_tracker.py           — NEW
├── test_orb_strategy.py              — NEW
├── test_vwap_strategy.py             — NEW
├── test_simulator.py                 — NEW
└── test_campaign_runner.py           — NEW
```

---

## Phase 0: Prerequisites

These tasks prepare the codebase for the evaluation bot. Must complete before Phase 1.

---

### Task 0.1: Add ATR indicator

The ORB and VWAP strategies both need ATR (Average True Range) for volatility-based position sizing and stop distances. ATR does not exist in the codebase — only SMA, EMA, RSI, MACD, BollingerBands.

**Files:**
- Modify: `indicators/technical.py` (append after BollingerBands class)
- Modify: `tests/test_indicators.py` (append ATR tests)

- [ ] **Step 1: Write failing tests for ATR**

Append to `tests/test_indicators.py`:

```python
# --- ATR Tests ---

from indicators.technical import ATR


def test_atr_warm_up_period():
    atr = ATR(period=3)
    assert atr.warm_up_period == 3


def test_atr_returns_none_before_warm_up():
    atr = ATR(period=3)
    atr.update(high=12.0, low=10.0, close=11.0)
    assert atr.value is None
    atr.update(high=13.0, low=10.5, close=12.0)
    assert atr.value is None


def test_atr_computes_after_warm_up():
    """ATR(3) with known values.
    Bar 1: H=12 L=10 C=11 → TR=2.0
    Bar 2: H=13 L=10.5 C=12 → TR=max(2.5, |13-11|, |10.5-11|)=2.5
    Bar 3: H=14 L=11 C=13 → TR=max(3.0, |14-12|, |11-12|)=3.0
    ATR(3) = (2.0+2.5+3.0)/3 = 2.5
    """
    atr = ATR(period=3)
    atr.update(high=12.0, low=10.0, close=11.0)
    atr.update(high=13.0, low=10.5, close=12.0)
    atr.update(high=14.0, low=11.0, close=13.0)
    assert atr.value == pytest.approx(2.5, abs=0.01)


def test_atr_subsequent_values_use_smoothing():
    """After initial ATR, subsequent values use Wilder smoothing:
    ATR_new = (ATR_prev * (period-1) + TR_new) / period
    """
    atr = ATR(period=3)
    atr.update(high=12.0, low=10.0, close=11.0)  # TR=2.0
    atr.update(high=13.0, low=10.5, close=12.0)  # TR=2.5
    atr.update(high=14.0, low=11.0, close=13.0)  # TR=3.0, ATR=2.5
    atr.update(high=15.0, low=13.0, close=14.0)  # TR=2.0, ATR=(2.5*2+2.0)/3=2.333
    assert atr.value == pytest.approx(2.333, abs=0.01)


def test_atr_invalid_period_raises():
    with pytest.raises(ValueError):
        ATR(period=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_indicators.py -k "test_atr" -v`
Expected: FAIL — `ImportError: cannot import name 'ATR'`

- [ ] **Step 3: Implement ATR indicator**

In `indicators/technical.py`, add after the `BollingerBands` class (at the end of the file):

```python
class ATR(Indicator):
    """Average True Range — measures volatility using Wilder smoothing.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    Initial ATR = simple average of first `period` TRs.
    Subsequent: ATR = (prev_ATR * (period-1) + TR) / period (Wilder smoothing).
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError(f"ATR period must be >= 1, got {period}")
        self._period = period
        self._prev_close: float | None = None
        self._tr_window: list[float] = []
        self._atr: float | None = None

    def update(self, high: float = 0.0, low: float = 0.0, close: float = 0.0, **kwargs) -> None:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close

        if self._atr is None:
            self._tr_window.append(tr)
            if len(self._tr_window) == self._period:
                self._atr = sum(self._tr_window) / self._period
                self._tr_window = []
        else:
            self._atr = (self._atr * (self._period - 1) + tr) / self._period

    @property
    def value(self) -> float | None:
        return self._atr

    @property
    def warm_up_period(self) -> int:
        return self._period
```

Note: The existing `Indicator` base class uses `update(price: float)` but our ATR needs `high`, `low`, `close`. Check the `Indicator` ABC in `indicators/base.py`. If `update` only takes `price`, override the signature — the ATR needs OHLC. The engine calls indicator updates per bar, so it already has access to these fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_indicators.py -k "test_atr" -v`
Expected: ALL PASS

- [ ] **Step 5: Run full indicator test suite for regressions**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_indicators.py -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add indicators/technical.py tests/test_indicators.py
git commit -m "feat: add ATR indicator with Wilder smoothing"
```

---

### Task 0.2: Add eval_campaigns table and EvalCampaignRecord model

**Files:**
- Modify: `data/storage/models.py` (append EvalCampaignRecord after WalkForwardStudy)
- Modify: `data/storage/database.py` (add table DDL at line ~195, add CRUD methods)
- Modify: `tests/test_persistence.py` (append tests)

- [ ] **Step 1: Write failing tests for eval campaign storage**

Append to `tests/test_persistence.py`:

```python
from data.storage.models import EvalCampaignRecord


def test_insert_and_get_eval_campaign(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    campaign = EvalCampaignRecord(
        campaign_id="camp1",
        strategy_name="ORB",
        instrument="MES",
        state_machine=True,
        topstep_config='{"account_size": 50000}',
        num_attempts=1000,
        seed=42,
        pass_rate=0.342,
        ev_per_attempt=68.30,
        cost_to_funded=143.0,
        avg_days_to_pass=3.8,
        annual_ev=3264.0,
        created_at=datetime.now(tz=timezone.utc),
        full_results='{"attempt_outcomes": []}',
    )
    db.insert_eval_campaign(campaign)
    result = db.get_eval_campaign("camp1")

    assert result is not None
    assert result.campaign_id == "camp1"
    assert result.pass_rate == 0.342
    assert result.strategy_name == "ORB"
    db.close()


def test_list_eval_campaigns(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    for i in range(3):
        db.insert_eval_campaign(EvalCampaignRecord(
            campaign_id=f"camp{i}",
            strategy_name="ORB" if i < 2 else "VWAP",
            instrument="MES",
            state_machine=True,
            topstep_config="{}",
            num_attempts=1000,
            seed=i,
            pass_rate=0.3 + i * 0.05,
            ev_per_attempt=50.0 + i * 10,
            cost_to_funded=150.0,
            avg_days_to_pass=4.0,
            annual_ev=3000.0,
            created_at=datetime.now(tz=timezone.utc),
            full_results="{}",
        ))

    all_campaigns = db.list_eval_campaigns()
    assert len(all_campaigns) == 3

    page = db.list_eval_campaigns(limit=2, offset=0)
    assert len(page) == 2

    sorted_camps = db.list_eval_campaigns(sort="pass_rate", order="desc")
    assert sorted_camps[0].pass_rate >= sorted_camps[-1].pass_rate
    db.close()


def test_delete_eval_campaign(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    db.insert_eval_campaign(EvalCampaignRecord(
        campaign_id="del1", strategy_name="ORB", instrument="MES",
        state_machine=True, topstep_config="{}", num_attempts=100,
        seed=1, pass_rate=0.25, ev_per_attempt=30.0,
        cost_to_funded=196.0, avg_days_to_pass=5.0, annual_ev=1440.0,
        created_at=datetime.now(tz=timezone.utc), full_results="{}",
    ))
    db.delete_eval_campaign("del1")
    assert db.get_eval_campaign("del1") is None
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py -k "test_insert_and_get_eval_campaign or test_list_eval_campaigns or test_delete_eval_campaign" -v`
Expected: FAIL — `EvalCampaignRecord` not defined

- [ ] **Step 3: Add EvalCampaignRecord dataclass**

In `data/storage/models.py`, append at the end of the file (after `WalkForwardStudy`):

```python
@dataclass
class EvalCampaignRecord:
    campaign_id: str
    strategy_name: str
    instrument: str
    state_machine: bool
    topstep_config: str       # JSON
    num_attempts: int
    seed: int
    pass_rate: float = 0.0
    ev_per_attempt: float = 0.0
    cost_to_funded: float = 0.0
    avg_days_to_pass: float = 0.0
    annual_ev: float = 0.0
    created_at: Optional[datetime] = None
    full_results: str = ""    # JSON blob of CampaignResult
```

- [ ] **Step 4: Add eval_campaigns table to DDL**

In `data/storage/database.py`, inside the `create_tables` DDL string (after the `walk_forward_studies` block, before the closing `"""`), add:

```sql
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
```

- [ ] **Step 5: Add CRUD methods for eval_campaigns**

In `data/storage/database.py`, add after the walk_forward methods section:

```python
# ------------------------------------------------------------------
# eval_campaigns
# ------------------------------------------------------------------

def insert_eval_campaign(self, record: "EvalCampaignRecord") -> None:
    from data.storage.models import EvalCampaignRecord
    sql = """
    INSERT OR REPLACE INTO eval_campaigns
        (campaign_id, strategy_name, instrument, state_machine,
         topstep_config, num_attempts, seed, pass_rate, ev_per_attempt,
         cost_to_funded, avg_days_to_pass, annual_ev, created_at, full_results)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    self._conn.execute(sql, (
        record.campaign_id, record.strategy_name, record.instrument,
        record.state_machine, record.topstep_config, record.num_attempts,
        record.seed, record.pass_rate, record.ev_per_attempt,
        record.cost_to_funded, record.avg_days_to_pass, record.annual_ev,
        record.created_at.isoformat() if record.created_at else None,
        record.full_results,
    ))
    self._conn.commit()

def get_eval_campaign(self, campaign_id: str):
    from data.storage.models import EvalCampaignRecord
    sql = """
    SELECT campaign_id, strategy_name, instrument, state_machine,
           topstep_config, num_attempts, seed, pass_rate, ev_per_attempt,
           cost_to_funded, avg_days_to_pass, annual_ev, created_at, full_results
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
        pass_rate=row["pass_rate"],
        ev_per_attempt=row["ev_per_attempt"],
        cost_to_funded=row["cost_to_funded"],
        avg_days_to_pass=row["avg_days_to_pass"],
        annual_ev=row["annual_ev"],
        created_at=self._to_datetime(row["created_at"]),
        full_results=row["full_results"] or "",
    )

def list_eval_campaigns(
    self,
    sort: str = "created_at",
    order: str = "desc",
    limit: int | None = None,
    offset: int = 0,
) -> list:
    from data.storage.models import EvalCampaignRecord
    allowed_sort = {"created_at", "pass_rate", "ev_per_attempt", "strategy_name", "annual_ev"}
    if sort not in allowed_sort:
        sort = "created_at"
    order = "DESC" if order.lower() == "desc" else "ASC"

    sql = f"""
    SELECT campaign_id, strategy_name, instrument, state_machine,
           topstep_config, num_attempts, seed, pass_rate, ev_per_attempt,
           cost_to_funded, avg_days_to_pass, annual_ev, created_at, full_results
    FROM   eval_campaigns
    ORDER  BY {sort} {order}
    """
    params: list = []
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

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
            pass_rate=row["pass_rate"],
            ev_per_attempt=row["ev_per_attempt"],
            cost_to_funded=row["cost_to_funded"],
            avg_days_to_pass=row["avg_days_to_pass"],
            annual_ev=row["annual_ev"],
            created_at=self._to_datetime(row["created_at"]),
            full_results=row["full_results"] or "",
        )
        for row in rows
    ]

def delete_eval_campaign(self, campaign_id: str) -> None:
    self._conn.execute("DELETE FROM eval_campaigns WHERE campaign_id = ?", (campaign_id,))
    self._conn.commit()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_persistence.py -k "eval_campaign" -v`
Expected: ALL PASS

- [ ] **Step 7: Run full test suite for regressions**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add data/storage/models.py data/storage/database.py tests/test_persistence.py
git commit -m "feat: add eval_campaigns table and EvalCampaignRecord model"
```

---

### Task 0.3: Create TopstepConfig and scaffold topstep package

**Files:**
- Create: `topstep/__init__.py`
- Create: `topstep/strategies/__init__.py`
- Create: `topstep/config.py`
- Create: `tests/test_topstep_config.py`

- [ ] **Step 1: Write failing tests for TopstepConfig**

```python
# tests/test_topstep_config.py
from topstep.config import TopstepConfig, TIER_50K, TIER_100K, TIER_150K


def test_default_config_is_50k():
    config = TopstepConfig()
    assert config.account_size == 50_000.0
    assert config.profit_target == 3_000.0
    assert config.max_loss == 2_000.0
    assert config.consistency_pct == 0.50
    assert config.subscription_fee == 49.0
    assert config.activation_fee == 149.0
    assert config.max_payout == 5_000.0
    assert config.payout_split == 0.90
    assert config.max_position_micros == 50
    assert config.max_attempt_days == 60


def test_tier_presets():
    assert TIER_50K.profit_target == 3_000.0
    assert TIER_50K.max_loss == 2_000.0
    assert TIER_50K.max_position_micros == 50

    assert TIER_100K.profit_target == 6_000.0
    assert TIER_100K.max_loss == 3_000.0
    assert TIER_100K.max_position_micros == 100

    assert TIER_150K.profit_target == 9_000.0
    assert TIER_150K.max_loss == 4_500.0
    assert TIER_150K.max_position_micros == 150


def test_config_custom_values():
    config = TopstepConfig(profit_target=5_000.0, max_loss=3_000.0)
    assert config.profit_target == 5_000.0
    assert config.max_loss == 3_000.0
    assert config.account_size == 50_000.0  # other defaults unchanged


def test_config_ev_break_even_pass_rate():
    """At break-even, EV per attempt = 0.
    EV = pass_rate * (payout - activation_fee) - subscription_fee
    0 = pass_rate * (5000*0.9 - 149) - 49
    pass_rate = 49 / (4500 - 149) = 49 / 4351 ≈ 1.13%
    """
    config = TopstepConfig()
    break_even = config.subscription_fee / (config.max_payout * config.payout_split - config.activation_fee)
    assert break_even < 0.02  # less than 2% needed to break even
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_topstep_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topstep'`

- [ ] **Step 3: Create package scaffolding**

```bash
cd /Users/chrislane/Desktop/Claude_Code/trading-bot
mkdir -p topstep/strategies
touch topstep/__init__.py topstep/strategies/__init__.py
```

- [ ] **Step 4: Implement TopstepConfig**

```python
# topstep/config.py
"""Topstep evaluation configuration and account tier presets."""

from dataclasses import dataclass


@dataclass
class TopstepConfig:
    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    max_loss: float = 2_000.0
    consistency_pct: float = 0.50
    subscription_fee: float = 49.0
    activation_fee: float = 149.0
    max_payout: float = 5_000.0
    payout_split: float = 0.90
    max_position_minis: int = 5
    max_position_micros: int = 50
    max_attempt_days: int = 60


TIER_50K = TopstepConfig(
    account_size=50_000.0,
    profit_target=3_000.0,
    max_loss=2_000.0,
    max_position_minis=5,
    max_position_micros=50,
    subscription_fee=49.0,
)

TIER_100K = TopstepConfig(
    account_size=100_000.0,
    profit_target=6_000.0,
    max_loss=3_000.0,
    max_position_minis=10,
    max_position_micros=100,
    subscription_fee=99.0,
)

TIER_150K = TopstepConfig(
    account_size=150_000.0,
    profit_target=9_000.0,
    max_loss=4_500.0,
    max_position_minis=15,
    max_position_micros=150,
    subscription_fee=149.0,
)

TIERS = {"50k": TIER_50K, "100k": TIER_100K, "150k": TIER_150K}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_topstep_config.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add topstep/ tests/test_topstep_config.py
git commit -m "feat: scaffold topstep package with TopstepConfig and tier presets"
```

---

## Phase 1: Core Evaluation Logic (3 parallel agents)

> **These 3 tasks have NO dependencies on each other.** Dispatch one agent per task. All depend on Phase 0 (TopstepConfig).

---

### Task 1.1: EvaluationRules

**Files:**
- Create: `topstep/evaluation_rules.py`
- Create: `tests/test_evaluation_rules.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluation_rules.py
from topstep.config import TopstepConfig
from topstep.evaluation_rules import EvaluationRules, EvalOutcome


def test_pass_when_target_reached_and_consistent():
    rules = EvaluationRules(TopstepConfig())
    # Day 1: +$1,600
    result = rules.evaluate_day(day_pnl=1600.0, eod_balance=51600.0)
    assert result == EvalOutcome.ACTIVE
    # Day 2: +$1,500 → total $3,100 > $3,000 target
    # Best day $1,600 / $3,100 = 51.6% > 50% → consistency NOT satisfied
    result = rules.evaluate_day(day_pnl=1500.0, eod_balance=53100.0)
    assert result == EvalOutcome.ACTIVE  # not pass yet, consistency violated
    # Day 3: +$500 → total $3,600, best day $1,600 / $3,600 = 44.4% < 50% ✓
    result = rules.evaluate_day(day_pnl=500.0, eod_balance=53600.0)
    assert result == EvalOutcome.PASS


def test_fail_when_drawdown_breached():
    rules = EvaluationRules(TopstepConfig())
    # Day 1: +$500 → balance $50,500, high water $50,500, floor = $48,500
    result = rules.evaluate_day(day_pnl=500.0, eod_balance=50500.0)
    assert result == EvalOutcome.ACTIVE
    # Day 2: -$2,100 → balance $48,400 < floor $48,500
    result = rules.evaluate_day(day_pnl=-2100.0, eod_balance=48400.0)
    assert result == EvalOutcome.FAIL


def test_fail_at_exact_drawdown_limit():
    config = TopstepConfig()
    rules = EvaluationRules(config)
    # Start at $50,000. Floor = $48,000.
    # Day 1: -$2,000 → balance $48,000 = floor exactly
    result = rules.evaluate_day(day_pnl=-2000.0, eod_balance=48000.0)
    assert result == EvalOutcome.FAIL


def test_trailing_drawdown_moves_up():
    rules = EvaluationRules(TopstepConfig())
    # Day 1: +$1,000 → balance $51,000, high water moves to $51,000, floor = $49,000
    rules.evaluate_day(day_pnl=1000.0, eod_balance=51000.0)
    assert rules.eod_high_water == 51000.0
    assert rules.drawdown_floor == 49000.0
    # Day 2: -$500 → balance $50,500 > floor $49,000
    result = rules.evaluate_day(day_pnl=-500.0, eod_balance=50500.0)
    assert result == EvalOutcome.ACTIVE
    assert rules.eod_high_water == 51000.0  # didn't move
    assert rules.drawdown_floor == 49000.0  # didn't move


def test_consistency_check():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=2800.0, eod_balance=52800.0)
    rules.evaluate_day(day_pnl=300.0, eod_balance=53100.0)
    # Total $3,100 > $3,000 target BUT best day $2,800 / $3,100 = 90% > 50%
    assert rules.consistency_satisfied is False
    assert rules.total_profit == 3100.0
    assert rules.best_day_pnl == 2800.0


def test_timeout():
    config = TopstepConfig(max_attempt_days=3)
    rules = EvaluationRules(config)
    rules.evaluate_day(day_pnl=100.0, eod_balance=50100.0)
    rules.evaluate_day(day_pnl=100.0, eod_balance=50200.0)
    result = rules.evaluate_day(day_pnl=100.0, eod_balance=50300.0)
    assert result == EvalOutcome.TIMEOUT


def test_reset_clears_state():
    rules = EvaluationRules(TopstepConfig())
    rules.evaluate_day(day_pnl=1000.0, eod_balance=51000.0)
    rules.reset()
    assert rules.eod_high_water == 50000.0
    assert rules.total_profit == 0.0
    assert rules.days_traded == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_evaluation_rules.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement EvaluationRules**

```python
# topstep/evaluation_rules.py
"""Topstep evaluation rules: trailing EOD drawdown, consistency, pass/fail."""

from enum import Enum, auto

from topstep.config import TopstepConfig


class EvalOutcome(Enum):
    ACTIVE = auto()
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()


class EvaluationRules:
    """Evaluates end-of-day account state against Topstep Combine rules."""

    def __init__(self, config: TopstepConfig) -> None:
        self.config = config
        self.eod_high_water: float = config.account_size
        self.drawdown_floor: float = config.account_size - config.max_loss
        self.total_profit: float = 0.0
        self.best_day_pnl: float = 0.0
        self.days_traded: int = 0
        self.daily_pnls: list[float] = []

    def evaluate_day(self, day_pnl: float, eod_balance: float) -> EvalOutcome:
        """Call at end of each trading day. Returns the evaluation outcome."""
        self.days_traded += 1
        self.daily_pnls.append(day_pnl)

        if day_pnl > 0:
            self.total_profit += day_pnl
        if day_pnl > self.best_day_pnl:
            self.best_day_pnl = day_pnl

        # Update trailing drawdown (EOD only)
        if eod_balance > self.eod_high_water:
            self.eod_high_water = eod_balance
            self.drawdown_floor = self.eod_high_water - self.config.max_loss

        # Check fail: balance at or below drawdown floor
        if eod_balance <= self.drawdown_floor:
            return EvalOutcome.FAIL

        # Check pass: profit target met AND consistency satisfied
        cumulative_pnl = eod_balance - self.config.account_size
        if cumulative_pnl >= self.config.profit_target and self.consistency_satisfied:
            return EvalOutcome.PASS

        # Check timeout
        if self.days_traded >= self.config.max_attempt_days:
            return EvalOutcome.TIMEOUT

        return EvalOutcome.ACTIVE

    @property
    def consistency_satisfied(self) -> bool:
        """Best single day must be ≤ consistency_pct of total profit."""
        if self.total_profit <= 0:
            return False
        return self.best_day_pnl <= self.config.consistency_pct * self.total_profit

    def reset(self) -> None:
        """Reset for a new evaluation attempt."""
        self.eod_high_water = self.config.account_size
        self.drawdown_floor = self.config.account_size - self.config.max_loss
        self.total_profit = 0.0
        self.best_day_pnl = 0.0
        self.days_traded = 0
        self.daily_pnls = []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_evaluation_rules.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add topstep/evaluation_rules.py tests/test_evaluation_rules.py
git commit -m "feat: add EvaluationRules with trailing drawdown, consistency, pass/fail"
```

---

### Task 1.2: StateManager

**Files:**
- Create: `topstep/state_manager.py`
- Create: `tests/test_state_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_state_manager.py
from topstep.state_manager import StateManager, EvalState


def test_normal_state_at_zero_pnl():
    sm = StateManager()
    assert sm.get_state(0.0) == EvalState.NORMAL
    assert sm.get_position_multiplier(0.0) == 1.0
    assert sm.get_stop_multiplier(0.0) == 1.0


def test_careful_state():
    sm = StateManager()
    assert sm.get_state(800.0) == EvalState.CAREFUL
    assert sm.get_position_multiplier(800.0) == 0.7
    assert sm.get_stop_multiplier(800.0) == 0.7


def test_repeat_state():
    sm = StateManager()
    assert sm.get_state(2000.0) == EvalState.REPEAT
    assert sm.get_position_multiplier(2000.0) == 0.8
    assert sm.get_stop_multiplier(2000.0) == 0.9


def test_aggressive_state():
    sm = StateManager()
    assert sm.get_state(-700.0) == EvalState.AGGRESSIVE
    assert sm.get_position_multiplier(-700.0) == 1.3
    assert sm.get_stop_multiplier(-700.0) == 1.2


def test_yolo_state():
    sm = StateManager()
    assert sm.get_state(-1200.0) == EvalState.YOLO
    assert sm.get_position_multiplier(-1200.0) == 1.8
    assert sm.get_stop_multiplier(-1200.0) == 1.5


def test_hail_mary_state():
    sm = StateManager()
    assert sm.get_state(-1700.0) == EvalState.HAIL_MARY
    assert sm.get_position_multiplier(-1700.0) == 2.5
    assert sm.get_stop_multiplier(-1700.0) == 2.0


def test_boundary_values():
    sm = StateManager()
    # Exact boundaries — test which side they fall on
    assert sm.get_state(500.0) == EvalState.CAREFUL     # +$500 is CAREFUL
    assert sm.get_state(499.99) == EvalState.NORMAL      # just below
    assert sm.get_state(1500.0) == EvalState.REPEAT      # +$1500 is REPEAT
    assert sm.get_state(-500.0) == EvalState.AGGRESSIVE   # -$500 is AGGRESSIVE
    assert sm.get_state(-1000.0) == EvalState.YOLO        # -$1000 is YOLO
    assert sm.get_state(-1500.0) == EvalState.HAIL_MARY   # -$1500 is HAIL_MARY


def test_disabled_state_manager():
    """When disabled, always returns 1.0 multipliers."""
    sm = StateManager(enabled=False)
    assert sm.get_position_multiplier(-1700.0) == 1.0
    assert sm.get_stop_multiplier(-1700.0) == 1.0
    assert sm.get_state(-1700.0) == EvalState.HAIL_MARY  # state still computed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_state_manager.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement StateManager**

```python
# topstep/state_manager.py
"""Adaptive position sizing state machine for Topstep evaluations."""

from enum import Enum, auto


class EvalState(Enum):
    NORMAL = auto()
    CAREFUL = auto()
    REPEAT = auto()
    AGGRESSIVE = auto()
    YOLO = auto()
    HAIL_MARY = auto()


# (state, position_multiplier, stop_multiplier)
_STATE_CONFIG = {
    EvalState.NORMAL:     (1.0, 1.0),
    EvalState.CAREFUL:    (0.7, 0.7),
    EvalState.REPEAT:     (0.8, 0.9),
    EvalState.AGGRESSIVE: (1.3, 1.2),
    EvalState.YOLO:       (1.8, 1.5),
    EvalState.HAIL_MARY:  (2.5, 2.0),
}


class StateManager:
    """Maps cumulative P&L to strategy behavior multipliers."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def get_state(self, cumulative_pnl: float) -> EvalState:
        """Determine the current evaluation state based on P&L."""
        if cumulative_pnl >= 1500.0:
            return EvalState.REPEAT
        elif cumulative_pnl >= 500.0:
            return EvalState.CAREFUL
        elif cumulative_pnl >= -500.0:
            return EvalState.NORMAL
        elif cumulative_pnl > -1000.0:
            return EvalState.AGGRESSIVE
        elif cumulative_pnl > -1500.0:
            return EvalState.YOLO
        else:
            return EvalState.HAIL_MARY

    def get_position_multiplier(self, cumulative_pnl: float) -> float:
        """Return position size multiplier for current state."""
        if not self.enabled:
            return 1.0
        state = self.get_state(cumulative_pnl)
        return _STATE_CONFIG[state][0]

    def get_stop_multiplier(self, cumulative_pnl: float) -> float:
        """Return stop distance multiplier for current state."""
        if not self.enabled:
            return 1.0
        state = self.get_state(cumulative_pnl)
        return _STATE_CONFIG[state][1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_state_manager.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add topstep/state_manager.py tests/test_state_manager.py
git commit -m "feat: add StateManager with 6-state adaptive position sizing"
```

---

### Task 1.3: AttemptTracker

**Files:**
- Create: `topstep/attempt_tracker.py`
- Create: `tests/test_attempt_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_attempt_tracker.py
from topstep.config import TopstepConfig
from topstep.attempt_tracker import AttemptTracker, AttemptStatus
from topstep.evaluation_rules import EvalOutcome


def test_initial_state():
    tracker = AttemptTracker(TopstepConfig())
    assert tracker.status == AttemptStatus.ACTIVE
    assert tracker.cumulative_pnl == 0.0
    assert tracker.days_traded == 0
    assert tracker.daily_pnls == []


def test_record_day():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=500.0, eod_balance=50500.0)
    assert tracker.cumulative_pnl == 500.0
    assert tracker.days_traded == 1
    assert tracker.daily_pnls == [500.0]


def test_pass_updates_status():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=1500.0, eod_balance=51500.0)
    assert tracker.status == AttemptStatus.ACTIVE
    tracker.record_day(pnl=1600.0, eod_balance=53100.0)
    assert tracker.status == AttemptStatus.PASS


def test_fail_updates_status():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=-2100.0, eod_balance=47900.0)
    assert tracker.status == AttemptStatus.FAIL


def test_state_history_tracked():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=600.0, eod_balance=50600.0)  # → CAREFUL
    tracker.record_day(pnl=-1300.0, eod_balance=49300.0)  # → AGGRESSIVE
    assert len(tracker.state_history) == 2


def test_to_dict():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=500.0, eod_balance=50500.0)
    d = tracker.to_dict()
    assert d["cumulative_pnl"] == 500.0
    assert d["days_traded"] == 1
    assert d["status"] == "active"
    assert "state_history" in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_attempt_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement AttemptTracker**

```python
# topstep/attempt_tracker.py
"""Per-attempt state tracking for Topstep evaluation simulation."""

from enum import Enum, auto

from topstep.config import TopstepConfig
from topstep.evaluation_rules import EvaluationRules, EvalOutcome
from topstep.state_manager import StateManager, EvalState


class AttemptStatus(Enum):
    ACTIVE = auto()
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()


_OUTCOME_TO_STATUS = {
    EvalOutcome.ACTIVE: AttemptStatus.ACTIVE,
    EvalOutcome.PASS: AttemptStatus.PASS,
    EvalOutcome.FAIL: AttemptStatus.FAIL,
    EvalOutcome.TIMEOUT: AttemptStatus.TIMEOUT,
}


class AttemptTracker:
    """Tracks a single evaluation attempt's progress."""

    def __init__(self, config: TopstepConfig, state_machine_enabled: bool = True) -> None:
        self.config = config
        self.rules = EvaluationRules(config)
        self.state_manager = StateManager(enabled=state_machine_enabled)
        self.cumulative_pnl: float = 0.0
        self.daily_pnls: list[float] = []
        self.state_history: list[EvalState] = []
        self.status: AttemptStatus = AttemptStatus.ACTIVE

    @property
    def days_traded(self) -> int:
        return self.rules.days_traded

    def record_day(self, pnl: float, eod_balance: float) -> AttemptStatus:
        """Record end-of-day results and evaluate rules."""
        self.cumulative_pnl += pnl
        self.daily_pnls.append(pnl)
        self.state_history.append(self.state_manager.get_state(self.cumulative_pnl))

        outcome = self.rules.evaluate_day(day_pnl=pnl, eod_balance=eod_balance)
        self.status = _OUTCOME_TO_STATUS[outcome]
        return self.status

    def to_dict(self) -> dict:
        """Serialize attempt state for storage."""
        return {
            "cumulative_pnl": self.cumulative_pnl,
            "days_traded": self.days_traded,
            "daily_pnls": self.daily_pnls,
            "best_day_pnl": self.rules.best_day_pnl,
            "worst_day_pnl": min(self.daily_pnls) if self.daily_pnls else 0.0,
            "status": self.status.name.lower(),
            "state_history": [s.name for s in self.state_history],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_attempt_tracker.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add topstep/attempt_tracker.py tests/test_attempt_tracker.py
git commit -m "feat: add AttemptTracker with evaluation rules and state history"
```

---

## Phase 2: Strategies (2 parallel agents)

> **These 2 tasks have NO dependencies on each other.** Both depend on Phase 0 (ATR, TopstepConfig) and Phase 1 (StateManager).

---

### Task 2.1: ORBStrategy

**Files:**
- Create: `topstep/strategies/orb_strategy.py`
- Create: `tests/test_orb_strategy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orb_strategy.py
from datetime import date
from core.events import BarEvent, SignalEvent
from topstep.strategies.orb_strategy import ORBStrategy
from topstep.state_manager import StateManager


def _make_bar(symbol="ES", d=date(2024, 6, 1), o=5000.0, h=5050.0, l=4960.0, c=5030.0, v=1000000):
    return BarEvent(symbol=symbol, date=d, open=o, high=h, low=l, close=c, adj_close=c, volume=v)


class MockPortfolio:
    def has_position(self, symbol):
        return False
    def get_equity(self, prices=None):
        return 50000.0


def test_orb_warm_up_period():
    strategy = ORBStrategy()
    assert strategy.warm_up_period() >= 5


def test_orb_no_signal_during_warm_up():
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    bar = _make_bar()
    # First bar during warm-up — should update indicators but not signal
    signals = strategy.generate_signals(bar, portfolio)
    assert signals == []


def test_orb_bullish_signal_after_warm_up():
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    # Feed warm-up bars
    for i in range(strategy.warm_up_period()):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0+i, h=5050.0+i, l=4960.0+i, c=5030.0+i)
        strategy.generate_signals(bar, portfolio)
    # Bullish bar: close > open by significant margin
    bullish_bar = _make_bar(d=date(2024, 2, 1), o=5000.0, h=5080.0, l=4990.0, c=5070.0)
    signals = strategy.generate_signals(bullish_bar, portfolio)
    # Should produce a long signal
    long_signals = [s for s in signals if s.direction == "long"]
    assert len(long_signals) >= 1


def test_orb_bearish_signal():
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0-i, h=5050.0-i, l=4960.0-i, c=4970.0-i)
        strategy.generate_signals(bar, portfolio)
    # Bearish bar: close < open by significant margin
    bearish_bar = _make_bar(d=date(2024, 2, 1), o=5000.0, h=5010.0, l=4920.0, c=4930.0)
    signals = strategy.generate_signals(bearish_bar, portfolio)
    short_signals = [s for s in signals if s.direction == "short"]
    assert len(short_signals) >= 1


def test_orb_state_manager_scales_strength():
    sm = StateManager(enabled=True)
    strategy = ORBStrategy(state_manager=sm)
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0+i, h=5050.0+i, l=4960.0+i, c=5030.0+i)
        strategy.generate_signals(bar, portfolio)
    bullish_bar = _make_bar(d=date(2024, 2, 1), o=5000.0, h=5080.0, l=4990.0, c=5070.0)
    signals = strategy.generate_signals(bullish_bar, portfolio)
    # Strength should reflect state manager multiplier
    assert len(signals) >= 1
    assert signals[0].strength > 0


def test_orb_no_signal_on_doji():
    """Small candle body relative to ATR — no conviction, no signal."""
    strategy = ORBStrategy()
    portfolio = MockPortfolio()
    for i in range(strategy.warm_up_period()):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0, h=5050.0, l=4950.0, c=5000.0+i)
        strategy.generate_signals(bar, portfolio)
    # Doji: close ≈ open
    doji = _make_bar(d=date(2024, 2, 1), o=5000.0, h=5020.0, l=4980.0, c=5001.0)
    signals = strategy.generate_signals(doji, portfolio)
    assert signals == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_orb_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ORBStrategy**

```python
# topstep/strategies/orb_strategy.py
"""Opening Range Breakout strategy for Topstep evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.events import BarEvent, SignalEvent
from indicators.technical import ATR, SMA
from strategy.base import Strategy

if TYPE_CHECKING:
    from topstep.state_manager import StateManager


class ORBStrategy(Strategy):
    """Trades daily bar direction as a proxy for opening range breakout.

    Long when close > open by a threshold (bullish structure).
    Short when close < open by a threshold (bearish structure).
    Threshold scaled by ATR to filter noise.
    """

    def __init__(
        self,
        atr_period: int = 14,
        sma_period: int = 5,
        body_threshold: float = 0.3,
        state_manager: "StateManager | None" = None,
    ) -> None:
        super().__init__()
        self._atr = ATR(period=atr_period)
        self._sma = SMA(period=sma_period)
        self._body_threshold = body_threshold
        self._state_manager = state_manager
        self._bar_count = 0
        self._warm_up = max(atr_period, sma_period)

    def warm_up_period(self) -> int:
        return self._warm_up

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        # Update indicators
        self._atr.update(high=bar.high, low=bar.low, close=bar.close)
        self._sma.update(bar.close)
        self._bar_count += 1

        if self._bar_count <= self._warm_up:
            return []

        atr_val = self._atr.value
        if atr_val is None or atr_val <= 0:
            return []

        body = bar.close - bar.open
        body_ratio = abs(body) / atr_val

        if body_ratio < self._body_threshold:
            return []  # Doji / indecision — no signal

        pos_mult = 1.0
        if self._state_manager is not None:
            equity = portfolio.get_equity() if hasattr(portfolio, "get_equity") else 50000.0
            pnl = equity - 50000.0
            pos_mult = self._state_manager.get_position_multiplier(pnl)

        direction = "long" if body > 0 else "short"
        reason = f"ORB {'bullish' if body > 0 else 'bearish'}: body/ATR={body_ratio:.2f}"

        return [SignalEvent(
            symbol=bar.symbol,
            direction=direction,
            reason=reason,
            strength=min(body_ratio * pos_mult, 1.0),
        )]

    @classmethod
    def get_parameter_space(cls) -> dict:
        return {
            "atr_period": [10, 14, 20],
            "sma_period": [3, 5, 10],
            "body_threshold": [0.2, 0.3, 0.5],
        }

    @classmethod
    def from_params(cls, params: dict) -> "ORBStrategy":
        return cls(**params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_orb_strategy.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add topstep/strategies/orb_strategy.py tests/test_orb_strategy.py
git commit -m "feat: add ORBStrategy for Topstep evaluation"
```

---

### Task 2.2: VWAPReversionStrategy

**Files:**
- Create: `topstep/strategies/vwap_reversion_strategy.py`
- Create: `tests/test_vwap_strategy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vwap_strategy.py
from datetime import date
from core.events import BarEvent, SignalEvent
from topstep.strategies.vwap_reversion_strategy import VWAPReversionStrategy
from topstep.state_manager import StateManager


def _make_bar(symbol="ES", d=date(2024, 6, 1), o=5000.0, h=5050.0, l=4960.0, c=5030.0, v=1000000):
    return BarEvent(symbol=symbol, date=d, open=o, high=h, low=l, close=c, adj_close=c, volume=v)


class MockPortfolio:
    def has_position(self, symbol):
        return False
    def get_equity(self, prices=None):
        return 50000.0


def test_vwap_warm_up_period():
    strategy = VWAPReversionStrategy()
    assert strategy.warm_up_period() >= 20


def test_vwap_no_signal_during_warm_up():
    strategy = VWAPReversionStrategy()
    portfolio = MockPortfolio()
    bar = _make_bar()
    signals = strategy.generate_signals(bar, portfolio)
    assert signals == []


def test_vwap_long_signal_on_oversold():
    """Price well below SMA → mean reversion long signal."""
    strategy = VWAPReversionStrategy(sma_period=5, atr_period=5, deviation_threshold=1.0)
    portfolio = MockPortfolio()
    # Feed warm-up bars at stable price
    for i in range(5):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0, h=5020.0, l=4980.0, c=5000.0)
        strategy.generate_signals(bar, portfolio)
    # Drop price well below SMA (SMA ≈ 5000, close at 4920 = 80pts below, ATR ≈ 40)
    oversold_bar = _make_bar(d=date(2024, 1, 8), o=4940.0, h=4950.0, l=4910.0, c=4920.0)
    signals = strategy.generate_signals(oversold_bar, portfolio)
    long_signals = [s for s in signals if s.direction == "long"]
    assert len(long_signals) >= 1


def test_vwap_short_signal_on_overbought():
    """Price well above SMA → mean reversion short signal."""
    strategy = VWAPReversionStrategy(sma_period=5, atr_period=5, deviation_threshold=1.0)
    portfolio = MockPortfolio()
    for i in range(5):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0, h=5020.0, l=4980.0, c=5000.0)
        strategy.generate_signals(bar, portfolio)
    overbought_bar = _make_bar(d=date(2024, 1, 8), o=5060.0, h=5090.0, l=5050.0, c=5080.0)
    signals = strategy.generate_signals(overbought_bar, portfolio)
    short_signals = [s for s in signals if s.direction == "short"]
    assert len(short_signals) >= 1


def test_vwap_no_signal_near_fair_value():
    """Price near SMA → no signal."""
    strategy = VWAPReversionStrategy(sma_period=5, atr_period=5, deviation_threshold=1.0)
    portfolio = MockPortfolio()
    for i in range(5):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0, h=5020.0, l=4980.0, c=5000.0)
        strategy.generate_signals(bar, portfolio)
    fair_bar = _make_bar(d=date(2024, 1, 8), o=4998.0, h=5010.0, l=4990.0, c=5002.0)
    signals = strategy.generate_signals(fair_bar, portfolio)
    assert signals == []


def test_vwap_state_manager_integration():
    sm = StateManager(enabled=True)
    strategy = VWAPReversionStrategy(sma_period=5, atr_period=5, state_manager=sm)
    portfolio = MockPortfolio()
    for i in range(5):
        bar = _make_bar(d=date(2024, 1, 2 + i), o=5000.0, h=5020.0, l=4980.0, c=5000.0)
        strategy.generate_signals(bar, portfolio)
    oversold_bar = _make_bar(d=date(2024, 1, 8), o=4940.0, h=4950.0, l=4910.0, c=4920.0)
    signals = strategy.generate_signals(oversold_bar, portfolio)
    assert len(signals) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_vwap_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement VWAPReversionStrategy**

```python
# topstep/strategies/vwap_reversion_strategy.py
"""VWAP Mean Reversion strategy for Topstep evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.events import BarEvent, SignalEvent
from indicators.technical import ATR, SMA
from strategy.base import Strategy

if TYPE_CHECKING:
    from topstep.state_manager import StateManager


class VWAPReversionStrategy(Strategy):
    """Fades deviations from a short-term SMA (VWAP proxy on daily bars).

    Long when price drops > deviation_threshold ATRs below SMA.
    Short when price rises > deviation_threshold ATRs above SMA.
    """

    def __init__(
        self,
        sma_period: int = 5,
        atr_period: int = 14,
        deviation_threshold: float = 1.0,
        state_manager: "StateManager | None" = None,
    ) -> None:
        super().__init__()
        self._sma = SMA(period=sma_period)
        self._atr = ATR(period=atr_period)
        self._deviation_threshold = deviation_threshold
        self._state_manager = state_manager
        self._bar_count = 0
        self._warm_up = max(sma_period, atr_period)

    def warm_up_period(self) -> int:
        return self._warm_up

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        self._sma.update(bar.close)
        self._atr.update(high=bar.high, low=bar.low, close=bar.close)
        self._bar_count += 1

        if self._bar_count <= self._warm_up:
            return []

        sma_val = self._sma.value
        atr_val = self._atr.value
        if sma_val is None or atr_val is None or atr_val <= 0:
            return []

        deviation = (bar.close - sma_val) / atr_val

        if abs(deviation) < self._deviation_threshold:
            return []  # Near fair value — no trade

        pos_mult = 1.0
        if self._state_manager is not None:
            equity = portfolio.get_equity() if hasattr(portfolio, "get_equity") else 50000.0
            pnl = equity - 50000.0
            pos_mult = self._state_manager.get_position_multiplier(pnl)

        if deviation < -self._deviation_threshold:
            direction = "long"
            reason = f"VWAP reversion long: {deviation:.2f} ATRs below SMA"
        else:
            direction = "short"
            reason = f"VWAP reversion short: {deviation:.2f} ATRs above SMA"

        return [SignalEvent(
            symbol=bar.symbol,
            direction=direction,
            reason=reason,
            strength=min(abs(deviation) / 3.0 * pos_mult, 1.0),
        )]

    @classmethod
    def get_parameter_space(cls) -> dict:
        return {
            "sma_period": [3, 5, 10],
            "atr_period": [10, 14, 20],
            "deviation_threshold": [0.8, 1.0, 1.5, 2.0],
        }

    @classmethod
    def from_params(cls, params: dict) -> "VWAPReversionStrategy":
        return cls(**params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_vwap_strategy.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add topstep/strategies/vwap_reversion_strategy.py tests/test_vwap_strategy.py
git commit -m "feat: add VWAPReversionStrategy for Topstep evaluation"
```

---

## Phase 3: Simulator + Campaign Runner (sequential — simulator first, then runner)

---

### Task 3.1: TopstepEvalSimulator

This is the core orchestrator. It wraps BacktestEngine, subscribes to PortfolioUpdateEvents, and runs evaluation rules after each trading day.

**Files:**
- Create: `topstep/simulator.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_simulator.py
import tempfile
from datetime import date

from core.events import BarEvent
from data.storage.database import Database
from data.storage.models import DailyBar
from topstep.config import TopstepConfig
from topstep.simulator import TopstepEvalSimulator
from topstep.attempt_tracker import AttemptStatus
from topstep.strategies.orb_strategy import ORBStrategy


def _seed_db(db: Database, symbol: str = "ES", num_bars: int = 60):
    """Create deterministic trending bar data that should trigger signals."""
    db.create_tables()
    bars = []
    base = 5000.0
    for i in range(num_bars):
        d = date(2024, 1, 2 + i)
        # Trending up: each bar opens higher, closes higher
        o = base + i * 2
        h = o + 30
        l = o - 10
        c = o + 20  # bullish candles
        bars.append(DailyBar(
            symbol=symbol, date=d, open=o, high=h, low=l,
            close=c, adj_close=c, volume=1_000_000, vwap=o + 10,
        ))
    db.insert_daily_bars(bars)
    return bars


def test_simulator_runs_to_completion():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 60)

        config = TopstepConfig(max_attempt_days=30)
        strategy = ORBStrategy()
        sim = TopstepEvalSimulator(
            strategy=strategy,
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))

        assert result["status"] in ("pass", "fail", "timeout")
        assert result["days_traded"] >= 1
        assert "cumulative_pnl" in result
        assert "daily_pnls" in result
        db.close()


def test_simulator_respects_max_days():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 60)

        config = TopstepConfig(max_attempt_days=5)
        strategy = ORBStrategy()
        sim = TopstepEvalSimulator(
            strategy=strategy,
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))

        assert result["days_traded"] <= 5
        db.close()


def test_simulator_with_state_machine():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 60)

        config = TopstepConfig(max_attempt_days=30)
        strategy = ORBStrategy()
        sim = TopstepEvalSimulator(
            strategy=strategy,
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=True,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))

        assert result["status"] in ("pass", "fail", "timeout")
        assert "state_history" in result
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_simulator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement TopstepEvalSimulator**

```python
# topstep/simulator.py
"""Topstep evaluation simulator — wraps BacktestEngine for single attempt."""

from __future__ import annotations

import threading
from datetime import date, timedelta

from core.engine import BacktestEngine
from core.events import EventType, PortfolioUpdateEvent
from data.storage.database import Database
from strategy.base import Strategy
from topstep.attempt_tracker import AttemptTracker, AttemptStatus
from topstep.config import TopstepConfig
from topstep.state_manager import StateManager


class TopstepEvalSimulator:
    """Runs a single Topstep evaluation attempt using the existing BacktestEngine."""

    def __init__(
        self,
        strategy: Strategy,
        database: Database,
        instrument: str,
        config: TopstepConfig,
        state_machine_enabled: bool = True,
    ) -> None:
        self.strategy = strategy
        self.database = database
        self.instrument = instrument
        self.config = config
        self.tracker = AttemptTracker(config, state_machine_enabled=state_machine_enabled)
        self._cancel = threading.Event()

    def run_attempt(self, start_date: date) -> dict:
        """Run a single evaluation attempt starting from the given date.

        Returns a dict with attempt outcome, P&L, days traded, etc.
        """
        end_date = start_date + timedelta(days=self.config.max_attempt_days * 2)

        engine = BacktestEngine(
            strategy=self.strategy,
            database=self.database,
            universe=[self.instrument],
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.account_size,
            benchmark_symbol=self.instrument,
            slippage_pct=0.0001,
            commission_per_share=0.005,
            cancel_event=self._cancel,
            quiet=True,
        )

        # Track daily equity snapshots from PortfolioUpdateEvent
        daily_equity: list[tuple[date, float]] = []

        def _on_portfolio_update(event: PortfolioUpdateEvent) -> None:
            daily_equity.append((event.timestamp.date() if hasattr(event.timestamp, 'date') else date.today(), event.equity))

        engine.event_bus.subscribe(EventType.PORTFOLIO_UPDATE, _on_portfolio_update)

        # Run the engine — it will iterate day by day
        metrics = engine.run()

        # Process equity snapshots into daily P&L for evaluation
        if daily_equity:
            prev_equity = self.config.account_size
            seen_dates: set[date] = set()
            for eq_date, equity in daily_equity:
                if eq_date in seen_dates:
                    continue  # Multiple updates per day — take last
                seen_dates.add(eq_date)
                day_pnl = equity - prev_equity
                status = self.tracker.record_day(pnl=day_pnl, eod_balance=equity)
                prev_equity = equity

                if status != AttemptStatus.ACTIVE:
                    self._cancel.set()
                    break

        return self.tracker.to_dict()
```

Note: The simulator runs the full BacktestEngine and intercepts PortfolioUpdateEvents. When a pass/fail/timeout occurs, it sets the cancel_event to stop the engine early. The `run_attempt` method returns the tracker's dict representation.

**Important implementation detail:** The engine's PortfolioUpdateEvent is emitted after each fill and after `record_equity()` per date. We collect all equity values and deduplicate per date, using the last equity value of each day as the EOD balance. This may need adjustment based on exactly when the engine emits these events — check `core/engine.py` and `portfolio/portfolio.py` during implementation to get the right EOD equity.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_simulator.py -v`
Expected: ALL PASS (or tests may need fixture adjustments based on actual engine behavior — the key is that the simulator runs to completion without errors)

- [ ] **Step 5: Commit**

```bash
git add topstep/simulator.py tests/test_simulator.py
git commit -m "feat: add TopstepEvalSimulator wrapping BacktestEngine"
```

---

### Task 3.2: CampaignRunner

**Files:**
- Create: `topstep/campaign_runner.py`
- Create: `tests/test_campaign_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_campaign_runner.py
import tempfile
from datetime import date

from data.storage.database import Database
from data.storage.models import DailyBar
from topstep.campaign_runner import CampaignRunner, CampaignResult
from topstep.config import TopstepConfig
from topstep.strategies.orb_strategy import ORBStrategy


def _seed_db(db: Database, symbol: str = "ES", num_bars: int = 200):
    db.create_tables()
    bars = []
    base = 5000.0
    for i in range(num_bars):
        d = date(2024, 1, 2) + __import__("datetime").timedelta(days=i)
        o = base + (i % 50) * 2
        h = o + 30 + (i % 7) * 5
        l = o - 10 - (i % 5) * 3
        c = o + 15 + ((-1) ** i) * 10  # alternating up/down bias
        bars.append(DailyBar(
            symbol=symbol, date=d, open=o, high=h, low=l,
            close=c, adj_close=c, volume=1_000_000, vwap=o + 10,
        ))
    db.insert_daily_bars(bars)


def test_campaign_runner_basic():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 200)

        config = TopstepConfig(max_attempt_days=10)
        runner = CampaignRunner(
            strategy_class=ORBStrategy,
            instrument="ES",
            config=config,
            database=db,
            state_machine_enabled=False,
            num_attempts=10,
            seed=42,
        )
        result = runner.run()

        assert isinstance(result, CampaignResult)
        assert result.num_attempts == 10
        assert 0.0 <= result.pass_rate <= 1.0
        assert len(result.attempt_outcomes) == 10
        assert result.campaign_id is not None
        db.close()


def test_campaign_runner_reproducible_with_seed():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 200)

        config = TopstepConfig(max_attempt_days=10)

        runner1 = CampaignRunner(
            strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=10, seed=42,
        )
        result1 = runner1.run()

        runner2 = CampaignRunner(
            strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=10, seed=42,
        )
        result2 = runner2.run()

        assert result1.pass_rate == result2.pass_rate
        db.close()


def test_campaign_runner_computes_ev():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 200)

        config = TopstepConfig(max_attempt_days=10)
        runner = CampaignRunner(
            strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=20, seed=42,
        )
        result = runner.run()

        # EV calculation should exist
        assert isinstance(result.ev_per_attempt, float)
        assert isinstance(result.cost_to_funded, float)
        assert isinstance(result.annual_ev, float)
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_campaign_runner.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CampaignRunner and CampaignResult**

```python
# topstep/campaign_runner.py
"""Campaign runner: executes N independent evaluation attempts and aggregates results."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Type

from data.storage.database import Database
from strategy.base import Strategy
from topstep.config import TopstepConfig
from topstep.simulator import TopstepEvalSimulator
from topstep.state_manager import StateManager


@dataclass
class CampaignResult:
    campaign_id: str
    strategy_name: str
    instrument: str
    state_machine_enabled: bool
    num_attempts: int
    seed: int

    # Pass-rate metrics
    pass_rate: float = 0.0
    avg_days_to_pass: float = 0.0
    avg_days_to_fail: float = 0.0
    ev_per_attempt: float = 0.0
    cost_to_funded: float = 0.0
    median_attempts_to_pass: float = 0.0
    annual_ev: float = 0.0

    # Distribution data
    attempt_outcomes: list[dict] = field(default_factory=list)
    pnl_distribution: list[float] = field(default_factory=list)
    days_distribution: list[int] = field(default_factory=list)
    state_usage: dict[str, float] = field(default_factory=dict)
    pass_by_regime: dict[str, float] = field(default_factory=dict)


class CampaignRunner:
    """Runs N independent evaluation attempts with random start dates."""

    def __init__(
        self,
        strategy_class: Type[Strategy],
        instrument: str,
        config: TopstepConfig,
        database: Database,
        state_machine_enabled: bool = True,
        num_attempts: int = 1000,
        seed: int = 42,
    ) -> None:
        self.strategy_class = strategy_class
        self.instrument = instrument
        self.config = config
        self.database = database
        self.state_machine_enabled = state_machine_enabled
        self.num_attempts = num_attempts
        self.seed = seed

    def run(self) -> CampaignResult:
        """Execute all attempts and return aggregated results."""
        rng = random.Random(self.seed)

        # Get available date range for the instrument
        date_range = self.database.get_cached_date_range(self.instrument)
        if date_range == (None, None):
            raise ValueError(f"No data for {self.instrument}")
        min_date, max_date = date_range

        # Need at least max_attempt_days * 2 of headroom for each attempt
        headroom = timedelta(days=self.config.max_attempt_days * 2)
        latest_start = max_date - headroom
        if latest_start <= min_date:
            raise ValueError(f"Not enough data: need {headroom.days} days, have {(max_date - min_date).days}")

        # Generate random start dates
        date_range_days = (latest_start - min_date).days
        start_dates = [
            min_date + timedelta(days=rng.randint(0, date_range_days))
            for _ in range(self.num_attempts)
        ]

        # Run attempts
        outcomes: list[dict] = []
        for start in start_dates:
            strategy = self.strategy_class()
            if self.state_machine_enabled:
                sm = StateManager(enabled=True)
                if hasattr(strategy, '_state_manager'):
                    strategy._state_manager = sm

            sim = TopstepEvalSimulator(
                strategy=strategy,
                database=self.database,
                instrument=self.instrument,
                config=self.config,
                state_machine_enabled=self.state_machine_enabled,
            )
            result = sim.run_attempt(start_date=start)
            outcomes.append(result)

        return self._aggregate(outcomes)

    def _aggregate(self, outcomes: list[dict]) -> CampaignResult:
        """Compute aggregate statistics from individual attempt outcomes."""
        passes = [o for o in outcomes if o["status"] == "pass"]
        fails = [o for o in outcomes if o["status"] == "fail"]

        pass_rate = len(passes) / len(outcomes) if outcomes else 0.0

        avg_days_to_pass = (
            sum(o["days_traded"] for o in passes) / len(passes)
            if passes else 0.0
        )
        avg_days_to_fail = (
            sum(o["days_traded"] for o in fails) / len(fails)
            if fails else 0.0
        )

        # EV calculation
        gross_payout = min(
            self.config.profit_target * self.config.payout_split,
            self.config.max_payout,
        )
        ev_per_attempt = (
            pass_rate * (gross_payout - self.config.activation_fee)
            - self.config.subscription_fee
        )

        cost_to_funded = (
            self.config.subscription_fee / pass_rate
            if pass_rate > 0 else float("inf")
        )
        median_attempts = 1.0 / pass_rate if pass_rate > 0 else float("inf")

        attempts_per_month = 4
        annual_ev = ev_per_attempt * attempts_per_month * 12

        # State usage across all attempts
        all_states: list[str] = []
        for o in outcomes:
            all_states.extend(o.get("state_history", []))
        state_counts: dict[str, int] = {}
        for s in all_states:
            state_counts[s] = state_counts.get(s, 0) + 1
        total_states = len(all_states) or 1
        state_usage = {k: v / total_states for k, v in state_counts.items()}

        return CampaignResult(
            campaign_id=str(uuid.uuid4())[:8],
            strategy_name=self.strategy_class.__name__,
            instrument=self.instrument,
            state_machine_enabled=self.state_machine_enabled,
            num_attempts=len(outcomes),
            seed=self.seed,
            pass_rate=pass_rate,
            avg_days_to_pass=avg_days_to_pass,
            avg_days_to_fail=avg_days_to_fail,
            ev_per_attempt=ev_per_attempt,
            cost_to_funded=cost_to_funded,
            median_attempts_to_pass=median_attempts,
            annual_ev=annual_ev,
            attempt_outcomes=outcomes,
            pnl_distribution=[o["cumulative_pnl"] for o in outcomes],
            days_distribution=[o["days_traded"] for o in outcomes],
            state_usage=state_usage,
            pass_by_regime={},  # TODO: integrate with existing regime detection in Phase 5
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -m pytest tests/test_campaign_runner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add topstep/campaign_runner.py tests/test_campaign_runner.py
git commit -m "feat: add CampaignRunner with EV calculation and result aggregation"
```

---

## Phase 4: Backend API

---

### Task 4.1: Eval router and schemas

**Files:**
- Modify: `backend/schemas.py` (append campaign schemas)
- Create: `backend/routers/eval.py`
- Modify: `backend/main.py` (add router include)

- [ ] **Step 1: Add campaign Pydantic schemas**

Append to `backend/schemas.py`:

```python
# --- Eval Campaign Schemas ---

class CampaignResponse(BaseModel):
    campaign_id: str
    strategy_name: str
    instrument: str
    state_machine: bool
    num_attempts: int
    pass_rate: float
    ev_per_attempt: float
    cost_to_funded: float
    avg_days_to_pass: float
    annual_ev: float
    created_at: Optional[datetime] = None


class CampaignListResponse(BaseModel):
    campaigns: list[CampaignResponse]
    total: int


class CampaignDetailResponse(CampaignResponse):
    full_results: Optional[dict] = None


class CampaignRunRequest(BaseModel):
    strategy: str = "orb"           # "orb" or "vwap"
    instrument: str = "ES"
    state_machine_enabled: bool = True
    account_tier: str = "50k"       # "50k", "100k", "150k"
    num_attempts: int = 1000
    seed: int = 42


class CampaignRunResponse(BaseModel):
    campaign_id: str
    status: str
```

- [ ] **Step 2: Create eval router**

```python
# backend/routers/eval.py
"""Eval campaign endpoints: list, get, delete, launch."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.dependencies import get_database
from backend.schemas import (
    CampaignDetailResponse,
    CampaignListResponse,
    CampaignResponse,
    CampaignRunRequest,
    CampaignRunResponse,
)
from data.storage.models import EvalCampaignRecord
from topstep.campaign_runner import CampaignRunner
from topstep.config import TIERS
from topstep.strategies.orb_strategy import ORBStrategy
from topstep.strategies.vwap_reversion_strategy import VWAPReversionStrategy

router = APIRouter(prefix="/api/eval", tags=["eval"])

_STRATEGIES = {"orb": ORBStrategy, "vwap": VWAPReversionStrategy}
_running: dict[str, threading.Thread] = {}


def _record_to_response(rec: EvalCampaignRecord) -> CampaignResponse:
    return CampaignResponse(
        campaign_id=rec.campaign_id,
        strategy_name=rec.strategy_name,
        instrument=rec.instrument,
        state_machine=rec.state_machine,
        num_attempts=rec.num_attempts,
        pass_rate=rec.pass_rate,
        ev_per_attempt=rec.ev_per_attempt,
        cost_to_funded=rec.cost_to_funded,
        avg_days_to_pass=rec.avg_days_to_pass,
        annual_ev=rec.annual_ev,
        created_at=rec.created_at,
    )


@router.get("/campaigns", response_model=CampaignListResponse)
def list_campaigns(
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    limit: int = Query(50),
    offset: int = Query(0),
):
    db = get_database()
    campaigns = db.list_eval_campaigns(sort=sort, order=order, limit=limit, offset=offset)
    total = len(db.list_eval_campaigns())
    return CampaignListResponse(
        campaigns=[_record_to_response(c) for c in campaigns],
        total=total,
    )


@router.get("/campaigns/{campaign_id}", response_model=CampaignDetailResponse)
def get_campaign(campaign_id: str):
    db = get_database()
    rec = db.get_eval_campaign(campaign_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    full = None
    if rec.full_results:
        try:
            full = json.loads(rec.full_results)
        except (json.JSONDecodeError, TypeError):
            full = None
    resp = _record_to_response(rec)
    return CampaignDetailResponse(**resp.model_dump(), full_results=full)


@router.delete("/campaigns/{campaign_id}")
def delete_campaign(campaign_id: str):
    db = get_database()
    if db.get_eval_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    db.delete_eval_campaign(campaign_id)
    return {"status": "deleted"}


@router.post("/campaigns/run", response_model=CampaignRunResponse)
def run_campaign(req: CampaignRunRequest):
    strategy_cls = _STRATEGIES.get(req.strategy)
    if strategy_cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")
    config = TIERS.get(req.account_tier)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown tier: {req.account_tier}")

    campaign_id = str(uuid.uuid4())[:8]

    def _run():
        db = get_database()
        runner = CampaignRunner(
            strategy_class=strategy_cls,
            instrument=req.instrument,
            config=config,
            database=db,
            state_machine_enabled=req.state_machine_enabled,
            num_attempts=req.num_attempts,
            seed=req.seed,
        )
        result = runner.run()
        result.campaign_id = campaign_id

        record = EvalCampaignRecord(
            campaign_id=campaign_id,
            strategy_name=result.strategy_name,
            instrument=result.instrument,
            state_machine=result.state_machine_enabled,
            topstep_config=json.dumps(config.__dict__),
            num_attempts=result.num_attempts,
            seed=result.seed,
            pass_rate=result.pass_rate,
            ev_per_attempt=result.ev_per_attempt,
            cost_to_funded=result.cost_to_funded,
            avg_days_to_pass=result.avg_days_to_pass,
            annual_ev=result.annual_ev,
            created_at=datetime.now(tz=timezone.utc),
            full_results=json.dumps({
                "attempt_outcomes": result.attempt_outcomes,
                "pnl_distribution": result.pnl_distribution,
                "days_distribution": result.days_distribution,
                "state_usage": result.state_usage,
                "pass_by_regime": result.pass_by_regime,
            }),
        )
        db.insert_eval_campaign(record)
        _running.pop(campaign_id, None)

    thread = threading.Thread(target=_run, daemon=True)
    _running[campaign_id] = thread
    thread.start()

    return CampaignRunResponse(campaign_id=campaign_id, status="running")


@router.get("/campaigns/{campaign_id}/status")
def campaign_status(campaign_id: str):
    if campaign_id in _running:
        return {"campaign_id": campaign_id, "status": "running"}
    db = get_database()
    rec = db.get_eval_campaign(campaign_id)
    if rec is not None:
        return {"campaign_id": campaign_id, "status": "completed"}
    return {"campaign_id": campaign_id, "status": "unknown"}
```

- [ ] **Step 3: Register eval router in main.py**

In `backend/main.py`, add the import and router include:

Add to imports: `from backend.routers import analytics, backtest, data, eval, runs, strategies, trades, walk_forward`

Note: `eval` is a Python builtin — if this causes a naming conflict, rename the import: `from backend.routers import eval as eval_router` and use `app.include_router(eval_router.router)`.

Add after the last `app.include_router(...)`:
```python
app.include_router(eval.router)
```

- [ ] **Step 4: Test the API manually**

Run: `cd /Users/chrislane/Desktop/Claude_Code/trading-bot && python -c "from backend.routers.eval import router; print('Router loaded:', router.prefix)"`
Expected: `Router loaded: /api/eval`

- [ ] **Step 5: Commit**

```bash
git add backend/schemas.py backend/routers/eval.py backend/main.py
git commit -m "feat: add eval campaign API endpoints"
```

---

## Phase 5: Dashboard Pages

> **These 2 tasks can run in parallel.** Both depend on Phase 4 (API endpoints exist). Before building these pages, read `frontend/AGENTS.md` which warns that Next.js APIs may differ from training data — check `node_modules/next/dist/docs/` for the actual API reference.

---

### Task 5.1: Campaign Browser page (`/eval`)

**Files:**
- Create: `frontend/app/eval/page.tsx`
- Create: `frontend/hooks/use-eval.ts`
- Create: `frontend/components/tables/campaigns-table.tsx`

This task creates the `/eval` page with the metrics strip and campaign table. Follow the exact patterns from the existing Run Browser page (`frontend/app/page.tsx`), sidebar (`frontend/components/layout/sidebar.tsx`), and hooks (`frontend/hooks/use-runs.ts`).

- [ ] **Step 1: Create TanStack Query hook for eval API**

Read `frontend/hooks/use-runs.ts` for the exact hook pattern (query key structure, fetch function, return shape). Create `frontend/hooks/use-eval.ts` following the same pattern but targeting `/api/eval/campaigns`.

- [ ] **Step 2: Create campaigns table component**

Read `frontend/components/tables/runs-table.tsx` for the DataTable pattern (column definitions, sorting, filtering). Create `frontend/components/tables/campaigns-table.tsx` with columns: campaign_id, strategy, state_machine, attempts, pass_rate, ev_per_attempt, cost_to_funded, avg_days, created_at.

Color coding: pass_rate green >30%, yellow 20-30%, red <20%. EV green if positive, red if negative.

- [ ] **Step 3: Create Campaign Browser page**

Read `frontend/app/page.tsx` (the Run Browser) for the page layout pattern. Create `frontend/app/eval/page.tsx` with:
- Metrics strip (6 KPIs): Best Pass Rate, Best EV/Attempt, Cost to Funded, Avg Days to Pass, Annual EV, Campaigns Run
- Campaign table below
- Click row → navigate to `/eval/[campaignId]`

- [ ] **Step 4: Add sidebar link**

Read `frontend/components/layout/sidebar.tsx`. Add "Eval" nav item between "Data" and "Paper" with an appropriate icon.

- [ ] **Step 5: Verify the page loads**

Run the frontend dev server and navigate to `/eval`. Verify the page renders without errors (will show empty state until campaigns are run).

- [ ] **Step 6: Commit**

```bash
git add frontend/app/eval/ frontend/hooks/use-eval.ts frontend/components/tables/campaigns-table.tsx frontend/components/layout/sidebar.tsx
git commit -m "feat: add Campaign Browser page at /eval"
```

---

### Task 5.2: Campaign Detail page (`/eval/[campaignId]`)

**Files:**
- Create: `frontend/app/eval/[campaignId]/page.tsx`
- Create: `frontend/components/charts/eval-fan-chart.tsx`
- Create: `frontend/components/charts/regime-pass-rate.tsx`
- Create: `frontend/components/charts/days-distribution.tsx`
- Create: `frontend/components/charts/state-usage.tsx`

This task creates the detail drill-down page. Follow existing Run Detail patterns from `frontend/app/runs/[runId]/page.tsx`.

- [ ] **Step 1: Create chart components**

Read existing chart components in `frontend/components/charts/` for the Recharts + dark theme pattern. Create four new chart components:

1. `eval-fan-chart.tsx` — Recharts AreaChart showing attempt equity curves as percentile bands (P5/P25/P50/P75/P95). Horizontal lines at +profit_target (green) and -max_loss (red).
2. `regime-pass-rate.tsx` — Horizontal bar chart with colored bars per regime (BULL green, BEAR red, SIDEWAYS yellow, HIGH_VOL purple).
3. `days-distribution.tsx` — Histogram with green bars (passes, clustered early) and red bars (failures, spread wider).
4. `state-usage.tsx` — Horizontal bar chart showing % time in each of the 6 states.

- [ ] **Step 2: Create Campaign Detail page**

Read `frontend/app/runs/[runId]/page.tsx` for the detail page layout. Create `frontend/app/eval/[campaignId]/page.tsx` with:
- Metrics strip (4 large KPIs): Pass Rate, EV/Attempt, Median Attempts to Pass, Annual EV
- 2×2 chart grid: fan chart, regime pass rate, days distribution, state usage

- [ ] **Step 3: Add hook for campaign detail**

Add `useCampaignDetail(campaignId)` to `frontend/hooks/use-eval.ts` — fetches `/api/eval/campaigns/{id}` with full_results.

- [ ] **Step 4: Verify the page loads**

Navigate to `/eval/[campaignId]` with a test campaign ID. Verify charts render correctly.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/eval/[campaignId]/ frontend/components/charts/eval-fan-chart.tsx frontend/components/charts/regime-pass-rate.tsx frontend/components/charts/days-distribution.tsx frontend/components/charts/state-usage.tsx frontend/hooks/use-eval.ts
git commit -m "feat: add Campaign Detail page with charts at /eval/[campaignId]"
```

---

## Dependency Graph Summary

```
Phase 0: Prerequisites
  Task 0.1: ATR indicator
  Task 0.2: eval_campaigns table + model
  Task 0.3: TopstepConfig + scaffold
    ↓
Phase 1: [EvaluationRules(1.1)] [StateManager(1.2)] [AttemptTracker(1.3)]  ← 3 PARALLEL
    ↓
Phase 2: [ORBStrategy(2.1)] [VWAPReversionStrategy(2.2)]  ← 2 PARALLEL
    ↓
Phase 3: Simulator(3.1) → CampaignRunner(3.2)  ← SEQUENTIAL
    ↓
Phase 4: Backend API(4.1)
    ↓
Phase 5: [Campaign Browser(5.1)] [Campaign Detail(5.2)]  ← 2 PARALLEL
```

**Total: 12 tasks across 6 phases. Maximum parallelism: 3 agents in Phase 1.**
