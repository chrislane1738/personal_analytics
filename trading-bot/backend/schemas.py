"""Pydantic response/request models for the trading bot dashboard API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunResponse(BaseModel):
    run_id: str
    mode: str
    strategy_name: str
    config: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    initial_capital: Optional[float] = None
    final_value: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    created_at: Optional[datetime] = None
    full_metrics: Optional[dict] = None


class RunListResponse(BaseModel):
    runs: list[RunResponse]
    total: int


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


class TradeResponse(BaseModel):
    trade_id: str
    run_id: str
    symbol: str
    direction: str
    entry_date: Optional[date] = None
    exit_date: Optional[date] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_reason: str = ""
    exit_reason: str = ""
    option_type: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[date] = None


# ---------------------------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------------------------


class EquityCurvePointResponse(BaseModel):
    date: date
    strategy_value: float
    benchmark_value: float


class EquityCurveResponse(BaseModel):
    points: list[EquityCurvePointResponse]


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


class RegimeStatResponse(BaseModel):
    regime: str
    trades: int
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Options Analytics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class StrategyFileResponse(BaseModel):
    name: str
    content: str


class StrategyCreateRequest(BaseModel):
    name: str
    content: str


class StrategyUpdateRequest(BaseModel):
    content: str


# ---------------------------------------------------------------------------
# Backtest Launcher
# ---------------------------------------------------------------------------


class BacktestRunRequest(BaseModel):
    strategy: str
    universe: str
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    benchmark: str = "SPY"
    position_size_pct: float = 0.06


class BacktestRunResponse(BaseModel):
    run_id: str
    status: str


class BacktestStatusResponse(BaseModel):
    run_id: str
    status: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data Manager
# ---------------------------------------------------------------------------


class SymbolInfoResponse(BaseModel):
    symbol: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    market_cap: Optional[float] = None
    bar_count: int = 0
    min_date: Optional[date] = None
    max_date: Optional[date] = None
    updated_at: Optional[datetime] = None


class SymbolListResponse(BaseModel):
    symbols: list[SymbolInfoResponse]
    total_symbols: int
    total_bars: int


class DataFetchRequest(BaseModel):
    symbol: str
    start_date: date
    end_date: date


class DataFetchResponse(BaseModel):
    symbol: str
    bars_fetched: int
    message: str


class DataValidateRequest(BaseModel):
    symbols: list[str]


class DataValidateResponse(BaseModel):
    results: dict[str, str]


class DataQualityLogEntry(BaseModel):
    symbol: str
    date: date
    issue_type: str
    severity: Optional[str] = None
    details: Optional[str] = None
    resolved: bool = False


class DataQualityResponse(BaseModel):
    entries: list[DataQualityLogEntry]
    total: int


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------


class WalkForwardRunRequest(BaseModel):
    strategy: str
    universe: str  # comma-separated
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    benchmark: str = "SPY"
    train_months: int = 24
    oos_months: int = 6
    step_months: int = 3
    objective: str = "sharpe"
    monte_carlo_modes: list[str] = []
    simulations: int = 1000


class WalkForwardRunResponse(BaseModel):
    study_id: str
    status: str


class WalkForwardStatusResponse(BaseModel):
    study_id: str
    status: str
    windows_completed: int = 0
    windows_total: int = 0
    current_phase: str = "initializing"


class WalkForwardStudyResponse(BaseModel):
    study_id: str
    strategy_name: str
    config: Optional[dict] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    initial_capital: float = 100_000.0
    train_months: int = 24
    oos_months: int = 6
    step_months: int = 3
    objective: str = "sharpe"
    status: str = "running"
    created_at: Optional[datetime] = None
    windows: list = []
    aggregate: dict = {}
    stitched_equity_curve: list = []
    parameter_stability: dict = {}
    monte_carlo: dict = {}


class WalkForwardListResponse(BaseModel):
    studies: list[WalkForwardStudyResponse]
    total: int


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
    strategy: str = "orb"
    instrument: str = "ES"
    state_machine_enabled: bool = True
    account_tier: str = "50k"
    num_attempts: int = 1000
    seed: int = 42

class CampaignRunResponse(BaseModel):
    campaign_id: str
    status: str
