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
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
