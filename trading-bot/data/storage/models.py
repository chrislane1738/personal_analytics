"""Data model dataclasses for the trading bot persistence layer."""
from dataclasses import dataclass, field
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
    created_at: Optional[datetime] = None
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
