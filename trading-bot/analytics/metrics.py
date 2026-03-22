"""Core metrics engine for evaluating trading strategy performance."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class DrawdownInfo:
    max_drawdown_pct: float
    max_drawdown_dollar: float
    peak_date: Optional[date]
    trough_date: Optional[date]
    recovery_date: Optional[date]
    duration_days: int


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.04) -> float:
    """Annualized Sharpe ratio. returns = list of daily returns (e.g., 0.01 = 1%)."""
    if len(returns) < 2:
        return 0.0
    excess = np.array(returns) - risk_free_rate / 252
    std = float(np.std(excess, ddof=1))
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def sortino_ratio(returns: list[float], risk_free_rate: float = 0.04) -> float:
    """Like Sharpe but only penalizes downside volatility."""
    if len(returns) < 2:
        return 0.0
    excess = np.array(returns) - risk_free_rate / 252
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float(np.std(downside))
    if downside_std < 1e-10:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(252))


def max_drawdown(equity_curve: list[tuple[date, float]]) -> DrawdownInfo:
    """Compute max drawdown from equity curve [(date, value), ...].

    Tracks the running peak and computes drawdown at each point.
    Returns worst drawdown with corresponding dates and duration.
    """
    if not equity_curve:
        return DrawdownInfo(
            max_drawdown_pct=0.0,
            max_drawdown_dollar=0.0,
            peak_date=None,
            trough_date=None,
            recovery_date=None,
            duration_days=0,
        )

    dates = [entry[0] for entry in equity_curve]
    values = [entry[1] for entry in equity_curve]

    running_peak = values[0]
    running_peak_date = dates[0]

    best_peak_value = values[0]
    best_peak_date = dates[0]
    best_trough_value = values[0]
    best_trough_date = dates[0]
    max_dd_pct = 0.0
    max_dd_dollar = 0.0

    # Temporary tracking for current drawdown period
    cur_peak_value = values[0]
    cur_peak_date = dates[0]

    for i, (d, v) in enumerate(zip(dates, values)):
        if v >= running_peak:
            running_peak = v
            running_peak_date = d
            cur_peak_value = v
            cur_peak_date = d

        dd_dollar = cur_peak_value - v
        dd_pct = dd_dollar / cur_peak_value if cur_peak_value > 0 else 0.0

        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_dollar = dd_dollar
            best_peak_value = cur_peak_value
            best_peak_date = cur_peak_date
            best_trough_value = v
            best_trough_date = d

    # Find recovery date: first date after trough where value >= peak at drawdown
    recovery_date: Optional[date] = None
    trough_idx = next(
        (i for i, (d, v) in enumerate(zip(dates, values)) if d == best_trough_date),
        None,
    )
    if trough_idx is not None and max_dd_pct > 0:
        for d, v in zip(dates[trough_idx + 1:], values[trough_idx + 1:]):
            if v >= best_peak_value:
                recovery_date = d
                break

    # Duration: from peak to recovery (or end of series if no recovery)
    if max_dd_pct > 0:
        end_date = recovery_date if recovery_date is not None else dates[-1]
        duration_days = (end_date - best_peak_date).days
    else:
        duration_days = 0

    return DrawdownInfo(
        max_drawdown_pct=max_dd_pct,
        max_drawdown_dollar=max_dd_dollar,
        peak_date=best_peak_date if max_dd_pct > 0 else None,
        trough_date=best_trough_date if max_dd_pct > 0 else None,
        recovery_date=recovery_date,
        duration_days=duration_days,
    )


def calmar_ratio(cagr_val: float, max_dd_pct: float) -> float:
    """CAGR / abs(max drawdown). Returns 0 if no drawdown."""
    if max_dd_pct == 0:
        return 0.0
    return cagr_val / abs(max_dd_pct)


def cagr(start_value: float, end_value: float, years: float) -> float:
    """Compound Annual Growth Rate."""
    if start_value <= 0 or years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1


def profit_factor(trades: list[dict]) -> float:
    """Gross profits / gross losses. Each trade dict has 'pnl' key."""
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def win_rate(trades: list[dict]) -> float:
    """Percentage of winning trades."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t['pnl'] > 0)
    return winners / len(trades)


def expectancy(trades: list[dict]) -> float:
    """Average expected P&L per trade = avg_win * win_rate - avg_loss * loss_rate."""
    if not trades:
        return 0.0
    winners = [t['pnl'] for t in trades if t['pnl'] > 0]
    losers = [t['pnl'] for t in trades if t['pnl'] <= 0]
    avg_win = float(np.mean(winners)) if winners else 0.0
    avg_loss = abs(float(np.mean(losers))) if losers else 0.0
    wr = len(winners) / len(trades)
    return float(avg_win * wr - avg_loss * (1 - wr))


def beta(strategy_returns: list[float], benchmark_returns: list[float]) -> float:
    """Beta = covariance(strategy, benchmark) / variance(benchmark).

    Uses ddof=1 (sample) consistently for both covariance and variance so that
    strategy = 2 * benchmark produces beta = 2.0 regardless of series length.
    """
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    # np.cov defaults to ddof=1; use ddof=1 for variance to match
    cov = np.cov(strategy_returns, benchmark_returns)[0][1]
    var = float(np.var(benchmark_returns, ddof=1))
    if var < 1e-20:
        return 0.0
    return float(cov / var)


def value_at_risk(returns: list[float], confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level. Returns negative number (loss)."""
    if not returns:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def annualized_volatility(returns: list[float]) -> float:
    """Annualized standard deviation of daily returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns) * np.sqrt(252))


def _daily_returns_from_curve(equity_curve: list[tuple[date, float]]) -> list[float]:
    """Compute daily returns from equity curve."""
    if len(equity_curve) < 2:
        return []
    values = [v for _, v in equity_curve]
    return [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] != 0
    ]


def compute_all_metrics(
    equity_curve: list[tuple[date, float]],
    trades: list[dict],
    benchmark_curve: list[tuple[date, float]] = None,
    risk_free_rate: float = 0.04,
    initial_capital: float = 100_000.0,
) -> dict:
    """Compute all metrics and return as a dict.

    Parameters
    ----------
    equity_curve:
        List of (date, portfolio_value) tuples sorted chronologically.
    trades:
        List of trade dicts, each containing at least a 'pnl' key.
    benchmark_curve:
        Optional list of (date, value) tuples for benchmark comparison.
    risk_free_rate:
        Annual risk-free rate (default 4%).
    initial_capital:
        Starting portfolio value for reference.
    """
    returns = _daily_returns_from_curve(equity_curve)

    # Time span in years
    years = 0.0
    if len(equity_curve) >= 2:
        years = (equity_curve[-1][0] - equity_curve[0][0]).days / 365.25

    start_val = equity_curve[0][1] if equity_curve else initial_capital
    end_val = equity_curve[-1][1] if equity_curve else initial_capital

    cagr_val = cagr(start_val, end_val, years)
    dd_info = max_drawdown(equity_curve)

    result: dict = {
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown_pct": dd_info.max_drawdown_pct,
        "max_drawdown_dollar": dd_info.max_drawdown_dollar,
        "drawdown_peak_date": dd_info.peak_date,
        "drawdown_trough_date": dd_info.trough_date,
        "drawdown_recovery_date": dd_info.recovery_date,
        "drawdown_duration_days": dd_info.duration_days,
        "cagr": cagr_val,
        "calmar_ratio": calmar_ratio(cagr_val, dd_info.max_drawdown_pct),
        "annualized_volatility": annualized_volatility(returns),
        "profit_factor": profit_factor(trades),
        "win_rate": win_rate(trades),
        "expectancy": expectancy(trades),
        "total_trades": len(trades),
        "value_at_risk_95": value_at_risk(returns, 0.95),
        "total_return": (end_val - start_val) / start_val if start_val > 0 else 0.0,
        "start_value": start_val,
        "end_value": end_val,
        "years": years,
    }

    if benchmark_curve is not None:
        bench_returns = _daily_returns_from_curve(benchmark_curve)
        # Align lengths for beta calculation
        min_len = min(len(returns), len(bench_returns))
        if min_len >= 2:
            result["beta"] = beta(returns[-min_len:], bench_returns[-min_len:])
        else:
            result["beta"] = 0.0

        if bench_returns:
            bench_start = benchmark_curve[0][1]
            bench_end = benchmark_curve[-1][1]
            result["benchmark_total_return"] = (
                (bench_end - bench_start) / bench_start if bench_start > 0 else 0.0
            )
            result["alpha"] = result["total_return"] - result["benchmark_total_return"]
        else:
            result["benchmark_total_return"] = 0.0
            result["alpha"] = 0.0
    else:
        result["beta"] = None
        result["benchmark_total_return"] = None
        result["alpha"] = None

    return result
