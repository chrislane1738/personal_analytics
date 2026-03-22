"""Analytics endpoints: equity curve, regime, monte carlo, options."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query

from analytics.monte_carlo import run_monte_carlo
from analytics.options_analytics import compute_options_analytics
from analytics.regime import compute_regime_stats, detect_regimes
from backend.dependencies import get_database
from backend.schemas import (
    EquityCurveResponse,
    EquityCurvePointResponse,
    MonteCarloResponse,
    OptionsAnalyticsResponse,
    RegimeStatResponse,
)
from data.storage.database import Database

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _get_run_or_404(db: Database, run_id: str):
    """Return the run record or raise 404."""
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


# ---------------------------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------------------------


@router.get("/{run_id}/equity-curve", response_model=EquityCurveResponse)
def get_equity_curve(
    run_id: str,
    db: Database = Depends(get_database),
) -> EquityCurveResponse:
    """Return the equity curve for a run."""
    _get_run_or_404(db, run_id)
    points = db.get_equity_curve(run_id)
    return EquityCurveResponse(
        points=[
            EquityCurvePointResponse(
                date=p.date,
                strategy_value=p.strategy_value,
                benchmark_value=p.benchmark_value,
            )
            for p in points
        ]
    )


# ---------------------------------------------------------------------------
# Regime Analysis
# ---------------------------------------------------------------------------


@router.get("/{run_id}/regime", response_model=list[RegimeStatResponse])
def get_regime_stats(
    run_id: str,
    db: Database = Depends(get_database),
) -> list[RegimeStatResponse]:
    """Detect market regimes and compute per-regime trade statistics."""
    run = _get_run_or_404(db, run_id)
    trades = db.get_trades(run_id)
    if not trades:
        return []

    # Parse benchmark_symbol from run config
    benchmark_symbol = "SPY"  # default
    if run.config:
        try:
            config = json.loads(run.config)
            benchmark_symbol = config.get("benchmark_symbol", "SPY")
        except (json.JSONDecodeError, TypeError):
            pass

    # Fetch benchmark bars for the run's date range
    if run.start_date and run.end_date:
        bars = db.get_daily_bars(benchmark_symbol, run.start_date, run.end_date)
    else:
        bars = []

    if not bars:
        return []

    # Build price list for detect_regimes
    prices = [(b.date, b.close) for b in bars]
    regimes = detect_regimes(prices)

    # Compute per-regime stats
    stats = compute_regime_stats(trades, regimes)

    return [
        RegimeStatResponse(
            regime=regime_name,
            trades=data.get("trades", 0),
            win_rate=data.get("win_rate", 0.0),
            avg_pnl=data.get("avg_pnl", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            best_trade=data.get("best_trade", 0.0),
            worst_trade=data.get("worst_trade", 0.0),
        )
        for regime_name, data in stats.items()
    ]


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


@router.get("/{run_id}/monte-carlo", response_model=MonteCarloResponse)
def get_monte_carlo(
    run_id: str,
    simulations: int = Query(10000, ge=100, le=100000),
    db: Database = Depends(get_database),
) -> MonteCarloResponse:
    """Run Monte Carlo simulation on trade P&Ls."""
    run = _get_run_or_404(db, run_id)
    trades = db.get_trades(run_id)
    if not trades:
        raise HTTPException(
            status_code=400,
            detail="No trades found for this run; cannot run Monte Carlo.",
        )

    trade_pnls = [t.pnl for t in trades]
    initial_capital = run.initial_capital or 100_000.0

    result = run_monte_carlo(
        trade_pnls=trade_pnls,
        initial_capital=initial_capital,
        num_simulations=simulations,
        seed=42,
    )

    return MonteCarloResponse(
        simulations=result.simulations,
        median_final_equity=result.median_final_equity,
        percentile_5=result.percentile_5,
        percentile_95=result.percentile_95,
        actual_final_equity=result.actual_final_equity,
        probability_of_ruin=result.probability_of_ruin,
        max_drawdown_median=result.max_drawdown_median,
        max_drawdown_95=result.max_drawdown_95,
        is_outlier=result.is_outlier,
        percentile_bands=result.percentile_bands,
        actual_curve=result.actual_curve,
        drawdown_distribution=result.drawdown_distribution,
    )


# ---------------------------------------------------------------------------
# Options Analytics
# ---------------------------------------------------------------------------


@router.get("/{run_id}/options", response_model=OptionsAnalyticsResponse)
def get_options_analytics(
    run_id: str,
    db: Database = Depends(get_database),
) -> OptionsAnalyticsResponse:
    """Compute options-specific analytics for a run."""
    _get_run_or_404(db, run_id)
    trades = db.get_trades(run_id)

    # Filter to options trades only
    options_trades = [t for t in trades if t.option_type is not None]
    if not options_trades:
        raise HTTPException(
            status_code=400,
            detail="No options trades found for this run.",
        )

    result = compute_options_analytics(trades)

    return OptionsAnalyticsResponse(
        total_premium_collected=result.total_premium_collected,
        total_premium_paid=result.total_premium_paid,
        net_premium=result.net_premium,
        assignment_count=result.assignment_count,
        total_short_options=result.total_short_options,
        assignment_rate=result.assignment_rate,
        win_rate_by_dte=result.win_rate_by_dte,
        avg_pnl_by_dte=result.avg_pnl_by_dte,
        greeks_timeline=result.greeks_timeline,
    )
