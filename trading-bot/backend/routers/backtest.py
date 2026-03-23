"""Backtest launcher endpoints: run, status, stop."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from data.storage.database import Database
from backend.dependencies import get_database
from backend.schemas import (
    BacktestRunRequest,
    BacktestRunResponse,
    BacktestStatusResponse,
)
from backend.services.backtest_service import get_backtest_runner
from data.storage.database import Database

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestRunResponse, status_code=202)
def launch_backtest(
    body: BacktestRunRequest,
    db: Database = Depends(get_database),
) -> BacktestRunResponse:
    """Start a new backtest in a background thread."""
    universe = [s.strip().upper() for s in body.universe.split(",") if s.strip()]
    if not universe:
        raise HTTPException(status_code=400, detail="Universe must not be empty")

    runner = get_backtest_runner()
    run_id = runner.start_run(
        strategy_name=body.strategy,
        universe=universe,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_capital=body.initial_capital,
        benchmark=body.benchmark,
        position_size_pct=body.position_size_pct,
        database=db,
    )
    return BacktestRunResponse(run_id=run_id, status="running")


@router.get("/status/{run_id}", response_model=BacktestStatusResponse)
def backtest_status(
    run_id: str,
    db: Database = Depends(get_database),
) -> BacktestStatusResponse:
    """Get the current status of a backtest run."""
    runner = get_backtest_runner()
    info = runner.get_status(run_id)
    if info is not None:
        return BacktestStatusResponse(
            run_id=info.run_id,
            status=info.status.value,
            error=info.error,
        )

    # Fallback: the run_id may have changed after engine completion.
    # Check if a recently completed run exists in the DB.
    runs = db.list_runs(sort="created_at", order="desc", limit=1)
    if runs:
        return BacktestStatusResponse(
            run_id=runs[0].run_id,
            status="completed",
            error=None,
        )

    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


@router.post("/stop/{run_id}")
def stop_backtest(run_id: str) -> dict:
    """Cancel a running backtest."""
    runner = get_backtest_runner()
    found = runner.stop_run(run_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"run_id": run_id, "status": "cancelled"}
