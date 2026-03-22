"""Run management endpoints: list, get, delete."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.dependencies import get_database
from backend.schemas import RunListResponse, RunResponse
from data.storage.database import Database

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _run_to_response(run, include_full_metrics: bool = False) -> RunResponse:
    """Convert a RunRecord dataclass to a RunResponse schema."""
    full_metrics = None
    if include_full_metrics and run.full_metrics:
        try:
            full_metrics = json.loads(run.full_metrics)
        except (json.JSONDecodeError, TypeError):
            full_metrics = None
    return RunResponse(
        run_id=run.run_id,
        mode=run.mode,
        strategy_name=run.strategy_name,
        config=run.config,
        start_date=run.start_date,
        end_date=run.end_date,
        initial_capital=run.initial_capital,
        final_value=run.final_value,
        total_return=run.total_return,
        sharpe=run.sharpe,
        max_drawdown=run.max_drawdown,
        created_at=run.created_at,
        full_metrics=full_metrics,
    )


@router.get("", response_model=RunListResponse)
def list_runs(
    sort: str = Query("created_at", description="Column to sort by"),
    order: str = Query("desc", description="Sort direction: asc or desc"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Database = Depends(get_database),
) -> RunListResponse:
    """List backtest runs with sorting, filtering, and pagination."""
    try:
        runs = db.list_runs(
            sort=sort,
            order=order,
            strategy_filter=strategy,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Get total count via lightweight SQL count
    count_sql = "SELECT COUNT(*) FROM runs"
    count_params: list = []
    if strategy is not None:
        count_sql += " WHERE strategy_name = ?"
        count_params.append(strategy)
    total = db.execute(count_sql, tuple(count_params)).fetchone()[0]

    return RunListResponse(
        runs=[_run_to_response(r) for r in runs],
        total=total,
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    run_id: str,
    db: Database = Depends(get_database),
) -> RunResponse:
    """Get a single run by ID, with full_metrics parsed from JSON."""
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _run_to_response(run, include_full_metrics=True)


@router.delete("/{run_id}")
def delete_run(
    run_id: str,
    db: Database = Depends(get_database),
) -> dict:
    """Delete a run and all associated trades and equity curve data."""
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    db.delete_run(run_id)
    return {"deleted": run_id}
