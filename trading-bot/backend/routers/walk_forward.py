"""Walk-forward study endpoints: run, list, status, stop, delete."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_database
from backend.schemas import (
    WalkForwardListResponse,
    WalkForwardRunRequest,
    WalkForwardRunResponse,
    WalkForwardStatusResponse,
    WalkForwardStudyResponse,
)
from backend.services.walk_forward_service import get_walk_forward_runner
from data.storage.database import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/walk-forward", tags=["walk-forward"])

# Resolve strategy directory relative to project root
_STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "config" / "strategies"


def _load_strategy_config(strategy_name: str) -> dict:
    """Load a strategy YAML file and return the parsed config dict."""
    yaml_path = _STRATEGIES_DIR / strategy_name
    if not yaml_path.suffix:
        yaml_path = _STRATEGIES_DIR / f"{strategy_name}.yaml"

    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    raise ValueError(
        f"Could not find strategy config '{strategy_name}'. "
        f"Provide a YAML filename from config/strategies/."
    )


@router.post("/run", response_model=WalkForwardRunResponse, status_code=202)
def launch_walk_forward(
    body: WalkForwardRunRequest,
    db: Database = Depends(get_database),
) -> WalkForwardRunResponse:
    """Start a new walk-forward study in a background thread."""
    universe = [s.strip().upper() for s in body.universe.split(",") if s.strip()]
    if not universe:
        raise HTTPException(status_code=400, detail="Universe must not be empty")

    # Load strategy config
    try:
        config_dict = _load_strategy_config(body.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Extract param_space from the strategy config's optimization block
    param_space: dict = {}
    optimization = config_dict.get("optimization", {})
    if isinstance(optimization, dict):
        param_space = optimization.get("parameters", {})

    runner = get_walk_forward_runner()
    study_id = runner.start_study(
        strategy_name=body.strategy,
        config_dict=config_dict,
        param_space=param_space,
        universe=universe,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_capital=body.initial_capital,
        benchmark=body.benchmark,
        train_months=body.train_months,
        oos_months=body.oos_months,
        step_months=body.step_months,
        objective=body.objective,
        monte_carlo_modes=body.monte_carlo_modes,
        simulations=body.simulations,
        database=db,
    )
    return WalkForwardRunResponse(study_id=study_id, status="running")


@router.get("", response_model=WalkForwardListResponse)
def list_walk_forward_studies(
    sort: str = "created_at",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
    db: Database = Depends(get_database),
) -> WalkForwardListResponse:
    """List walk-forward studies with pagination."""
    try:
        studies = db.list_walk_forward_studies(
            sort=sort, order=order, limit=limit, offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    total = db.count_walk_forward_studies()

    study_responses = []
    for s in studies:
        # Parse config JSON for display
        try:
            config_parsed = json.loads(s.config) if s.config else {}
        except (json.JSONDecodeError, TypeError):
            config_parsed = {}

        # Parse results JSON for summary fields
        results_parsed = {}
        if s.results:
            try:
                results_parsed = json.loads(s.results)
            except (json.JSONDecodeError, TypeError):
                results_parsed = {}

        # Parse monte_carlo JSON
        mc_parsed = {}
        if s.monte_carlo:
            try:
                mc_parsed = json.loads(s.monte_carlo)
            except (json.JSONDecodeError, TypeError):
                mc_parsed = {}

        study_responses.append(
            WalkForwardStudyResponse(
                study_id=s.study_id,
                strategy_name=s.strategy_name,
                config=config_parsed,
                start_date=s.start_date,
                end_date=s.end_date,
                initial_capital=s.initial_capital,
                train_months=s.train_months,
                oos_months=s.oos_months,
                step_months=s.step_months,
                objective=s.objective,
                status=s.status,
                created_at=s.created_at,
                windows=results_parsed.get("windows", []),
                aggregate=results_parsed.get("aggregate", {}),
                stitched_equity_curve=results_parsed.get("stitched_equity_curve", []),
                parameter_stability=results_parsed.get("parameter_stability", {}),
                monte_carlo=mc_parsed,
            )
        )

    return WalkForwardListResponse(studies=study_responses, total=total)


@router.get("/{study_id}", response_model=WalkForwardStudyResponse)
def get_walk_forward_study(
    study_id: str,
    db: Database = Depends(get_database),
) -> WalkForwardStudyResponse:
    """Get full walk-forward study with parsed results."""
    study = db.get_walk_forward_study(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail=f"Study {study_id} not found")

    # Parse config JSON
    try:
        config_parsed = json.loads(study.config) if study.config else {}
    except (json.JSONDecodeError, TypeError):
        config_parsed = {}

    # Parse results JSON
    results_parsed = {}
    if study.results:
        try:
            results_parsed = json.loads(study.results)
        except (json.JSONDecodeError, TypeError):
            results_parsed = {}

    # Parse monte_carlo JSON
    mc_parsed = {}
    if study.monte_carlo:
        try:
            mc_parsed = json.loads(study.monte_carlo)
        except (json.JSONDecodeError, TypeError):
            mc_parsed = {}

    return WalkForwardStudyResponse(
        study_id=study.study_id,
        strategy_name=study.strategy_name,
        config=config_parsed,
        start_date=study.start_date,
        end_date=study.end_date,
        initial_capital=study.initial_capital,
        train_months=study.train_months,
        oos_months=study.oos_months,
        step_months=study.step_months,
        objective=study.objective,
        status=study.status,
        created_at=study.created_at,
        windows=results_parsed.get("windows", []),
        aggregate=results_parsed.get("aggregate", {}),
        stitched_equity_curve=results_parsed.get("stitched_equity_curve", []),
        parameter_stability=results_parsed.get("parameter_stability", {}),
        monte_carlo=mc_parsed,
    )


@router.get("/{study_id}/status", response_model=WalkForwardStatusResponse)
def walk_forward_status(
    study_id: str,
    db: Database = Depends(get_database),
) -> WalkForwardStatusResponse:
    """Get the current status of a walk-forward study."""
    runner = get_walk_forward_runner()
    info = runner.get_status(study_id)
    if info is not None:
        return WalkForwardStatusResponse(
            study_id=info.study_id,
            status=info.status.value,
            windows_completed=info.windows_completed,
            windows_total=info.windows_total,
            current_phase=info.current_phase,
        )

    # Fallback: check the database
    study = db.get_walk_forward_study(study_id)
    if study is not None:
        return WalkForwardStatusResponse(
            study_id=study.study_id,
            status=study.status,
            windows_completed=0,
            windows_total=0,
            current_phase=study.status,
        )

    raise HTTPException(status_code=404, detail=f"Study {study_id} not found")


@router.post("/{study_id}/stop")
def stop_walk_forward(study_id: str) -> dict:
    """Cancel a running walk-forward study."""
    runner = get_walk_forward_runner()
    found = runner.stop_study(study_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Study {study_id} not found")
    return {"study_id": study_id, "status": "cancelled"}


@router.delete("/{study_id}")
def delete_walk_forward_study(
    study_id: str,
    db: Database = Depends(get_database),
) -> dict:
    """Delete a walk-forward study."""
    study = db.get_walk_forward_study(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail=f"Study {study_id} not found")
    db.delete_walk_forward_study(study_id)
    return {"study_id": study_id, "deleted": True}
