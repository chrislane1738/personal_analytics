"""Strategy YAML file CRUD endpoints."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    StrategyCreateRequest,
    StrategyFileResponse,
    StrategyUpdateRequest,
)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# Resolve strategy directory relative to project root
_STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "config" / "strategies"


def _validate_name(name: str) -> str:
    """Validate strategy name: no path traversal, must end with .yaml."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid strategy name")
    # Strip .yaml if present, then re-add for consistency
    stem = name.removesuffix(".yaml").removesuffix(".yml")
    if not re.match(r"^[a-zA-Z0-9_\-]+$", stem):
        raise HTTPException(
            status_code=400,
            detail="Strategy name must contain only letters, numbers, hyphens, and underscores",
        )
    return f"{stem}.yaml"


@router.get("", response_model=list[StrategyFileResponse])
def list_strategies() -> list[StrategyFileResponse]:
    """List all YAML strategy files."""
    _STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    results: list[StrategyFileResponse] = []
    for path in sorted(_STRATEGIES_DIR.glob("*.yaml")):
        results.append(
            StrategyFileResponse(name=path.name, content=path.read_text())
        )
    for path in sorted(_STRATEGIES_DIR.glob("*.yml")):
        results.append(
            StrategyFileResponse(name=path.name, content=path.read_text())
        )
    return results


@router.get("/{name}", response_model=StrategyFileResponse)
def get_strategy(name: str) -> StrategyFileResponse:
    """Get a single strategy file by name."""
    filename = _validate_name(name)
    path = _STRATEGIES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Strategy '{filename}' not found")
    return StrategyFileResponse(name=path.name, content=path.read_text())


@router.put("/{name}", response_model=StrategyFileResponse)
def update_strategy(name: str, body: StrategyUpdateRequest) -> StrategyFileResponse:
    """Update an existing strategy file."""
    filename = _validate_name(name)
    path = _STRATEGIES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Strategy '{filename}' not found")
    path.write_text(body.content)
    return StrategyFileResponse(name=path.name, content=body.content)


@router.post("", response_model=StrategyFileResponse, status_code=201)
def create_strategy(body: StrategyCreateRequest) -> StrategyFileResponse:
    """Create a new strategy file."""
    filename = _validate_name(body.name)
    _STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    path = _STRATEGIES_DIR / filename
    if path.exists():
        raise HTTPException(
            status_code=409, detail=f"Strategy '{filename}' already exists"
        )
    path.write_text(body.content)
    return StrategyFileResponse(name=path.name, content=body.content)


@router.delete("/{name}")
def delete_strategy(name: str) -> dict:
    """Delete a strategy file."""
    filename = _validate_name(name)
    path = _STRATEGIES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Strategy '{filename}' not found")
    path.unlink()
    return {"deleted": filename}
