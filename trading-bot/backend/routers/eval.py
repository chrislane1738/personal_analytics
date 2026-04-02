"""Eval campaign endpoints: list, get, delete, launch."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone

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
    """Convert an EvalCampaignRecord dataclass to a CampaignResponse schema."""
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
    sort: str = Query("created_at", description="Column to sort by"),
    order: str = Query("desc", description="Sort direction: asc or desc"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> CampaignListResponse:
    """List eval campaigns with sorting and pagination."""
    db = get_database()
    try:
        records = db.list_eval_campaigns(
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    total = db.execute("SELECT COUNT(*) FROM eval_campaigns").fetchone()[0]

    return CampaignListResponse(
        campaigns=[_record_to_response(r) for r in records],
        total=total,
    )


@router.get("/campaigns/{campaign_id}", response_model=CampaignDetailResponse)
def get_campaign(campaign_id: str) -> CampaignDetailResponse:
    """Get a single campaign by ID, with full_results parsed from JSON."""
    db = get_database()
    rec = db.get_eval_campaign(campaign_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")

    full_results = None
    if rec.full_results:
        try:
            full_results = json.loads(rec.full_results)
        except (json.JSONDecodeError, TypeError):
            full_results = None

    return CampaignDetailResponse(
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
        full_results=full_results,
    )


@router.delete("/campaigns/{campaign_id}")
def delete_campaign(campaign_id: str) -> dict:
    """Delete a campaign, 404 if not found."""
    db = get_database()
    rec = db.get_eval_campaign(campaign_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    db.delete_eval_campaign(campaign_id)
    return {"deleted": campaign_id}


def _run(
    campaign_id: str,
    strategy_key: str,
    instrument: str,
    state_machine_enabled: bool,
    account_tier: str,
    num_attempts: int,
    seed: int,
    config_dict: dict,
) -> None:
    """Background thread: run the campaign and persist results."""
    # Use a fresh DB connection — SQLite connections aren't thread-safe
    db = get_database()

    strategy_class = _STRATEGIES[strategy_key]
    config = TIERS.get(account_tier, TIERS["50k"])

    runner = CampaignRunner(
        strategy_class=strategy_class,
        instrument=instrument,
        config=config,
        database=db,
        state_machine_enabled=state_machine_enabled,
        num_attempts=num_attempts,
        seed=seed,
    )

    try:
        result = runner.run()
    except Exception:
        _running.pop(campaign_id, None)
        return

    record = EvalCampaignRecord(
        campaign_id=campaign_id,
        strategy_name=result.strategy_name,
        instrument=result.instrument,
        state_machine=result.state_machine_enabled,
        topstep_config=json.dumps(config_dict),
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


@router.post("/campaigns/run", response_model=CampaignRunResponse)
def run_campaign(req: CampaignRunRequest) -> CampaignRunResponse:
    """Launch a campaign in a background thread, return campaign_id + 'running'."""
    strategy_key = req.strategy.lower()
    if strategy_key not in _STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy {req.strategy!r}. Valid: {list(_STRATEGIES)}",
        )
    if req.account_tier not in TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown account tier {req.account_tier!r}. Valid: {list(TIERS)}",
        )

    campaign_id = str(uuid.uuid4())
    config = TIERS[req.account_tier]
    config_dict = config.__dict__.copy()

    thread = threading.Thread(
        target=_run,
        kwargs={
            "campaign_id": campaign_id,
            "strategy_key": strategy_key,
            "instrument": req.instrument,
            "state_machine_enabled": req.state_machine_enabled,
            "account_tier": req.account_tier,
            "num_attempts": req.num_attempts,
            "seed": req.seed,
            "config_dict": config_dict,
        },
        daemon=True,
    )
    _running[campaign_id] = thread
    thread.start()

    return CampaignRunResponse(campaign_id=campaign_id, status="running")


@router.get("/campaigns/{campaign_id}/status")
def campaign_status(campaign_id: str) -> dict:
    """Check if a campaign is running, completed, or unknown."""
    if campaign_id in _running and _running[campaign_id].is_alive():
        return {"campaign_id": campaign_id, "status": "running"}

    db = get_database()
    rec = db.get_eval_campaign(campaign_id)
    if rec is not None:
        return {"campaign_id": campaign_id, "status": "completed"}

    return {"campaign_id": campaign_id, "status": "unknown"}
