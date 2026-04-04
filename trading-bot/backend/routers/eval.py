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
from topstep.campaign_runner import CampaignRunner, FundedCampaignResult
from topstep.config import FUNDED_TIERS, TIERS
from topstep.strategies.orb_strategy import ORBStrategy
from topstep.strategies.vwap_reversion_strategy import VWAPReversionStrategy

import os
from pathlib import Path

router = APIRouter(prefix="/api/eval", tags=["eval"])

_STRATEGIES = {"orb": ORBStrategy, "vwap": VWAPReversionStrategy}
_running: dict[str, threading.Thread] = {}


def _record_to_response(rec: EvalCampaignRecord) -> CampaignResponse:
    """Convert an EvalCampaignRecord dataclass to a CampaignResponse schema."""
    return CampaignResponse(
        campaign_id=rec.campaign_id,
        strategy_name=rec.strategy_name,
        instrument=rec.instrument,
        timeframe=rec.timeframe,
        state_machine=rec.state_machine,
        num_attempts=rec.num_attempts,
        pass_rate=rec.pass_rate,
        ev_per_attempt=rec.ev_per_attempt,
        cost_to_funded=rec.cost_to_funded,
        avg_days_to_pass=rec.avg_days_to_pass,
        annual_ev=rec.annual_ev,
        created_at=rec.created_at,
        mode=rec.mode,
        survival_rate=rec.survival_rate,
        avg_monthly_pnl=rec.avg_monthly_pnl,
        sharpe_ratio=rec.sharpe_ratio,
        max_drawdown_median=rec.max_drawdown_median,
        avg_monthly_withdrawal=rec.avg_monthly_withdrawal,
        annual_expected_income=rec.annual_expected_income,
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
        timeframe=rec.timeframe,
        state_machine=rec.state_machine,
        num_attempts=rec.num_attempts,
        pass_rate=rec.pass_rate,
        ev_per_attempt=rec.ev_per_attempt,
        cost_to_funded=rec.cost_to_funded,
        avg_days_to_pass=rec.avg_days_to_pass,
        annual_ev=rec.annual_ev,
        created_at=rec.created_at,
        full_results=full_results,
        mode=rec.mode,
        survival_rate=rec.survival_rate,
        avg_monthly_pnl=rec.avg_monthly_pnl,
        sharpe_ratio=rec.sharpe_ratio,
        max_drawdown_median=rec.max_drawdown_median,
        avg_monthly_withdrawal=rec.avg_monthly_withdrawal,
        annual_expected_income=rec.annual_expected_income,
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
    timeframe: str,
    state_machine_enabled: bool,
    account_tier: str,
    num_attempts: int,
    seed: int,
    config_dict: dict,
    mode: str = "eval",
) -> None:
    """Background thread: run the campaign and persist results."""
    import logging
    import os
    from data.storage.database import Database

    logger = logging.getLogger(__name__)

    # Fresh DB connection for this thread — SQLite connections aren't thread-safe
    db_path = os.environ.get("TRADING_BOT_DB", "db/trading_bot.db")
    db = Database(db_path)
    db.create_tables()

    strategy_class = _STRATEGIES[strategy_key]
    if mode == "funded":
        config = FUNDED_TIERS.get(account_tier, FUNDED_TIERS["50k"])
    else:
        config = TIERS.get(account_tier, TIERS["50k"])

    runner = CampaignRunner(
        strategy_class=strategy_class,
        instrument=instrument,
        config=config,
        database=db,
        state_machine_enabled=state_machine_enabled,
        num_attempts=num_attempts,
        seed=seed,
        timeframe=timeframe,
    )

    try:
        result = runner.run()
    except Exception:
        logger.exception("Campaign %s failed", campaign_id)
        _running.pop(campaign_id, None)
        return

    if isinstance(result, FundedCampaignResult):
        record = EvalCampaignRecord(
            campaign_id=campaign_id,
            strategy_name=result.strategy_name,
            instrument=result.instrument,
            state_machine=result.state_machine_enabled,
            topstep_config=json.dumps(config_dict),
            timeframe=result.timeframe,
            num_attempts=result.num_attempts,
            seed=result.seed,
            mode="funded",
            # Eval-specific fields zeroed out for funded
            pass_rate=0.0,
            ev_per_attempt=0.0,
            cost_to_funded=0.0,
            avg_days_to_pass=0.0,
            annual_ev=0.0,
            # Funded-specific fields
            survival_rate=result.survival_rate,
            avg_monthly_pnl=result.avg_monthly_pnl,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown_median=result.max_drawdown_median,
            avg_monthly_withdrawal=result.avg_monthly_withdrawal,
            annual_expected_income=result.annual_expected_income,
            created_at=datetime.now(tz=timezone.utc),
            full_results=json.dumps({
                "attempt_outcomes": result.attempt_outcomes,
                "monthly_pnl_distribution": result.monthly_pnl_distribution,
                "survival_curve": result.survival_curve,
            }),
        )
    else:
        record = EvalCampaignRecord(
            campaign_id=campaign_id,
            strategy_name=result.strategy_name,
            instrument=result.instrument,
            state_machine=result.state_machine_enabled,
            topstep_config=json.dumps(config_dict),
            timeframe=result.timeframe,
            num_attempts=result.num_attempts,
            seed=result.seed,
            mode="eval",
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
    mode = req.mode.lower()
    if mode not in ("eval", "funded"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode {req.mode!r}. Valid: ['eval', 'funded']",
        )
    tier_map = FUNDED_TIERS if mode == "funded" else TIERS
    if req.account_tier not in tier_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown account tier {req.account_tier!r}. Valid: {list(tier_map)}",
        )

    campaign_id = str(uuid.uuid4())
    config = tier_map[req.account_tier]
    config_dict = {
        k: (v.name if hasattr(v, "name") else v)
        for k, v in config.__dict__.items()
    }

    thread = threading.Thread(
        target=_run,
        kwargs={
            "campaign_id": campaign_id,
            "strategy_key": strategy_key,
            "instrument": req.instrument,
            "timeframe": req.timeframe,
            "state_machine_enabled": req.state_machine_enabled,
            "account_tier": req.account_tier,
            "num_attempts": req.num_attempts,
            "seed": req.seed,
            "config_dict": config_dict,
            "mode": mode,
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


# ---------------------------------------------------------------------------
# Strategy source code viewer
# ---------------------------------------------------------------------------

_STRATEGY_FILES = {
    "orb_strategy": "topstep/strategies/orb_strategy.py",
    "vwap_reversion_strategy": "topstep/strategies/vwap_reversion_strategy.py",
    "state_manager": "topstep/state_manager.py",
    "config": "topstep/config.py",
    "evaluation_rules": "topstep/evaluation_rules.py",
    "simulator": "topstep/simulator.py",
    "campaign_runner": "topstep/campaign_runner.py",
}


@router.get("/strategies")
def list_strategy_files() -> list[dict]:
    """List available strategy source files."""
    result = []
    for key, filepath in _STRATEGY_FILES.items():
        full = Path(filepath)
        result.append({
            "key": key,
            "filename": full.name,
            "path": filepath,
            "exists": full.exists(),
        })
    return result


@router.get("/strategies/{key}")
def get_strategy_source(key: str) -> dict:
    """Return the source code of a strategy file."""
    filepath = _STRATEGY_FILES.get(key)
    if filepath is None:
        raise HTTPException(status_code=404, detail=f"Unknown strategy key: {key}")
    full = Path(filepath)
    if not full.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    return {
        "key": key,
        "filename": full.name,
        "path": filepath,
        "source": full.read_text(),
    }
