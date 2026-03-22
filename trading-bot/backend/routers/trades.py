"""Trade listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.dependencies import get_database
from backend.schemas import TradeResponse
from data.storage.database import Database

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("", response_model=list[TradeResponse])
def list_trades(
    run_id: str = Query(..., description="Run ID to fetch trades for"),
    db: Database = Depends(get_database),
) -> list[TradeResponse]:
    """List all trades for a given run."""
    # Verify the run exists
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    trades = db.get_trades(run_id)
    return [
        TradeResponse(
            trade_id=t.trade_id,
            run_id=t.run_id,
            symbol=t.symbol,
            direction=t.direction,
            entry_date=t.entry_date,
            exit_date=t.exit_date,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            quantity=t.quantity,
            pnl=t.pnl,
            pnl_pct=t.pnl_pct,
            entry_reason=t.entry_reason,
            exit_reason=t.exit_reason,
            option_type=t.option_type,
            strike=t.strike,
            expiration=t.expiration,
        )
        for t in trades
    ]
