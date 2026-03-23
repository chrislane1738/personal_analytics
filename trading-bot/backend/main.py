"""FastAPI application for the trading bot dashboard."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import analytics, backtest, data, runs, strategies, trades, walk_forward
from backend.services.ws_manager import ws_manager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading Bot Dashboard API",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the Next.js dev server
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3004"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(runs.router)
app.include_router(trades.router)
app.include_router(analytics.router)
app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(data.router)
app.include_router(walk_forward.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WebSocket: Live event monitor
# ---------------------------------------------------------------------------

async def _ws_ping_loop(websocket: WebSocket, interval: float = 15.0) -> None:
    """Send periodic ping frames to keep the connection alive."""
    try:
        while True:
            await asyncio.sleep(interval)
            await websocket.send_json({"type": "ping"})
    except Exception:
        pass  # Connection closed — task will be cancelled


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket) -> None:
    """Live event streaming endpoint.

    Protocol:
        Client sends: {"action": "subscribe", "run_id": "..."}
        Client sends: {"action": "stop", "run_id": "..."}
        Server sends: {"type": "<event_type>", "data": {...}}
        Server sends: {"type": "ping"} every 15 s
        Server sends: {"type": "subscribed", "run_id": "..."}
        Server sends: {"type": "error", "data": {"message": "..."}}
    """
    await websocket.accept()
    subscribed_run_id: str | None = None
    ping_task: asyncio.Task | None = None

    try:
        # Start keepalive ping loop
        ping_task = asyncio.create_task(_ws_ping_loop(websocket))

        while True:
            raw = await websocket.receive_json()
            action = raw.get("action")
            run_id = raw.get("run_id", "")

            if action == "subscribe":
                # Unsubscribe from previous run if any
                if subscribed_run_id is not None:
                    ws_manager.disconnect(websocket, subscribed_run_id)

                subscribed_run_id = run_id
                await ws_manager.connect(websocket, run_id)
                await ws_manager.send_personal(websocket, {
                    "type": "subscribed",
                    "run_id": run_id,
                })
                logger.info("Client subscribed to run_id=%s", run_id)

            elif action == "stop":
                if run_id:
                    ws_manager.request_stop(run_id)
                    await ws_manager.send_personal(websocket, {
                        "type": "stop_requested",
                        "run_id": run_id,
                    })
                    logger.info("Stop requested for run_id=%s", run_id)

            elif action == "pong":
                # Client pong response — no action needed
                pass

            else:
                await ws_manager.send_personal(websocket, {
                    "type": "error",
                    "data": {"message": f"Unknown action: {action}"},
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (run_id=%s)", subscribed_run_id)
    except Exception:
        logger.exception("WebSocket error (run_id=%s)", subscribed_run_id)
    finally:
        if ping_task is not None:
            ping_task.cancel()
        if subscribed_run_id is not None:
            ws_manager.disconnect(websocket, subscribed_run_id)
