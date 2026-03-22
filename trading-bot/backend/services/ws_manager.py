"""WebSocket connection manager that bridges the EventBus to WebSocket clients.

Manages per-run WebSocket connections, broadcasts backtest events as JSON,
and supports stop-run signalling from clients.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manage WebSocket connections grouped by run_id.

    Each backtest run can have multiple WebSocket clients subscribed.
    The manager broadcasts events to all clients watching a given run.
    """

    def __init__(self) -> None:
        # run_id -> list of active WebSocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}
        # run_id -> asyncio.Event that signals "stop requested"
        self._stop_events: dict[str, asyncio.Event] = {}
        # Callbacks that the backtest_service can register to be notified
        # when a client requests a stop.  Keyed by run_id.
        self._stop_callbacks: dict[str, Callable[[], None]] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket, run_id: str) -> None:
        """Accept and register a WebSocket for *run_id*."""
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
        logger.info("WS connect: run_id=%s (total=%d)", run_id, len(self.active_connections[run_id]))

    def disconnect(self, websocket: WebSocket, run_id: str) -> None:
        """Remove a WebSocket from *run_id* tracking."""
        conns = self.active_connections.get(run_id, [])
        try:
            conns.remove(websocket)
        except ValueError:
            pass
        if not conns:
            self.active_connections.pop(run_id, None)
        logger.info("WS disconnect: run_id=%s (remaining=%d)", run_id, len(conns))

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, run_id: str, message: dict) -> None:
        """Send *message* as JSON to every client watching *run_id*.

        Silently removes connections that have closed.
        """
        conns = self.active_connections.get(run_id, [])
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, run_id)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        """Send a message to a single client."""
        try:
            await websocket.send_json(message)
        except Exception:
            logger.debug("Failed to send personal message")

    # ------------------------------------------------------------------
    # Stop signalling
    # ------------------------------------------------------------------

    def request_stop(self, run_id: str) -> None:
        """Signal that a client wants to stop *run_id*."""
        evt = self._stop_events.get(run_id)
        if evt is not None:
            evt.set()
        cb = self._stop_callbacks.get(run_id)
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.exception("Error in stop callback for run_id=%s", run_id)

    def register_stop_event(self, run_id: str) -> asyncio.Event:
        """Create and return an asyncio.Event for stop signalling."""
        evt = asyncio.Event()
        self._stop_events[run_id] = evt
        return evt

    def register_stop_callback(self, run_id: str, callback: Callable[[], None]) -> None:
        """Register a callback invoked when a client requests stop."""
        self._stop_callbacks[run_id] = callback

    def cleanup_run(self, run_id: str) -> None:
        """Remove all state for a completed/cancelled run."""
        self.active_connections.pop(run_id, None)
        self._stop_events.pop(run_id, None)
        self._stop_callbacks.pop(run_id, None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def client_count(self, run_id: str) -> int:
        """Return number of connected clients for *run_id*."""
        return len(self.active_connections.get(run_id, []))

    def active_runs(self) -> list[str]:
        """Return list of run_ids with active connections."""
        return list(self.active_connections.keys())


# ---------------------------------------------------------------------------
# Singleton instance — imported by main.py and backtest_service
# ---------------------------------------------------------------------------
ws_manager = WebSocketManager()
