"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type WsEventType =
  | "signal"
  | "order"
  | "fill"
  | "risk"
  | "portfolio"
  | "progress"
  | "complete"
  | "error";

export interface WsEvent {
  type: WsEventType;
  data: Record<string, unknown>;
  receivedAt: number; // Date.now() when we got the message
}

export interface PortfolioState {
  equity: number;
  cash: number;
  unrealized_pnl: number;
  realized_pnl: number;
  drawdown_pct: number;
}

export interface ProgressState {
  current: number;
  total: number;
  pct: number;
}

export type WsStatus = "idle" | "connecting" | "connected" | "error" | "completed";

export interface UseWebSocketReturn {
  events: WsEvent[];
  equityCurve: Array<{ timestamp: string; equity: number }>;
  portfolioState: PortfolioState | null;
  progress: ProgressState | null;
  status: WsStatus;
  isConnected: boolean;
  requestStop: () => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8001";
const MAX_EVENTS = 500;
const MAX_EQUITY_POINTS = 2000;
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useWebSocket(runId: string | null): UseWebSocketReturn {
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [equityCurve, setEquityCurve] = useState<
    Array<{ timestamp: string; equity: number }>
  >([]);
  const [portfolioState, setPortfolioState] = useState<PortfolioState | null>(
    null,
  );
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [status, setStatus] = useState<WsStatus>("idle");

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const doneRef = useRef(false); // true after "complete" or "error" — stop reconnecting

  // Reset state when runId changes
  useEffect(() => {
    setEvents([]);
    setEquityCurve([]);
    setPortfolioState(null);
    setProgress(null);
    setStatus(runId ? "connecting" : "idle");
    doneRef.current = false;
    reconnectAttempt.current = 0;
  }, [runId]);

  // ------------------------------------------------------------------
  // Message handler
  // ------------------------------------------------------------------
  const handleMessage = useCallback((msgData: Record<string, unknown>) => {
    const type = msgData.type as string;

    // Keepalive ping — ignore
    if (type === "ping") return;

    // Subscription confirmation
    if (type === "subscribed") {
      setStatus("connected");
      return;
    }

    // Stop acknowledged
    if (type === "stop_requested") return;

    // Map event_type names from the backend to our WsEventType
    const typeMap: Record<string, WsEventType> = {
      SIGNAL: "signal",
      signal: "signal",
      ORDER: "order",
      order: "order",
      FILL: "fill",
      fill: "fill",
      RISK: "risk",
      risk: "risk",
      PORTFOLIO_UPDATE: "portfolio",
      portfolio: "portfolio",
      progress: "progress",
      complete: "complete",
      error: "error",
    };

    const wsType = typeMap[type];
    if (!wsType) return; // unknown type — ignore

    const data = (msgData.data as Record<string, unknown>) ?? msgData;

    const event: WsEvent = { type: wsType, data, receivedAt: Date.now() };

    setEvents((prev) => {
      const next = [...prev, event];
      return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next;
    });

    // Update derived state
    if (wsType === "portfolio") {
      const ps: PortfolioState = {
        equity: Number(data.equity ?? 0),
        cash: Number(data.cash ?? 0),
        unrealized_pnl: Number(data.unrealized_pnl ?? 0),
        realized_pnl: Number(data.realized_pnl ?? 0),
        drawdown_pct: Number(data.drawdown_pct ?? 0),
      };
      setPortfolioState(ps);
      setEquityCurve((prev) => {
        const point = {
          timestamp: (data.timestamp as string) ?? new Date().toISOString(),
          equity: ps.equity,
        };
        const next = [...prev, point];
        return next.length > MAX_EQUITY_POINTS
          ? next.slice(-MAX_EQUITY_POINTS)
          : next;
      });
    }

    if (wsType === "progress") {
      setProgress({
        current: Number(data.current ?? 0),
        total: Number(data.total ?? 1),
        pct: Number(data.pct ?? 0),
      });
    }

    if (wsType === "complete") {
      setStatus("completed");
      doneRef.current = true;
    }

    if (wsType === "error") {
      setStatus("error");
      doneRef.current = true;
    }
  }, []);

  // ------------------------------------------------------------------
  // WebSocket connection
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!runId) {
      // Close any existing connection
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    function connect() {
      if (doneRef.current) return;

      const ws = new WebSocket(`${WS_BASE}/ws/monitor`);
      wsRef.current = ws;
      setStatus("connecting");

      ws.onopen = () => {
        reconnectAttempt.current = 0;
        // Subscribe to the run
        ws.send(JSON.stringify({ action: "subscribe", run_id: runId }));
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          handleMessage(parsed);
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onerror = () => {
        // onclose will fire after onerror
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (doneRef.current) return;

        // Exponential backoff reconnect
        const delay = Math.min(
          RECONNECT_BASE_MS * Math.pow(2, reconnectAttempt.current),
          RECONNECT_MAX_MS,
        );
        reconnectAttempt.current += 1;
        setStatus("connecting");
        reconnectTimer.current = setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [runId, handleMessage]);

  // ------------------------------------------------------------------
  // Stop request
  // ------------------------------------------------------------------
  const requestStop = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && runId) {
      wsRef.current.send(
        JSON.stringify({ action: "stop", run_id: runId }),
      );
    }
  }, [runId]);

  return {
    events,
    equityCurve,
    portfolioState,
    progress,
    status,
    isConnected: status === "connected",
    requestStop,
  };
}
