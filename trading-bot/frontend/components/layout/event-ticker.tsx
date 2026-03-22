"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Activity } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TickerEvent {
  type: string;
  message: string;
  timestamp: number;
}

interface LastRunSummary {
  strategy_name: string;
  total_return: number;
  sharpe: number;
}

// ---------------------------------------------------------------------------
// Color mapping for event types
// ---------------------------------------------------------------------------

function eventColor(type: string): string {
  const upper = type.toUpperCase();
  if (upper === "SIGNAL" || upper === "signal") return "text-[#22c55e]";
  if (upper === "FILL" || upper === "fill") return "text-[#f97316]";
  if (upper === "RISK" || upper === "risk" || upper === "ERROR" || upper === "error")
    return "text-[#ef4444]";
  if (upper === "ORDER" || upper === "order") return "text-[#3b82f6]";
  if (upper === "PORTFOLIO" || upper === "portfolio" || upper === "PORTFOLIO_UPDATE")
    return "text-[#8b5cf6]";
  return "text-zinc-500";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8001";
const MAX_TICKER_EVENTS = 50;

export function EventTicker() {
  const [events, setEvents] = useState<TickerEvent[]>([]);
  const [lastRun, setLastRun] = useState<LastRunSummary | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempt = useRef(0);

  // Fetch last run summary from REST API
  useEffect(() => {
    const API_BASE =
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
    fetch(`${API_BASE}/api/runs?limit=1&sort=created_at&order=desc`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.runs && data.runs.length > 0) {
          const run = data.runs[0];
          setLastRun({
            strategy_name: run.strategy_name,
            total_return: run.total_return,
            sharpe: run.sharpe,
          });
        }
      })
      .catch(() => {
        // Silently fail — ticker is non-critical
      });
  }, []);

  // WebSocket event listener for live events
  const handleMessage = useCallback((msgData: Record<string, unknown>) => {
    const type = msgData.type as string;
    if (type === "ping" || type === "subscribed" || type === "stop_requested")
      return;

    const data = (msgData.data as Record<string, unknown>) ?? msgData;
    let message = type;

    if (type === "SIGNAL" || type === "signal") {
      message = `SIGNAL ${data.symbol ?? ""} ${data.direction ?? ""}`;
    } else if (type === "FILL" || type === "fill") {
      message = `FILL ${data.symbol ?? ""} ${data.quantity ?? ""} @ ${data.price ?? ""}`;
    } else if (type === "RISK" || type === "risk") {
      message = `RISK ${data.message ?? data.reason ?? "alert"}`;
    } else if (type === "ORDER" || type === "order") {
      message = `ORDER ${data.symbol ?? ""} ${data.side ?? ""}`;
    } else if (type === "complete") {
      message = "COMPLETE backtest finished";
    } else if (type === "error") {
      message = `ERROR ${data.message ?? "unknown"}`;
    } else if (
      type === "PORTFOLIO_UPDATE" ||
      type === "portfolio"
    ) {
      return; // Too noisy for ticker
    }

    const event: TickerEvent = {
      type,
      message: message.trim(),
      timestamp: Date.now(),
    };

    setEvents((prev) => {
      const next = [event, ...prev];
      return next.length > MAX_TICKER_EVENTS
        ? next.slice(0, MAX_TICKER_EVENTS)
        : next;
    });
  }, []);

  // Connect to WebSocket for ambient event streaming
  useEffect(() => {
    function connect() {
      try {
        const ws = new WebSocket(`${WS_BASE}/ws/monitor`);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          reconnectAttempt.current = 0;
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
          setConnected(false);
        };

        ws.onclose = () => {
          wsRef.current = null;
          setConnected(false);
          // Exponential backoff reconnect
          const delay = Math.min(
            2000 * Math.pow(2, reconnectAttempt.current),
            60000
          );
          reconnectAttempt.current += 1;
          reconnectTimer.current = setTimeout(connect, delay);
        };
      } catch {
        setConnected(false);
      }
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
  }, [handleMessage]);

  // Build display content
  const latestEvent = events.length > 0 ? events[0] : null;

  return (
    <div className="flex h-7 items-center gap-3 border-t border-[#1a1a1a] bg-[#0a0a0a] px-3 shrink-0">
      {/* Connection indicator */}
      <div className="flex items-center gap-1.5">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            connected ? "bg-[#22c55e] animate-pulse" : "bg-zinc-600"
          }`}
        />
        <Activity className="h-3 w-3 text-zinc-600" />
      </div>

      {/* Event content */}
      <div className="flex-1 overflow-hidden">
        {latestEvent ? (
          <div className="flex items-center gap-2 font-mono text-[10px]">
            <span className={eventColor(latestEvent.type)}>
              {latestEvent.message}
            </span>
            <span className="text-zinc-700">
              {new Date(latestEvent.timestamp).toLocaleTimeString()}
            </span>
          </div>
        ) : lastRun ? (
          <div className="flex items-center gap-3 font-mono text-[10px] text-zinc-500">
            <span>Last run: {lastRun.strategy_name}</span>
            <span
              className={
                lastRun.total_return >= 0
                  ? "text-[#22c55e]"
                  : "text-[#ef4444]"
              }
            >
              {(lastRun.total_return * 100).toFixed(2)}%
            </span>
            <span className="text-zinc-500">
              Sharpe {lastRun.sharpe.toFixed(2)}
            </span>
          </div>
        ) : (
          <span className="font-mono text-[10px] text-zinc-600">
            No recent activity
          </span>
        )}
      </div>

      {/* Event count badge */}
      {events.length > 0 && (
        <span className="font-mono text-[10px] text-zinc-600">
          {events.length} events
        </span>
      )}
    </div>
  );
}
