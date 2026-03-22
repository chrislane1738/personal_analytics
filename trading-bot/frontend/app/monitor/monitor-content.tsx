"use client";

import { useEffect, useState } from "react";
import { Activity, Square, Wifi, WifiOff } from "lucide-react";
import {
  useWebSocket,
  type WsEventType,
  type WsStatus,
} from "@/hooks/use-websocket";
import { LiveEquityChart } from "./live-equity-chart";
import { PortfolioPanel } from "./portfolio-panel";
import { EventFeed } from "./event-feed";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Status bar helpers
// ---------------------------------------------------------------------------

const STATUS_LABELS: Record<WsStatus, string> = {
  idle: "Idle",
  connecting: "Connecting...",
  connected: "Running",
  completed: "Completed",
  error: "Error",
};

const STATUS_COLORS: Record<WsStatus, string> = {
  idle: "text-zinc-500",
  connecting: "text-yellow-400",
  connected: "text-[#22c55e]",
  completed: "text-[#3b82f6]",
  error: "text-[#ef4444]",
};

function StatusDot({ status }: { status: WsStatus }) {
  const dotColor: Record<WsStatus, string> = {
    idle: "bg-zinc-500",
    connecting: "bg-yellow-400 animate-pulse",
    connected: "bg-[#22c55e] animate-pulse",
    completed: "bg-[#3b82f6]",
    error: "bg-[#ef4444]",
  };
  return <span className={cn("inline-block h-2 w-2 rounded-full", dotColor[status])} />;
}

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

function ProgressBar({ pct }: { pct: number }) {
  return (
    <div className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-800">
      <div
        className="h-full rounded-full bg-[#f97316] transition-all duration-300 ease-out"
        style={{ width: `${Math.min(pct * 100, 100)}%` }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Elapsed timer
// ---------------------------------------------------------------------------

function useElapsed(startTime: number | null) {
  const [, setTick] = useState(0);

  // Re-render every second while running
  useEffect(() => {
    if (!startTime) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [startTime]);

  if (!startTime) return "0s";
  const seconds = Math.floor((Date.now() - startTime) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

// ---------------------------------------------------------------------------
// Main content
// ---------------------------------------------------------------------------

interface MonitorContentProps {
  initialRunId: string | null;
}

export function MonitorContent({ initialRunId }: MonitorContentProps) {
  const [runId] = useState<string | null>(initialRunId);

  const {
    events,
    equityCurve,
    portfolioState,
    progress,
    status,
    isConnected,
    requestStop,
  } = useWebSocket(runId);

  const startTime = events.length > 0 ? events[0].receivedAt : null;
  const elapsed = useElapsed(status === "connected" ? startTime : null);

  // Event type filters
  const [filters, setFilters] = useState<Record<WsEventType, boolean>>({
    signal: true,
    order: true,
    fill: true,
    risk: true,
    portfolio: true,
    progress: false,
    complete: true,
    error: true,
  });

  const toggleFilter = (type: WsEventType) => {
    setFilters((prev) => ({ ...prev, [type]: !prev[type] }));
  };

  // ------------------------------------------------------------------
  // No active run — show placeholder
  // ------------------------------------------------------------------
  if (!runId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-6">
        <Activity className="h-12 w-12 text-zinc-700" />
        <p className="text-sm text-zinc-500">
          No active backtest. Launch one from the{" "}
          <a href="/launch" className="text-[#f97316] hover:underline">
            Launch page
          </a>
          .
        </p>
      </div>
    );
  }

  // ------------------------------------------------------------------
  // Active run
  // ------------------------------------------------------------------
  return (
    <div className="flex h-full flex-col">
      {/* ---- Status bar ---- */}
      <div className="flex items-center gap-4 border-b border-[#1a1a1a] px-6 py-3">
        <div className="flex items-center gap-2">
          <StatusDot status={status} />
          <span className={cn("text-sm font-medium", STATUS_COLORS[status])}>
            {STATUS_LABELS[status]}
          </span>
        </div>

        {/* Connection indicator */}
        <div className="flex items-center gap-1 text-xs text-zinc-500">
          {isConnected ? (
            <Wifi className="h-3 w-3 text-[#22c55e]" />
          ) : (
            <WifiOff className="h-3 w-3 text-zinc-600" />
          )}
          <span className="font-mono">{runId.slice(0, 8)}...</span>
        </div>

        {/* Progress */}
        {progress && (
          <>
            <ProgressBar pct={progress.pct} />
            <span className="font-mono text-xs text-zinc-500">
              {(progress.pct * 100).toFixed(1)}%
            </span>
          </>
        )}

        {/* Elapsed time */}
        <span className="font-mono text-xs text-zinc-500">{elapsed}</span>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Stop button */}
        {status === "connected" && (
          <button
            onClick={requestStop}
            className="flex items-center gap-1.5 rounded-md border border-[#ef4444]/30 bg-[#ef4444]/10 px-3 py-1.5 text-xs font-medium text-[#ef4444] transition-colors hover:bg-[#ef4444]/20"
          >
            <Square className="h-3 w-3" />
            Stop
          </button>
        )}
      </div>

      {/* ---- Three-panel layout ---- */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Equity curve (60%) */}
        <div className="flex w-[60%] flex-col border-r border-[#1a1a1a]">
          <div className="border-b border-[#1a1a1a] px-4 py-2">
            <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Live Equity Curve
            </h2>
          </div>
          <div className="flex-1 p-4">
            <LiveEquityChart data={equityCurve} />
          </div>
        </div>

        {/* Right (40%) */}
        <div className="flex w-[40%] flex-col">
          {/* Right top: Portfolio state */}
          <div className="flex flex-col border-b border-[#1a1a1a]">
            <div className="border-b border-[#1a1a1a] px-4 py-2">
              <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                Portfolio
              </h2>
            </div>
            <div className="p-4">
              <PortfolioPanel state={portfolioState} />
            </div>
          </div>

          {/* Right bottom: Event feed */}
          <div className="flex flex-1 flex-col overflow-hidden">
            <div className="flex items-center gap-2 border-b border-[#1a1a1a] px-4 py-2">
              <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                Event Feed
              </h2>
              <div className="flex-1" />
              <EventFilters filters={filters} onToggle={toggleFilter} />
            </div>
            <div className="flex-1 overflow-hidden">
              <EventFeed events={events} filters={filters} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Event filter toggles
// ---------------------------------------------------------------------------

const FILTER_TYPES: { type: WsEventType; label: string; color: string }[] = [
  { type: "signal", label: "SIG", color: "#22c55e" },
  { type: "order", label: "ORD", color: "#3b82f6" },
  { type: "fill", label: "FILL", color: "#f97316" },
  { type: "risk", label: "RISK", color: "#ef4444" },
  { type: "portfolio", label: "PORT", color: "#a855f7" },
];

function EventFilters({
  filters,
  onToggle,
}: {
  filters: Record<WsEventType, boolean>;
  onToggle: (type: WsEventType) => void;
}) {
  return (
    <div className="flex gap-1">
      {FILTER_TYPES.map(({ type, label, color }) => (
        <button
          key={type}
          onClick={() => onToggle(type)}
          className={cn(
            "rounded px-1.5 py-0.5 font-mono text-[10px] font-medium transition-colors",
            filters[type]
              ? "opacity-100"
              : "opacity-30",
          )}
          style={{
            color: filters[type] ? color : undefined,
            backgroundColor: filters[type] ? `${color}15` : undefined,
          }}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
