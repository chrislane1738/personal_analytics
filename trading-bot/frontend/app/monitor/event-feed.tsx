"use client";

import { useEffect, useRef } from "react";
import type { WsEvent, WsEventType } from "@/hooks/use-websocket";

// ---------------------------------------------------------------------------
// Event type styling
// ---------------------------------------------------------------------------

const EVENT_STYLES: Record<
  WsEventType,
  { label: string; color: string; bg: string }
> = {
  signal: { label: "SIGNAL", color: "#22c55e", bg: "#22c55e15" },
  order: { label: "ORDER", color: "#3b82f6", bg: "#3b82f615" },
  fill: { label: "FILL", color: "#f97316", bg: "#f9731615" },
  risk: { label: "RISK", color: "#ef4444", bg: "#ef444415" },
  portfolio: { label: "PORT", color: "#a855f7", bg: "#a855f715" },
  progress: { label: "PROG", color: "#71717a", bg: "#71717a15" },
  complete: { label: "DONE", color: "#3b82f6", bg: "#3b82f615" },
  error: { label: "ERR", color: "#ef4444", bg: "#ef444415" },
};

// ---------------------------------------------------------------------------
// Event summary formatters
// ---------------------------------------------------------------------------

function formatEventSummary(event: WsEvent): string {
  const d = event.data;
  switch (event.type) {
    case "signal":
      return `${d.symbol ?? "?"} ${d.direction ?? ""} strength=${d.strength ?? "?"} — ${d.reason ?? ""}`;
    case "order":
      return `${d.action ?? ""} ${d.quantity ?? 0} ${d.symbol ?? "?"} @ ${d.order_type ?? "market"}${d.price ? ` $${d.price}` : ""}`;
    case "fill":
      return `${d.action ?? ""} ${d.quantity ?? 0} ${d.symbol ?? "?"} filled @ $${d.fill_price ?? 0} (comm=$${d.commission ?? 0})`;
    case "risk":
      return `${d.rule_name ?? "rule"}: ${d.reason ?? "blocked"}`;
    case "portfolio":
      return `equity=$${Number(d.equity ?? 0).toLocaleString()} cash=$${Number(d.cash ?? 0).toLocaleString()} dd=${((Number(d.drawdown_pct ?? 0)) * 100).toFixed(2)}%`;
    case "progress":
      return `${d.current ?? 0}/${d.total ?? 0} (${((Number(d.pct ?? 0)) * 100).toFixed(1)}%)`;
    case "complete":
      return d.message ? String(d.message) : "Backtest completed";
    case "error":
      return d.message ? String(d.message) : "An error occurred";
    default:
      return JSON.stringify(d);
  }
}

function formatTimestamp(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface EventFeedProps {
  events: WsEvent[];
  filters: Record<WsEventType, boolean>;
}

export function EventFeed({ events, filters }: EventFeedProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isUserScrolledUp = useRef(false);

  // Auto-scroll to bottom unless user scrolled up
  useEffect(() => {
    const el = containerRef.current;
    if (!el || isUserScrolledUp.current) return;
    el.scrollTop = el.scrollHeight;
  }, [events]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    // If user is within 40px of bottom, consider "at bottom"
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    isUserScrolledUp.current = !atBottom;
  };

  const filtered = events.filter((e) => filters[e.type] !== false);

  if (filtered.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-600">
        Waiting for events...
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="h-full overflow-y-auto px-3 py-2 font-mono text-[11px] leading-relaxed"
    >
      {filtered.map((event, i) => {
        const style = EVENT_STYLES[event.type];
        return (
          <div key={i} className="flex gap-2 py-[2px]">
            <span className="shrink-0 text-zinc-600">
              {formatTimestamp(event.receivedAt)}
            </span>
            <span
              className="shrink-0 rounded px-1 font-semibold"
              style={{ color: style.color, backgroundColor: style.bg }}
            >
              {style.label}
            </span>
            <span className="truncate text-zinc-400">
              {formatEventSummary(event)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
