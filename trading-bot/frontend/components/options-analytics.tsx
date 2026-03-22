"use client";

import { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
  Legend,
} from "recharts";
import type { OptionsAnalyticsResponse } from "@/lib/types";
import { formatCurrency, formatPercent } from "@/lib/format";
import { CHART_COLORS } from "./charts/chart-theme";

interface OptionsAnalyticsProps {
  data: OptionsAnalyticsResponse;
}

/** Color mapping for Greeks */
const GREEKS_COLORS: Record<string, string> = {
  delta: CHART_COLORS.strategy,
  gamma: CHART_COLORS.purple,
  theta: CHART_COLORS.orange,
  vega: CHART_COLORS.benchmark,
};

function PremiumCard({
  label,
  value,
  colorClass,
}: {
  label: string;
  value: number;
  colorClass: string;
}) {
  return (
    <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2.5">
      <p className="font-sans text-[9px] uppercase tracking-wider text-zinc-500">
        {label}
      </p>
      <p className={`mt-0.5 font-mono text-lg font-bold ${colorClass}`}>
        {formatCurrency(value)}
      </p>
    </div>
  );
}

function WinRateTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{
    payload: { bucket: string; win_rate: number; trade_count: number };
  }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const { bucket, win_rate, trade_count } = payload[0].payload;
  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="font-mono text-[11px] text-zinc-500">{bucket}</p>
      <p className="font-mono text-xs text-zinc-300">
        Win Rate: {formatPercent(win_rate)}
      </p>
      <p className="font-mono text-xs text-zinc-500">
        {trade_count} trades
      </p>
    </div>
  );
}

function GreeksTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string; color: string }>;
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;

  const date = label
    ? new Date(label).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : "";

  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500">{date}</p>
      {payload.map((entry) => (
        <p
          key={entry.dataKey}
          className="font-mono text-xs"
          style={{ color: entry.color }}
        >
          {entry.dataKey}: {entry.value.toFixed(4)}
        </p>
      ))}
    </div>
  );
}

function formatAxisDate(timestamp: string): string {
  const d = new Date(timestamp);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function OptionsAnalytics({ data }: OptionsAnalyticsProps) {
  const [activeGreeks, setActiveGreeks] = useState<Record<string, boolean>>({
    delta: true,
    gamma: true,
    theta: true,
    vega: true,
  });

  const netPremiumColor =
    data.net_premium >= 0 ? "text-[#22c55e]" : "text-[#ef4444]";

  const assignmentPct = data.total_short_options > 0
    ? data.assignment_rate
    : 0;

  const hasGreeksData = data.greeks_timeline.length > 0;

  return (
    <div className="flex flex-col gap-6">
      {/* Premium summary cards */}
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Premium Summary
        </h3>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <PremiumCard
            label="Total Collected"
            value={data.total_collected}
            colorClass="text-[#22c55e]"
          />
          <PremiumCard
            label="Total Paid"
            value={data.total_paid}
            colorClass="text-[#ef4444]"
          />
          <PremiumCard
            label="Net Premium"
            value={data.net_premium}
            colorClass={netPremiumColor}
          />
        </div>
      </div>

      {/* Assignment metrics */}
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Assignment Metrics
        </h3>
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-3">
          <div className="mb-3 grid grid-cols-3 gap-4">
            <div>
              <p className="text-[9px] uppercase tracking-wider text-zinc-500">
                Assignments
              </p>
              <p className="mt-0.5 font-mono text-sm font-bold text-zinc-200">
                {data.assignment_count}
              </p>
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-wider text-zinc-500">
                Short Options
              </p>
              <p className="mt-0.5 font-mono text-sm font-bold text-zinc-200">
                {data.total_short_options}
              </p>
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-wider text-zinc-500">
                Assignment Rate
              </p>
              <p className="mt-0.5 font-mono text-sm font-bold text-zinc-200">
                {formatPercent(assignmentPct)}
              </p>
            </div>
          </div>
          {/* Progress bar */}
          <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-[#ef4444] transition-all"
              style={{ width: `${Math.min(assignmentPct * 100, 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Win rate by DTE bucket */}
      {data.win_rate_by_dte.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
            Win Rate by DTE Bucket
          </h3>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.win_rate_by_dte}
                margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={CHART_COLORS.grid}
                  vertical={false}
                />
                <XAxis
                  dataKey="bucket"
                  stroke={CHART_COLORS.text}
                  tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                  axisLine={{ stroke: CHART_COLORS.grid }}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  stroke={CHART_COLORS.text}
                  tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                  axisLine={false}
                  tickLine={false}
                  width={44}
                  domain={[0, 1]}
                />
                <Tooltip content={<WinRateTooltip />} />
                <Bar dataKey="win_rate" radius={[3, 3, 0, 0]}>
                  {data.win_rate_by_dte.map((entry, index) => (
                    <Cell
                      key={index}
                      fill={CHART_COLORS.strategy}
                      fillOpacity={0.3 + 0.7 * entry.win_rate}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Greeks timeline */}
      {hasGreeksData && (
        <div>
          <div className="mb-2 flex items-center gap-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Greeks Timeline
            </h3>
            <div className="flex gap-2">
              {(["delta", "gamma", "theta", "vega"] as const).map((greek) => (
                <button
                  key={greek}
                  type="button"
                  onClick={() =>
                    setActiveGreeks((prev) => ({
                      ...prev,
                      [greek]: !prev[greek],
                    }))
                  }
                  className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-medium transition-colors ${
                    activeGreeks[greek]
                      ? "text-white"
                      : "text-zinc-600 line-through"
                  }`}
                  style={{
                    backgroundColor: activeGreeks[greek]
                      ? GREEKS_COLORS[greek] + "30"
                      : "transparent",
                    color: activeGreeks[greek]
                      ? GREEKS_COLORS[greek]
                      : undefined,
                  }}
                >
                  {greek}
                </button>
              ))}
            </div>
          </div>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data.greeks_timeline}
                margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={CHART_COLORS.grid}
                  vertical={false}
                />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatAxisDate}
                  stroke={CHART_COLORS.text}
                  tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                  axisLine={{ stroke: CHART_COLORS.grid }}
                  tickLine={false}
                  minTickGap={40}
                />
                <YAxis
                  stroke={CHART_COLORS.text}
                  tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                  axisLine={false}
                  tickLine={false}
                  width={56}
                />
                <Tooltip content={<GreeksTooltip />} />
                <Legend
                  verticalAlign="top"
                  height={0}
                  content={() => null}
                />
                {(["delta", "gamma", "theta", "vega"] as const).map(
                  (greek) =>
                    activeGreeks[greek] && (
                      <Line
                        key={greek}
                        type="monotone"
                        dataKey={greek}
                        stroke={GREEKS_COLORS[greek]}
                        strokeWidth={1.5}
                        dot={false}
                        activeDot={{ r: 3, fill: GREEKS_COLORS[greek] }}
                      />
                    )
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
