"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import type { MonteCarloResult, FanChartPoint } from "@/lib/types";
import { formatCurrency, formatPercent } from "@/lib/format";
import { CHART_COLORS } from "./chart-theme";

interface FanChartProps {
  data: MonteCarloResult;
}

/** Colors for the fan chart percentile bands */
const FAN_COLORS = {
  p5_p25: "#ef4444",    // red
  p25_p50: "#f97316",   // orange
  p50_p75: "#22c55e",   // green
  p75_p95: "#16a34a",   // dark green
  actual: "#3b82f6",    // blue
} as const;

/**
 * Transform fan chart data for stacked area rendering.
 * Recharts stacked areas add values, so we compute deltas between bands.
 */
function prepareFanData(
  points: FanChartPoint[]
): Array<{
  step: number;
  base: number;
  band_p5_p25: number;
  band_p25_p50: number;
  band_p50_p75: number;
  band_p75_p95: number;
  actual: number | null;
}> {
  return points.map((p) => ({
    step: p.step,
    base: p.p5,
    band_p5_p25: p.p25 - p.p5,
    band_p25_p50: p.p50 - p.p25,
    band_p50_p75: p.p75 - p.p50,
    band_p75_p95: p.p95 - p.p75,
    actual: p.actual,
  }));
}

/** Prepare drawdown distribution as histogram buckets */
function prepareDrawdownHistogram(
  drawdowns: number[]
): Array<{ bucket: string; count: number; midpoint: number }> {
  if (drawdowns.length === 0) return [];

  const min = Math.min(...drawdowns);
  const max = Math.max(...drawdowns);
  const bucketCount = 20;
  const bucketSize = (max - min) / bucketCount || 0.01;

  const buckets: Array<{ bucket: string; count: number; midpoint: number }> = [];
  for (let i = 0; i < bucketCount; i++) {
    const lo = min + i * bucketSize;
    const hi = lo + bucketSize;
    const count = drawdowns.filter((d) => d >= lo && (i === bucketCount - 1 ? d <= hi : d < hi)).length;
    buckets.push({
      bucket: `${(lo * 100).toFixed(0)}%`,
      count,
      midpoint: (lo + hi) / 2,
    });
  }
  return buckets;
}

function FanChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string; color: string }>;
  label?: number;
}) {
  if (!active || !payload || payload.length === 0) return null;

  // Reconstruct percentile values from stacked data
  const base = payload.find((p) => p.dataKey === "base")?.value ?? 0;
  const bp5p25 = payload.find((p) => p.dataKey === "band_p5_p25")?.value ?? 0;
  const bp25p50 = payload.find((p) => p.dataKey === "band_p25_p50")?.value ?? 0;
  const bp50p75 = payload.find((p) => p.dataKey === "band_p50_p75")?.value ?? 0;
  const bp75p95 = payload.find((p) => p.dataKey === "band_p75_p95")?.value ?? 0;
  const actual = payload.find((p) => p.dataKey === "actual")?.value;

  const p5 = base;
  const p25 = base + bp5p25;
  const p50 = p25 + bp25p50;
  const p75 = p50 + bp50p75;
  const p95 = p75 + bp75p95;

  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500">Step {label}</p>
      <p className="font-mono text-xs" style={{ color: FAN_COLORS.p75_p95 }}>
        P95: {formatCurrency(p95)}
      </p>
      <p className="font-mono text-xs" style={{ color: FAN_COLORS.p50_p75 }}>
        P75: {formatCurrency(p75)}
      </p>
      <p className="font-mono text-xs text-zinc-300">
        P50: {formatCurrency(p50)}
      </p>
      <p className="font-mono text-xs" style={{ color: FAN_COLORS.p25_p50 }}>
        P25: {formatCurrency(p25)}
      </p>
      <p className="font-mono text-xs" style={{ color: FAN_COLORS.p5_p25 }}>
        P5: {formatCurrency(p5)}
      </p>
      {actual != null && (
        <p className="mt-1 font-mono text-xs" style={{ color: FAN_COLORS.actual }}>
          Actual: {formatCurrency(actual)}
        </p>
      )}
    </div>
  );
}

function DrawdownHistogramTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { bucket: string; count: number } }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const { bucket, count } = payload[0].payload;
  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="font-mono text-[11px] text-zinc-500">Max DD: {bucket}</p>
      <p className="font-mono text-xs text-zinc-300">{count} simulations</p>
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-[9px] uppercase tracking-wider text-zinc-500">
        {label}
      </span>
      <span className="font-mono text-xs text-zinc-300">{value}</span>
    </div>
  );
}

export function FanChart({ data }: FanChartProps) {
  const fanData = prepareFanData(data.fan_chart ?? []);
  const ddHist = prepareDrawdownHistogram(data.drawdown_distribution ?? []);

  if (fanData.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No Monte Carlo data available
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Top section: fan chart + stats panel */}
      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Fan chart */}
        <div className="h-[360px] flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={fanData}
              margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
            >
              <defs>
                <linearGradient id="fanP5P25" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={FAN_COLORS.p5_p25} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={FAN_COLORS.p5_p25} stopOpacity={0.08} />
                </linearGradient>
                <linearGradient id="fanP25P50" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={FAN_COLORS.p25_p50} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={FAN_COLORS.p25_p50} stopOpacity={0.08} />
                </linearGradient>
                <linearGradient id="fanP50P75" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={FAN_COLORS.p50_p75} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={FAN_COLORS.p50_p75} stopOpacity={0.08} />
                </linearGradient>
                <linearGradient id="fanP75P95" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={FAN_COLORS.p75_p95} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={FAN_COLORS.p75_p95} stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={CHART_COLORS.grid}
                vertical={false}
              />
              <XAxis
                dataKey="step"
                stroke={CHART_COLORS.text}
                tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                axisLine={{ stroke: CHART_COLORS.grid }}
                tickLine={false}
                minTickGap={40}
              />
              <YAxis
                tickFormatter={(v: number) => formatCurrency(v)}
                stroke={CHART_COLORS.text}
                tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                axisLine={false}
                tickLine={false}
                width={80}
              />
              <Tooltip content={<FanChartTooltip />} />
              {/* Invisible base (P5 level) */}
              <Area
                type="monotone"
                dataKey="base"
                stackId="fan"
                stroke="none"
                fill="transparent"
                dot={false}
                activeDot={false}
              />
              {/* P5-P25 band */}
              <Area
                type="monotone"
                dataKey="band_p5_p25"
                stackId="fan"
                stroke="none"
                fill="url(#fanP5P25)"
                dot={false}
                activeDot={false}
              />
              {/* P25-P50 band */}
              <Area
                type="monotone"
                dataKey="band_p25_p50"
                stackId="fan"
                stroke="none"
                fill="url(#fanP25P50)"
                dot={false}
                activeDot={false}
              />
              {/* P50-P75 band */}
              <Area
                type="monotone"
                dataKey="band_p50_p75"
                stackId="fan"
                stroke="none"
                fill="url(#fanP50P75)"
                dot={false}
                activeDot={false}
              />
              {/* P75-P95 band */}
              <Area
                type="monotone"
                dataKey="band_p75_p95"
                stackId="fan"
                stroke="none"
                fill="url(#fanP75P95)"
                dot={false}
                activeDot={false}
              />
              {/* Actual equity curve overlay */}
              <Area
                type="monotone"
                dataKey="actual"
                stroke={FAN_COLORS.actual}
                strokeWidth={2}
                fill="transparent"
                dot={false}
                activeDot={{ r: 3, fill: FAN_COLORS.actual }}
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Stats panel */}
        <div className="w-full shrink-0 lg:w-56">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-3">
            <div className="mb-3 flex items-center gap-2">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Simulation Stats
              </h3>
              {data.is_outlier && (
                <span className="inline-flex h-5 items-center rounded-full bg-amber-500/15 px-2 text-[10px] font-medium text-amber-400">
                  Outlier
                </span>
              )}
            </div>
            <div className="grid gap-2">
              <StatRow
                label="Simulations"
                value={data.simulations.toLocaleString()}
              />
              <StatRow
                label="Median Final Equity"
                value={formatCurrency(data.median_final_equity)}
              />
              <StatRow
                label="5th Percentile"
                value={formatCurrency(data.percentiles.p5)}
              />
              <StatRow
                label="95th Percentile"
                value={formatCurrency(data.percentiles.p95)}
              />
              <StatRow
                label="Prob. of Ruin"
                value={formatPercent(data.probability_of_ruin)}
              />
              <StatRow
                label="Prob. of Profit"
                value={formatPercent(data.probability_of_profit)}
              />
              <div className="flex items-baseline justify-between">
                <span className="text-[9px] uppercase tracking-wider text-zinc-500">
                  Outlier
                </span>
                <span
                  className={`inline-flex h-5 items-center rounded-full px-2 text-[10px] font-medium ${
                    data.is_outlier
                      ? "bg-amber-500/15 text-amber-400"
                      : "bg-zinc-800 text-zinc-400"
                  }`}
                >
                  {data.is_outlier ? "Yes" : "No"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Drawdown distribution histogram */}
      {ddHist.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
            Max Drawdown Distribution
          </h3>
          <div className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={ddHist}
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
                  tick={{ fontSize: 9, fill: CHART_COLORS.text }}
                  axisLine={{ stroke: CHART_COLORS.grid }}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  stroke={CHART_COLORS.text}
                  tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                />
                <Tooltip content={<DrawdownHistogramTooltip />} />
                <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                  {ddHist.map((entry, index) => (
                    <Cell
                      key={index}
                      fill={CHART_COLORS.drawdown}
                      fillOpacity={0.3 + 0.7 * (entry.count / Math.max(...ddHist.map((d) => d.count)))}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
