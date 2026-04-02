"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { formatCurrency } from "@/lib/format";
import { CHART_COLORS } from "./chart-theme";

interface AttemptOutcome {
  cumulative_pnl: number;
  days_traded: number;
  status: string;
  daily_pnls: number[];
}

interface EvalFanChartProps {
  attempts: AttemptOutcome[];
}

/** Colors for the eval fan chart */
const FAN_COLORS = {
  p5_p25: "#ef4444",
  p25_p50: "#f97316",
  p50_p75: "#22c55e",
  p75_p95: "#16a34a",
  pass_line: "#22c55e",
  fail_line: "#ef4444",
} as const;

/**
 * Build percentile fan chart data from daily PnL arrays across all attempts.
 * Each step = a trading day. We compute cumulative equity, then percentile bands.
 */
function buildFanData(attempts: AttemptOutcome[]) {
  if (attempts.length === 0) return [];

  const maxDays = Math.max(...attempts.map((a) => a.daily_pnls.length));
  if (maxDays === 0) return [];

  // Build cumulative equity curves for each attempt
  const curves: number[][] = attempts.map((a) => {
    const curve: number[] = [];
    let cumulative = 0;
    for (const pnl of a.daily_pnls) {
      cumulative += pnl;
      curve.push(cumulative);
    }
    return curve;
  });

  // For each day, compute percentile bands
  const fanData: Array<{
    day: number;
    base: number;
    band_p5_p25: number;
    band_p25_p50: number;
    band_p50_p75: number;
    band_p75_p95: number;
  }> = [];

  for (let d = 0; d < maxDays; d++) {
    const values = curves
      .filter((c) => d < c.length)
      .map((c) => c[d])
      .sort((a, b) => a - b);

    if (values.length === 0) continue;

    const pct = (p: number) => {
      const idx = Math.floor(p * (values.length - 1));
      return values[idx];
    };

    const p5 = pct(0.05);
    const p25 = pct(0.25);
    const p50 = pct(0.5);
    const p75 = pct(0.75);
    const p95 = pct(0.95);

    fanData.push({
      day: d + 1,
      base: p5,
      band_p5_p25: p25 - p5,
      band_p25_p50: p50 - p25,
      band_p50_p75: p75 - p50,
      band_p75_p95: p95 - p75,
    });
  }

  return fanData;
}

function EvalFanTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string }>;
  label?: number;
}) {
  if (!active || !payload || payload.length === 0) return null;

  const base = payload.find((p) => p.dataKey === "base")?.value ?? 0;
  const bp5p25 = payload.find((p) => p.dataKey === "band_p5_p25")?.value ?? 0;
  const bp25p50 =
    payload.find((p) => p.dataKey === "band_p25_p50")?.value ?? 0;
  const bp50p75 =
    payload.find((p) => p.dataKey === "band_p50_p75")?.value ?? 0;
  const bp75p95 =
    payload.find((p) => p.dataKey === "band_p75_p95")?.value ?? 0;

  const p5 = base;
  const p25 = base + bp5p25;
  const p50 = p25 + bp25p50;
  const p75 = p50 + bp50p75;
  const p95 = p75 + bp75p95;

  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500">Day {label}</p>
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
    </div>
  );
}

export function EvalFanChart({ attempts }: EvalFanChartProps) {
  const fanData = buildFanData(attempts);

  if (fanData.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No attempt data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart
        data={fanData}
        margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
      >
        <defs>
          <linearGradient id="evalFanP5P25" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={FAN_COLORS.p5_p25}
              stopOpacity={0.25}
            />
            <stop
              offset="100%"
              stopColor={FAN_COLORS.p5_p25}
              stopOpacity={0.08}
            />
          </linearGradient>
          <linearGradient id="evalFanP25P50" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={FAN_COLORS.p25_p50}
              stopOpacity={0.25}
            />
            <stop
              offset="100%"
              stopColor={FAN_COLORS.p25_p50}
              stopOpacity={0.08}
            />
          </linearGradient>
          <linearGradient id="evalFanP50P75" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={FAN_COLORS.p50_p75}
              stopOpacity={0.25}
            />
            <stop
              offset="100%"
              stopColor={FAN_COLORS.p50_p75}
              stopOpacity={0.08}
            />
          </linearGradient>
          <linearGradient id="evalFanP75P95" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={FAN_COLORS.p75_p95}
              stopOpacity={0.35}
            />
            <stop
              offset="100%"
              stopColor={FAN_COLORS.p75_p95}
              stopOpacity={0.1}
            />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={CHART_COLORS.grid}
          vertical={false}
        />
        <XAxis
          dataKey="day"
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={{ stroke: CHART_COLORS.grid }}
          tickLine={false}
          minTickGap={40}
          label={{
            value: "Trading Day",
            position: "insideBottom",
            offset: -2,
            style: { fontSize: 10, fill: CHART_COLORS.text },
          }}
        />
        <YAxis
          tickFormatter={(v: number) => formatCurrency(v)}
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={false}
          tickLine={false}
          width={80}
        />
        <Tooltip content={<EvalFanTooltip />} />
        <ReferenceLine y={0} stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
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
          fill="url(#evalFanP5P25)"
          dot={false}
          activeDot={false}
        />
        {/* P25-P50 band */}
        <Area
          type="monotone"
          dataKey="band_p25_p50"
          stackId="fan"
          stroke="none"
          fill="url(#evalFanP25P50)"
          dot={false}
          activeDot={false}
        />
        {/* P50-P75 band */}
        <Area
          type="monotone"
          dataKey="band_p50_p75"
          stackId="fan"
          stroke="none"
          fill="url(#evalFanP50P75)"
          dot={false}
          activeDot={false}
        />
        {/* P75-P95 band */}
        <Area
          type="monotone"
          dataKey="band_p75_p95"
          stackId="fan"
          stroke="none"
          fill="url(#evalFanP75P95)"
          dot={false}
          activeDot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
