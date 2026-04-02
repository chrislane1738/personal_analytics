"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";
import { CHART_COLORS } from "./chart-theme";

interface AttemptOutcome {
  days_traded: number;
  status: string;
}

interface DaysDistributionProps {
  attempts: AttemptOutcome[];
}

function DaysTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{
    payload: { bucket: string; pass: number; fail: number };
  }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const { bucket, pass, fail } = payload[0].payload;
  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500">{bucket} days</p>
      <p className="font-mono text-xs text-[#22c55e]">Pass: {pass}</p>
      <p className="font-mono text-xs text-[#ef4444]">Fail: {fail}</p>
    </div>
  );
}

/**
 * Build histogram buckets from attempt outcomes, split by pass/fail status.
 */
function buildHistogram(
  attempts: AttemptOutcome[]
): Array<{ bucket: string; pass: number; fail: number }> {
  if (attempts.length === 0) return [];

  const days = attempts.map((a) => a.days_traded);
  const min = Math.min(...days);
  const max = Math.max(...days);
  const bucketCount = Math.min(15, Math.max(5, Math.ceil(Math.sqrt(days.length))));
  const bucketSize = Math.max(1, Math.ceil((max - min + 1) / bucketCount));

  const buckets: Array<{ bucket: string; pass: number; fail: number }> = [];

  for (let i = 0; i < bucketCount; i++) {
    const lo = min + i * bucketSize;
    const hi = lo + bucketSize - 1;
    const inBucket = attempts.filter(
      (a) => a.days_traded >= lo && a.days_traded <= hi
    );
    const pass = inBucket.filter(
      (a) => a.status === "pass" || a.status === "passed"
    ).length;
    const fail = inBucket.filter(
      (a) => a.status !== "pass" && a.status !== "passed"
    ).length;

    if (lo === hi) {
      buckets.push({ bucket: `${lo}`, pass, fail });
    } else {
      buckets.push({ bucket: `${lo}-${hi}`, pass, fail });
    }
  }

  return buckets.filter((b) => b.pass > 0 || b.fail > 0);
}

export function DaysDistribution({ attempts }: DaysDistributionProps) {
  const histData = buildHistogram(attempts);

  if (histData.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No days distribution data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={histData}
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
        <Tooltip content={<DaysTooltip />} />
        <Bar
          dataKey="pass"
          stackId="days"
          fill="#22c55e"
          fillOpacity={0.7}
          radius={[0, 0, 0, 0]}
        />
        <Bar
          dataKey="fail"
          stackId="days"
          fill="#ef4444"
          fillOpacity={0.7}
          radius={[2, 2, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
