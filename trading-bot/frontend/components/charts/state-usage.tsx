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

interface StateUsageProps {
  data: Record<string, number>;
}

/** Assign distinct colors to state entries */
const STATE_COLORS = [
  "#3b82f6", // blue
  "#22c55e", // green
  "#f97316", // orange
  "#a855f7", // purple
  "#eab308", // yellow
  "#ef4444", // red
  "#06b6d4", // cyan
  "#ec4899", // pink
];

function StateTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { state: string; count: number } }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const { state, count } = payload[0].payload;
  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500">{state}</p>
      <p className="font-mono text-xs text-zinc-300">
        Usage: {count.toLocaleString()}
      </p>
    </div>
  );
}

export function StateUsage({ data }: StateUsageProps) {
  const chartData = Object.entries(data)
    .map(([state, count]) => ({ state, count }))
    .sort((a, b) => b.count - a.count);

  if (chartData.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No state usage data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={CHART_COLORS.grid}
          horizontal={false}
        />
        <XAxis
          type="number"
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={{ stroke: CHART_COLORS.grid }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="state"
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={false}
          tickLine={false}
          width={100}
        />
        <Tooltip content={<StateTooltip />} />
        <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={20}>
          {chartData.map((_, index) => (
            <Cell
              key={index}
              fill={STATE_COLORS[index % STATE_COLORS.length]}
              fillOpacity={0.7}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
