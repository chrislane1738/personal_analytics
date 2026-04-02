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
import { CHART_COLORS, REGIME_COLORS } from "./chart-theme";

interface RegimePassRateProps {
  data: Record<string, number>;
}

function RegimeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { regime: string; pass_rate: number } }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const { regime, pass_rate } = payload[0].payload;
  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500 capitalize">
        {regime}
      </p>
      <p className="font-mono text-xs text-zinc-300">
        Pass Rate: {(pass_rate * 100).toFixed(1)}%
      </p>
    </div>
  );
}

export function RegimePassRate({ data }: RegimePassRateProps) {
  const chartData = Object.entries(data)
    .map(([regime, pass_rate]) => ({ regime, pass_rate }))
    .sort((a, b) => b.pass_rate - a.pass_rate);

  if (chartData.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No regime data available
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
          domain={[0, 1]}
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={{ stroke: CHART_COLORS.grid }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="regime"
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 11, fill: CHART_COLORS.text }}
          axisLine={false}
          tickLine={false}
          width={80}
        />
        <Tooltip content={<RegimeTooltip />} />
        <Bar dataKey="pass_rate" radius={[0, 4, 4, 0]} barSize={24}>
          {chartData.map((entry) => (
            <Cell
              key={entry.regime}
              fill={
                REGIME_COLORS[entry.regime.toLowerCase()] ?? CHART_COLORS.orange
              }
              fillOpacity={0.7}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
