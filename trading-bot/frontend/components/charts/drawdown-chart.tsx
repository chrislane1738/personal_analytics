"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { CHART_COLORS } from "./chart-theme";

interface DrawdownPoint {
  date: string;
  drawdown: number;
}

interface DrawdownChartProps {
  data: DrawdownPoint[];
}

function formatAxisDate(date: string): string {
  const d = new Date(date);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number }>;
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
      <p className="font-mono text-xs text-[#ef4444]">
        {(payload[0].value * 100).toFixed(2)}%
      </p>
    </div>
  );
}

export function DrawdownChart({ data }: DrawdownChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No drawdown data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={CHART_COLORS.drawdown}
              stopOpacity={0}
            />
            <stop
              offset="100%"
              stopColor={CHART_COLORS.drawdown}
              stopOpacity={0.4}
            />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={CHART_COLORS.grid}
          vertical={false}
        />
        <XAxis
          dataKey="date"
          tickFormatter={formatAxisDate}
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={{ stroke: CHART_COLORS.grid }}
          tickLine={false}
          minTickGap={40}
        />
        <YAxis
          tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={false}
          tickLine={false}
          width={56}
          domain={["dataMin", 0]}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke={CHART_COLORS.drawdown}
          strokeWidth={1.5}
          fill="url(#drawdownGradient)"
          dot={false}
          activeDot={{ r: 3, fill: CHART_COLORS.drawdown }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
