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
import { formatCurrency } from "@/lib/format";
import { CHART_COLORS } from "@/components/charts/chart-theme";

interface LiveEquityChartProps {
  data: Array<{ timestamp: string; equity: number }>;
}

function formatAxisTime(timestamp: string): string {
  const d = new Date(timestamp);
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
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500">{date}</p>
      <p className="font-mono text-xs" style={{ color: CHART_COLORS.strategy }}>
        Equity: {formatCurrency(payload[0].value)}
      </p>
    </div>
  );
}

export function LiveEquityChart({ data }: LiveEquityChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-600">
        Waiting for portfolio data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart
        data={data}
        margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
      >
        <defs>
          <linearGradient id="liveEquityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={CHART_COLORS.strategy}
              stopOpacity={0.3}
            />
            <stop
              offset="100%"
              stopColor={CHART_COLORS.strategy}
              stopOpacity={0}
            />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={CHART_COLORS.grid}
          vertical={false}
        />
        <XAxis
          dataKey="timestamp"
          tickFormatter={formatAxisTime}
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={{ stroke: CHART_COLORS.grid }}
          tickLine={false}
          minTickGap={60}
        />
        <YAxis
          tickFormatter={(v: number) => formatCurrency(v)}
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={false}
          tickLine={false}
          width={80}
          domain={["auto", "auto"]}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="equity"
          stroke={CHART_COLORS.strategy}
          strokeWidth={2}
          fill="url(#liveEquityGradient)"
          dot={false}
          activeDot={{ r: 3, fill: CHART_COLORS.strategy }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
