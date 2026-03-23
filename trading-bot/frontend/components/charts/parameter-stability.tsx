"use client";

import { useMemo } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { CHART_COLORS } from "./chart-theme";

/** Ordered palette for parameter bars */
const PARAM_COLORS = [
  CHART_COLORS.strategy,   // green
  CHART_COLORS.benchmark,  // blue
  CHART_COLORS.orange,     // orange
  CHART_COLORS.purple,     // purple
  CHART_COLORS.yellow,     // yellow
  CHART_COLORS.drawdown,   // red
];

interface ParameterStabilityProps {
  /** Map of parameter name to array of values (one per window) */
  data: Record<string, number[]>;
}

interface NormalizedRow {
  window: string;
  [paramName: string]: string | number;
}

/**
 * Normalize each parameter series to 0-1 range so they can be compared
 * on the same Y axis regardless of their original scale.
 */
function normalizeData(
  data: Record<string, number[]>
): { rows: NormalizedRow[]; paramNames: string[] } {
  const paramNames = Object.keys(data);
  if (paramNames.length === 0) return { rows: [], paramNames: [] };

  const windowCount = Math.max(...paramNames.map((k) => data[k].length));

  // Compute min/max per param for normalization
  const ranges: Record<string, { min: number; max: number }> = {};
  for (const name of paramNames) {
    const values = data[name];
    const min = Math.min(...values);
    const max = Math.max(...values);
    ranges[name] = { min, max };
  }

  const rows: NormalizedRow[] = [];
  for (let i = 0; i < windowCount; i++) {
    const row: NormalizedRow = { window: `W${i + 1}` };
    for (const name of paramNames) {
      const values = data[name];
      const raw = i < values.length ? values[i] : 0;
      const { min, max } = ranges[name];
      // If all values are the same, normalize to 0.5
      row[name] = max === min ? 0.5 : (raw - min) / (max - min);
    }
    rows.push(row);
  }
  return { rows, paramNames };
}

function StabilityTooltip({
  active,
  payload,
  label,
  rawData,
}: {
  active?: boolean;
  payload?: Array<{ dataKey: string; color: string; value: number }>;
  label?: string;
  rawData: Record<string, number[]>;
}) {
  if (!active || !payload || payload.length === 0) return null;

  // Extract window index from label (e.g. "W1" -> 0)
  const windowIdx = label ? parseInt(label.replace("W", ""), 10) - 1 : 0;

  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2 shadow-lg">
      <p className="mb-1 font-mono text-[11px] text-zinc-500">{label}</p>
      {payload.map((entry) => {
        const rawValues = rawData[entry.dataKey];
        const rawValue =
          rawValues && windowIdx < rawValues.length
            ? rawValues[windowIdx]
            : null;
        return (
          <p
            key={entry.dataKey}
            className="font-mono text-xs"
            style={{ color: entry.color }}
          >
            {entry.dataKey}: {rawValue != null ? rawValue.toFixed(4) : "-"}
          </p>
        );
      })}
    </div>
  );
}

export function ParameterStability({ data }: ParameterStabilityProps) {
  const { rows, paramNames } = useMemo(() => normalizeData(data), [data]);

  if (rows.length === 0 || paramNames.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No parameter stability data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={CHART_COLORS.grid}
          vertical={false}
        />
        <XAxis
          dataKey="window"
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={{ stroke: CHART_COLORS.grid }}
          tickLine={false}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => v.toFixed(1)}
          stroke={CHART_COLORS.text}
          tick={{ fontSize: 10, fill: CHART_COLORS.text }}
          axisLine={false}
          tickLine={false}
          width={30}
          label={{
            value: "Normalized",
            angle: -90,
            position: "insideLeft",
            style: { fontSize: 9, fill: CHART_COLORS.text },
          }}
        />
        <Tooltip
          content={
            <StabilityTooltip rawData={data} />
          }
        />
        <Legend
          wrapperStyle={{ fontSize: 10, color: CHART_COLORS.text }}
          iconSize={8}
        />
        {paramNames.map((name, i) => (
          <Bar
            key={name}
            dataKey={name}
            fill={PARAM_COLORS[i % PARAM_COLORS.length]}
            fillOpacity={0.8}
            radius={[2, 2, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
