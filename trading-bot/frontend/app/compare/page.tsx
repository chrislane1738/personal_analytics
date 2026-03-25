"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { ArrowLeft } from "lucide-react";
import { useRuns } from "@/hooks/use-runs";
import { useEquityCurve } from "@/hooks/use-analytics";
import { formatPercent, pnlColor } from "@/lib/format";
import { CHART_COLORS } from "@/components/charts/chart-theme";
import type { Run, EquityCurvePoint } from "@/lib/types";

const RUN_COLORS = [
  CHART_COLORS.strategy,
  CHART_COLORS.benchmark,
  CHART_COLORS.orange,
  CHART_COLORS.purple,
];

/** Metrics we compare across runs */
const METRIC_DEFS: {
  key: keyof Run;
  label: string;
  format: (v: number) => string;
  higherIsBetter: boolean;
}[] = [
  {
    key: "total_return",
    label: "Total Return",
    format: (v) => formatPercent(v),
    higherIsBetter: true,
  },
  {
    key: "sharpe_ratio",
    label: "Sharpe Ratio",
    format: (v) => v.toFixed(2),
    higherIsBetter: true,
  },
  {
    key: "max_drawdown",
    label: "Max Drawdown",
    format: (v) => formatPercent(v),
    higherIsBetter: false,
  },
  {
    key: "win_rate",
    label: "Win Rate",
    format: (v) => formatPercent(v),
    higherIsBetter: true,
  },
];

function formatAxisDate(timestamp: string): string {
  const d = new Date(timestamp);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/** A single run selector row with toggle */
function RunSelector({
  run,
  selected,
  color,
  onToggle,
  disabled,
}: {
  run: Run;
  selected: boolean;
  color: string | undefined;
  onToggle: () => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled && !selected}
      className={`flex w-full items-center gap-3 rounded-md border px-3 py-2 text-left text-sm transition-colors ${
        selected
          ? "border-zinc-600 bg-[#1a1a1a]"
          : disabled
            ? "cursor-not-allowed border-[#1a1a1a] bg-[#0a0a0a] opacity-40"
            : "border-[#1a1a1a] bg-[#0a0a0a] hover:border-zinc-700 hover:bg-[#111]"
      }`}
    >
      <span
        className="h-3 w-3 shrink-0 rounded-full"
        style={{
          backgroundColor: selected ? color : "#333",
        }}
      />
      <span className="truncate font-medium text-zinc-300">
        {run.strategy}
      </span>
      <span className="shrink-0 font-mono text-[10px] text-zinc-600">
        {run.run_id.slice(0, 8)}
      </span>
    </button>
  );
}

/** Hook to get equity curve for a run (only when selected) */
function useConditionalEquityCurve(runId: string | undefined) {
  return useEquityCurve(runId);
}

/**
 * Custom tooltip for the comparison chart
 */
function ComparisonTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{
    value: number;
    dataKey: string;
    color: string;
    name: string;
  }>;
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
          {entry.name}: {(entry.value * 100).toFixed(2)}%
        </p>
      ))}
    </div>
  );
}

export default function ComparePage() {
  const { data: runsData, isLoading: runsLoading } = useRuns({
    page_size: 100,
  });
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // Fetch equity curves for all selected runs (up to 4)
  const curve0 = useConditionalEquityCurve(selectedIds[0]);
  const curve1 = useConditionalEquityCurve(selectedIds[1]);
  const curve2 = useConditionalEquityCurve(selectedIds[2]);
  const curve3 = useConditionalEquityCurve(selectedIds[3]);

  const curveQueries = [curve0, curve1, curve2, curve3];

  const runs = runsData?.runs ?? [];

  const toggleRun = (runId: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(runId)) {
        return prev.filter((id) => id !== runId);
      }
      if (prev.length >= 4) return prev;
      return [...prev, runId];
    });
  };

  // Build a color map for selected runs
  const colorMap = useMemo(() => {
    const map = new Map<string, string>();
    selectedIds.forEach((id, i) => {
      map.set(id, RUN_COLORS[i]);
    });
    return map;
  }, [selectedIds]);

  // Normalize equity curves to % return and merge into unified time series
  const chartData = useMemo(() => {
    if (selectedIds.length === 0) return [];

    // Collect all curves with their initial equity
    const curves: { id: string; data: EquityCurvePoint[]; initial: number }[] =
      [];
    for (let i = 0; i < selectedIds.length; i++) {
      const curveData = curveQueries[i]?.data;
      if (curveData && curveData.length > 0) {
        curves.push({
          id: selectedIds[i],
          data: curveData,
          initial: curveData[0].strategy_value,
        });
      }
    }

    if (curves.length === 0) return [];

    // Build a union of all timestamps, then for each timestamp fill in the
    // normalized return for each run that has data at that point.
    // Since curves may have different timestamps, we index by timestamp string.
    const timestampSet = new Set<string>();
    for (const curve of curves) {
      for (const point of curve.data) {
        timestampSet.add(point.date);
      }
    }

    const sortedTimestamps = Array.from(timestampSet).sort();

    // Build lookup maps for each curve: timestamp -> equity
    const lookups = curves.map((curve) => {
      const map = new Map<string, number>();
      for (const point of curve.data) {
        map.set(point.date, point.strategy_value);
      }
      return { id: curve.id, map, initial: curve.initial, lastValue: 0 };
    });

    // For each timestamp, compute normalized returns and carry forward missing values
    return sortedTimestamps.map((ts) => {
      const row: Record<string, string | number> = { timestamp: ts };
      for (const lookup of lookups) {
        const equity = lookup.map.get(ts);
        if (equity !== undefined) {
          lookup.lastValue = equity;
          row[lookup.id] =
            lookup.initial > 0 ? equity / lookup.initial - 1 : 0;
        } else if (lookup.lastValue > 0) {
          // Carry forward
          row[lookup.id] =
            lookup.initial > 0 ? lookup.lastValue / lookup.initial - 1 : 0;
        }
      }
      return row;
    });
  }, [selectedIds, curveQueries]);

  // Selected run objects for metrics table
  const selectedRuns = useMemo(() => {
    return selectedIds
      .map((id) => runs.find((r) => r.id === id))
      .filter((r): r is Run => r !== undefined);
  }, [selectedIds, runs]);

  // For each metric, find best and worst among selected runs
  const metricHighlights = useMemo(() => {
    if (selectedRuns.length < 2) return new Map<string, { best: string; worst: string }>();
    const highlights = new Map<string, { best: string; worst: string }>();

    for (const def of METRIC_DEFS) {
      const values = selectedRuns
        .map((r) => ({
          id: r.id,
          value: r[def.key] as number | null,
        }))
        .filter((v) => v.value !== null) as { id: string; value: number }[];

      if (values.length < 2) continue;

      const sorted = [...values].sort((a, b) => a.value - b.value);
      const best = def.higherIsBetter
        ? sorted[sorted.length - 1].id
        : sorted[0].id;
      const worst = def.higherIsBetter
        ? sorted[0].id
        : sorted[sorted.length - 1].id;

      highlights.set(def.key, { best, worst });
    }

    return highlights;
  }, [selectedRuns]);

  const anyLoading = selectedIds.some(
    (_, i) => curveQueries[i]?.isLoading
  );

  return (
    <div className="flex flex-col gap-4 p-6">
      {/* Header */}
      <div>
        <Link
          href="/"
          className="mb-2 inline-flex items-center gap-1 text-xs text-zinc-500 transition-colors hover:text-zinc-300"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to Runs
        </Link>
        <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
          Run Comparison
        </h1>
        <p className="mt-1 text-xs text-zinc-500">
          Select 2-4 runs to compare side-by-side
        </p>
      </div>

      {/* Run Selector */}
      <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Select Runs ({selectedIds.length}/4)
        </h2>
        {runsLoading ? (
          <p className="text-sm text-zinc-500">Loading runs...</p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-zinc-500">No runs available</p>
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {runs.map((run) => (
              <RunSelector
                key={run.run_id}
                run={run}
                selected={selectedIds.includes(run.run_id)}
                color={colorMap.get(run.run_id)}
                onToggle={() => toggleRun(run.run_id)}
                disabled={selectedIds.length >= 4}
              />
            ))}
          </div>
        )}
      </div>

      {/* Equity Curve Comparison Chart */}
      {selectedIds.length >= 2 && (
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Normalized Equity Curves (% Return)
          </h2>
          <div className="h-[400px]">
            {anyLoading ? (
              <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                Loading curves...
              </div>
            ) : chartData.length === 0 ? (
              <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                No equity data available
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
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
                    tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                    stroke={CHART_COLORS.text}
                    tick={{ fontSize: 10, fill: CHART_COLORS.text }}
                    axisLine={false}
                    tickLine={false}
                    width={64}
                  />
                  <Tooltip content={<ComparisonTooltip />} />
                  <Legend
                    wrapperStyle={{
                      fontSize: "11px",
                      color: CHART_COLORS.text,
                    }}
                  />
                  {selectedIds.map((id, i) => {
                    const run = runs.find((r) => r.id === id);
                    const name = run
                      ? `${run.strategy} (${id.slice(0, 8)})`
                      : id.slice(0, 8);
                    return (
                      <Line
                        key={id}
                        type="monotone"
                        dataKey={id}
                        name={name}
                        stroke={RUN_COLORS[i]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 3, fill: RUN_COLORS[i] }}
                        connectNulls
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {/* Metrics Comparison Table */}
      {selectedRuns.length >= 2 && (
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Metrics Comparison
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                    Metric
                  </th>
                  {selectedRuns.map((run, i) => (
                    <th
                      key={run.run_id}
                      className="px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-wider"
                      style={{ color: RUN_COLORS[i] }}
                    >
                      {run.strategy}
                      <span className="ml-1 text-zinc-600">
                        ({run.run_id.slice(0, 8)})
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRIC_DEFS.map((def) => {
                  const highlight = metricHighlights.get(def.key);
                  return (
                    <tr
                      key={def.key}
                      className="border-t border-[#1a1a1a]"
                    >
                      <td className="px-3 py-2 text-xs text-zinc-400">
                        {def.label}
                      </td>
                      {selectedRuns.map((run) => {
                        const value = run[def.key] as number | null;
                        const isBest = highlight?.best === run.run_id;
                        const isWorst = highlight?.worst === run.run_id;

                        let cellClass =
                          "px-3 py-2 text-right font-mono text-xs ";
                        if (isBest) cellClass += "text-[#22c55e] bg-[#22c55e]/10";
                        else if (isWorst) cellClass += "text-[#ef4444] bg-[#ef4444]/10";
                        else cellClass += "text-zinc-300";

                        return (
                          <td key={run.run_id} className={cellClass}>
                            {value !== null ? def.format(value) : "-"}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Prompt when fewer than 2 selected */}
      {selectedIds.length < 2 && (
        <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-zinc-800 text-sm text-zinc-600">
          Select at least 2 runs above to see comparison charts and metrics
        </div>
      )}
    </div>
  );
}
