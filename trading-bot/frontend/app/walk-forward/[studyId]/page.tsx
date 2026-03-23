"use client";

import { use, useMemo } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useWalkForwardStudy } from "@/hooks/use-walk-forward";
import { MetricsStrip } from "@/components/metrics-strip";
import { EquityCurve } from "@/components/charts/equity-curve";
import { ParameterStability } from "@/components/charts/parameter-stability";
import { FanChart } from "@/components/charts/fan-chart";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  formatPercent,
  formatDate,
  pnlColor,
} from "@/lib/format";
import type { WalkForwardStudy, WalkForwardWindow } from "@/lib/types";

/** Build aggregate OOS metrics for the MetricsStrip */
function buildAggregateMetrics(study: WalkForwardStudy) {
  const agg = study.aggregate ?? {};
  const num = (key: string) => {
    const v = agg[key];
    return typeof v === "number" ? v : null;
  };

  return [
    {
      label: "OOS Sharpe",
      value: num("sharpe_ratio") != null ? num("sharpe_ratio")!.toFixed(2) : "-",
    },
    {
      label: "OOS Return",
      value: num("total_return") != null ? formatPercent(num("total_return")!) : "-",
      colorClass: num("total_return") != null ? pnlColor(num("total_return")!) : undefined,
    },
    {
      label: "Max Drawdown",
      value: num("max_drawdown") != null ? formatPercent(num("max_drawdown")!) : "-",
      colorClass: "text-[#ef4444]",
    },
    {
      label: "Win Rate",
      value: num("win_rate") != null ? formatPercent(num("win_rate")!) : "-",
    },
    {
      label: "Profit Factor",
      value: num("profit_factor") != null ? num("profit_factor")!.toFixed(2) : "-",
    },
    {
      label: "Total Trades",
      value: num("total_trades") != null ? String(Math.round(num("total_trades")!)) : "-",
    },
  ];
}

/** Format window config string like "24mo train / 6mo OOS / 3mo step" */
function formatWindowConfig(study: WalkForwardStudy): string {
  return `${study.train_months}mo train / ${study.oos_months}mo OOS / ${study.step_months}mo step`;
}

/** Truncate JSON string for display */
function truncateParams(params: Record<string, number>, maxLen = 40): string {
  const str = JSON.stringify(params);
  return str.length > maxLen ? str.slice(0, maxLen) + "\u2026" : str;
}

/** Get the Monte Carlo mode keys, sorted for consistent tab order */
const MC_MODE_LABELS: Record<string, string> = {
  trade_shuffle: "Trade Shuffle",
  window_shuffle: "Window Shuffle",
  bootstrap: "Bootstrap",
};

export default function WalkForwardStudyDetailPage({
  params,
}: {
  params: Promise<{ studyId: string }>;
}) {
  const { studyId } = use(params);

  const {
    data: study,
    isLoading,
    error,
  } = useWalkForwardStudy(studyId);

  const equityData = study?.stitched_equity_curve ?? [];

  const metrics = useMemo(
    () => (study ? buildAggregateMetrics(study) : []),
    [study]
  );

  const mcModes = useMemo(() => {
    if (!study?.monte_carlo) return [];
    return Object.keys(MC_MODE_LABELS).filter(
      (mode) => mode in study.monte_carlo
    );
  }, [study]);

  // ---------- Loading state ----------
  if (isLoading) {
    return (
      <div className="flex flex-col gap-4 p-6">
        {/* Header skeleton */}
        <div>
          <div className="h-3 w-20 rounded bg-zinc-800/60 animate-pulse mb-2" />
          <div className="h-7 w-64 rounded bg-zinc-800/60 animate-pulse" />
          <div className="mt-2 flex gap-3">
            <div className="h-3 w-32 rounded bg-zinc-800/40 animate-pulse" />
            <div className="h-3 w-48 rounded bg-zinc-800/40 animate-pulse" />
          </div>
        </div>
        {/* Metrics strip skeleton */}
        <div className="flex gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="flex-1 rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-3"
            >
              <div className="h-2 w-12 rounded bg-zinc-800/40 animate-pulse mb-2" />
              <div className="h-5 w-16 rounded bg-zinc-800/60 animate-pulse" />
            </div>
          ))}
        </div>
        {/* Chart skeletons */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4 lg:col-span-2">
            <div className="h-3 w-32 rounded bg-zinc-800/40 animate-pulse mb-4" />
            <div className="h-[300px] rounded bg-zinc-800/20 animate-pulse" />
          </div>
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
            <div className="h-3 w-28 rounded bg-zinc-800/40 animate-pulse mb-4" />
            <div className="h-[300px] rounded bg-zinc-800/20 animate-pulse" />
          </div>
        </div>
      </div>
    );
  }

  // ---------- Error state ----------
  if (error || !study) {
    return (
      <div className="p-6">
        <p className="text-sm text-[#ef4444]">
          {error ? `Error: ${error.message}` : "Study not found"}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 p-6">
      {/* ─── Header ─── */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            href="/walk-forward"
            className="mb-2 inline-flex items-center gap-1 text-xs text-zinc-500 transition-colors hover:text-zinc-300"
          >
            <ArrowLeft className="h-3 w-3" />
            Back to Walk-Forward Studies
          </Link>
          <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
            {study.strategy_name}
          </h1>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-zinc-500">
            <span>
              {formatDate(study.start_date)} &mdash;{" "}
              {formatDate(study.end_date)}
            </span>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px]">
              {formatWindowConfig(study)}
            </span>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px]">
              obj: {study.objective}
            </span>
          </div>
        </div>
      </div>

      {/* ─── Aggregate OOS Metrics ─── */}
      <MetricsStrip metrics={metrics} />

      {/* ─── Charts Row: Stitched Equity (2/3) + Parameter Stability (1/3) ─── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4 lg:col-span-2">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Stitched OOS Equity Curve
          </h2>
          <div className="h-[300px]">
            {equityData.length > 0 ? (
              <EquityCurve data={equityData} />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                No equity data available
              </div>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Parameter Stability
          </h2>
          <div className="h-[300px]">
            <ParameterStability
              data={study.parameter_stability ?? {}}
            />
          </div>
        </div>
      </div>

      {/* ─── Per-Window Results Table ─── */}
      <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Per-Window Results
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#1a1a1a] text-[10px] uppercase tracking-wider text-zinc-500">
                <th className="px-3 py-2 font-medium">Win #</th>
                <th className="px-3 py-2 font-medium">Train Period</th>
                <th className="px-3 py-2 font-medium">OOS Period</th>
                <th className="px-3 py-2 font-medium">Best Params</th>
                <th className="px-3 py-2 font-medium text-right">
                  OOS Return
                </th>
                <th className="px-3 py-2 font-medium text-right">
                  OOS Sharpe
                </th>
                <th className="px-3 py-2 font-medium text-right">
                  OOS Trades
                </th>
              </tr>
            </thead>
            <tbody>
              {(study.windows ?? []).map((w: WalkForwardWindow) => {
                const oosReturn = w.oos_metrics?.total_return ?? null;
                const oosSharpe = w.oos_metrics?.sharpe_ratio ?? null;
                const oosTrades = w.oos_metrics?.total_trades ?? w.oos_trades?.length ?? null;

                return (
                  <tr
                    key={w.window_index}
                    className="border-b border-[#1a1a1a]/50 transition-colors hover:bg-zinc-900/30"
                  >
                    <td className="px-3 py-2 font-mono text-zinc-300">
                      {w.window_index + 1}
                    </td>
                    <td className="px-3 py-2 text-zinc-400">
                      {formatDate(w.train_start)} &mdash;{" "}
                      {formatDate(w.train_end)}
                    </td>
                    <td className="px-3 py-2 text-zinc-400">
                      {formatDate(w.oos_start)} &mdash;{" "}
                      {formatDate(w.oos_end)}
                    </td>
                    <td className="max-w-[200px] truncate px-3 py-2 font-mono text-[10px] text-zinc-500">
                      <span title={JSON.stringify(w.best_params)}>
                        {truncateParams(w.best_params)}
                      </span>
                    </td>
                    <td
                      className={`px-3 py-2 text-right font-mono font-semibold ${
                        oosReturn != null ? pnlColor(oosReturn) : "text-zinc-500"
                      }`}
                    >
                      {oosReturn != null ? formatPercent(oosReturn) : "-"}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-zinc-300">
                      {oosSharpe != null ? oosSharpe.toFixed(2) : "-"}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-zinc-300">
                      {oosTrades != null ? Math.round(oosTrades) : "-"}
                    </td>
                  </tr>
                );
              })}
              {(!study.windows || study.windows.length === 0) && (
                <tr>
                  <td
                    colSpan={7}
                    className="px-3 py-6 text-center text-zinc-500"
                  >
                    No window results available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ─── Monte Carlo Tabs ─── */}
      {mcModes.length > 0 ? (
        <Tabs defaultValue={mcModes[0]}>
          <TabsList variant="line">
            {Object.keys(MC_MODE_LABELS).map((mode) => (
              <TabsTrigger key={mode} value={mode}>
                {MC_MODE_LABELS[mode]}
                {!(mode in study.monte_carlo) && (
                  <span className="ml-1 text-[9px] text-zinc-600">
                    (not run)
                  </span>
                )}
              </TabsTrigger>
            ))}
          </TabsList>

          {Object.keys(MC_MODE_LABELS).map((mode) => (
            <TabsContent key={mode} value={mode}>
              <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
                {mode in study.monte_carlo ? (
                  <FanChart data={study.monte_carlo[mode]} />
                ) : (
                  <p className="py-8 text-center text-sm text-zinc-500">
                    {MC_MODE_LABELS[mode]} simulation was not run for this study
                  </p>
                )}
              </div>
            </TabsContent>
          ))}
        </Tabs>
      ) : (
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Monte Carlo Analysis
          </h2>
          <p className="text-sm text-zinc-500">
            No Monte Carlo simulations were run for this study
          </p>
        </div>
      )}
    </div>
  );
}
