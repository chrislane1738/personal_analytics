"use client";

import { use, useMemo } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useRun } from "@/hooks/use-runs";
import { useTrades } from "@/hooks/use-trades";
import { useEquityCurve, useRegimeStats, useMonteCarlo } from "@/hooks/use-analytics";
import { useOptionsAnalytics } from "@/hooks/use-options";
import { FanChart } from "@/components/charts/fan-chart";
import { OptionsAnalytics } from "@/components/options-analytics";
import { EquityCurve } from "@/components/charts/equity-curve";
import { DrawdownChart } from "@/components/charts/drawdown-chart";
import { MonthlyHeatmap } from "@/components/charts/monthly-heatmap";
import { MetricsStrip } from "@/components/metrics-strip";
import { TradesTable } from "@/components/tables/trades-table";
import { RegimeCards } from "@/components/regime-cards";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  formatCurrency,
  formatPercent,
  formatDate,
  pnlColor,
} from "@/lib/format";
import type { EquityCurvePoint } from "@/lib/types";

export default function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = use(params);

  const { data: run, isLoading: runLoading, error: runError } = useRun(runId);
  const { data: trades, isLoading: tradesLoading } = useTrades(runId);
  const { data: equityCurve, isLoading: curveLoading } = useEquityCurve(runId);
  const { data: regimeStats, isLoading: regimeLoading } = useRegimeStats(runId);
  const { data: mcData, isLoading: mcLoading } = useMonteCarlo(runId);
  const { data: optionsData, isLoading: optionsLoading } = useOptionsAnalytics(runId);

  // Compute drawdown series from equity curve
  const drawdownData = useMemo(() => {
    if (!equityCurve || equityCurve.length === 0) return [];
    let peak = equityCurve[0].strategy_value;
    return equityCurve.map((point: EquityCurvePoint) => {
      if (point.strategy_value > peak) peak = point.strategy_value;
      const drawdown = peak > 0 ? -((peak - point.strategy_value) / peak) : 0;
      return { date: point.date, drawdown };
    });
  }, [equityCurve]);

  // Build metrics from run data
  const metrics = useMemo(() => {
    if (!run) return [];
    const fm = (run.full_metrics ?? {}) as Record<string, unknown>;
    const num = (key: string, fallback?: number) => {
      const v = fm[key];
      return typeof v === "number" ? v : typeof v === "string" ? parseFloat(v) : (fallback ?? null);
    };
    return [
      {
        label: "Total Return",
        value: run.total_return != null ? formatPercent(run.total_return) : "-",
        colorClass: run.total_return != null ? pnlColor(run.total_return) : undefined,
      },
      {
        label: "Sharpe Ratio",
        value: run.sharpe != null ? run.sharpe.toFixed(2) : "-",
      },
      {
        label: "Sortino",
        value: num("sortino_ratio") != null ? num("sortino_ratio")!.toFixed(2) : "-",
      },
      {
        label: "Max Drawdown",
        value: run.max_drawdown != null ? formatPercent(run.max_drawdown) : "-",
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
      {
        label: "Strategy",
        value: run.strategy_name ?? "-",
      },
    ];
  }, [run]);

  // Risk metrics for the side panel
  const riskMetrics = useMemo(() => {
    if (!equityCurve || equityCurve.length === 0 || !run) return [];
    const lastPoint = equityCurve[equityCurve.length - 1];
    const maxDd = drawdownData.reduce(
      (min: number, p: { drawdown: number }) => Math.min(min, p.drawdown),
      0
    );
    return [
      {
        label: "Final Equity",
        value: formatCurrency(lastPoint.strategy_value ?? 0),
      },
      {
        label: "Peak Drawdown",
        value: `${(maxDd * 100).toFixed(2)}%`,
        colorClass: "text-[#ef4444]",
      },
      {
        label: "Initial Capital",
        value: formatCurrency(run.initial_capital ?? 0),
      },
      {
        label: "Benchmark",
        value: formatCurrency(lastPoint.benchmark_value ?? 0),
      },
    ];
  }, [equityCurve, drawdownData, run]);

  if (runLoading) {
    return (
      <div className="flex flex-col gap-4 p-6">
        {/* Header skeleton */}
        <div>
          <div className="h-3 w-20 rounded bg-zinc-800/60 animate-pulse mb-2" />
          <div className="h-7 w-48 rounded bg-zinc-800/60 animate-pulse" />
          <div className="mt-2 flex gap-3">
            <div className="h-3 w-16 rounded bg-zinc-800/40 animate-pulse" />
            <div className="h-3 w-32 rounded bg-zinc-800/40 animate-pulse" />
            <div className="h-3 w-20 rounded bg-zinc-800/40 animate-pulse" />
          </div>
        </div>
        {/* Metrics strip skeleton */}
        <div className="flex gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="flex-1 rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-3">
              <div className="h-2 w-12 rounded bg-zinc-800/40 animate-pulse mb-2" />
              <div className="h-5 w-16 rounded bg-zinc-800/60 animate-pulse" />
            </div>
          ))}
        </div>
        {/* Chart skeletons */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4 lg:col-span-2">
            <div className="h-3 w-20 rounded bg-zinc-800/40 animate-pulse mb-4" />
            <div className="h-[300px] rounded bg-zinc-800/20 animate-pulse" />
          </div>
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
            <div className="h-3 w-28 rounded bg-zinc-800/40 animate-pulse mb-4" />
            <div className="space-y-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-16 rounded bg-zinc-800/20 animate-pulse" />
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (runError || !run) {
    return (
      <div className="p-6">
        <p className="text-sm text-[#ef4444]">
          {runError ? `Error: ${runError.message}` : "Run not found"}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 p-6">
      {/* Run Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            href="/"
            className="mb-2 inline-flex items-center gap-1 text-xs text-zinc-500 transition-colors hover:text-zinc-300"
          >
            <ArrowLeft className="h-3 w-3" />
            Back to Runs
          </Link>
          <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
            {run.strategy_name}
          </h1>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-zinc-500">
            <span className="font-mono">{run.run_id}</span>
            <span>
              {formatDate(run.start_date)} &mdash; {formatDate(run.end_date)}
            </span>
            <span>{formatCurrency(run.initial_capital)}</span>
          </div>
        </div>
      </div>

      {/* Metrics Strip */}
      <MetricsStrip metrics={metrics} />

      {/* Main content grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Equity Curve — takes 2/3 */}
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4 lg:col-span-2">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Equity Curve
          </h2>
          <div className="h-[300px]">
            {curveLoading ? (
              <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                Loading chart...
              </div>
            ) : (
              <EquityCurve data={equityCurve ?? []} />
            )}
          </div>
        </div>

        {/* Regime Cards — takes 1/3 */}
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Regime Analysis
          </h2>
          {regimeLoading ? (
            <p className="text-sm text-zinc-500">Loading...</p>
          ) : (
            <RegimeCards data={regimeStats ?? []} />
          )}
        </div>
      </div>

      {/* Bottom row */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Trade Log — takes 2/3 */}
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4 lg:col-span-2">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Trade Log
          </h2>
          {tradesLoading ? (
            <p className="text-sm text-zinc-500">Loading trades...</p>
          ) : (
            <TradesTable data={trades ?? []} />
          )}
        </div>

        {/* Drawdown + Risk Metrics — takes 1/3 */}
        <div className="flex flex-col gap-4">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Drawdown
            </h2>
            <div className="h-[180px]">
              {curveLoading ? (
                <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                  Loading chart...
                </div>
              ) : (
                <DrawdownChart data={drawdownData} />
              )}
            </div>
          </div>

          <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Risk Metrics
            </h2>
            <div className="flex flex-col gap-2">
              {riskMetrics.map((m) => (
                <div
                  key={m.label}
                  className="flex items-baseline justify-between"
                >
                  <span className="text-[10px] uppercase tracking-wider text-zinc-500">
                    {m.label}
                  </span>
                  <span
                    className={`font-mono text-sm font-semibold ${m.colorClass ?? "text-zinc-200"}`}
                  >
                    {m.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Tabs */}
      <Tabs defaultValue="monthly-returns">
        <TabsList variant="line">
          <TabsTrigger value="monthly-returns">Monthly Returns</TabsTrigger>
          <TabsTrigger value="monte-carlo">Monte Carlo</TabsTrigger>
          <TabsTrigger value="options">Options</TabsTrigger>
        </TabsList>

        <TabsContent value="monthly-returns">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
            {curveLoading ? (
              <p className="text-sm text-zinc-500">Loading...</p>
            ) : (
              <MonthlyHeatmap data={equityCurve ?? []} />
            )}
          </div>
        </TabsContent>

        <TabsContent value="monte-carlo">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
            {mcLoading ? (
              <p className="text-sm text-zinc-500">Running Monte Carlo simulation...</p>
            ) : mcData ? (
              <FanChart data={mcData} />
            ) : (
              <p className="text-sm text-zinc-500">No trade data available for simulation</p>
            )}
          </div>
        </TabsContent>

        <TabsContent value="options">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
            {optionsLoading ? (
              <p className="text-sm text-zinc-500">Loading options analytics...</p>
            ) : optionsData ? (
              <OptionsAnalytics data={optionsData} />
            ) : (
              <p className="text-sm text-zinc-500">No options trades in this run</p>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
