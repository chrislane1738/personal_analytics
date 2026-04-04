"use client";

import { use, useMemo } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useCampaign } from "@/hooks/use-eval";
import { MetricsStrip } from "@/components/metrics-strip";
import { EvalFanChart } from "@/components/charts/eval-fan-chart";
import { RegimePassRate } from "@/components/charts/regime-pass-rate";
import { DaysDistribution } from "@/components/charts/days-distribution";
import { StateUsage } from "@/components/charts/state-usage";
import { formatCurrency, pnlColor } from "@/lib/format";

export default function CampaignDetailPage({
  params,
}: {
  params: Promise<{ campaignId: string }>;
}) {
  const { campaignId } = use(params);

  const {
    data: campaign,
    isLoading,
    error,
  } = useCampaign(campaignId);

  const isFunded = campaign?.mode === "funded";

  const metrics = useMemo(() => {
    if (!campaign) return [];

    if (isFunded) {
      return [
        {
          label: "Survival Rate",
          value: `${((campaign.survival_rate ?? 0) * 100).toFixed(1)}%`,
          colorClass:
            (campaign.survival_rate ?? 0) > 0.5
              ? "text-[#22c55e]"
              : (campaign.survival_rate ?? 0) >= 0.3
                ? "text-[#eab308]"
                : "text-[#ef4444]",
        },
        {
          label: "Avg Monthly P&L",
          value: formatCurrency(campaign.avg_monthly_pnl ?? 0),
          colorClass: pnlColor(campaign.avg_monthly_pnl ?? 0),
        },
        {
          label: "Sharpe Ratio",
          value: (campaign.sharpe_ratio ?? 0).toFixed(2),
          colorClass:
            (campaign.sharpe_ratio ?? 0) > 1
              ? "text-[#22c55e]"
              : (campaign.sharpe_ratio ?? 0) > 0.5
                ? "text-[#eab308]"
                : "text-[#ef4444]",
        },
        {
          label: "Annual Expected Income",
          value: formatCurrency(campaign.annual_expected_income ?? 0),
          colorClass: pnlColor(campaign.annual_expected_income ?? 0),
        },
      ];
    }

    return [
      {
        label: "Pass Rate",
        value: `${(campaign.pass_rate * 100).toFixed(1)}%`,
        colorClass:
          campaign.pass_rate > 0.3
            ? "text-[#22c55e]"
            : campaign.pass_rate >= 0.2
              ? "text-[#eab308]"
              : "text-[#ef4444]",
      },
      {
        label: "EV/Attempt",
        value: formatCurrency(campaign.ev_per_attempt),
        colorClass: pnlColor(campaign.ev_per_attempt),
      },
      {
        label: "Avg Days to Pass",
        value: campaign.avg_days_to_pass.toFixed(1),
      },
      {
        label: "Annual EV",
        value: formatCurrency(campaign.annual_ev),
        colorClass: pnlColor(campaign.annual_ev),
      },
    ];
  }, [campaign, isFunded]);

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4 p-6">
        {/* Header skeleton */}
        <div>
          <div className="h-3 w-20 rounded bg-zinc-800/60 animate-pulse mb-2" />
          <div className="h-7 w-48 rounded bg-zinc-800/60 animate-pulse" />
          <div className="mt-2 flex gap-3">
            <div className="h-3 w-16 rounded bg-zinc-800/40 animate-pulse" />
            <div className="h-3 w-32 rounded bg-zinc-800/40 animate-pulse" />
          </div>
        </div>
        {/* Metrics strip skeleton */}
        <div className="flex gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
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
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4"
            >
              <div className="h-3 w-28 rounded bg-zinc-800/40 animate-pulse mb-4" />
              <div className="h-[240px] rounded bg-zinc-800/20 animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !campaign) {
    return (
      <div className="p-6">
        <p className="text-sm text-[#ef4444]">
          {error ? `Error: ${error.message}` : "Campaign not found"}
        </p>
      </div>
    );
  }

  const results = campaign.full_results;

  return (
    <div className="flex flex-col gap-4 p-6">
      {/* Campaign Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            href="/eval"
            className="mb-2 inline-flex items-center gap-1 text-xs text-zinc-500 transition-colors hover:text-zinc-300"
          >
            <ArrowLeft className="h-3 w-3" />
            Back to Campaigns
          </Link>
          <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
            {campaign.strategy_name}
          </h1>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-zinc-500">
            <span className="font-mono">{campaign.campaign_id}</span>
            <span
              className={`inline-flex h-5 items-center rounded-full px-2 text-[10px] font-medium ${
                isFunded
                  ? "bg-blue-500/15 text-blue-400"
                  : "bg-zinc-800 text-zinc-400"
              }`}
            >
              {isFunded ? "FUNDED" : "EVAL"}
            </span>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[#fafafa]">
              {campaign.instrument}
            </span>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[#fafafa]">
              {campaign.timeframe === "1D"
                ? "Daily"
                : campaign.timeframe === "1m"
                  ? "1-Min"
                  : campaign.timeframe === "5m"
                    ? "5-Min"
                    : campaign.timeframe === "15m"
                      ? "15-Min"
                      : campaign.timeframe === "1h"
                        ? "Hourly"
                        : campaign.timeframe}
            </span>
            <span
              className={
                campaign.state_machine ? "text-[#22c55e]" : "text-zinc-500"
              }
            >
              State Machine: {campaign.state_machine ? "ON" : "OFF"}
            </span>
            <span>{campaign.num_attempts} {isFunded ? "simulations" : "attempts"}</span>
          </div>
        </div>
      </div>

      {/* Metrics Strip */}
      <MetricsStrip metrics={metrics} />

      {/* Funded-specific additional metrics */}
      {isFunded && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-3">
            <p className="text-[10px] uppercase tracking-wider text-zinc-500">Max Drawdown (Median)</p>
            <p className="mt-1 font-mono text-sm text-zinc-300">
              {formatCurrency(campaign.max_drawdown_median ?? 0)}
            </p>
          </div>
          <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-3">
            <p className="text-[10px] uppercase tracking-wider text-zinc-500">Avg Monthly Withdrawal</p>
            <p className={`mt-1 font-mono text-sm ${pnlColor(campaign.avg_monthly_withdrawal ?? 0)}`}>
              {formatCurrency(campaign.avg_monthly_withdrawal ?? 0)}
            </p>
          </div>
        </div>
      )}

      {/* 2x2 chart grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {isFunded ? (
          <>
            {/* Fan Chart -- reuse for funded equity curves */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Simulation Equity Fan Chart
              </h2>
              <div className="h-[280px]">
                {results?.attempt_outcomes ? (
                  <EvalFanChart attempts={results.attempt_outcomes} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No attempt data available
                  </div>
                )}
              </div>
            </div>

            {/* Survival Curve placeholder */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Survival Curve
              </h2>
              <div className="h-[280px]">
                {results?.survival_curve && results.survival_curve.length > 0 ? (
                  <div className="flex h-full flex-col justify-end px-2 pb-4">
                    <div className="flex h-full items-end gap-1">
                      {results.survival_curve.map((rate: number, i: number) => (
                        <div
                          key={i}
                          className="flex-1 rounded-t bg-blue-500/40 transition-all hover:bg-blue-500/60"
                          style={{ height: `${rate * 100}%` }}
                          title={`Month ${i + 1}: ${(rate * 100).toFixed(1)}%`}
                        />
                      ))}
                    </div>
                    <div className="mt-2 flex justify-between text-[10px] text-zinc-600">
                      <span>Month 1</span>
                      <span>Month {results.survival_curve.length}</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No survival data available
                  </div>
                )}
              </div>
            </div>

            {/* Days Distribution -- show funded sim days */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Days Survived Distribution
              </h2>
              <div className="h-[280px]">
                {results?.attempt_outcomes ? (
                  <DaysDistribution attempts={results.attempt_outcomes} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No distribution data available
                  </div>
                )}
              </div>
            </div>

            {/* Monthly PnL Distribution placeholder */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Monthly P&L Distribution
              </h2>
              <div className="h-[280px]">
                {results?.monthly_pnl_distribution &&
                results.monthly_pnl_distribution.length > 0 ? (
                  <div className="flex h-full flex-col justify-end px-2 pb-4">
                    {(() => {
                      const values: number[] = results.monthly_pnl_distribution;
                      const min = Math.min(...values);
                      const max = Math.max(...values);
                      const range = max - min || 1;
                      const bucketCount = 20;
                      const bucketSize = range / bucketCount;
                      const buckets = Array(bucketCount).fill(0);
                      for (const v of values) {
                        const idx = Math.min(
                          Math.floor((v - min) / bucketSize),
                          bucketCount - 1
                        );
                        buckets[idx]++;
                      }
                      const maxBucket = Math.max(...buckets);
                      return (
                        <>
                          <div className="flex h-full items-end gap-0.5">
                            {buckets.map((count: number, i: number) => {
                              const bucketMid = min + (i + 0.5) * bucketSize;
                              return (
                                <div
                                  key={i}
                                  className={`flex-1 rounded-t transition-all hover:opacity-80 ${
                                    bucketMid >= 0
                                      ? "bg-[#22c55e]/50"
                                      : "bg-[#ef4444]/50"
                                  }`}
                                  style={{
                                    height: `${(count / maxBucket) * 100}%`,
                                  }}
                                  title={`$${bucketMid.toFixed(0)}: ${count} months`}
                                />
                              );
                            })}
                          </div>
                          <div className="mt-2 flex justify-between text-[10px] text-zinc-600">
                            <span>{formatCurrency(min)}</span>
                            <span>{formatCurrency(max)}</span>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No monthly P&L data available
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <>
            {/* Fan Chart -- equity curves as percentile bands */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Attempt Equity Fan Chart
              </h2>
              <div className="h-[280px]">
                {results?.attempt_outcomes ? (
                  <EvalFanChart attempts={results.attempt_outcomes} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No attempt data available
                  </div>
                )}
              </div>
            </div>

            {/* Regime Pass Rate -- horizontal bars per regime */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Pass Rate by Regime
              </h2>
              <div className="h-[280px]">
                {results?.pass_by_regime ? (
                  <RegimePassRate data={results.pass_by_regime} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No regime data available
                  </div>
                )}
              </div>
            </div>

            {/* Days Distribution -- histogram (green=pass, red=fail) */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Days to Resolution
              </h2>
              <div className="h-[280px]">
                {results?.attempt_outcomes ? (
                  <DaysDistribution attempts={results.attempt_outcomes} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No distribution data available
                  </div>
                )}
              </div>
            </div>

            {/* State Usage -- horizontal bars per state */}
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                State Machine Usage
              </h2>
              <div className="h-[280px]">
                {results?.state_usage ? (
                  <StateUsage data={results.state_usage} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                    No state usage data available
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
