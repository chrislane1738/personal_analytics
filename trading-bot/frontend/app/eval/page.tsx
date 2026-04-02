"use client";

import { useMemo } from "react";
import { useCampaigns, useDeleteCampaign } from "@/hooks/use-eval";
import { CampaignsTable } from "@/components/tables/campaigns-table";
import { MetricsStrip } from "@/components/metrics-strip";
import { formatCurrency } from "@/lib/format";
import { pnlColor } from "@/lib/format";
import type { Campaign } from "@/lib/types";

function SkeletonRows() {
  return (
    <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-1">
      {/* Header skeleton */}
      <div className="flex gap-4 border-b border-[#1a1a1a] px-4 py-3">
        {Array.from({ length: 9 }).map((_, i) => (
          <div
            key={i}
            className="h-3 w-16 rounded bg-zinc-800/60 animate-pulse"
          />
        ))}
      </div>
      {/* Row skeletons */}
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="flex gap-4 border-b border-[#1a1a1a]/50 px-4 py-3 last:border-0"
        >
          {Array.from({ length: 9 }).map((_, j) => (
            <div
              key={j}
              className="h-4 rounded bg-zinc-800/40 animate-pulse"
              style={{ width: `${60 + ((j * 17 + i * 31) % 40)}px` }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

/**
 * Return pass rate color class: green >30%, yellow 20-30%, red <20%.
 */
function passRateColorClass(value: number): string {
  if (value > 0.3) return "text-[#22c55e]";
  if (value >= 0.2) return "text-[#eab308]";
  return "text-[#ef4444]";
}

function buildKpiMetrics(campaigns: Campaign[]) {
  if (campaigns.length === 0) return [];

  const bestPassRate = Math.max(...campaigns.map((c) => c.pass_rate));
  const bestEv = Math.max(...campaigns.map((c) => c.ev_per_attempt));
  const avgCost =
    campaigns.reduce((sum, c) => sum + c.cost_to_funded, 0) / campaigns.length;
  const avgDays =
    campaigns.reduce((sum, c) => sum + c.avg_days_to_pass, 0) /
    campaigns.length;
  const bestAnnualEv = Math.max(...campaigns.map((c) => c.annual_ev));

  return [
    {
      label: "Best Pass Rate",
      value: `${(bestPassRate * 100).toFixed(1)}%`,
      colorClass: passRateColorClass(bestPassRate),
    },
    {
      label: "Best EV/Attempt",
      value: formatCurrency(bestEv),
      colorClass: pnlColor(bestEv),
    },
    {
      label: "Avg Cost to Fund",
      value: formatCurrency(avgCost),
    },
    {
      label: "Avg Days to Pass",
      value: avgDays.toFixed(1),
    },
    {
      label: "Best Annual EV",
      value: formatCurrency(bestAnnualEv),
      colorClass: pnlColor(bestAnnualEv),
    },
    {
      label: "Campaigns Run",
      value: String(campaigns.length),
    },
  ];
}

export default function EvalPage() {
  const { data, isLoading, error } = useCampaigns();
  const deleteCampaign = useDeleteCampaign();

  const kpiMetrics = useMemo(
    () => buildKpiMetrics(data?.campaigns ?? []),
    [data]
  );

  return (
    <div className="p-6">
      <div className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
          Campaign Browser
        </h1>
        {data && (
          <span className="text-sm text-zinc-500">
            {data.total} {data.total === 1 ? "campaign" : "campaigns"}
          </span>
        )}
      </div>

      {/* KPI Metrics Strip */}
      {data && data.campaigns.length > 0 && (
        <div className="mb-4">
          <MetricsStrip metrics={kpiMetrics} />
        </div>
      )}

      {isLoading && <SkeletonRows />}

      {error && (
        <p className="text-sm text-red-500">
          Failed to load campaigns: {error.message}
        </p>
      )}

      {data && (
        <CampaignsTable
          data={data.campaigns}
          onDelete={(campaignId) => deleteCampaign.mutate(campaignId)}
        />
      )}
    </div>
  );
}
