"use client";

import type { RegimeStat } from "@/lib/types";
import { REGIME_COLORS } from "@/components/charts/chart-theme";
import { formatPercent, formatCurrency } from "@/lib/format";

interface RegimeCardsProps {
  data: RegimeStat[];
}

function regimeLabel(regime: string): string {
  return regime
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function RegimeCards({ data }: RegimeCardsProps) {
  if (data.length === 0) {
    return (
      <p className="text-sm text-zinc-500">No regime data available</p>
    );
  }

  return (
    <div className="grid gap-2">
      {data.map((stat) => {
        const color = REGIME_COLORS[stat.regime] ?? "#71717a";
        return (
          <div
            key={stat.regime}
            className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2.5"
            style={{ borderLeftColor: color, borderLeftWidth: 3 }}
          >
            {/* Header */}
            <div className="mb-2 flex items-center gap-2">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs font-semibold text-zinc-200">
                {regimeLabel(stat.regime)}
              </span>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <StatRow label="Trades" value={String(stat.trade_count)} />
              <StatRow label="Win Rate" value={formatPercent(stat.win_rate)} />
              <StatRow label="Avg P&L" value={formatCurrency(stat.avg_pnl)} />
              <StatRow
                label="Sharpe"
                value={stat.sharpe_ratio.toFixed(2)}
              />
              <StatRow
                label="Return"
                value={formatPercent(stat.total_return)}
              />
              <StatRow
                label="Max DD"
                value={formatPercent(stat.max_drawdown)}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-[9px] uppercase tracking-wider text-zinc-500">
        {label}
      </span>
      <span className="font-mono text-xs text-zinc-300">{value}</span>
    </div>
  );
}
