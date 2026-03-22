"use client";

import type { PortfolioState } from "@/hooks/use-websocket";
import { formatCurrency, formatPercent, pnlColor } from "@/lib/format";

interface PortfolioPanelProps {
  state: PortfolioState | null;
}

function MetricRow({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`font-mono text-sm ${className ?? "text-zinc-200"}`}>
        {value}
      </span>
    </div>
  );
}

export function PortfolioPanel({ state }: PortfolioPanelProps) {
  if (!state) {
    return (
      <div className="flex h-24 items-center justify-center text-sm text-zinc-600">
        Waiting for portfolio update...
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <MetricRow
        label="Equity"
        value={formatCurrency(state.equity)}
        className="text-[#fafafa] font-medium"
      />
      <MetricRow
        label="Cash"
        value={formatCurrency(state.cash)}
      />
      <MetricRow
        label="Unrealized P&L"
        value={formatCurrency(state.unrealized_pnl)}
        className={pnlColor(state.unrealized_pnl)}
      />
      <MetricRow
        label="Realized P&L"
        value={formatCurrency(state.realized_pnl)}
        className={pnlColor(state.realized_pnl)}
      />
      <MetricRow
        label="Drawdown"
        value={formatPercent(state.drawdown_pct)}
        className={state.drawdown_pct < 0 ? "text-[#ef4444]" : "text-zinc-400"}
      />
    </div>
  );
}
