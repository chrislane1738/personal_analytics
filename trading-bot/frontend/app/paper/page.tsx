"use client";

import { DollarSign, Play, Square, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function PaperTradingPage() {
  return (
    <div className="p-6 flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <DollarSign className="h-5 w-5 text-[#f97316]" />
        <div>
          <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
            Paper Trading
          </h1>
          <p className="text-sm text-zinc-500">Coming Soon</p>
        </div>
      </div>

      {/* Description card */}
      <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-6 max-w-2xl">
        <div className="flex items-start gap-4">
          <div className="mt-1 rounded-full bg-[#f97316]/10 p-2">
            <Lock className="h-5 w-5 text-[#f97316]" />
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-medium text-zinc-100">
              Schwab API Integration
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-zinc-400">
              Paper trading will allow you to test strategies in real-time
              with simulated orders through the Charles Schwab API. Connect
              your account, select a strategy, and monitor live paper
              trades with real market data — all without risking capital.
            </p>
            <ul className="mt-4 space-y-2 text-sm text-zinc-500">
              <li className="flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-[#f97316]" />
                Real-time market data streaming
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-[#f97316]" />
                Simulated order execution with realistic fills
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-[#f97316]" />
                Live portfolio tracking and P&L monitoring
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-[#f97316]" />
                Strategy performance comparison vs. backtest results
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Disabled controls */}
      <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-6 max-w-2xl">
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Trading Controls
        </h2>
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-[10px] uppercase tracking-wider text-zinc-600">
              Strategy
            </label>
            <div className="flex h-8 w-48 items-center rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] px-2.5 text-sm text-zinc-600">
              Select strategy...
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[10px] uppercase tracking-wider text-zinc-600">
              Account
            </label>
            <div className="flex h-8 w-48 items-center rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] px-2.5 text-sm text-zinc-600">
              Not connected
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              disabled
              className="gap-1.5 bg-[#22c55e]/20 text-[#22c55e] opacity-50 cursor-not-allowed"
            >
              <Play className="h-4 w-4" />
              Start
            </Button>
            <Button
              disabled
              className="gap-1.5 bg-[#ef4444]/20 text-[#ef4444] opacity-50 cursor-not-allowed"
            >
              <Square className="h-4 w-4" />
              Stop
            </Button>
          </div>
        </div>
        <p className="mt-4 text-xs text-zinc-600">
          Connect your Schwab account to enable paper trading controls.
        </p>
      </div>
    </div>
  );
}
