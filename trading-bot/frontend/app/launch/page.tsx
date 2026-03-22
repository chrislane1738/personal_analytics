"use client";

import { useState } from "react";
import Link from "next/link";
import { Play, Loader2, CheckCircle2, XCircle, StopCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useStrategies } from "@/hooks/use-strategies";
import {
  useLaunchBacktest,
  useBacktestStatus,
  useStopBacktest,
} from "@/hooks/use-backtest";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function LaunchPage() {
  const { data: strategies, isLoading: strategiesLoading } = useStrategies();
  const launchBacktest = useLaunchBacktest();
  const stopBacktest = useStopBacktest();

  // Form state
  const [strategy, setStrategy] = useState("");
  const [universe, setUniverse] = useState("SPY,AAPL,MSFT");
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");
  const [initialCapital, setInitialCapital] = useState(100000);
  const [benchmark, setBenchmark] = useState("SPY");
  const [positionSizePct, setPositionSizePct] = useState(6);

  // Active run tracking
  const [activeRunId, setActiveRunId] = useState<string | undefined>(undefined);
  const { data: runStatus } = useBacktestStatus(activeRunId);

  const handleLaunch = () => {
    if (!strategy) return;
    launchBacktest.mutate(
      {
        strategy,
        universe,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        benchmark,
        position_size_pct: positionSizePct / 100,
      },
      {
        onSuccess: (result) => {
          setActiveRunId(result.run_id);
        },
      }
    );
  };

  const handleStop = () => {
    if (activeRunId) {
      stopBacktest.mutate(activeRunId);
    }
  };

  const isRunning = runStatus?.status === "running";
  const isCompleted = runStatus?.status === "completed";
  const isFailed = runStatus?.status === "failed";
  const isCancelled = runStatus?.status === "cancelled";

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
          Backtest Launcher
        </h1>
        <p className="mt-1 text-xs text-zinc-500">
          Configure and launch a new backtest run
        </p>
      </div>

      {/* Form */}
      <div className="grid max-w-2xl gap-6">
        {/* Strategy Picker */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-zinc-400">Strategy</label>
          {strategiesLoading ? (
            <p className="text-sm text-zinc-500">Loading strategies...</p>
          ) : (
            <Select value={strategy} onValueChange={(v) => setStrategy(v ?? "")}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a strategy..." />
              </SelectTrigger>
              <SelectContent>
                {strategies?.map((s) => (
                  <SelectItem key={s.name} value={s.name}>
                    {s.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        {/* Universe */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-zinc-400">
            Universe (comma-separated symbols)
          </label>
          <Input
            value={universe}
            onChange={(e) => setUniverse(e.target.value)}
            placeholder="SPY,AAPL,MSFT"
          />
        </div>

        {/* Date Range */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Start Date
            </label>
            <Input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              End Date
            </label>
            <Input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
        </div>

        {/* Capital, Benchmark, Position Size */}
        <div className="grid grid-cols-3 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Initial Capital ($)
            </label>
            <Input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              min={1000}
              step={1000}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Benchmark
            </label>
            <Input
              value={benchmark}
              onChange={(e) => setBenchmark(e.target.value)}
              placeholder="SPY"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Position Size (%)
            </label>
            <Input
              type="number"
              value={positionSizePct}
              onChange={(e) => setPositionSizePct(Number(e.target.value))}
              min={1}
              max={100}
              step={1}
            />
          </div>
        </div>

        {/* Launch Button */}
        <div className="flex items-center gap-3">
          <Button
            onClick={handleLaunch}
            disabled={
              !strategy || launchBacktest.isPending || isRunning
            }
            size="lg"
          >
            {launchBacktest.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="ml-1.5">Launching...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span className="ml-1.5">Run Backtest</span>
              </>
            )}
          </Button>
          {isRunning && (
            <Button variant="destructive" size="lg" onClick={handleStop}>
              <StopCircle className="h-4 w-4" />
              <span className="ml-1.5">Stop</span>
            </Button>
          )}
        </div>

        {/* Launch Error */}
        {launchBacktest.error && (
          <div className="rounded-md border border-red-900 bg-red-950/50 px-4 py-3 text-sm text-red-400">
            Launch failed: {launchBacktest.error.message}
          </div>
        )}
      </div>

      {/* Run Status */}
      {activeRunId && runStatus && (
        <div className="max-w-2xl rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Run Status
          </h2>
          <div className="flex items-center gap-3">
            {isRunning && (
              <>
                <Loader2 className="h-5 w-5 animate-spin text-[#f97316]" />
                <span className="text-sm text-[#f97316]">Running...</span>
              </>
            )}
            {isCompleted && (
              <>
                <CheckCircle2 className="h-5 w-5 text-[#22c55e]" />
                <span className="text-sm text-[#22c55e]">Completed</span>
              </>
            )}
            {isFailed && (
              <>
                <XCircle className="h-5 w-5 text-[#ef4444]" />
                <span className="text-sm text-[#ef4444]">Failed</span>
              </>
            )}
            {isCancelled && (
              <>
                <StopCircle className="h-5 w-5 text-zinc-400" />
                <span className="text-sm text-zinc-400">Cancelled</span>
              </>
            )}
            <span className="font-mono text-xs text-zinc-600">
              ID: {activeRunId}
            </span>
          </div>

          {runStatus.error && (
            <div className="mt-3 rounded-md border border-red-900 bg-red-950/50 px-3 py-2 text-xs text-red-400">
              {runStatus.error}
            </div>
          )}

          {isCompleted && (
            <div className="mt-3">
              <Link
                href={`/runs/${runStatus?.run_id ?? activeRunId}`}
                className="text-sm text-[#f97316] underline-offset-4 hover:underline"
              >
                View Run Details
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
