"use client";

import { useState } from "react";
import { Database, Download, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { useSymbols, useFetchData } from "@/hooks/use-data";
import { SymbolsTable } from "@/components/tables/symbols-table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

function formatNumber(n: number): string {
  return new Intl.NumberFormat("en-US").format(n);
}

function SkeletonRows() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-10 rounded-md bg-zinc-800/50 animate-pulse"
        />
      ))}
    </div>
  );
}

export default function DataPage() {
  const { data, isLoading, error } = useSymbols();
  const fetchData = useFetchData();

  const [symbol, setSymbol] = useState("");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState(
    new Date().toISOString().split("T")[0]
  );

  function handleFetch(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = symbol.trim().toUpperCase();
    if (!trimmed) return;
    fetchData.mutate({
      symbol: trimmed,
      start_date: startDate,
      end_date: endDate,
    });
  }

  return (
    <div className="p-6 flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-baseline justify-between">
        <div className="flex items-center gap-3">
          <Database className="h-5 w-5 text-[#f97316]" />
          <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
            Data Manager
          </h1>
        </div>
      </div>

      {/* Storage stats + Fetch panel */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Stats cards */}
        <div className="flex gap-4 lg:col-span-1">
          <div className="flex-1 rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
            <p className="text-[10px] uppercase tracking-wider text-zinc-500">
              Total Symbols
            </p>
            <p className="mt-1 font-mono text-2xl font-semibold text-zinc-100">
              {isLoading ? "--" : formatNumber(data?.total_symbols ?? 0)}
            </p>
          </div>
          <div className="flex-1 rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4">
            <p className="text-[10px] uppercase tracking-wider text-zinc-500">
              Total Bars
            </p>
            <p className="mt-1 font-mono text-2xl font-semibold text-zinc-100">
              {isLoading ? "--" : formatNumber(data?.total_bars ?? 0)}
            </p>
          </div>
        </div>

        {/* Fetch panel */}
        <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-4 lg:col-span-2">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Fetch Market Data
          </h2>
          <form onSubmit={handleFetch} className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <label
                htmlFor="symbol-input"
                className="text-[10px] uppercase tracking-wider text-zinc-500"
              >
                Symbol
              </label>
              <Input
                id="symbol-input"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                placeholder="AAPL"
                className="w-28 bg-[#0a0a0a] border-[#1a1a1a] font-mono uppercase"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label
                htmlFor="start-date-input"
                className="text-[10px] uppercase tracking-wider text-zinc-500"
              >
                Start Date
              </label>
              <Input
                id="start-date-input"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-36 bg-[#0a0a0a] border-[#1a1a1a] font-mono"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label
                htmlFor="end-date-input"
                className="text-[10px] uppercase tracking-wider text-zinc-500"
              >
                End Date
              </label>
              <Input
                id="end-date-input"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-36 bg-[#0a0a0a] border-[#1a1a1a] font-mono"
              />
            </div>
            <Button
              type="submit"
              disabled={fetchData.isPending || !symbol.trim()}
              className="gap-1.5 bg-[#f97316] text-black hover:bg-[#ea580c] disabled:opacity-50"
            >
              {fetchData.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              {fetchData.isPending ? "Fetching..." : "Fetch"}
            </Button>
          </form>

          {/* Fetch result feedback */}
          {fetchData.isSuccess && (
            <div className="mt-3 flex items-center gap-2 text-sm text-[#22c55e]">
              <CheckCircle className="h-4 w-4" />
              {fetchData.data.message}
            </div>
          )}
          {fetchData.isError && (
            <div className="mt-3 flex items-center gap-2 text-sm text-[#ef4444]">
              <AlertCircle className="h-4 w-4" />
              {fetchData.error.message}
            </div>
          )}
        </div>
      </div>

      {/* Symbols table */}
      <div>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Cached Symbols
        </h2>
        {isLoading ? (
          <SkeletonRows />
        ) : error ? (
          <p className="text-sm text-[#ef4444]">
            Failed to load symbols: {error.message}
          </p>
        ) : (
          <SymbolsTable data={data?.symbols ?? []} />
        )}
      </div>
    </div>
  );
}
