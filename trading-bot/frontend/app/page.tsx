"use client";

import { useRuns, useDeleteRun } from "@/hooks/use-runs";
import { RunsTable } from "@/components/tables/runs-table";

function SkeletonRows() {
  return (
    <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] p-1">
      {/* Header skeleton */}
      <div className="flex gap-4 border-b border-[#1a1a1a] px-4 py-3">
        {Array.from({ length: 7 }).map((_, i) => (
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
          {Array.from({ length: 7 }).map((_, j) => (
            <div
              key={j}
              className="h-4 rounded bg-zinc-800/40 animate-pulse"
              style={{ width: `${60 + Math.random() * 40}px` }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export default function HomePage() {
  const { data, isLoading, error } = useRuns();
  const deleteRun = useDeleteRun();

  return (
    <div className="p-6">
      <div className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
          Run Browser
        </h1>
        {data && (
          <span className="text-sm text-zinc-500">
            {data.total} {data.total === 1 ? "run" : "runs"}
          </span>
        )}
      </div>

      {isLoading && <SkeletonRows />}

      {error && (
        <p className="text-sm text-red-500">
          Failed to load runs: {error.message}
        </p>
      )}

      {data && (
        <RunsTable
          data={data.items}
          onDelete={(runId) => deleteRun.mutate(runId)}
        />
      )}
    </div>
  );
}
