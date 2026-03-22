"use client";

import { useRuns, useDeleteRun } from "@/hooks/use-runs";
import { RunsTable } from "@/components/tables/runs-table";

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

      {isLoading && (
        <p className="text-sm text-zinc-500">Loading runs...</p>
      )}

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
