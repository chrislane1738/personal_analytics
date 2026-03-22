import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Run, RunListResponse } from "@/lib/types";

export interface RunsParams {
  page?: number;
  page_size?: number;
  strategy?: string;
  status?: string;
  sort_by?: string;
  sort_dir?: "asc" | "desc";
}

/**
 * Fetch paginated list of backtest runs.
 */
export function useRuns(params?: RunsParams) {
  const searchParams = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.set(key, String(value));
      }
    });
  }
  const query = searchParams.toString();
  const path = `/api/runs${query ? `?${query}` : ""}`;

  return useQuery<RunListResponse>({
    queryKey: ["runs", params],
    queryFn: () => apiFetch<RunListResponse>(path),
  });
}

/**
 * Fetch a single run by ID.
 */
export function useRun(runId: string | undefined) {
  return useQuery<Run>({
    queryKey: ["run", runId],
    queryFn: () => apiFetch<Run>(`/api/runs/${runId}`),
    enabled: !!runId,
  });
}

/**
 * Delete a run and invalidate the runs cache.
 */
export function useDeleteRun() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (runId: string) =>
      apiFetch<void>(`/api/runs/${runId}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}
