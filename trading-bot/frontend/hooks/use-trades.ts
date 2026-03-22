import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Trade } from "@/lib/types";

/**
 * Fetch all trades for a given run.
 */
export function useTrades(runId: string | undefined) {
  return useQuery<Trade[]>({
    queryKey: ["trades", runId],
    queryFn: () => apiFetch<Trade[]>(`/api/trades?run_id=${runId}`),
    enabled: !!runId,
  });
}
