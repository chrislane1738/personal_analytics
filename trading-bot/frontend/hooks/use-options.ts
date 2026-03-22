import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { OptionsAnalyticsResponse } from "@/lib/types";

/**
 * Fetch options analytics for a run.
 */
export function useOptionsAnalytics(runId: string | undefined) {
  return useQuery<OptionsAnalyticsResponse>({
    queryKey: ["options-analytics", runId],
    queryFn: () =>
      apiFetch<OptionsAnalyticsResponse>(
        `/api/analytics/${runId}/options`
      ),
    enabled: !!runId,
  });
}
