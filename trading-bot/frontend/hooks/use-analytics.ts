import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type {
  EquityCurvePoint,
  RegimeStat,
  MonteCarloResult,
} from "@/lib/types";

/**
 * Fetch equity curve data for a run.
 */
export function useEquityCurve(runId: string | undefined) {
  return useQuery<EquityCurvePoint[]>({
    queryKey: ["equity-curve", runId],
    queryFn: () =>
      apiFetch<{ points: EquityCurvePoint[] }>(`/api/analytics/${runId}/equity-curve`).then(r => r.points),
    enabled: !!runId,
  });
}

/**
 * Fetch regime-based performance statistics for a run.
 */
export function useRegimeStats(runId: string | undefined) {
  return useQuery<RegimeStat[]>({
    queryKey: ["regime-stats", runId],
    queryFn: () =>
      apiFetch<RegimeStat[]>(`/api/analytics/${runId}/regime`),
    enabled: !!runId,
  });
}

/**
 * Fetch Monte Carlo simulation results for a run.
 */
export function useMonteCarlo(
  runId: string | undefined,
  simulations: number = 1000
) {
  return useQuery<MonteCarloResult>({
    queryKey: ["monte-carlo", runId, simulations],
    queryFn: () =>
      apiFetch<MonteCarloResult>(
        `/api/analytics/${runId}/monte-carlo?simulations=${simulations}`
      ),
    enabled: !!runId,
  });
}
