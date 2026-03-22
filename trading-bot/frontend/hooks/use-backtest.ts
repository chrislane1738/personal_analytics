import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface BacktestRunRequest {
  strategy: string;
  universe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  benchmark: string;
  position_size_pct: number;
}

export interface BacktestRunResponse {
  run_id: string;
  status: string;
}

export interface BacktestStatusResponse {
  run_id: string;
  status: string;
  error?: string | null;
}

/**
 * Launch a new backtest run.
 */
export function useLaunchBacktest() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (params: BacktestRunRequest) =>
      apiFetch<BacktestRunResponse>("/api/backtest/run", {
        method: "POST",
        body: JSON.stringify(params),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

/**
 * Poll backtest status for a given run_id.
 */
export function useBacktestStatus(runId: string | undefined) {
  return useQuery<BacktestStatusResponse>({
    queryKey: ["backtest-status", runId],
    queryFn: () =>
      apiFetch<BacktestStatusResponse>(`/api/backtest/status/${runId}`),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "running") return 2000;
      return false;
    },
  });
}

/**
 * Stop a running backtest.
 */
export function useStopBacktest() {
  return useMutation({
    mutationFn: (runId: string) =>
      apiFetch<{ run_id: string; status: string }>(
        `/api/backtest/stop/${runId}`,
        { method: "POST" }
      ),
  });
}
