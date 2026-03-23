import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type {
  WalkForwardStudy,
  WalkForwardListResponse,
  WalkForwardStatusResponse,
} from "@/lib/types";

export interface WalkForwardRunRequest {
  strategy: string;
  universe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  train_months: number;
  oos_months: number;
  step_months: number;
  objective: string;
}

export interface WalkForwardRunResponse {
  study_id: string;
  status: string;
}

/**
 * Launch a new walk-forward study.
 */
export function useLaunchWalkForward() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (params: WalkForwardRunRequest) =>
      apiFetch<WalkForwardRunResponse>("/api/walk-forward/run", {
        method: "POST",
        body: JSON.stringify(params),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["walk-forward"] });
    },
  });
}

/**
 * Fetch paginated list of walk-forward studies.
 */
export function useWalkForwardStudies() {
  return useQuery<WalkForwardListResponse>({
    queryKey: ["walk-forward"],
    queryFn: () => apiFetch<WalkForwardListResponse>("/api/walk-forward"),
  });
}

/**
 * Fetch a single walk-forward study by ID.
 */
export function useWalkForwardStudy(studyId: string | undefined) {
  return useQuery<WalkForwardStudy>({
    queryKey: ["walk-forward", studyId],
    queryFn: () => apiFetch<WalkForwardStudy>(`/api/walk-forward/${studyId}`),
    enabled: !!studyId,
  });
}

/**
 * Poll walk-forward study status while running (3s interval).
 */
export function useWalkForwardStatus(studyId: string | undefined) {
  return useQuery<WalkForwardStatusResponse>({
    queryKey: ["walk-forward-status", studyId],
    queryFn: () =>
      apiFetch<WalkForwardStatusResponse>(
        `/api/walk-forward/${studyId}/status`
      ),
    enabled: !!studyId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "running") return 3000;
      return false;
    },
  });
}

/**
 * Stop a running walk-forward study.
 */
export function useStopWalkForward() {
  return useMutation({
    mutationFn: (studyId: string) =>
      apiFetch<{ study_id: string; status: string }>(
        `/api/walk-forward/${studyId}/stop`,
        { method: "POST" }
      ),
  });
}

/**
 * Delete a walk-forward study and invalidate the list cache.
 */
export function useDeleteWalkForward() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (studyId: string) =>
      apiFetch<void>(`/api/walk-forward/${studyId}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["walk-forward"] });
    },
  });
}
