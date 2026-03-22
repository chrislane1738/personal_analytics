import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface StrategyFile {
  name: string;
  content: string;
}

/**
 * Fetch all strategy files.
 */
export function useStrategies() {
  return useQuery<StrategyFile[]>({
    queryKey: ["strategies"],
    queryFn: () => apiFetch<StrategyFile[]>("/api/strategies"),
  });
}

/**
 * Fetch a single strategy file by name.
 */
export function useStrategy(name: string | undefined) {
  return useQuery<StrategyFile>({
    queryKey: ["strategy", name],
    queryFn: () => apiFetch<StrategyFile>(`/api/strategies/${name}`),
    enabled: !!name,
  });
}

/**
 * Save (update) an existing strategy file.
 */
export function useSaveStrategy() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, content }: { name: string; content: string }) =>
      apiFetch<StrategyFile>(`/api/strategies/${name}`, {
        method: "PUT",
        body: JSON.stringify({ content }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies"] });
      queryClient.invalidateQueries({ queryKey: ["strategy"] });
    },
  });
}

/**
 * Create a new strategy file.
 */
export function useCreateStrategy() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, content }: { name: string; content: string }) =>
      apiFetch<StrategyFile>("/api/strategies", {
        method: "POST",
        body: JSON.stringify({ name, content }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies"] });
    },
  });
}

/**
 * Delete a strategy file.
 */
export function useDeleteStrategy() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<void>(`/api/strategies/${name}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies"] });
    },
  });
}
