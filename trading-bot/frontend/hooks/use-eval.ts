import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type {
  CampaignListResponse,
  CampaignDetail,
  CampaignRunRequest,
  CampaignStatusResponse,
} from "@/lib/types";

/**
 * Fetch list of eval campaigns.
 */
export function useCampaigns() {
  return useQuery<CampaignListResponse>({
    queryKey: ["campaigns"],
    queryFn: () => apiFetch<CampaignListResponse>("/api/eval/campaigns"),
  });
}

/**
 * Fetch a single campaign by ID.
 */
export function useCampaign(campaignId: string | undefined) {
  return useQuery<CampaignDetail>({
    queryKey: ["campaign", campaignId],
    queryFn: () =>
      apiFetch<CampaignDetail>(`/api/eval/campaigns/${campaignId}`),
    enabled: !!campaignId,
  });
}

/**
 * Fetch campaign run status.
 */
export function useCampaignStatus(campaignId: string | undefined) {
  return useQuery<CampaignStatusResponse>({
    queryKey: ["campaign-status", campaignId],
    queryFn: () =>
      apiFetch<CampaignStatusResponse>(
        `/api/eval/campaigns/${campaignId}/status`
      ),
    enabled: !!campaignId,
    refetchInterval: 3000,
  });
}

/**
 * Run a new eval campaign and invalidate the campaigns cache.
 */
export function useRunCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CampaignRunRequest) =>
      apiFetch<CampaignStatusResponse>("/api/eval/campaigns/run", {
        method: "POST",
        body: JSON.stringify(request),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["campaigns"] });
    },
  });
}

/**
 * Delete a campaign and invalidate the campaigns cache.
 */
export function useDeleteCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (campaignId: string) =>
      apiFetch<void>(`/api/eval/campaigns/${campaignId}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["campaigns"] });
    },
  });
}
