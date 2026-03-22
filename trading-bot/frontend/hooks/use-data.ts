import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SymbolInfo {
  symbol: string;
  company_name: string | null;
  sector: string | null;
  industry: string | null;
  exchange: string | null;
  market_cap: number | null;
  bar_count: number;
  min_date: string | null;
  max_date: string | null;
  updated_at: string | null;
}

export interface SymbolListResponse {
  symbols: SymbolInfo[];
  total_symbols: number;
  total_bars: number;
}

export interface DataFetchRequest {
  symbol: string;
  start_date: string;
  end_date: string;
}

export interface DataFetchResponse {
  symbol: string;
  bars_fetched: number;
  message: string;
}

export interface DataValidateResponse {
  results: Record<string, string>;
}

export interface DataQualityEntry {
  symbol: string;
  date: string;
  issue_type: string;
  severity: string | null;
  details: string | null;
  resolved: boolean;
}

export interface DataQualityResponse {
  entries: DataQualityEntry[];
  total: number;
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/**
 * Fetch the list of cached symbols with bar counts and date ranges.
 */
export function useSymbols() {
  return useQuery<SymbolListResponse>({
    queryKey: ["data-symbols"],
    queryFn: () => apiFetch<SymbolListResponse>("/api/data/symbols"),
  });
}

/**
 * Fetch data for a symbol via POST /api/data/fetch.
 * Invalidates the symbols cache on success.
 */
export function useFetchData() {
  const queryClient = useQueryClient();

  return useMutation<DataFetchResponse, Error, DataFetchRequest>({
    mutationFn: (req) =>
      apiFetch<DataFetchResponse>("/api/data/fetch", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data-symbols"] });
    },
  });
}

/**
 * Validate data for a list of symbols.
 */
export function useValidateData() {
  return useMutation<DataValidateResponse, Error, string[]>({
    mutationFn: (symbols) =>
      apiFetch<DataValidateResponse>("/api/data/validate", {
        method: "POST",
        body: JSON.stringify({ symbols }),
      }),
  });
}

/**
 * Fetch data quality log for a specific symbol.
 */
export function useDataQuality(symbol: string | null) {
  return useQuery<DataQualityResponse>({
    queryKey: ["data-quality", symbol],
    queryFn: () =>
      apiFetch<DataQualityResponse>(`/api/data/quality?symbol=${symbol}`),
    enabled: !!symbol,
  });
}
