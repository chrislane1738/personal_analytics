/**
 * Format a number as currency (USD).
 */
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

/**
 * Format a number as a percentage with 2 decimal places.
 * Input is expected as a decimal (0.15 = 15%).
 */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

/**
 * Format an ISO date string to a human-readable date.
 */
export function formatDate(dateStr: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(dateStr));
}

/**
 * Return CSS class for profit/loss coloring.
 */
export function pnlColor(value: number | null): string {
  if (value === null || value === 0) return "text-zinc-400";
  return value > 0 ? "text-[#22c55e]" : "text-[#ef4444]";
}
