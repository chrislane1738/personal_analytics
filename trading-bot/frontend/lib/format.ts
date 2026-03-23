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
export function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  // Parse as local date (YYYY-MM-DD) to avoid timezone shift
  const parts = dateStr.split("T")[0].split("-");
  if (parts.length === 3) {
    const d = new Date(+parts[0], +parts[1] - 1, +parts[2]);
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  }
  return new Date(dateStr).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

/**
 * Return CSS class for profit/loss coloring.
 */
export function pnlColor(value: number | null): string {
  if (value === null || value === 0) return "text-zinc-400";
  return value > 0 ? "text-[#22c55e]" : "text-[#ef4444]";
}
