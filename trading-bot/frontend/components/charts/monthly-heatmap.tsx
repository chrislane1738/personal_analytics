"use client";

import { useMemo } from "react";
import type { EquityCurvePoint } from "@/lib/types";

interface MonthlyHeatmapProps {
  data: EquityCurvePoint[];
}

interface MonthlyReturn {
  year: number;
  month: number;
  returnPct: number;
}

const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/**
 * Interpolate between red, neutral, and green based on return value.
 * Negative returns map to red, zero to neutral gray, positive to green.
 */
function returnToColor(value: number): string {
  // Clamp to [-50%, +50%] for color scaling
  const clamped = Math.max(-0.5, Math.min(0.5, value));
  const t = clamped / 0.5; // -1 to 1

  if (t < 0) {
    // Red scale: interpolate from neutral (#1a1a1a) to red (#ef4444)
    const intensity = Math.abs(t);
    const r = Math.round(26 + (239 - 26) * intensity);
    const g = Math.round(26 + (68 - 26) * intensity);
    const b = Math.round(26 + (68 - 26) * intensity);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Green scale: interpolate from neutral (#1a1a1a) to green (#22c55e)
    const intensity = t;
    const r = Math.round(26 + (34 - 26) * intensity);
    const g = Math.round(26 + (197 - 26) * intensity);
    const b = Math.round(26 + (94 - 26) * intensity);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

function returnToTextColor(value: number): string {
  const absVal = Math.abs(value);
  if (absVal < 0.02) return "#a1a1aa"; // zinc-400 for near-zero
  return "#fafafa";
}

export function MonthlyHeatmap({ data }: MonthlyHeatmapProps) {
  const { years, grid } = useMemo(() => {
    if (!data || data.length === 0) return { years: [], grid: new Map() };

    // Group equity values by year-month, keeping first and last per month
    const monthBounds = new Map<
      string,
      { startEquity: number; endEquity: number }
    >();

    for (const point of data) {
      const d = new Date(point.date);
      const year = d.getFullYear();
      const month = d.getMonth();
      const key = `${year}-${month}`;

      const existing = monthBounds.get(key);
      if (!existing) {
        monthBounds.set(key, {
          startEquity: point.strategy_value,
          endEquity: point.strategy_value,
        });
      } else {
        existing.endEquity = point.strategy_value;
      }
    }

    // Compute monthly returns
    const returns: MonthlyReturn[] = [];
    for (const [key, bounds] of monthBounds) {
      const [yearStr, monthStr] = key.split("-");
      const year = parseInt(yearStr, 10);
      const month = parseInt(monthStr, 10);
      const returnPct =
        bounds.startEquity > 0
          ? bounds.endEquity / bounds.startEquity - 1
          : 0;
      returns.push({ year, month, returnPct });
    }

    // Build grid keyed by "year-month"
    const gridMap = new Map<string, number>();
    const yearSet = new Set<number>();
    for (const r of returns) {
      gridMap.set(`${r.year}-${r.month}`, r.returnPct);
      yearSet.add(r.year);
    }

    const sortedYears = Array.from(yearSet).sort((a, b) => a - b);
    return { years: sortedYears, grid: gridMap };
  }, [data]);

  if (years.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        No data available for monthly returns
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
              Year
            </th>
            {MONTH_LABELS.map((label) => (
              <th
                key={label}
                className="px-2 py-2 text-center text-[10px] font-semibold uppercase tracking-wider text-zinc-500"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {years.map((year) => (
            <tr key={year}>
              <td className="px-3 py-1 font-mono text-xs font-semibold text-zinc-400">
                {year}
              </td>
              {Array.from({ length: 12 }, (_, month) => {
                const key = `${year}-${month}`;
                const value = grid.get(key);

                if (value === undefined || value === null || Number.isNaN(value)) {
                  return (
                    <td
                      key={month}
                      className="border border-[#1a1a1a] px-2 py-2 text-center"
                      style={{ backgroundColor: "#0a0a0a" }}
                    >
                      <span className="font-mono text-[11px] text-zinc-700">
                        &mdash;
                      </span>
                    </td>
                  );
                }

                const bgColor = returnToColor(value);
                const textColor = returnToTextColor(value);
                const prefix = value >= 0 ? "+" : "";
                const display = `${prefix}${(value * 100).toFixed(1)}%`;

                return (
                  <td
                    key={month}
                    className="border border-[#1a1a1a] px-2 py-2 text-center"
                    style={{ backgroundColor: bgColor }}
                  >
                    <span
                      className="font-mono text-[11px] font-medium"
                      style={{ color: textColor }}
                    >
                      {display}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
