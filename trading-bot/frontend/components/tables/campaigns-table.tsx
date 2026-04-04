"use client";

import { useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";
import { ArrowUpDown, Trash2 } from "lucide-react";
import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatCurrency } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Campaign } from "@/lib/types";

interface CampaignsTableProps {
  data: Campaign[];
  mode?: "eval" | "funded";
  onDelete: (campaignId: string) => void;
}

/**
 * Format an ISO date string as a relative time (e.g. "2h ago", "3d ago").
 */
function relativeTime(dateStr: string | null): string {
  if (!dateStr) return "\u2014";
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMs = now - then;

  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return `${seconds}s ago`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;

  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;

  const years = Math.floor(months / 12);
  return `${years}y ago`;
}

/**
 * Return a color class for pass rate / survival rate values.
 */
function passRateColor(value: number): string {
  if (value > 0.3) return "text-[#22c55e]";
  if (value >= 0.2) return "text-[#eab308]";
  return "text-[#ef4444]";
}

/**
 * Return a color class for EV values (positive = green, negative = red).
 */
function evColor(value: number): string {
  if (value > 0) return "text-[#22c55e]";
  if (value < 0) return "text-[#ef4444]";
  return "text-zinc-400";
}

function SortableHeader({
  label,
  column,
}: {
  label: string;
  column: { toggleSorting: (desc?: boolean) => void };
}) {
  return (
    <button
      type="button"
      className="flex items-center gap-1 hover:text-zinc-300 transition-colors"
      onClick={() => column.toggleSorting()}
    >
      {label}
      <ArrowUpDown className="h-3 w-3" />
    </button>
  );
}

function buildEvalColumns(
  onDelete: (id: string) => void
): ColumnDef<Campaign>[] {
  return [
    {
      accessorKey: "campaign_id",
      header: ({ column }) => (
        <SortableHeader label="Campaign" column={column} />
      ),
      cell: ({ row }) => {
        const id = row.original.campaign_id;
        return (
          <Link
            href={`/eval/${id}`}
            className="font-mono text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            onClick={(e) => e.stopPropagation()}
          >
            {id.slice(0, 8)}...
          </Link>
        );
      },
    },
    {
      accessorKey: "strategy_name",
      header: ({ column }) => (
        <SortableHeader label="Strategy" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="text-sm text-zinc-200">{getValue<string>()}</span>
      ),
    },
    {
      id: "mode",
      header: () => <span>Mode</span>,
      cell: ({ row }) => {
        const m = row.original.mode ?? "eval";
        return (
          <span
            className={cn(
              "inline-flex h-5 items-center rounded-full px-2 text-[10px] font-medium",
              m === "funded"
                ? "bg-blue-500/15 text-blue-400"
                : "bg-zinc-800 text-zinc-400"
            )}
          >
            {m === "funded" ? "FUNDED" : "EVAL"}
          </span>
        );
      },
      enableSorting: false,
    },
    {
      id: "instrument_timeframe",
      header: () => <span>Instrument</span>,
      cell: ({ row }) => {
        const tf = row.original.timeframe;
        const label =
          tf === "1D"
            ? "Daily"
            : tf === "1m"
              ? "1-Min"
              : tf === "5m"
                ? "5-Min"
                : tf === "15m"
                  ? "15-Min"
                  : tf === "1h"
                    ? "Hourly"
                    : tf;
        return (
          <span className="text-sm text-zinc-200">
            <span className="font-mono">{row.original.instrument}</span>
            <span className="ml-1.5 text-zinc-500">{label}</span>
          </span>
        );
      },
      enableSorting: false,
    },
    {
      accessorKey: "state_machine",
      header: ({ column }) => (
        <SortableHeader label="State Machine" column={column} />
      ),
      cell: ({ getValue }) => {
        const enabled = getValue<boolean>();
        return (
          <span
            className={cn(
              "inline-flex h-5 items-center rounded-full px-2 text-[10px] font-medium",
              enabled
                ? "bg-[#22c55e]/15 text-[#22c55e]"
                : "bg-zinc-800 text-zinc-400"
            )}
          >
            {enabled ? "ON" : "OFF"}
          </span>
        );
      },
    },
    {
      accessorKey: "num_attempts",
      header: ({ column }) => (
        <SortableHeader label="Attempts" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="font-mono text-sm text-zinc-300">
          {getValue<number>().toLocaleString()}
        </span>
      ),
    },
    {
      accessorKey: "pass_rate",
      header: ({ column }) => (
        <SortableHeader label="Pass Rate" column={column} />
      ),
      cell: ({ getValue }) => {
        const value = getValue<number>();
        return (
          <span className={cn("font-mono text-sm", passRateColor(value))}>
            {(value * 100).toFixed(1)}%
          </span>
        );
      },
    },
    {
      accessorKey: "ev_per_attempt",
      header: ({ column }) => (
        <SortableHeader label="EV/Attempt" column={column} />
      ),
      cell: ({ getValue }) => {
        const value = getValue<number>();
        return (
          <span className={cn("font-mono text-sm", evColor(value))}>
            {formatCurrency(value)}
          </span>
        );
      },
    },
    {
      accessorKey: "cost_to_funded",
      header: ({ column }) => (
        <SortableHeader label="Cost to Fund" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="font-mono text-sm text-zinc-300">
          {formatCurrency(getValue<number>())}
        </span>
      ),
    },
    {
      accessorKey: "avg_days_to_pass",
      header: ({ column }) => (
        <SortableHeader label="Avg Days" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="font-mono text-sm text-zinc-300">
          {getValue<number>().toFixed(1)}
        </span>
      ),
    },
    {
      accessorKey: "created_at",
      header: ({ column }) => (
        <SortableHeader label="Created" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="text-sm text-zinc-400">
          {relativeTime(getValue<string | null>())}
        </span>
      ),
    },
    {
      id: "actions",
      header: () => null,
      cell: ({ row }) => (
        <button
          type="button"
          className="rounded p-1 text-zinc-600 hover:bg-zinc-800 hover:text-red-400 transition-colors"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(row.original.campaign_id);
          }}
          aria-label="Delete campaign"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      ),
      enableSorting: false,
    },
  ];
}

function buildFundedColumns(
  onDelete: (id: string) => void
): ColumnDef<Campaign>[] {
  return [
    {
      accessorKey: "campaign_id",
      header: ({ column }) => (
        <SortableHeader label="Campaign" column={column} />
      ),
      cell: ({ row }) => {
        const id = row.original.campaign_id;
        return (
          <Link
            href={`/eval/${id}`}
            className="font-mono text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            onClick={(e) => e.stopPropagation()}
          >
            {id.slice(0, 8)}...
          </Link>
        );
      },
    },
    {
      accessorKey: "strategy_name",
      header: ({ column }) => (
        <SortableHeader label="Strategy" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="text-sm text-zinc-200">{getValue<string>()}</span>
      ),
    },
    {
      id: "mode",
      header: () => <span>Mode</span>,
      cell: () => (
        <span className="inline-flex h-5 items-center rounded-full px-2 text-[10px] font-medium bg-blue-500/15 text-blue-400">
          FUNDED
        </span>
      ),
      enableSorting: false,
    },
    {
      id: "instrument_timeframe",
      header: () => <span>Instrument</span>,
      cell: ({ row }) => {
        const tf = row.original.timeframe;
        const label =
          tf === "1D"
            ? "Daily"
            : tf === "1m"
              ? "1-Min"
              : tf === "5m"
                ? "5-Min"
                : tf === "15m"
                  ? "15-Min"
                  : tf === "1h"
                    ? "Hourly"
                    : tf;
        return (
          <span className="text-sm text-zinc-200">
            <span className="font-mono">{row.original.instrument}</span>
            <span className="ml-1.5 text-zinc-500">{label}</span>
          </span>
        );
      },
      enableSorting: false,
    },
    {
      accessorKey: "num_attempts",
      header: ({ column }) => (
        <SortableHeader label="Simulations" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="font-mono text-sm text-zinc-300">
          {getValue<number>().toLocaleString()}
        </span>
      ),
    },
    {
      accessorKey: "survival_rate",
      header: ({ column }) => (
        <SortableHeader label="Survival" column={column} />
      ),
      cell: ({ row }) => {
        const value = row.original.survival_rate ?? 0;
        return (
          <span className={cn("font-mono text-sm", passRateColor(value))}>
            {(value * 100).toFixed(1)}%
          </span>
        );
      },
    },
    {
      accessorKey: "avg_monthly_pnl",
      header: ({ column }) => (
        <SortableHeader label="Monthly P&L" column={column} />
      ),
      cell: ({ row }) => {
        const value = row.original.avg_monthly_pnl ?? 0;
        return (
          <span className={cn("font-mono text-sm", evColor(value))}>
            {formatCurrency(value)}
          </span>
        );
      },
    },
    {
      accessorKey: "sharpe_ratio",
      header: ({ column }) => (
        <SortableHeader label="Sharpe" column={column} />
      ),
      cell: ({ row }) => {
        const value = row.original.sharpe_ratio ?? 0;
        return (
          <span
            className={cn(
              "font-mono text-sm",
              value > 1
                ? "text-[#22c55e]"
                : value > 0.5
                  ? "text-[#eab308]"
                  : "text-[#ef4444]"
            )}
          >
            {value.toFixed(2)}
          </span>
        );
      },
    },
    {
      accessorKey: "annual_expected_income",
      header: ({ column }) => (
        <SortableHeader label="Annual Income" column={column} />
      ),
      cell: ({ row }) => {
        const value = row.original.annual_expected_income ?? 0;
        return (
          <span className={cn("font-mono text-sm", evColor(value))}>
            {formatCurrency(value)}
          </span>
        );
      },
    },
    {
      accessorKey: "created_at",
      header: ({ column }) => (
        <SortableHeader label="Created" column={column} />
      ),
      cell: ({ getValue }) => (
        <span className="text-sm text-zinc-400">
          {relativeTime(getValue<string | null>())}
        </span>
      ),
    },
    {
      id: "actions",
      header: () => null,
      cell: ({ row }) => (
        <button
          type="button"
          className="rounded p-1 text-zinc-600 hover:bg-zinc-800 hover:text-red-400 transition-colors"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(row.original.campaign_id);
          }}
          aria-label="Delete campaign"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      ),
      enableSorting: false,
    },
  ];
}

export function CampaignsTable({ data, mode = "eval", onDelete }: CampaignsTableProps) {
  const router = useRouter();
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(
    () =>
      mode === "funded"
        ? buildFundedColumns(onDelete)
        : buildEvalColumns(onDelete),
    [mode, onDelete]
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f]">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow
              key={headerGroup.id}
              className="border-[#1a1a1a] hover:bg-transparent"
            >
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className="text-zinc-500 text-xs uppercase tracking-wider font-medium"
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={columns.length}
                className="h-24 text-center text-zinc-500"
              >
                No campaigns found.
              </TableCell>
            </TableRow>
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="cursor-pointer border-[#1a1a1a] hover:bg-zinc-900/50 transition-colors"
                onClick={() =>
                  router.push(`/eval/${row.original.campaign_id}`)
                }
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
