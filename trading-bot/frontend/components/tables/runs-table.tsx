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
import { formatPercent, formatDate, pnlColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Run } from "@/lib/types";

interface RunsTableProps {
  data: Run[];
  onDelete: (runId: string) => void;
}

/**
 * Format an ISO date string as a relative time (e.g. "2h ago", "3d ago").
 */
function relativeTime(dateStr: string): string {
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
 * Return a color class for Sharpe ratio values.
 */
function sharpeColor(value: number | null): string {
  if (value === null) return "text-zinc-400";
  if (value > 1) return "text-[#22c55e]";
  if (value > 0) return "text-[#eab308]";
  return "text-[#ef4444]";
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

export function RunsTable({ data, onDelete }: RunsTableProps) {
  const router = useRouter();
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo<ColumnDef<Run>[]>(
    () => [
      {
        accessorKey: "id",
        header: ({ column }) => (
          <SortableHeader label="Run ID" column={column} />
        ),
        cell: ({ row }) => {
          const runId = row.original.id;
          return (
            <Link
              href={`/runs/${runId}`}
              className="font-mono text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              {runId.slice(0, 8)}...
            </Link>
          );
        },
      },
      {
        accessorKey: "strategy",
        header: ({ column }) => (
          <SortableHeader label="Strategy" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="text-sm text-zinc-200">{getValue<string>()}</span>
        ),
      },
      {
        accessorKey: "start_date",
        header: ({ column }) => (
          <SortableHeader label="Start" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="text-sm text-zinc-300">
            {formatDate(getValue<string>())}
          </span>
        ),
      },
      {
        accessorKey: "end_date",
        header: ({ column }) => (
          <SortableHeader label="End" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="text-sm text-zinc-300">
            {formatDate(getValue<string>())}
          </span>
        ),
      },
      {
        accessorKey: "total_return",
        header: ({ column }) => (
          <SortableHeader label="Return" column={column} />
        ),
        cell: ({ getValue }) => {
          const value = getValue<number | null>();
          return (
            <span className={cn("font-mono text-sm", pnlColor(value))}>
              {value !== null ? formatPercent(value) : "—"}
            </span>
          );
        },
      },
      {
        accessorKey: "sharpe_ratio",
        header: ({ column }) => (
          <SortableHeader label="Sharpe" column={column} />
        ),
        cell: ({ getValue }) => {
          const value = getValue<number | null>();
          return (
            <span className={cn("font-mono text-sm", sharpeColor(value))}>
              {value !== null ? value.toFixed(2) : "—"}
            </span>
          );
        },
      },
      {
        accessorKey: "max_drawdown",
        header: ({ column }) => (
          <SortableHeader label="Max DD" column={column} />
        ),
        cell: ({ getValue }) => {
          const value = getValue<number | null>();
          return (
            <span className="font-mono text-sm text-[#ef4444]">
              {value !== null ? formatPercent(value) : "—"}
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
            {relativeTime(getValue<string>())}
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
              onDelete(row.original.id);
            }}
            aria-label="Delete run"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        ),
        enableSorting: false,
      },
    ],
    [onDelete]
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
                No runs found.
              </TableCell>
            </TableRow>
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="cursor-pointer border-[#1a1a1a] hover:bg-zinc-900/50 transition-colors"
                onClick={() => router.push(`/runs/${row.original.id}`)}
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
