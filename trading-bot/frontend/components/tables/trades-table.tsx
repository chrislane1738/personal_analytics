"use client";

import { useState, useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Trade } from "@/lib/types";
import { formatCurrency, formatDate } from "@/lib/format";
import { pnlColor } from "@/lib/format";
import { ArrowUpDown } from "lucide-react";

interface TradesTableProps {
  data: Trade[];
}

const columnHelper = createColumnHelper<Trade>();

function holdingDays(durationSeconds: number | null): string {
  if (durationSeconds === null) return "-";
  const days = Math.round(durationSeconds / 86400);
  return days === 1 ? "1d" : `${days}d`;
}

export function TradesTable({ data }: TradesTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(
    () => [
      columnHelper.accessor("entry_time", {
        header: "Date",
        cell: (info) => (
          <span className="font-mono text-xs text-zinc-300">
            {formatDate(info.getValue())}
          </span>
        ),
      }),
      columnHelper.accessor("side", {
        header: "Direction",
        cell: (info) => {
          const side = info.getValue();
          return (
            <span
              className={`font-mono text-xs font-semibold ${
                side === "long" ? "text-[#22c55e]" : "text-[#ef4444]"
              }`}
            >
              {side === "long" ? "BUY" : "SELL"}
            </span>
          );
        },
      }),
      columnHelper.accessor("symbol", {
        header: "Symbol",
        cell: (info) => (
          <span className="font-mono text-xs font-medium text-zinc-200">
            {info.getValue()}
          </span>
        ),
      }),
      columnHelper.accessor("entry_price", {
        header: "Entry",
        cell: (info) => (
          <span className="font-mono text-xs text-zinc-300">
            {formatCurrency(info.getValue())}
          </span>
        ),
      }),
      columnHelper.accessor("quantity", {
        header: "Qty",
        cell: (info) => (
          <span className="font-mono text-xs text-zinc-400">
            {info.getValue()}
          </span>
        ),
      }),
      columnHelper.accessor("exit_price", {
        header: "Exit",
        cell: (info) => {
          const val = info.getValue();
          return (
            <span className="font-mono text-xs text-zinc-300">
              {val !== null ? formatCurrency(val) : "-"}
            </span>
          );
        },
      }),
      columnHelper.accessor("pnl", {
        header: "P&L ($)",
        cell: (info) => {
          const val = info.getValue();
          return (
            <span className={`font-mono text-xs font-semibold ${pnlColor(val)}`}>
              {val !== null ? formatCurrency(val) : "-"}
            </span>
          );
        },
      }),
      columnHelper.accessor("pnl_pct", {
        header: "P&L (%)",
        cell: (info) => {
          const val = info.getValue();
          return (
            <span className={`font-mono text-xs font-semibold ${pnlColor(val)}`}>
              {val !== null ? `${(val * 100).toFixed(2)}%` : "-"}
            </span>
          );
        },
      }),
      columnHelper.accessor("duration_seconds", {
        header: "Hold",
        cell: (info) => (
          <span className="font-mono text-xs text-zinc-400">
            {holdingDays(info.getValue())}
          </span>
        ),
      }),
    ],
    []
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
    <div className="overflow-auto rounded-lg border border-[#1a1a1a]">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow
              key={headerGroup.id}
              className="border-b border-[#1a1a1a] bg-[#0a0a0a] hover:bg-[#0a0a0a]"
            >
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className="cursor-pointer select-none text-[10px] uppercase tracking-wider text-zinc-500"
                  onClick={header.column.getToggleSortingHandler()}
                >
                  <span className="inline-flex items-center gap-1">
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                    <ArrowUpDown className="h-3 w-3 text-zinc-600" />
                  </span>
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
                className="py-8 text-center text-sm text-zinc-500"
              >
                No trades recorded
              </TableCell>
            </TableRow>
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="border-b border-[#1a1a1a] hover:bg-[#111]"
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} className="py-1.5">
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
