"use client";

import { useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";
import { ArrowUpDown } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { SymbolInfo } from "@/hooks/use-data";

interface SymbolsTableProps {
  data: SymbolInfo[];
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

function formatNumber(n: number): string {
  return new Intl.NumberFormat("en-US").format(n);
}

function formatMarketCap(value: number | null): string {
  if (value === null || value === 0) return "--";
  if (value >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(0)}M`;
  return `$${formatNumber(value)}`;
}

export function SymbolsTable({ data }: SymbolsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo<ColumnDef<SymbolInfo>[]>(
    () => [
      {
        accessorKey: "symbol",
        header: ({ column }) => (
          <SortableHeader label="Symbol" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="font-mono text-sm font-semibold text-[#f97316]">
            {getValue<string>()}
          </span>
        ),
      },
      {
        accessorKey: "company_name",
        header: ({ column }) => (
          <SortableHeader label="Company" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="text-sm text-zinc-300 truncate max-w-[200px] block">
            {getValue<string | null>() ?? "--"}
          </span>
        ),
      },
      {
        accessorKey: "sector",
        header: ({ column }) => (
          <SortableHeader label="Sector" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="text-xs text-zinc-400">
            {getValue<string | null>() ?? "--"}
          </span>
        ),
      },
      {
        accessorKey: "bar_count",
        header: ({ column }) => (
          <SortableHeader label="Bars" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="font-mono text-sm text-zinc-300">
            {formatNumber(getValue<number>())}
          </span>
        ),
      },
      {
        id: "date_range",
        header: "Date Range",
        accessorFn: (row) => row.min_date,
        cell: ({ row }) => {
          const min = row.original.min_date;
          const max = row.original.max_date;
          if (!min || !max) return <span className="text-zinc-500">--</span>;
          return (
            <span className="font-mono text-xs text-zinc-400">
              {min} to {max}
            </span>
          );
        },
      },
      {
        accessorKey: "market_cap",
        header: ({ column }) => (
          <SortableHeader label="Mkt Cap" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="font-mono text-xs text-zinc-400">
            {formatMarketCap(getValue<number | null>())}
          </span>
        ),
      },
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
                No symbols cached yet. Use the fetch panel to add data.
              </TableCell>
            </TableRow>
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="border-[#1a1a1a] hover:bg-zinc-900/50 transition-colors"
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(
                      cell.column.columnDef.cell,
                      cell.getContext()
                    )}
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
