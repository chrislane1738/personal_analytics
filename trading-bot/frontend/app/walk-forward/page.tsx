"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  StopCircle,
  Trash2,
  ArrowUpDown,
} from "lucide-react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useStrategies } from "@/hooks/use-strategies";
import {
  useLaunchWalkForward,
  useWalkForwardStudies,
  useWalkForwardStatus,
  useStopWalkForward,
  useDeleteWalkForward,
} from "@/hooks/use-walk-forward";
import { formatPercent, formatDate, pnlColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { WalkForwardStudySummary } from "@/lib/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OBJECTIVES = ["sharpe", "sortino", "calmar", "total_return", "profit_factor"];

const OBJECTIVE_LABELS: Record<string, string> = {
  sharpe: "Sharpe",
  sortino: "Sortino",
  calmar: "Calmar",
  total_return: "Total Return",
  profit_factor: "Profit Factor",
};

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

function sharpeColor(value: number | null | undefined): string {
  if (value === null || value === undefined) return "text-zinc-400";
  if (value > 1) return "text-[#22c55e]";
  if (value > 0) return "text-[#eab308]";
  return "text-[#ef4444]";
}

function statusBadge(status: string) {
  switch (status) {
    case "running":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-orange-500/10 px-2 py-0.5 text-xs font-medium text-[#f97316]">
          <Loader2 className="h-3 w-3 animate-spin" />
          Running
        </span>
      );
    case "completed":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-xs font-medium text-[#22c55e]">
          <CheckCircle2 className="h-3 w-3" />
          Completed
        </span>
      );
    case "failed":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-red-500/10 px-2 py-0.5 text-xs font-medium text-[#ef4444]">
          <XCircle className="h-3 w-3" />
          Failed
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-zinc-500/10 px-2 py-0.5 text-xs font-medium text-zinc-400">
          {status}
        </span>
      );
  }
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

// ---------------------------------------------------------------------------
// Page Component
// ---------------------------------------------------------------------------

export default function WalkForwardPage() {
  const { data: strategies, isLoading: strategiesLoading } = useStrategies();
  const launchWF = useLaunchWalkForward();
  const stopWF = useStopWalkForward();
  const deleteWF = useDeleteWalkForward();
  const { data: studiesData, isLoading: studiesLoading } = useWalkForwardStudies();

  // Form state
  const [strategy, setStrategy] = useState("");
  const [universe, setUniverse] = useState("SPY");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");
  const [initialCapital, setInitialCapital] = useState(100000);
  const [benchmark, setBenchmark] = useState("SPY");
  const [trainMonths, setTrainMonths] = useState(24);
  const [oosMonths, setOosMonths] = useState(6);
  const [stepMonths, setStepMonths] = useState(3);
  const [objective, setObjective] = useState("sharpe");
  const [mcTradeShuffle, setMcTradeShuffle] = useState(true);
  const [mcWindowShuffle, setMcWindowShuffle] = useState(false);
  const [mcBootstrap, setMcBootstrap] = useState(true);
  const [simulations, setSimulations] = useState(10000);

  // Active study tracking
  const [activeStudyId, setActiveStudyId] = useState<string | undefined>(
    undefined,
  );
  const { data: studyStatus } = useWalkForwardStatus(activeStudyId);

  const handleLaunch = () => {
    if (!strategy) return;

    const modes: string[] = [];
    if (mcTradeShuffle) modes.push("trade_shuffle");
    if (mcWindowShuffle) modes.push("window_shuffle");
    if (mcBootstrap) modes.push("bootstrap");

    launchWF.mutate(
      {
        strategy,
        universe,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        benchmark,
        train_months: trainMonths,
        oos_months: oosMonths,
        step_months: stepMonths,
        objective,
        monte_carlo_modes: modes,
        simulations,
      },
      {
        onSuccess: (result) => {
          setActiveStudyId(result.study_id);
        },
      },
    );
  };

  const handleStop = () => {
    if (activeStudyId) {
      stopWF.mutate(activeStudyId);
    }
  };

  const handleDelete = (studyId: string) => {
    deleteWF.mutate(studyId);
  };

  const isRunning = studyStatus?.status === "running";
  const isCompleted = studyStatus?.status === "completed";
  const isFailed = studyStatus?.status === "failed";
  const isCancelled = studyStatus?.status === "cancelled";

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
          Walk-Forward Analysis
        </h1>
        <p className="mt-1 text-xs text-zinc-500">
          Launch walk-forward optimization studies and review results
        </p>
      </div>

      {/* Configuration Form */}
      <div className="grid max-w-2xl gap-6">
        {/* Strategy Picker */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-zinc-400">Strategy</label>
          {strategiesLoading ? (
            <p className="text-sm text-zinc-500">Loading strategies...</p>
          ) : (
            <Select
              value={strategy}
              onValueChange={(v) => setStrategy(v ?? "")}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a strategy..." />
              </SelectTrigger>
              <SelectContent>
                {strategies?.map((s) => (
                  <SelectItem key={s.name} value={s.name}>
                    {s.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        {/* Universe */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-zinc-400">
            Universe (comma-separated symbols)
          </label>
          <Input
            value={universe}
            onChange={(e) => setUniverse(e.target.value)}
            placeholder="SPY"
          />
        </div>

        {/* Date Range */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Start Date
            </label>
            <Input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              End Date
            </label>
            <Input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
        </div>

        {/* Capital & Benchmark */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Initial Capital ($)
            </label>
            <Input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              min={1000}
              step={1000}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Benchmark
            </label>
            <Input
              value={benchmark}
              onChange={(e) => setBenchmark(e.target.value)}
              placeholder="SPY"
            />
          </div>
        </div>

        {/* Window Configuration */}
        <div className="grid grid-cols-3 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Train Window (months)
            </label>
            <Input
              type="number"
              value={trainMonths}
              onChange={(e) => setTrainMonths(Number(e.target.value))}
              min={1}
              step={1}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              OOS Window (months)
            </label>
            <Input
              type="number"
              value={oosMonths}
              onChange={(e) => setOosMonths(Number(e.target.value))}
              min={1}
              step={1}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-zinc-400">
              Step Size (months)
            </label>
            <Input
              type="number"
              value={stepMonths}
              onChange={(e) => setStepMonths(Number(e.target.value))}
              min={1}
              step={1}
            />
          </div>
        </div>

        {/* Objective */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-zinc-400">
            Objective
          </label>
          <Select
            value={objective}
            onValueChange={(v) => setObjective(v ?? "sharpe")}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {OBJECTIVES.map((obj) => (
                <SelectItem key={obj} value={obj}>
                  {OBJECTIVE_LABELS[obj]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Monte Carlo Options */}
        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium text-zinc-400">
            Monte Carlo Modes
          </label>
          <div className="flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
              <input
                type="checkbox"
                checked={mcTradeShuffle}
                onChange={(e) => setMcTradeShuffle(e.target.checked)}
                className="h-4 w-4 rounded border-zinc-600 bg-zinc-900 text-[#f97316] accent-[#f97316]"
              />
              Trade Shuffle
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
              <input
                type="checkbox"
                checked={mcWindowShuffle}
                onChange={(e) => setMcWindowShuffle(e.target.checked)}
                className="h-4 w-4 rounded border-zinc-600 bg-zinc-900 text-[#f97316] accent-[#f97316]"
              />
              Window Shuffle
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
              <input
                type="checkbox"
                checked={mcBootstrap}
                onChange={(e) => setMcBootstrap(e.target.checked)}
                className="h-4 w-4 rounded border-zinc-600 bg-zinc-900 text-[#f97316] accent-[#f97316]"
              />
              Bootstrap
            </label>
          </div>
        </div>

        {/* Simulations */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-zinc-400">
            Simulations
          </label>
          <Input
            type="number"
            value={simulations}
            onChange={(e) => setSimulations(Number(e.target.value))}
            min={100}
            step={1000}
            className="max-w-48"
          />
        </div>

        {/* Launch Button */}
        <div className="flex items-center gap-3">
          <Button
            onClick={handleLaunch}
            disabled={!strategy || launchWF.isPending || isRunning}
            size="lg"
          >
            {launchWF.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="ml-1.5">Launching...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span className="ml-1.5">Run Study</span>
              </>
            )}
          </Button>
          {isRunning && (
            <Button variant="destructive" size="lg" onClick={handleStop}>
              <StopCircle className="h-4 w-4" />
              <span className="ml-1.5">Stop</span>
            </Button>
          )}
        </div>

        {/* Launch Error */}
        {launchWF.error && (
          <div className="rounded-md border border-red-900 bg-red-950/50 px-4 py-3 text-sm text-red-400">
            Launch failed: {launchWF.error.message}
          </div>
        )}
      </div>

      {/* Study Status */}
      {activeStudyId && studyStatus && (
        <div className="max-w-2xl rounded-lg border border-[#1a1a1a] bg-[#0a0a0a] p-4">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Study Status
          </h2>
          <div className="flex items-center gap-3">
            {isRunning && (
              <>
                <Loader2 className="h-5 w-5 animate-spin text-[#f97316]" />
                <span className="text-sm text-[#f97316]">Running...</span>
              </>
            )}
            {isCompleted && (
              <>
                <CheckCircle2 className="h-5 w-5 text-[#22c55e]" />
                <span className="text-sm text-[#22c55e]">Completed</span>
              </>
            )}
            {isFailed && (
              <>
                <XCircle className="h-5 w-5 text-[#ef4444]" />
                <span className="text-sm text-[#ef4444]">Failed</span>
              </>
            )}
            {isCancelled && (
              <>
                <StopCircle className="h-5 w-5 text-zinc-400" />
                <span className="text-sm text-zinc-400">Cancelled</span>
              </>
            )}
            <span className="font-mono text-xs text-zinc-600">
              ID: {activeStudyId}
            </span>
          </div>

          {/* Progress */}
          {isRunning &&
            studyStatus.windows_total > 0 && (
              <div className="mt-3 text-sm text-zinc-400">
                Window {studyStatus.windows_completed}/{studyStatus.windows_total}
                {studyStatus.current_phase && (
                  <span className="ml-1 text-zinc-500">
                    &mdash; {studyStatus.current_phase}
                  </span>
                )}
              </div>
            )}

          {isCompleted && (
            <div className="mt-3">
              <Link
                href={`/walk-forward/${studyStatus?.study_id ?? activeStudyId}`}
                className="text-sm text-[#f97316] underline-offset-4 hover:underline"
              >
                View Results
              </Link>
            </div>
          )}
        </div>
      )}

      {/* Previous Studies Table */}
      <div>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Previous Studies
        </h2>
        {studiesLoading ? (
          <p className="text-sm text-zinc-500">Loading studies...</p>
        ) : (
          <StudiesTable
            data={studiesData?.studies ?? []}
            onDelete={handleDelete}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Studies DataTable
// ---------------------------------------------------------------------------

function StudiesTable({
  data,
  onDelete,
}: {
  data: WalkForwardStudySummary[];
  onDelete: (studyId: string) => void;
}) {
  const router = useRouter();
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo<ColumnDef<WalkForwardStudySummary>[]>(
    () => [
      {
        accessorKey: "study_id",
        header: ({ column }) => (
          <SortableHeader label="Study ID" column={column} />
        ),
        cell: ({ row }) => {
          const id = row.original.study_id;
          return (
            <Link
              href={`/walk-forward/${id}`}
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
        id: "date_range",
        header: "Date Range",
        accessorFn: (row) => row.start_date,
        cell: ({ row }) => (
          <span className="text-sm text-zinc-300">
            {formatDate(row.original.start_date)} &mdash;{" "}
            {formatDate(row.original.end_date)}
          </span>
        ),
      },
      {
        id: "window_config",
        header: "Windows",
        accessorFn: (row) => row.train_months,
        cell: ({ row }) => (
          <span className="font-mono text-xs text-zinc-400">
            {row.original.train_months}/{row.original.oos_months}/
            {row.original.step_months}mo
          </span>
        ),
      },
      {
        accessorKey: "objective",
        header: ({ column }) => (
          <SortableHeader label="Objective" column={column} />
        ),
        cell: ({ getValue }) => (
          <span className="text-sm text-zinc-300 capitalize">
            {OBJECTIVE_LABELS[getValue<string>()] ?? getValue<string>()}
          </span>
        ),
      },
      {
        id: "oos_return",
        header: ({ column }) => (
          <SortableHeader label="OOS Return" column={column} />
        ),
        accessorFn: (row) => row.aggregate?.total_return ?? null,
        cell: ({ getValue }) => {
          const value = getValue<number | null>();
          return (
            <span className={cn("font-mono text-sm", pnlColor(value))}>
              {value !== null && value !== undefined
                ? formatPercent(value)
                : "\u2014"}
            </span>
          );
        },
      },
      {
        id: "oos_sharpe",
        header: ({ column }) => (
          <SortableHeader label="OOS Sharpe" column={column} />
        ),
        accessorFn: (row) => row.aggregate?.sharpe ?? null,
        cell: ({ getValue }) => {
          const value = getValue<number | null>();
          return (
            <span className={cn("font-mono text-sm", sharpeColor(value))}>
              {value !== null && value !== undefined ? value.toFixed(2) : "\u2014"}
            </span>
          );
        },
      },
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ getValue }) => statusBadge(getValue<string>()),
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
              onDelete(row.original.study_id);
            }}
            aria-label="Delete study"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        ),
        enableSorting: false,
      },
    ],
    [onDelete],
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
                        header.getContext(),
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
                No walk-forward studies found.
              </TableCell>
            </TableRow>
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="cursor-pointer border-[#1a1a1a] hover:bg-zinc-900/50 transition-colors"
                onClick={() =>
                  router.push(`/walk-forward/${row.original.study_id}`)
                }
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(
                      cell.column.columnDef.cell,
                      cell.getContext(),
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
