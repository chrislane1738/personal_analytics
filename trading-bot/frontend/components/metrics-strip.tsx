"use client";

interface Metric {
  label: string;
  value: string;
  colorClass?: string;
}

interface MetricsStripProps {
  metrics: Metric[];
}

export function MetricsStrip({ metrics }: MetricsStripProps) {
  return (
    <div
      className="grid gap-2"
      style={{
        gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
      }}
    >
      {metrics.map((metric) => (
        <div
          key={metric.label}
          className="rounded-lg border border-[#1a1a1a] bg-[#0f0f0f] px-3 py-2.5"
        >
          <p className="font-sans text-[9px] uppercase tracking-wider text-zinc-500">
            {metric.label}
          </p>
          <p
            className={`mt-0.5 font-mono text-lg font-bold ${metric.colorClass ?? "text-zinc-100"}`}
          >
            {metric.value}
          </p>
        </div>
      ))}
    </div>
  );
}
