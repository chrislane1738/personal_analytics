"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { MonitorContent } from "./monitor-content";

function MonitorWithParams() {
  const searchParams = useSearchParams();
  const runId = searchParams.get("runId");
  return <MonitorContent initialRunId={runId} />;
}

export default function MonitorPage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-full items-center justify-center text-sm text-zinc-500">
          Loading monitor...
        </div>
      }
    >
      <MonitorWithParams />
    </Suspense>
  );
}
