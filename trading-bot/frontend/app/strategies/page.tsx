"use client";

import { useState, useEffect } from "react";
import { Settings2, Plus, Save, Trash2, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  useStrategies,
  useSaveStrategy,
  useCreateStrategy,
  useDeleteStrategy,
} from "@/hooks/use-strategies";

const STARTER_YAML = `name: "New Strategy"
universe: ["SPY"]
timeframe: daily

indicators:
  sma_20: { type: SMA, period: 20 }
  rsi_14: { type: RSI, period: 14 }

entry_rules:
  - condition: "rsi_14 < 30"
    direction: long
    reason: "Oversold RSI"

exit_rules:
  - condition: "rsi_14 > 70"
    reason: "Overbought RSI"
  - condition: "pnl_pct < -0.05"
    reason: "Stop loss at -5%"

position_sizing:
  method: fixed_pct
  value: 0.06
`;

export default function StrategiesPage() {
  const { data: strategies, isLoading, error } = useStrategies();
  const saveStrategy = useSaveStrategy();
  const createStrategy = useCreateStrategy();
  const deleteStrategy = useDeleteStrategy();

  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState("");
  const [isDirty, setIsDirty] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  // When strategies load, select the first one
  useEffect(() => {
    if (strategies && strategies.length > 0 && selectedName === null) {
      setSelectedName(strategies[0].name);
    }
  }, [strategies, selectedName]);

  // When selection changes, load content into editor
  useEffect(() => {
    if (strategies && selectedName) {
      const file = strategies.find((s) => s.name === selectedName);
      if (file) {
        setEditorContent(file.content);
        setIsDirty(false);
      }
    }
  }, [selectedName, strategies]);

  const handleSave = () => {
    if (!selectedName) return;
    saveStrategy.mutate(
      { name: selectedName, content: editorContent },
      {
        onSuccess: () => {
          setIsDirty(false);
          setSaveMessage("Saved");
          setTimeout(() => setSaveMessage(null), 2000);
        },
      }
    );
  };

  const handleCreate = () => {
    const name = prompt("Strategy name (letters, numbers, hyphens, underscores):");
    if (!name) return;
    createStrategy.mutate(
      { name, content: STARTER_YAML },
      {
        onSuccess: (result) => {
          setSelectedName(result.name);
          setEditorContent(STARTER_YAML);
          setIsDirty(false);
        },
      }
    );
  };

  const handleDelete = () => {
    if (!selectedName) return;
    if (!confirm(`Delete strategy "${selectedName}"? This cannot be undone.`)) return;
    deleteStrategy.mutate(selectedName, {
      onSuccess: () => {
        setSelectedName(null);
        setEditorContent("");
        setIsDirty(false);
      },
    });
  };

  return (
    <div className="flex h-full flex-col p-6">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-medium tracking-tight text-[#fafafa]">
            Strategy Editor
          </h1>
          <p className="mt-1 text-xs text-zinc-500">
            Create and edit YAML strategy configurations
          </p>
        </div>
      </div>

      {isLoading && (
        <p className="text-sm text-zinc-500">Loading strategies...</p>
      )}

      {error && (
        <p className="text-sm text-red-500">
          Failed to load strategies: {error.message}
        </p>
      )}

      {strategies && (
        <div className="flex flex-1 gap-4 overflow-hidden">
          {/* Left panel: file list */}
          <div className="flex w-[40%] min-w-[240px] flex-col rounded-lg border border-[#1a1a1a] bg-[#0a0a0a]">
            <div className="flex items-center justify-between border-b border-[#1a1a1a] px-4 py-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Strategies ({strategies.length})
              </h2>
              <Button variant="ghost" size="icon-xs" onClick={handleCreate}>
                <Plus className="h-3.5 w-3.5" />
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              {strategies.length === 0 ? (
                <p className="px-2 py-4 text-center text-sm text-zinc-600">
                  No strategies yet
                </p>
              ) : (
                <div className="flex flex-col gap-1">
                  {strategies.map((file) => (
                    <button
                      key={file.name}
                      type="button"
                      onClick={() => setSelectedName(file.name)}
                      className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors ${
                        selectedName === file.name
                          ? "bg-[#18181b] text-[#fafafa]"
                          : "text-zinc-400 hover:bg-[#18181b] hover:text-zinc-300"
                      }`}
                    >
                      <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
                      <span className="truncate font-mono text-xs">
                        {file.name}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right panel: editor */}
          <div className="flex flex-1 flex-col rounded-lg border border-[#1a1a1a] bg-[#0a0a0a]">
            {selectedName ? (
              <>
                {/* Toolbar */}
                <div className="flex items-center justify-between border-b border-[#1a1a1a] px-4 py-2">
                  <div className="flex items-center gap-2">
                    <Settings2 className="h-3.5 w-3.5 text-zinc-600" />
                    <span className="font-mono text-xs text-zinc-300">
                      {selectedName}
                    </span>
                    {isDirty && (
                      <span className="rounded bg-[#f97316]/10 px-1.5 py-0.5 text-[10px] font-medium text-[#f97316]">
                        MODIFIED
                      </span>
                    )}
                    {saveMessage && (
                      <span className="text-[10px] font-medium text-[#22c55e]">
                        {saveMessage}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleSave}
                      disabled={!isDirty || saveStrategy.isPending}
                    >
                      <Save className="h-3.5 w-3.5" />
                      <span className="ml-1">Save</span>
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleDelete}
                      disabled={deleteStrategy.isPending}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      <span className="ml-1">Delete</span>
                    </Button>
                  </div>
                </div>

                {/* Textarea editor */}
                <div className="flex-1 p-2">
                  <textarea
                    value={editorContent}
                    onChange={(e) => {
                      setEditorContent(e.target.value);
                      setIsDirty(true);
                    }}
                    onKeyDown={(e) => {
                      // Ctrl/Cmd+S to save
                      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
                        e.preventDefault();
                        if (isDirty) handleSave();
                      }
                      // Tab inserts 2 spaces
                      if (e.key === "Tab") {
                        e.preventDefault();
                        const target = e.target as HTMLTextAreaElement;
                        const start = target.selectionStart;
                        const end = target.selectionEnd;
                        const newContent =
                          editorContent.substring(0, start) +
                          "  " +
                          editorContent.substring(end);
                        setEditorContent(newContent);
                        setIsDirty(true);
                        // Restore cursor after React re-render
                        requestAnimationFrame(() => {
                          target.selectionStart = start + 2;
                          target.selectionEnd = start + 2;
                        });
                      }
                    }}
                    spellCheck={false}
                    className="h-full w-full resize-none rounded-md border border-[#1a1a1a] bg-[#0f0f0f] p-4 font-mono text-xs leading-relaxed text-zinc-300 outline-none transition-colors focus:border-zinc-700 placeholder:text-zinc-700"
                    placeholder="# YAML strategy configuration..."
                  />
                </div>
              </>
            ) : (
              <div className="flex flex-1 items-center justify-center text-sm text-zinc-600">
                Select a strategy from the left panel or create a new one
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error toasts */}
      {createStrategy.error && (
        <div className="fixed bottom-4 right-4 rounded-md border border-red-900 bg-red-950 px-4 py-2 text-sm text-red-400">
          Create failed: {createStrategy.error.message}
        </div>
      )}
      {saveStrategy.error && (
        <div className="fixed bottom-4 right-4 rounded-md border border-red-900 bg-red-950 px-4 py-2 text-sm text-red-400">
          Save failed: {saveStrategy.error.message}
        </div>
      )}
      {deleteStrategy.error && (
        <div className="fixed bottom-4 right-4 rounded-md border border-red-900 bg-red-950 px-4 py-2 text-sm text-red-400">
          Delete failed: {deleteStrategy.error.message}
        </div>
      )}
    </div>
  );
}
