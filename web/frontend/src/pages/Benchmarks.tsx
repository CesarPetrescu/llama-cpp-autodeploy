import { FormEvent, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  BenchmarkRecord,
  BenchmarkRow,
  GpuInfo,
} from "@/api/client";
import { LogPane } from "@/components/LogPane";
import { ModelSelect } from "@/components/ModelSelect";
import { PageHeader } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { useManagedWebSocket } from "@/hooks/useManagedWebSocket";

type BenchmarkFormState = {
  name: string;
  modelRef: string;
  hfToken: string;
  gpuDevices: string;
  repetitions: number;
  delay: number;
  nPrompt: string;
  nGen: string;
  pg: string;
  nDepth: string;
  batchSize: string;
  ubatchSize: string;
  threads: string;
  nGpuLayers: string;
  splitMode: string;
  mainGpu: string;
  tensorSplit: string;
  flashAttn: boolean;
  embeddings: boolean;
  noKvOffload: boolean;
  noWarmup: boolean;
  extraFlags: string;
};

const DEFAULT_FORM: BenchmarkFormState = {
  name: "",
  modelRef: "",
  hfToken: "",
  gpuDevices: "",
  repetitions: 5,
  delay: 0,
  nPrompt: "512",
  nGen: "128",
  pg: "",
  nDepth: "0",
  batchSize: "2048",
  ubatchSize: "512",
  threads: "8",
  nGpuLayers: "99",
  splitMode: "layer",
  mainGpu: "0",
  tensorSplit: "",
  flashAttn: true,
  embeddings: false,
  noKvOffload: false,
  noWarmup: false,
  extraFlags: "",
};

const CONTROL_NOTES = [
  {
    label: "Start benchmark",
    detail: "Resolves the selected GGUF, pins the chosen GPU set, and launches llama-bench with structured JSON capture.",
  },
  {
    label: "History row",
    detail: "Loads the stored command, parsed throughput rows, durable log, and result summary for that run.",
  },
  {
    label: "Stop benchmark",
    detail: "Sends a stop request to the benchmark process group. Partial results may still be unavailable if the run is interrupted early.",
  },
  {
    label: "Prompt / gen / pg",
    detail: "Use n_prompt and n_gen for the usual pp/tg runs. Fill pg when you also want a combined prefill+generation test in the same job.",
  },
  {
    label: "GPU devices",
    detail: "This maps to CUDA_VISIBLE_DEVICES before llama-bench starts. Use the CUDA indices shown below; if the host order differs, the nvidia-smi index is shown beside it.",
  },
  {
    label: "Extra flags",
    detail: "Use this for advanced llama-bench options like cache types, NUMA, polling, or priority without changing the UI schema.",
  },
];

function formatRelativeTime(timestamp: number | null): string {
  if (!timestamp) return "—";
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatDuration(record: BenchmarkRecord | null): string {
  if (!record?.started_at) return "—";
  const end = record.finished_at ?? Date.now() / 1000;
  const seconds = Math.max(0, Math.floor(end - record.started_at));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatTimestamp(timestamp: number | null): string {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleString();
}

function formatTs(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)} t/s`;
}

function isActive(status?: BenchmarkRecord["status"]): boolean {
  return status === "running" || status === "cancelling";
}

function gpuLabel(gpu: GpuInfo): string {
  const vram = gpu.total_h ? ` · ${gpu.total_h}` : "";
  const systemIndex =
    gpu.system_index != null && gpu.system_index !== gpu.index
      ? ` · nvidia-smi #${gpu.system_index}`
      : "";
  return `CUDA #${gpu.index}${systemIndex} · ${gpu.name}${vram}`;
}

function benchmarkCommandPreview(form: BenchmarkFormState): string {
  const prefix = form.gpuDevices.trim()
    ? [`CUDA_VISIBLE_DEVICES=${form.gpuDevices.trim()}`]
    : [];
  const cmd = ["llama-bench", "-m", form.modelRef || "<model>", "-o", "json", "-oe", "md", "--progress"];
  cmd.push("-r", String(form.repetitions));
  cmd.push("--delay", String(form.delay));
  cmd.push("-p", form.nPrompt || "512");
  cmd.push("-n", form.nGen || "128");
  if (form.pg.trim()) cmd.push("-pg", form.pg.trim());
  cmd.push("-d", form.nDepth || "0");
  cmd.push("-b", form.batchSize || "2048");
  cmd.push("-ub", form.ubatchSize || "512");
  cmd.push("-t", form.threads || "8");
  cmd.push("-ngl", form.nGpuLayers || "99");
  cmd.push("-sm", form.splitMode || "layer");
  cmd.push("-mg", form.mainGpu || "0");
  if (form.tensorSplit.trim()) cmd.push("-ts", form.tensorSplit.trim());
  cmd.push("-fa", form.flashAttn ? "1" : "0");
  if (form.embeddings) cmd.push("-embd", "1");
  if (form.noKvOffload) cmd.push("-nkvo", "1");
  if (form.noWarmup) cmd.push("--no-warmup");
  if (form.extraFlags.trim()) cmd.push(form.extraFlags.trim());
  return [...prefix, ...cmd].join(" ");
}

export default function Benchmarks() {
  const qc = useQueryClient();
  const [form, setForm] = useState<BenchmarkFormState>(DEFAULT_FORM);
  const [selected, setSelected] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [formError, setFormError] = useState<string | null>(null);

  const benchmarks = useQuery({
    queryKey: ["benchmarks"],
    queryFn: api.listBenchmarks,
    refetchInterval: 4000,
  });
  const gpus = useQuery({
    queryKey: ["gpus"],
    queryFn: api.listGpus,
    refetchInterval: 10000,
  });
  const selectedDetails = useQuery({
    queryKey: ["benchmark", selected],
    queryFn: () => api.getBenchmark(selected || ""),
    enabled: !!selected,
  });

  const benchmarkList = [...(benchmarks.data?.benchmarks ?? [])].sort(
    (a, b) => (b.started_at ?? 0) - (a.started_at ?? 0),
  );
  const selectedBenchmark =
    benchmarkList.find((item) => item.id === selected) ??
    selectedDetails.data?.benchmark ??
    null;
  const runningCount = benchmarkList.filter((item) => item.status === "running").length;
  const latestBenchmark = benchmarkList[0] ?? null;
  const fastestOverall = benchmarkList
    .map((item) => item.summary?.best_overall?.avg_ts ?? null)
    .filter((value): value is number => value != null)
    .sort((a, b) => b - a)[0] ?? null;
  const fastestGeneration = benchmarkList
    .map((item) => item.summary?.best_tg?.avg_ts ?? null)
    .filter((value): value is number => value != null)
    .sort((a, b) => b - a)[0] ?? null;

  const start = useMutation({
    mutationFn: () =>
      api.startBenchmark({
        name: form.name || null,
        model_ref: form.modelRef,
        hf_token: form.hfToken || null,
        gpu_devices: form.gpuDevices,
        repetitions: form.repetitions,
        delay: form.delay,
        n_prompt: form.nPrompt,
        n_gen: form.nGen,
        pg: form.pg,
        n_depth: form.nDepth,
        batch_size: form.batchSize,
        ubatch_size: form.ubatchSize,
        threads: form.threads,
        n_gpu_layers: form.nGpuLayers,
        split_mode: form.splitMode,
        main_gpu: form.mainGpu,
        tensor_split: form.tensorSplit,
        flash_attn: form.flashAttn,
        embeddings: form.embeddings,
        no_kv_offload: form.noKvOffload,
        no_warmup: form.noWarmup,
        extra_flags: form.extraFlags,
      }),
    onSuccess: (data) => {
      setFormError(null);
      setSelected(data.benchmark.id);
      setLines([]);
      qc.invalidateQueries({ queryKey: ["benchmarks"] });
      qc.invalidateQueries({ queryKey: ["benchmark", data.benchmark.id] });
    },
    onError: (err: Error) => setFormError(err.message),
  });

  const stop = useMutation({
    mutationFn: (id: string) => api.stopBenchmark(id),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["benchmarks"] });
      qc.invalidateQueries({ queryKey: ["benchmark", data.benchmark.id] });
    },
    onError: (err: Error) => setFormError(err.message),
  });

  useEffect(() => {
    if (!selected && benchmarkList.length > 0) {
      setSelected(benchmarkList[0].id);
      return;
    }
    if (selected && benchmarkList.length > 0 && !benchmarkList.some((item) => item.id === selected)) {
      setSelected(benchmarkList[0].id);
    }
  }, [benchmarkList, selected]);

  useEffect(() => {
    setLines([]);
  }, [selected]);

  useEffect(() => {
    if (!selectedDetails.data || selectedDetails.data.benchmark.id !== selected || lines.length > 0) return;
    setLines(selectedDetails.data.logs);
  }, [selected, selectedDetails.data, lines.length]);

  const socket = useManagedWebSocket({
    path:
      selected && isActive(selectedBenchmark?.status)
        ? `/api/benchmarks/${selected}/logs?history=0`
        : null,
    enabled: Boolean(selected && isActive(selectedBenchmark?.status)),
    beforeConnect: async (phase) => {
      if (phase !== "reconnect" || !selected) return;
      const response = await api.getBenchmark(selected);
      qc.setQueryData(["benchmark", selected], response);
      setLines(response.logs);
    },
    onMessage: (ev) => {
      setLines((prev) => {
        const next = [...prev, String(ev.data)];
        if (next.length > 5000) next.splice(0, next.length - 5000);
        return next;
      });
    },
  });

  function submit(e: FormEvent) {
    e.preventDefault();
    setFormError(null);
    start.mutate();
  }

  function update<K extends keyof BenchmarkFormState>(key: K, value: BenchmarkFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow="Performance"
        title="Benchmarks"
        description="Run llama-bench against a local or Hugging Face GGUF, pin the job to a chosen GPU set, and keep structured throughput history in the control plane."
      />

      <div className="grid gap-4 grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="History"
          value={`${benchmarkList.length}`}
          meta={[
            latestBenchmark ? `latest ${latestBenchmark.id}` : "no runs yet",
            latestBenchmark ? formatRelativeTime(latestBenchmark.started_at) : "—",
          ]}
        />
        <MetricCard
          label="Running"
          value={`${runningCount}`}
          meta={[
            latestBenchmark?.config?.gpu_devices
              ? `gpu ${String(latestBenchmark.config.gpu_devices)}`
              : "gpu auto",
            latestBenchmark ? formatDuration(latestBenchmark) : "—",
          ]}
          tone={runningCount > 0 ? "warning" : "default"}
        />
        <MetricCard
          label="Best tg"
          value={formatTs(fastestGeneration)}
          meta={[
            latestBenchmark?.summary?.best_tg?.test ?? "no tg result",
            latestBenchmark?.summary?.backend ?? "—",
          ]}
          tone={fastestGeneration ? "success" : "default"}
        />
        <MetricCard
          label="Best overall"
          value={formatTs(fastestOverall)}
          meta={[
            latestBenchmark?.summary?.best_overall?.test ?? "no result",
            latestBenchmark?.summary?.model_type ?? "—",
          ]}
          tone={fastestOverall ? "success" : "default"}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Panel title="Benchmark configuration">
          <form onSubmit={submit} className="space-y-4">
            <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
              <div className="border border-white/10 bg-ink-300/70 p-4">
                <div className="brand-label">Model target</div>
                <div className="mt-2">
                  <ModelSelect
                    value={form.modelRef}
                    onChange={(value) => update("modelRef", value)}
                    manualPlaceholder="or paste org/repo:quant (for example unsloth/Qwen3.5-0.8B-GGUF:Q4_K_XL)"
                  />
                </div>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  <input
                    value={form.name}
                    onChange={(e) => update("name", e.target.value)}
                    className="brand-input"
                    placeholder="Run name (optional)"
                  />
                  <input
                    value={form.hfToken}
                    onChange={(e) => update("hfToken", e.target.value)}
                    className="brand-input"
                    placeholder="hf_token (optional)"
                    type="password"
                  />
                </div>
              </div>

              <div className="border border-white/10 bg-ink-300/70 p-4">
                <div className="brand-label">Device targeting</div>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  <Field
                    label="CUDA_VISIBLE_DEVICES"
                    value={form.gpuDevices}
                    onChange={(value) => update("gpuDevices", value)}
                    placeholder="auto or e.g. 1"
                  />
                  <Field
                    label="Main GPU"
                    value={form.mainGpu}
                    onChange={(value) => update("mainGpu", value)}
                    placeholder="0"
                  />
                  <Field
                    label="n_gpu_layers"
                    value={form.nGpuLayers}
                    onChange={(value) => update("nGpuLayers", value)}
                    placeholder="99"
                  />
                  <Field
                    label="tensor-split"
                    value={form.tensorSplit}
                    onChange={(value) => update("tensorSplit", value)}
                    placeholder="optional"
                  />
                  <label className="flex flex-col gap-2 text-sm">
                    <span className="brand-label">split-mode</span>
                    <select
                      className="brand-input"
                      value={form.splitMode}
                      onChange={(e) => update("splitMode", e.target.value)}
                    >
                      <option value="layer">layer</option>
                      <option value="none">none</option>
                      <option value="row">row</option>
                      <option value="tensor">tensor</option>
                    </select>
                  </label>
                  <ToggleField
                    label="Flash attention"
                    checked={form.flashAttn}
                    onChange={(checked) => update("flashAttn", checked)}
                  />
                </div>
                {gpus.data && (
                  <div className="mt-3 border border-white/10 bg-ink-400/72 p-3">
                    <div className="brand-label">Detected GPUs</div>
                    <div className="mt-2 space-y-1 text-xs text-bone-300">
                      {gpus.data.gpus.map((gpu) => (
                        <div key={gpu.index}>{gpuLabel(gpu)}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <Field label="repetitions" value={String(form.repetitions)} onChange={(value) => update("repetitions", Number(value) || 1)} />
              <Field label="delay (s)" value={String(form.delay)} onChange={(value) => update("delay", Number(value) || 0)} />
              <Field label="threads" value={form.threads} onChange={(value) => update("threads", value)} />
              <Field label="n_depth" value={form.nDepth} onChange={(value) => update("nDepth", value)} />
              <Field label="n_prompt" value={form.nPrompt} onChange={(value) => update("nPrompt", value)} />
              <Field label="n_gen" value={form.nGen} onChange={(value) => update("nGen", value)} />
              <Field label="pg (pp,tg)" value={form.pg} onChange={(value) => update("pg", value)} placeholder="optional" />
              <Field label="batch-size" value={form.batchSize} onChange={(value) => update("batchSize", value)} />
              <Field label="ubatch-size" value={form.ubatchSize} onChange={(value) => update("ubatchSize", value)} />
              <ToggleField
                label="embeddings"
                checked={form.embeddings}
                onChange={(checked) => update("embeddings", checked)}
              />
              <ToggleField
                label="no-kv-offload"
                checked={form.noKvOffload}
                onChange={(checked) => update("noKvOffload", checked)}
              />
              <ToggleField
                label="no-warmup"
                checked={form.noWarmup}
                onChange={(checked) => update("noWarmup", checked)}
              />
            </div>

            <div className="border border-white/10 bg-ink-300/70 p-4">
              <div className="brand-label">extra_flags</div>
              <textarea
                value={form.extraFlags}
                onChange={(e) => update("extraFlags", e.target.value)}
                className="brand-input mt-2 min-h-24"
                placeholder="--numa distribute --poll 0"
              />
            </div>

            <div className="border border-white/10 bg-ink-400/72 p-4">
              <div className="brand-label">Command preview</div>
              <div className="brand-code-block brand-scroll-pane mt-2 max-h-[8rem]">
                {benchmarkCommandPreview(form)}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={start.isPending || !form.modelRef.trim()}
                className="brand-btn-primary"
              >
                {start.isPending ? "Starting…" : "Start benchmark"}
              </button>
              {formError && <span className="text-sm text-rose-300">{formError}</span>}
            </div>
          </form>
        </Panel>

        <Panel title="Controls and notes">
          <div className="space-y-2">
            {CONTROL_NOTES.map((item) => (
              <div key={item.label} className="border border-white/10 bg-ink-300/70 px-3 py-3">
                <div className="text-sm font-semibold text-bone-50">{item.label}</div>
                <div className="mt-1 text-xs leading-5 text-bone-300">{item.detail}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.82fr_1.18fr]">
        <Panel title="Benchmark history">
          <div className="brand-table-wrap">
            <table className="min-w-[760px] w-full text-sm">
              <thead className="bg-ink-400/78">
                <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                  <th className="px-4 py-3 font-semibold">Run</th>
                  <th className="px-4 py-3 font-semibold">Model</th>
                  <th className="px-4 py-3 font-semibold">GPU</th>
                  <th className="px-4 py-3 font-semibold">Status</th>
                  <th className="px-4 py-3 font-semibold">Best tg</th>
                  <th className="px-4 py-3 font-semibold">Started</th>
                </tr>
              </thead>
              <tbody>
                {benchmarkList.map((item) => (
                  <tr
                    key={item.id}
                    className={`border-t border-white/10 ${selected === item.id ? "bg-ink-200/55" : "hover:bg-ink-300/50"}`}
                  >
                    <td className="px-4 py-3">
                      <button
                        type="button"
                        onClick={() => setSelected(item.id)}
                        className="font-mono text-[12px] text-bone-50 hover:text-lime-200"
                      >
                        {item.id}
                      </button>
                    </td>
                    <td className="max-w-[12rem] truncate px-4 py-3 text-bone-300" title={String(item.resolved_model ?? item.config.model_ref ?? "—")}>
                      {item.name || String(item.config.model_ref ?? "—")}
                    </td>
                    <td className="px-4 py-3 text-bone-300">
                      {String(item.config.gpu_devices || "auto")}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={item.status} />
                    </td>
                    <td className="px-4 py-3 text-bone-300">
                      {formatTs(item.summary?.best_tg?.avg_ts)}
                    </td>
                    <td className="px-4 py-3 text-bone-300">
                      {formatRelativeTime(item.started_at)}
                    </td>
                  </tr>
                ))}
                {benchmarkList.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-10 text-center text-bone-500">
                      No benchmarks yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel
          title={selectedBenchmark ? `Benchmark ${selectedBenchmark.id}` : "Benchmark details"}
          actions={
            selectedBenchmark ? (
              <>
                <StatusBadge status={selectedBenchmark.status} />
                {isActive(selectedBenchmark.status) && (
                  <button
                    className="brand-btn-warning px-3 py-1.5 text-xs"
                    disabled={selectedBenchmark.status === "cancelling" || stop.isPending}
                    onClick={() => stop.mutate(selectedBenchmark.id)}
                  >
                    {selectedBenchmark.status === "cancelling" ? "Cancelling…" : "Stop benchmark"}
                  </button>
                )}
              </>
            ) : undefined
          }
        >
          {selectedBenchmark ? (
            <div className="space-y-4">
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <DetailBox label="Started" value={formatTimestamp(selectedBenchmark.started_at)} />
                <DetailBox label="Finished" value={formatTimestamp(selectedBenchmark.finished_at)} />
                <DetailBox label="Duration" value={formatDuration(selectedBenchmark)} />
                <DetailBox label="Exit code" value={selectedBenchmark.exit_code == null ? "—" : String(selectedBenchmark.exit_code)} />
                <DetailBox label="PID" value={selectedBenchmark.pid == null ? "—" : String(selectedBenchmark.pid)} />
                <DetailBox label="GPU set" value={String(selectedBenchmark.config.gpu_devices || "auto")} />
                <DetailBox label="Resolved model" value={selectedBenchmark.resolved_model ?? "—"} mono />
                <DetailBox label="Best tg" value={formatTs(selectedBenchmark.summary?.best_tg?.avg_ts)} />
              </div>

              {selectedBenchmark.parse_error && (
                <div className="border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
                  Result parse note: {selectedBenchmark.parse_error}
                </div>
              )}

              <div className="border border-white/10 bg-ink-400/72 p-4">
                <div className="brand-label">Executed command</div>
                <div className="mt-2 font-mono text-[12px] text-bone-100 break-all">
                  {selectedBenchmark.cmdline?.join(" ") || "—"}
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <MetricMini label="best overall" value={formatTs(selectedBenchmark.summary?.best_overall?.avg_ts)} meta={selectedBenchmark.summary?.best_overall?.test ?? "—"} />
                <MetricMini label="best pp" value={formatTs(selectedBenchmark.summary?.best_pp?.avg_ts)} meta={selectedBenchmark.summary?.best_pp?.test ?? "—"} />
                <MetricMini label="best tg" value={formatTs(selectedBenchmark.summary?.best_tg?.avg_ts)} meta={selectedBenchmark.summary?.best_tg?.test ?? "—"} />
                <MetricMini label="best pg" value={formatTs(selectedBenchmark.summary?.best_pg?.avg_ts)} meta={selectedBenchmark.summary?.best_pg?.test ?? "—"} />
              </div>

              <div className="brand-table-wrap">
                <table className="min-w-[760px] w-full text-sm">
                  <thead className="bg-ink-400/78">
                    <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                      <th className="px-4 py-3 font-semibold">Test</th>
                      <th className="px-4 py-3 font-semibold">t/s</th>
                      <th className="px-4 py-3 font-semibold">Stddev</th>
                      <th className="px-4 py-3 font-semibold">Threads</th>
                      <th className="px-4 py-3 font-semibold">ngl</th>
                      <th className="px-4 py-3 font-semibold">Batch</th>
                      <th className="px-4 py-3 font-semibold">Backend</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(selectedBenchmark.result_rows ?? []).map((row, idx) => (
                      <BenchmarkRowView key={`${row.test}-${idx}`} row={row} />
                    ))}
                    {(selectedBenchmark.result_rows ?? []).length === 0 && (
                      <tr>
                        <td colSpan={7} className="px-4 py-10 text-center text-bone-500">
                          No parsed benchmark rows yet.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="border border-white/10 bg-ink-400/72 px-3 py-2 text-[11px] uppercase tracking-wider text-bone-400">
                {isActive(selectedBenchmark.status)
                  ? `Log stream · ${socket.status}`
                  : "Viewing durable log history"}
              </div>
              <LogPane lines={lines} height="52vh" />
            </div>
          ) : (
            <div className="text-sm text-bone-500">
              Select a benchmark from history to inspect parsed throughput rows and logs.
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="flex flex-col gap-2 text-sm">
      <span className="brand-label">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="brand-input"
        placeholder={placeholder}
      />
    </label>
  );
}

function ToggleField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-3 border border-white/10 bg-ink-300/70 px-3 py-3 text-sm text-bone-100">
      <span>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 rounded-none accent-lime-300"
      />
    </label>
  );
}

function DetailBox({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="border border-white/10 bg-ink-300/70 px-4 py-3">
      <div className="brand-label">{label}</div>
      <div className={`mt-2 text-sm text-bone-100 ${mono ? "font-mono break-all" : ""}`}>
        {value}
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  meta,
  tone = "default",
}: {
  label: string;
  value: string;
  meta: string[];
  tone?: "default" | "success" | "warning";
}) {
  const toneClass =
    tone === "success"
      ? "text-lime-200"
      : tone === "warning"
        ? "text-amber-200"
        : "text-bone-50";
  return (
    <div className="brand-stat">
      <div className="brand-label">{label}</div>
      <div className={`mt-4 text-3xl font-semibold tracking-tight ${toneClass}`}>{value}</div>
      <div className="mt-4 space-y-1 text-sm text-bone-400">
        {meta.map((line) => (
          <div key={line}>{line}</div>
        ))}
      </div>
    </div>
  );
}

function MetricMini({
  label,
  value,
  meta,
}: {
  label: string;
  value: string;
  meta: string;
}) {
  return (
    <div className="border border-white/10 bg-ink-300/70 px-4 py-3">
      <div className="brand-label">{label}</div>
      <div className="mt-2 text-lg font-semibold text-bone-50">{value}</div>
      <div className="mt-1 text-xs text-bone-400">{meta}</div>
    </div>
  );
}

function BenchmarkRowView({ row }: { row: BenchmarkRow }) {
  return (
    <tr className="border-t border-white/10">
      <td className="px-4 py-3 font-mono text-[12px] text-bone-100">{row.test}</td>
      <td className="px-4 py-3 text-bone-300">{formatTs(row.avg_ts)}</td>
      <td className="px-4 py-3 text-bone-300">
        {row.stddev_ts == null ? "—" : row.stddev_ts.toFixed(2)}
      </td>
      <td className="px-4 py-3 text-bone-300">{row.threads ?? "—"}</td>
      <td className="px-4 py-3 text-bone-300">{row.n_gpu_layers ?? "—"}</td>
      <td className="px-4 py-3 text-bone-300">
        {row.batch_size ?? "—"}
        {row.ubatch_size ? ` / ${row.ubatch_size}` : ""}
      </td>
      <td className="px-4 py-3 text-bone-300">{row.backend ?? "—"}</td>
    </tr>
  );
}
