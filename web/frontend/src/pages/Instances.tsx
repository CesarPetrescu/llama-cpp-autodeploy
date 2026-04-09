import { FormEvent, ReactNode, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { GpuInfo, Instance, InstanceConfig, api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { ModelSelect } from "@/components/ModelSelect";
import { PageHeader } from "@/components/PageHeader";

type GpuStrategy = "balanced" | "vram" | "priority" | "auto" | "single" | "cpu" | "custom";
type AutoSplitPolicy = "vram" | "free" | "even";

const DEFAULT_CONFIG: InstanceConfig = {
  mode: "llm",
  model_ref: "",
  host: "127.0.0.1",
  port: 45540,
  n_gpu_layers: 999,
  gpu_strategy: "balanced",
  gpu_devices: "all",
  auto_split_policy: "vram",
  tensor_split: "",
  split_mode: "default",
  ctx_size: 4096,
  n_cpu_moe: null,
  cpu_moe: false,
  mmproj: "",
  jinja: false,
  reasoning_format: "",
  no_context_shift: false,
  extra_flags: "",
};

const GPU_STRATEGY_OPTIONS: Array<{ value: GpuStrategy; label: string }> = [
  { value: "balanced", label: "Even split" },
  { value: "vram", label: "% by total VRAM" },
  { value: "priority", label: "Bias first GPU" },
  { value: "auto", label: "Auto at launch" },
  { value: "custom", label: "Custom %" },
  { value: "single", label: "Single GPU" },
  { value: "cpu", label: "CPU only" },
];

const AUTO_SPLIT_POLICY_OPTIONS: Array<{ value: AutoSplitPolicy; label: string }> = [
  { value: "vram", label: "By total VRAM" },
  { value: "free", label: "By free VRAM" },
  { value: "even", label: "Even split" },
];

const SPLIT_MODE_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "layer", label: "Layer split" },
  { value: "row", label: "Row split" },
  { value: "none", label: "No split" },
];

function formatUptime(seconds: number | null): string {
  if (!seconds || seconds < 0) return "—";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
}

function parseGpuSelection(value: string | null | undefined, gpus: GpuInfo[]): number[] {
  const raw = (value || "").trim().toLowerCase();
  if (!raw || raw === "all" || raw === "auto" || raw === "default") {
    return gpus.map((gpu) => gpu.index);
  }
  if (raw === "cpu" || raw === "none" || raw === "off") {
    return [];
  }
  const tokens = raw.split(/[,\s]+/).map((token) => token.trim()).filter(Boolean);
  const selected: number[] = [];
  for (const token of tokens) {
    let cleaned = token;
    if (cleaned.startsWith("cuda:")) cleaned = cleaned.slice(5);
    if (cleaned.startsWith("gpu:")) cleaned = cleaned.slice(4);
    const idx = Number.parseInt(cleaned, 10);
    if (Number.isNaN(idx)) continue;
    if (!selected.includes(idx) && gpus.some((gpu) => gpu.index === idx)) {
      selected.push(idx);
    }
  }
  return selected;
}

function encodeGpuSelection(indices: number[], gpus: GpuInfo[]): string {
  if (indices.length === 0) return "cpu";
  const sorted = [...indices].sort((a, b) => a - b);
  const all = gpus.map((gpu) => gpu.index).sort((a, b) => a - b);
  if (sorted.length === all.length && sorted.every((value, idx) => value === all[idx])) {
    return "all";
  }
  return sorted.join(",");
}

function distributePercentages(weights: number[]): number[] {
  if (weights.length === 0) return [];
  const safe = weights.map((weight) => (weight > 0 ? weight : 1));
  const total = safe.reduce((sum, weight) => sum + weight, 0);
  if (total <= 0) {
    const even = Math.floor(100 / safe.length);
    const parts = safe.map(() => even);
    for (let i = 0; i < 100 - even * safe.length; i += 1) {
      parts[i % parts.length] += 1;
    }
    return parts;
  }
  const raw = safe.map((weight) => (weight / total) * 100);
  const parts = raw.map((value) => Math.floor(value));
  let remainder = 100 - parts.reduce((sum, value) => sum + value, 0);
  const order = raw
    .map((value, idx) => ({ idx, frac: value - parts[idx] }))
    .sort((a, b) => b.frac - a.frac);
  while (remainder > 0 && order.length > 0) {
    parts[order[(100 - remainder) % order.length].idx] += 1;
    remainder -= 1;
  }
  return parts;
}

function buildBalancedSplit(count: number): number[] {
  return distributePercentages(Array.from({ length: count }, () => 1));
}

function buildPrioritySplit(count: number): number[] {
  if (count <= 0) return [];
  if (count === 1) return [100];
  const first = 60;
  const remaining = 100 - first;
  const parts = [first, ...buildBalancedSplit(count - 1).map((value) => Math.round((value / 100) * remaining))];
  const total = parts.reduce((sum, value) => sum + value, 0);
  parts[parts.length - 1] += 100 - total;
  return parts;
}

function buildPolicySplit(gpus: GpuInfo[], policy: AutoSplitPolicy): number[] {
  if (gpus.length <= 1) return gpus.length === 1 ? [100] : [];
  if (policy === "even") {
    return buildBalancedSplit(gpus.length);
  }
  const weights = gpus.map((gpu) => {
    if (policy === "free") {
      return gpu.free ?? gpu.total ?? 1;
    }
    return gpu.total ?? gpu.free ?? 1;
  });
  return distributePercentages(weights);
}

function parseTensorSplit(value: string | null | undefined): number[] | null {
  const raw = (value || "").trim();
  if (!raw || raw.toLowerCase() === "auto") return null;
  const tokens = raw.replace(/;/g, ",").split(/[,:\s]+/).map((token) => token.trim()).filter(Boolean);
  if (tokens.length === 0) return null;
  const parts: number[] = [];
  for (const token of tokens) {
    const cleaned = token.endsWith("%") ? token.slice(0, -1).trim() : token;
    const parsed = Number.parseFloat(cleaned);
    if (!Number.isFinite(parsed) || parsed < 0) return null;
    parts.push(parsed);
  }
  return parts;
}

function normalizeSplitValues(parts: number[] | null, count: number): number[] {
  if (count <= 0) return [];
  if (!parts || parts.length === 0) return buildBalancedSplit(count);
  const next = parts.slice(0, count);
  while (next.length < count) next.push(0);
  return next;
}

function formatGpuVisibility(config: InstanceConfig, gpus: GpuInfo[]): string {
  const selected = parseGpuSelection(config.gpu_devices, gpus);
  if (selected.length === 0) return "CPU only";
  if (selected.length === gpus.length) return gpus.length > 0 ? `All detected GPUs (${selected.join(",")})` : "All detected GPUs";
  return `CUDA_VISIBLE_DEVICES=${selected.join(",")}`;
}

function applyGpuStrategy(
  current: InstanceConfig,
  strategy: GpuStrategy,
  gpus: GpuInfo[],
): InstanceConfig {
  let next: InstanceConfig = { ...current, gpu_strategy: strategy };
  let selected = gpus.filter((gpu) => parseGpuSelection(next.gpu_devices, gpus).includes(gpu.index));

  if (strategy === "cpu" || selected.length === 0) {
    return {
      ...next,
      gpu_strategy: "cpu",
      gpu_devices: "cpu",
      n_gpu_layers: 0,
      tensor_split: "",
    };
  }

  if (strategy === "single" && selected.length > 1) {
    next = { ...next, gpu_devices: String(selected[0].index) };
    selected = [selected[0]];
  }

  if (selected.length === 1) {
    return {
      ...next,
      gpu_strategy: "single",
      n_gpu_layers: next.n_gpu_layers && next.n_gpu_layers > 0 ? next.n_gpu_layers : 999,
      tensor_split: "",
    };
  }

  if (strategy === "auto") {
    return {
      ...next,
      n_gpu_layers: next.n_gpu_layers && next.n_gpu_layers > 0 ? next.n_gpu_layers : 999,
      tensor_split: "auto",
    };
  }

  if (strategy === "balanced") {
    return {
      ...next,
      n_gpu_layers: next.n_gpu_layers && next.n_gpu_layers > 0 ? next.n_gpu_layers : 999,
      tensor_split: buildBalancedSplit(selected.length).join(","),
    };
  }

  if (strategy === "vram") {
    return {
      ...next,
      n_gpu_layers: next.n_gpu_layers && next.n_gpu_layers > 0 ? next.n_gpu_layers : 999,
      tensor_split: buildPolicySplit(selected, "vram").join(","),
    };
  }

  if (strategy === "priority") {
    return {
      ...next,
      n_gpu_layers: next.n_gpu_layers && next.n_gpu_layers > 0 ? next.n_gpu_layers : 999,
      tensor_split: buildPrioritySplit(selected.length).join(","),
    };
  }

  return {
    ...next,
    gpu_strategy: "custom",
    n_gpu_layers: next.n_gpu_layers && next.n_gpu_layers > 0 ? next.n_gpu_layers : 999,
    tensor_split: normalizeSplitValues(parseTensorSplit(next.tensor_split), selected.length).join(","),
  };
}

export default function Instances() {
  const qc = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [config, setConfig] = useState<InstanceConfig>({ ...DEFAULT_CONFIG });
  const [autoStart, setAutoStart] = useState(true);
  const [formError, setFormError] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["instances"],
    queryFn: api.listInstances,
  });

  const gpuQuery = useQuery({
    queryKey: ["gpus"],
    queryFn: api.listGpus,
    refetchInterval: 4000,
  });

  const gpus = gpuQuery.data?.gpus ?? [];
  const selectedGpuIndices = parseGpuSelection(config.gpu_devices, gpus);
  const selectedGpus = gpus.filter((gpu) => selectedGpuIndices.includes(gpu.index));
  const tensorSplitAuto = (config.tensor_split ?? "").trim().toLowerCase() === "auto";
  const splitEditorValues = normalizeSplitValues(parseTensorSplit(config.tensor_split), selectedGpus.length);
  const autoPreview = buildPolicySplit(selectedGpus, (config.auto_split_policy as AutoSplitPolicy | null) ?? "vram");
  const splitTotal = splitEditorValues.reduce((sum, value) => sum + value, 0);

  const createMut = useMutation({
    mutationFn: async () => api.createInstance(name, config, autoStart),
    onSuccess: () => {
      setShowForm(false);
      setName("");
      setConfig({ ...DEFAULT_CONFIG });
      setFormError(null);
      qc.invalidateQueries({ queryKey: ["instances"] });
    },
    onError: (err: Error) => setFormError(err.message),
  });

  const action = (fn: (id: string) => Promise<unknown>) => async (id: string) => {
    try {
      await fn(id);
      qc.invalidateQueries({ queryKey: ["instances"] });
    } catch (err) {
      alert((err as Error).message);
    }
  };

  const onStart = action(api.startInstance);
  const onStop = action(api.stopInstance);
  const onRestart = action(api.restartInstance);
  const onDelete = action(async (id) => {
    if (!confirm("Delete this instance?")) return;
    return api.deleteInstance(id);
  });

  function submit(e: FormEvent) {
    e.preventDefault();
    setFormError(null);
    createMut.mutate();
  }

  function upd<K extends keyof InstanceConfig>(k: K, v: InstanceConfig[K]) {
    setConfig((current) => ({ ...current, [k]: v }));
  }

  function setGpuSelection(indices: number[]) {
    setConfig((current) => {
      const next = { ...current, gpu_devices: encodeGpuSelection(indices, gpus) };
      const strategy = (current.gpu_strategy as GpuStrategy | null) ?? "balanced";
      const fallback = indices.length <= 1 ? "single" : strategy === "cpu" ? "balanced" : strategy;
      return applyGpuStrategy(next, fallback, gpus);
    });
  }

  function toggleGpu(index: number) {
    if (selectedGpuIndices.includes(index)) {
      setGpuSelection(selectedGpuIndices.filter((value) => value !== index));
      return;
    }
    setGpuSelection([...selectedGpuIndices, index]);
  }

  function setStrategy(strategy: GpuStrategy) {
    setConfig((current) => applyGpuStrategy(current, strategy, gpus));
  }

  function setSplitValue(idx: number, raw: string) {
    const parsed = Number.parseFloat(raw);
    setConfig((current) => {
      const next = normalizeSplitValues(parseTensorSplit(current.tensor_split), selectedGpus.length);
      next[idx] = Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
      return {
        ...current,
        gpu_strategy: "custom",
        n_gpu_layers: selectedGpus.length > 0 ? (current.n_gpu_layers && current.n_gpu_layers > 0 ? current.n_gpu_layers : 999) : 0,
        tensor_split: next.join(","),
      };
    });
  }

  function updateTensorSplit(raw: string) {
    setConfig((current) => {
      const cleaned = raw.trim();
      const nextStrategy: GpuStrategy =
        cleaned.toLowerCase() === "auto"
          ? "auto"
          : selectedGpus.length <= 1
            ? "single"
            : "custom";
      return {
        ...current,
        gpu_strategy: nextStrategy,
        n_gpu_layers: selectedGpus.length > 0 ? (current.n_gpu_layers && current.n_gpu_layers > 0 ? current.n_gpu_layers : 999) : 0,
        tensor_split: raw,
      };
    });
  }

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Workloads"
        title="Instances"
        description="Launch, stop, and monitor llama-server processes."
        actions={
          <button
            className="brand-btn-primary"
            onClick={() => setShowForm((s) => !s)}
          >
            {showForm ? "Cancel" : "+ New instance"}
          </button>
        }
      />

      {showForm && (
        <Panel title="New instance">
          <form onSubmit={submit} className="grid gap-3 md:grid-cols-2">
            <Field label="Name">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="input"
                placeholder="qwen-30b"
              />
            </Field>
            <Field label="Mode">
              <select
                value={config.mode}
                onChange={(e) => upd("mode", e.target.value as "llm" | "embed")}
                className="input"
              >
                <option value="llm">llm</option>
                <option value="embed">embed</option>
              </select>
            </Field>
            <div className="md:col-span-2 flex flex-col gap-1 text-sm">
              <span className="brand-label">Model (pick from library or paste HF ref)</span>
              <ModelSelect
                value={config.model_ref}
                onChange={(value) => upd("model_ref", value)}
              />
            </div>
            <Field label="HF token (optional)">
              <input
                type="password"
                value={config.hf_token ?? ""}
                onChange={(e) => upd("hf_token", e.target.value)}
                className="input"
              />
            </Field>
            <Field label="Host">
              <input value={config.host} onChange={(e) => upd("host", e.target.value)} className="input" />
            </Field>
            <Field label="Port">
              <input
                type="number"
                value={config.port}
                onChange={(e) => upd("port", Number.parseInt(e.target.value, 10) || 0)}
                className="input"
              />
            </Field>
            <Field label="--n-gpu-layers">
              <input
                type="number"
                value={config.n_gpu_layers ?? 0}
                onChange={(e) => upd("n_gpu_layers", Number.parseInt(e.target.value, 10))}
                className="input"
              />
            </Field>
            <Field label="--split-mode">
              <select
                value={config.split_mode ?? "default"}
                onChange={(e) => upd("split_mode", e.target.value)}
                className="input"
              >
                {SPLIT_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </Field>

            <div className="md:col-span-2 brand-surface-muted p-4">
              <div className="flex flex-col gap-3">
                <div>
                  <div className="font-display text-sm font-semibold text-bone-50">GPU placement</div>
                  <div className="text-[11px] uppercase tracking-wider text-bone-500">
                    Visible GPUs become <code className="text-lime-300">CUDA_VISIBLE_DEVICES</code> for this instance.
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${selectedGpuIndices.length === gpus.length && gpus.length > 0 ? "bg-lime-300 text-ink-900 shadow-glow-lime" : "border border-white/10 bg-white/5 text-bone-200 hover:border-lime-300/40"}`}
                    onClick={() => setGpuSelection(gpus.map((gpu) => gpu.index))}
                    disabled={gpus.length === 0}
                  >
                    All GPUs
                  </button>
                  <button
                    type="button"
                    className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${selectedGpuIndices.length === 0 ? "bg-amber-400 text-ink-900" : "border border-white/10 bg-white/5 text-bone-200 hover:border-amber-300/40"}`}
                    onClick={() => setStrategy("cpu")}
                  >
                    CPU only
                  </button>
                </div>
                <div className="grid gap-2 md:grid-cols-3">
                  {gpus.map((gpu) => {
                    const active = selectedGpuIndices.includes(gpu.index);
                    return (
                      <button
                        type="button"
                        key={gpu.index}
                        onClick={() => toggleGpu(gpu.index)}
                        className={`rounded-xl border p-3 text-left transition ${active ? "border-lime-300/60 bg-lime-300/10 shadow-[0_0_0_1px_rgba(213,255,64,0.2)]" : "border-white/5 bg-ink-300/60 hover:border-white/20"}`}
                      >
                        <div className="font-medium text-bone-50">#{gpu.index} {gpu.name}</div>
                        <div className="mt-1 text-[11px] uppercase tracking-wider text-bone-500">
                          {gpu.free_h} free · {gpu.total_h} total
                        </div>
                      </button>
                    );
                  })}
                  {gpus.length === 0 && (
                    <div className="rounded-xl border border-white/5 bg-ink-300/60 p-3 text-sm text-bone-400">
                      No CUDA devices detected. The instance can still run CPU-only.
                    </div>
                  )}
                </div>
                <div className="text-[11px] uppercase tracking-wider text-bone-500">
                  Visibility · <span className="text-lime-300">{formatGpuVisibility(config, gpus)}</span>
                </div>
              </div>
            </div>

            <Field label="GPU memory strategy">
              <select
                value={(config.gpu_strategy as string) || "balanced"}
                onChange={(e) => setStrategy(e.target.value as GpuStrategy)}
                className="input"
              >
                {GPU_STRATEGY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </Field>
            <Field label="Auto split policy">
              <select
                value={(config.auto_split_policy as string) || "vram"}
                onChange={(e) => upd("auto_split_policy", e.target.value)}
                className="input"
              >
                {AUTO_SPLIT_POLICY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </Field>
            <Field label="--tensor-split (blank, auto, or 60,40)">
              <input
                value={config.tensor_split ?? ""}
                onChange={(e) => updateTensorSplit(e.target.value)}
                className="input"
                placeholder={selectedGpus.length > 1 ? "60,40" : ""}
              />
            </Field>
            <Field label="--ctx-size">
              <input
                type="number"
                value={config.ctx_size ?? 0}
                onChange={(e) => upd("ctx_size", Number.parseInt(e.target.value, 10))}
                className="input"
              />
            </Field>
            <Field label="--n-cpu-moe">
              <input
                type="number"
                value={config.n_cpu_moe ?? 0}
                onChange={(e) => upd("n_cpu_moe", Number.parseInt(e.target.value, 10))}
                className="input"
              />
            </Field>
            <Field label="Extra flags">
              <input
                value={config.extra_flags}
                onChange={(e) => upd("extra_flags", e.target.value)}
                className="input"
                placeholder="--parallel 4"
              />
            </Field>

            {selectedGpus.length > 1 && (
              <div className="md:col-span-2 brand-surface-muted p-4">
                <div className="font-display text-sm font-semibold text-bone-50">Per-GPU split</div>
                <div className="mt-1 text-[11px] uppercase tracking-wider text-bone-500">
                  {tensorSplitAuto
                    ? "Preview of the auto policy that will be resolved on launch."
                    : "Edit percentages directly for each selected GPU."}
                </div>
                <div className="mt-3 grid gap-3 md:grid-cols-3">
                  {(tensorSplitAuto ? autoPreview : splitEditorValues).map((value, idx) => (
                    <label key={selectedGpus[idx].index} className="flex flex-col gap-1 text-sm">
                      <span className="brand-label">
                        GPU #{selectedGpus[idx].index} {selectedGpus[idx].name}
                      </span>
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          min={0}
                          step="0.1"
                          value={value}
                          disabled={tensorSplitAuto}
                          onChange={(e) => setSplitValue(idx, e.target.value)}
                          className="input"
                        />
                        <span className="text-xs text-bone-400">%</span>
                      </div>
                    </label>
                  ))}
                </div>
                {!tensorSplitAuto && (
                  <div className={`mt-3 text-[11px] uppercase tracking-wider ${Math.abs(splitTotal - 100) < 0.001 ? "text-lime-300" : "text-amber-300"}`}>
                    Current total · {splitTotal}%
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.cpu_moe}
                  onChange={(e) => upd("cpu_moe", e.target.checked)}
                />
                --cpu-moe
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.jinja}
                  onChange={(e) => upd("jinja", e.target.checked)}
                />
                --jinja
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={autoStart}
                  onChange={(e) => setAutoStart(e.target.checked)}
                />
                start now
              </label>
            </div>
            <div className="col-span-full flex items-center gap-3">
              <button
                type="submit"
                disabled={createMut.isPending}
                className="brand-btn-primary"
              >
                {createMut.isPending ? "Creating…" : "Create instance"}
              </button>
              {formError && <span className="text-sm text-rose-300">{formError}</span>}
            </div>
          </form>
        </Panel>
      )}

      <Panel title="Managed instances">
        <div className="overflow-hidden rounded-xl border border-white/5">
          <table className="w-full text-sm">
            <thead className="bg-white/[0.03]">
              <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                <th className="px-4 py-2.5 font-semibold">Name</th>
                <th className="px-4 py-2.5 font-semibold">Status</th>
                <th className="px-4 py-2.5 font-semibold">Model</th>
                <th className="px-4 py-2.5 font-semibold">Endpoint</th>
                <th className="px-4 py-2.5 font-semibold">Uptime</th>
                <th className="px-4 py-2.5 text-right font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {query.data?.instances.map((inst: Instance) => (
                <tr
                  key={inst.id}
                  className="border-t border-white/5 hover:bg-white/[0.02]"
                >
                  <td className="px-4 py-3 font-medium text-bone-100">
                    {inst.name}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={inst.status} />
                  </td>
                  <td
                    className="max-w-xs truncate px-4 py-3 text-bone-300"
                    title={inst.config.model_ref}
                  >
                    {inst.config.model_ref}
                  </td>
                  <td className="px-4 py-3 font-mono text-[12px] text-bone-300">
                    {inst.host}:{inst.port}
                  </td>
                  <td className="px-4 py-3 text-bone-300">
                    {formatUptime(inst.uptime_s)}
                  </td>
                  <td className="space-x-2 px-4 py-3 text-right">
                    <Link
                      to={`/instances/${inst.id}/logs`}
                      className="brand-btn-ghost px-3 py-1.5 text-xs"
                    >
                      Logs
                    </Link>
                    {inst.status === "running" ? (
                      <button
                        className="brand-btn-warning px-3 py-1.5 text-xs"
                        onClick={() => onStop(inst.id)}
                      >
                        Stop
                      </button>
                    ) : inst.status === "stopping" ? (
                      <button
                        className="brand-btn-warning px-3 py-1.5 text-xs"
                        disabled
                      >
                        Stopping…
                      </button>
                    ) : (
                      <button
                        className="brand-btn-primary px-3 py-1.5 text-xs"
                        onClick={() => onStart(inst.id)}
                      >
                        Start
                      </button>
                    )}
                    <button
                      className="brand-btn-ghost px-3 py-1.5 text-xs"
                      disabled={inst.status === "stopping"}
                      onClick={() => onRestart(inst.id)}
                    >
                      Restart
                    </button>
                    <button
                      className="brand-btn-danger px-3 py-1.5 text-xs"
                      disabled={inst.status === "stopping"}
                      onClick={() => onDelete(inst.id)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {query.data?.instances.length === 0 && (
                <tr>
                  <td
                    colSpan={6}
                    className="px-4 py-10 text-center text-bone-500"
                  >
                    No instances yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      <style>{`
        .input {
          width: 100%;
          border: 1px solid rgba(255, 255, 255, 0.10);
          background: rgba(13, 14, 9, 0.6);
          color: rgb(244 245 242);
          padding: 0.5rem 0.75rem;
          border-radius: 0.5rem;
          font-size: 0.875rem;
          font-family: Poppins, system-ui, sans-serif;
          transition: border-color 120ms, box-shadow 120ms;
        }
        .input::placeholder { color: rgb(131 135 123); }
        .input:focus {
          outline: none;
          border-color: rgba(213, 255, 64, 0.55);
          box-shadow: 0 0 0 3px rgba(213, 255, 64, 0.18);
        }
      `}</style>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="brand-label">{label}</span>
      {children}
    </label>
  );
}
