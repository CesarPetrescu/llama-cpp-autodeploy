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
  spec_type: "none",
  spec_draft_model: "",
  spec_draft_n_max: null,
  spec_draft_n_min: null,
  spec_draft_p_split: null,
  spec_draft_p_min: null,
  spec_draft_backend_sampling: "default",
  spec_draft_hf: "",
  spec_draft_ngl: "",
  spec_draft_device: "",
  spec_draft_type_k: "",
  spec_draft_type_v: "",
  spec_draft_override_tensor: "",
  spec_draft_cpu_moe: false,
  spec_draft_n_cpu_moe: null,
  spec_draft_threads: null,
  spec_draft_threads_batch: null,
  spec_draft_cpu_mask: "",
  spec_draft_cpu_range: "",
  spec_draft_cpu_strict: null,
  spec_draft_prio: null,
  spec_draft_poll: null,
  spec_draft_cpu_mask_batch: "",
  spec_draft_cpu_strict_batch: null,
  spec_draft_prio_batch: null,
  spec_draft_poll_batch: null,
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

const SPEC_DRAFT_CACHE_TYPES = ["", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"];

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

function formatGpuIndexLabel(gpu: GpuInfo): string {
  if (gpu.system_index != null && gpu.system_index !== gpu.index) {
    return `CUDA #${gpu.index} · nvidia-smi #${gpu.system_index}`;
  }
  return `CUDA #${gpu.index}`;
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
  const [instanceMessage, setInstanceMessage] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["instances"],
    queryFn: api.listInstances,
  });

  const gpuQuery = useQuery({
    queryKey: ["gpus"],
    queryFn: api.listGpus,
    refetchInterval: 4000,
  });

  const binaryCaps = useQuery({
    queryKey: ["binary-caps"],
    queryFn: api.binaryCaps,
  });

  const gpus = gpuQuery.data?.gpus ?? [];
  const mtpEnabled = config.spec_type === "draft-mtp";
  const mtpSupported = binaryCaps.data?.has_draft_mtp ?? false;
  const selectedGpuIndices = parseGpuSelection(config.gpu_devices, gpus);
  const selectedGpus = gpus.filter((gpu) => selectedGpuIndices.includes(gpu.index));
  const tensorSplitAuto = (config.tensor_split ?? "").trim().toLowerCase() === "auto";
  const splitEditorValues = normalizeSplitValues(parseTensorSplit(config.tensor_split), selectedGpus.length);
  const autoPreview = buildPolicySplit(selectedGpus, (config.auto_split_policy as AutoSplitPolicy | null) ?? "vram");
  const splitTotal = splitEditorValues.reduce((sum, value) => sum + value, 0);
  const instances = query.data?.instances ?? [];
  const runningCount = instances.filter((inst) => inst.status === "running").length;
  const stoppedCount = instances.filter((inst) => inst.status === "stopped").length;
  const crashedCount = instances.filter((inst) => inst.status === "crashed").length;

  const createMut = useMutation({
    mutationFn: async () => api.createInstance(name, config, autoStart),
    onSuccess: () => {
      setShowForm(false);
      setName("");
      setConfig({ ...DEFAULT_CONFIG });
      setFormError(null);
      setInstanceMessage(autoStart ? "Instance created and launch requested." : "Instance saved to the control plane.");
      qc.invalidateQueries({ queryKey: ["instances"] });
    },
    onError: (err: Error) => setFormError(err.message),
  });

  const recoverMut = useMutation({
    mutationFn: () => api.recoverInstances(),
    onSuccess: (data) => {
      setInstanceMessage(
        data.recovered.length > 0
          ? `Recovered ${data.recovered.length} orphaned llama-server instance${data.recovered.length === 1 ? "" : "s"}.`
          : "No orphaned llama-server processes were found.",
      );
      qc.invalidateQueries({ queryKey: ["instances"] });
    },
    onError: (err: Error) => setInstanceMessage(err.message),
  });

  const action = (fn: (id: string) => Promise<unknown>) => async (id: string) => {
    try {
      await fn(id);
      setInstanceMessage(null);
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

  function updTextKey(k: keyof InstanceConfig, value: string) {
    setConfig((current) => ({ ...current, [k]: value }));
  }

  function updOptionalInt<K extends keyof InstanceConfig>(k: K, raw: string) {
    const parsed = raw.trim() === "" ? null : Number.parseInt(raw, 10);
    upd(k, (Number.isFinite(parsed) ? parsed : null) as InstanceConfig[K]);
  }

  function updOptionalFloat<K extends keyof InstanceConfig>(k: K, raw: string) {
    const parsed = raw.trim() === "" ? null : Number.parseFloat(raw);
    upd(k, (Number.isFinite(parsed) ? parsed : null) as InstanceConfig[K]);
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
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow="Workloads"
        title="Instances"
        description="Launch, recover, and monitor llama-server processes from one place. The page now exposes backend recovery directly so orphaned servers do not disappear after a crash."
        actions={
          <>
            <button
              type="button"
              className="brand-btn-ghost"
              disabled={recoverMut.isPending}
              onClick={() => recoverMut.mutate()}
            >
              {recoverMut.isPending ? "Scanning…" : "Recover running servers"}
            </button>
            <button
              type="button"
              className="brand-btn-primary"
              onClick={() => setShowForm((s) => !s)}
            >
              {showForm ? "Close editor" : "New instance"}
            </button>
          </>
        }
      />

      {instanceMessage && (
        <div className="rounded-none border border-lime-300/25 bg-lime-300/10 px-4 py-3 text-sm text-lime-100">
          {instanceMessage}
        </div>
      )}

      <div className="grid gap-4 grid-cols-2 xl:grid-cols-4">
        <SummaryCard label="Managed" value={`${instances.length}`} hint="Persisted control-plane records" />
        <SummaryCard
          label="Running"
          value={`${runningCount}`}
          hint="Currently answering requests"
          tone="success"
        />
        <SummaryCard label="Stopped" value={`${stoppedCount}`} hint="Ready to restart" />
        <SummaryCard
          label="Crashed"
          value={`${crashedCount}`}
          hint="Needs inspection or relaunch"
          tone={crashedCount > 0 ? "danger" : "default"}
        />
      </div>

      {showForm && (
        <Panel
          title="Instance editor"
          subtitle="Group the launch settings before you start a new workload. This layout is denser and easier to scan than the previous flat form."
        >
          <form onSubmit={submit} className="space-y-4">
            <div className="grid gap-4 xl:grid-cols-[1.08fr_0.92fr]">
              <div className="space-y-4">
                <div className="brand-surface-muted p-4">
                  <div className="brand-label">Identity</div>
                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <Field label="Name">
                      <input
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className="brand-input"
                        placeholder="qwen-30b"
                      />
                    </Field>
                    <Field label="Mode">
                      <select
                        value={config.mode}
                        onChange={(e) => upd("mode", e.target.value as "llm" | "embed")}
                        className="brand-input"
                      >
                        <option value="llm">llm</option>
                        <option value="embed">embed</option>
                      </select>
                    </Field>
                    <div className="md:col-span-2 flex flex-col gap-1 text-sm">
                      <span className="brand-label">Model</span>
                      <ModelSelect
                        value={config.model_ref}
                        onChange={(value) => upd("model_ref", value)}
                      />
                    </div>
                    <div className="md:col-span-2">
                      <Field label="HF token (optional)">
                        <input
                          type="password"
                          value={config.hf_token ?? ""}
                          onChange={(e) => upd("hf_token", e.target.value)}
                          className="brand-input"
                          placeholder="Used only for gated/private model pulls"
                        />
                      </Field>
                    </div>
                  </div>
                </div>

                <div className="brand-surface-muted p-4">
                  <div className="brand-label">Endpoint and runtime</div>
                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <Field label="Host">
                      <input
                        value={config.host}
                        onChange={(e) => upd("host", e.target.value)}
                        className="brand-input"
                      />
                    </Field>
                    <Field label="Port">
                      <input
                        type="number"
                        value={config.port}
                        onChange={(e) => upd("port", Number.parseInt(e.target.value, 10) || 0)}
                        className="brand-input"
                      />
                    </Field>
                    <Field label="--n-gpu-layers">
                      <input
                        type="number"
                        value={config.n_gpu_layers ?? 0}
                        onChange={(e) => upd("n_gpu_layers", Number.parseInt(e.target.value, 10))}
                        className="brand-input"
                      />
                    </Field>
                    <Field label="--ctx-size">
                      <input
                        type="number"
                        value={config.ctx_size ?? 0}
                        onChange={(e) => upd("ctx_size", Number.parseInt(e.target.value, 10))}
                        className="brand-input"
                      />
                    </Field>
                    <Field label="--split-mode">
                      <select
                        value={config.split_mode ?? "default"}
                        onChange={(e) => upd("split_mode", e.target.value)}
                        className="brand-input"
                      >
                        {SPLIT_MODE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <Field label="--n-cpu-moe">
                      <input
                        type="number"
                        value={config.n_cpu_moe ?? 0}
                        onChange={(e) => upd("n_cpu_moe", Number.parseInt(e.target.value, 10))}
                        className="brand-input"
                      />
                    </Field>
                    <div className="md:col-span-2">
                      <Field label="Extra flags">
                        <input
                          value={config.extra_flags}
                          onChange={(e) => upd("extra_flags", e.target.value)}
                          className="brand-input"
                          placeholder="--parallel 4 --flash-attn on"
                        />
                      </Field>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="brand-surface-muted p-4">
                  <div className="brand-label">GPU placement</div>
                  <p className="mt-2 text-sm text-bone-300">
                    Selected devices become <code className="text-lime-300">CUDA_VISIBLE_DEVICES</code> for this instance.
                  </p>

                  <div className="mt-4 grid gap-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <Field label="GPU memory strategy">
                        <select
                          value={(config.gpu_strategy as string) || "balanced"}
                          onChange={(e) => setStrategy(e.target.value as GpuStrategy)}
                          className="brand-input"
                        >
                          {GPU_STRATEGY_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </Field>
                      <Field label="Auto split policy">
                        <select
                          value={(config.auto_split_policy as string) || "vram"}
                          onChange={(e) => upd("auto_split_policy", e.target.value)}
                          className="brand-input"
                        >
                          {AUTO_SPLIT_POLICY_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </Field>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        className={`rounded-none px-3 py-2 text-xs font-semibold ${selectedGpuIndices.length === gpus.length && gpus.length > 0 ? "bg-lime-300 text-ink-900 shadow-glow-lime" : "border border-white/10 bg-ink-300/72 text-bone-200 hover:border-lime-300/40 hover:bg-ink-200/82"}`}
                        onClick={() => setGpuSelection(gpus.map((gpu) => gpu.index))}
                        disabled={gpus.length === 0}
                      >
                        All GPUs
                      </button>
                      <button
                        type="button"
                        className={`rounded-none px-3 py-2 text-xs font-semibold ${selectedGpuIndices.length === 0 ? "bg-amber-400 text-ink-900" : "border border-white/10 bg-ink-300/72 text-bone-200 hover:border-amber-300/40 hover:bg-ink-200/82"}`}
                        onClick={() => setStrategy("cpu")}
                      >
                        CPU only
                      </button>
                    </div>

                    <div className="grid gap-2 md:grid-cols-2">
                      {gpus.map((gpu) => {
                        const active = selectedGpuIndices.includes(gpu.index);
                        return (
                          <button
                            type="button"
                            key={gpu.index}
                            onClick={() => toggleGpu(gpu.index)}
                            className={`rounded-none border p-3 text-left ${active ? "border-lime-300/60 bg-lime-300/10 shadow-[0_0_0_1px_rgba(213,255,64,0.2)]" : "border-white/10 bg-ink-300/60 hover:border-white/20"}`}
                          >
                            <div className="font-medium text-bone-50">
                              {formatGpuIndexLabel(gpu)} {gpu.name}
                            </div>
                            <div className="mt-1 text-[11px] uppercase tracking-wider text-bone-500">
                              {gpu.free_h} free · {gpu.total_h} total
                            </div>
                          </button>
                        );
                      })}
                      {gpus.length === 0 && (
                        <div className="rounded-none border border-white/10 bg-ink-300/60 p-3 text-sm text-bone-400">
                          No CUDA devices detected. CPU-only launch is still available.
                        </div>
                      )}
                    </div>

                    <Field label="--tensor-split (blank, auto, or 60,40)">
                      <input
                        value={config.tensor_split ?? ""}
                        onChange={(e) => updateTensorSplit(e.target.value)}
                        className="brand-input"
                        placeholder={selectedGpus.length > 1 ? "60,40" : "auto"}
                      />
                    </Field>

                    <div className="rounded-none border border-white/10 bg-ink-400/72 px-3 py-3 text-xs text-bone-300">
                      Visibility: <span className="text-lime-300">{formatGpuVisibility(config, gpus)}</span>
                    </div>
                  </div>
                </div>

                {selectedGpus.length > 1 && (
                  <div className="brand-surface-muted p-4">
                    <div className="brand-label">Per-GPU split</div>
                    <p className="mt-2 text-sm text-bone-300">
                      {tensorSplitAuto
                        ? "Preview of the selected auto policy resolved against the current GPU set."
                        : "Edit percentages directly when you want explicit placement."}
                    </p>
                    <div className="mt-4 grid gap-3">
                      {(tensorSplitAuto ? autoPreview : splitEditorValues).map((value, idx) => (
                        <label key={selectedGpus[idx].index} className="flex flex-col gap-1 text-sm">
                          <span className="brand-label">
                            {formatGpuIndexLabel(selectedGpus[idx])} {selectedGpus[idx].name}
                          </span>
                          <div className="flex items-center gap-2">
                            <input
                              type="number"
                              min={0}
                              step="0.1"
                              value={value}
                              disabled={tensorSplitAuto}
                              onChange={(e) => setSplitValue(idx, e.target.value)}
                              className="brand-input"
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

                <div className="brand-surface-muted p-4">
                  <div className="brand-label">Speculative / MTP</div>
                  <div className="mt-4 grid gap-4">
                    {binaryCaps.isSuccess && !mtpSupported && (
                      <div className="rounded-none border border-amber-300/25 bg-amber-300/10 px-3 py-2 text-sm text-amber-100">
                        Current llama-server does not expose <code>draft-mtp</code>. Rebuild from a recent upstream commit before launching MTP.
                      </div>
                    )}
                    <Field label="Mode">
                      <select
                        value={config.spec_type ?? "none"}
                        onChange={(e) => upd("spec_type", e.target.value as "none" | "draft-mtp")}
                        className="brand-input"
                      >
                        <option value="none">Disabled</option>
                        <option value="draft-mtp" disabled={binaryCaps.isSuccess && !mtpSupported}>
                          MTP (draft-mtp)
                        </option>
                      </select>
                    </Field>

                    {mtpEnabled && (
                      <>
                        <Field label="--spec-draft-model">
                          <input
                            value={config.spec_draft_model ?? ""}
                            onChange={(e) => upd("spec_draft_model", e.target.value)}
                            className="brand-input"
                            placeholder="Optional path or HF spec for an MTP GGUF"
                          />
                        </Field>
                        <Field label="--spec-draft-hf">
                          <input
                            value={config.spec_draft_hf ?? ""}
                            onChange={(e) => upd("spec_draft_hf", e.target.value)}
                            className="brand-input"
                            placeholder="unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ1_M"
                          />
                        </Field>

                        <div className="grid gap-3 md:grid-cols-2">
                          <Field label="--spec-draft-n-max">
                            <input
                              type="number"
                              value={config.spec_draft_n_max ?? ""}
                              onChange={(e) => updOptionalInt("spec_draft_n_max", e.target.value)}
                              className="brand-input"
                              placeholder="llama default"
                            />
                          </Field>
                          <Field label="--spec-draft-n-min">
                            <input
                              type="number"
                              value={config.spec_draft_n_min ?? ""}
                              onChange={(e) => updOptionalInt("spec_draft_n_min", e.target.value)}
                              className="brand-input"
                              placeholder="llama default"
                            />
                          </Field>
                          <Field label="--spec-draft-p-min">
                            <input
                              type="number"
                              step="0.01"
                              value={config.spec_draft_p_min ?? ""}
                              onChange={(e) => updOptionalFloat("spec_draft_p_min", e.target.value)}
                              className="brand-input"
                              placeholder="llama default"
                            />
                          </Field>
                          <Field label="Backend sampling">
                            <select
                              value={config.spec_draft_backend_sampling ?? "default"}
                              onChange={(e) => upd("spec_draft_backend_sampling", e.target.value as "default" | "on" | "off")}
                              className="brand-input"
                            >
                              <option value="default">Default</option>
                              <option value="on">On</option>
                              <option value="off">Off</option>
                            </select>
                          </Field>
                        </div>

                        <details className="rounded-none border border-white/10 bg-ink-400/60 p-3">
                          <summary className="cursor-pointer text-sm font-semibold text-bone-100">
                            Advanced MTP parameters
                          </summary>
                          <div className="mt-4 grid gap-3 md:grid-cols-2">
                            <Field label="--spec-draft-p-split">
                              <input
                                type="number"
                                step="0.01"
                                value={config.spec_draft_p_split ?? ""}
                                onChange={(e) => updOptionalFloat("spec_draft_p_split", e.target.value)}
                                className="brand-input"
                              />
                            </Field>
                            <Field label="--spec-draft-ngl">
                              <input
                                value={config.spec_draft_ngl ?? ""}
                                onChange={(e) => upd("spec_draft_ngl", e.target.value)}
                                className="brand-input"
                                placeholder="auto, all, or number"
                              />
                            </Field>
                            <Field label="--spec-draft-device">
                              <input
                                value={config.spec_draft_device ?? ""}
                                onChange={(e) => upd("spec_draft_device", e.target.value)}
                                className="brand-input"
                                placeholder="0,1"
                              />
                            </Field>
                            <Field label="--spec-draft-type-k">
                              <select
                                value={config.spec_draft_type_k ?? ""}
                                onChange={(e) => upd("spec_draft_type_k", e.target.value)}
                                className="brand-input"
                              >
                                {SPEC_DRAFT_CACHE_TYPES.map((value) => (
                                  <option key={value || "default"} value={value}>
                                    {value || "Default"}
                                  </option>
                                ))}
                              </select>
                            </Field>
                            <Field label="--spec-draft-type-v">
                              <select
                                value={config.spec_draft_type_v ?? ""}
                                onChange={(e) => upd("spec_draft_type_v", e.target.value)}
                                className="brand-input"
                              >
                                {SPEC_DRAFT_CACHE_TYPES.map((value) => (
                                  <option key={value || "default"} value={value}>
                                    {value || "Default"}
                                  </option>
                                ))}
                              </select>
                            </Field>
                            <Field label="--spec-draft-override-tensor">
                              <input
                                value={config.spec_draft_override_tensor ?? ""}
                                onChange={(e) => upd("spec_draft_override_tensor", e.target.value)}
                                className="brand-input"
                                placeholder="pattern=CPU"
                              />
                            </Field>
                          </div>

                          <div className="mt-4 grid gap-3 text-sm text-bone-200">
                            <label className="flex items-center gap-3">
                              <input
                                type="checkbox"
                                checked={Boolean(config.spec_draft_cpu_moe)}
                                onChange={(e) => upd("spec_draft_cpu_moe", e.target.checked)}
                                className="h-4 w-4 rounded border-white/20 bg-ink-400 text-lime-300 accent-lime-300"
                              />
                              Keep all draft MoE weights on CPU
                            </label>
                          </div>

                          <div className="mt-4 grid gap-3 md:grid-cols-2">
                            {[
                              ["spec_draft_n_cpu_moe", "--spec-draft-n-cpu-moe"],
                              ["spec_draft_threads", "--spec-draft-threads"],
                              ["spec_draft_threads_batch", "--spec-draft-threads-batch"],
                              ["spec_draft_cpu_strict", "--spec-draft-cpu-strict"],
                              ["spec_draft_prio", "--spec-draft-prio"],
                              ["spec_draft_poll", "--spec-draft-poll"],
                              ["spec_draft_cpu_strict_batch", "--spec-draft-cpu-strict-batch"],
                              ["spec_draft_prio_batch", "--spec-draft-prio-batch"],
                              ["spec_draft_poll_batch", "--spec-draft-poll-batch"],
                            ].map(([key, label]) => (
                              <Field key={key} label={label}>
                                <input
                                  type="number"
                                  value={(config[key as keyof InstanceConfig] as number | null | undefined) ?? ""}
                                  onChange={(e) => updOptionalInt(key as keyof InstanceConfig, e.target.value)}
                                  className="brand-input"
                                />
                              </Field>
                            ))}
                            {[
                              ["spec_draft_cpu_mask", "--spec-draft-cpu-mask"],
                              ["spec_draft_cpu_range", "--spec-draft-cpu-range"],
                              ["spec_draft_cpu_mask_batch", "--spec-draft-cpu-mask-batch"],
                            ].map(([key, label]) => (
                              <Field key={key} label={label}>
                                <input
                                  value={(config[key as keyof InstanceConfig] as string | null | undefined) ?? ""}
                                  onChange={(e) => updTextKey(key as keyof InstanceConfig, e.target.value)}
                                  className="brand-input"
                                />
                              </Field>
                            ))}
                          </div>
                        </details>
                      </>
                    )}
                  </div>
                </div>

                <div className="brand-surface-muted p-4">
                  <div className="brand-label">Launch toggles</div>
                  <div className="mt-4 grid gap-3 text-sm text-bone-200">
                    <label className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        checked={config.cpu_moe}
                        onChange={(e) => upd("cpu_moe", e.target.checked)}
                        className="h-4 w-4 rounded border-white/20 bg-ink-400 text-lime-300 accent-lime-300"
                      />
                      Offload all MoE experts to CPU
                    </label>
                    <label className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        checked={config.jinja}
                        onChange={(e) => upd("jinja", e.target.checked)}
                        className="h-4 w-4 rounded border-white/20 bg-ink-400 text-lime-300 accent-lime-300"
                      />
                      Enable Jinja chat template support
                    </label>
                    <label className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        checked={autoStart}
                        onChange={(e) => setAutoStart(e.target.checked)}
                        className="h-4 w-4 rounded border-white/20 bg-ink-400 text-lime-300 accent-lime-300"
                      />
                      Start immediately after save
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 border-t border-white/10 pt-4">
              <button
                type="submit"
                disabled={createMut.isPending}
                className="brand-btn-primary"
              >
                {createMut.isPending ? "Creating…" : "Create instance"}
              </button>
              <span className="text-sm text-bone-400">
                {autoStart ? "The backend will try to launch it right away." : "The instance record will be saved in stopped state."}
              </span>
              {formError && <span className="text-sm text-rose-300">{formError}</span>}
            </div>
          </form>
        </Panel>
      )}

      <Panel
        title="Managed instances"
        subtitle="Scrollable on smaller screens, with the high-friction actions grouped at the far right."
      >
        <div className="brand-table-wrap">
          <table className="min-w-[980px] w-full text-sm">
            <thead className="bg-ink-400/78">
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
                  className="border-t border-white/10 hover:bg-ink-300/50"
                >
                  <td className="px-4 py-3 font-medium text-bone-50">
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
                  <td className="px-4 py-3">
                    <div className="flex justify-end gap-2">
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
                    </div>
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

function SummaryCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint: string;
  tone?: "default" | "success" | "danger";
}) {
  const toneClass =
    tone === "success"
      ? "text-lime-200"
      : tone === "danger"
        ? "text-rose-200"
        : "text-bone-50";
  return (
    <div className="brand-stat">
      <div className="brand-label">{label}</div>
      <div className={`mt-3 text-4xl font-bold tracking-tight ${toneClass}`}>
        {value}
      </div>
      <div className="mt-4 text-sm text-bone-300">{hint}</div>
    </div>
  );
}
