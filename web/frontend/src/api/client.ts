// Tiny typed fetch wrapper for the backend API.

const TOKEN_KEY = "llama_web_token";
const BASE_KEY = "llama_web_base";

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function getBaseUrl(): string {
  return localStorage.getItem(BASE_KEY) || "";
}

export function setBaseUrl(base: string): void {
  localStorage.setItem(BASE_KEY, base.replace(/\/$/, ""));
}

function resolvePath(path: string): string {
  const base = getBaseUrl();
  if (!base) return path;
  return `${base}${path}`;
}

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers || {});
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(resolvePath(path), { ...init, headers });
  let payload: unknown = null;
  const text = await res.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!res.ok) {
    const detail =
      (payload && typeof payload === "object" && "detail" in payload
        ? (payload as { detail: unknown }).detail
        : payload) || res.statusText;
    throw new ApiError(res.status, detail);
  }
  return payload as T;
}

export function buildWsUrl(path: string): string {
  const base = getBaseUrl();
  let url: URL;
  if (base) {
    url = new URL(path, base);
  } else {
    url = new URL(path, window.location.origin);
  }
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const token = getToken();
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

// ---------------- typed helpers ----------------

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  auth_required: boolean;
  llama_server_present: boolean;
  models_dir: string;
}

export interface GpuInfo {
  index: number;
  system_index?: number | null;
  name: string;
  uuid?: string | null;
  total: number | null;
  free: number | null;
  used?: number | null;
  total_h: string;
  free_h: string;
  used_h?: string;
  utilization_gpu?: number | null;
  utilization_memory?: number | null;
  memory_percent?: number | null;
  processes?: GpuProcessInfo[];
}

export interface GpuProcessInfo {
  pid: number;
  process_name: string;
  raw_process_name?: string | null;
  label?: string | null;
  kind?: string | null;
  status?: string | null;
  detail?: string | null;
  used_memory: number | null;
  used_memory_h: string;
  memory_percent?: number | null;
}

export interface SystemCoreUsage {
  index: number;
  percent: number | null;
}

export interface SystemUsage {
  cpu_percent: number | null;
  cpu_count_logical: number | null;
  cpu_count_physical: number | null;
  load_1: number | null;
  load_5: number | null;
  load_15: number | null;
  memory_total: number | null;
  memory_total_h: string;
  memory_available: number | null;
  memory_available_h: string;
  memory_used: number | null;
  memory_used_h: string;
  memory_percent: number | null;
  cores: SystemCoreUsage[];
}

export interface GpuUsage {
  info: GpuInfo;
  weights: number;
  weights_h: string;
  kv: number;
  kv_h: string;
}

export interface MemoryPlan {
  source_label: string;
  model_label: string;
  quant: string | null;
  param_count: number | null;
  param_count_h: string;
  weights_total: number | null;
  weights_total_h: string;
  weights_gpu: number;
  weights_gpu_h: string;
  weights_cpu: number;
  weights_cpu_h: string;
  kv_total: number;
  kv_total_h: string;
  ctx_size: number | null;
  layers_est: number | null;
  gpus: GpuUsage[];
  cpu_total: number | null;
  cpu_total_h: string;
  cpu_available: number | null;
  cpu_available_h: string;
  cpu_weights: number;
  cpu_weights_h: string;
  cpu_kv: number;
  cpu_kv_h: string;
  warnings: string[];
  summary: string[];
}

export interface LocalModel {
  name: string;
  path: string;
  rel?: string;
  size: number | null;
  size_h: string;
  params: number | null;
  params_h: string;
  quant: string | null;
}

export interface SupportedFlags {
  bool_flags: string[];
  choice_flags: Record<string, string[]>;
  value_flags: string[];
  usage: string;
  summary: string;
  options: Array<{
    flag: string;
    syntax: string;
    description: string;
    choices: string[];
    metavar: string | null;
    kind: "bool" | "choice" | "value" | "meta";
  }>;
}

export interface InstancesEvent {
  type: "instances.snapshot";
  total: number;
  running: number;
  instances: Instance[];
}

export interface BuildsEvent {
  type: "builds.snapshot";
  total: number;
  running: number;
  builds: BuildRecord[];
}

export interface BenchmarksEvent {
  type: "benchmarks.snapshot";
  total: number;
  running: number;
  benchmarks: BenchmarkRecord[];
}

export interface InstanceConfig {
  mode: "llm" | "embed";
  model_ref: string;
  models_dir?: string | null;
  hf_token?: string | null;
  host: string;
  port: number;
  n_gpu_layers?: number | null;
  gpu_strategy?: string | null;
  gpu_devices?: string | null;
  auto_split_policy?: string | null;
  tensor_split?: string | null;
  split_mode?: string | null;
  ctx_size?: number | null;
  n_cpu_moe?: number | null;
  cpu_moe: boolean;
  mmproj?: string | null;
  jinja: boolean;
  reasoning_format?: string | null;
  no_context_shift: boolean;
  extra_flags: string;
}

export interface Instance {
  id: string;
  name: string;
  kind: string;
  config: InstanceConfig;
  pid: number | null;
  cmdline: string[];
  started_at: number | null;
  stopped_at: number | null;
  last_exit: number | null;
  status: "running" | "stopping" | "stopped" | "crashed" | "starting";
  host: string | null;
  port: number | null;
  log_file: string | null;
  restart_policy: string;
  uptime_s: number | null;
  alive: boolean;
}

export interface BuildRecord {
  id: string;
  config: Record<string, unknown>;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  status: "pending" | "running" | "cancelling" | "cancelled" | "success" | "failure";
  log_file: string | null;
  pid: number | null;
  pgid?: number | null;
  cmdline?: string[];
  alive: boolean;
}

export interface RecoverInstancesResponse {
  recovered: Instance[];
  instances: Instance[];
}

export interface BenchmarkRow {
  test: string;
  avg_ts: number | null;
  stddev_ts: number | null;
  backend: string | null;
  model_type: string | null;
  threads: number | null;
  n_gpu_layers: number | null;
  batch_size: number | null;
  ubatch_size: number | null;
  n_prompt: number | null;
  n_gen: number | null;
  n_depth: number | null;
  raw: Record<string, unknown>;
}

export interface BenchmarkSummaryStat {
  test?: string | null;
  avg_ts?: number | null;
}

export interface BenchmarkSummary {
  row_count: number;
  model_type?: string | null;
  backend?: string | null;
  best_overall?: BenchmarkSummaryStat;
  best_pp?: BenchmarkSummaryStat;
  best_tg?: BenchmarkSummaryStat;
  best_pg?: BenchmarkSummaryStat;
}

export interface BenchmarkRecord {
  id: string;
  name: string;
  config: Record<string, unknown>;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  status: "pending" | "running" | "cancelling" | "cancelled" | "success" | "failure";
  log_file: string | null;
  result_file?: string | null;
  pid: number | null;
  pgid?: number | null;
  cmdline?: string[];
  alive: boolean;
  resolved_model?: string | null;
  result_rows?: BenchmarkRow[];
  summary?: BenchmarkSummary;
  parse_error?: string | null;
}

// API methods --------------------------------------------------------------

export const api = {
  health: () => apiFetch<HealthResponse>("/api/health"),
  listGpus: () => apiFetch<{ gpus: GpuInfo[]; system: SystemUsage }>("/api/memory/gpus"),
  planMemory: (state: Record<string, unknown>) =>
    apiFetch<MemoryPlan>("/api/memory/plan", {
      method: "POST",
      body: JSON.stringify(state),
    }),
  listLocal: () =>
    apiFetch<{ models_dir: string; models: LocalModel[] }>("/api/models/local"),
  downloadModel: (spec: string, hf_token?: string) =>
    apiFetch<{ model: LocalModel }>("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ spec, hf_token: hf_token || null }),
    }),
  renameModel: (name: string, new_name: string) =>
    apiFetch<{ model: LocalModel; old_name: string; new_name: string }>(
      `/api/models/local/${encodeURIComponent(name)}/rename`,
      {
        method: "POST",
        body: JSON.stringify({ new_name }),
      },
    ),
  deleteModel: (name: string) =>
    apiFetch<{ deleted: string; path: string }>(
      `/api/models/local/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  supportedFlags: () =>
    apiFetch<SupportedFlags>("/api/builds/supported-flags"),
  listInstances: () =>
    apiFetch<{ instances: Instance[] }>("/api/instances"),
  createInstance: (
    name: string,
    config: InstanceConfig,
    auto_start: boolean,
  ) =>
    apiFetch<{ instance: Instance }>("/api/instances", {
      method: "POST",
      body: JSON.stringify({ name, config, auto_start }),
    }),
  getInstance: (id: string) =>
    apiFetch<{ instance: Instance; logs: string[] }>(`/api/instances/${id}`),
  startInstance: (id: string) =>
    apiFetch<{ instance: Instance }>(`/api/instances/${id}/start`, {
      method: "POST",
    }),
  recoverInstances: () =>
    apiFetch<RecoverInstancesResponse>("/api/instances/recover", {
      method: "POST",
    }),
  stopInstance: (id: string) =>
    apiFetch<{ instance: Instance }>(`/api/instances/${id}/stop`, {
      method: "POST",
    }),
  restartInstance: (id: string) =>
    apiFetch<{ instance: Instance }>(`/api/instances/${id}/restart`, {
      method: "POST",
    }),
  deleteInstance: (id: string) =>
    apiFetch<{ deleted: string }>(`/api/instances/${id}`, { method: "DELETE" }),
  listBuilds: () => apiFetch<{ builds: BuildRecord[] }>("/api/builds"),
  startBuild: (payload: Record<string, unknown>) =>
    apiFetch<{ build: BuildRecord }>("/api/builds", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getBuild: (id: string) =>
    apiFetch<{ build: BuildRecord; logs: string[] }>(`/api/builds/${id}`),
  stopBuild: (id: string) =>
    apiFetch<{ build: BuildRecord }>(`/api/builds/${id}/stop`, {
      method: "POST",
    }),
  listBenchmarks: () => apiFetch<{ benchmarks: BenchmarkRecord[] }>("/api/benchmarks"),
  startBenchmark: (payload: Record<string, unknown>) =>
    apiFetch<{ benchmark: BenchmarkRecord }>("/api/benchmarks", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getBenchmark: (id: string) =>
    apiFetch<{ benchmark: BenchmarkRecord; logs: string[] }>(`/api/benchmarks/${id}`),
  stopBenchmark: (id: string) =>
    apiFetch<{ benchmark: BenchmarkRecord }>(`/api/benchmarks/${id}/stop`, {
      method: "POST",
    }),
};
