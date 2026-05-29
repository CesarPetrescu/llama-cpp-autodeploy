import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, type GpuInfo, type GpuProcessInfo } from "@/api/client";
import { PageHeader } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";

function formatUptime(seconds: number | null): string {
  if (!seconds || seconds < 0) return "—";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function formatRelativeTime(timestamp: number | null): string {
  if (!timestamp) return "—";
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatBytes(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let next = value;
  let idx = 0;
  while (next >= 1024 && idx < units.length - 1) {
    next /= 1024;
    idx += 1;
  }
  return `${next >= 10 || idx === 0 ? next.toFixed(0) : next.toFixed(1)} ${units[idx]}`;
}

function formatPercent(value: number | null, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}%`;
}

function formatLoad(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(2);
}

function toneForPressure(value: number | null): "default" | "success" | "danger" {
  if (value == null) return "default";
  if (value >= 90) return "danger";
  if (value <= 55) return "success";
  return "default";
}

function computeGpuUsed(gpu: GpuInfo): number | null {
  if (gpu.used != null) return gpu.used;
  if (gpu.total != null && gpu.free != null) return Math.max(gpu.total - gpu.free, 0);
  return null;
}

function computeGpuPercent(gpu: GpuInfo): number | null {
  const used = computeGpuUsed(gpu);
  if (gpu.memory_percent != null) return gpu.memory_percent;
  if (used != null && gpu.total != null && gpu.total > 0) {
    return Number(((used / gpu.total) * 100).toFixed(1));
  }
  return null;
}

function gpuIndexLabel(gpu: GpuInfo): string {
  if (gpu.system_index != null && gpu.system_index !== gpu.index) {
    return `CUDA GPU #${gpu.index} · nvidia-smi #${gpu.system_index}`;
  }
  return `CUDA GPU #${gpu.index}`;
}

function buildRef(build: { config: Record<string, unknown> }): string {
  return String(build.config.ref ?? "latest");
}

export default function Dashboard() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15000,
  });
  const instances = useQuery({
    queryKey: ["instances"],
    queryFn: api.listInstances,
    refetchInterval: 4000,
  });
  const gpus = useQuery({
    queryKey: ["gpus"],
    queryFn: api.listGpus,
    refetchInterval: 4000,
  });
  const builds = useQuery({
    queryKey: ["builds"],
    queryFn: api.listBuilds,
    refetchInterval: 4000,
  });
  const benchmarks = useQuery({
    queryKey: ["benchmarks"],
    queryFn: api.listBenchmarks,
    refetchInterval: 4000,
  });
  const local = useQuery({
    queryKey: ["local-models"],
    queryFn: api.listLocal,
    refetchInterval: 30000,
  });

  const allInstances = [...(instances.data?.instances ?? [])].sort((a, b) => {
    const statusRank = (status: string) =>
      status === "running" ? 0 : status === "starting" ? 1 : status === "crashed" ? 2 : 3;
    return statusRank(a.status) - statusRank(b.status) || (b.started_at ?? 0) - (a.started_at ?? 0);
  });
  const buildList = [...(builds.data?.builds ?? [])].sort(
    (a, b) => (b.started_at ?? 0) - (a.started_at ?? 0),
  );
  const benchmarkList = [...(benchmarks.data?.benchmarks ?? [])].sort(
    (a, b) => (b.started_at ?? 0) - (a.started_at ?? 0),
  );
  const models = [...(local.data?.models ?? [])].sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
  const gpuList = gpus.data?.gpus ?? [];
  const host = gpus.data?.system ?? null;

  const runningInstances = allInstances.filter((i) => i.status === "running").length;
  const startingInstances = allInstances.filter((i) => i.status === "starting").length;
  const stoppedInstances = allInstances.filter((i) => i.status === "stopped").length;
  const crashedInstances = allInstances.filter((i) => i.status === "crashed").length;
  const runningBuilds = buildList.filter((build) => build.status === "running").length;
  const failedBuilds = buildList.filter((build) => build.status === "failure").length;
  const runningBenchmarks = benchmarkList.filter((benchmark) => benchmark.status === "running").length;
  const failedBenchmarks = benchmarkList.filter((benchmark) => benchmark.status === "failure").length;
  const bestBenchmarkTg = benchmarkList
    .map((benchmark) => benchmark.summary?.best_tg?.avg_ts ?? null)
    .filter((value): value is number => value != null)
    .sort((a, b) => b - a)[0] ?? null;

  const gpuTotals = gpuList.reduce(
    (acc, gpu) => {
      if (gpu.total != null) acc.total += gpu.total;
      if (gpu.free != null) acc.free += gpu.free;
      const used = computeGpuUsed(gpu);
      if (used != null) acc.used += used;
      return acc;
    },
    { total: 0, free: 0, used: 0 },
  );
  const gpuMemoryPercent =
    gpuTotals.total > 0 ? Number(((gpuTotals.used / gpuTotals.total) * 100).toFixed(1)) : null;
  const gpuComputeSamples = gpuList
    .map((gpu) => gpu.utilization_gpu)
    .filter((value): value is number => value != null);
  const gpuComputePercent =
    gpuComputeSamples.length > 0
      ? Number(
          (
            gpuComputeSamples.reduce((sum, value) => sum + value, 0) /
            gpuComputeSamples.length
          ).toFixed(1),
        )
      : null;
  const hottestGpu = [...gpuList]
    .sort(
      (a, b) =>
        (b.utilization_gpu ?? computeGpuPercent(b) ?? -1) -
        (a.utilization_gpu ?? computeGpuPercent(a) ?? -1),
    )[0];
  const busiestCore = [...(host?.cores ?? [])].sort((a, b) => (b.percent ?? -1) - (a.percent ?? -1))[0];

  const alerts = [
    health.isError
      ? {
          tone: "danger" as const,
          label: "Backend unreachable",
          detail: (health.error as Error).message,
        }
      : null,
    health.data && !health.data.llama_server_present
      ? {
          tone: "danger" as const,
          label: "llama-server missing",
          detail: "Build or install the binary before launching workloads.",
        }
      : null,
    crashedInstances > 0
      ? {
          tone: "danger" as const,
          label: `${crashedInstances} crashed instance${crashedInstances === 1 ? "" : "s"}`,
          detail: "Review logs or restart from Instances.",
        }
      : null,
    failedBuilds > 0
      ? {
          tone: "danger" as const,
          label: `${failedBuilds} failed build${failedBuilds === 1 ? "" : "s"}`,
          detail: "Toolchain output needs attention.",
        }
      : null,
    failedBenchmarks > 0
      ? {
          tone: "danger" as const,
          label: `${failedBenchmarks} failed benchmark${failedBenchmarks === 1 ? "" : "s"}`,
          detail: "Inspect the benchmark history and logs.",
        }
      : null,
    host?.cpu_percent != null && host.cpu_percent >= 90
      ? {
          tone: "warning" as const,
          label: `CPU pressure ${formatPercent(host.cpu_percent)}`,
          detail: "Host compute headroom is tight.",
        }
      : null,
    host?.memory_percent != null && host.memory_percent >= 90
      ? {
          tone: "warning" as const,
          label: `System RAM ${formatPercent(host.memory_percent)}`,
          detail: "Memory is close to saturation.",
        }
      : null,
    gpuList.some((gpu) => (gpu.utilization_gpu ?? 0) >= 95 || (computeGpuPercent(gpu) ?? 0) >= 95)
      ? {
          tone: "warning" as const,
          label: `GPU pressure ${formatPercent(gpuMemoryPercent, 1)}`,
          detail: hottestGpu ? `${gpuIndexLabel(hottestGpu)} is the current hotspot.` : "GPU headroom is low.",
        }
      : null,
    runningBuilds > 0
      ? {
          tone: "warning" as const,
          label: `${runningBuilds} active build${runningBuilds === 1 ? "" : "s"}`,
          detail: "Compiler work is running now.",
        }
      : null,
    runningBenchmarks > 0
      ? {
          tone: "warning" as const,
          label: `${runningBenchmarks} active benchmark${runningBenchmarks === 1 ? "" : "s"}`,
          detail: "Throughput testing is running now.",
        }
      : null,
    health.data && !health.data.auth_required
      ? {
          tone: "warning" as const,
          label: "Auth disabled",
          detail: "Backend is open unless another gate is in front of it.",
        }
      : null,
  ].filter(Boolean) as Array<{ tone: "warning" | "danger"; label: string; detail: string }>;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow="Control plane"
        title="Overview"
        description="Summary first. Expand GPUs, builds, and fleet only when you need the details."
        actions={
          <>
            <Link to="/instances" className="brand-btn-ghost">
              Instances
            </Link>
            <Link to="/builds" className="brand-btn-ghost">
              Builds
            </Link>
            <Link to="/benchmarks" className="brand-btn-ghost">
              Benchmarks
            </Link>
            <Link to="/library" className="brand-btn-primary">
              Library
            </Link>
          </>
        }
      />

      <div className="grid gap-4 grid-cols-2 xl:grid-cols-6">
        <MetricCard
          label="Backend"
          value={health.data?.status === "ok" ? "ok" : health.isError ? "down" : "check"}
          tone={health.data?.status === "ok" ? "success" : health.isError ? "danger" : "default"}
          meta={[
            health.data?.service ?? "service —",
            health.data?.version ? `v${health.data.version}` : "version —",
          ]}
        />
        <MetricCard
          label="Fleet"
          value={`${runningInstances}/${allInstances.length}`}
          tone={runningInstances > 0 ? "success" : "default"}
          meta={[`${startingInstances} starting`, `${crashedInstances} crashed`]}
        />
        <MetricCard
          label="GPU compute"
          value={formatPercent(gpuComputePercent, 1)}
          tone={toneForPressure(gpuComputePercent)}
          meta={[
            `${gpuList.length} device${gpuList.length === 1 ? "" : "s"}`,
            hottestGpu
              ? `hot #${hottestGpu.index} ${formatPercent(hottestGpu.utilization_gpu ?? computeGpuPercent(hottestGpu), 1)}`
              : "no probe",
          ]}
        />
        <MetricCard
          label="VRAM"
          value={formatPercent(gpuMemoryPercent, 1)}
          tone={toneForPressure(gpuMemoryPercent)}
          meta={[`used ${formatBytes(gpuTotals.used)}`, `free ${formatBytes(gpuTotals.free)}`]}
        />
        <MetricCard
          label="CPU"
          value={formatPercent(host?.cpu_percent ?? null, 1)}
          tone={toneForPressure(host?.cpu_percent ?? null)}
          meta={[
            host?.cpu_count_logical != null ? `${host.cpu_count_logical} logical` : "logical —",
            busiestCore ? `hot core ${busiestCore.index} ${formatPercent(busiestCore.percent, 1)}` : "core probe —",
          ]}
        />
        <MetricCard
          label="RAM"
          value={formatPercent(host?.memory_percent ?? null, 1)}
          tone={toneForPressure(host?.memory_percent ?? null)}
          meta={[
            host?.memory_used_h ? `used ${host.memory_used_h}` : "used —",
            host?.memory_total_h ? `total ${host.memory_total_h}` : "total —",
          ]}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.78fr_1.22fr]">
        <Panel
          title="Attention"
          actions={
            <Link to="/instances" className="brand-btn-ghost px-3 py-1.5 text-xs">
              Open instances
            </Link>
          }
        >
          <div className="space-y-2">
            {alerts.length > 0 ? (
              alerts.map((alert) => (
                <AlertRow
                  key={alert.label}
                  tone={alert.tone}
                  label={alert.label}
                  detail={alert.detail}
                />
              ))
            ) : (
              <div className="border border-white/10 bg-ink-300/70 px-4 py-4 text-sm text-lime-200">
                No active alerts.
              </div>
            )}
          </div>
        </Panel>

        <Panel title="Runtime facts">
          {health.isLoading && <div className="text-sm text-bone-400">Loading runtime state…</div>}
          {health.data && (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <FactCard label="Service" value={health.data.service} />
              <FactCard label="Version" value={health.data.version} />
              <FactCard label="Auth" value={health.data.auth_required ? "token" : "open"} />
              <FactCard label="llama-server" value={health.data.llama_server_present ? "present" : "missing"} />
              <FactCard label="Models dir" value={health.data.models_dir} mono />
              <FactCard label="Local weights" value={`${models.length}`} />
              <FactCard label="Latest build" value={buildList[0]?.status ?? "—"} />
              <FactCard label="Latest ref" value={buildList[0] ? buildRef(buildList[0]) : "latest"} mono />
              <FactCard label="Latest bench" value={benchmarkList[0]?.status ?? "—"} />
              <FactCard label="Best tg" value={bestBenchmarkTg == null ? "—" : `${bestBenchmarkTg.toFixed(2)} t/s`} />
            </div>
          )}
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.02fr_0.98fr]">
        <Panel title="Host resources" subtitle="Core host stats stay visible. Per-core detail is expandable.">
          {host ? (
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <HostFact
                  label="CPU"
                  value={formatPercent(host.cpu_percent, 1)}
                  detail={`logical ${host.cpu_count_logical ?? "—"} / physical ${host.cpu_count_physical ?? "—"}`}
                />
                <HostFact
                  label="RAM"
                  value={formatPercent(host.memory_percent, 1)}
                  detail={`${host.memory_used_h} used / ${host.memory_total_h} total`}
                />
                <HostFact
                  label="Load 1m"
                  value={formatLoad(host.load_1)}
                  detail={`5m ${formatLoad(host.load_5)} / 15m ${formatLoad(host.load_15)}`}
                />
                <HostFact
                  label="Available RAM"
                  value={host.memory_available_h}
                  detail="System free headroom"
                />
              </div>

              <div className="grid gap-3 xl:grid-cols-2">
                <ResourceMeter
                  label="CPU total"
                  value={host.cpu_percent}
                  detail={
                    busiestCore
                      ? `hot core ${busiestCore.index} at ${formatPercent(busiestCore.percent, 1)}`
                      : "per-core probe unavailable"
                  }
                />
                <ResourceMeter
                  label="System RAM"
                  value={host.memory_percent}
                  detail={`${host.memory_used_h} used / ${host.memory_available_h} free`}
                />
              </div>

              <ExpandableSection
                summary={
                  <div className="flex min-w-0 flex-1 flex-wrap items-center gap-3">
                    <div className="brand-label">Per-core load</div>
                    <CompactSummary label="cores" value={`${host.cores.length}`} />
                    <CompactSummary
                      label="hot core"
                      value={busiestCore ? `${busiestCore.index}` : "—"}
                      meta={busiestCore ? formatPercent(busiestCore.percent, 1) : "no data"}
                    />
                  </div>
                }
              >
                {host.cores.length > 0 ? (
                  <div className="brand-scroll-pane max-h-[16rem] pr-1 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                    {host.cores.map((core) => (
                      <CoreMeter key={core.index} index={core.index} value={core.percent} />
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-bone-400">No core telemetry available.</div>
                )}
              </ExpandableSection>
            </div>
          ) : (
            <div className="text-sm text-bone-400">No host telemetry available.</div>
          )}
        </Panel>

        <Panel title="Recent builds" subtitle="Latest builds stay compact until you expand one.">
          <div className="space-y-2">
            {buildList.slice(0, 6).map((build) => (
              <BuildDisclosure key={build.id} build={build} />
            ))}
            {buildList.length === 0 && (
              <div className="border border-white/10 bg-ink-300/70 px-4 py-8 text-center text-sm text-bone-500">
                No builds yet.
              </div>
            )}
          </div>
        </Panel>
      </div>

      <Panel
        title="GPU pressure"
        subtitle="Each GPU stays collapsed until you want process owners, VRAM, and runtime detail."
        actions={
          <Link to="/memory" className="brand-btn-ghost px-3 py-1.5 text-xs">
            Memory planner
          </Link>
        }
      >
        {gpuList.length > 0 ? (
          <div className="space-y-2">
            {gpuList.map((gpu) => (
              <GpuDisclosure key={gpu.index} gpu={gpu} />
            ))}
          </div>
        ) : (
          <div className="text-sm text-bone-400">No CUDA devices detected.</div>
        )}
      </Panel>

      <Panel title="Fleet" subtitle="The full table is hidden until you open it.">
        <ExpandableSection
          summary={
            <div className="flex min-w-0 flex-1 flex-wrap items-center gap-3">
              <div className="brand-label">Managed instances</div>
              <CompactSummary label="running" value={`${runningInstances}`} />
              <CompactSummary label="starting" value={`${startingInstances}`} />
              <CompactSummary label="stopped" value={`${stoppedInstances}`} />
              <CompactSummary label="crashed" value={`${crashedInstances}`} />
            </div>
          }
        >
          <div className="brand-table-wrap brand-scroll-pane max-h-[24rem]">
            <table className="min-w-[900px] w-full text-sm">
              <thead className="bg-ink-400/78">
                <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                  <th className="px-4 py-3 font-semibold">Name</th>
                  <th className="px-4 py-3 font-semibold">Status</th>
                  <th className="px-4 py-3 font-semibold">Model</th>
                  <th className="px-4 py-3 font-semibold">Endpoint</th>
                  <th className="px-4 py-3 font-semibold">PID</th>
                  <th className="px-4 py-3 font-semibold">Uptime</th>
                </tr>
              </thead>
              <tbody>
                {allInstances.map((inst) => (
                  <tr key={inst.id} className="border-t border-white/10 hover:bg-ink-300/50">
                    <td className="px-4 py-3">
                      <div className="font-semibold text-bone-50">{inst.name}</div>
                      <div className="mt-1 text-xs text-bone-400">{inst.id}</div>
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={inst.status} />
                    </td>
                    <td className="max-w-[18rem] truncate px-4 py-3 text-bone-300" title={inst.config.model_ref}>
                      {inst.config.model_ref || "—"}
                    </td>
                    <td className="px-4 py-3 font-mono text-[12px] text-bone-300">
                      {inst.host ?? "—"}:{inst.port ?? "—"}
                    </td>
                    <td className="px-4 py-3 font-mono text-[12px] text-bone-300">
                      {inst.pid ?? "—"}
                    </td>
                    <td className="px-4 py-3 text-bone-300">{formatUptime(inst.uptime_s)}</td>
                  </tr>
                ))}
                {allInstances.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-10 text-center text-bone-500">
                      No instances yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </ExpandableSection>
      </Panel>
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
  tone?: "default" | "success" | "danger";
}) {
  const valueClass =
    tone === "success" ? "text-lime-200" : tone === "danger" ? "text-rose-200" : "text-bone-50";
  return (
    <div className="brand-stat">
      <div className="brand-label">{label}</div>
      <div className={`mt-3 break-words text-4xl font-bold tracking-tight ${valueClass}`}>{value}</div>
      <div className="mt-4 space-y-1 text-xs text-bone-400">
        {meta.map((item) => (
          <div key={item}>{item}</div>
        ))}
      </div>
    </div>
  );
}

function AlertRow({
  tone,
  label,
  detail,
}: {
  tone: "warning" | "danger";
  label: string;
  detail: string;
}) {
  const toneClass =
    tone === "danger"
      ? "border-rose-500/40 bg-rose-500/10 text-rose-200"
      : "border-amber-400/40 bg-amber-400/10 text-amber-200";
  return (
    <div className={`border px-4 py-3 ${toneClass}`}>
      <div className="text-sm font-semibold">{label}</div>
      <div className="mt-1 text-xs leading-5">{detail}</div>
    </div>
  );
}

function FactCard({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="border border-white/10 bg-ink-300/70 p-3">
      <div className="brand-label">{label}</div>
      <div className={`mt-2 break-all text-sm ${mono ? "font-mono text-[12px] text-bone-100" : "text-bone-100"}`}>
        {value}
      </div>
    </div>
  );
}

function HostFact({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="border border-white/10 bg-ink-400/50 p-3">
      <div className="brand-label">{label}</div>
      <div className="mt-2 text-2xl font-semibold tracking-tight text-bone-50">{value}</div>
      <div className="mt-2 text-xs leading-5 text-bone-400">{detail}</div>
    </div>
  );
}

function MiniStat({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="border border-white/10 bg-ink-500/70 p-2.5">
      <div className="brand-label">{label}</div>
      <div className="mt-1 break-words text-sm font-semibold text-bone-50">{value}</div>
      <div className="mt-1 text-[11px] leading-5 text-bone-400">{detail}</div>
    </div>
  );
}

function CompactSummary({
  label,
  value,
  meta,
}: {
  label: string;
  value: string;
  meta?: string;
}) {
  return (
    <div className="border border-white/10 bg-ink-500/70 px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-[0.18em] text-bone-400">{label}</div>
      <div className="mt-1 text-sm font-semibold text-bone-50">{value}</div>
      {meta && <div className="mt-1 text-[10px] text-bone-500">{meta}</div>}
    </div>
  );
}

function ResourceMeter({
  label,
  value,
  detail,
}: {
  label: string;
  value: number | null;
  detail: string;
}) {
  const percent = value == null ? 0 : Math.max(0, Math.min(100, value));
  const fill =
    value == null ? "bg-bone-500/40" : value >= 90 ? "bg-rose-400" : value >= 70 ? "bg-amber-300" : "bg-lime-300";

  return (
    <div className="border border-white/10 bg-ink-400/50 p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="brand-label">{label}</div>
        <div className="text-sm font-semibold text-bone-50">{formatPercent(value, 1)}</div>
      </div>
      <div className="mt-3 h-2 overflow-hidden bg-ink-500/90">
        <div className={`h-full ${fill}`} style={{ width: `${percent}%` }} />
      </div>
      <div className="mt-2 text-xs leading-5 text-bone-400">{detail}</div>
    </div>
  );
}

function CoreMeter({ index, value }: { index: number; value: number | null }) {
  const percent = value == null ? 0 : Math.max(0, Math.min(100, value));
  const fill =
    value == null ? "bg-bone-500/40" : value >= 90 ? "bg-rose-400" : value >= 70 ? "bg-amber-300" : "bg-lime-300";

  return (
    <div className="border border-white/10 bg-ink-500/70 p-2">
      <div className="flex items-center justify-between gap-3 text-[11px] uppercase tracking-[0.18em] text-bone-400">
        <span>core {index}</span>
        <span className="text-bone-200">{formatPercent(value, 1)}</span>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden bg-ink-200/90">
        <div className={`h-full ${fill}`} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function ExpandableSection({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="brand-disclosure" open={defaultOpen}>
      <summary>
        <div className="min-w-0 flex-1">{summary}</div>
        <DisclosureChevron />
      </summary>
      <div className="brand-disclosure-body">{children}</div>
    </details>
  );
}

function DisclosureChevron() {
  return (
    <svg
      viewBox="0 0 16 16"
      aria-hidden="true"
      className="brand-disclosure-chevron shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path d="M6 3.5 10.5 8 6 12.5" strokeLinecap="square" strokeLinejoin="miter" />
    </svg>
  );
}

function BuildDisclosure({
  build,
}: {
  build: {
    id: string;
    config: Record<string, unknown>;
    started_at: number | null;
    finished_at: number | null;
    exit_code: number | null;
    status: string;
    pid: number | null;
    alive: boolean;
    cmdline?: string[];
  };
}) {
  return (
    <ExpandableSection
      summary={
        <div className="flex min-w-0 flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <div className="font-mono text-[12px] text-bone-50">{build.id}</div>
            <div className="mt-1 break-all text-xs text-bone-300">ref {buildRef(build)}</div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <CompactSummary label="started" value={formatRelativeTime(build.started_at)} />
            <CompactSummary label="pid" value={String(build.pid ?? "—")} />
            <StatusBadge status={build.status} />
          </div>
        </div>
      }
    >
      <div className="space-y-3">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <MiniStat label="Ref" value={buildRef(build)} detail="source target" />
          <MiniStat label="Exit" value={String(build.exit_code ?? "—")} detail="process exit code" />
          <MiniStat label="Alive" value={build.alive ? "yes" : "no"} detail="process liveness" />
          <MiniStat label="Finished" value={formatRelativeTime(build.finished_at)} detail="last completion" />
        </div>
        {build.cmdline && build.cmdline.length > 0 && (
          <div>
            <div className="brand-label">Command</div>
            <div className="brand-code-block brand-scroll-pane mt-2 max-h-[8rem]">{build.cmdline.join(" ")}</div>
          </div>
        )}
      </div>
    </ExpandableSection>
  );
}

function GpuDisclosure({ gpu }: { gpu: GpuInfo }) {
  const used = computeGpuUsed(gpu);
  const vramPercent = computeGpuPercent(gpu);
  const usedLabel = gpu.used_h ?? formatBytes(used);

  return (
    <ExpandableSection
      summary={
        <div className="flex min-w-0 flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <div className="brand-label">{gpuIndexLabel(gpu)}</div>
            <div className="mt-1 break-words text-sm font-semibold text-bone-50">{gpu.name}</div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <CompactSummary label="compute" value={formatPercent(gpu.utilization_gpu ?? null, 1)} />
            <CompactSummary label="VRAM" value={formatPercent(vramPercent, 1)} meta={usedLabel} />
            <CompactSummary
              label="procs"
              value={String(gpu.processes?.length ?? 0)}
              meta={gpu.processes?.length ? "attached" : "idle"}
            />
          </div>
        </div>
      }
    >
      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="grid gap-3">
          <ResourceMeter label="Compute" value={gpu.utilization_gpu ?? null} detail="overall GPU engine load" />
          <ResourceMeter label="Memory bus" value={gpu.utilization_memory ?? null} detail="overall VRAM traffic" />
          <ResourceMeter label="VRAM" value={vramPercent} detail={`${usedLabel} used / ${gpu.total_h} total`} />
        </div>
        <div>
          <div className="flex items-center justify-between gap-3">
            <div className="brand-label">Processes</div>
            <div className="text-xs text-bone-400">{gpu.processes?.length ?? 0} active</div>
          </div>
          <div className="brand-scroll-pane mt-3 max-h-[16rem] space-y-2 pr-1">
            {gpu.processes && gpu.processes.length > 0 ? (
              gpu.processes.map((process) => (
                <GpuProcessRow key={`${gpu.index}-${process.pid}-${process.process_name}`} process={process} />
              ))
            ) : (
              <div className="border border-white/10 bg-ink-400/50 px-3 py-3 text-sm text-bone-400">
                No active compute processes detected.
              </div>
            )}
          </div>
        </div>
      </div>
    </ExpandableSection>
  );
}

function GpuProcessRow({ process }: { process: GpuProcessInfo }) {
  const title = process.label || process.process_name || `pid ${process.pid}`;
  const detail = [process.kind, process.status, process.detail, `pid ${process.pid}`]
    .filter(Boolean)
    .join(" • ");

  return (
    <div className="grid gap-3 border border-white/10 bg-ink-400/50 px-3 py-3 lg:grid-cols-[minmax(0,1fr)_13rem]">
      <div className="min-w-0">
        <div className="break-words font-medium text-bone-100">{title}</div>
        <div className="mt-1 break-words text-xs leading-5 text-bone-400">{detail}</div>
        {process.label && process.process_name && process.label !== process.process_name && (
          <div className="mt-1 break-words text-[11px] text-bone-500">{process.process_name}</div>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2">
        <MiniStat label="% VRAM" value={formatPercent(process.memory_percent ?? null, 1)} detail="share of device VRAM" />
        <MiniStat label="VRAM" value={process.used_memory_h} detail="process framebuffer" />
      </div>
    </div>
  );
}
