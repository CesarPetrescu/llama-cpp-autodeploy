import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { PageHeader } from "@/components/PageHeader";

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

function MetricCard({
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
  const valueClass =
    tone === "success"
      ? "text-lime-200"
      : tone === "danger"
        ? "text-rose-200"
        : "text-bone-50";

  return (
    <div className="brand-stat">
      <div className="brand-label">{label}</div>
      <div className={`mt-3 text-4xl font-bold tracking-tight ${valueClass}`}>
        {value}
      </div>
      <div className="mt-4 text-sm text-bone-300">{hint}</div>
    </div>
  );
}

export default function Dashboard() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const instances = useQuery({
    queryKey: ["instances"],
    queryFn: api.listInstances,
  });
  const gpus = useQuery({
    queryKey: ["gpus"],
    queryFn: api.listGpus,
    refetchInterval: 4000,
  });

  const allInstances = instances.data?.instances ?? [];
  const totalInstances = allInstances.length;
  const runningInstances = allInstances.filter((i) => i.status === "running").length;
  const crashedInstances = allInstances.filter((i) => i.status === "crashed").length;
  const gpuCount = gpus.data?.gpus.length ?? 0;
  const busiestGpu = (gpus.data?.gpus ?? [])
    .map((g) => {
      const used = g.total != null && g.free != null ? g.total - g.free : null;
      const pct =
        g.total && used != null ? Math.min(100, Math.round((used / g.total) * 100)) : 0;
      return { gpu: g, pct };
    })
    .sort((a, b) => b.pct - a.pct)[0];

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow="Control plane"
        title="Overview"
        description="Monitor backend health, GPU pressure, and managed llama.cpp endpoints at a glance."
        actions={
          <>
            <Link to="/instances" className="brand-btn-ghost">
              Manage instances
            </Link>
            <Link to="/builds" className="brand-btn-primary">
              Run build
            </Link>
          </>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Backend"
          value={health.data?.status === "ok" ? "Healthy" : health.isError ? "Offline" : "Checking"}
          hint={
            health.data?.llama_server_present
              ? "llama-server binary detected"
              : "binary missing or health unavailable"
          }
          tone={health.data?.status === "ok" ? "success" : health.isError ? "danger" : "default"}
        />
        <MetricCard
          label="Running instances"
          value={`${runningInstances}`}
          hint={`${totalInstances} managed in total`}
          tone={runningInstances > 0 ? "success" : "default"}
        />
        <MetricCard
          label="Crashed instances"
          value={`${crashedInstances}`}
          hint={
            crashedInstances > 0
              ? "Needs review or restart"
              : "No crash-reported workloads"
          }
          tone={crashedInstances > 0 ? "danger" : "default"}
        />
        <MetricCard
          label="Detected GPUs"
          value={`${gpuCount}`}
          hint={
            busiestGpu
              ? `Busiest: #${busiestGpu.gpu.index} at ${busiestGpu.pct}%`
              : "No CUDA devices reported"
          }
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <Panel title="Runtime status" subtitle="Backend service state and core runtime paths.">
          {health.isLoading && <div className="text-sm text-bone-400">Loading health state…</div>}
          {health.isError && (
            <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
              {(health.error as Error).message}. Check Settings and confirm the backend URL and token.
            </div>
          )}
          {health.data && (
            <div className="grid gap-3 md:grid-cols-2">
              <RuntimeRow label="Service status" value={health.data.status} highlight />
              <RuntimeRow
                label="llama-server"
                value={health.data.llama_server_present ? "present" : "missing"}
              />
              <RuntimeRow
                label="Auth"
                value={health.data.auth_required ? "token required" : "open"}
              />
              <RuntimeRow label="Models path" value={health.data.models_dir} mono />
            </div>
          )}
        </Panel>

        <Panel title="Quick actions" subtitle="Most common next steps from the control plane.">
          <div className="grid gap-3">
            <Link
              to="/instances"
              className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 hover:border-lime-300/40 hover:bg-lime-300/5"
            >
              <div className="text-sm font-semibold text-bone-50">Launch or recover an instance</div>
              <p className="mt-1 text-sm text-bone-300">
                Create workloads, restart stopped endpoints, or reclaim orphaned servers after a backend crash.
              </p>
            </Link>
            <Link
              to="/memory"
              className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 hover:border-lime-300/40 hover:bg-lime-300/5"
            >
              <div className="text-sm font-semibold text-bone-50">Validate placement before launch</div>
              <p className="mt-1 text-sm text-bone-300">
                Estimate weights and KV-cache placement before committing a large model to GPU memory.
              </p>
            </Link>
            <Link
              to="/library"
              className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 hover:border-lime-300/40 hover:bg-lime-300/5"
            >
              <div className="text-sm font-semibold text-bone-50">Refresh the model catalog</div>
              <p className="mt-1 text-sm text-bone-300">
                Check what GGUFs are already local and pull new weights from Hugging Face when needed.
              </p>
            </Link>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Panel title="GPU availability" subtitle="Live usage snapshot from the backend memory probe.">
          {gpus.data?.gpus.length === 0 && (
            <div className="text-sm text-bone-400">No CUDA devices detected.</div>
          )}
          <div className="grid gap-3">
            {gpus.data?.gpus.map((g) => {
              const used = g.total != null && g.free != null ? g.total - g.free : null;
              const pct =
                g.total && used != null ? Math.min(100, Math.round((used / g.total) * 100)) : 0;
              return (
                <div key={g.index} className="brand-surface-muted p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div className="text-sm font-semibold text-bone-50">
                        GPU #{g.index}
                      </div>
                      <div className="text-sm text-bone-300">{g.name}</div>
                    </div>
                    <span className="brand-chip">{pct}% used</span>
                  </div>
                  <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-lime-300 via-emerald-400 to-sky-400"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-bone-400">
                    <span>{g.free_h} free</span>
                    <span>{g.total_h} total</span>
                  </div>
                </div>
              );
            })}
          </div>
        </Panel>

        <Panel title="Managed inventory" subtitle="Current workloads and the endpoints they expose.">
          <div className="space-y-3">
            {allInstances.slice(0, 5).map((inst) => (
              <div
                key={inst.id}
                className="rounded-2xl border border-white/10 bg-white/[0.04] p-4"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-bone-50">
                      {inst.name}
                    </div>
                    <div className="mt-1 truncate text-sm text-bone-300" title={inst.config.model_ref}>
                      {inst.config.model_ref || "No model configured"}
                    </div>
                  </div>
                  <StatusBadge status={inst.status} />
                </div>
                <div className="mt-3 flex flex-wrap gap-4 text-xs text-bone-400">
                  <span>{inst.host}:{inst.port}</span>
                  <span>uptime {formatUptime(inst.uptime_s)}</span>
                </div>
              </div>
            ))}
            {allInstances.length === 0 && (
              <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.03] px-4 py-8 text-center text-sm text-bone-400">
                No managed instances yet. Start from the Instances page.
              </div>
            )}
          </div>
        </Panel>
      </div>

      <Panel title="All managed instances" subtitle="Responsive table for quick scanning across status, model, and endpoint.">
        <div className="brand-table-wrap">
          <table className="min-w-[720px] w-full text-sm">
            <thead className="bg-white/[0.03]">
              <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                <th className="px-4 py-3 font-semibold">Name</th>
                <th className="px-4 py-3 font-semibold">Status</th>
                <th className="px-4 py-3 font-semibold">Model</th>
                <th className="px-4 py-3 font-semibold">Endpoint</th>
                <th className="px-4 py-3 font-semibold">Uptime</th>
              </tr>
            </thead>
            <tbody>
              {allInstances.map((inst) => (
                <tr key={inst.id} className="border-t border-white/10 hover:bg-white/[0.03]">
                  <td className="px-4 py-3 font-medium text-bone-50">{inst.name}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={inst.status} />
                  </td>
                  <td className="max-w-xs truncate px-4 py-3 text-bone-300" title={inst.config.model_ref}>
                    {inst.config.model_ref}
                  </td>
                  <td className="px-4 py-3 font-mono text-[12px] text-bone-300">
                    {inst.host}:{inst.port}
                  </td>
                  <td className="px-4 py-3 text-bone-300">{formatUptime(inst.uptime_s)}</td>
                </tr>
              ))}
              {allInstances.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-bone-500">
                    No instances yet. Create one in the Instances page.
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

function RuntimeRow({
  label,
  value,
  highlight,
  mono,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  mono?: boolean;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className="brand-label">{label}</div>
      <div
        className={`mt-2 break-all text-sm ${
          highlight
            ? "font-semibold text-lime-200"
            : mono
              ? "font-mono text-[12px] text-bone-200"
              : "text-bone-200"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
