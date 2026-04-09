import { useQuery } from "@tanstack/react-query";
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

  const totalInstances = instances.data?.instances.length ?? 0;
  const runningInstances =
    instances.data?.instances.filter((i) => i.status === "running").length ?? 0;

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Overview"
        title="Dashboard"
        description="Real-time status of your managed llama.cpp deployments and host resources."
      />

      <div className="grid gap-5 md:grid-cols-3">
        <Panel title="Backend">
          {health.isLoading && (
            <div className="text-sm text-bone-400">Loading…</div>
          )}
          {health.isError && (
            <div className="text-sm text-rose-300">
              {(health.error as Error).message}. Check Settings → token.
            </div>
          )}
          {health.data && (
            <dl className="space-y-2 text-sm">
              <Row label="Status" value={health.data.status} highlight />
              <Row
                label="llama-server"
                value={health.data.llama_server_present ? "present" : "missing"}
              />
              <div className="flex flex-col gap-1 pt-1">
                <dt className="text-[11px] uppercase tracking-wider text-bone-500">
                  models dir
                </dt>
                <dd className="break-all font-mono text-[12px] text-bone-200">
                  {health.data.models_dir}
                </dd>
              </div>
            </dl>
          )}
        </Panel>

        <Panel title="GPUs">
          {gpus.data?.gpus.length === 0 && (
            <div className="text-sm text-bone-400">
              No CUDA devices detected.
            </div>
          )}
          <ul className="space-y-2">
            {gpus.data?.gpus.map((g) => {
              const used =
                g.total != null && g.free != null ? g.total - g.free : null;
              const pct =
                g.total && used != null
                  ? Math.min(100, Math.round((used / g.total) * 100))
                  : 0;
              return (
                <li
                  key={g.index}
                  className="brand-surface-muted px-3 py-2.5 text-sm"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-bone-100">
                      #{g.index} {g.name}
                    </span>
                    <span className="text-xs text-bone-400">{pct}%</span>
                  </div>
                  <div className="mt-1.5 text-[11px] text-bone-400">
                    {g.free_h} free · {g.total_h}
                  </div>
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-lime-300 to-lime-500 shadow-[0_0_10px_rgba(213,255,64,0.45)]"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        </Panel>

        <Panel title="Instances">
          <div className="flex items-end gap-3">
            <div className="font-display text-5xl font-bold leading-none text-bone-50">
              {totalInstances}
            </div>
            <div className="pb-1 text-sm text-bone-400">
              total
            </div>
          </div>
          <div className="mt-3 flex items-center gap-2 text-sm text-bone-300">
            <span className="h-1.5 w-1.5 rounded-full bg-lime-300 shadow-[0_0_10px_rgba(213,255,64,0.8)]" />
            {runningInstances} running
          </div>
          <div className="mt-4 h-px w-full bg-white/5" />
          <div className="mt-3 text-[11px] uppercase tracking-[0.18em] text-bone-500">
            auto-refresh via websocket
          </div>
        </Panel>
      </div>

      <Panel title="Active instances">
        <div className="overflow-hidden rounded-xl border border-white/5">
          <table className="w-full text-sm">
            <thead className="bg-white/[0.03]">
              <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                <th className="px-4 py-2.5 font-semibold">Name</th>
                <th className="px-4 py-2.5 font-semibold">Status</th>
                <th className="px-4 py-2.5 font-semibold">Model</th>
                <th className="px-4 py-2.5 font-semibold">Endpoint</th>
                <th className="px-4 py-2.5 font-semibold">Uptime</th>
              </tr>
            </thead>
            <tbody>
              {instances.data?.instances.map((inst) => (
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
                </tr>
              ))}
              {instances.data?.instances.length === 0 && (
                <tr>
                  <td
                    colSpan={5}
                    className="px-4 py-8 text-center text-bone-500"
                  >
                    No instances yet. Create one in the{" "}
                    <span className="text-lime-300">Instances</span> tab.
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

function Row({
  label,
  value,
  highlight,
}: {
  label: string;
  value: React.ReactNode;
  highlight?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-[11px] uppercase tracking-wider text-bone-500">
        {label}
      </dt>
      <dd
        className={
          highlight
            ? "font-semibold text-lime-300"
            : "max-w-[12rem] text-bone-200"
        }
      >
        {value}
      </dd>
    </div>
  );
}
