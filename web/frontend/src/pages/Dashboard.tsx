import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
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

export default function Dashboard() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  // Instances are pushed live via /api/instances/events (see useLiveFeeds).
  const instances = useQuery({
    queryKey: ["instances"],
    queryFn: api.listInstances,
  });
  // GPUs are host state so we still poll; cheap request.
  const gpus = useQuery({
    queryKey: ["gpus"],
    queryFn: api.listGpus,
    refetchInterval: 4000,
  });

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h2 className="text-2xl font-semibold">Dashboard</h2>
        <p className="text-sm text-slate-400">
          Overview of managed llama-server instances and host resources.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-3">
        <Panel title="Backend">
          {health.isLoading && <div className="text-sm">Loading…</div>}
          {health.isError && (
            <div className="text-sm text-rose-400">
              {(health.error as Error).message}. Check Settings → token.
            </div>
          )}
          {health.data && (
            <dl className="space-y-1 text-sm">
              <div className="flex justify-between">
                <dt className="text-slate-400">Status</dt>
                <dd>{health.data.status}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">llama-server</dt>
                <dd>{health.data.llama_server_present ? "present" : "missing"}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">models dir</dt>
                <dd className="truncate max-w-[12rem]" title={health.data.models_dir}>
                  {health.data.models_dir}
                </dd>
              </div>
            </dl>
          )}
        </Panel>

        <Panel title="GPUs">
          {gpus.data?.gpus.length === 0 && (
            <div className="text-sm text-slate-400">No CUDA devices detected.</div>
          )}
          <ul className="space-y-2 text-sm">
            {gpus.data?.gpus.map((g) => (
              <li key={g.index} className="rounded bg-slate-800/60 px-3 py-2">
                <div className="font-medium">#{g.index} {g.name}</div>
                <div className="text-xs text-slate-400">
                  free {g.free_h} / {g.total_h}
                </div>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Instances">
          <div className="text-3xl font-bold">
            {instances.data?.instances.length ?? 0}
          </div>
          <div className="text-sm text-slate-400">
            {instances.data?.instances.filter((i) => i.status === "running")
              .length ?? 0}{" "}
            running
          </div>
        </Panel>
      </div>

      <Panel title="Active instances">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase text-slate-500">
              <th className="py-1">Name</th>
              <th>Status</th>
              <th>Model</th>
              <th>Host:Port</th>
              <th>Uptime</th>
            </tr>
          </thead>
          <tbody>
            {instances.data?.instances.map((inst) => (
              <tr key={inst.id} className="border-t border-slate-800">
                <td className="py-2">{inst.name}</td>
                <td><StatusBadge status={inst.status} /></td>
                <td className="max-w-xs truncate" title={inst.config.model_ref}>
                  {inst.config.model_ref}
                </td>
                <td>
                  {inst.host}:{inst.port}
                </td>
                <td>{formatUptime(inst.uptime_s)}</td>
              </tr>
            ))}
            {instances.data?.instances.length === 0 && (
              <tr>
                <td colSpan={5} className="py-4 text-center text-slate-500">
                  No instances yet. Create one in the Instances tab.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
