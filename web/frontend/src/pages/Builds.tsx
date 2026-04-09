import { FormEvent, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BuildRecord, api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { LogPane } from "@/components/LogPane";
import { StatusBadge } from "@/components/StatusBadge";
import { useManagedWebSocket } from "@/hooks/useManagedWebSocket";

function isActiveBuild(status?: BuildRecord["status"]): boolean {
  return status === "running" || status === "cancelling";
}

export default function Builds() {
  const qc = useQueryClient();
  const [ref, setRef] = useState("latest");
  const [fastMath, setFastMath] = useState(false);
  const [forceMmq, setForceMmq] = useState("auto");
  const [blas, setBlas] = useState("auto");
  const [distributed, setDistributed] = useState(false);
  const [cpuOnly, setCpuOnly] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [formError, setFormError] = useState<string | null>(null);

  const builds = useQuery({ queryKey: ["builds"], queryFn: api.listBuilds });
  const flags = useQuery({
    queryKey: ["supported-flags"],
    queryFn: api.supportedFlags,
  });
  const selectedDetails = useQuery({
    queryKey: ["build", selected],
    queryFn: () => api.getBuild(selected || ""),
    enabled: !!selected,
  });

  const selectedBuild =
    builds.data?.builds.find((b) => b.id === selected) ??
    selectedDetails.data?.build ??
    null;
  const forceMmqChoices = flags.data?.choice_flags["--force-mmq"];
  const blasChoices = flags.data?.choice_flags["--blas"];
  const boolFlags = new Set(flags.data?.bool_flags ?? []);
  const hasFastMath = boolFlags.has("--fast-math");
  const hasDistributed = boolFlags.has("--distributed");
  const hasCpuOnly = boolFlags.has("--cpu-only");

  useEffect(() => {
    if (forceMmqChoices && !forceMmqChoices.includes(forceMmq)) {
      setForceMmq(forceMmqChoices[0] ?? "auto");
    }
  }, [forceMmqChoices, forceMmq]);

  useEffect(() => {
    if (blasChoices && !blasChoices.includes(blas)) {
      setBlas(blasChoices[0] ?? "auto");
    }
  }, [blasChoices, blas]);

  const start = useMutation({
    mutationFn: () =>
      api.startBuild({
        ref,
        now: true,
        fast_math: fastMath,
        force_mmq: forceMmq,
        blas,
        distributed,
        cpu_only: cpuOnly,
      }),
    onSuccess: (data) => {
      setFormError(null);
      setSelected(data.build.id);
      setLines([]);
      qc.invalidateQueries({ queryKey: ["build", data.build.id] });
    },
    onError: (err: Error) => setFormError(err.message),
  });

  const stop = useMutation({
    mutationFn: (id: string) => api.stopBuild(id),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["builds"] });
      qc.invalidateQueries({ queryKey: ["build", data.build.id] });
    },
    onError: (err: Error) => setFormError(err.message),
  });

  useEffect(() => {
    setLines([]);
  }, [selected]);

  useEffect(() => {
    if (!selectedDetails.data || selectedDetails.data.build.id !== selected || lines.length > 0) return;
    setLines(selectedDetails.data.logs);
  }, [selected, selectedDetails.data, lines.length]);

  const socket = useManagedWebSocket({
    path: selected && isActiveBuild(selectedBuild?.status) ? `/api/builds/${selected}/logs?history=0` : null,
    enabled: Boolean(selected && isActiveBuild(selectedBuild?.status)),
    beforeConnect: async (phase) => {
      if (phase !== "reconnect" || !selected) return;
      const response = await api.getBuild(selected);
      qc.setQueryData(["build", selected], response);
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

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h2 className="text-2xl font-semibold">llama.cpp builds</h2>
        <p className="text-sm text-slate-400">
          Trigger and monitor autodevops.py builds. Options are detected live
          from <code>autodevops.py --help</code>, so unsupported flags are
          hidden automatically.
        </p>
      </header>

      <Panel title="New build">
        {flags.isLoading && (
          <div className="text-sm text-slate-400">Probing supported flags…</div>
        )}
        {flags.isError && (
          <div className="text-sm text-rose-400">
            Failed to probe supported flags: {(flags.error as Error).message}
          </div>
        )}
        <form onSubmit={submit} className="grid gap-3 md:grid-cols-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">--ref (tag/branch/commit)</span>
            <input
              value={ref}
              onChange={(e) => setRef(e.target.value)}
              className="rounded border border-slate-700 bg-slate-950 px-2 py-1"
            />
          </label>
          {forceMmqChoices && (
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-slate-400">--force-mmq</span>
              <select
                value={forceMmq}
                onChange={(e) => setForceMmq(e.target.value)}
                className="rounded border border-slate-700 bg-slate-950 px-2 py-1"
              >
                {forceMmqChoices.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          )}
          {blasChoices && (
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-slate-400">--blas</span>
              <select
                value={blas}
                onChange={(e) => setBlas(e.target.value)}
                className="rounded border border-slate-700 bg-slate-950 px-2 py-1"
              >
                {blasChoices.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          )}
          {hasFastMath && (
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={fastMath}
                onChange={(e) => setFastMath(e.target.checked)}
              />
              --fast-math
            </label>
          )}
          {hasDistributed && (
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={distributed}
                onChange={(e) => setDistributed(e.target.checked)}
              />
              --distributed (RPC)
            </label>
          )}
          {hasCpuOnly && (
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={cpuOnly}
                onChange={(e) => setCpuOnly(e.target.checked)}
              />
              --cpu-only
            </label>
          )}
          <div className="md:col-span-3 flex items-center gap-3">
            <button
              type="submit"
              disabled={start.isPending || flags.isLoading}
              className="rounded bg-emerald-500 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-400 disabled:opacity-50"
            >
              {start.isPending ? "Starting…" : "Start build"}
            </button>
            {formError && (
              <span className="text-sm text-rose-400">{formError}</span>
            )}
          </div>
        </form>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="History" className="lg:col-span-1">
          <ul className="divide-y divide-slate-800 text-sm">
            {builds.data?.builds.map((b) => (
              <li
                key={b.id}
                className={`flex cursor-pointer items-center justify-between gap-2 py-2 ${
                  selected === b.id ? "text-sky-300" : ""
                }`}
                onClick={() => setSelected(b.id)}
              >
                <span className="truncate">{b.id}</span>
                <StatusBadge status={b.status} />
              </li>
            ))}
            {builds.data?.builds.length === 0 && (
              <li className="py-4 text-center text-slate-500">No builds yet.</li>
            )}
          </ul>
        </Panel>

        <Panel
          title={selected ? `Build ${selected}` : "Build log"}
          actions={
            selectedBuild ? (
              <>
                <StatusBadge status={selectedBuild.status} />
                {isActiveBuild(selectedBuild.status) && (
                  <button
                    className="rounded bg-amber-600 px-3 py-1 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-60"
                    disabled={selectedBuild.status === "cancelling" || stop.isPending}
                    onClick={() => stop.mutate(selectedBuild.id)}
                  >
                    {selectedBuild.status === "cancelling" ? "Cancelling…" : "Cancel build"}
                  </button>
                )}
              </>
            ) : undefined
          }
          className="lg:col-span-2"
        >
          {selected ? (
            <>
              <div className="mb-2 text-xs text-slate-400">
                {isActiveBuild(selectedBuild?.status)
                  ? `Log stream: ${socket.status}`
                  : "Viewing durable log history"}
              </div>
              <LogPane lines={lines} height="60vh" />
            </>
          ) : (
            <div className="text-sm text-slate-500">
              Start a new build or select one from the history to view its log.
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
