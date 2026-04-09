import { FormEvent, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BuildRecord, api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { LogPane } from "@/components/LogPane";
import { StatusBadge } from "@/components/StatusBadge";
import { useManagedWebSocket } from "@/hooks/useManagedWebSocket";
import { PageHeader } from "@/components/PageHeader";

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
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Toolchain"
        title="llama.cpp builds"
        description={
          <>
            Trigger and monitor <code className="text-lime-300">autodevops.py</code>{" "}
            builds. Options are detected live from{" "}
            <code className="text-lime-300">autodevops.py --help</code>, so
            unsupported flags are hidden automatically.
          </>
        }
      />

      <Panel title="New build">
        {flags.isLoading && (
          <div className="text-sm text-bone-400">Probing supported flags…</div>
        )}
        {flags.isError && (
          <div className="text-sm text-rose-300">
            Failed to probe supported flags: {(flags.error as Error).message}
          </div>
        )}
        <form onSubmit={submit} className="grid gap-4 md:grid-cols-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="brand-label">--ref (tag/branch/commit)</span>
            <input
              value={ref}
              onChange={(e) => setRef(e.target.value)}
              className="brand-input"
            />
          </label>
          {forceMmqChoices && (
            <label className="flex flex-col gap-1 text-sm">
              <span className="brand-label">--force-mmq</span>
              <select
                value={forceMmq}
                onChange={(e) => setForceMmq(e.target.value)}
                className="brand-input"
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
              <span className="brand-label">--blas</span>
              <select
                value={blas}
                onChange={(e) => setBlas(e.target.value)}
                className="brand-input"
              >
                {blasChoices.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="md:col-span-3 flex flex-wrap gap-x-6 gap-y-2 border-t border-white/5 pt-4">
            {hasFastMath && (
              <label className="flex items-center gap-2 text-sm text-bone-200">
                <input
                  type="checkbox"
                  checked={fastMath}
                  onChange={(e) => setFastMath(e.target.checked)}
                  className="accent-lime-300"
                />
                --fast-math
              </label>
            )}
            {hasDistributed && (
              <label className="flex items-center gap-2 text-sm text-bone-200">
                <input
                  type="checkbox"
                  checked={distributed}
                  onChange={(e) => setDistributed(e.target.checked)}
                  className="accent-lime-300"
                />
                --distributed (RPC)
              </label>
            )}
            {hasCpuOnly && (
              <label className="flex items-center gap-2 text-sm text-bone-200">
                <input
                  type="checkbox"
                  checked={cpuOnly}
                  onChange={(e) => setCpuOnly(e.target.checked)}
                  className="accent-lime-300"
                />
                --cpu-only
              </label>
            )}
          </div>
          <div className="md:col-span-3 flex items-center gap-3">
            <button
              type="submit"
              disabled={start.isPending || flags.isLoading}
              className="brand-btn-primary"
            >
              {start.isPending ? "Starting…" : "Start build"}
            </button>
            {formError && (
              <span className="text-sm text-rose-300">{formError}</span>
            )}
          </div>
        </form>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-3">
        <Panel title="History" className="lg:col-span-1">
          <ul className="flex flex-col gap-2">
            {builds.data?.builds.map((b) => (
              <li
                key={b.id}
                className={`flex cursor-pointer items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm transition ${
                  selected === b.id
                    ? "border-lime-300/40 bg-lime-300/5 text-lime-200"
                    : "border-white/5 bg-white/[0.02] text-bone-200 hover:border-white/10"
                }`}
                onClick={() => setSelected(b.id)}
              >
                <span className="truncate font-mono text-[12px]">{b.id}</span>
                <StatusBadge status={b.status} />
              </li>
            ))}
            {builds.data?.builds.length === 0 && (
              <li className="py-6 text-center text-sm text-bone-500">
                No builds yet.
              </li>
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
                    className="brand-btn-warning px-3 py-1.5 text-xs"
                    disabled={
                      selectedBuild.status === "cancelling" || stop.isPending
                    }
                    onClick={() => stop.mutate(selectedBuild.id)}
                  >
                    {selectedBuild.status === "cancelling"
                      ? "Cancelling…"
                      : "Cancel build"}
                  </button>
                )}
              </>
            ) : undefined
          }
          className="lg:col-span-2"
        >
          {selected ? (
            <>
              <div className="mb-2 text-[11px] uppercase tracking-wider text-bone-500">
                {isActiveBuild(selectedBuild?.status)
                  ? `Log stream · ${socket.status}`
                  : "Viewing durable log history"}
              </div>
              <LogPane lines={lines} height="60vh" />
            </>
          ) : (
            <div className="text-sm text-bone-500">
              Start a new build or select one from the history to view its log.
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
