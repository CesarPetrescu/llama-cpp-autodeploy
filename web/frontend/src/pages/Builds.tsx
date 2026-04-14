import { FormEvent, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BuildRecord, SupportedFlags, api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { LogPane } from "@/components/LogPane";
import { StatusBadge } from "@/components/StatusBadge";
import { useManagedWebSocket } from "@/hooks/useManagedWebSocket";
import { PageHeader } from "@/components/PageHeader";

type BuildFormState = {
  now: boolean;
  ref: string;
  fastMath: boolean;
  forceMmq: string;
  blas: string;
  distributed: boolean;
  cpuOnly: boolean;
};

const DEFAULT_FORM: BuildFormState = {
  now: true,
  ref: "latest",
  fastMath: false,
  forceMmq: "auto",
  blas: "auto",
  distributed: false,
  cpuOnly: false,
};

const OPTION_NOTES: Record<string, string> = {
  "--now": "Off means autodevops checks freshness and schedules the build for 02:00 when a newer ref exists.",
  "--ref": "Use a tag, branch, commit SHA, or latest.",
  "--fast-math": "Enables NVCC fast math for CUDA kernels.",
  "--force-mmq": "MMQ controls the matrix-multiply kernel path used by CUDA builds.",
  "--blas": "BLAS affects the CPU path and host-side math implementation.",
  "--distributed": "Builds the GGML RPC backend for distributed inference workers.",
  "--cpu-only": "Skips NVIDIA driver checks when you only care about CPU execution.",
};

const CHOICE_NOTES: Record<string, Record<string, string>> = {
  "--force-mmq": {
    auto: "use script default",
    on: "force MMQ on",
    off: "force MMQ off",
  },
  "--blas": {
    auto: "use script default",
    openblas: "force OpenBLAS",
    mkl: "force MKL",
    off: "disable BLAS",
  },
};

const CONTROL_NOTES = [
  {
    label: "Start build",
    detail: "Launches autodevops.py with the exact command preview shown in the form.",
  },
  {
    label: "History row",
    detail: "Loads saved build metadata, command, and durable logs for that run.",
  },
  {
    label: "Cancel build",
    detail: "Sends a stop request to the active build process group.",
  },
  {
    label: "Pause / Resume",
    detail: "Only pauses log autoscroll. The underlying build keeps running.",
  },
];

const FALLBACK_USAGE =
  "usage: autodevops.py [--now] [--ref REF] [--fast-math] [--force-mmq {auto,on,off}] [--blas {auto,openblas,mkl,off}] [--distributed] [--cpu-only]";

const FALLBACK_SUMMARY = "Automated llama.cpp build (CUDA + BLAS).";

const FALLBACK_OPTIONS: SupportedFlags["options"] = [
  {
    flag: "--now",
    syntax: "--now",
    description: "build immediately",
    choices: [],
    metavar: null,
    kind: "bool",
  },
  {
    flag: "--ref",
    syntax: "--ref REF",
    description: "git tag/branch/commit, or 'latest'",
    choices: [],
    metavar: "REF",
    kind: "value",
  },
  {
    flag: "--fast-math",
    syntax: "--fast-math",
    description: "pass --use_fast_math to NVCC",
    choices: [],
    metavar: null,
    kind: "bool",
  },
  {
    flag: "--force-mmq",
    syntax: "--force-mmq {auto,on,off}",
    description: "toggle MMQ CUDA kernels",
    choices: ["auto", "on", "off"],
    metavar: null,
    kind: "choice",
  },
  {
    flag: "--blas",
    syntax: "--blas {auto,openblas,mkl,off}",
    description: "choose BLAS for CPU path",
    choices: ["auto", "openblas", "mkl", "off"],
    metavar: null,
    kind: "choice",
  },
  {
    flag: "--distributed",
    syntax: "--distributed",
    description: "enable GGML RPC backend for distributed inference",
    choices: [],
    metavar: null,
    kind: "bool",
  },
  {
    flag: "--cpu-only",
    syntax: "--cpu-only",
    description: "skip NVIDIA driver checks when GPU execution is not needed",
    choices: [],
    metavar: null,
    kind: "bool",
  },
];

function isActiveBuild(status?: BuildRecord["status"]): boolean {
  return status === "running" || status === "cancelling";
}

function formatRelativeTime(timestamp: number | null): string {
  if (!timestamp) return "—";
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatDuration(build: BuildRecord | null): string {
  if (!build?.started_at) return "—";
  const end = build.finished_at ?? Date.now() / 1000;
  const seconds = Math.max(0, Math.floor(end - build.started_at));
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

function formatConfigValue(flag: string, config: Record<string, unknown>): string {
  switch (flag) {
    case "--now":
      return config.now === false ? "off" : "on";
    case "--ref":
      return String(config.ref ?? "latest");
    case "--fast-math":
      return config.fast_math ? "on" : "off";
    case "--force-mmq":
      return String(config.force_mmq ?? "auto");
    case "--blas":
      return String(config.blas ?? "auto");
    case "--distributed":
      return config.distributed ? "on" : "off";
    case "--cpu-only":
      return config.cpu_only ? "on" : "off";
    default:
      return "—";
  }
}

function buildCommandPreview(form: BuildFormState): string {
  const parts = ["python", "autodevops.py"];
  if (form.now) parts.push("--now");
  if (form.ref.trim()) parts.push("--ref", form.ref.trim());
  parts.push("--force-mmq", form.forceMmq);
  parts.push("--blas", form.blas);
  if (form.fastMath) parts.push("--fast-math");
  if (form.distributed) parts.push("--distributed");
  if (form.cpuOnly) parts.push("--cpu-only");
  return parts.join(" ");
}

function findOptionSpec(flags: SupportedFlags | undefined, flag: string) {
  const options = flags?.options?.length ? flags.options : FALLBACK_OPTIONS;
  return options.find((option) => option.flag === flag);
}

export default function Builds() {
  const qc = useQueryClient();
  const [form, setForm] = useState<BuildFormState>(DEFAULT_FORM);
  const [selected, setSelected] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [formError, setFormError] = useState<string | null>(null);

  const builds = useQuery({
    queryKey: ["builds"],
    queryFn: api.listBuilds,
    refetchInterval: 4000,
  });
  const flags = useQuery({
    queryKey: ["supported-flags"],
    queryFn: api.supportedFlags,
  });
  const selectedDetails = useQuery({
    queryKey: ["build", selected],
    queryFn: () => api.getBuild(selected || ""),
    enabled: !!selected,
  });

  const buildList = [...(builds.data?.builds ?? [])].sort(
    (a, b) => (b.started_at ?? 0) - (a.started_at ?? 0),
  );
  const latestBuild = buildList[0] ?? null;
  const selectedBuild =
    buildList.find((build) => build.id === selected) ??
    selectedDetails.data?.build ??
    null;
  const runningCount = buildList.filter((build) => build.status === "running").length;
  const failedCount = buildList.filter((build) => build.status === "failure").length;
  const completedCount = buildList.filter((build) => build.status === "success").length;

  const forceMmqChoices = flags.data?.choice_flags["--force-mmq"] ?? ["auto"];
  const blasChoices = flags.data?.choice_flags["--blas"] ?? ["auto"];
  const boolFlags = new Set(flags.data?.bool_flags ?? []);
  const optionSpecs = flags.data?.options?.length ? flags.data.options : FALLBACK_OPTIONS;

  const start = useMutation({
    mutationFn: () =>
      api.startBuild({
        ref: form.ref,
        now: form.now,
        fast_math: form.fastMath,
        force_mmq: form.forceMmq,
        blas: form.blas,
        distributed: form.distributed,
        cpu_only: form.cpuOnly,
      }),
    onSuccess: (data) => {
      setFormError(null);
      setSelected(data.build.id);
      setLines([]);
      qc.invalidateQueries({ queryKey: ["builds"] });
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
    if (!selected && buildList.length > 0) {
      setSelected(buildList[0].id);
      return;
    }
    if (selected && buildList.length > 0 && !buildList.some((build) => build.id === selected)) {
      setSelected(buildList[0].id);
    }
  }, [buildList, selected]);

  useEffect(() => {
    if (!forceMmqChoices.includes(form.forceMmq)) {
      setForm((current) => ({ ...current, forceMmq: forceMmqChoices[0] ?? "auto" }));
    }
  }, [forceMmqChoices, form.forceMmq]);

  useEffect(() => {
    if (!blasChoices.includes(form.blas)) {
      setForm((current) => ({ ...current, blas: blasChoices[0] ?? "auto" }));
    }
  }, [blasChoices, form.blas]);

  useEffect(() => {
    setLines([]);
  }, [selected]);

  useEffect(() => {
    if (!selectedDetails.data || selectedDetails.data.build.id !== selected || lines.length > 0) return;
    setLines(selectedDetails.data.logs);
  }, [selected, selectedDetails.data, lines.length]);

  const socket = useManagedWebSocket({
    path:
      selected && isActiveBuild(selectedBuild?.status)
        ? `/api/builds/${selected}/logs?history=0`
        : null,
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

  function update<K extends keyof BuildFormState>(key: K, value: BuildFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  const refSpec = findOptionSpec(flags.data, "--ref");
  const nowSpec = findOptionSpec(flags.data, "--now");
  const fastMathSpec = findOptionSpec(flags.data, "--fast-math");
  const forceMmqSpec = findOptionSpec(flags.data, "--force-mmq");
  const blasSpec = findOptionSpec(flags.data, "--blas");
  const distributedSpec = findOptionSpec(flags.data, "--distributed");
  const cpuOnlySpec = findOptionSpec(flags.data, "--cpu-only");

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow="Toolchain"
        title="Builds"
        description={flags.data?.summary || FALLBACK_SUMMARY}
        actions={
          (flags.data?.usage || FALLBACK_USAGE) ? (
            <div className="brand-code-block text-left sm:max-w-[40rem]">
              {flags.data?.usage || FALLBACK_USAGE}
            </div>
          ) : undefined
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <BuildStat
          label="History"
          value={`${buildList.length}`}
          meta={[
            `${completedCount} success`,
            `${failedCount} failure`,
          ]}
        />
        <BuildStat
          label="Running"
          value={`${runningCount}`}
          meta={[
            latestBuild ? `latest ${latestBuild.id}` : "no runs yet",
            latestBuild ? formatRelativeTime(latestBuild.started_at) : "—",
          ]}
          tone={runningCount > 0 ? "warning" : "default"}
        />
        <BuildStat
          label="Target ref"
          value={String(latestBuild?.config.ref ?? form.ref)}
          meta={[
            `--now ${form.now ? "on" : "off"}`,
            `--blas ${form.blas}`,
          ]}
        />
        <BuildStat
          label="Latest status"
          value={latestBuild?.status ?? "idle"}
          meta={[
            latestBuild?.pid ? `pid ${latestBuild.pid}` : "pid —",
            latestBuild ? `duration ${formatDuration(latestBuild)}` : "duration —",
          ]}
          tone={latestBuild?.status === "failure" ? "danger" : latestBuild?.status === "running" ? "warning" : "default"}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
        <Panel
          title="Build configuration"
          actions={
            <div className="brand-code-block text-left sm:max-w-[36rem]">
              {buildCommandPreview(form)}
            </div>
          }
        >
          {flags.isLoading && (
            <div className="text-sm text-bone-400">Probing autodevops.py --help…</div>
          )}
          {flags.isError && (
            <div className="text-sm text-rose-300">
              Failed to probe supported flags: {(flags.error as Error).message}
            </div>
          )}

          <form onSubmit={submit} className="space-y-4">
            <div className="grid gap-4 lg:grid-cols-3">
              <FieldCard
                syntax={refSpec?.syntax ?? "--ref REF"}
                description={refSpec?.description ?? ""}
                note={OPTION_NOTES["--ref"]}
              >
                <input
                  value={form.ref}
                  onChange={(e) => update("ref", e.target.value)}
                  className="brand-input"
                />
              </FieldCard>

              <FieldCard
                syntax={forceMmqSpec?.syntax ?? "--force-mmq {auto,on,off}"}
                description={forceMmqSpec?.description ?? ""}
                note={formatChoiceNote("--force-mmq", form.forceMmq)}
              >
                <select
                  value={form.forceMmq}
                  onChange={(e) => update("forceMmq", e.target.value)}
                  className="brand-input"
                >
                  {forceMmqChoices.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </FieldCard>

              <FieldCard
                syntax={blasSpec?.syntax ?? "--blas {auto,openblas,mkl,off}"}
                description={blasSpec?.description ?? ""}
                note={formatChoiceNote("--blas", form.blas)}
              >
                <select
                  value={form.blas}
                  onChange={(e) => update("blas", e.target.value)}
                  className="brand-input"
                >
                  {blasChoices.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </FieldCard>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {boolFlags.has("--now") && (
                <ToggleCard
                  syntax={nowSpec?.syntax ?? "--now"}
                  description={nowSpec?.description ?? ""}
                  note={OPTION_NOTES["--now"]}
                  checked={form.now}
                  onChange={(checked) => update("now", checked)}
                />
              )}
              {boolFlags.has("--fast-math") && (
                <ToggleCard
                  syntax={fastMathSpec?.syntax ?? "--fast-math"}
                  description={fastMathSpec?.description ?? ""}
                  note={OPTION_NOTES["--fast-math"]}
                  checked={form.fastMath}
                  onChange={(checked) => update("fastMath", checked)}
                />
              )}
              {boolFlags.has("--distributed") && (
                <ToggleCard
                  syntax={distributedSpec?.syntax ?? "--distributed"}
                  description={distributedSpec?.description ?? ""}
                  note={OPTION_NOTES["--distributed"]}
                  checked={form.distributed}
                  onChange={(checked) => update("distributed", checked)}
                />
              )}
              {boolFlags.has("--cpu-only") && (
                <ToggleCard
                  syntax={cpuOnlySpec?.syntax ?? "--cpu-only"}
                  description={cpuOnlySpec?.description ?? ""}
                  note={OPTION_NOTES["--cpu-only"]}
                  checked={form.cpuOnly}
                  onChange={(checked) => update("cpuOnly", checked)}
                />
              )}
            </div>

            <div className="border border-white/10 bg-ink-400/72 p-4">
              <div className="brand-label">Command preview</div>
              <div className="mt-2 font-mono text-[12px] text-bone-100 break-all">
                {buildCommandPreview(form)}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={start.isPending || flags.isLoading}
                className="brand-btn-primary"
              >
                {start.isPending ? "Starting…" : "Start build"}
              </button>
              <span className="text-xs text-bone-400">
                {form.now
                  ? "Immediate run."
                  : "Only builds immediately at 02:00, otherwise schedules when a newer ref exists."}
              </span>
              {formError && <span className="text-sm text-rose-300">{formError}</span>}
            </div>
          </form>
        </Panel>

        <Panel title="Controls and options">
          <div className="space-y-5">
            <div>
              <div className="brand-label">Buttons</div>
              <div className="mt-3 space-y-2">
                {CONTROL_NOTES.map((item) => (
                  <div key={item.label} className="border border-white/10 bg-ink-300/70 px-3 py-3">
                    <div className="text-sm font-semibold text-bone-50">{item.label}</div>
                    <div className="mt-1 text-xs leading-5 text-bone-300">{item.detail}</div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="brand-label">autodevops options</div>
              <div className="mt-3 space-y-2">
                {optionSpecs.map((option) => (
                  <div key={option.flag} className="border border-white/10 bg-ink-300/70 px-3 py-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1 font-mono text-[12px] text-bone-50 break-all">{option.syntax}</div>
                      <span className="brand-chip shrink-0">{option.kind}</span>
                    </div>
                    <div className="mt-2 text-xs leading-5 text-bone-300">
                      {option.description}
                    </div>
                    <div className="mt-2 text-[11px] leading-5 text-bone-400">
                      {OPTION_NOTES[option.flag] ?? "No extra UI note."}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="History">
          <div className="brand-table-wrap">
            <table className="min-w-[680px] w-full text-sm">
              <thead className="bg-ink-400/78">
                <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                  <th className="px-4 py-3 font-semibold">Build</th>
                  <th className="px-4 py-3 font-semibold">Ref</th>
                  <th className="px-4 py-3 font-semibold">Status</th>
                  <th className="px-4 py-3 font-semibold">PID</th>
                  <th className="px-4 py-3 font-semibold">Started</th>
                </tr>
              </thead>
              <tbody>
                {buildList.map((build) => (
                  <tr
                    key={build.id}
                    className={`border-t border-white/10 ${selected === build.id ? "bg-ink-200/55" : "hover:bg-ink-300/50"}`}
                  >
                    <td className="px-4 py-3">
                      <button
                        type="button"
                        onClick={() => setSelected(build.id)}
                        className="font-mono text-[12px] text-bone-50 hover:text-lime-200"
                      >
                        {build.id}
                      </button>
                    </td>
                    <td className="max-w-[12rem] truncate px-4 py-3 text-bone-300" title={String(build.config.ref ?? "latest")}>
                      {String(build.config.ref ?? "latest")}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={build.status} />
                    </td>
                    <td className="px-4 py-3 font-mono text-[12px] text-bone-300">
                      {build.pid ?? "—"}
                    </td>
                    <td className="px-4 py-3 text-bone-300">
                      {formatRelativeTime(build.started_at)}
                    </td>
                  </tr>
                ))}
                {buildList.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-10 text-center text-bone-500">
                      No builds yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel
          title={selectedBuild ? `Build ${selectedBuild.id}` : "Build details"}
          actions={
            selectedBuild ? (
              <>
                <StatusBadge status={selectedBuild.status} />
                {isActiveBuild(selectedBuild.status) && (
                  <button
                    className="brand-btn-warning px-3 py-1.5 text-xs"
                    disabled={selectedBuild.status === "cancelling" || stop.isPending}
                    onClick={() => stop.mutate(selectedBuild.id)}
                  >
                    {selectedBuild.status === "cancelling" ? "Cancelling…" : "Cancel build"}
                  </button>
                )}
              </>
            ) : undefined
          }
        >
          {selectedBuild ? (
            <div className="space-y-4">
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <DetailBox label="Started" value={formatTimestamp(selectedBuild.started_at)} />
                <DetailBox label="Finished" value={formatTimestamp(selectedBuild.finished_at)} />
                <DetailBox label="Duration" value={formatDuration(selectedBuild)} />
                <DetailBox label="Exit code" value={selectedBuild.exit_code == null ? "—" : String(selectedBuild.exit_code)} />
                <DetailBox label="PID" value={selectedBuild.pid == null ? "—" : String(selectedBuild.pid)} />
                <DetailBox label="PGID" value={selectedBuild.pgid == null ? "—" : String(selectedBuild.pgid)} />
                <DetailBox label="Log file" value={selectedBuild.log_file ?? "—"} mono />
                <DetailBox label="Alive" value={selectedBuild.alive ? "yes" : "no"} />
              </div>

              <div className="border border-white/10 bg-ink-400/72 p-4">
                <div className="brand-label">Executed command</div>
                <div className="mt-2 font-mono text-[12px] text-bone-100 break-all">
                  {selectedBuild.cmdline?.join(" ") ||
                    buildCommandPreview({
                      now: Boolean(selectedBuild.config.now ?? true),
                      ref: String(selectedBuild.config.ref ?? "latest"),
                      fastMath: Boolean(selectedBuild.config.fast_math),
                      forceMmq: String(selectedBuild.config.force_mmq ?? "auto"),
                      blas: String(selectedBuild.config.blas ?? "auto"),
                      distributed: Boolean(selectedBuild.config.distributed),
                      cpuOnly: Boolean(selectedBuild.config.cpu_only),
                    })}
                </div>
              </div>

              <div className="brand-table-wrap">
                <table className="min-w-[560px] w-full text-sm">
                  <thead className="bg-ink-400/78">
                    <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                      <th className="px-4 py-3 font-semibold">Option</th>
                      <th className="px-4 py-3 font-semibold">Value</th>
                      <th className="px-4 py-3 font-semibold">Meaning</th>
                    </tr>
                  </thead>
                  <tbody>
                    {optionSpecs.map((option) => (
                      <tr key={option.flag} className="border-t border-white/10">
                        <td className="px-4 py-3 font-mono text-[12px] text-bone-100">
                          {option.flag}
                        </td>
                        <td className="px-4 py-3 text-bone-300">
                          {formatConfigValue(option.flag, selectedBuild.config)}
                        </td>
                        <td className="px-4 py-3 text-bone-300">
                          {option.description}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="border border-white/10 bg-ink-400/72 px-3 py-2 text-[11px] uppercase tracking-wider text-bone-400">
                {isActiveBuild(selectedBuild.status)
                  ? `Log stream · ${socket.status}`
                  : "Viewing durable log history"}
              </div>
              <LogPane lines={lines} height="56vh" />
            </div>
          ) : (
            <div className="text-sm text-bone-500">
              Select a build from history to inspect its command, options, and log.
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function formatChoiceNote(flag: string, value: string): string {
  const note = CHOICE_NOTES[flag]?.[value];
  return note ? `${OPTION_NOTES[flag]} Current: ${note}.` : OPTION_NOTES[flag] ?? "";
}

function FieldCard({
  syntax,
  description,
  note,
  children,
}: {
  syntax: string;
  description: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-w-0 border border-white/10 bg-ink-300/70 p-4">
      <div className="font-mono text-[12px] text-bone-50 break-all">{syntax}</div>
      <div className="mt-1 text-xs leading-5 text-bone-300">{description}</div>
      <div className="mt-2">{children}</div>
      <div className="mt-2 text-[11px] leading-5 text-bone-400">{note}</div>
    </div>
  );
}

function ToggleCard({
  syntax,
  description,
  note,
  checked,
  onChange,
}: {
  syntax: string;
  description: string;
  note: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="min-w-0 border border-white/10 bg-ink-300/70 p-4 text-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[12px] text-bone-50 break-all">{syntax}</div>
          <div className="mt-1 text-xs leading-5 text-bone-300">{description}</div>
        </div>
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-1 h-4 w-4 border-white/20 bg-ink-400 text-lime-300 accent-lime-300"
        />
      </div>
      <div className="mt-2 text-[11px] leading-5 text-bone-400">{note}</div>
    </label>
  );
}

function DetailBox({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="border border-white/10 bg-ink-300/70 p-3">
      <div className="brand-label">{label}</div>
      <div className={`mt-2 break-all text-sm ${mono ? "font-mono text-[12px] text-bone-100" : "text-bone-100"}`}>
        {value}
      </div>
    </div>
  );
}

function BuildStat({
  label,
  value,
  meta,
  tone = "default",
}: {
  label: string;
  value: string;
  meta: string[];
  tone?: "default" | "warning" | "danger";
}) {
  const valueClass =
    tone === "warning"
      ? "text-amber-200"
      : tone === "danger"
        ? "text-rose-200"
        : "text-bone-50";
  return (
    <div className="brand-stat">
      <div className="brand-label">{label}</div>
      <div className={`mt-3 text-4xl font-bold tracking-tight ${valueClass}`}>
        {value}
      </div>
      <div className="mt-4 space-y-1 text-xs text-bone-400">
        {meta.map((item) => (
          <div key={item}>{item}</div>
        ))}
      </div>
    </div>
  );
}
