const COLORS: Record<string, string> = {
  running:
    "bg-lime-300/15 text-lime-200 ring-lime-300/40 shadow-[0_0_18px_-8px_rgba(213,255,64,0.8)]",
  stopping: "bg-amber-400/15 text-amber-200 ring-amber-400/40",
  stopped: "bg-bone-500/20 text-bone-200 ring-bone-500/30",
  crashed: "bg-rose-500/15 text-rose-200 ring-rose-500/40",
  cancelling: "bg-amber-400/15 text-amber-200 ring-amber-400/40",
  cancelled: "bg-bone-500/20 text-bone-200 ring-bone-500/30",
  starting: "bg-amber-400/15 text-amber-200 ring-amber-400/40",
  success:
    "bg-lime-300/15 text-lime-200 ring-lime-300/40 shadow-[0_0_18px_-8px_rgba(213,255,64,0.8)]",
  failure: "bg-rose-500/15 text-rose-200 ring-rose-500/40",
  pending: "bg-bone-500/20 text-bone-200 ring-bone-500/30",
};

export function StatusBadge({ status }: { status: string }) {
  const klass = COLORS[status] || "bg-white/5 text-bone-200 ring-white/10";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider ring-1 ${klass}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          status === "running" || status === "success"
            ? "bg-lime-300"
            : status === "crashed" || status === "failure"
              ? "bg-rose-400"
              : status === "stopping" ||
                  status === "cancelling" ||
                  status === "starting"
                ? "bg-amber-300"
                : "bg-bone-400"
        }`}
      />
      {status}
    </span>
  );
}
