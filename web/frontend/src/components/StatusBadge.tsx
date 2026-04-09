const COLORS: Record<string, string> = {
  running: "bg-emerald-500/20 text-emerald-300 ring-emerald-500/40",
  stopping: "bg-amber-500/20 text-amber-300 ring-amber-500/40",
  stopped: "bg-slate-500/20 text-slate-300 ring-slate-500/30",
  crashed: "bg-rose-500/20 text-rose-300 ring-rose-500/40",
  cancelling: "bg-amber-500/20 text-amber-300 ring-amber-500/40",
  cancelled: "bg-slate-500/20 text-slate-300 ring-slate-500/30",
  starting: "bg-amber-500/20 text-amber-300 ring-amber-500/40",
  success: "bg-emerald-500/20 text-emerald-300 ring-emerald-500/40",
  failure: "bg-rose-500/20 text-rose-300 ring-rose-500/40",
  pending: "bg-slate-500/20 text-slate-300 ring-slate-500/30",
};

export function StatusBadge({ status }: { status: string }) {
  const klass = COLORS[status] || "bg-slate-700 text-slate-200";
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs ring-1 ${klass}`}
    >
      {status}
    </span>
  );
}
