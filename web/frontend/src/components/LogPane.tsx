import { useEffect, useRef, useState } from "react";

interface Props {
  lines: string[];
  autoscroll?: boolean;
  height?: string;
}

export function LogPane({ lines, autoscroll = true, height = "60vh" }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (!autoscroll || paused || !ref.current) return;
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines, autoscroll, paused]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-3 text-[11px] uppercase tracking-wider text-bone-400">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPaused((p) => !p)}
            className="rounded-md border border-white/10 bg-white/5 px-2.5 py-1 font-semibold text-bone-100 hover:border-lime-300/50 hover:text-lime-200"
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <span className="text-bone-400">autoscroll</span>
        </div>
        <span>{lines.length} lines</span>
      </div>
      <div
        ref={ref}
        className="log-pane overflow-auto rounded-xl border border-white/5 bg-ink-700/80 p-4 text-bone-200 shadow-panel"
        style={{ height }}
      >
        {lines.length === 0 ? (
          <div className="text-bone-500">(no output yet)</div>
        ) : (
          lines.map((l, i) => <div key={i}>{l}</div>)
        )}
      </div>
    </div>
  );
}
