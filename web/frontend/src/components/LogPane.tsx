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
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <button
          onClick={() => setPaused((p) => !p)}
          className="rounded bg-slate-800 px-2 py-1 text-slate-200 hover:bg-slate-700"
        >
          {paused ? "Resume" : "Pause"} autoscroll
        </button>
        <span>{lines.length} lines</span>
      </div>
      <div
        ref={ref}
        className="log-pane overflow-auto rounded border border-slate-800 bg-black/60 p-3"
        style={{ height }}
      >
        {lines.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
    </div>
  );
}
