import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { ModelSelect } from "@/components/ModelSelect";

export default function Memory() {
  const gpus = useQuery({
    queryKey: ["gpus"],
    queryFn: api.listGpus,
    refetchInterval: 2000,
  });

  const [modelRef, setModelRef] = useState("");
  const [ctxSize, setCtxSize] = useState("4096");
  const [nGpuLayers, setNGpuLayers] = useState("999");
  const [tensorSplit, setTensorSplit] = useState("");

  const local = useQuery({ queryKey: ["local-models"], queryFn: api.listLocal });
  const modelsDir = local.data?.models_dir;

  const plan = useMutation({
    mutationFn: () =>
      api.planMemory({
        model_source: "local",
        model_ref: modelRef,
        selected_local_model: modelRef,
        models_dir: modelsDir || undefined,
        ctx_size: ctxSize,
        n_gpu_layers: nGpuLayers,
        tensor_split: tensorSplit,
      }),
  });

  // Re-run the plan whenever the inputs change so the preview stays live.
  useEffect(() => {
    if (!modelRef) return;
    const t = setTimeout(() => plan.mutate(), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelRef, ctxSize, nGpuLayers, tensorSplit]);

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h2 className="text-2xl font-semibold">Memory planning</h2>
        <p className="text-sm text-slate-400">
          Live GPU probe and weight/KV split preview.
        </p>
      </header>

      <Panel title="GPUs">
        <div className="grid gap-3 md:grid-cols-3">
          {gpus.data?.gpus.map((g) => {
            const used = g.total && g.free != null ? g.total - g.free : null;
            const pct = g.total && used != null ? Math.round((used / g.total) * 100) : 0;
            return (
              <div key={g.index} className="rounded border border-slate-800 bg-slate-950/40 p-3">
                <div className="font-medium">#{g.index} {g.name}</div>
                <div className="mt-1 text-xs text-slate-400">
                  {g.free_h} free / {g.total_h} total
                </div>
                <div className="mt-2 h-2 w-full overflow-hidden rounded bg-slate-800">
                  <div
                    className="h-full bg-sky-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
          {gpus.data?.gpus.length === 0 && (
            <div className="text-sm text-slate-400">No CUDA devices detected.</div>
          )}
        </div>
      </Panel>

      <Panel title="Plan preview">
        <form
          className="grid gap-3 md:grid-cols-2"
          onSubmit={(e) => {
            e.preventDefault();
            plan.mutate();
          }}
        >
          <div className="flex flex-col gap-1 text-sm md:col-span-2">
            <span className="text-slate-400">Model (library or HF ref)</span>
            <ModelSelect value={modelRef} onChange={setModelRef} />
          </div>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">--ctx-size</span>
            <input
              className="rounded border border-slate-700 bg-slate-950 px-2 py-1"
              value={ctxSize}
              onChange={(e) => setCtxSize(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">--n-gpu-layers</span>
            <input
              className="rounded border border-slate-700 bg-slate-950 px-2 py-1"
              value={nGpuLayers}
              onChange={(e) => setNGpuLayers(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-sm md:col-span-2">
            <span className="text-slate-400">--tensor-split (blank, "auto", or 50,50)</span>
            <input
              className="rounded border border-slate-700 bg-slate-950 px-2 py-1"
              value={tensorSplit}
              onChange={(e) => setTensorSplit(e.target.value)}
            />
          </label>
          <div className="md:col-span-2">
            <button
              type="submit"
              className="rounded bg-sky-500 px-4 py-2 text-sm font-medium text-white hover:bg-sky-400"
              disabled={plan.isPending}
            >
              {plan.isPending ? "Computing…" : "Estimate"}
            </button>
          </div>
        </form>

        {plan.data && (
          <div className="mt-4 space-y-3 text-sm">
            <ul className="space-y-1">
              {plan.data.summary.map((line, i) => (
                <li key={i} className="text-slate-200">{line}</li>
              ))}
            </ul>
            {plan.data.gpus.length > 0 && (
              <div className="mt-3 space-y-2">
                <div className="text-xs uppercase text-slate-500">Per-GPU split</div>
                {plan.data.gpus.map((g, i) => (
                  <div key={i} className="rounded border border-slate-800 p-2">
                    <div className="font-medium">#{g.info.index} {g.info.name}</div>
                    <div className="text-xs text-slate-400">
                      weights {g.weights_h} · kv {g.kv_h}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {plan.data.warnings.length > 0 && (
              <div className="mt-3 rounded border border-amber-600/40 bg-amber-900/20 p-2 text-xs text-amber-200">
                {plan.data.warnings.map((w, i) => (
                  <div key={i}>⚠ {w}</div>
                ))}
              </div>
            )}
          </div>
        )}
        {plan.isError && (
          <div className="mt-4 text-sm text-rose-400">
            {(plan.error as Error).message}
          </div>
        )}
      </Panel>
    </div>
  );
}
