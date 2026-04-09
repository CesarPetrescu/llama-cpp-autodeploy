import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { ModelSelect } from "@/components/ModelSelect";
import { PageHeader } from "@/components/PageHeader";

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

  const local = useQuery({
    queryKey: ["local-models"],
    queryFn: api.listLocal,
  });
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

  useEffect(() => {
    if (!modelRef) return;
    const t = setTimeout(() => plan.mutate(), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelRef, ctxSize, nGpuLayers, tensorSplit]);

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Resources"
        title="Memory planning"
        description="Live GPU probe and weight / KV-cache split preview, powered by memory_utils."
      />

      <Panel title="GPUs">
        <div className="grid gap-3 md:grid-cols-3">
          {gpus.data?.gpus.map((g) => {
            const used =
              g.total != null && g.free != null ? g.total - g.free : null;
            const pct =
              g.total && used != null
                ? Math.min(100, Math.round((used / g.total) * 100))
                : 0;
            return (
              <div key={g.index} className="brand-surface-muted p-4">
                <div className="flex items-center justify-between">
                  <div className="font-display text-sm font-semibold text-bone-50">
                    #{g.index} {g.name}
                  </div>
                  <span className="brand-chip">{pct}%</span>
                </div>
                <div className="mt-2 text-[11px] uppercase tracking-wider text-bone-500">
                  {g.free_h} free · {g.total_h} total
                </div>
                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-white/5">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-lime-300 to-lime-500 shadow-[0_0_14px_rgba(213,255,64,0.45)]"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
          {gpus.data?.gpus.length === 0 && (
            <div className="col-span-full rounded-xl border border-white/5 bg-white/[0.02] p-4 text-sm text-bone-400">
              No CUDA devices detected.
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Plan preview">
        <form
          className="grid gap-4 md:grid-cols-2"
          onSubmit={(e) => {
            e.preventDefault();
            plan.mutate();
          }}
        >
          <div className="flex flex-col gap-1 text-sm md:col-span-2">
            <span className="brand-label">Model (library or HF ref)</span>
            <ModelSelect value={modelRef} onChange={setModelRef} />
          </div>
          <label className="flex flex-col gap-1 text-sm">
            <span className="brand-label">--ctx-size</span>
            <input
              className="brand-input"
              value={ctxSize}
              onChange={(e) => setCtxSize(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="brand-label">--n-gpu-layers</span>
            <input
              className="brand-input"
              value={nGpuLayers}
              onChange={(e) => setNGpuLayers(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-sm md:col-span-2">
            <span className="brand-label">
              --tensor-split (blank, auto, or 50,50)
            </span>
            <input
              className="brand-input"
              value={tensorSplit}
              onChange={(e) => setTensorSplit(e.target.value)}
            />
          </label>
          <div className="md:col-span-2">
            <button
              type="submit"
              className="brand-btn-primary"
              disabled={plan.isPending}
            >
              {plan.isPending ? "Computing…" : "Re-estimate"}
            </button>
          </div>
        </form>

        {plan.data && (
          <div className="mt-6 space-y-4 text-sm">
            <ul className="space-y-1.5 text-bone-200">
              {plan.data.summary.map((line, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1 w-1 rounded-full bg-lime-300" />
                  {line}
                </li>
              ))}
            </ul>
            {plan.data.gpus.length > 0 && (
              <div className="space-y-2">
                <div className="brand-label">Per-GPU split</div>
                <div className="grid gap-2 md:grid-cols-2">
                  {plan.data.gpus.map((g, i) => (
                    <div key={i} className="brand-surface-muted p-3">
                      <div className="font-medium text-bone-50">
                        #{g.info.index} {g.info.name}
                      </div>
                      <div className="mt-1 text-[11px] uppercase tracking-wider text-bone-400">
                        weights{" "}
                        <span className="text-lime-300">{g.weights_h}</span> ·
                        kv <span className="text-lime-300">{g.kv_h}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {plan.data.warnings.length > 0 && (
              <div className="rounded-xl border border-amber-400/40 bg-amber-400/10 p-3 text-xs text-amber-200">
                {plan.data.warnings.map((w, i) => (
                  <div key={i}>⚠ {w}</div>
                ))}
              </div>
            )}
          </div>
        )}
        {plan.isError && (
          <div className="mt-4 text-sm text-rose-300">
            {(plan.error as Error).message}
          </div>
        )}
      </Panel>
    </div>
  );
}
