import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LocalModel, api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { PageHeader } from "@/components/PageHeader";

export default function Library() {
  const qc = useQueryClient();
  const [spec, setSpec] = useState("");
  const [token, setToken] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  const local = useQuery({ queryKey: ["local-models"], queryFn: api.listLocal });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["local-models"] });
  };

  const download = useMutation({
    mutationFn: () => api.downloadModel(spec, token || undefined),
    onSuccess: (data) => {
      setMsg(`Downloaded ${data.model.name}`);
      invalidate();
    },
    onError: (err: Error) => setMsg(err.message),
  });

  const renameMut = useMutation({
    mutationFn: (args: { name: string; newName: string }) =>
      api.renameModel(args.name, args.newName),
    onSuccess: (_data, vars) => {
      setMsg(`Renamed to ${vars.newName}`);
      invalidate();
    },
    onError: (err: Error) => setMsg(err.message),
  });

  const deleteMut = useMutation({
    mutationFn: (name: string) => api.deleteModel(name),
    onSuccess: (_data, name) => {
      setMsg(`Deleted ${name}`);
      invalidate();
    },
    onError: (err: Error) => setMsg(err.message),
  });

  function submit(e: FormEvent) {
    e.preventDefault();
    if (!spec.trim()) return;
    setMsg(null);
    download.mutate();
  }

  function onRename(m: LocalModel) {
    const key = m.rel ?? m.name;
    const next = window.prompt(`Rename '${key}' to:`, key);
    if (!next || next === key) return;
    setMsg(null);
    renameMut.mutate({ name: key, newName: next });
  }

  function onDelete(m: LocalModel) {
    const key = m.rel ?? m.name;
    if (!window.confirm(`Permanently delete ${key}? This cannot be undone.`)) {
      return;
    }
    setMsg(null);
    deleteMut.mutate(key);
  }

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Storage"
        title="Model library"
        description="Browse, rename, and delete local GGUFs. Pull new models directly from Hugging Face."
      />

      <Panel title="Download from Hugging Face">
        <form onSubmit={submit} className="flex flex-col gap-3 md:flex-row">
          <input
            value={spec}
            onChange={(e) => setSpec(e.target.value)}
            placeholder="Qwen/Qwen3-Embedding-8B-GGUF:Q8_0"
            className="brand-input flex-1"
          />
          <input
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="hf_token (optional)"
            type="password"
            className="brand-input w-60"
          />
          <button
            type="submit"
            disabled={download.isPending}
            className="brand-btn-primary"
          >
            {download.isPending ? "Downloading…" : "Download"}
          </button>
        </form>
        {msg && (
          <p className="mt-3 text-sm text-lime-200">
            <span className="mr-2 inline-block h-1 w-1 rounded-full bg-lime-300" />
            {msg}
          </p>
        )}
      </Panel>

      <Panel
        title={`Local GGUFs · ${local.data?.models.length ?? 0}`}
        subtitle={local.data?.models_dir}
      >
        <div className="overflow-hidden rounded-xl border border-white/5">
          <table className="w-full text-sm">
            <thead className="bg-white/[0.03]">
              <tr className="text-left text-[10px] uppercase tracking-[0.18em] text-bone-500">
                <th className="px-4 py-2.5 font-semibold">Name</th>
                <th className="px-4 py-2.5 font-semibold">Size</th>
                <th className="px-4 py-2.5 font-semibold">Params</th>
                <th className="px-4 py-2.5 font-semibold">Quant</th>
                <th className="px-4 py-2.5 text-right font-semibold">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {local.data?.models.map((m) => (
                <tr
                  key={m.path}
                  className="border-t border-white/5 hover:bg-white/[0.02]"
                >
                  <td
                    className="px-4 py-3 font-medium text-bone-100"
                    title={m.path}
                  >
                    {m.rel ?? m.name}
                  </td>
                  <td className="px-4 py-3 text-bone-300">{m.size_h}</td>
                  <td className="px-4 py-3 text-bone-300">{m.params_h}</td>
                  <td className="px-4 py-3">
                    {m.quant ? (
                      <span className="brand-chip text-lime-200">
                        {m.quant}
                      </span>
                    ) : (
                      <span className="text-bone-500">—</span>
                    )}
                  </td>
                  <td className="space-x-2 px-4 py-3 text-right">
                    <button
                      className="brand-btn-ghost px-3 py-1.5 text-xs"
                      onClick={() => onRename(m)}
                      disabled={renameMut.isPending}
                    >
                      Rename
                    </button>
                    <button
                      className="brand-btn-danger px-3 py-1.5 text-xs"
                      onClick={() => onDelete(m)}
                      disabled={deleteMut.isPending}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {local.data?.models.length === 0 && (
                <tr>
                  <td
                    colSpan={5}
                    className="px-4 py-10 text-center text-bone-500"
                  >
                    No GGUF files found. Try downloading one above.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
