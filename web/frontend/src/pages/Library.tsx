import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LocalModel, api } from "@/api/client";
import { Panel } from "@/components/Panel";

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
    const next = window.prompt(
      `Rename '${m.rel ?? m.name}' to:`,
      m.rel ?? m.name,
    );
    if (!next || next === (m.rel ?? m.name)) return;
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
    <div className="flex flex-col gap-6">
      <header>
        <h2 className="text-2xl font-semibold">Model library</h2>
        <p className="text-sm text-slate-400">
          Browse, rename, and delete local GGUFs. New models can be pulled
          directly from Hugging Face.
        </p>
      </header>

      <Panel title="Download from Hugging Face">
        <form onSubmit={submit} className="flex flex-col gap-3 md:flex-row">
          <input
            value={spec}
            onChange={(e) => setSpec(e.target.value)}
            placeholder="Qwen/Qwen3-Embedding-8B-GGUF:Q8_0"
            className="flex-1 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
          />
          <input
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="hf_token (optional)"
            type="password"
            className="w-60 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={download.isPending}
            className="rounded bg-sky-500 px-4 py-2 text-sm font-medium text-white hover:bg-sky-400 disabled:opacity-50"
          >
            {download.isPending ? "Downloading…" : "Download"}
          </button>
        </form>
        {msg && <p className="mt-2 text-sm text-slate-300">{msg}</p>}
      </Panel>

      <Panel title={`Local GGUFs (${local.data?.models.length ?? 0})`}>
        <p className="mb-2 text-xs text-slate-500">{local.data?.models_dir}</p>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase text-slate-500">
              <th className="py-1">Name</th>
              <th>Size</th>
              <th>Params</th>
              <th>Quant</th>
              <th className="text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {local.data?.models.map((m) => (
              <tr key={m.path} className="border-t border-slate-800">
                <td className="py-2" title={m.path}>
                  {m.rel ?? m.name}
                </td>
                <td>{m.size_h}</td>
                <td>{m.params_h}</td>
                <td>{m.quant ?? "—"}</td>
                <td className="space-x-1 text-right">
                  <button
                    className="rounded bg-slate-800 px-2 py-1 text-xs hover:bg-slate-700"
                    onClick={() => onRename(m)}
                    disabled={renameMut.isPending}
                  >
                    Rename
                  </button>
                  <button
                    className="rounded bg-rose-700 px-2 py-1 text-xs hover:bg-rose-600"
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
                <td colSpan={5} className="py-4 text-center text-slate-500">
                  No GGUF files found. Try downloading one above.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
