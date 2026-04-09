import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

interface Props {
  value: string;
  onChange: (value: string) => void;
  /** Allow free-form input alongside the dropdown (HF refs, etc.). */
  allowManual?: boolean;
  /** Extra placeholder text for the manual box. */
  manualPlaceholder?: string;
  className?: string;
}

/**
 * Dropdown populated from the local GGUF library with an optional
 * manual-entry field for HF references or paths that aren't downloaded yet.
 *
 * The query is prefetched by usePrefetch so the dropdown is populated the
 * moment this component mounts.
 */
export function ModelSelect({
  value,
  onChange,
  allowManual = true,
  manualPlaceholder = "or paste org/repo:quant (e.g. Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M)",
  className = "",
}: Props) {
  const local = useQuery({
    queryKey: ["local-models"],
    queryFn: api.listLocal,
  });

  const models = local.data?.models ?? [];
  const modelsDir = local.data?.models_dir ?? "";

  // Match by either absolute path (what the backend gives us) or plain name.
  const selectedInList = models.some(
    (m) => m.path === value || m.name === value || m.rel === value,
  );

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <select
        className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
        value={selectedInList ? value : ""}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">
          {models.length === 0
            ? "— library is empty —"
            : "— pick a local GGUF —"}
        </option>
        {models.map((m) => (
          <option key={m.path} value={m.path}>
            {m.rel ?? m.name}
            {m.size_h ? ` · ${m.size_h}` : ""}
            {m.quant ? ` · ${m.quant}` : ""}
          </option>
        ))}
      </select>
      {allowManual && (
        <input
          type="text"
          className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
          placeholder={manualPlaceholder}
          value={selectedInList ? "" : value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      {modelsDir && (
        <p className="text-[10px] text-slate-500">library: {modelsDir}</p>
      )}
    </div>
  );
}
