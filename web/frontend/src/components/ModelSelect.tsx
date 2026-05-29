import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

interface Props {
  value: string;
  onChange: (value: string) => void;
  allowManual?: boolean;
  manualPlaceholder?: string;
  className?: string;
}

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

  const selectedInList = models.some(
    (m) => m.path === value || m.name === value || m.rel === value,
  );

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <select
        className="brand-input"
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
          className="brand-input"
          placeholder={manualPlaceholder}
          value={selectedInList ? "" : value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      {modelsDir && (
        <p className="text-[10px] uppercase tracking-wider text-bone-500">
          library · {modelsDir}
        </p>
      )}
    </div>
  );
}
