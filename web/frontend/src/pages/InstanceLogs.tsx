import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Panel } from "@/components/Panel";
import { LogPane } from "@/components/LogPane";
import { StatusBadge } from "@/components/StatusBadge";
import { useManagedWebSocket } from "@/hooks/useManagedWebSocket";

function isActiveInstance(status?: string): boolean {
  return status === "running" || status === "stopping";
}

export default function InstanceLogs() {
  const { id = "" } = useParams();
  const [lines, setLines] = useState<string[]>([]);

  const meta = useQuery({
    queryKey: ["instance", id],
    queryFn: () => api.getInstance(id),
    enabled: !!id,
    refetchInterval: 5000,
  });

  const inst = meta.data?.instance;
  const active = isActiveInstance(inst?.status);

  useEffect(() => {
    setLines([]);
  }, [id]);

  useEffect(() => {
    if (!meta.data || meta.data.instance.id !== id || lines.length > 0) return;
    setLines(meta.data.logs);
  }, [id, meta.data, lines.length]);

  const socket = useManagedWebSocket({
    path: id && active ? `/api/instances/${id}/logs?history=0` : null,
    enabled: Boolean(id && active),
    beforeConnect: async (phase) => {
      if (phase !== "reconnect" || !id) return;
      const response = await api.getInstance(id);
      setLines(response.logs);
    },
    onMessage: (ev) => {
      setLines((prev) => {
        const next = [...prev, String(ev.data)];
        if (next.length > 5000) next.splice(0, next.length - 5000);
        return next;
      });
    },
  });

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-center justify-between gap-4">
        <div>
          <Link
            to="/instances"
            className="text-[11px] font-semibold uppercase tracking-[0.18em] text-lime-300 hover:text-lime-200"
          >
            ← back to instances
          </Link>
          <h1 className="mt-2 font-display text-3xl font-bold tracking-tight text-bone-50">
            {inst?.name ?? id}{" "}
            <span className="text-bone-400">· logs</span>
          </h1>
          <p className="mt-1 text-[11px] uppercase tracking-wider text-bone-500">
            {active
              ? `Log stream · ${socket.status}`
              : "Viewing durable log history"}
          </p>
        </div>
        {inst && (
          <div className="flex items-center gap-3 text-xs text-bone-400">
            <StatusBadge status={inst.status} />
            <span className="brand-chip font-mono normal-case tracking-normal">
              {inst.host}:{inst.port}
            </span>
            {inst.pid && <span className="brand-chip">pid {inst.pid}</span>}
          </div>
        )}
      </header>

      <Panel>
        <LogPane lines={lines} height="70vh" />
      </Panel>
    </div>
  );
}
