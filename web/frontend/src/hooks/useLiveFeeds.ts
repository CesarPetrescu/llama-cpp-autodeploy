import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { BenchmarksEvent, BuildsEvent, InstancesEvent } from "@/api/client";
import { useManagedWebSocket } from "./useManagedWebSocket";

type AnyEvent = InstancesEvent | BuildsEvent | BenchmarksEvent;

/**
 * Opens both /api/instances/events and /api/builds/events WebSockets once
 * per app lifetime and mirrors incoming snapshots straight into the
 * React Query cache so every page sees live data without polling.
 *
 * Returns a small summary (counts) for the header/badges.
 */
export function useLiveFeeds() {
  const qc = useQueryClient();
  const [instances, setInstances] = useState({ total: 0, running: 0 });
  const [builds, setBuilds] = useState({ total: 0, running: 0 });
  const [benchmarks, setBenchmarks] = useState({ total: 0, running: 0 });
  const instancesSocket = useManagedWebSocket({
    path: "/api/instances/events",
    onMessage: (ev) => {
      try {
        const payload = JSON.parse(String(ev.data)) as AnyEvent;
        if (payload.type !== "instances.snapshot") return;
        setInstances({ total: payload.total, running: payload.running });
        qc.setQueryData(["instances"], { instances: payload.instances });
      } catch {
        /* ignore malformed */
      }
    },
  });
  const buildsSocket = useManagedWebSocket({
    path: "/api/builds/events",
    onMessage: (ev) => {
      try {
        const payload = JSON.parse(String(ev.data)) as AnyEvent;
        if (payload.type !== "builds.snapshot") return;
        setBuilds({ total: payload.total, running: payload.running });
        qc.setQueryData(["builds"], { builds: payload.builds });
      } catch {
        /* ignore malformed */
      }
    },
  });
  const benchmarksSocket = useManagedWebSocket({
    path: "/api/benchmarks/events",
    onMessage: (ev) => {
      try {
        const payload = JSON.parse(String(ev.data)) as AnyEvent;
        if (payload.type !== "benchmarks.snapshot") return;
        setBenchmarks({ total: payload.total, running: payload.running });
        qc.setQueryData(["benchmarks"], { benchmarks: payload.benchmarks });
      } catch {
        /* ignore malformed */
      }
    },
  });

  const connected =
    instancesSocket.connected || buildsSocket.connected || benchmarksSocket.connected;

  return { instances, builds, benchmarks, connected };
}
