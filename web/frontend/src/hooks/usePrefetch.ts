import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";

/**
 * Prefetch every main query as soon as the app mounts so switching tabs
 * renders instantly. React Query will reuse the prefetched data within
 * ``staleTime`` (see main.tsx) and WebSockets keep it current.
 */
export function usePrefetch() {
  const qc = useQueryClient();

  useEffect(() => {
    const tasks: Array<Promise<unknown>> = [
      qc.prefetchQuery({ queryKey: ["health"], queryFn: api.health }),
      qc.prefetchQuery({ queryKey: ["gpus"], queryFn: api.listGpus }),
      qc.prefetchQuery({
        queryKey: ["instances"],
        queryFn: api.listInstances,
      }),
      qc.prefetchQuery({
        queryKey: ["local-models"],
        queryFn: api.listLocal,
      }),
      qc.prefetchQuery({ queryKey: ["builds"], queryFn: api.listBuilds }),
      qc.prefetchQuery({
        queryKey: ["supported-flags"],
        queryFn: api.supportedFlags,
      }),
    ];
    Promise.allSettled(tasks);
  }, [qc]);
}
