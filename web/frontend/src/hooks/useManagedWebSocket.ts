import { useEffect, useRef, useState } from "react";
import { buildWsUrl } from "@/api/client";

export type WebSocketStatus = "connecting" | "connected" | "reconnecting" | "disconnected";
type ConnectPhase = "initial" | "reconnect";

interface UseManagedWebSocketOptions {
  path: string | null;
  enabled?: boolean;
  beforeConnect?: (phase: ConnectPhase) => Promise<void> | void;
  onMessage?: (event: MessageEvent) => void;
}

export function useManagedWebSocket({
  path,
  enabled = true,
  beforeConnect,
  onMessage,
}: UseManagedWebSocketOptions) {
  const beforeConnectRef = useRef(beforeConnect);
  const onMessageRef = useRef(onMessage);
  const [status, setStatus] = useState<WebSocketStatus>(
    enabled && path ? "connecting" : "disconnected",
  );

  beforeConnectRef.current = beforeConnect;
  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!enabled || !path) {
      setStatus("disconnected");
      return;
    }

    let cancelled = false;
    let attempt = 0;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const clearReconnect = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const scheduleReconnect = () => {
      if (cancelled) return;
      setStatus("reconnecting");
      const delay = Math.min(10_000, 500 * 2 ** attempt++);
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        void openSocket("reconnect");
      }, delay);
    };

    const openSocket = async (phase: ConnectPhase) => {
      if (cancelled) return;
      setStatus(phase === "initial" ? "connecting" : "reconnecting");
      try {
        await beforeConnectRef.current?.(phase);
      } catch {
        // A failed reseed should not prevent the live socket from reconnecting.
      }
      if (cancelled) return;

      const ws = new WebSocket(buildWsUrl(path));
      socket = ws;

      ws.onopen = () => {
        if (cancelled || socket !== ws) return;
        attempt = 0;
        clearReconnect();
        setStatus("connected");
      };

      ws.onmessage = (event) => {
        if (cancelled || socket !== ws) return;
        onMessageRef.current?.(event);
      };

      ws.onerror = () => {
        try {
          ws.close();
        } catch {
          /* noop */
        }
      };

      ws.onclose = () => {
        if (cancelled || socket !== ws) return;
        socket = null;
        scheduleReconnect();
      };
    };

    void openSocket("initial");

    return () => {
      cancelled = true;
      clearReconnect();
      const active = socket;
      socket = null;
      if (active) {
        try {
          active.close();
        } catch {
          /* noop */
        }
      }
    };
  }, [enabled, path]);

  return {
    status,
    connected: status === "connected",
  };
}
