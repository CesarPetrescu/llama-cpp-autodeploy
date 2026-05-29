import { FormEvent, useState } from "react";
import { getBaseUrl, getToken, setBaseUrl, setToken } from "@/api/client";
import { Panel } from "@/components/Panel";
import { PageHeader } from "@/components/PageHeader";

export default function Settings() {
  const [token, setTokenInput] = useState(getToken());
  const [base, setBaseInput] = useState(getBaseUrl());
  const [msg, setMsg] = useState<string | null>(null);

  function submit(e: FormEvent) {
    e.preventDefault();
    setToken(token.trim());
    setBaseUrl(base.trim());
    setMsg("Saved. Refreshing…");
    setTimeout(() => window.location.reload(), 400);
  }

  return (
    <div className="flex max-w-2xl flex-col gap-8">
      <PageHeader
        eyebrow="Config"
        title="Settings"
        description="Configure the backend URL and bearer token. The token is stored in localStorage and sent with every request."
      />

      <Panel title="Backend credentials">
        <form onSubmit={submit} className="flex flex-col gap-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="brand-label">
              Backend URL (blank = same origin)
            </span>
            <input
              value={base}
              onChange={(e) => setBaseInput(e.target.value)}
              className="brand-input"
              placeholder="http://192.168.1.10:8787"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="brand-label">Bearer token</span>
            <input
              value={token}
              onChange={(e) => setTokenInput(e.target.value)}
              type="password"
              className="brand-input"
              placeholder="from .web_config.json"
            />
          </label>
          <div className="flex items-center gap-3">
            <button type="submit" className="brand-btn-primary">
              Save
            </button>
            {msg && <p className="text-sm text-lime-200">{msg}</p>}
          </div>
        </form>
      </Panel>

      <Panel title="Security notes">
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-bone-300 marker:text-lime-300">
          <li>
            The backend binds to <code className="text-lime-300">0.0.0.0</code>{" "}
            by default so it's reachable from other hosts on your LAN. Protect
            it with the token or bind it to 127.0.0.1 in{" "}
            <code className="text-lime-300">.web_config.json</code>.
          </li>
          <li>
            WebSocket endpoints accept the token as a{" "}
            <code className="text-lime-300">?token=</code> query parameter
            because browsers can't set Authorization headers on WS upgrade. Use
            HTTPS via a reverse proxy when exposing outside your LAN.
          </li>
          <li>
            Starting <code className="text-lime-300">llama-server</code>{" "}
            inherits your environment, so{" "}
            <code className="text-lime-300">HF_TOKEN</code>,{" "}
            <code className="text-lime-300">CUDA_VISIBLE_DEVICES</code>, etc.
            work the same as the CLI.
          </li>
        </ul>
      </Panel>
    </div>
  );
}
