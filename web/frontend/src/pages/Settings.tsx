import { FormEvent, useState } from "react";
import { getBaseUrl, getToken, setBaseUrl, setToken } from "@/api/client";
import { Panel } from "@/components/Panel";

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
    <div className="flex max-w-xl flex-col gap-6">
      <header>
        <h2 className="text-2xl font-semibold">Settings</h2>
        <p className="text-sm text-slate-400">
          Configure the backend URL and bearer token. The token is stored in
          localStorage and sent with every request.
        </p>
      </header>

      <Panel title="Backend credentials">
        <form onSubmit={submit} className="flex flex-col gap-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">Backend URL (blank = same origin)</span>
            <input
              value={base}
              onChange={(e) => setBaseInput(e.target.value)}
              className="rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
              placeholder="http://192.168.1.10:8787"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">Bearer token</span>
            <input
              value={token}
              onChange={(e) => setTokenInput(e.target.value)}
              type="password"
              className="rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
              placeholder="from .web_config.json"
            />
          </label>
          <button
            type="submit"
            className="self-start rounded bg-sky-500 px-4 py-2 text-sm font-medium text-white hover:bg-sky-400"
          >
            Save
          </button>
          {msg && <p className="text-sm text-emerald-400">{msg}</p>}
        </form>
      </Panel>

      <Panel title="Security notes">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-300">
          <li>
            The backend binds to <code>0.0.0.0</code> by default so it is
            reachable from other hosts on your LAN. Protect it with the token
            or bind it to 127.0.0.1 in <code>.web_config.json</code>.
          </li>
          <li>
            WebSocket endpoints accept the token as a <code>?token=</code> query
            parameter because browsers can't set Authorization headers on WS
            upgrade. Use HTTPS via a reverse proxy when exposing outside your
            LAN.
          </li>
          <li>
            Starting <code>llama-server</code> inherits your environment, so
            <code>HF_TOKEN</code>, <code>CUDA_VISIBLE_DEVICES</code>, etc. work
            the same as the CLI.
          </li>
        </ul>
      </Panel>
    </div>
  );
}
