# llama-cpp-autodeploy web frontend

React + Vite + TypeScript frontend for the `web/backend` FastAPI service.

## Develop

```bash
cd web/frontend
npm install
npm run dev         # http://localhost:5173 (proxies /api -> http://127.0.0.1:8787)
```

Point at a different backend during dev:

```bash
VITE_BACKEND_URL=http://192.168.1.10:8787 npm run dev
```

In the running app, open **Settings** and paste the bearer token printed by
`python web_cli.py --init` (or found in `.web_config.json`).

## Build

```bash
npm run build       # writes web/frontend/dist/
```

When `web/frontend/dist` exists, `python web_cli.py` mounts it at `/` so the
full app is available at `http://<host>:8787`.
