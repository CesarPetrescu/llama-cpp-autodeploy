# llama-cpp-autodeploy

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white" alt="Linux" />
  <img src="https://img.shields.io/badge/backend-FastAPI-05998B?logo=fastapi&logoColor=white" alt="FastAPI backend" />
  <img src="https://img.shields.io/badge/frontend-React%20%2B%20Vite-1F2937?logo=vite&logoColor=FACC15" alt="React and Vite frontend" />
  <img src="https://img.shields.io/badge/runtime-llama.cpp-3F3F46" alt="llama.cpp runtime" />
</p>

<p align="center">
  Turn a local <code>llama.cpp</code> checkout into something you can actually use:
  build it, launch models, inspect GPU pressure, recover orphaned servers, and control everything from a browser or terminal.
</p>

<p align="center">
  <a href="#why-this-is-useful">Why</a> |
  <a href="#see-the-app">Screenshots</a> |
  <a href="#start-here">Start Here</a> |
  <a href="#what-you-can-do-in-the-ui">UI</a> |
  <a href="#technical-reference">Technical Reference</a>
</p>

## Why this is useful

- You do not have to manually juggle `llama.cpp` builds, `llama-server` launch commands, VRAM checks, and logs across separate scripts.
- The web UI gives you one place to build, launch, monitor, and recover running instances.
- If the backend restarts or crashes, it can re-adopt repo-launched `llama-server` processes instead of losing control of them.
- The browser UI is layered on top of the same local tools, so terminal users and UI users are working against the same repo, binaries, and models.

## Why it is easy to use

- One repo checkout.
- One Python environment.
- One backend command to bring up the control plane.
- One token for the browser UI.
- The backend can serve the built frontend directly, so normal users do not need to run a separate frontend dev server.

## See the app

#### Overview

The main control-plane page shows backend health, fleet status, host load,
GPU pressure, and recent activity without making you dig through logs first.

<p align="center">
  <img src="docs/screenshots/web-dashboard-overview.png" alt="Dashboard overview" width="100%" />
</p>

#### GPU Runtime Detail

Expandable GPU detail shows compute load, VRAM use, and which managed
processes currently own memory on each device.

<p align="center">
  <img src="docs/screenshots/web-dashboard-gpu.png" alt="Dashboard GPU runtime detail" width="100%" />
</p>

#### Instances

The Instances page gives you a proper launcher for `llama-server` and a way to
recover servers that survived a backend restart.

<p align="center">
  <img src="docs/screenshots/web-instances.png" alt="Instances page" width="100%" />
</p>

#### Builds

The Builds page wraps `autodevops.py` with real options, history, command
preview, and logs.

<p align="center">
  <img src="docs/screenshots/web-builds.png" alt="Builds page" width="100%" />
</p>

#### Mobile

The same control plane also works on narrow screens.

<p align="center">
  <img src="docs/screenshots/web-dashboard-mobile.png" alt="Dashboard on mobile" width="42%" />
  <img src="docs/screenshots/web-builds-mobile.png" alt="Builds on mobile" width="42%" />
</p>

## Start here

This is the simplest browser-first path for a normal user.

### 1. Clone the repo

```bash
git clone https://github.com/CesarPetrescu/llama-cpp-autodeploy.git
cd llama-cpp-autodeploy
```

### 2. Install the backend requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3. Build the frontend once

```bash
cd web/frontend
npm install
npm run build
cd ../..
```

That gives the backend a production frontend to serve at `/`.

### 4. Initialize and start the backend

```bash
python web_cli.py --init
python web_cli.py
```

What happens here:

- `python web_cli.py --init` creates `.web_config.json` and prints a bearer token.
- `python web_cli.py` starts the backend on port `8787` by default.
- If `web/frontend/dist/` exists, the backend also serves the frontend UI.

### 5. Open the app

Open `http://localhost:8787`.

On first use:

- go to **Settings**
- paste the bearer token printed during `--init`
- save it once

After that, the app can build `llama.cpp`, launch models, show logs, inspect
GPU pressure, and recover managed instances from the browser.

### 6. Frontend development mode (optional)

If you are editing the UI instead of just using it:

```bash
cd web/frontend
npm run dev
```

That starts the Vite frontend at `http://localhost:5173` and proxies API
requests to the backend at `http://127.0.0.1:8787`.

## What you can do in the UI

| Page | What it is for |
| --- | --- |
| **Dashboard** | See backend health, host CPU/RAM/load, GPU pressure, builds, and fleet state |
| **Instances** | Create, recover, start, stop, restart, and delete `llama-server` processes |
| **Instance logs** | Watch live stdout with pause/resume |
| **Memory** | Estimate placement and VRAM needs before launch |
| **Library** | Scan local GGUFs and download new ones from Hugging Face |
| **Builds** | Run `autodevops.py`, inspect supported options, and stream logs |
| **Settings** | Set backend URL and bearer token |

## How the app fits together

| Layer | Role |
| --- | --- |
| `autodevops.py` | Build local `llama.cpp` binaries |
| `loadmodel.py` | Launch `llama-server` and reranker processes |
| `memory_utils.py` | Probe VRAM, RAM, and placement estimates |
| `web/backend/` | Auth, state, logs, recovery, and API surface |
| `web/frontend/` | Browser UI for overview, builds, instances, memory, and library |

## Technical reference

### Requirements

- Linux with Python 3.10+.
- Build tools for `llama.cpp`: `git`, `cmake`, `make`, `gcc`, `g++`, `pkg-config`.
- NVIDIA drivers and CUDA toolkit if you want CUDA builds or GPU runtime.
- Optional BLAS libraries:
  - Intel MKL for `--blas mkl`
  - OpenBLAS for `--blas openblas`

Python dependencies are in `requirements.txt`.

### Build llama.cpp

Interactive build flow:

```bash
python autodevops_cli.py
```

Non-interactive build flow:

```bash
python autodevops.py --help
python autodevops.py --ref latest --now
```

Supported build flags:

| Flag | Meaning |
| --- | --- |
| `--ref <tag|branch|commit|latest>` | Build a specific upstream ref |
| `--now` | Build immediately instead of waiting for the scheduled path |
| `--fast-math` | Pass fast-math CUDA flags to NVCC |
| `--force-mmq {auto,on,off}` | Control MMQ CUDA kernels |
| `--blas {auto,openblas,mkl,off}` | Choose the CPU BLAS backend |
| `--distributed` | Build GGML RPC support |
| `--cpu-only` | Skip NVIDIA driver prechecks |

### Launch services

Interactive launcher:

```bash
python loadmodel_cli.py
```

Unified launcher:

```bash
python loadmodel.py --help
```

`loadmodel.py` supports three mutually exclusive modes:

| Mode | Result |
| --- | --- |
| `--llm` | Start `./bin/llama-server` for completion/chat |
| `--embed` | Start `./bin/llama-server` for embeddings |
| `--rerank` | Start the Transformers reranker HTTP service |

Examples:

```bash
# LLM (local GGUF)
python loadmodel.py --llm ./models/model.gguf --port 45540

# Embeddings (download GGUF from HF repo, auto-select quant/file)
python loadmodel.py --embed Qwen/Qwen3-Embedding-8B-GGUF:Q8_0 --port 45541

# Reranker HTTP server
python loadmodel.py --rerank Qwen/Qwen3-Reranker-8B --host 127.0.0.1 --port 45542
```

For MoE-capable `llama-server` builds, `loadmodel.py` also accepts:

- `--cpu-moe`
- `--n-cpu-moe <N>`

If the local `llama-server` binary does not expose these flags,
`loadmodel.py` exits with a rebuild hint.

### Distributed inference (RPC)

Interactive distributed launcher:

```bash
python loadmodel_dist_cli.py
```

This flow can:

- scan private subnets for RPC workers
- manage the worker host list
- optionally start a local `rpc-server`
- launch `llama-cli` with `--rpc` workers

Standalone `rpc-server` helper:

```bash
python rpc_server_cli.py --help
python rpc_server_cli.py --host 0.0.0.0 --port 5515 --devices 0
```

`rpc_server_cli.py` requires `./bin/rpc-server` to exist.

### Web backend

Backend startup:

```bash
python web_cli.py --init
python web_cli.py
```

The backend:

- binds to `0.0.0.0` by default
- requires a bearer token on every request except `GET /api/health`
- persists managed instances and builds in `.web_state.json`
- tees logs to `web/logs/<id>.log`
- can re-adopt orphaned repo-launched `llama-server` processes on startup
- can force that same recovery flow through `POST /api/instances/recover`

<details>
<summary>API surface</summary>

- Health: `GET /api/health`
- Memory: `GET /api/memory/gpus`, `POST /api/memory/plan`, `POST /api/memory/auto-split`
- Models: `GET /api/models/local`, `GET /api/models/binary-caps`, `POST /api/models/download`
- Instances: `GET /POST /api/instances`, `GET /api/instances/{id}`, `POST /api/instances/{id}/start|stop|restart`, `DELETE /api/instances/{id}`, `POST /api/instances/recover`, `WS /api/instances/{id}/logs?token=...`
- Builds: `GET /POST /api/builds`, `GET /api/builds/{id}`, `POST /api/builds/{id}/stop`, `WS /api/builds/{id}/logs?token=...`

Full schema: `GET /docs`
</details>

### Security notes

- The bearer token is the only built-in auth layer.
- Keep `.web_config.json` readable only by you.
- Prefer binding to `127.0.0.1` when you do not need remote access.
- WebSocket endpoints use `?token=` because browsers cannot attach
  `Authorization` headers during the upgrade request.
- If you expose the backend beyond a trusted LAN, put it behind HTTPS.

<details>
<summary>Refresh screenshot assets</summary>

```bash
cd web/frontend
npx playwright install chromium
WEB_BEARER_TOKEN="$(python - <<'PY'
import json
print(json.load(open('../../.web_config.json', 'r', encoding='utf-8'))['token'])
PY
)" npm run screenshots:readme
```
</details>

## Convenience launcher

`./start` uses `./venv/bin/python` and offers a small menu:

```bash
./start
./start autodevops
./start loadmodel
./start web [--init]
./start --help
```

## Tests

Run unit tests:

```bash
python -m unittest discover -s tests
```

Current tests cover:

- CUDA home resolution behavior in `autodevops.py`
- option and config assembly helpers in `autodevops_cli.py`

## Sample scripts

`run/` currently includes:

- `run_qwen30b_llm.sh`
- `run_qwen_embed8b.sh`
- `run_qwen_reranker8b.sh`

These are example launchers for fixed ports and model targets.
