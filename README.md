# llama-cpp-autodeploy

Python tooling to build `llama.cpp` and launch inference services from local or Hugging Face-hosted models.

This repository includes:

- Build automation (`autodevops.py`) for fetching/building `llama.cpp` with configurable CUDA/MMQ/BLAS/RPC options.
- Curses TUIs for build and launch workflows (`autodevops_cli.py`, `loadmodel_cli.py`, `loadmodel_dist_cli.py`).
- Runtime launchers for:
  - `llama-server` (LLM + embeddings) via `loadmodel.py`
  - distributed `llama-cli` + `rpc-server` via `loadmodel_dist_cli.py`
  - standalone `rpc-server` via `rpc_server_cli.py`
  - Transformers reranking via `loadmodel.py --rerank` or `reranker.py`

## Repository layout

- `autodevops.py` — non-interactive `llama.cpp` build script.
- `autodevops_cli.py` — interactive build wizard (curses UI).
- `loadmodel.py` — launcher for `llama-server` and Transformers reranker service.
- `loadmodel_cli.py` — interactive launcher for local/HF GGUF and reranker workflows.
- `loadmodel_dist_cli.py` — interactive distributed launcher (RPC worker discovery + launch).
- `rpc_server_cli.py` — helper wrapper for `./bin/rpc-server`.
- `reranker.py` — standalone Transformers reranker CLI/HTTP server.
- `run/` — sample shell launch scripts.
- `tests/` — unit tests for build + build-TUI configuration logic.

## Requirements

- Linux with Python 3.10+ (tested here with Python 3.12).
- Build tools for `llama.cpp`: `git`, `cmake`, `make`, `gcc`, `g++`, `pkg-config`.
- NVIDIA drivers + CUDA toolkit if using CUDA builds/runtime.
- Optional BLAS libraries:
  - Intel MKL (for `--blas mkl`)
  - OpenBLAS (for `--blas openblas`)

Python dependencies are in `requirements.txt` (PyTorch CUDA 12.9 index + Transformers stack).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Build llama.cpp

### Interactive (recommended)

```bash
python autodevops_cli.py
```

> Note: this is a curses TUI and must run in a real terminal (not a non-interactive CI shell).

### Non-interactive

```bash
python autodevops.py --help
python autodevops.py --ref latest --now
```

Supported build flags:

- `--ref <tag|branch|commit|latest>`
- `--now`
- `--fast-math`
- `--force-mmq {auto,on,off}`
- `--blas {auto,openblas,mkl,off}`
- `--distributed` (builds GGML RPC backend)
- `--cpu-only` (skip NVIDIA driver precheck)

## Run model servers

### Interactive launcher

```bash
python loadmodel_cli.py
```

### Unified CLI launcher (`loadmodel.py`)

`loadmodel.py` supports mutually exclusive modes:

- `--llm` → starts `./bin/llama-server` completion API
- `--embed` → starts `./bin/llama-server` embeddings API
- `--rerank` → starts Transformers reranker HTTP service

```bash
python loadmodel.py --help
```

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

If the local `llama-server` binary does not expose these flags, `loadmodel.py` exits with a rebuild hint.

## Distributed inference (RPC)

### Interactive distributed launcher

```bash
python loadmodel_dist_cli.py
```

This TUI can:

- scan private subnets for RPC workers,
- manage worker host list,
- optionally start a local `rpc-server`,
- launch `llama-cli` with `--rpc` workers.

### Standalone rpc-server helper

```bash
python rpc_server_cli.py --help
python rpc_server_cli.py --host 0.0.0.0 --port 5515 --devices 0
```

`rpc_server_cli.py` requires `./bin/rpc-server` to exist (build with `--distributed` / distributed backend enabled).

## Web UI (browser backend + frontend)

A FastAPI backend and React + Vite + TypeScript frontend under [web/](web/)
let you manage `llama-server` instances, stream logs over WebSocket, preview
memory plans, browse the model library, and trigger `autodevops.py` builds
from a browser. It reuses the same Python helpers as the TUIs
([loadmodel.py](loadmodel.py), [memory_utils.py](memory_utils.py),
[autodevops.py](autodevops.py)) so behaviour is consistent across both
surfaces.

### Backend

```bash
source venv/bin/activate
pip install -r requirements.txt          # adds fastapi, uvicorn, pydantic, websockets
python web_cli.py --init                 # writes .web_config.json with a fresh bearer token
python web_cli.py                        # serves on http://0.0.0.0:8787 by default
```

The backend binds to `0.0.0.0` by default and requires a bearer token on every
request (`/api/health` is the only public endpoint). Change
`host`/`port`/`models_dir` in [.web_config.json](.web_config.json). Managed
instances and build history persist in `.web_state.json`; per-instance logs
tee to `web/logs/<id>.log` so earlier output survives a backend restart.
On backend startup, the manager also scans for orphaned `llama-server`
processes launched from this repo and re-adopts them so stop/restart control
can be recovered after a backend crash. You can force the same scan manually
with `POST /api/instances/recover`.

Endpoints (see `GET /docs` for the full OpenAPI schema):

- `GET  /api/health` — public health check
- `GET  /api/memory/gpus`, `POST /api/memory/plan`, `POST /api/memory/auto-split`
- `GET  /api/models/local`, `GET /api/models/binary-caps`, `POST /api/models/download`
- `GET /POST /api/instances`, `GET/POST /api/instances/{id}[/start|/stop|/restart]`, `DELETE /api/instances/{id}`
- `POST /api/instances/recover` — rescan `/proc` and adopt orphaned managed `llama-server` processes
- `WS   /api/instances/{id}/logs?token=…` — live stdout tail
- `GET /POST /api/builds`, `GET /api/builds/{id}`, `POST /api/builds/{id}/stop`
- `WS   /api/builds/{id}/logs?token=…`

### Frontend

```bash
cd web/frontend
npm install
npm run dev        # http://localhost:5173, proxies /api -> http://127.0.0.1:8787
# or for production:
npm run build      # writes web/frontend/dist/
```

When `web/frontend/dist/` exists, `python web_cli.py` automatically mounts it
at `/` so the full app is served at `http://<host>:8787`. In the UI, open
**Settings** and paste the token printed by `python web_cli.py --init` (or
read it from `.web_config.json`). Pages:

- **Dashboard** — backend health, GPU stats, running-instance summary
- **Instances** — create/start/stop/restart/delete `llama-server` processes
  with a form mirroring `loadmodel_cli.py` (mode, model ref, n-gpu-layers,
  tensor-split, ctx-size, cpu-moe, jinja, extra flags, …)
- **Instance logs** — live WebSocket tail with pause/resume
- **Memory** — live `detect_gpus()` probe + `estimate_memory_profile()`
  per-GPU weight/KV preview
- **Library** — scan `./models` for GGUFs and download new ones from
  Hugging Face via `resolve_gguf()`
- **Builds** — trigger `autodevops.py` and stream the build log
- **Settings** — backend URL + bearer token

### Security notes

- The bearer token is the only auth layer; keep `.web_config.json` readable
  only by you and prefer binding to `127.0.0.1` when you don't need remote
  access.
- WebSocket endpoints accept the token as a `?token=` query parameter because
  browsers can't set `Authorization` headers on WS upgrade. If you expose the
  backend beyond a trusted LAN, put it behind an HTTPS reverse proxy.
- Managed `llama-server` processes inherit the backend's environment, so
  `HF_TOKEN`, `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS` and friends work the
  same as running `loadmodel.py` directly.

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
- option/config assembly helpers in `autodevops_cli.py`

## Sample scripts

`run/` currently includes:

- `run_qwen30b_llm.sh`
- `run_qwen_embed8b.sh`
- `run_qwen_reranker8b.sh`

These are examples for fixed ports/model targets and can be adapted to your environment.
