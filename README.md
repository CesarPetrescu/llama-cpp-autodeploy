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

## Convenience launcher

`./start` uses `./venv/bin/python` and offers a small menu:

```bash
./start
./start autodevops
./start loadmodel
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
