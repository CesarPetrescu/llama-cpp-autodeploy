# llama.cpp Automated Build & Deployment (Python) — NVIDIA + Intel oneAPI MKL

Automated scripts to build llama.cpp and run llama-server with your models. Supports LLM generation, embeddings, and document reranking. Optimized for NVIDIA GPUs and Intel CPUs via oneAPI MKL.

## Features

- LLM server: chat and text completions
- Embeddings server: vector embeddings with optional pooling
- Reranking: semantic doc rerankers (GGUF and Transformers)
- Auto-build: CUDA-enabled, MMQ kernels, deterministic builds
- MKL acceleration: fastest CPU BLAS via Intel oneAPI MKL
- Multi-GPU: simple tensor split across NVIDIA GPUs
- Model management: download GGUF models automatically

## Quick start (Intel i9‑13900KS + 2×RTX 4090)

```bash
# 0) System deps
sudo apt update && sudo apt install -y git cmake build-essential pkg-config

# 1) NVIDIA: driver + CUDA Toolkit
nvidia-smi            # verify driver and 2x 4090 detected
nvcc --version        # CUDA 12.8+ / 13.x recommended

# 2) Intel oneAPI MKL (APT)
sudo mkdir -p /usr/share/keyrings
wget -qO- https://apt.repos.intel.com/oneapi/Intel-GPG-KEY-INTEL-PSR2.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneapi.list
sudo apt update
sudo apt install -y intel-oneapi-mkl intel-oneapi-mkl-devel intel-oneapi-openmp

# 3) Python env
python3 -m venv venv && source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 4) Build (MKL + CUDA fast-math + MMQ)
python autodevops.py --now --fast-math --force-mmq=on --blas mkl

# 5) Run-time env (OpenMP + MKL threads)
source /opt/intel/oneapi/setvars.sh --force
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export GGML_N_THREADS=8

# 6) Start the server using both GPUs
./bin/llama-server \
  --model ./models/llama-7b.gguf \
  --n-gpu-layers 999 \
  --tensor-split 50,50
```

Notes:
- --fast-math slightly relaxes FP math for speed (fine for inference).
- --n-gpu-layers 999 offloads as much as possible to GPUs.
- --tensor-split 50,50 splits weights across both 4090s (adjust per VRAM).

## Requirements

- Debian/Ubuntu with NVIDIA driver + CUDA Toolkit (12.8+ or 13.x)
- Build tools: git, cmake, make, gcc, g++, pkg-config
- Python 3.8+
- Intel oneAPI MKL (recommended for fastest CPU BLAS)

### Install Intel oneAPI MKL (APT)

```bash
sudo mkdir -p /usr/share/keyrings
wget -qO- https://apt.repos.intel.com/oneapi/Intel-GPG-KEY-INTEL-PSR2.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneapi.list
sudo apt update
sudo apt install -y intel-oneapi-mkl intel-oneapi-mkl-devel intel-oneapi-openmp
```

If APT complains about a missing public key, re‑fetch the PSR2 key above.

## Install & Build

```bash
# Python deps (Transformers used by the reranker path)
pip install -r requirements.txt

# Build latest llama.cpp with CUDA + MMQ + MKL
python autodevops.py --now --fast-math --force-mmq=on --blas mkl
```

What this does:
- Enables CUDA backend and MMQ kernels
- Detects your GPU arch (e.g., 8.9 for RTX 4090)
- Links Intel MKL when --blas mkl (auto-detected if already present)
- Builds all tools and symlinks the latest binaries into ./bin/

Verify MKL is linked:

```bash
ldd ./bin/llama-server | egrep 'mkl|iomp|omp'
```

## Run‑time Environment (Intel P+E cores)

For hybrid Intel CPUs (e.g., 13900KS), start here and tune:

```bash
# Make oneAPI libs available
source /opt/intel/oneapi/setvars.sh --force
# Threading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
# ggml CPU-side threads (tokenization, some ops)
export GGML_N_THREADS=8
```

Tip: Tune *_NUM_THREADS to 8–12 and pick what yields the best end‑to‑end tokens/sec without starving the GPUs (watch `nvidia-smi dmon -s pucmem`).

Optional wrapper script `env/run-intel-nvidia.sh`:

```bash
cat > env/run-intel-nvidia.sh <<'EOF'
#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export GGML_N_THREADS=8
exec "$@"
EOF
chmod +x env/run-intel-nvidia.sh
```

## Run the server

Single command:

```bash
./bin/llama-server \
  --model ./models/llama-7b.gguf \
  --n-gpu-layers 999 \
  --tensor-split 50,50
```

Or through the wrapper to ensure MKL/OpenMP env is set:

```bash
./env/run-intel-nvidia.sh ./bin/llama-server \
  --model ./models/llama-7b.gguf \
  --n-gpu-layers 999 \
  --tensor-split 50,50
```

Handy flags:
- --ctx-size 8192 (or higher if VRAM permits)
- --host 0.0.0.0 --port 8080 to listen on the network
- --parallel 4 for concurrent requests (tune per VRAM/latency)
- --mlock to reduce paging on Linux if you have enough RAM

## Model types & usage

1) LLM models (text generation)

```bash
# from Hugging Face
python loadmodel.py bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0 --llm
# local model
python loadmodel.py ./models/llama-7b.gguf --llm --local
```

2) Embedding models

```bash
# from Hugging Face
python loadmodel.py sentence-transformers/all-MiniLM-L6-v2-GGUF:Q8_0 --embed
# local model
python loadmodel.py ./models/bge-embedding-model.gguf --embed --local
```

3) Reranking models

```bash
# GGUF reranker
python loadmodel.py gpustack/bge-reranker-v2-m3-GGUF:Q8_0 --rerank

# Qwen3 reranker server via Transformers
python loadmodel.py QuantFactory/Qwen3-Reranker-4B-GGUF --rerank

# Run Qwen3 reranker locally via CLI
python reranker.py QuantFactory/Qwen3-Reranker-4B-GGUF \
  "What is machine learning?" \
  "ML is an AI subset" \
  "Deep learning uses neural nets"

# local reranker model
python loadmodel.py ./models/bge-reranker-v2-m3-Q8_0.gguf --rerank --local
```

Private models: set `HF_TOKEN` in `.env.local`. Downloads use `huggingface-cli` and are stored under `./models/`.

## API examples

Health check

```bash
curl -X GET http://127.0.0.1:8080/health
```

Chat completions

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b",
    "messages": [
      {"role": "user", "content": "What is artificial intelligence?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Text completions

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Embeddings

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": [
      "Artificial intelligence is transforming technology",
      "Machine learning enables pattern recognition"
    ]
  }'
```

Document reranking

```bash
# Basic
curl -X POST http://127.0.0.1:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "documents": [
      "Artificial intelligence (AI) is intelligence demonstrated by machines.",
      "Machine learning is a subset of artificial intelligence.",
      "Python is a programming language used for AI development.",
      "The weather is nice today."
    ]
  }'

# Multilingual
curl -X POST http://127.0.0.1:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "documents": [
      "Linear regression is a basic statistical method.",
      "Random forests are ensemble learning methods.",
      "Les algorithmes d'\''apprentissage automatique sont puissants.",
      "神经网络是深度学习的基础。"
    ]
  }'
```

## Updating

Running `python autodevops.py` without `--now` checks GitHub releases and schedules a build at 2 AM if a new version is found. Latest binaries are always symlinked into `./bin/`.

## Benchmarking & verification

```bash
# ggml/llama.cpp benchmark
./bin/llama-bench -m ./models/llama-7b.gguf \
  --n-gpu-layers 999 --tensor-split 50,50

# Watch GPU utilization
nvidia-smi dmon -s pucmem
```

## Troubleshooting

MKL libs not found at runtime

```bash
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib/intel64_lin:$LD_LIBRARY_PATH
```

Confirm MKL is linked

```bash
ldd ./bin/llama-server | egrep 'mkl|iomp|omp'
```

CUDA MMQ vs cuBLAS

```bash
# We default to MMQ (faster on many consumer GPUs). Use cuBLAS if you prefer:
export GGML_CUDA_FORCE_CUBLAS=1
./bin/llama-server ...
```

GCC warning about stringop-overflow in repack.cpp

Known benign compile warning in ggml’s CPU path; safe to ignore.

## Repository layout

- `autodevops.py` – automated build (CUDA + BLAS aware)
- `loadmodel.py` – server launcher + model download
- `requirements.txt` – Python dependencies
- `bin/` – symlinks to the latest built binaries
- `models/` – downloaded models
- `env/run-intel-nvidia.sh` – optional run‑time env wrapper

## Notes on performance

- With 2×4090, use `--tensor-split 50,50` (adjust per VRAM).
- Keep CPU threads modest (8–12) so GPU kernels stay fed.
- `--fast-math` adds a small throughput bump with tiny numerical diffs; recommended for inference.

README — Loader & Services (llama.cpp + Transformers)
This controls the three backend servers LiteLLM talks to. All live in /root/llama-cpp-server/ and are launched via systemd.

1) Layout
bash
Copy code
/root/llama-cpp-server/
  bin/llama-server             # built binary
  loadmodel.py                 # launcher (llm/embed/rerank)
  models/                      # downloaded GGUF files
  venv/                        # python venv
  run/
    run_qwen30b_llm.sh
    run_qwen_embed8b.sh
    run_qwen_reranker8b.sh
2) Run scripts (already created)
/root/llama-cpp-server/run/run_qwen30b_llm.sh

bash
Copy code
#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
exec python loadmodel.py --llm unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q5_K_XL \
  --host 0.0.0.0 --port 45540 \
  --n-gpu-layers 999 --tensor-split 50,50 \
  --ctx-size 16384 \
  --extra --threads 32
/root/llama-cpp-server/run/run_qwen_embed8b.sh

bash
Copy code
#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
exec python loadmodel.py --embed Qwen/Qwen3-Embedding-8B-GGUF:Q8_0 \
  --host 0.0.0.0 --port 45541 \
  --n-gpu-layers 999 --tensor-split 50,50 \
  --extra --ctx-size 6144 --threads 32
/root/llama-cpp-server/run/run_qwen_reranker8b.sh

bash
Copy code
#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python loadmodel.py --rerank Qwen/Qwen3-Reranker-8B \
  --host 0.0.0.0 --port 45542 \
  --device cuda --device-map auto \
  --dtype bf16 --quant 8bit \
  --doc-batch 64 --max-len 8192 \
  --max-memory 22GiB,22GiB \
  --trust-remote-code
These match the settings you’re successfully running now (2×4090, ctx sizes, etc.).

3) systemd units (already installed)
/etc/systemd/system/qwen30b-llm.service

ini
Copy code
[Unit]
Description=Qwen3 30B A3B (llama.cpp) on 45540
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/root/llama-cpp-server
ExecStart=/bin/bash -lc '/root/llama-cpp-server/run/run_qwen30b_llm.sh'
Restart=always
RestartSec=2
User=root
Environment=HF_HOME=/root/.cache/huggingface
Environment=CUDA_VISIBLE_DEVICES=0,1
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
/etc/systemd/system/qwen-embed-8b.service

ini
Copy code
[Unit]
Description=Qwen3 Embedding 8B (llama.cpp) on 45541
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/root/llama-cpp-server
ExecStart=/bin/bash -lc '/root/llama-cpp-server/run/run_qwen_embed8b.sh'
Restart=always
RestartSec=2
User=root
Environment=HF_HOME=/root/.cache/huggingface
Environment=CUDA_VISIBLE_DEVICES=0,1
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
/etc/systemd/system/qwen-reranker-8b.service

ini
Copy code
[Unit]
Description=Qwen3 Reranker 8B (Transformers 8-bit) on 45542
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/root/llama-cpp-server
ExecStart=/bin/bash -lc '/root/llama-cpp-server/run/run_qwen_reranker8b.sh'
Restart=always
RestartSec=2
User=root
Environment=HF_HOME=/root/.cache/huggingface
Environment=CUDA_VISIBLE_DEVICES=0,1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
Enable/start (already done):

bash
Copy code
systemctl daemon-reload
systemctl enable --now qwen30b-llm qwen-embed-8b qwen-reranker-8b
4) Check status & logs
bash
Copy code
systemctl --no-pager --full status qwen30b-llm
systemctl --no-pager --full status qwen-embed-8b
systemctl --no-pager --full status qwen-reranker-8b

journalctl -u qwen30b-llm -e --no-pager
journalctl -u qwen-embed-8b -e --no-pager
journalctl -u qwen-reranker-8b -e --no-pager
5) Direct backend health (bypassing LiteLLM)
bash
Copy code
# LLM
curl -s http://192.168.100.199:45540/health

# Embeddings
curl -s http://192.168.100.199:45541/health

# Reranker
curl -s http://192.168.100.199:45542/health
6) Notes
Embeddings dim from Qwen3‑Embedding‑8B GGUF is fixed at 4096 (you verified via /v1/embeddings length).

Reranker response has been patched in loadmodel.py to Jina/Cohere shape (relevance_score, document: {text: ...}); LiteLLM now validates it cleanly.

For even lower VRAM on reranker: switch --quant 8bit → 4bit (you’ll sacrifice some speed but save VRAM).

If you want these as separate files, I can spit them out as README.litellm.md and README.backends.md. Want me to generate them to disk right now?








Sources

Ask ChatGPT


