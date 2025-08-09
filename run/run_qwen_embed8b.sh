#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3 Embedding 8B (GGUF) – llama.cpp embeddings server
# Listens on 0.0.0.0:45541
CUDA_VISIBLE_DEVICES=0,1 \
exec python loadmodel.py --embed Qwen/Qwen3-Embedding-8B-GGUF:Q8_0 \
  --host 0.0.0.0 \
  --port 45541 \
  --n-gpu-layers 999 \
  --tensor-split 50,50 \
  --extra --ctx-size 6144 --threads 32
