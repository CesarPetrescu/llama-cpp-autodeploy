#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3 Embedding 8B (GGUF) – llama.cpp embeddings server
# Listens on 0.0.0.0:45541
TENSOR_SPLIT="${TENSOR_SPLIT:-35,65}"
PARALLEL="${PARALLEL:-1}"
CUDA_VISIBLE_DEVICES=0,1 \
exec python loadmodel.py --embed Qwen/Qwen3-Embedding-8B-GGUF:Q8_0 \
  --host 0.0.0.0 \
  --port 45541 \
  --n-gpu-layers 999 \
  --tensor-split "${TENSOR_SPLIT}" \
  --extra --ctx-size 6144 --parallel "${PARALLEL}" --threads 32
