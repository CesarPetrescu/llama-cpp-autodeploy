#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3 Embedding 8B (GGUF) - llama.cpp embeddings server
# Listens on 0.0.0.0:45541
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_SPLIT="${TENSOR_SPLIT:-}"
PARALLEL="${PARALLEL:-1}"

cmd=(python loadmodel.py --embed Qwen/Qwen3-Embedding-8B-GGUF:Q8_0
  --host 0.0.0.0 \
  --port 45541 \
  --n-gpu-layers 999)

if [[ -n "${TENSOR_SPLIT}" ]]; then
  cmd+=(--tensor-split "${TENSOR_SPLIT}")
fi

cmd+=(--extra --ctx-size 6144 --parallel "${PARALLEL}" --threads 32)

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" exec "${cmd[@]}"
