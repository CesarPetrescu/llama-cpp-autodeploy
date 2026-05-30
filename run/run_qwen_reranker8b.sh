#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3 Reranker 4B GGUF - llama.cpp reranker server
# Listens on 0.0.0.0:45542
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_SPLIT="${TENSOR_SPLIT:-50,50}"
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL="${PARALLEL:-1}"

cmd=(python loadmodel.py --rerank 'DevQuasar/Qwen.Qwen3-Reranker-4B-GGUF:Qwen.Qwen3-Reranker-4B.Q6_K.gguf'
  --host 0.0.0.0 \
  --port 45542 \
  --n-gpu-layers 999 \
  --ctx-size "${CTX_SIZE}")

if [[ -n "${TENSOR_SPLIT}" ]]; then
  cmd+=(--tensor-split "${TENSOR_SPLIT}")
fi

cmd+=(--extra --parallel "${PARALLEL}" --threads 32)

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" exec "${cmd[@]}"
