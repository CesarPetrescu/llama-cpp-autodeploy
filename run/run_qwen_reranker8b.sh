#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3 Reranker 4B GGUF – llama.cpp reranker server
# Listens on 0.0.0.0:45542
TENSOR_SPLIT="${TENSOR_SPLIT:-50,50}"
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL="${PARALLEL:-1}"

CUDA_VISIBLE_DEVICES=0,1 \
exec python loadmodel.py --rerank 'DevQuasar/Qwen.Qwen3-Reranker-4B-GGUF:Qwen.Qwen3-Reranker-4B.Q6_K.gguf' \
  --host 0.0.0.0 \
  --port 45542 \
  --n-gpu-layers 999 \
  --tensor-split "${TENSOR_SPLIT}" \
  --ctx-size "${CTX_SIZE}" \
  --extra --parallel "${PARALLEL}" --threads 32
