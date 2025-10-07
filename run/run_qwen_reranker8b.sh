#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3 Reranker 8B – Transformers + bitsandbytes 8-bit
# Listens on 0.0.0.0:45542
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python loadmodel.py --rerank Qwen/Qwen3-Reranker-4B \
  --host 0.0.0.0 \
  --port 45542 \
  --device cuda \
  --device-map auto \
  --dtype bf16 \
  --quant 8bit \
  --doc-batch 1 \
  --max-len 8192 \
  --max-memory "4GiB,4GiB,cpu=48GiB" \
  --trust-remote-code
