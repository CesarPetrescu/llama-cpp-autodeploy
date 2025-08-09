#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3-30B A3B Thinking (Unsloth) – llama.cpp server
# Listens on 0.0.0.0:45540 so LiteLLM (on another host) can reach it.
CUDA_VISIBLE_DEVICES=0,1 \
exec python loadmodel.py --llm unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q5_K_XL \
  --host 0.0.0.0 \
  --port 45540 \
  --n-gpu-layers 999 \
  --tensor-split 50,50 \
  --ctx-size 16384 \
  --extra --threads 32
