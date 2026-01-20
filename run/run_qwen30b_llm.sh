#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3-30B A3B Thinking (Unsloth) – llama.cpp server
# Listens on 0.0.0.0:45540 so LiteLLM (on another host) can reach it.
N_CPU_MOE="${N_CPU_MOE:-}"  # Set to a number to offload experts for the first N MoE layers
EXTRA_FLAGS=(--threads 32)
if [[ -n "${N_CPU_MOE}" ]]; then
  EXTRA_FLAGS+=(--n-cpu-moe "${N_CPU_MOE}")
fi

CUDA_VISIBLE_DEVICES=0,1 \
exec python loadmodel.py --llm unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q5_K_XL \
  --host 0.0.0.0 \
  --port 45540 \
  --n-gpu-layers 999 \
  --tensor-split 50,50 \
  --ctx-size 16384 \
  --extra "${EXTRA_FLAGS[@]}"
