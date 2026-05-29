#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3.6-35B A3B (Unsloth) – llama.cpp server
# Listens on 0.0.0.0:45540 so LiteLLM (on another host) can reach it.
N_CPU_MOE="${N_CPU_MOE:-}"  # Set to a number to offload experts for the first N MoE layers
TENSOR_SPLIT="${TENSOR_SPLIT:-46,54}"
PARALLEL="${PARALLEL:-1}"
IMAGE_TOKENS="${IMAGE_TOKENS:-1024}"
EXTRA_FLAGS=(
  --threads 32
  --parallel "${PARALLEL}"
  --image-min-tokens "${IMAGE_TOKENS}"
  --image-max-tokens "${IMAGE_TOKENS}"
)
if [[ -n "${N_CPU_MOE}" ]]; then
  EXTRA_FLAGS+=(--n-cpu-moe "${N_CPU_MOE}")
fi

MMPROJ="${MMPROJ:-/root/llama-cpp-server/models/mmproj-F32-3.6.gguf}"
CUDA_VISIBLE_DEVICES=0,1 \
exec python loadmodel.py --llm unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_XL \
  --host 0.0.0.0 \
  --port 45540 \
  --n-gpu-layers 999 \
  --tensor-split "${TENSOR_SPLIT}" \
  --ctx-size 22000 \
  --mmproj "${MMPROJ}" \
  --jinja \
  --reasoning-format deepseek \
  --no-context-shift \
  --extra "${EXTRA_FLAGS[@]}"
