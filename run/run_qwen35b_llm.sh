#!/usr/bin/env bash
set -euo pipefail
cd /root/llama-cpp-server
source venv/bin/activate

# Qwen3.5-35B A3B (Unsloth) – llama.cpp server
# Listens on 0.0.0.0:45540 so LiteLLM (on another host) can reach it.
#
# Vision/image support:
#   Set MMPROJ to the local path of the multimodal projector GGUF to enable image input.
#   Download it first:
#     huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF --include "*mmproj*" \
#       --local-dir /root/llama-cpp-server/models
#   Then set: MMPROJ=/root/llama-cpp-server/models/mmproj-Qwen3.5-35B-A3B-F16.gguf
#
# Reasoning toggle (per-request via API):
#   With --jinja + --reasoning-format deepseek enabled below, clients can toggle thinking
#   per request by passing:  extra_body={"chat_template_kwargs": {"enable_thinking": False}}
#
N_CPU_MOE="${N_CPU_MOE:-}"  # Set to a number to offload experts for the first N MoE layers
MMPROJ="${MMPROJ:-}"        # Optional: path to mmproj GGUF for vision/image input

EXTRA_FLAGS=(--threads 32)
if [[ -n "${N_CPU_MOE}" ]]; then
  EXTRA_FLAGS+=(--n-cpu-moe "${N_CPU_MOE}")
fi

MMPROJ_FLAGS=()
if [[ -n "${MMPROJ}" ]]; then
  MMPROJ_FLAGS=(--mmproj "${MMPROJ}")
fi

CUDA_VISIBLE_DEVICES=0,1 \
exec python loadmodel.py --llm unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_XL \
  --host 0.0.0.0 \
  --port 45540 \
  --n-gpu-layers 999 \
  --tensor-split 50,50 \
  --ctx-size 24576 \
  --jinja \
  --reasoning-format deepseek \
  --no-context-shift \
  "${MMPROJ_FLAGS[@]}" \
  --extra "${EXTRA_FLAGS[@]}"
