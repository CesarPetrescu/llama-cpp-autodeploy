#!/usr/bin/env bash
set -euo pipefail

cd /root/llama-cpp-server/llama-builds/llama-cpp-ggml-mtp

MODEL_PATH="${MODEL_PATH:-/root/llama-cpp-server/models/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
MMPROJ="${MMPROJ:-/root/llama-cpp-server/models/mmproj-BF16.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-45540}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_SPLIT="${TENSOR_SPLIT:-35,65}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q8_0}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.00}"
SPEC_DRAFT_KV_TYPE="${SPEC_DRAFT_KV_TYPE:-q8_0}"
IMAGE_TOKENS="${IMAGE_TOKENS:-1024}"
THREADS="${THREADS:-32}"

cmd=(/root/llama-cpp-server/llama-builds/llama-cpp-ggml-mtp/build/bin/llama-server
  -m "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  -ngl "${N_GPU_LAYERS}"
  -c "${CTX_SIZE}"
  -fa on
  -np 1
  --mmproj "${MMPROJ}"
  --no-mmproj-offload
  --image-min-tokens "${IMAGE_TOKENS}"
  --image-max-tokens "${IMAGE_TOKENS}"
  -ctk "${KV_CACHE_TYPE}"
  -ctv "${KV_CACHE_TYPE}"
  --spec-type draft-mtp
  --spec-draft-n-max "${SPEC_DRAFT_N_MAX}"
  --spec-draft-n-min "${SPEC_DRAFT_N_MIN}"
  --spec-draft-p-min "${SPEC_DRAFT_P_MIN}"
  --spec-draft-type-k "${SPEC_DRAFT_KV_TYPE}"
  --spec-draft-type-v "${SPEC_DRAFT_KV_TYPE}"
  --cache-prompt
  --metrics
  --threads "${THREADS}")

if [[ -n "${TENSOR_SPLIT}" ]]; then
  cmd+=(--tensor-split "${TENSOR_SPLIT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" exec "${cmd[@]}"
