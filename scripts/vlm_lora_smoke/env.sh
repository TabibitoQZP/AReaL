#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export SMOKE_ROOT=${SMOKE_ROOT:-/mnt/data/zipengqiu/areal/vlm_lora_smoke}
export MODEL_PATH=${MODEL_PATH:-/mnt/data/zipengqiu/areal/models/Qwen2.5-VL-3B-Instruct}

export LORA_NAME=${LORA_NAME:-vlm-smoke-lora}
export LORA_VERSION=${LORA_VERSION:-0}
export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
export LORA_ADAPTER_DIR=${LORA_ADAPTER_DIR:-$SMOKE_ROOT/adapters/${LORA_NAME}-v${LORA_VERSION}}

export SGLANG_HOST=${SGLANG_HOST:-127.0.0.1}
export SGLANG_PORT=${SGLANG_PORT:-30005}
export SGLANG_BASE_URL=${SGLANG_BASE_URL:-http://${SGLANG_HOST}:${SGLANG_PORT}}
export SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.60}
export SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-8192}
export SGLANG_MAX_LORAS_PER_BATCH=${SGLANG_MAX_LORAS_PER_BATCH:-8}
export SGLANG_MAX_LOADED_LORAS=${SGLANG_MAX_LOADED_LORAS:-$SGLANG_MAX_LORAS_PER_BATCH}

export DATA_PROXY_HOST=${DATA_PROXY_HOST:-127.0.0.1}
export DATA_PROXY_PORT=${DATA_PROXY_PORT:-8085}
export DATA_PROXY_BASE_URL=${DATA_PROXY_BASE_URL:-http://${DATA_PROXY_HOST}:${DATA_PROXY_PORT}}
export ADMIN_API_KEY=${ADMIN_API_KEY:-areal-admin-key}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export HF_HOME=${HF_HOME:-/mnt/data/zipengqiu/areal/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/data/zipengqiu/areal/huggingface/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/data/zipengqiu/areal/datasets}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/mnt/data/zipengqiu/areal/cache/triton}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/mnt/data/zipengqiu/areal/cache/vllm}
export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-/mnt/data/zipengqiu/areal/cache/torch/kernels}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

export LOG_DIR=$SMOKE_ROOT/logs
export PID_DIR=$SMOKE_ROOT/pids
export TMP_DIR=$SMOKE_ROOT/tmp

mkdir -p "$SMOKE_ROOT" "$SMOKE_ROOT/adapters" "$LOG_DIR" "$PID_DIR" "$TMP_DIR"
