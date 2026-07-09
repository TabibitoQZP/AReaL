#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cd "$REPO_ROOT"

PID_FILE="$PID_DIR/sglang.pid"
LOG_FILE="$LOG_DIR/sglang.log"
IFS=',' read -r -a LORA_TARGET_MODULE_ARGS <<< "$LORA_TARGET_MODULES"

if (( SGLANG_MAX_LOADED_LORAS < SGLANG_MAX_LORAS_PER_BATCH )); then
  echo "[sglang] SGLANG_MAX_LOADED_LORAS must be >= SGLANG_MAX_LORAS_PER_BATCH" >&2
  echo "[sglang] got loaded=$SGLANG_MAX_LOADED_LORAS per_batch=$SGLANG_MAX_LORAS_PER_BATCH" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[sglang] already running pid=$(cat "$PID_FILE")"
  exit 0
fi

echo "[sglang] starting on GPU(s) $CUDA_VISIBLE_DEVICES at $SGLANG_BASE_URL"
echo "[sglang] log: $LOG_FILE"

nohup env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  PYTORCH_KERNEL_CACHE_PATH="$PYTORCH_KERNEL_CACHE_PATH" \
  PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
  uv run python -m areal.v2.inference_service.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    --dtype bfloat16 \
    --mem-fraction-static "$SGLANG_MEM_FRACTION" \
    --context-length "$SGLANG_CONTEXT_LENGTH" \
    --enable-multimodal \
    --enable-lora \
    --max-lora-rank "$LORA_RANK" \
    --lora-target-modules "${LORA_TARGET_MODULE_ARGS[@]}" \
    --max-loras-per-batch "$SGLANG_MAX_LORAS_PER_BATCH" \
    --max-loaded-loras "$SGLANG_MAX_LOADED_LORAS" \
    --random-seed 1 \
  >"$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "[sglang] pid=$(cat "$PID_FILE")"

for _ in $(seq 1 180); do
  if curl -fsS "$SGLANG_BASE_URL/model_info" >/dev/null 2>&1; then
    echo "[sglang] ready"
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[sglang] exited early; tailing log" >&2
    tail -n 80 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 2
done

echo "[sglang] did not become ready; tailing log" >&2
tail -n 120 "$LOG_FILE" >&2
exit 1
