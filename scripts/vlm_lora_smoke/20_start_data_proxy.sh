#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cd "$REPO_ROOT"

PID_FILE="$PID_DIR/data_proxy.pid"
LOG_FILE="$LOG_DIR/data_proxy.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[data-proxy] already running pid=$(cat "$PID_FILE")"
  exit 0
fi

echo "[data-proxy] starting at $DATA_PROXY_BASE_URL -> $SGLANG_BASE_URL"
echo "[data-proxy] LoRA: ${LORA_NAME}-v${LORA_VERSION}"
echo "[data-proxy] log: $LOG_FILE"

nohup env \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  PYTORCH_KERNEL_CACHE_PATH="$PYTORCH_KERNEL_CACHE_PATH" \
  PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
  uv run python -m areal.v2.inference_service.data_proxy \
    --host "$DATA_PROXY_HOST" \
    --port "$DATA_PROXY_PORT" \
    --backend-addr "$SGLANG_BASE_URL" \
    --backend-type sglang \
    --tokenizer-path "$MODEL_PATH" \
    --log-level info \
    --request-timeout 180 \
    --engine-max-tokens "$SGLANG_CONTEXT_LENGTH" \
    --use-lora \
    --lora-name "$LORA_NAME" \
  >"$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "[data-proxy] pid=$(cat "$PID_FILE")"

for _ in $(seq 1 90); do
  if curl -fsS "$DATA_PROXY_BASE_URL/health" >/dev/null 2>&1; then
    echo "[data-proxy] ready"
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[data-proxy] exited early; tailing log" >&2
    tail -n 80 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 1
done

echo "[data-proxy] did not become ready; tailing log" >&2
tail -n 120 "$LOG_FILE" >&2
exit 1
