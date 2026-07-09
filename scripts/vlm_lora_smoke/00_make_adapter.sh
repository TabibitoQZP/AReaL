#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cd "$REPO_ROOT"

echo "[make-adapter] model: $MODEL_PATH"
echo "[make-adapter] output: $LORA_ADAPTER_DIR"

uv run python "$SCRIPT_DIR/make_adapter.py" \
  --model-path "$MODEL_PATH" \
  --output-dir "$LORA_ADAPTER_DIR" \
  --rank "$LORA_RANK" \
  --alpha "$LORA_ALPHA" \
  --target-modules "$LORA_TARGET_MODULES"

echo "[make-adapter] done"
