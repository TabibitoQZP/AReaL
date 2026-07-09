#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

VERSIONED_LORA_NAME="${LORA_NAME}-v${LORA_VERSION}"

if [[ ! -f "$LORA_ADAPTER_DIR/adapter_config.json" ]]; then
  echo "[load-lora] missing adapter_config.json under $LORA_ADAPTER_DIR" >&2
  echo "[load-lora] run scripts/vlm_lora_smoke/00_make_adapter.sh first" >&2
  exit 1
fi

echo "[load-lora] loading $VERSIONED_LORA_NAME from $LORA_ADAPTER_DIR"
curl -fsS \
  -X POST "$SGLANG_BASE_URL/load_lora_adapter" \
  -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"${VERSIONED_LORA_NAME}\",\"lora_path\":\"${LORA_ADAPTER_DIR}\"}"
echo

echo "[load-lora] setting data proxy version to $LORA_VERSION"
curl -fsS \
  -X POST "$DATA_PROXY_BASE_URL/set_version" \
  -H "Content-Type: application/json" \
  -d "{\"version\":${LORA_VERSION}}"
echo

echo "[load-lora] done"
