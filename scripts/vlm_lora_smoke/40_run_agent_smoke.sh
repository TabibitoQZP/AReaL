#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cd "$REPO_ROOT"

echo "[agent-smoke] data proxy: $DATA_PROXY_BASE_URL"
echo "[agent-smoke] expected LoRA adapter: ${LORA_NAME}-v${LORA_VERSION}"
uv run python "$SCRIPT_DIR/run_agent_smoke.py"
