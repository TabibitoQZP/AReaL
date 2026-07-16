#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

usage() {
  cat <<'EOF'
Run the Geometry3K v2 VLM LoRA example in the background.

Environment overrides:
  CUDA_VISIBLE_DEVICES   GPU list to expose to the local scheduler (default: 5,6)
  MODEL_PATH             Local or HF model path
  RUN_ROOT               Root for logs, pids, experiments, and name_resolve
  CONFIG                 Example YAML path
  EXPERIMENT_ROOT        AReaL experiment output root
  NAME_RESOLVE_ROOT      AReaL name-resolve root
  SGLANG_MEM_FRACTION_STATIC
                         Optional override for sglang.mem_fraction_static

Examples:
  bash scripts/vlm_lora_example/run_geometry3k_v2_lora.sh
  CUDA_VISIBLE_DEVICES=6,7 bash scripts/vlm_lora_example/run_geometry3k_v2_lora.sh
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

RUN_ROOT=${RUN_ROOT:-/mnt/data/zipengqiu/areal/vlm_lora_example}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PID_DIR=${PID_DIR:-$RUN_ROOT/pids}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$RUN_ROOT/experiments}
NAME_RESOLVE_ROOT=${NAME_RESOLVE_ROOT:-$RUN_ROOT/name_resolve}

CONFIG=${CONFIG:-examples/vlm/geometry3k_grpo_v2_lora.yaml}
MODEL_PATH=${MODEL_PATH:-/mnt/data/zipengqiu/areal/models/Qwen2.5-VL-3B-Instruct}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6}
PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

mkdir -p "$LOG_DIR" "$PID_DIR" "$EXPERIMENT_ROOT" "$NAME_RESOLVE_ROOT"

PID_FILE="$PID_DIR/geometry3k_v2_lora.pid"
STATUS_FILE="$PID_DIR/geometry3k_v2_lora.status"
EXIT_FILE="$PID_DIR/geometry3k_v2_lora.exitcode"
LATEST_LOG="$LOG_DIR/geometry3k_v2_lora.latest.log"
CMD_FILE="$LOG_DIR/geometry3k_v2_lora.latest.cmd"
RUNNER_FILE="$LOG_DIR/geometry3k_v2_lora.latest.runner.sh"
LOG_FILE="$LOG_DIR/geometry3k_v2_lora.$(date +%Y%m%d_%H%M%S).log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[geometry3k-v2-lora] already running pid=$(cat "$PID_FILE")"
  echo "[geometry3k-v2-lora] log: $LATEST_LOG"
  exit 0
fi

cd "$REPO_ROOT"

cmd=(
  uv run python examples/vlm/geometry3k_grpo_v2_lora.py
  --config "$CONFIG"
  actor.path="$MODEL_PATH"
  cluster.fileroot="$EXPERIMENT_ROOT"
  cluster.name_resolve.nfs_record_root="$NAME_RESOLVE_ROOT"
)

if [[ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]]; then
  cmd+=(sglang.mem_fraction_static="$SGLANG_MEM_FRACTION_STATIC")
fi

{
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf 'PYTHONUNBUFFERED=%q ' "$PYTHONUNBUFFERED"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$CMD_FILE"

ln -sfn "$LOG_FILE" "$LATEST_LOG"
rm -f "$EXIT_FILE"
echo "starting" > "$STATUS_FILE"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set +e\n'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'echo running > %q\n' "$STATUS_FILE"
  printf 'echo "[geometry3k-v2-lora] started at $(date -Is)"\n'
  printf 'env '
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf 'PYTHONUNBUFFERED=%q ' "$PYTHONUNBUFFERED"
  printf 'HF_HOME=%q ' "${HF_HOME:-}"
  printf 'HF_HUB_CACHE=%q ' "${HF_HUB_CACHE:-}"
  printf 'HF_DATASETS_CACHE=%q ' "${HF_DATASETS_CACHE:-}"
  printf 'TRITON_CACHE_DIR=%q ' "${TRITON_CACHE_DIR:-}"
  printf 'VLLM_CACHE_ROOT=%q ' "${VLLM_CACHE_ROOT:-}"
  printf 'PYTORCH_KERNEL_CACHE_PATH=%q ' "${PYTORCH_KERNEL_CACHE_PATH:-}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  printf 'rc=$?\n'
  printf 'echo "$rc" > %q\n' "$EXIT_FILE"
  printf 'if [[ "$rc" -eq 0 ]]; then echo succeeded > %q; else echo failed > %q; fi\n' "$STATUS_FILE" "$STATUS_FILE"
  printf 'echo "[geometry3k-v2-lora] finished at $(date -Is) rc=$rc"\n'
  printf 'exit "$rc"\n'
} > "$RUNNER_FILE"
chmod +x "$RUNNER_FILE"

echo "[geometry3k-v2-lora] starting"
echo "[geometry3k-v2-lora] gpus: $CUDA_VISIBLE_DEVICES"
echo "[geometry3k-v2-lora] config: $CONFIG"
echo "[geometry3k-v2-lora] model: $MODEL_PATH"
echo "[geometry3k-v2-lora] experiment root: $EXPERIMENT_ROOT"
echo "[geometry3k-v2-lora] name resolve root: $NAME_RESOLVE_ROOT"
echo "[geometry3k-v2-lora] log: $LOG_FILE"
echo "[geometry3k-v2-lora] cmd: $CMD_FILE"
echo "[geometry3k-v2-lora] status: $STATUS_FILE"

if command -v setsid >/dev/null 2>&1; then
  nohup setsid "$RUNNER_FILE" >"$LOG_FILE" 2>&1 &
else
  nohup "$RUNNER_FILE" >"$LOG_FILE" 2>&1 &
fi

echo "$!" > "$PID_FILE"
echo "[geometry3k-v2-lora] pid=$(cat "$PID_FILE")"
echo "[geometry3k-v2-lora] tail:"
echo "tail -f $LATEST_LOG"
echo "[geometry3k-v2-lora] status:"
echo "bash scripts/vlm_lora_example/status_geometry3k_v2_lora.sh"
