#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/mnt/data/zipengqiu/areal/vlm_lora_example}
PID_FILE=${PID_FILE:-$RUN_ROOT/pids/geometry3k_v2_lora.pid}
STATUS_FILE=${STATUS_FILE:-$RUN_ROOT/pids/geometry3k_v2_lora.status}
EXIT_FILE=${EXIT_FILE:-$RUN_ROOT/pids/geometry3k_v2_lora.exitcode}
LATEST_LOG=${LATEST_LOG:-$RUN_ROOT/logs/geometry3k_v2_lora.latest.log}
CMD_FILE=${CMD_FILE:-$RUN_ROOT/logs/geometry3k_v2_lora.latest.cmd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-geometry3k-grpo-v2-lora}
SCRIPT_NAME=${SCRIPT_NAME:-geometry3k_grpo_v2_lora.py}

collect_related_pids() {
  ps -u "$USER" -o pid=,args= | while read -r proc_pid proc_args; do
    [[ -z "${proc_pid:-}" ]] && continue
    [[ "$proc_pid" == "$$" ]] && continue
    if [[ "$proc_args" == *"$EXPERIMENT_NAME"* ]] ||
      [[ "$proc_args" == *"$SCRIPT_NAME"* ]]; then
      echo "$proc_pid"
    fi
  done | awk 'NF' | sort -u
}

status="unknown"
if [[ -f "$STATUS_FILE" ]]; then
  status=$(cat "$STATUS_FILE")
fi

echo "[geometry3k-v2-lora] status: $status"

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "[geometry3k-v2-lora] pid: $pid (running)"
    ps -p "$pid" -o pid,ppid,etime,stat,cmd
  else
    echo "[geometry3k-v2-lora] pid: $pid (not running)"
  fi
else
  echo "[geometry3k-v2-lora] pid: none"
fi

related=$(collect_related_pids)
if [[ -n "$related" ]]; then
  echo "[geometry3k-v2-lora] related processes:"
  ps -p "$(echo "$related" | paste -sd, -)" -o pid,ppid,etime,stat,cmd || true
else
  echo "[geometry3k-v2-lora] related processes: none"
fi

if [[ -f "$EXIT_FILE" ]]; then
  echo "[geometry3k-v2-lora] exit code: $(cat "$EXIT_FILE")"
fi

if [[ -f "$CMD_FILE" ]]; then
  echo "[geometry3k-v2-lora] cmd: $CMD_FILE"
fi

if [[ -f "$LATEST_LOG" ]]; then
  echo "[geometry3k-v2-lora] log: $LATEST_LOG"
  if tail -n 200 "$LATEST_LOG" | grep -E "Traceback|RuntimeError|AssertionError|ERROR" >/dev/null; then
    echo "[geometry3k-v2-lora] recent error markers:"
    tail -n 200 "$LATEST_LOG" | grep -E "Traceback|RuntimeError|AssertionError|ERROR" | tail -n 10
  fi
  echo "[geometry3k-v2-lora] last 40 log lines:"
  tail -n 40 "$LATEST_LOG"
else
  echo "[geometry3k-v2-lora] log: none"
fi
