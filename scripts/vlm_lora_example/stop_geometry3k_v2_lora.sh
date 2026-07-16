#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/mnt/data/zipengqiu/areal/vlm_lora_example}
PID_FILE=${PID_FILE:-$RUN_ROOT/pids/geometry3k_v2_lora.pid}
STATUS_FILE=${STATUS_FILE:-$RUN_ROOT/pids/geometry3k_v2_lora.status}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-geometry3k-grpo-v2-lora}
SCRIPT_NAME=${SCRIPT_NAME:-geometry3k_grpo_v2_lora.py}

collect_related_pids() {
  {
    if [[ -f "$PID_FILE" ]]; then
      pid=$(cat "$PID_FILE")
      if kill -0 "$pid" 2>/dev/null; then
        echo "$pid"
      fi
    fi
    ps -u "$USER" -o pid=,args= | while read -r proc_pid proc_args; do
      [[ -z "${proc_pid:-}" ]] && continue
      [[ "$proc_pid" == "$$" ]] && continue
      if [[ "$proc_args" == *"$EXPERIMENT_NAME"* ]] ||
        [[ "$proc_args" == *"$SCRIPT_NAME"* ]]; then
        echo "$proc_pid"
      fi
    done
  } | awk 'NF' | sort -u
}

mkdir -p "$(dirname "$STATUS_FILE")"

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "[geometry3k-v2-lora] stopping pid=$pid"
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  else
    echo "[geometry3k-v2-lora] pid not running: $pid"
  fi
else
  echo "[geometry3k-v2-lora] no pid file: $PID_FILE"
fi

sleep 1

related=$(collect_related_pids)
if [[ -n "$related" ]]; then
  echo "[geometry3k-v2-lora] stopping related processes:"
  ps -p "$(echo "$related" | paste -sd, -)" -o pid,ppid,etime,stat,cmd || true
  echo "$related" | xargs -r kill 2>/dev/null || true
fi

for _ in $(seq 1 30); do
  remaining=$(collect_related_pids)
  if [[ -z "$remaining" ]]; then
    rm -f "$PID_FILE"
    echo "stopped" > "$STATUS_FILE"
    echo "[geometry3k-v2-lora] stopped"
    exit 0
  fi
  sleep 1
done

remaining=$(collect_related_pids)
if [[ -z "$remaining" ]]; then
  rm -f "$PID_FILE"
  echo "stopped" > "$STATUS_FILE"
  echo "[geometry3k-v2-lora] stopped"
  exit 0
fi

echo "[geometry3k-v2-lora] still running; sending SIGKILL"
ps -p "$(echo "$remaining" | paste -sd, -)" -o pid,ppid,etime,stat,cmd || true
echo "$remaining" | xargs -r kill -9 2>/dev/null || true
rm -f "$PID_FILE"
echo "stopped" > "$STATUS_FILE"
echo "[geometry3k-v2-lora] stopped"
