#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

stop_one() {
  local name=$1
  local pid_file=$2

  if [[ ! -f "$pid_file" ]]; then
    echo "[$name] no pid file"
    return
  fi

  local pid
  pid=$(cat "$pid_file")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[$name] not running"
    rm -f "$pid_file"
    return
  fi

  echo "[$name] stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "[$name] stopped"
      return
    fi
    sleep 1
  done

  echo "[$name] still running; sending SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pid_file"
}

stop_one "data-proxy" "$PID_DIR/data_proxy.pid"
stop_one "sglang" "$PID_DIR/sglang.pid"
