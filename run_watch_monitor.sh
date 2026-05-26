#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/luiscarlosht/WeinsteinAgent"
cd "$PROJECT_DIR"

OUT_DIR="${OUT_DIR:-output/watch_monitor}"
SEND_EMAIL="${SEND_EMAIL:-0}"

ARGS=(
  --strict-debug "output/intraday_debug.csv"
  --validation-debug "output/intraday_debug_validation.csv"
  --out-dir "$OUT_DIR"
)

if [[ "$SEND_EMAIL" == "1" ]]; then
  ARGS+=(--send-email)
fi

python3 weinstein_watch_monitor.py "${ARGS[@]}"
