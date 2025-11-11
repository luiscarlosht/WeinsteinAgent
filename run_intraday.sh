#!/usr/bin/env bash
# ============================================================
# run_intraday.sh – Launches Weinstein Intraday Watcher
# Also writes intraday_debug.csv/json for Signal Engine
# ------------------------------------------------------------
# Example cron (ET every 10 min during session):
# */10 9-16 * * 1-5 /bin/bash -lc 'cd ~/WeinsteinAgent && ./run_intraday.sh'
# ============================================================

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"
OUTDIR="./output"
DEBUG_CSV="${OUTDIR}/intraday_debug.csv"
DEBUG_JSON="${OUTDIR}/intraday_debug.json"

[[ -r "$CONFIG_PATH" ]] || { red "Config not found: $CONFIG_PATH"; exit 2; }
[[ -d ".venv" ]] && source .venv/bin/activate 2>/dev/null || true
mkdir -p "$OUTDIR"

PY_SCRIPT="weinstein_intraday_watcher.py"
[[ -f "$PY_SCRIPT" ]] || { red "Cannot find $PY_SCRIPT in repo root."; exit 2; }

bold "⚡ Intraday watcher using config: $CONFIG_PATH"
yellow "• Running: python3 $PY_SCRIPT --config $CONFIG_PATH"

# Add --log-csv / --log-json so tools can consume the snapshot
python3 "$PY_SCRIPT" \
  --config "$CONFIG_PATH" \
  --log-csv "$DEBUG_CSV" \
  --log-json "$DEBUG_JSON" \
  "$@" || { red "❌ Intraday watcher error."; exit 1; }

green "✅ Intraday tick complete."
