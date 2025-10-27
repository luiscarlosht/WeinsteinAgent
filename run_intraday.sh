#!/usr/bin/env bash
# ============================================================
# run_intraday.sh – Launches Weinstein Intraday Watcher
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

if [[ ! -r "$CONFIG_PATH" ]]; then
  red "Config file not found or unreadable: $CONFIG_PATH"
  red "Set CONFIG_FILE=./config.yaml or create ./config.yaml"
  exit 2
fi

# ------------------------------------------------------------
# Activate virtual environment if it exists
# ------------------------------------------------------------
if [[ -d ".venv" ]]; then
  source .venv/bin/activate 2>/dev/null || true
fi

bold "⚡ Intraday watcher using config: $CONFIG_PATH"

# ------------------------------------------------------------
# Hard-wire the correct intraday Python script
# ------------------------------------------------------------
PY_SCRIPT="weinstein_intraday_watcher.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  red "Error: Cannot find $PY_SCRIPT in the current directory."
  red "Make sure you're in the ~/WeinsteinAgent folder."
  exit 2
fi

# ------------------------------------------------------------
# Run the intraday watcher
# ------------------------------------------------------------
yellow "• Running: python3 $PY_SCRIPT --config $CONFIG_PATH"
python3 "$PY_SCRIPT" --config "$CONFIG_PATH" "$@" || {
  red "❌ Intraday watcher encountered an error."
  exit 1
}

green "✅ Intraday tick complete."
