#!/usr/bin/env bash
# ============================================================
# run_intraday.sh – Launches Weinstein Intraday Watcher
# and then runs Signal Engine + Diagnostics
# ------------------------------------------------------------
# Example:
#   ./run_intraday.sh --dry-run
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

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
  source .venv/bin/activate 2>/dev/null || true
fi

bold "⚡ Intraday watcher using config: $CONFIG_PATH"

PY_SCRIPT="weinstein_intraday_watcher.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
  red "Error: Cannot find $PY_SCRIPT in the current directory."
  red "Make sure you're in the ~/WeinsteinAgent folder."
  exit 2
fi

yellow "• Running: python3 $PY_SCRIPT --config $CONFIG_PATH $*"
python3 "$PY_SCRIPT" --config "$CONFIG_PATH" "$@" || {
  red "❌ Intraday watcher encountered an error."
  exit 1
}
green "✅ Intraday tick complete."

# ---------------------------------------------
# Follow-ups: Signal Engine, then Diagnostics
# ---------------------------------------------
if [[ -x "./run_signal_engine.sh" ]]; then
  ./run_signal_engine.sh
else
  yellow "run_signal_engine.sh not found/executable; skipping."
fi

if [[ -x "./run_diag_intraday.sh" ]]; then
  ./run_diag_intraday.sh
else
  yellow "run_diag_intraday.sh not found/executable; skipping."
fi
