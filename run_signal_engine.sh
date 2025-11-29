#!/usr/bin/env bash
# ============================================================
# run_signal_engine.sh – Wraps tools/signal_engine.py
# ------------------------------------------------------------
# Examples:
#   ./run_signal_engine.sh
#   ./run_signal_engine.sh --bps-threshold 35 --window-min 120 --explain MU,DDOG
# ============================================================

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

# Activate venv if present
if [[ -d ".venv" ]]; then
  source .venv/bin/activate 2>/dev/null || true
fi

# ----------------------------------------------------------------------
# DEFAULT PATHS (override with ENV if needed)
# ----------------------------------------------------------------------
CSV="${CSV_PATH:-./output/intraday_debug.csv}"
OUTDIR="${OUTDIR:-./output}"
STATE="${STATE_PATH:-./output/diag_state.json}"
SIGNALS="${SIGNALS_PATH:-./output/signals_log.csv}"
HTML_GLOB="${HTML_GLOB:-./output/intraday_watch_*.html}"

# ----------------------------------------------------------------------
# DEFAULT ARGUMENTS PASSED TO tools/signal_engine.py
# You can still override them (e.g., ./run_signal_engine.sh --window-min 300)
# ----------------------------------------------------------------------
DEFAULT_ARGS=(
  --csv "$CSV"
  --outdir "$OUTDIR"
  --state "$STATE"
  --write-signals "$SIGNALS"
  --html-glob "$HTML_GLOB"
  --bps-threshold 35
  --window-min 10000
)

# ----------------------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------------------
bold "⚡ Signal Engine on: $CSV"

python3 tools/signal_engine.py "${DEFAULT_ARGS[@]}" "$@" \
  && green "✅ Signal Engine complete." \
  || { red "❌ Signal Engine error."; exit 1; }
