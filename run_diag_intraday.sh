#!/usr/bin/env bash
# ============================================================
# run_diag_intraday.sh – Wraps tools/diagnose_intraday.py
# ------------------------------------------------------------
# Examples:
#   ./run_diag_intraday.sh
#   ./run_diag_intraday.sh --explain MU,DDOG --bps-threshold 35 --window-min 120
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

CSV="${CSV_PATH:-./output/intraday_debug.csv}"
OUTDIR="${OUTDIR:-./output}"
STATE="${STATE_PATH:-./output/diag_state.json}"
HTML_GLOB="${HTML_GLOB:-./output/intraday_watch_*.html}"
SIGNALS_CSV="${SIGNALS_CSV:-./output/signals_log.csv}"

bold "🔎 Diagnostics on: $CSV"
python3 tools/diagnose_intraday.py \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --state "$STATE" \
  --html-glob "$HTML_GLOB" \
  --signals-csv "$SIGNALS_CSV" \
  "$@" && green "✅ Diagnostics complete." || {
  red "❌ Diagnostics error."
  exit 1
}
