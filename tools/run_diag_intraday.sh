#!/usr/bin/env bash
# ============================================================
# run_diag_intraday.sh – Runs tools/diagnose_intraday.py for a readable summary
# ============================================================

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

OUTDIR="./output"
CSV="${OUTDIR}/intraday_debug.csv"
STATE="${OUTDIR}/diag_state.json"
HTML_GLOB="${OUTDIR}/intraday_watch_*.html"
SIGNALS_CSV="${OUTDIR}/signals_log.csv"

[[ -d ".venv" ]] && source .venv/bin/activate 2>/dev/null || true
mkdir -p "$OUTDIR"

bold "🔎 Diagnostics on: $CSV"
python3 tools/diagnose_intraday.py \
  --csv "$CSV" \
  --signals-csv "$SIGNALS_CSV" \
  --outdir "$OUTDIR" \
  --state "$STATE" \
  --html-glob "$HTML_GLOB" \
  --bps-threshold 35 \
  --window-min 120 \
  "$@" || { red "❌ Diagnostics error."; exit 1; }

green "✅ Diagnostics complete."
