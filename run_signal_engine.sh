#!/usr/bin/env bash
# ============================================================
# run_signal_engine.sh – Evaluates near/armed/buy/sell from intraday_debug.csv
# ============================================================

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

OUTDIR="./output"
CSV="${OUTDIR}/intraday_debug.csv"
STATE="${OUTDIR}/diag_state.json"
SIGNALS="${OUTDIR}/signals_log.csv"
HTML_GLOB="${OUTDIR}/intraday_watch_*.html"

[[ -d ".venv" ]] && source .venv/bin/activate 2>/dev/null || true
mkdir -p "$OUTDIR"

bold "⚡ Signal Engine on: $CSV"
if [[ ! -s "$CSV" ]]; then
  yellow "• CSV missing or empty: $CSV"
  # still run to refresh near_universe from HTML
fi

python3 tools/signal_engine.py \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --state "$STATE" \
  --write-signals "$SIGNALS" \
  --html-glob "$HTML_GLOB" \
  --bps-threshold 35 \
  --window-min 120 \
  "$@" || { red "❌ Signal Engine error."; exit 1; }

green "✅ Signal Engine complete."
