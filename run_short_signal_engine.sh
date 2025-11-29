#!/usr/bin/env bash
# ============================================================
# run_short_signal_engine.sh – Wraps tools/short_signal_engine.py
# ------------------------------------------------------------
# Examples:
#   ./run_short_signal_engine.sh
#   ./run_short_signal_engine.sh --window-min 180 --explain CRM
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

CSV_DEFAULT="./output/short_debug.csv"
OUTDIR_DEFAULT="./output"
BPS_DEFAULT=50          # basis-point threshold placeholder (not really used yet)
WINDOW_DEFAULT=390      # minutes (full regular US session)

CSV="$CSV_DEFAULT"
OUTDIR="$OUTDIR_DEFAULT"
BPS="$BPS_DEFAULT"
WINDOW_MIN="$WINDOW_DEFAULT"

EXTRA_ARGS=()

# Known options: --csv, --outdir, --bps, --window-min
# Everything else (e.g. --explain CRM) is forwarded to Python.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)
      CSV="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --bps)
      BPS="$2"
      shift 2
      ;;
    --window-min)
      WINDOW_MIN="$2"
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

bold "⚡ Short Signal Engine on: $CSV"
echo "   → outdir:      $OUTDIR"
echo "   → bps:         $BPS"
echo "   → window-min:  $WINDOW_MIN"

python3 tools/short_signal_engine.py \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --window-min "$WINDOW_MIN" \
  --bps "$BPS" \
  "${EXTRA_ARGS[@]}" && green "✅ Short Signal Engine complete." || {
    red "❌ Short Signal Engine error."
    exit 1
  }
