#!/usr/bin/env bash
set -euo pipefail

# Defaults
CSV_DEFAULT="./output/short_debug.csv"
OUTDIR_DEFAULT="./output"
BPS_DEFAULT=50          # basis points threshold for short signals
WINDOW_DEFAULT=390      # minutes (full regular session)

CSV="$CSV_DEFAULT"
OUTDIR="$OUTDIR_DEFAULT"
BPS="$BPS_DEFAULT"
WINDOW_MIN="$WINDOW_DEFAULT"

EXTRA_ARGS=()

# Simple CLI parsing for the wrapper
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
      # Explicit end of wrapper options; everything after this goes to Python
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      # Unknown flag/arg → pass through to Python (e.g. --explain CRM)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "⚡ Short Signal Engine on: $CSV"
echo "   → outdir:      $OUTDIR"
echo "   → bps:         $BPS"
echo "   → window-min:  $WINDOW_MIN"

python3 tools/short_signal_engine.py \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --window-min "$WINDOW_MIN" \
  --bps-threshold "$BPS" \
  "${EXTRA_ARGS[@]}"
