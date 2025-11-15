#!/usr/bin/env bash
set -euo pipefail

# Wrapper for tools/short_signal_engine.py

CSV_DEFAULT="./output/short_debug.csv"
OUTDIR_DEFAULT="./output"
BPS_DEFAULT=50          # basis-point threshold placeholder (not used yet)
WINDOW_DEFAULT=390      # minutes (full regular session)

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

echo "⚡ Short Signal Engine on: $CSV"
echo "   → outdir:      $OUTDIR"
echo "   → bps:         $BPS"
echo "   → window-min:  $WINDOW_MIN"

python3 tools/short_signal_engine.py \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --window-min "$WINDOW_MIN" \
  --bps "$BPS" \
  "${EXTRA_ARGS[@]}"
