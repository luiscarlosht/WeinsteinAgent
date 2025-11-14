#!/usr/bin/env bash
set -euo pipefail

# Default locations
CSV="./output/short_debug.csv"
OUTDIR="./output"

# Default thresholds (can be overridden via flags below)
BPS_THRESHOLD=50      # 0.50% move
WINDOW_MIN=390        # up to one full regular session in minutes

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra_args...]

Options:
  --csv PATH            Path to short debug CSV (default: ${CSV})
  --outdir DIR          Output directory (default: ${OUTDIR})
  --bps N               Basis-point threshold (default: ${BPS_THRESHOLD})
  --window-min N        Max elapsed_min window in minutes (default: ${WINDOW_MIN})
  -h, --help            Show this help

Any arguments after '--' are passed directly to short_signal_engine.py
(e.g. --explain CRM).
EOF
}

# Parse our simple flags; stop at '--'
EXTRA_ARGS=()
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
      BPS_THRESHOLD="$2"
      shift 2
      ;;
    --window-min)
      WINDOW_MIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      # Unknown option; treat as extra arg for the Python tool
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "⚡ Short Signal Engine on: ${CSV}"
echo "   → outdir:      ${OUTDIR}"
echo "   → bps:         ${BPS_THRESHOLD}"
echo "   → window-min:  ${WINDOW_MIN}"

python3 tools/short_signal_engine.py \
  --csv "${CSV}" \
  --outdir "${OUTDIR}" \
  --bps-threshold "${BPS_THRESHOLD}" \
  --window-min "${WINDOW_MIN}" \
  "${EXTRA_ARGS[@]:-}"

echo "✅ Short signal engine complete."
