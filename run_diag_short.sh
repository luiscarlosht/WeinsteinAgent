#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
cd "$(dirname "$0")"

CFG="./config.yaml"
CSV="./output/short_debug.csv"
OUTDIR="./output"

echo "⚡ Short-side *diagnostic* run using config: ${CFG}"
echo "• Running short watcher (test-ease, dry-run) with CSV: ${CSV}"

python3 weinstein_short_watcher.py \
  --config "${CFG}" \
  --test-ease \
  --dry-run \
  --log-csv "${CSV}"

echo "⚡ Short Signal Engine on: ${CSV}"
python3 tools/short_signal_engine.py \
  --csv "${CSV}" \
  --outdir "${OUTDIR}" \
  --window-min 120

echo "✅ Short diagnostics complete."

# --------------------------------------------------------------------
# Aggregated “production-style” summary with a tighter window
# --------------------------------------------------------------------
# This uses the wrapper so you can override params if needed:
#   ./run_short_signal_engine.sh --bps 0 --window-min 10000 --explain FDS
# etc.
echo "./run_short_signal_engine.sh --bps 40 --window-min 180 $*"
./run_short_signal_engine.sh --bps 40 --window-min 180 "$@"
