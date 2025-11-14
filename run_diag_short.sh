#!/usr/bin/env bash
set -euo pipefail

CFG="./config.yaml"
OUTDIR="./output"
CSV="${OUTDIR}/short_debug.csv"

echo "⚡ Short-side *diagnostic* run using config: ${CFG}"

# 1) Short watcher in test/dry-run mode + CSV logging
echo "• Running short watcher (test-ease, dry-run) with CSV: ${CSV}"
python3 weinstein_short_watcher.py \
  --config "${CFG}" \
  --test-ease \
  --dry-run \
  --log-csv "${CSV}"

# 2) Short signal engine over that CSV
echo "⚡ Short Signal Engine on: ${CSV}"
python3 tools/short_signal_engine.py \
  --csv "${CSV}" \
  --outdir "${OUTDIR}" \
  --window-min 120

echo "✅ Short diagnostics complete."
