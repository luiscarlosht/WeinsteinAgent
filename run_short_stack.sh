#!/usr/bin/env bash
set -euo pipefail
# Activate venv if present
if [[ -d ".venv" ]]; then
  source .venv/bin/activate 2>/dev/null || true
fi

CFG=${1:-./config.yaml}
OUTDIR=./output
LOGCSV="$OUTDIR/short_debug.csv"
LOGJSON="$OUTDIR/short_debug.json"

echo "⚡ Short stack using config: $CFG"

echo "▶️ Step 1/2: Short watcher..."
python3 weinstein_short_watcher.py \
  --config "$CFG" \
  --log-csv "$LOGCSV" \
  --log-json "$LOGJSON"

echo "⚡ Step 2/2: Short Signal Engine on: $LOGCSV"
python3 short_signal_engine.py \
  --csv "$LOGCSV" \
  --outdir "$OUTDIR" \
  --window-min 390 \
  --bps 50

echo "✅ Short stack complete."
