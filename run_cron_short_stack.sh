#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Activate virtual environment
source /home/luiscarlosht/WeinsteinAgent/.venv/bin/activate

echo "⚡ Intraday watcher using config: ./config.yaml"
# Long-side intraday: keep this as test-only for now
python3 weinstein_intraday_watcher.py \
  --config ./config.yaml \
  --test-ease \
  --log-csv ./output/intraday_debug.csv 
  #\
  #--dry-run

echo "✅ Intraday tick complete."
echo "⚡ Signal Engine on: ./output/intraday_debug.csv"
./run_signal_engine.sh

echo "🔎 Diagnostics on: ./output/intraday_debug.csv"
./run_diag_intraday.sh

echo "⚡ Short-side intraday run using config: ./config.yaml"
# SHORT-SIDE: TEST-EASE BUT LIVE EMAIL (NO --dry-run)
python3 weinstein_short_watcher.py \
  --config ./config.yaml \
  --test-ease \
  --log-csv ./output/short_debug.csv

echo "⚡ Short Signal Engine on: ./output/short_debug.csv"
./run_short_signal_engine.sh --bps 40 --window-min 180
./run_short_signal_engine.sh --bps 0 --window-min 10000

echo "✅ Short stack complete."
