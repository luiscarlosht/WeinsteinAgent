#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Activate virtual environment
source /home/luiscarlosht/WeinsteinAgent/.venv/bin/activate

echo "⚡ Intraday watcher using config: ./config.yaml"
# Long-side intraday: currently with test-ease
python3 weinstein_intraday_watcher.py \
  --config ./config.yaml \
  --test-ease \
  --log-csv ./output/intraday_debug.csv
  # --dry-run    # uncomment if you want to suppress email

echo "✅ Intraday tick complete."

echo "⚡ Signal Engine on: ./output/intraday_debug.csv"
./run_signal_engine.sh

echo "🔎 Diagnostics on: ./output/intraday_debug.csv"
./run_diag_intraday.sh

echo "⚡ Short-side intraday run using config: ./config.yaml"
# SHORT-SIDE: TEST-EASE BUT LIVE EMAIL (no --dry-run)
python3 weinstein_short_watcher.py \
  --config ./config.yaml \
  --test-ease \
  --log-csv ./output/short_debug.csv

echo "⚡ Short Signal Engine on: ./output/short_debug.csv"
# 1) Intraday-window summary (e.g. 3 hours = 180 min)
./run_short_signal_engine.sh --bps 40 --window-min 180

# 2) Full-history summary (effectively “no window”)
./run_short_signal_engine.sh --bps 0 --window-min 10000

echo "✅ Short stack complete."
