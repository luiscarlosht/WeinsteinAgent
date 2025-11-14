#!/bin/bash
cd /home/luiscarlosht/WeinsteinAgent
source .venv/bin/activate

# 1) Intraday watcher (test mode)
./run_intraday.sh --test-ease --dry-run

# 2) Short watcher with CSV logging for the engine
./run_diag_short.sh

# 3) Short signal engine (window 10000 min, threshold 0 → show all)
./run_short_signal_engine.sh --bps 0 --window-min 10000

exit 0
