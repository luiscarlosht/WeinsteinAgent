#!/bin/bash
# ---------------------------------------------------------
# run_buffett.sh
# Runs Buffett CSP Engine with correct venv & logging
# ---------------------------------------------------------

# Go to project directory
cd /home/luiscarlosht/WeinsteinAgent || exit 1

# Activate virtual environment
source /home/luiscarlosht/WeinsteinAgent/.venv/bin/activate

# Run engine
python3 buffett_options_engine.py >> /home/luiscarlosht/WeinsteinAgent/cron_buffett.log 2>&1
