#!/usr/bin/env bash
# run_intraday.sh – intraday watcher (YAML-driven)
#
# Example cron (ET every 10 min during session):
# */10 9-16 * * 1-5 /bin/bash -lc 'cd ~/WeinsteinAgent && ./run_intraday.sh'
#
# You can still pin timezone with CRON_TZ if desired.

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"

if [[ ! -r "$CONFIG_PATH" ]]; then
  red "Config file not found or unreadable: $CONFIG_PATH"
  red "Set CONFIG_FILE=./config.yaml or create ./config.yaml"
  exit 2
fi

# Activate venv if present
source .venv/bin/activate 2>/dev/null || true

bold "⚡ Intraday watcher using config: $CONFIG_PATH"

# If your intraday Python supports --config, pass it here.
# Replace the script name below with the one you actually use for intraday.
# Common names from this project: weinstein_intraday.py or intraday_watcher.py
PY_SCRIPT=""
for cand in weinstein_intraday.py intraday_watcher.py intraday.py; do
  [[ -f "$cand" ]] && PY_SCRIPT="$cand" && break
done

if [[ -z "$PY_SCRIPT" ]]; then
  red "Could not find an intraday Python script (tried: weinstein_intraday.py, intraday_watcher.py, intraday.py)."
  red "Add or rename your intraday script accordingly."
  exit 2
fi

yellow "• Running: python3 $PY_SCRIPT --config $CONFIG_PATH"
python3 "$PY_SCRIPT" --config "$CONFIG_PATH" "$@"

green "✅ Intraday tick complete."
