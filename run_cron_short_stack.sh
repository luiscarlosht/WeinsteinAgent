#!/usr/bin/env bash
# ============================================================
# run_cron_short_stack.sh – Full intraday + short stack
# ------------------------------------------------------------
# Runs:
#   1) Long-side intraday watcher (+ Signal Engine + Diagnostics)
#   2) Short-side intraday watcher (Chapter 8 aware)
#   3) Short Signal Engine (intraday window + full history)
#
# Safe to call from cron every N minutes.
# Chapter 8:
#   - Long/short regime is enforced *inside* the watchers.
#   - If short_ok=False, short watcher writes an empty CSV and
#     short_signal_engine exits cleanly with "nothing to do".
# ============================================================

set -euo pipefail

# Move to repo root
cd "$(dirname "$0")"

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

# Activate virtual environment if present
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
fi

# ---------------------------------------------
# 1) LONG-SIDE STACK (intraday + signals + diag)
# ---------------------------------------------
bold "⚡ Long-side intraday + Signal Engine + Diagnostics"
# We force test-ease + explicit log-csv; everything else is handled
# by weinstein_intraday_watcher.py and the run_ helpers.
./run_intraday.sh \
  --test-ease \
  --log-csv ./output/intraday_debug.csv

# run_intraday.sh already calls:
#   - ./run_signal_engine.sh
#   - ./run_diag_intraday.sh
# so we don't re-run them here.

# ---------------------------------------------
# 2) SHORT-SIDE INTRADAY (Chapter 8 aware)
# ---------------------------------------------
bold "⚡ Short-side intraday run using config: $CONFIG_PATH"

python3 weinstein_short_watcher.py \
  --config "$CONFIG_PATH" \
  --test-ease \
  --log-csv ./output/short_debug.csv

# Note:
#   - If Chapter 8 says short_ok=False, this will log:
#       "short side is DISABLED in current regime — skipping short scan."
#     and write an empty ./output/short_debug.csv.
#   - That’s expected; the engine below will then no-op cleanly.

# ---------------------------------------------
# 3) SHORT SIGNAL ENGINE (two windows)
# ---------------------------------------------
bold "⚡ Short Signal Engine on: ./output/short_debug.csv (intraday window 180m)"
./run_short_signal_engine.sh --bps 40 --window-min 180 || {
  red "❌ Short Signal Engine (window 180) error."
  exit 1
}

bold "⚡ Short Signal Engine on: ./output/short_debug.csv (full history)"
./run_short_signal_engine.sh --bps 0 --window-min 10000 || {
  red "❌ Short Signal Engine (full history) error."
  exit 1
}

green "✅ Short stack complete."
