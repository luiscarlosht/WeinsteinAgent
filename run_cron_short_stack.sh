#!/usr/bin/env bash
# ============================================================
# run_cron_short_stack.sh – Full intraday + short stack
# ------------------------------------------------------------
# Long side:
#   1) Intraday watcher  → ./output/intraday_debug.csv
#   2) Signal Engine     → ./output/signals_log.csv
#   3) Diagnostics       → ./output/diag_summary_*.txt
#
# Short side:
#   4) Short watcher     → ./output/short_debug.csv
#   5) Short Signal Eng. (180m window)
#   6) Short Signal Eng. (full-history)
#
# TEST_EASE toggle:
#   TEST_EASE=1 ./run_cron_short_stack.sh   # dev / validation (low thresholds)
#   ./run_cron_short_stack.sh               # production thresholds (no --test-ease)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

# Activate virtual environment if present
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
fi

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"

if [[ ! -r "$CONFIG_PATH" ]]; then
  red "Config file not found or unreadable: $CONFIG_PATH"
  red "Set CONFIG_FILE=./config.yaml or create ./config.yaml"
  exit 2
fi

# ------------------------------------------------------------
# TEST-EASE toggle
# ------------------------------------------------------------
# TEST_EASE=1 → add --test-ease to both long & short watchers
# default (empty / 0) → production thresholds
TEST_EASE="${TEST_EASE:-0}"

INTRADAY_EXTRA=()
SHORT_EXTRA=()

if [[ "$TEST_EASE" == "1" ]]; then
  yellow "⚠️  TEST-EASE ENABLED for this run (lowered thresholds)."
  INTRADAY_EXTRA+=(--test-ease)
  SHORT_EXTRA+=(--test-ease)
fi

# =======================
# 1) LONG-SIDE STACK
# =======================
bold "⚡ Long-side intraday + Signal Engine + Diagnostics"

bold "⚡ Intraday watcher using config: $CONFIG_PATH"
yellow "• Running: python3 weinstein_intraday_watcher.py --config $CONFIG_PATH --log-csv ./output/intraday_debug.csv ${INTRADAY_EXTRA[*]}"

python3 weinstein_intraday_watcher.py \
  --config "$CONFIG_PATH" \
  --log-csv ./output/intraday_debug.csv \
  "${INTRADAY_EXTRA[@]}" || {
    red "❌ Intraday watcher encountered an error."
    exit 1
  }

green "✅ Intraday tick complete."

# Signal Engine
if [[ -x "./run_signal_engine.sh" ]]; then
  ./run_signal_engine.sh
else
  yellow "run_signal_engine.sh not found/executable; skipping."
fi

# Diagnostics
if [[ -x "./run_diag_intraday.sh" ]]; then
  ./run_diag_intraday.sh
else
  yellow "run_diag_intraday.sh not found/executable; skipping."
fi

# =======================
# 2) SHORT-SIDE STACK
# =======================
bold "⚡ Short-side intraday run using config: $CONFIG_PATH"

python3 weinstein_short_watcher.py \
  --config "$CONFIG_PATH" \
  --log-csv ./output/short_debug.csv \
  "${SHORT_EXTRA[@]}" || {
    red "❌ Short watcher encountered an error."
    exit 1
  }

green "✅ Short tick complete."

# Short Signal Engine – intraday window
bold "⚡ Short Signal Engine on: ./output/short_debug.csv (intraday window 180m)"
if [[ -x "./run_short_signal_engine.sh" ]]; then
  ./run_short_signal_engine.sh --bps 40 --window-min 180
else
  yellow "run_short_signal_engine.sh not found/executable; skipping."
fi

# Short Signal Engine – full history
bold "⚡ Short Signal Engine on: ./output/short_debug.csv (full history)"
if [[ -x "./run_short_signal_engine.sh" ]]; then
  ./run_short_signal_engine.sh --bps 0 --window-min 10000
fi

green "✅ Short stack complete."
