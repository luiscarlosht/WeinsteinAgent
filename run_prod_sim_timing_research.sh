#!/usr/bin/env bash
set -euo pipefail

cd /home/luiscarlosht/WeinsteinAgent

OUT_DIR="${OUT_DIR:-output/prod_sim_timing_research}"
PROD_HISTORY="${PROD_HISTORY:-output/prod_intraday_signal_history.csv}"
PARITY_DIR="${PARITY_DIR:-}"

ARGS=(--prod-history "$PROD_HISTORY" --out-dir "$OUT_DIR")

if [[ -n "$PARITY_DIR" ]]; then
  ARGS+=(--parity-dir "$PARITY_DIR")
fi

python3 weinstein_prod_sim_timing_research.py "${ARGS[@]}"
