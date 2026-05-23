#!/usr/bin/env bash
# ============================================================
# run_prod_validation_scan.sh
# ------------------------------------------------------------
# Safe validation-mode PROD scan.
#
# Purpose:
# - Run the existing intraday watcher with --test-ease
# - Write to output/intraday_debug_validation.csv
# - Build a comparison report versus normal PROD output/intraday_debug.csv
#
# This does NOT change:
# - production config.yaml
# - production cron
# - output/intraday_debug.csv
# - Weinstein CORE logic
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/luiscarlosht/WeinsteinAgent"
cd "$PROJECT_DIR"

LOCK_FILE="/tmp/weinstein_prod_validation_scan.lock"
exec 9>"$LOCK_FILE"

if ! flock -n 9; then
  echo "Another PROD validation scan is already active. Exiting."
  exit 0
fi

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"
STRICT_DEBUG="${STRICT_DEBUG:-./output/intraday_debug.csv}"
VALIDATION_DEBUG="${VALIDATION_DEBUG:-./output/intraday_debug_validation.csv}"
OUT_DIR="${OUT_DIR:-./output/prod_validation}"
SEND_EMAIL="${SEND_EMAIL:-1}"

mkdir -p "$OUT_DIR"

echo "PROD validation-mode scan"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "STRICT_DEBUG=$STRICT_DEBUG"
echo "VALIDATION_DEBUG=$VALIDATION_DEBUG"
echo "OUT_DIR=$OUT_DIR"
echo "SEND_EMAIL=$SEND_EMAIL"

echo
echo "Running intraday watcher in TEST-EASE validation mode..."
python3 weinstein_intraday_watcher.py \
  --config "$CONFIG_PATH" \
  --log-csv "$VALIDATION_DEBUG" \
  --test-ease

echo
echo "Building validation summary..."
ARGS=(
  --strict-debug "$STRICT_DEBUG"
  --validation-debug "$VALIDATION_DEBUG"
  --out-dir "$OUT_DIR"
)

if [[ "$SEND_EMAIL" == "1" ]]; then
  ARGS+=(--send-email)
fi

python3 weinstein_prod_validation_summary.py "${ARGS[@]}"

echo
echo "DONE PROD validation scan"
ls -lh "$OUT_DIR" | tail -20
