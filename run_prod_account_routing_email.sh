#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/luiscarlosht/WeinsteinAgent"
cd "$PROJECT_DIR"

LOCK_FILE="/tmp/weinstein_prod_account_routing.lock"
exec 9>"$LOCK_FILE"

if ! flock -n 9; then
  echo "Another PROD account-routing report is already active. Exiting."
  exit 0
fi

POSITIONS_CSV="${POSITIONS_CSV:-}"
PROD_DEBUG="${PROD_DEBUG:-$PROJECT_DIR/output/intraday_debug.csv}"
PARITY_DIR="${PARITY_DIR:-}"
D_SOURCE="${D_SOURCE:-auto}"
SEND_EMAIL="${SEND_EMAIL:-1}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/output/prod_account_routing}"

# Auto-detect latest Fidelity positions export if not provided or path is missing.
if [[ "$POSITIONS_CSV" != "GOOGLE_SHEET" && ( -z "$POSITIONS_CSV" || ! -f "$POSITIONS_CSV" ) ]]; then
  if [[ -n "$POSITIONS_CSV" && ! -f "$POSITIONS_CSV" ]]; then
    echo "WARNING: Provided POSITIONS_CSV does not exist: $POSITIONS_CSV"
    echo "Attempting auto-detect instead..."
  fi

  POSITIONS_CSV="$(
    ls -t \
      "$PROJECT_DIR"/Portfolio_Positions*.csv \
      "$PROJECT_DIR"/output/Portfolio_Positions*.csv \
      "$HOME"/Portfolio_Positions*.csv \
      2>/dev/null | head -1 || true
  )"
fi

ARGS=(
  --prod-debug "$PROD_DEBUG"
  --profiles account_strategy_profiles.yaml
  --out-dir "$OUT_DIR"
  --d-source "$D_SOURCE"
)

if [[ "$POSITIONS_CSV" == "GOOGLE_SHEET" || ( -n "$POSITIONS_CSV" && -f "$POSITIONS_CSV" ) ]]; then
  ARGS+=(--positions-csv "$POSITIONS_CSV")
else
  echo "WARNING: No valid positions CSV found. Owned-position filtering will be limited."
fi

if [[ -n "$PARITY_DIR" ]]; then
  ARGS+=(--parity-dir "$PARITY_DIR")
fi

if [[ "$SEND_EMAIL" == "1" ]]; then
  ARGS+=(--send-email)
fi

echo "PROD account-routing report"
echo "PROD_DEBUG=$PROD_DEBUG"
echo "POSITIONS_CSV=${POSITIONS_CSV:-NONE}"
echo "PARITY_DIR=${PARITY_DIR:-latest auto-detect}"
echo "D_SOURCE=$D_SOURCE"
echo "OUT_DIR=$OUT_DIR"

python3 weinstein_prod_account_router.py "${ARGS[@]}"
