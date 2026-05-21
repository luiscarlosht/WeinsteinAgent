#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/luiscarlosht/WeinsteinAgent"
cd "$PROJECT_DIR"

START_DATE="${START_DATE:-$(date -d '30 days ago' +%F)}"
END_DATE="${END_DATE:-$(date +%F)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PROJECT_DIR/output/daily_parity/$STAMP"
mkdir -p "$RUN_DIR"

POSITIONS_CSV="${POSITIONS_CSV:-}"
PROD_DEBUG="${PROD_DEBUG:-$PROJECT_DIR/output/intraday_debug.csv}"
SEND_EMAIL="${SEND_EMAIL:-1}"
UPLOAD_SHEETS="${UPLOAD_SHEETS:-0}"

# Auto-detect latest Fidelity positions export if not provided.
if [[ -z "$POSITIONS_CSV" ]]; then
  POSITIONS_CSV="$(
    ls -t \
      "$PROJECT_DIR"/Portfolio_Positions*.csv \
      "$PROJECT_DIR"/output/Portfolio_Positions*.csv \
      "$HOME"/Portfolio_Positions*.csv \
      2>/dev/null | head -1 || true
  )"
fi

echo "Daily SIM vs PROD parity"
echo "START_DATE=$START_DATE"
echo "END_DATE=$END_DATE"
echo "RUN_DIR=$RUN_DIR"
echo "PROD_DEBUG=$PROD_DEBUG"
echo "POSITIONS_CSV=${POSITIONS_CSV:-NONE}"

SIM_D_EVENTS="$RUN_DIR/sim_D_replay_events.csv"
SIM_F_EVENTS="$RUN_DIR/sim_F_base_events.csv"
SIM_F_META="$RUN_DIR/sim_F_meta_equity.csv"

###############################################################################
# SIM D
###############################################################################

echo "Generating SIM D replay events..."

python3 weinstein_replay_portfolio_backtest_fast.py \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --mode both \
  --snapshot-mode auto \
  --config ./config.yaml \
  --regime-mode prod \
  --neutral-policy long \
  --exposure-mode scaled \
  --max-leverage 1.0 \
  --max-pos-frac 0.20 \
  --min-equity-frac 0.25 \
  --replay-only \
  --replay-events-out "$SIM_D_EVENTS" \
  --save-events

###############################################################################
# SIM F BASE
###############################################################################

echo "Generating broad replay events for SIM F..."

python3 weinstein_replay_portfolio_backtest_fast_meta.py \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --mode both \
  --snapshot-mode auto \
  --config ./config.yaml \
  --regime-mode off \
  --exposure-mode off \
  --max-leverage 1.0 \
  --max-pos-frac 0.20 \
  --min-equity-frac 0.25 \
  --replay-only \
  --replay-events-out "$SIM_F_EVENTS" \
  --save-events

###############################################################################
# SIM F META PORTFOLIO
###############################################################################

echo "Running SIM F meta portfolio for daily decision log..."

python3 weinstein_replay_portfolio_backtest_fast_meta.py \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --mode both \
  --snapshot-mode auto \
  --config ./config.yaml \
  --max-leverage 1.0 \
  --max-pos-frac 0.20 \
  --min-equity-frac 0.25 \
  --replay-events-in "$SIM_F_EVENTS" \
  --meta-strategy F \
  --meta-log \
  > "$RUN_DIR/sim_F_meta_run.log" 2>&1 || {
    echo "SIM F meta run failed. Last 80 lines:"
    tail -80 "$RUN_DIR/sim_F_meta_run.log"
    exit 1
  }

###############################################################################
# BUILD COMPARISON REPORT
###############################################################################

ARGS=(
  --prod-debug "$PROD_DEBUG"
  --sim-d-events "$SIM_D_EVENTS"
  --sim-f-events "$SIM_F_EVENTS"
  --sim-f-meta "$SIM_F_META"
  --profiles account_strategy_profiles.yaml
  --out-dir "$RUN_DIR"
)

if [[ -n "$POSITIONS_CSV" ]]; then
  ARGS+=(--positions-csv "$POSITIONS_CSV")
fi

if [[ "$SEND_EMAIL" == "1" ]]; then
  ARGS+=(--send-email)
fi

if [[ "$UPLOAD_SHEETS" == "1" ]]; then
  ARGS+=(--upload-sheets)
fi

echo "Building comparison report..."

python3 weinstein_daily_sim_prod_compare.py "${ARGS[@]}"

echo
echo "DONE daily parity run"
echo "Output folder: $RUN_DIR"

ls -lh "$RUN_DIR"
```
