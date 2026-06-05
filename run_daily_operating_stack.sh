#!/usr/bin/env bash
set -euo pipefail

# Weinstein Daily Operating Stack
#
# Purpose:
#   One command to run the daily operating flow across Fidelity sync,
#   crypto monitoring, and SIM-vs-PROD parity.
#
# Design:
#   - Uses the current working directory as PROJECT_DIR by default.
#   - Loads ~/.weinstein_env when present.
#   - Does not require hardcoded VM paths.
#   - Does not fail the entire stack if optional Fidelity export files are missing.
#   - Writes one timestamped run folder under output/daily_command_center/.
#
# Usage:
#   ./run_daily_operating_stack.sh
#
# Common overrides:
#   SEND_EMAIL=1 ./run_daily_operating_stack.sh
#   RUN_FIDELITY_SYNC=0 ./run_daily_operating_stack.sh
#   RUN_PARITY=0 ./run_daily_operating_stack.sh
#   RUN_CRYPTO=1 CRYPTO_FORCE_EMAIL=1 ./run_daily_operating_stack.sh

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

if [[ -f "$HOME/.weinstein_env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.weinstein_env"
fi

if [[ -d ".venv" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DATE="$(date +%Y%m%d)"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/output/daily_command_center/${RUN_DATE}_${STAMP}}"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/daily_operating_stack.log"
SUMMARY_FILE="$RUN_DIR/daily_operating_stack_summary.txt"
HTML_FILE="$RUN_DIR/daily_operating_stack_summary.html"

RUN_FIDELITY_SYNC="${RUN_FIDELITY_SYNC:-1}"
RUN_CRYPTO="${RUN_CRYPTO:-1}"
RUN_PARITY="${RUN_PARITY:-1}"
RUN_PROD_DAILY="${RUN_PROD_DAILY:-0}"

SEND_EMAIL="${SEND_EMAIL:-1}"
UPLOAD_SHEETS="${UPLOAD_SHEETS:-0}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"

CRYPTO_ONLY="${CRYPTO_ONLY:-BTC-USD,ETH-USD,SOL-USD,LTC-USD}"
CRYPTO_FORCE_EMAIL="${CRYPTO_FORCE_EMAIL:-0}"

POSITIONS_CSV="${POSITIONS_CSV:-}"
HISTORY_CSV="${HISTORY_CSV:-}"

# Auto-detect common Fidelity export filenames when env vars are not provided.
if [[ -z "${POSITIONS_CSV}" ]]; then
  POSITIONS_CSV="$(ls -1t Portfolio_Positions*.csv 2>/dev/null | head -1 || true)"
fi

if [[ -z "${HISTORY_CSV}" ]]; then
  HISTORY_CSV="$(ls -1t Accounts_History*.csv 2>/dev/null | head -1 || true)"
fi

log() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

section() {
  local name="$1"
  echo "" | tee -a "$LOG_FILE"
  echo "============================================================" | tee -a "$LOG_FILE"
  echo "$name" | tee -a "$LOG_FILE"
  echo "============================================================" | tee -a "$LOG_FILE"
}

run_step() {
  local name="$1"
  shift

  section "$name"
  log "START: $name"

  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    log "DONE: $name"
    echo "OK | $name" >> "$SUMMARY_FILE"
  else
    log "FAILED ($rc): $name"
    echo "FAILED($rc) | $name" >> "$SUMMARY_FILE"
  fi

  return "$rc"
}

write_header() {
  cat > "$SUMMARY_FILE" <<EOF
Weinstein Daily Operating Stack
RunUTC: $(date -u '+%Y-%m-%d %H:%M:%S')
Host: $(hostname)
ProjectDir: $PROJECT_DIR
RunDir: $RUN_DIR

Configuration:
RUN_FIDELITY_SYNC=$RUN_FIDELITY_SYNC
RUN_PROD_DAILY=$RUN_PROD_DAILY
RUN_CRYPTO=$RUN_CRYPTO
RUN_PARITY=$RUN_PARITY
SEND_EMAIL=$SEND_EMAIL
UPLOAD_SHEETS=$UPLOAD_SHEETS
LOOKBACK_DAYS=$LOOKBACK_DAYS
CRYPTO_ONLY=$CRYPTO_ONLY
CRYPTO_FORCE_EMAIL=$CRYPTO_FORCE_EMAIL
POSITIONS_CSV=${POSITIONS_CSV:-NONE}
HISTORY_CSV=${HISTORY_CSV:-NONE}

Step Results:
EOF
}

build_html_summary() {
  local latest_crypto=""
  local latest_parity_html=""
  local latest_account_recs=""
  local latest_comparison=""

  latest_crypto="$(ls -1t output/crypto_watch_*.html 2>/dev/null | head -1 || true)"
  latest_parity_html="$(ls -1t output/daily_parity/*/daily_prod_sim_summary_*.html 2>/dev/null | head -1 || true)"
  latest_account_recs="$(ls -1t output/daily_parity/*/daily_account_recommendations_*.csv 2>/dev/null | head -1 || true)"
  latest_comparison="$(ls -1t output/daily_parity/*/daily_prod_sim_signal_comparison_*.csv 2>/dev/null | head -1 || true)"

  cat > "$HTML_FILE" <<EOF
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Weinstein Daily Operating Stack</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1, h2 { margin-bottom: 6px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin: 14px 0; }
    .ok { color: #146c2e; font-weight: bold; }
    .warn { color: #9a6700; font-weight: bold; }
    .fail { color: #b42318; font-weight: bold; }
    code { background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }
    pre { background: #f6f8fa; padding: 12px; border-radius: 8px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>Weinstein Daily Operating Stack</h1>
  <p><b>Run UTC:</b> $(date -u '+%Y-%m-%d %H:%M:%S')</p>
  <p><b>Host:</b> $(hostname)</p>
  <p><b>Project:</b> <code>$PROJECT_DIR</code></p>
  <p><b>Run folder:</b> <code>$RUN_DIR</code></p>

  <div class="card">
    <h2>Step Results</h2>
    <pre>$(sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' "$SUMMARY_FILE")</pre>
  </div>

  <div class="card">
    <h2>Latest Outputs</h2>
    <p><b>Crypto Report:</b> <code>${latest_crypto:-NONE}</code></p>
    <p><b>Parity HTML:</b> <code>${latest_parity_html:-NONE}</code></p>
    <p><b>Account Recommendations:</b> <code>${latest_account_recs:-NONE}</code></p>
    <p><b>Signal Comparison:</b> <code>${latest_comparison:-NONE}</code></p>
  </div>

  <div class="card">
    <h2>Next Action</h2>
    <p>Review the latest crypto report and SIM-vs-PROD parity report. This command center currently orchestrates the stack; the next milestone is to aggregate the detailed report contents into this single HTML.</p>
  </div>
</body>
</html>
EOF
}

write_header

section "Environment"
log "PROJECT_DIR=$PROJECT_DIR"
log "RUN_DIR=$RUN_DIR"
log "Python=$(command -v python3 || true)"
log "GitCommit=$(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"

# Step 1: Fidelity sync to Google Sheet.
if [[ "$RUN_FIDELITY_SYNC" == "1" ]]; then
  if [[ -n "$POSITIONS_CSV" && -n "$HISTORY_CSV" && -f "$POSITIONS_CSV" && -f "$HISTORY_CSV" ]]; then
    run_step "Fidelity export sync to Google Sheet" \
      python3 sync_fidelity_exports_to_google_sheet.py \
        --positions-csv "$POSITIONS_CSV" \
        --history-csv "$HISTORY_CSV" \
        --write-sheet || true
  else
    section "Fidelity export sync to Google Sheet"
    log "SKIP: Missing positions/history CSV. POSITIONS_CSV=${POSITIONS_CSV:-NONE}; HISTORY_CSV=${HISTORY_CSV:-NONE}"
    echo "SKIPPED | Fidelity export sync to Google Sheet | missing CSV exports" >> "$SUMMARY_FILE"
  fi
else
  echo "SKIPPED | Fidelity export sync to Google Sheet | RUN_FIDELITY_SYNC=0" >> "$SUMMARY_FILE"
fi

# Step 2: Optional PROD daily watcher placeholder.
# Keep this disabled until the production daily command is confirmed.
if [[ "$RUN_PROD_DAILY" == "1" ]]; then
  if [[ -x "./run_all.sh" ]]; then
    run_step "PROD daily watcher / run_all.sh" ./run_all.sh || true
  else
    section "PROD daily watcher / run_all.sh"
    log "SKIP: run_all.sh not found or not executable."
    echo "SKIPPED | PROD daily watcher | run_all.sh missing/not executable" >> "$SUMMARY_FILE"
  fi
else
  echo "SKIPPED | PROD daily watcher | RUN_PROD_DAILY=0" >> "$SUMMARY_FILE"
fi

# Step 3: Crypto watcher.
if [[ "$RUN_CRYPTO" == "1" ]]; then
  crypto_cmd=(python3 weinstein_crypto_watcher.py --config ./config.yaml --only "$CRYPTO_ONLY")
  if [[ "$CRYPTO_FORCE_EMAIL" == "1" ]]; then
    crypto_cmd+=(--force-email)
  fi
  run_step "Crypto watcher" "${crypto_cmd[@]}" || true
else
  echo "SKIPPED | Crypto watcher | RUN_CRYPTO=0" >> "$SUMMARY_FILE"
fi

# Step 4: SIM vs PROD parity.
if [[ "$RUN_PARITY" == "1" ]]; then
  if [[ -x "./run_daily_sim_vs_prod_compare.sh" ]]; then
    run_step "SIM vs PROD parity" env \
      SEND_EMAIL="$SEND_EMAIL" \
      UPLOAD_SHEETS="$UPLOAD_SHEETS" \
      LOOKBACK_DAYS="$LOOKBACK_DAYS" \
      POSITIONS_SOURCE="${POSITIONS_SOURCE:-GOOGLE_SHEET}" \
      POSITIONS_CSV="${POSITIONS_CSV_OVERRIDE:-GOOGLE_SHEET}" \
      ./run_daily_sim_vs_prod_compare.sh || true
  else
    section "SIM vs PROD parity"
    log "SKIP: run_daily_sim_vs_prod_compare.sh not found or not executable."
    echo "SKIPPED | SIM vs PROD parity | script missing/not executable" >> "$SUMMARY_FILE"
  fi
else
  echo "SKIPPED | SIM vs PROD parity | RUN_PARITY=0" >> "$SUMMARY_FILE"
fi

section "Build command center summary"
build_html_summary
log "Summary text: $SUMMARY_FILE"
log "Summary HTML: $HTML_FILE"

section "Done"
log "Daily operating stack complete."
echo ""
echo "DONE daily operating stack"
echo "Run folder: $RUN_DIR"
echo "Summary HTML: $HTML_FILE"
