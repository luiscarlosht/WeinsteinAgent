#!/usr/bin/env bash
# run_all.sh
# Upload Fidelity CSVs → refresh merged Signals → rebuild dashboard tabs
# Supports passthrough flags for build_performance_dashboard.py

set -euo pipefail

# ------------------------------
# Pretty printing helpers
# ------------------------------
bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

usage() {
  cat <<'USAGE'
Usage:
  ./run_all.sh <HOLDINGS_CSV> <TXNS_CSV> [options]

Positional:
  HOLDINGS_CSV   Path to the Portfolio Positions CSV you downloaded from Fidelity
  TXNS_CSV       Path to the Accounts History CSV you downloaded from Fidelity

Options (passed to build_performance_dashboard.py unless noted):
  --no-live                Do NOT add GOOGLEFINANCE formulas to Open_Positions
  --strict-signals         Do NOT fallback to a later/any signal if a BUY has no prior signal
  --sell-cutoff YYYY-MM-DD Ignore unmatched SELLs on/after this date (treat as legacy exits)
  --debug                  Verbose debug for upload + build

Pipeline control (handled here):
  --skip-upload            Skip uploading the CSVs to Sheets
  --skip-merge             Skip the "merge_fidelity_with_signals.py" step

Examples:
  ./run_all.sh Portfolio_Positions_Oct-24-2025.csv Accounts_History_10-24-2025.csv
  ./run_all.sh holdings.csv tx.csv --no-live --strict-signals --sell-cutoff 2025-08-01 --debug
USAGE
}

# ------------------------------
# Args
# ------------------------------
if [[ $# -lt 2 ]]; then
  usage; exit 1
fi

HOLDINGS_CSV="$1"; shift
TXNS_CSV="$1"; shift

# Defaults
BUILD_FLAGS=()
DEBUG_FLAG=""
SKIP_UPLOAD="false"
SKIP_MERGE="false"

# Parse remaining long options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-live)
      BUILD_FLAGS+=("--no-live")
      shift
      ;;
    --strict-signals)
      BUILD_FLAGS+=("--strict-signals")
      shift
      ;;
    --sell-cutoff)
      if [[ $# -lt 2 ]]; then red "Error: --sell-cutoff requires a date (YYYY-MM-DD)"; exit 2; fi
      BUILD_FLAGS+=("--sell-cutoff" "$2")
      shift 2
      ;;
    --debug)
      DEBUG_FLAG="--debug"
      BUILD_FLAGS+=("--debug")
      shift
      ;;
    --skip-upload)
      SKIP_UPLOAD="true"
      shift
      ;;
    --skip-merge)
      SKIP_MERGE="true"
      shift
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      red "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

# ------------------------------
# Sanity checks
# ------------------------------
if [[ "$SKIP_UPLOAD" != "true" ]]; then
  if [[ ! -f "$HOLDINGS_CSV" ]]; then red "Holdings CSV not found: $HOLDINGS_CSV"; exit 3; fi
  if [[ ! -f "$TXNS_CSV" ]]; then red "Transactions CSV not found: $TXNS_CSV"; exit 3; fi
fi

# Helpful echo of plan
bold "🏁 Starting pipeline…"
yellow "• Holdings:    $HOLDINGS_CSV"
yellow "• Transactions: $TXNS_CSV"
if [[ "${#BUILD_FLAGS[@]}" -gt 0 ]]; then
  yellow "• Build flags:  ${BUILD_FLAGS[*]}"
fi
if [[ -n "$DEBUG_FLAG" ]]; then
  yellow "• Debug:        ON"
fi

# Activate venv if present
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ------------------------------
# 1) Upload CSVs to Google Sheets
# ------------------------------
if [[ "$SKIP_UPLOAD" != "true" ]]; then
  bold "🔑 Authorizing & uploading CSVs to Google Sheets…"
  set -x
  python3 upload_fidelity_to_sheets.py \
    --holdings "$HOLDINGS_CSV" \
    --txns "$TXNS_CSV" \
    $DEBUG_FLAG
  { set +x; } 2>/dev/null
else
  yellow "⏭️  Skipping upload step (--skip-upload)."
fi

# ------------------------------
# 2) Merge Signals with Transactions/Holdings
# ------------------------------
if [[ "$SKIP_MERGE" != "true" ]]; then
  bold "🔗 Merging Signals with Transactions/Holdings…"
  # This script is your existing backfiller that normalizes the Signals tab
  # and replaces blank prices with GOOGLEFINANCE or yfinance where applicable.
  if [[ -f "merge_fidelity_with_signals.py" ]]; then
    set -x
    python3 merge_fidelity_with_signals.py $DEBUG_FLAG
    { set +x; } 2>/dev/null
  else
    yellow "⚠️  merge_fidelity_with_signals.py not found. Continuing without merge."
  fi
else
  yellow "⏭️  Skipping merge step (--skip-merge)."
fi

# ------------------------------
# 3) Build dashboard tabs
# ------------------------------
bold "📊 Building Performance dashboard tabs…"
set -x
python3 build_performance_dashboard.py "${BUILD_FLAGS[@]}"
STATUS=$?
{ set +x; } 2>/dev/null

if [[ $STATUS -ne 0 ]]; then
  red "❌ Dashboard build failed (exit $STATUS)."
  exit $STATUS
fi

green "🎯 All done!"
