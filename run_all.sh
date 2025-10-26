#!/usr/bin/env bash
# run_all.sh – upload + merge + build

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

usage() {
  cat <<'USAGE'
Usage:
  ./run_all.sh <HOLDINGS_CSV> <TXNS_CSV> [flags]

Flags:
  --no-live           Skip GOOGLEFINANCE formulas in Open_Positions
  --strict-signals    Disable fallback signal matching
  --sell-cutoff DATE  Ignore unmatched SELLs on/after date
  --debug             Verbose debug
  --skip-upload       Skip upload step
  --skip-merge        Skip merge step
USAGE
}

if [[ $# -lt 2 ]]; then usage; exit 1; fi

HOLDINGS_CSV="$1"; shift
TXNS_CSV="$1"; shift

BUILD_FLAGS=()
DEBUG_FLAG=""
SKIP_UPLOAD=false
SKIP_MERGE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-live|--strict-signals|--debug)
      BUILD_FLAGS+=("$1"); shift ;;
    --sell-cutoff)
      BUILD_FLAGS+=("--sell-cutoff" "$2"); shift 2 ;;
    --skip-upload)
      SKIP_UPLOAD=true; shift ;;
    --skip-merge)
      SKIP_MERGE=true; shift ;;
    *) red "Unknown flag: $1"; exit 2 ;;
  esac
done

bold "🏁 Starting pipeline…"
yellow "• Holdings:    $HOLDINGS_CSV"
yellow "• Transactions: $TXNS_CSV"
yellow "• Build flags:  ${BUILD_FLAGS[*]:-(none)}"

source .venv/bin/activate || true

if ! $SKIP_UPLOAD; then
  bold "🔑 Uploading CSVs to Google Sheets…"
  python3 upload_fidelity_to_sheets.py --holdings "$HOLDINGS_CSV" --txns "$TXNS_CSV" "${DEBUG_FLAG:-}"
else
  yellow "⏭️ Skipping upload."
fi

if ! $SKIP_MERGE; then
  bold "🔗 Merging Signals with Transactions/Holdings…"
  python3 merge_fidelity_with_signals.py ${DEBUG_FLAG:-}
else
  yellow "⏭️ Skipping merge."
fi

bold "📊 Building dashboard tabs…"
python3 build_performance_dashboard.py "${BUILD_FLAGS[@]}"
green "🎯 Done!"
