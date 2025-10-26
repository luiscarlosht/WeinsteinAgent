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

Flags (merge step):
  --merge-debug        Pass --debug to merge_fidelity_with_signals.py
  --merge-strict       Pass --strict to merge_fidelity_with_signals.py
  --merge-no-google    Pass --no-google to merge_fidelity_with_signals.py

Flags (build step):
  --no-live            Skip GOOGLEFINANCE formulas in Open_Positions
  --strict-signals     Disable fallback signal matching
  --sell-cutoff DATE   Ignore unmatched SELLs on/after date (YYYY-MM-DD)
  --debug              Verbose debug for build

General:
  --skip-upload        Skip upload step
  --skip-merge         Skip merge step
USAGE
}

if [[ $# -lt 2 ]]; then usage; exit 1; fi

HOLDINGS_CSV="$1"; shift
TXNS_CSV="$1"; shift

BUILD_FLAGS=()
MERGE_FLAGS=()
SKIP_UPLOAD=false
SKIP_MERGE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    # build flags
    --no-live|--strict-signals|--debug)
      BUILD_FLAGS+=("$1"); shift ;;
    --sell-cutoff)
      [[ $# -ge 2 ]] || { red "--sell-cutoff needs a DATE"; exit 2; }
      BUILD_FLAGS+=("--sell-cutoff" "$2"); shift 2 ;;

    # merge flags
    --merge-debug)
      MERGE_FLAGS+=("--debug"); shift ;;
    --merge-strict)
      MERGE_FLAGS+=("--strict"); shift ;;
    --merge-no-google)
      MERGE_FLAGS+=("--no-google"); shift ;;

    # control
    --skip-upload)
      SKIP_UPLOAD=true; shift ;;
    --skip-merge)
      SKIP_MERGE=true; shift ;;

    *)
      red "Unknown flag: $1"; exit 2 ;;
  esac
done

bold "🏁 Starting pipeline…"
yellow "• Holdings:    $HOLDINGS_CSV"
yellow "• Transactions: $TXNS_CSV"
yellow "• Merge flags:  ${MERGE_FLAGS[*]:-(none)}"
yellow "• Build flags:  ${BUILD_FLAGS[*]:-(none)}"

# Activate venv if present
source .venv/bin/activate 2>/dev/null || true

if ! $SKIP_UPLOAD; then
  bold "🔑 Authorizing & uploading CSVs to Google Sheets…"
  python3 upload_fidelity_to_sheets.py --holdings "$HOLDINGS_CSV" --txns "$TXNS_CSV"
else
  yellow "⏭️ Skipping upload."
fi

if ! $SKIP_MERGE; then
  bold "🔗 Merging Signals with Transactions/Holdings…"
  python3 merge_fidelity_with_signals.py "${MERGE_FLAGS[@]}"
else
  yellow "⏭️ Skipping merge."
fi

bold "📊 Building Performance dashboard tabs…"
python3 build_performance_dashboard.py "${BUILD_FLAGS[@]}"
green "🎯 All done!"
