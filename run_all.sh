#!/usr/bin/env bash
# run_all.sh – upload + merge + build + export-signals + simulate

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

Simulation flags:
  --sim-year YEAR      Run backtest for a specific YEAR (default: current year)
  --sim-capital AMT    Starting capital (default: 10000)
  --sim-risk RATE      Risk fraction per trade (default: 0.01)
  --sim-max-long N     Max concurrent long positions (default: 10)
  --sim-skip           Skip simulation step

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

SIM_YEAR=$(date +%Y)
SIM_CAPITAL=10000
SIM_RISK=0.01
SIM_MAX_LONG=10
SIM_RUN=true

SKIP_UPLOAD=false
SKIP_MERGE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    # build flags
    --no-live|--strict-signals|--debug)
      BUILD_FLAGS+=("$1"); shift ;;
    --sell-cutoff)
      [[ $# -ge 2 ]] || { red "--sell-cutoff needs DATE"; exit 2; }
      BUILD_FLAGS+=("--sell-cutoff" "$2"); shift 2 ;;

    # merge flags
    --merge-debug)
      MERGE_FLAGS+=("--debug"); shift ;;
    --merge-strict)
      MERGE_FLAGS+=("--strict"); shift ;;
    --merge-no-google)
      MERGE_FLAGS+=("--no-google"); shift ;;

    # simulation flags
    --sim-year)
      SIM_YEAR="$2"; shift 2 ;;
    --sim-capital)
      SIM_CAPITAL="$2"; shift 2 ;;
    --sim-risk)
      SIM_RISK="$2"; shift 2 ;;
    --sim-max-long)
      SIM_MAX_LONG="$2"; shift 2 ;;
    --sim-skip)
      SIM_RUN=false; shift ;;

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
yellow "• Holdings:       $HOLDINGS_CSV"
yellow "• Transactions:   $TXNS_CSV"
yellow "• Merge flags:    ${MERGE_FLAGS[*]:-(none)}"
yellow "• Build flags:    ${BUILD_FLAGS[*]:-(none)}"
yellow "• Simulation year: $SIM_YEAR"
yellow "• Simulation on:   $SIM_RUN"

# Load venv
source .venv/bin/activate 2>/dev/null || true

# STEP 1: Upload CSVs
if ! $SKIP_UPLOAD; then
  bold "🔑 Uploading CSVs to Google Sheets…"
  python3 upload_fidelity_to_sheets.py --holdings "$HOLDINGS_CSV" --txns "$TXNS_CSV"
else
  yellow "⏭️ Skip upload."
fi

# STEP 2: Merge
if ! $SKIP_MERGE; then
  bold "🔗 Merging Signals with Transactions/Holdings…"
  python3 merge_fidelity_with_signals.py "${MERGE_FLAGS[@]}"
else
  yellow "⏭️ Skip merge."
fi

# STEP 3: Build Dashboard
bold "📊 Building Performance Dashboard…"
python3 build_performance_dashboard.py "${BUILD_FLAGS[@]}"

# STEP 4: Export Signals
bold "📤 Exporting Signals from Google Sheets → ./output/signals_log.csv"
python3 export_signals_from_sheets.py \
  --config config.yaml \
  --start "${SIM_YEAR}-01-01" \
  --end   "${SIM_YEAR}-12-31" \
  --output ./output/signals_log.csv

# STEP 5: Run Backtest (optional)
if $SIM_RUN; then
  bold "📈 Running Live-Logic Backtest for $SIM_YEAR…"
  python3 weinstein_live_logic_backtest.py \
    --start "${SIM_YEAR}-01-01" \
    --end   "${SIM_YEAR}-12-31" \
    --capital "$SIM_CAPITAL" \
    --risk-per-trade "$SIM_RISK" \
    --max-long "$SIM_MAX_LONG" \
    --mode long \
    --save-trades "./data/weinstein_sim_trades_${SIM_YEAR}.csv"
else
  yellow "⏭️ Skipping simulation."
fi

green "🎯 All done!"
