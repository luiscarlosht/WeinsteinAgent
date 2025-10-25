#!/usr/bin/env bash
# Runs the full pipeline:
# 1) Upload Fidelity CSVs -> Google Sheet (Holdings & Transactions tabs)
# 2) Merge Signals with Fidelity data
# 3) Build/refresh performance dashboard tabs

set -euo pipefail

# ---- usage/help --------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage:
  ./run_all.sh <HOLDINGS_CSV> <TRANSACTIONS_CSV>

Examples:
  ./run_all.sh Portfolio_Positions_Oct-24-2025.csv Accounts_History_10-24-2025.csv

Notes:
- Requires your virtualenv at ./.venv already set up with project deps.
- Uses the sheet URL embedded in the Python scripts.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
  usage
  exit 1
fi

HOLDINGS_CSV="$1"
TXNS_CSV="$2"

# ---- sanity checks -----------------------------------------------------------
if [[ ! -f "$HOLDINGS_CSV" ]]; then
  echo "❌ Holdings CSV not found: $HOLDINGS_CSV" >&2
  exit 2
fi
if [[ ! -f "$TXNS_CSV" ]]; then
  echo "❌ Transactions CSV not found: $TXNS_CSV" >&2
  exit 2
fi

if [[ ! -d ".venv" ]]; then
  echo "❌ Python venv not found at .venv"
  echo "   Create it (example):  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
  exit 3
fi

# ---- activate venv -----------------------------------------------------------
# shellcheck source=/dev/null
source ".venv/bin/activate"

# ---- step 1: upload to Google Sheets ----------------------------------------
echo "🔑 Authorizing & uploading CSVs to Google Sheets…"
python3 upload_fidelity_to_sheets.py \
  --holdings "$HOLDINGS_CSV" \
  --txns "$TXNS_CSV"

# ---- step 2: merge signals with fidelity data -------------------------------
echo "🔗 Merging Signals with Transactions/Holdings…"
python3 merge_fidelity_with_signals.py --verbose

# ---- step 3: build dashboard tabs -------------------------------------------
echo "📊 Building Performance dashboard tabs…"
python3 build_performance_dashboard.py --debug

echo "🎯 All done!"
