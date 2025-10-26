#!/usr/bin/env bash
# Run the weekly report (Holdings + Universe) in one go.

set -euo pipefail

# --- Directories -------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

# --- Python / venv -----------------------------------------------------------
if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "[run_weekly] .venv not found. Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# --- Config via .env (optional, but recommended) -----------------------------
# If a .env exists, we load defaults from it
if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

# Required: SHEET_URL for Google Sheet
: "${SHEET_URL:?SHEET_URL is required. Put it in .env like: SHEET_URL='https://docs.google.com/...' }"

# Optional: email behavior
EMAIL_FLAG="--email"
if [[ "${NO_EMAIL:-0}" == "1" ]]; then
  EMAIL_FLAG=""   # disable sending if NO_EMAIL=1
fi

# Optional: attach HTML (universe + weekly)
ATTACH_FLAG="--attach-html"
if [[ "${NO_ATTACH:-0}" == "1" ]]; then
  ATTACH_FLAG=""
fi

# Optional: include CSV exports for universe scan
CSV_FLAG=""
if [[ "${UNIVERSE_CSV:-0}" == "1" ]]; then
  CSV_FLAG="--export-csv"
fi

# Pass-through for any extra args you specify at the CLI
EXTRA_ARGS="${*:-}"

ts="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/weekly_${ts}.log"

echo "[run_weekly] Starting weekly run at ${ts}"
echo "[run_weekly] Sheet: ${SHEET_URL}"

# Core command: write Weekly tab + run Universe scan (like the “before” table)
# - --write updates the sheet (Weekly_Report + Universe_Signals)
# - --scan-universe generates the big Buy/Watch/Avoid table + HTML/CSV
# - --attach-html attaches the HTML reports to the email (if emailing)
# - --email sends the email (omit with NO_EMAIL=1)
set -x
python3 "${ROOT_DIR}/weinstein_report_weekly.py" \
  --write \
  --sheet-url "${SHEET_URL}" \
  --scan-universe \
  ${ATTACH_FLAG} \
  ${EMAIL_FLAG} \
  ${CSV_FLAG} \
  ${EXTRA_ARGS} \
  |& tee -a "${LOG_FILE}"
set +x

echo "[run_weekly] Done. Log: ${LOG_FILE}"
