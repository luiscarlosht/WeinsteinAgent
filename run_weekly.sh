#!/usr/bin/env bash
# run_weekly.sh – build BOTH weekly outputs:
#   1) Performance Dashboard (Google Sheets tabs)
#   2) Weinstein Weekly report (HTML/CSV + Email)
#
# Uses YAML ONLY to discover the Google Sheet URL.
# No --config is passed to the Python scripts.

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"

# Activate virtual environment if present
source .venv/bin/activate 2>/dev/null || true

bold "🧾 Using config: $CONFIG_PATH"

# Read config.yaml and compute SHEET_URL (prefer sheets.sheet_url, else build from sheets.daily_intake_sheet_id)
SHEET_URL="$(python3 - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    print("", end=""); sys.exit(0)

s = (cfg.get("sheets") or {})
url = s.get("sheet_url") or s.get("daily_intake_sheet_url")  # allow either name
if not url:
    sheet_id = s.get("daily_intake_sheet_id") or s.get("holdings_sheet_id") or s.get("control_sheet_id")
    if sheet_id:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
print(url or "", end="")
PY
)"

# Soft info printout
{
  python3 - <<PY || true
import yaml, os
p = os.environ.get("CONFIG_PATH","./config.yaml")
try:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    s = cfg.get("sheets", {}) or {}
    print(f"• Google Sheet: {os.environ.get('SHEET_URL','(missing)')}")
    print(f"• Open Positions tab: {s.get('open_positions_tab','Open_Positions')}")
    print(f"• Signals tab:        {s.get('signals_tab','Signals')}")
    print(f"• Output dir:         {s.get('output_dir','./output')}")
except Exception as e:
    print(f"(info) Skipping YAML preview: {e}")
PY
} 2>/dev/null

# 1️⃣ Build the Google Sheets dashboard tabs
bold "📊 Building portfolio dashboard (Sheets)…"
python3 build_performance_dashboard.py

# 2️⃣ Generate + email the Weinstein Weekly report
if [[ -z "${SHEET_URL}" ]]; then
  red "SHEET_URL not found in YAML. Set sheets.sheet_url OR sheets.daily_intake_sheet_id."
  exit 1
fi

bold "📰 Generating + emailing Weinstein Weekly report…"
# Always send email with HTML attachment by default
python3 weinstein_report_weekly.py --sheet-url "${SHEET_URL}" --write --email --attach-html

green "✅ Weekly pipeline finished (Sheets + Weinstein report emailed)."
