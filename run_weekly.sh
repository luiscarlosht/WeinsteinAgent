#!/usr/bin/env bash
# run_weekly.sh – build BOTH weekly outputs:
#   1) Performance Dashboard (Google Sheets tabs)
#   2) Weinstein Weekly report email (portfolio summary + per-position + classic scan)
#
# Uses YAML ONLY to discover the Google Sheet URL.
# No --config is passed to the Python scripts.

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"

# Activate venv if present
source .venv/bin/activate 2>/dev/null || true

bold "🧾 Using config: $CONFIG_PATH"

# Read config.yaml and compute SHEET_URL.
# Priority:
#  1) sheets.sheet_url  OR sheets.daily_intake_sheet_url
#  2) google.sheet_url  (fallback to older key some users had)
#  3) Build from any of *_sheet_id under sheets (daily_intake_sheet_id / holdings_sheet_id / control_sheet_id)
SHEET_URL="$(python3 - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    print("", end=""); sys.exit(0)

def pick_url(c):
    # New-style first
    sheets = (c.get("sheets") or {})
    url = sheets.get("sheet_url") or sheets.get("daily_intake_sheet_url")
    if url:
        return url
    # Older "google.sheet_url"
    g = (c.get("google") or {})
    url = g.get("sheet_url")
    if url:
        return url
    # Build from any known sheet_id
    sid = sheets.get("daily_intake_sheet_id") or sheets.get("holdings_sheet_id") or sheets.get("control_sheet_id")
    if sid:
        return f"https://docs.google.com/spreadsheets/d/{sid}/edit"
    return ""

print(pick_url(cfg), end="")
PY
)"

export SHEET_URL

# Soft info printout (don’t fail the run if YAML preview has an issue)
{
  python3 - <<'PY' || true
import yaml, os, sys
p = os.environ.get("CONFIG_PATH","./config.yaml")
sheet_url = os.environ.get("SHEET_URL","(missing)")
try:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    s = (cfg.get("sheets") or {})
    open_tab   = s.get("open_positions_tab","Open_Positions")
    signals    = s.get("signals_tab","Signals")
    out_dir    = s.get("output_dir","./output")
    print(f"• Google Sheet: {sheet_url if sheet_url else '(missing)'}")
    print(f"• Open Positions tab: {open_tab}")
    print(f"• Signals tab:        {signals}")
    print(f"• Output dir:         {out_dir}")
except Exception as e:
    print(f"(info) Skipping YAML preview: {e}")
PY
} 2>/dev/null

# 1) Build the Google Sheets dashboard tabs (NO --config here)
bold "📊 Building portfolio dashboard (Sheets)…"
python3 build_performance_dashboard.py

# 2) Generate + email the combined Weinstein Weekly report
if [[ -z "${SHEET_URL}" ]]; then
  red "SHEET_URL not found in YAML. Set sheets.sheet_url OR sheets.daily_intake_sheet_id."
  exit 1
fi

bold "📰 Generating + emailing Weinstein Weekly report…"
python3 weinstein_report_weekly.py \
  --sheet-url "${SHEET_URL}" \
  --write \
  --email \
  --attach-html \
  --include-scan \
  --scan-universe sp500 \
  --scan-benchmark SPY \
  --scan-max-rows 200

green "✅ Weekly pipeline finished (Sheets + Weinstein report emailed)."
