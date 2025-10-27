#!/usr/bin/env bash
# run_weekly.sh – build BOTH weekly outputs:
#   1) Performance Dashboard (Google Sheets tabs)
#   2) Weinstein Weekly report (file export)
#
# Reads Google Sheet URL/tabs and output dir from a YAML config (default ./config.yaml).
# No need for SHEET_URL in .env anymore.

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"

# Allow override via CLI: CONFIG_FILE=./myconfig.yaml ./run_weekly.sh
if [[ ! -r "$CONFIG_PATH" ]]; then
  red "Config file not found or unreadable: $CONFIG_PATH"
  red "Set CONFIG_FILE=./myconfig.yaml or create ./config.yaml"
  exit 2
fi

# Activate venv if present
source .venv/bin/activate 2>/dev/null || true

bold "🧾 Using config: $CONFIG_PATH"

# Print the essentials by reading YAML with Python (no dependence on SHEET_URL env)
python3 - <<PY
import sys, yaml
p = "$CONFIG_PATH"
try:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"Failed to read YAML: {e}", file=sys.stderr)
    sys.exit(1)

sheets = cfg.get("sheets", {})
print(f"• Google Sheet: {sheets.get('sheet_url','(missing)')}")
print(f"• Open Positions tab: {sheets.get('open_positions_tab','Open_Positions')}")
print(f"• Signals tab:        {sheets.get('signals_tab','Signals')}")
print(f"• Output dir:         {sheets.get('output_dir','./output')}")
PY

# 1) Build the Google Sheets dashboard tabs
bold "📊 Building portfolio dashboard (Sheets)…"
python3 build_performance_dashboard.py --config "$CONFIG_PATH" "$@"

# 2) Generate the Weinstein Weekly report (PDF/MD/HTML as your script does)
bold "📰 Generating Weinstein Weekly report…"
python3 weinstein_report_weekly.py --config "$CONFIG_PATH" "$@"

green "✅ Weekly pipeline finished (Sheets + Weinstein report)."
