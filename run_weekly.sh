#!/usr/bin/env bash
# run_weekly.sh – build BOTH weekly outputs:
#   1) Performance Dashboard (Google Sheets tabs)
#   2) Weinstein Weekly report (file export)
#
# Uses YAML ONLY for the Weinstein report. No --config is sent to build_performance_dashboard.py.

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"

# Activate venv if present
source .venv/bin/activate 2>/dev/null || true

bold "🧾 Using config: $CONFIG_PATH"

# Print a few YAML fields for info only (OK if missing)
python3 - <<PY || true
import yaml, sys, os
p = os.environ.get("CONFIG_PATH","./config.yaml")
try:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sheets = cfg.get("sheets", {})
    print(f"• Google Sheet: {sheets.get('sheet_url','(missing)')}")
    print(f"• Open Positions tab: {sheets.get('open_positions_tab','Open_Positions')}")
    print(f"• Signals tab:        {sheets.get('signals_tab','Signals')}")
    print(f"• Output dir:         {sheets.get('output_dir','./output')}")
except Exception as e:
    print(f"(info) Skipping YAML preview: {e}", file=sys.stderr)
PY

# 1) Build the Google Sheets dashboard tabs (NO --config here)
bold "📊 Building portfolio dashboard (Sheets)…"
python3 build_performance_dashboard.py "$@"

# 2) Generate the Weinstein Weekly report (this one DOES take --config)
bold "📰 Generating Weinstein Weekly report…"
python3 weinstein_report_weekly.py --config "$CONFIG_PATH" "$@"

green "✅ Weekly pipeline finished (Sheets + Weinstein report)."
