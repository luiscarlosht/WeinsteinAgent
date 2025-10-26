#!/usr/bin/env bash
# run_weekly.sh — Weekly reports from Google Sheets (no local CSVs)
set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

cd "$(dirname "$0")"
# activate venv if present
source .venv/bin/activate 2>/dev/null || true

choose_config() {
  if [[ -n "${CONFIG_FILE:-}" && -f "${CONFIG_FILE:-}" ]]; then
    printf "%s" "$CONFIG_FILE"; return 0
  fi
  for c in ./config.yaml ./config.yml ./.config.yaml ./.config.yml; do
    [[ -f "$c" ]] && { printf "%s" "$c"; return 0; }
  done
  return 1
}

CONFIG_PATH="$(choose_config || true)" || { red "No YAML config found."; exit 2; }

if [[ ! -r "$CONFIG_PATH" ]]; then
  yellow "Config not readable — attempting chmod 644…"
  chmod 644 "$CONFIG_PATH" 2>/dev/null || { red "Cannot read $CONFIG_PATH."; exit 3; }
fi

bold "🧾 Using config: $CONFIG_PATH"

# Pull SHEET_URL (and optional tab/output names) from YAML.
# We accept several key paths so your YAML can be flexible.
eval "$(
python3 - "$CONFIG_PATH" <<'PY'
import sys, yaml, json
cfg_path = sys.argv[1]

with open(cfg_path, 'r', encoding='utf-8') as f:
    d = yaml.safe_load(f) or {}

def dig(paths):
    for p in paths:
        x = d
        ok = True
        for k in p:
            if isinstance(x, dict) and k in x:
                x = x[k]
            else:
                ok = False; break
        if ok:
            return x
    return None

sheet_url = dig([
    ('google','sheet_url'),
    ('sheets','url'),
    ('sheet_url',),
    ('google','spreadsheet_url'),
    ('google','sheet'),
])

ops_tab = dig([('sheets','open_positions_tab')]) or ''
sig_tab = dig([('sheets','signals_tab')]) or ''
out_dir = dig([('sheets','output_dir')]) or ''

if not sheet_url:
    print('echo "Missing Google Sheets URL in config.yaml." >&2')
    print('echo "Add, for example:" >&2')
    print('echo "  google:" >&2')
    print('echo "    sheet_url: \\"https://docs.google.com/spreadsheets/d/.../edit\\"" >&2')
    print('exit 5')
    raise SystemExit

def sh_escape(s):  # minimal safe
    return s.replace('"', '\\"')

print(f'SHEET_URL="{sh_escape(sheet_url)}"')
if ops_tab: print(f'OPEN_POS_TAB="{sh_escape(ops_tab)}"')
if sig_tab: print(f'SIGNALS_TAB="{sh_escape(sig_tab)}"')
if out_dir: print(f'OUTPUT_DIR="{sh_escape(out_dir)}"')
PY
)"

# Export internally so legacy python expects SHEET_URL but user doesn’t have to set it.
export SHEET_URL

[[ -n "${OPEN_POS_TAB:-}" ]] && export OPEN_POS_TAB
[[ -n "${SIGNALS_TAB:-}"  ]] && export SIGNALS_TAB
[[ -n "${OUTPUT_DIR:-}"   ]] && export OUTPUT_DIR

yellow "• Google Sheet: $SHEET_URL"
[[ -n "${OPEN_POS_TAB:-}" ]] && yellow "• Open Positions tab: $OPEN_POS_TAB"
[[ -n "${SIGNALS_TAB:-}"  ]] && yellow "• Signals tab:        $SIGNALS_TAB"
[[ -n "${OUTPUT_DIR:-}"   ]] && yellow "• Output dir:         $OUTPUT_DIR"

# 1) Build/refresh the performance dashboard from the Sheet
bold "📊 Building portfolio dashboard (Sheets)…"
python3 build_performance_dashboard.py --config "$CONFIG_PATH"

# 2) Build Weinstein Weekly signals (reads price data & outputs the big table)
bold "📈 Building Weinstein Weekly signals…"
python3 weinstein_report_weekly.py --config "$CONFIG_PATH"

green "🎯 Weekly run finished (dashboard + signals from Google Sheets)."
