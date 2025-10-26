#!/usr/bin/env bash
# run_weekly.sh — Weekly: build portfolio dashboard + Weinstein signals
set -euo pipefail

# --- pretty prints
bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

cd "$(dirname "$0")"

# --- activate venv if present
source .venv/bin/activate 2>/dev/null || true

# --- pick a config file (ENV overrides; otherwise first matching)
choose_config() {
  if [[ -n "${CONFIG_FILE:-}" && -f "${CONFIG_FILE:-}" ]]; then
    printf "%s" "$CONFIG_FILE"; return 0
  fi
  for c in ./config.yaml ./config.yml ./.config.yaml ./.config.yml; do
    [[ -f "$c" ]] && { printf "%s" "$c"; return 0; }
  done
  return 1
}

CONFIG_PATH="$(choose_config || true)" || {
  red "No YAML config found. Provide one via CONFIG_FILE=... or create config.yaml"
  exit 2
}

# --- ensure readable
if [[ ! -r "$CONFIG_PATH" ]]; then
  yellow "Config not readable — attempting chmod 644…"
  chmod 644 "$CONFIG_PATH" 2>/dev/null || {
    red "Still cannot read $CONFIG_PATH — fix permissions and retry."
    exit 3
  }
fi

bold "🧾 Using config: $CONFIG_PATH"

# --- extract inputs for the portfolio dashboard from YAML (NO env needed)
# Expected keys (adjust to your YAML):
# fidelity.holdings_csv
# fidelity.txns_csv
eval "$(
python3 - <<'PY' "$CONFIG_PATH"
import sys, json
from pathlib import Path

cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
if not cfg_path:
    print('echo "YAML reader error: missing config path." >&2'; print('exit 4'))
    raise SystemExit

try:
    import yaml
except Exception:
    print('echo "PyYAML not installed. pip install pyyaml" >&2'; print('exit 4'))
    raise SystemExit

with open(cfg_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}

def dig(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d

holdings = (
    dig(data, 'fidelity', 'holdings_csv') or
    dig(data, 'inputs',   'holdings_csv')
)
txns = (
    dig(data, 'fidelity', 'txns_csv') or
    dig(data, 'inputs',   'txns_csv')
)

if not holdings or not txns:
    print('echo "Missing CSV paths in YAML. Expected keys:', 
          'fidelity.holdings_csv and fidelity.txns_csv" >&2')
    print('exit 5')
    raise SystemExit

# Normalize to repo-relative paths where possible
def clean_path(p):
    return str(Path(p).expanduser().resolve())

print(f'HOLDINGS_CSV="{clean_path(holdings)}"')
print(f'TXNS_CSV="{clean_path(txns)}"')
PY
)"

# --- show what we found
yellow "• Holdings CSV:    $HOLDINGS_CSV"
yellow "• Transactions CSV: $TXNS_CSV"

# --- 1) build the portfolio dashboard (“Currently”) via your existing pipeline
bold "📊 Building portfolio dashboard (run_all.sh)…"
./run_all.sh "$HOLDINGS_CSV" "$TXNS_CSV"

# --- 2) build the Weinstein signals report (“What was before”)
# Your existing weekly script typically is weinstein_report_weekly.py
# and it already reads everything from the YAML.
bold "📈 Building Weinstein Weekly signals…"
python3 weinstein_report_weekly.py --config "$CONFIG_PATH"

green "🎯 Weekly run finished (dashboard + signals)."
