#!/usr/bin/env bash
# Run the weekly report (Holdings + Universe) in one go.
# Auto-reads SHEET_URL from YAML so you don't have to set an env var.

set -euo pipefail

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

# --- Find config YAML --------------------------------------------------------
CONFIG_FILE_DEFAULTS=(
  "${ROOT_DIR}/config.yml"
  "${ROOT_DIR}/config.yaml"
  "${ROOT_DIR}/settings.yml"
  "${ROOT_DIR}/settings.yaml"
  "${ROOT_DIR}/weinstein.yml"
  "${ROOT_DIR}/weinstein.yaml"
)

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
  for c in "${CONFIG_FILE_DEFAULTS[@]}"; do
    if [[ -f "$c" ]]; then CONFIG_FILE="$c"; break; fi
  done
fi

if [[ -z "${CONFIG_FILE}" || ! -f "${CONFIG_FILE}" ]]; then
  echo "[run_weekly] Could not find a YAML config. Set CONFIG_FILE=/path/to/your.yml"
  exit 1
fi

# --- Extract SHEET_URL from YAML (recursively) -------------------------------
PYCODE=$(cat <<'PY'
import sys, re, json

# Try PyYAML if available; otherwise do a very loose grep-style fallback
def load_yaml(path):
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def iter_kv(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else str(k)
            yield kp, v
            yield from iter_kv(v, kp)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            kp = f"{path}[{i}]"
            yield kp, v
            yield from iter_kv(v, kp)

def looks_like_sheet_url(s):
    return isinstance(s, str) and ("docs.google.com/spreadsheets" in s or "google" in s and "sheet" in s.lower())

def key_is_sheety(k):
    k = k.lower()
    return any(w in k for w in ["sheet", "spreadsheet", "gsheet", "gspread"])

path = sys.argv[1]
data = load_yaml(path)

# Preferred: key looks "sheety" AND value looks like a sheet url
if data is not None:
    candidates = []
    for k, v in iter_kv(data):
        if looks_like_sheet_url(v):
            score = 0
            if key_is_sheety(k): score += 2
            if "url" in k.lower(): score += 1
            candidates.append((score, k, v))
    if candidates:
        # Best-scoring candidate first
        candidates.sort(key=lambda x: (-x[0], x[1]))
        print(candidates[0][2])
        sys.exit(0)

# Fallback: grep the file for a plausible URL
import re
txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
m = re.search(r'https?://[^\s"]*docs\.google\.com/[^\s"]*spreadsheets[^\s"]*', txt)
if m:
    print(m.group(0))
    sys.exit(0)

sys.exit(2)
PY
)

SHEET_URL="$(python3 - <<PY
${PYCODE}
PY
"${CONFIG_FILE}"
)"

if [[ -z "${SHEET_URL}" ]]; then
  echo "[run_weekly] Could not extract a Google Sheet URL from: ${CONFIG_FILE}"
  echo "            Provide CONFIG_FILE=/path/to/config.yml or ensure it contains a docs.google.com/spreadsheets URL."
  exit 1
fi

# --- Optional controls (still supported but not required) --------------------
EMAIL_FLAG="--email"
if [[ "${NO_EMAIL:-0}" == "1" ]]; then EMAIL_FLAG=""; fi

ATTACH_FLAG="--attach-html"
if [[ "${NO_ATTACH:-0}" == "1" ]]; then ATTACH_FLAG=""; fi

CSV_FLAG=""
if [[ "${UNIVERSE_CSV:-0}" == "1" ]]; then CSV_FLAG="--export-csv"; fi

EXTRA_ARGS="${*:-}"
ts="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/weekly_${ts}.log"

echo "[run_weekly] Using config: ${CONFIG_FILE}"
echo "[run_weekly] Sheet URL:   ${SHEET_URL}"

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
