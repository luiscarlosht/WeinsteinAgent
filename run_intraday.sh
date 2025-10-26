#!/usr/bin/env bash
# Intraday watcher/alerts. Auto-reads SHEET_URL from YAML.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "[run_intraday] .venv not found."
  exit 1
fi

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
  echo "[run_intraday] Could not find a YAML config. Set CONFIG_FILE=/path/to/your.yml"
  exit 1
fi

PYCODE=$(cat <<'PY'
import sys, re, json
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
if data is not None:
    candidates = []
    for k, v in iter_kv(data):
        if looks_like_sheet_url(v):
            score = 0
            if key_is_sheety(k): score += 2
            if "url" in k.lower(): score += 1
            candidates.append((score, k, v))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        print(candidates[0][2]); sys.exit(0)
import re
txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
m = re.search(r'https?://[^\s"]*docs\.google\.com/[^\s"]*spreadsheets[^\s"]*', txt)
if m:
    print(m.group(0)); sys.exit(0)
sys.exit(2)
PY
)

SHEET_URL="$(python3 - <<PY
${PYCODE}
PY
"${CONFIG_FILE}"
)"

if [[ -z "${SHEET_URL}" ]]; then
  echo "[run_intraday] Could not extract a Google Sheet URL from: ${CONFIG_FILE}"
  exit 1
fi

ts="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/intraday_${ts}.log"

set -x
python3 "${ROOT_DIR}/weinstein_intraday.py" \
  --sheet-url "${SHEET_URL}" \
  --quiet \
  |& tee -a "${LOG_FILE}"
set +x
