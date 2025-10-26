#!/usr/bin/env bash
# Weekly: builds "what is now" (holdings snapshot) AND "what was before" (universe scan)
# Pulls the Google Sheet URL from a YAML, no .env needed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"; mkdir -p "$LOG_DIR"

# --- venv ---------------------------------------------------------
if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

# --- locate YAML --------------------------------------------------
CANDIDATES=(config.yml config.yaml settings.yml settings.yaml weinstein.yml weinstein.yaml)
CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
  for f in "${CANDIDATES[@]}"; do
    [[ -f "${ROOT_DIR}/${f}" ]] && CONFIG_FILE="${ROOT_DIR}/${f}" && break
  done
fi
[[ -z "${CONFIG_FILE}" ]] && { echo "[weekly] no YAML found; set CONFIG_FILE=/path/to.yml"; exit 1; }

# --- extract sheet url from YAML ---------------------------------
read -r -d '' PY <<'PY' || true
import sys,re
def load_yaml(p):
    try:
        import yaml
        with open(p,'r',encoding='utf-8') as f: return yaml.safe_load(f)
    except Exception: return None
def iter_kv(x):
    if isinstance(x,dict):
        for k,v in x.items():
            yield k,v
            yield from ( (f"{k}.{kk}",vv) for kk,vv in iter_kv(v) )
    elif isinstance(x,list):
        for i,v in enumerate(x):
            yield f"[{i}]",v
            yield from iter_kv(v)
def looks_url(s): return isinstance(s,str) and "docs.google.com/spreadsheets" in s
def sheety(k): k=k.lower(); return any(w in k for w in ("sheet","spreadsheet","gsheet","gspread"))
p=sys.argv[1]
y=load_yaml(p)
if y is not None:
    c=[]
    for k,v in iter_kv(y):
        if looks_url(v):
            score=(2 if sheety(k) else 0)+(1 if "url" in k.lower() else 0)
            c.append((score,k,v))
    if c:
        c.sort(key=lambda t:(-t[0],t[1])); print(c[0][2]); sys.exit(0)
txt=open(p,'r',encoding='utf-8',errors='ignore').read()
m=re.search(r'https?://\S*docs\.google\.com/\S*spreadsheets\S*',txt)
if m: print(m.group(0)); sys.exit(0)
sys.exit(2)
PY
SHEET_URL="$(python3 - <<PY
$PY
PY
"${CONFIG_FILE}"
)"
[[ -z "${SHEET_URL}" ]] && { echo "[weekly] couldn’t find a Google Sheet URL in ${CONFIG_FILE}"; exit 1; }

# --- args & log ---------------------------------------------------
EXTRA_ARGS="${*:-}"
ts="$(date +'%Y%m%d_%H%M%S')"
LOG="${LOG_DIR}/weekly_${ts}.log"

echo "[weekly] YAML: ${CONFIG_FILE}"
echo "[weekly] SHEET: ${SHEET_URL}"

set -x
# "what is now" + "what was before" (universe scan)
python3 "${ROOT_DIR}/weinstein_report_weekly.py" \
  --write \
  --scan-universe \
  --sheet-url "${SHEET_URL}" \
  ${EXTRA_ARGS} |& tee -a "${LOG}"
set +x

echo "[weekly] done → ${LOG}"
