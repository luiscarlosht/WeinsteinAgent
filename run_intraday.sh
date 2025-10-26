#!/usr/bin/env bash
# Intraday watcher. No .env; reads sheet URL from YAML.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"; mkdir -p "$LOG_DIR"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

CANDIDATES=(config.yml config.yaml settings.yml settings.yaml weinstein.yml weinstein.yaml)
CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
  for f in "${CANDIDATES[@]}"; do
    [[ -f "${ROOT_DIR}/${f}" ]] && CONFIG_FILE="${ROOT_DIR}/${f}" && break
  done
fi
[[ -z "${CONFIG_FILE}" ]] && { echo "[intraday] no YAML; set CONFIG_FILE=/path/to.yml"; exit 1; }

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
[[ -z "${SHEET_URL}" ]] && { echo "[intraday] couldn’t find a Google Sheet URL in ${CONFIG_FILE}"; exit 1; }

ts="$(date +'%Y%m%d_%H%M%S')"
LOG="${LOG_DIR}/intraday_${ts}.log"

set -x
python3 "${ROOT_DIR}/weinstein_intraday.py" \
  --sheet-url "${SHEET_URL}" \
  --quiet |& tee -a "${LOG}"
set +x

echo "[intraday] done → ${LOG}"
