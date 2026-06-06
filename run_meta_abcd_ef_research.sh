#!/usr/bin/env bash
set -euo pipefail

START="${START:-2022-01-01}"
END="${END:-2026-06-05}"
CAPITAL="${CAPITAL:-20000}"
MIN_MEM_MB="${MIN_MEM_MB:-700}"
MIN_DISK_MB="${MIN_DISK_MB:-3000}"
MAX_LOAD="${MAX_LOAD:-2.5}"

OUTDIR="./output/meta_abcd_ef_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

check_guardrails () {
  echo "---- Guardrails ----"

  avail_mem_mb=$(free -m | awk '/Mem:/ {print $7}')
  disk_mb=$(df -Pm ./output | awk 'NR==2 {print $4}')
  load_1m=$(awk '{print $1}' /proc/loadavg)

  echo "Available memory MB: $avail_mem_mb"
  echo "Available disk MB:   $disk_mb"
  echo "Load 1m:             $load_1m"

  if [ "$avail_mem_mb" -lt "$MIN_MEM_MB" ]; then
    echo "❌ Not enough memory. Need ${MIN_MEM_MB}MB."
    exit 1
  fi

  if [ "$disk_mb" -lt "$MIN_DISK_MB" ]; then
    echo "❌ Not enough disk. Need ${MIN_DISK_MB}MB."
    exit 1
  fi

  python3 - <<PY
import sys
load=float("$load_1m")
max_load=float("$MAX_LOAD")
if load > max_load:
    print(f"❌ Load too high: {load} > {max_load}")
    sys.exit(1)
print("✅ Guardrails OK")
PY
}

copy_latest_outputs () {
  local name="$1"

  for kind in summary equity trades; do
    latest=$(ls -1t output/replay_portfolio_${kind}_*.csv 2>/dev/null | head -1 || true)
    if [ -n "$latest" ]; then
      cp -v "$latest" "$OUTDIR/${name}_${kind}.csv"
    fi
  done
}

run_case () {
  local name="$1"
  shift

  echo
  echo "===== Running $name ====="
  date
  check_guardrails

  python3 weinstein_replay_portfolio_backtest_fast_meta.py \
    --start "$START" \
    --end "$END" \
    --mode both \
    --capital "$CAPITAL" \
    --config ./config.yaml \
    "$@" \
    2>&1 | tee "$OUTDIR/${name}.log"

  copy_latest_outputs "$name"

  echo "✅ Finished $name"
  date
}

run_case A_baseline \
  --regime-mode off \
  --exposure-mode off \
  --signal-quality-mode off \
  --meta-strategy off

run_case B_prod_regime \
  --regime-mode prod \
  --neutral-policy current \
  --exposure-mode off \
  --signal-quality-mode off \
  --meta-strategy off

run_case C_scaled_exposure \
  --regime-mode prod \
  --neutral-policy current \
  --exposure-mode scaled \
  --signal-quality-mode off \
  --meta-strategy off

run_case D_adaptive_quality \
  --regime-mode prod \
  --neutral-policy current \
  --exposure-mode scaled \
  --signal-quality-mode adaptive \
  --meta-strategy off

run_case E_strict_quality \
  --regime-mode prod \
  --neutral-policy current \
  --exposure-mode scaled \
  --signal-quality-mode strict \
  --meta-strategy off

run_case F_meta_selector \
  --regime-mode prod \
  --neutral-policy current \
  --exposure-mode scaled \
  --signal-quality-mode adaptive \
  --meta-strategy F \
  --meta-log

echo
echo "Done. Results in: $OUTDIR"
