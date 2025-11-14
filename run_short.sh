#!/usr/bin/env bash
set -euo pipefail

# Default paths
CFG="./config.yaml"
CSV="./output/short_debug.csv"

# Collect extra args to forward to the watcher (e.g. --test-ease, --dry-run, --only XYZ)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CFG="$2"
      shift 2
      ;;
    --log-csv)
      CSV="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "⚡ Short watcher using config: ${CFG}"
echo "• Running: python3 weinstein_short_watcher.py --config ${CFG} --log-csv ${CSV} ${EXTRA_ARGS[*]:-}"

python3 weinstein_short_watcher.py \
  --config "${CFG}" \
  --log-csv "${CSV}" \
  "${EXTRA_ARGS[@]:-}"

echo "✅ Short tick complete."
echo "📄 Debug CSV: ${CSV}"
