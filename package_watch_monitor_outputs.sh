#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/luiscarlosht/WeinsteinAgent"
cd "$PROJECT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="output/watch_monitor_package_${STAMP}.tar.gz"

tar -czf "$OUT" \
  output/watch_monitor \
  output/intraday_debug.csv \
  output/intraday_debug_validation.csv \
  2>/dev/null || true

echo "Created: $OUT"
ls -lh "$OUT"
