#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/home/luiscarlosht/WeinsteinAgent"
cd "$PROJECT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="output/prod_signal_research_package_${STAMP}.tar.gz"
tar -czf "$OUT" output/html_signal_research output/intraday_signal_history.csv 2>/dev/null || true
echo "Created: $OUT"
ls -lh "$OUT"
