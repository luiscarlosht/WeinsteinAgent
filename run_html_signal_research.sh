#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/home/luiscarlosht/WeinsteinAgent"
cd "$PROJECT_DIR"
DAYS="${DAYS:-45}"
OUT_DIR="${OUT_DIR:-output/html_signal_research}"
python3 weinstein_html_signal_research.py --days "$DAYS" --out-dir "$OUT_DIR"
