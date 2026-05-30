#!/usr/bin/env bash
set -euo pipefail

cd /home/luiscarlosht/WeinsteinAgent

OUT="${POSITIONS_CSV_OUT:-/home/luiscarlosht/WeinsteinAgent/current_positions.csv}"

python3 sync_positions_from_google_sheet.py --out "$OUT"

echo "Current positions:"
ls -lh "$OUT"
head -5 "$OUT"
