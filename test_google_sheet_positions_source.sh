#!/usr/bin/env bash
set -euo pipefail

cd /home/luiscarlosht/WeinsteinAgent

echo "Testing Google Sheet Holdings source..."
POSITIONS_SOURCE=GOOGLE_SHEET POSITIONS_CSV=GOOGLE_SHEET python3 weinstein_positions_source.py

echo
echo "Testing account profile loader with Google Sheet source..."
POSITIONS_SOURCE=GOOGLE_SHEET POSITIONS_CSV=GOOGLE_SHEET python3 weinstein_account_profiles.py GOOGLE_SHEET
