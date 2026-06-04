#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

POSITIONS_CSV="${POSITIONS_CSV:-}"
HISTORY_CSV="${HISTORY_CSV:-}"
MODE="${MODE:-dry-run}"

ARGS=()

if [[ -n "$POSITIONS_CSV" ]]; then
  ARGS+=(--positions-csv "$POSITIONS_CSV")
fi

if [[ -n "$HISTORY_CSV" ]]; then
  ARGS+=(--history-csv "$HISTORY_CSV")
fi

if [[ "$MODE" == "write" || "$MODE" == "write-sheet" ]]; then
  ARGS+=(--write-sheet)
else
  ARGS+=(--dry-run)
fi

python3 sync_fidelity_exports_to_google_sheet.py "${ARGS[@]}"
