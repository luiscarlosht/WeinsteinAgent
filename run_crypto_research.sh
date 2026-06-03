#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

START_DATE="${START_DATE:-2020-01-01}"
CAPITAL="${CAPITAL:-10000}"
FIDELITY_ONLY="${FIDELITY_ONLY:-0}"

ARGS=(--start "$START_DATE" --capital "$CAPITAL")
if [[ "$FIDELITY_ONLY" == "1" ]]; then
  ARGS+=(--fidelity-only)
fi

python3 crypto_abcd_ef_research.py "${ARGS[@]}"
