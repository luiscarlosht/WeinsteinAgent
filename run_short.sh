#!/usr/bin/env bash
#
# Thin wrapper to run the Weinstein short-side watcher.
# - Uses ./config.yaml by default
# - Forwards any extra args you pass to this script
#   e.g.:
#     ./run_short.sh
#     ./run_short.sh --test-ease --dry-run
#     ./run_short.sh --only CRM --test-ease --dry-run
#

set -euo pipefail

CONFIG="./config.yaml"

echo "⚡ Short watcher using config: ${CONFIG}"

# Everything you pass to this script is forwarded to the Python program
EXTRA_ARGS=("$@")

echo -n "• Running: python3 weinstein_short_watcher.py --config ${CONFIG}"
if ((${#EXTRA_ARGS[@]})); then
  printf ' %q' "${EXTRA_ARGS[@]}"
fi
echo

python3 weinstein_short_watcher.py --config "${CONFIG}" "${EXTRA_ARGS[@]}"
