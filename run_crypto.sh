#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"
if [[ -d ".venv" ]]; then source .venv/bin/activate 2>/dev/null || true; fi
echo "⚡ Crypto watcher using config: $CONFIG_PATH"
python3 weinstein_crypto_watcher.py --config "$CONFIG_PATH" "$@"
