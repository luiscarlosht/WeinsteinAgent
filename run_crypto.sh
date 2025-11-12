#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate 2>/dev/null || true
echo "⚡ Crypto watcher using config: ./config.yaml"
python3 weinstein_crypto_watcher.py --config ./config.yaml "$@"
