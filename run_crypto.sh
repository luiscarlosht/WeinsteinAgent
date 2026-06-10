#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate 2>/dev/null || true
echo "⚡ Crypto watcher using config: ./config.yaml"
python3 weinstein_crypto_watcher.py --config ./config.yaml "$@"
