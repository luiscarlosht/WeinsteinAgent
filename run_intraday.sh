#!/usr/bin/env bash
# run_intraday.sh — Intraday watcher (reads YAML only)
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true

choose_config() {
  if [[ -n "${CONFIG_FILE:-}" && -f "${CONFIG_FILE:-}" ]]; then
    printf "%s" "$CONFIG_FILE"; return 0
  fi
  for c in ./config.yaml ./config.yml ./.config.yaml ./.config.yml; do
    [[ -f "$c" ]] && { printf "%s" "$c"; return 0; }
  done
  return 1
}

CONFIG_PATH="$(choose_config || true)" || {
  echo "No YAML config found. Provide CONFIG_FILE=... or create config.yaml" >&2
  exit 2
}

[[ -r "$CONFIG_PATH" ]] || chmod 644 "$CONFIG_PATH" 2>/dev/null || {
  echo "Cannot read $CONFIG_PATH — fix permissions." >&2
  exit 3
}

echo "Using config: $CONFIG_PATH"
python3 weinstein_intraday_watch.py --config "$CONFIG_PATH"
