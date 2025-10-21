#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# activate venv
. .venv/bin/activate

# prevent overlapping runs
/usr/bin/flock -n /tmp/weinstein_watcher.lock \
  python3 weinstein_intraday_watcher.py >> ./output/intraday.log 2>&1
