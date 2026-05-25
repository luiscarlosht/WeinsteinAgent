#!/usr/bin/env bash
# Load WeinsteinAgent local environment variables.
# Usage from cron/scripts:
#   source ./load_weinstein_env.sh

set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
