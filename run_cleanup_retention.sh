#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "===== Cleanup started: $(date -Is) ====="
df -h /

# Keep newest 5 stock parity runs.
if [ -d output/daily_parity ]; then
  (
    cd output/daily_parity
    ls -1dt */ 2>/dev/null | tail -n +6 | xargs -r rm -rf
  )
fi

# Compress large trade-outcome CSVs.
find output/daily_parity \
  -type f \
  -name "*trade_outcomes.csv" \
  -exec gzip -9 {} \; 2>/dev/null || true

# Keep newest 20 crypto validation runs.
if [ -d output/crypto_validation ]; then
  (
    cd output/crypto_validation
    ls -1dt */ 2>/dev/null | tail -n +21 | xargs -r rm -rf
  )
fi

# Keep newest 10 crypto PROD/SIM comparison runs.
if [ -d output/crypto_prod_sim_compare ]; then
  (
    cd output/crypto_prod_sim_compare
    ls -1dt */ 2>/dev/null | tail -n +11 | xargs -r rm -rf
  )
fi

# Delete regenerable artifacts after retention window.
find output -type f -name "*.html" -mtime +7 -delete 2>/dev/null || true
find output/daily_cache -type f -mtime +3 -delete 2>/dev/null || true
find output/charts -type f -mtime +7 -delete 2>/dev/null || true
find output -type f -name "*.tmp" -delete 2>/dev/null || true

# Keep research archive permanently.
find research/profile_history -type f -print >/dev/null 2>&1 || true

echo
du -xh output --max-depth=1 2>/dev/null | sort -h
echo
df -h /
echo "===== Cleanup finished: $(date -Is) ====="
