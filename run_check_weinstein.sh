#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/WeinsteinAgent"
cd "$BASE"

today_ymd=$(date +%Y%m%d)
today_iso=$(date +%Y-%m-%d)

echo "=== Weinstein checks for ${today_iso} ==="
echo

echo "---- Intraday HTML (today) ----"
ls -lt "output/intraday_watch_${today_ymd}_"*.html 2>/dev/null | head || \
  echo "No intraday HTML for today."

echo
echo "---- Short HTML (today) ----"
ls -lt "output/short_watch_${today_ymd}_"*.html 2>/dev/null | head || \
  echo "No short HTML for today."

echo
echo "---- Cron intraday entries (cron.log) ----"
grep "${today_iso}" cron.log 2>/dev/null | grep -i "Intraday watcher starting" | tail || \
  echo "No intraday lines for today in cron.log."

echo
echo "---- Cron short entries (cron_short.log) ----"
grep "${today_iso}" cron_short.log 2>/dev/null | tail || \
  echo "No short lines for today in cron_short.log (or file missing)."

echo
echo "---- Cron short 390 entries (cron_short_390.log) ----"
grep "${today_iso}" cron_short_390.log 2>/dev/null | tail || \
  echo "No short-390 lines for today in cron_short_390.log (or file missing)."

echo
echo "---- Crypto cron entries (cron_crypto.log, if present) ----"
grep "${today_iso}" cron_crypto.log 2>/dev/null | tail || \
  echo "No crypto lines for today in cron_crypto.log (or file missing)."

echo
echo "Done."
