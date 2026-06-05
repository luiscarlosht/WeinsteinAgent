#!/usr/bin/env bash
set -euo pipefail

BASE="${WEINSTEIN_BASE:-$HOME/WeinsteinAgent}"
cd "$BASE"

today_ymd="$(date +%Y%m%d)"
today_iso="$(date +%Y-%m-%d)"

line() { echo; echo "---- $1 ----"; }
pass() { echo "✅ $1"; }
warn() { echo "⚠️  $1"; }
fail() { echo "❌ $1"; }

latest_file() {
  local pattern="$1"
  ls -1t $pattern 2>/dev/null | head -1 || true
}

show_recent_files() {
  local title="$1"
  local pattern="$2"
  line "$title"
  local files
  files="$(ls -lt $pattern 2>/dev/null | head -10 || true)"
  if [[ -n "$files" ]]; then
    echo "$files"
  else
    warn "No files found for pattern: $pattern"
  fi
}

show_log_tail() {
  local title="$1"
  local logfile="$2"
  local pattern="${3:-}"
  line "$title"
  if [[ ! -f "$logfile" ]]; then
    warn "$logfile missing"
    return 0
  fi

  if [[ -n "$pattern" ]]; then
    local out
    out="$(grep -E "$pattern" "$logfile" 2>/dev/null | tail -20 || true)"
    if [[ -n "$out" ]]; then
      echo "$out"
    else
      warn "No matching lines in $logfile for: $pattern"
    fi
  else
    tail -30 "$logfile" || true
  fi
}

check_log_errors() {
  local logfile="$1"
  line "Error scan: $logfile"
  if [[ ! -f "$logfile" ]]; then
    warn "$logfile missing"
    return 0
  fi

  local errors
  errors="$(grep -Ei "Traceback|KeyError|ValueError|Exception|ERROR|failed|fatal|aborting|cannot|unrecognized arguments" "$logfile" 2>/dev/null | tail -30 || true)"
  if [[ -n "$errors" ]]; then
    fail "Potential errors found in $logfile"
    echo "$errors"
  else
    pass "No obvious errors found in $logfile"
  fi
}

echo "=== Weinstein health check for ${today_iso} ==="
echo "Base: $BASE"
echo "Time: $(date)"
echo

line "Git"
git status --short | head -60 || true
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

show_recent_files "Intraday HTML today" "output/intraday_watch_${today_ymd}_"*.html
show_recent_files "Crypto HTML today" "output/crypto_watch_${today_ymd}_"*.html
show_recent_files "Daily command center today" "output/daily_command_center/${today_ymd}_"*
show_recent_files "Daily parity today" "output/daily_parity/${today_ymd}_"*
show_recent_files "Attribution recent" "output/attribution/"*

line "Latest intraday diagnostics"
if [[ -s output/intraday_debug.csv ]]; then
  pass "output/intraday_debug.csv exists and is non-empty"
  python3 - <<'PY'
import pandas as pd
p="output/intraday_debug.csv"
df=pd.read_csv(p)
print(f"rows={len(df)} cols={len(df.columns)}")
if "Signal" in df.columns:
    print(df["Signal"].astype(str).value_counts(dropna=False).head(20).to_string())
if "WatchSignal" in df.columns:
    ws=df["WatchSignal"].fillna("").astype(str)
    vc=ws[ws.ne("")].value_counts().head(20)
    if len(vc):
        print("\nWatchSignal:")
        print(vc.to_string())
PY
else
  fail "output/intraday_debug.csv missing or empty"
fi

line "Latest portfolio holdings in intraday report"
latest_intraday="$(latest_file "output/intraday_watch_${today_ymd}_*.html")"
if [[ -n "$latest_intraday" ]]; then
  pass "Latest intraday HTML: $latest_intraday"
  grep -E "Reviewed [0-9]+ owned tickers|Portfolio Holdings Review|Portfolio Action|Ticker not present" "$latest_intraday" | head -20 || true
else
  fail "No intraday HTML for today"
fi

show_log_tail "Cron short stack recent lines" "cron_short.log" "${today_iso}|Intraday watcher starting|Saved HTML|Traceback|KeyError|Aborting|Done"
check_log_errors "cron_short.log"

show_log_tail "Daily parity recent lines" "cron_daily_parity.log" "${today_iso}|DONE|ERROR|Traceback|SIM F trade outcomes|HTML:"
check_log_errors "cron_daily_parity.log"

show_log_tail "Prod account routing recent lines" "cron_prod_account_routing.log" "${today_iso}|DONE|ERROR|Traceback|Email sent"
check_log_errors "cron_prod_account_routing.log"

show_log_tail "Weekly recent lines" "cron_weekly.log" "${today_iso}|DONE|ERROR|Traceback|Email sent"
check_log_errors "cron_weekly.log"

line "Cron schedule"
crontab -l | grep -v '^#' | grep -v '^$' || warn "No active crontab lines"

line "Recent files modified in last 3 hours"
find output -type f -mmin -180 2>/dev/null | sort | tail -80 || true

line "Overall interpretation"
if [[ -n "$latest_intraday" && -s output/intraday_debug.csv ]]; then
  pass "Intraday stack is producing current output."
else
  fail "Intraday stack is not producing current output."
fi

echo
echo "Done."
