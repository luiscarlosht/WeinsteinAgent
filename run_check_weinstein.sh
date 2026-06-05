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

file_epoch() {
  local f="$1"
  if [[ -n "$f" && -e "$f" ]]; then
    stat -c %Y "$f"
  else
    echo 0
  fi
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

scan_errors_since_epoch() {
  local logfile="$1"
  local since_epoch="$2"
  local title="$3"

  line "Error scan: $logfile ($title)"

  if [[ ! -f "$logfile" ]]; then
    warn "$logfile missing"
    return 0
  fi

  python3 - "$logfile" "$since_epoch" <<'PY'
import datetime as dt
import re
import sys

logfile = sys.argv[1]
since_epoch = int(float(sys.argv[2] or "0"))

error_re = re.compile(
    r"Traceback|KeyError|ValueError|Exception|ERROR|failed|fatal|cannot|unrecognized arguments|SMTPAuthenticationError|Username and Password not accepted|Failed download|Failed downloads|JSONDecodeError|YFPricesMissingError",
    re.I,
)

benign_re = re.compile(
    r"short_debug\.csv is empty|shorts disabled|Aborting: short_debug CSV is missing|FutureWarning",
    re.I,
)

email_error_re = re.compile(
    r"SMTPAuthenticationError|Username and Password not accepted|BadCredentials|email failed|smtplib\.SMTPAuthenticationError",
    re.I,
)

email_success_re = re.compile(
    r"Email sent\.|DONE$|DONE daily parity run|Weekly report complete|Routing email sent",
    re.I,
)

# Yahoo can intermittently fail one ticker, often SPGI, due to a temporary JSON
# decode / quote response issue. Treat small isolated download failures as WARN.
yahoo_failure_re = re.compile(
    r"Failed download|Failed downloads|JSONDecodeError|YFPricesMissingError|possibly delisted|No data found",
    re.I,
)

# Log timestamps can appear as:
# [15:40:23]
# [2026-06-05 15:41:27]
time_re_full = re.compile(r"\[(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})\]")
time_re_hms = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")

today = dt.datetime.now().date()
latest_ts = None
events = []
pending_traceback = []

def parse_ts(line, latest):
    m = time_re_full.search(line)
    if m:
        return dt.datetime(
            int(m.group(1)[0:4]), int(m.group(1)[5:7]), int(m.group(1)[8:10]),
            int(m.group(2)), int(m.group(3)), int(m.group(4))
        )
    m = time_re_hms.search(line)
    if m:
        return dt.datetime(today.year, today.month, today.day, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return latest

def classify_yahoo_failure(line):
    # If a single ticker failed, warn only. A broad failure remains an error.
    m = re.search(r"(\d+)\s+Failed download", line, flags=re.I)
    if m:
        n = int(m.group(1))
        return "WARN" if n <= 2 else "ERROR"
    if "1 Failed download" in line:
        return "WARN"
    if "JSONDecodeError" in line and ("SPGI" in line or "Failed download" in line):
        return "WARN"
    return "ERROR"

def flush_traceback():
    global pending_traceback
    if not pending_traceback:
        return
    block = "\n".join(x[2] for x in pending_traceback)
    ts = pending_traceback[-1][1]

    if email_error_re.search(block):
        events.append(("EMAIL_ERROR_BLOCK", ts, block))
    elif yahoo_failure_re.search(block):
        events.append(("WARN", ts, block))
    else:
        events.append(("ERROR", ts, block))

    pending_traceback = []

with open(logfile, "r", errors="replace") as f:
    for raw_line in f:
        ts = parse_ts(raw_line, latest_ts)
        if ts is not None:
            latest_ts = ts
        use_ts = ts or latest_ts

        if use_ts is not None and use_ts.timestamp() < since_epoch:
            continue

        line = raw_line.rstrip()

        # Capture traceback blocks so stale SMTP tracebacks can be suppressed as
        # one unit instead of leaving orphaned "Traceback" lines behind.
        if line.startswith("Traceback (most recent call last):"):
            flush_traceback()
            pending_traceback = [("TRACEBACK", use_ts, line)]
            continue

        if pending_traceback:
            pending_traceback.append(("TRACEBACK", use_ts, line))
            # A blank line or a final exception-looking line usually ends a traceback.
            if (
                line == ""
                or re.search(r"^[A-Za-z_][\w.]*Error:", line)
                or "SMTPAuthenticationError" in line
                or "KeyError:" in line
                or "ValueError:" in line
            ):
                flush_traceback()
            continue

        if email_success_re.search(line):
            events.append(("EMAIL_OK", use_ts, line))
        elif error_re.search(line):
            if benign_re.search(line):
                events.append(("WARN", use_ts, line))
            elif email_error_re.search(line):
                events.append(("EMAIL_ERROR", use_ts, line))
            elif yahoo_failure_re.search(line):
                events.append((classify_yahoo_failure(line), use_ts, line))
            else:
                events.append(("ERROR", use_ts, line))

flush_traceback()

last_email_ok_idx = max([i for i, (kind, _, _) in enumerate(events) if kind == "EMAIL_OK"], default=-1)

current = []
stale_email_errors = []
for i, event in enumerate(events):
    kind, ts, line = event

    # Suppress stale SMTP failures and entire SMTP traceback blocks when the same
    # log later proves email recovered.
    if kind in {"EMAIL_ERROR", "EMAIL_ERROR_BLOCK"} and i < last_email_ok_idx:
        stale_email_errors.append(event)
        continue

    if kind in {"ERROR", "EMAIL_ERROR", "EMAIL_ERROR_BLOCK", "WARN"}:
        current.append(event)

errors = [x for x in current if x[0] in {"ERROR", "EMAIL_ERROR", "EMAIL_ERROR_BLOCK"}]
warns = [x for x in current if x[0] == "WARN"]

if errors:
    print("❌ Current errors found:")
    for _, _, line in errors[-20:]:
        print(line)
elif warns:
    print("⚠️  Only benign/current warnings found:")
    for _, _, line in warns[-20:]:
        print(line)
else:
    print("✅ No current errors found.")

if stale_email_errors:
    print(f"ℹ️  Suppressed {len(stale_email_errors)} stale email/auth traceback or error block(s) because a later email success marker exists in this log.")
PY
}

echo "=== Weinstein health check for ${today_iso} ==="
echo "Base: $BASE"
echo "Time: $(date)"
echo

line "Git"
git status --short | head -80 || true
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

latest_intraday="$(latest_file "output/intraday_watch_${today_ymd}_*.html")"
latest_intraday_epoch="$(file_epoch "$latest_intraday")"

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
if [[ -n "$latest_intraday" ]]; then
  pass "Latest intraday HTML: $latest_intraday"
  grep -E "Reviewed [0-9]+ owned tickers|Portfolio Holdings Review|Portfolio Action|Ticker not present" "$latest_intraday" | head -20 || true
else
  fail "No intraday HTML for today"
fi

line "Bad holdings symbol check"
bad_symbols="857480172|857480180|87283J616|58805T275|NON40TVFA|NON40TP8J"
bad_hits="$(grep -E "$bad_symbols" output/intraday_debug.csv output/intraday_watch_${today_ymd}_*.html 2>/dev/null || true)"
if [[ -n "$bad_hits" ]]; then
  fail "Bad non-Yahoo holdings symbols are still present in current outputs:"
  echo "$bad_hits" | tail -20
else
  pass "No known bad holdings symbols found in current intraday outputs."
fi

show_log_tail "Cron short stack recent lines" "cron_short.log" "${today_iso}|Intraday watcher starting|Saved HTML|Traceback|KeyError|Aborting|Done|Short tick complete|shorts disabled"
if [[ "$latest_intraday_epoch" != "0" ]]; then
  scan_errors_since_epoch "cron_short.log" "$latest_intraday_epoch" "since latest intraday HTML"
else
  scan_errors_since_epoch "cron_short.log" "$(date -d "${today_iso} 00:00:00" +%s)" "since start of today"
fi

show_log_tail "Daily parity recent lines" "cron_daily_parity.log" "${today_iso}|DONE|ERROR|Traceback|SIM F trade outcomes|HTML:"
scan_errors_since_epoch "cron_daily_parity.log" "$(date -d "${today_iso} 00:00:00" +%s)" "since start of today"

show_log_tail "Prod account routing recent lines" "cron_prod_account_routing.log" "${today_iso}|DONE|ERROR|Traceback|Email sent|SMTPAuthenticationError"
scan_errors_since_epoch "cron_prod_account_routing.log" "$(date -d "${today_iso} 00:00:00" +%s)" "since start of today"

show_log_tail "Weekly recent lines" "cron_weekly.log" "${today_iso}|DONE|ERROR|Traceback|Email sent"
scan_errors_since_epoch "cron_weekly.log" "$(date -d "${today_iso} 00:00:00" +%s)" "since start of today"

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
