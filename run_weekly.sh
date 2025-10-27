#!/usr/bin/env bash
# run_weekly.sh – build BOTH weekly outputs:
#   1) Performance Dashboard (Google Sheets tabs)
#   2) Weinstein Weekly combined report (portfolio + classic scan)
#
# Uses YAML ONLY to discover the Google Sheet URL.
# No --config is passed to the Python scripts.
#
# Env toggles:
#   ENABLE_SCAN=1|0        (default 1) run classic Weinstein scan if watcher exists
#   SCAN_UNIVERSE=sp500    universe for classic scan
#   SCAN_BENCHMARK=SPY     benchmark for RS calc
#   SCAN_MAX_ROWS=200      max rows to include in email/attachments
#   WEEKLY_FLAGS="--email --attach-html"  let report script send the email itself
#
# Fallback mailer:
#   If the report script fails emailing, we try sending output/combined_weekly_email.html
#   using WeinsteinMinimalEmailSender and recipients in config.yaml under:
#     email:
#       to: ["you@example.com", "other@example.com"]
#       cc: []
#       bcc: []
#       subject: "Weinstein Weekly"
#   (All fields optional; subject defaults to "Weinstein Weekly".)

set -euo pipefail

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

CONFIG_PATH="${CONFIG_FILE:-./config.yaml}"
ENABLE_SCAN="${ENABLE_SCAN:-1}"
SCAN_UNIVERSE="${SCAN_UNIVERSE:-sp500}"
SCAN_BENCHMARK="${SCAN_BENCHMARK:-SPY}"
SCAN_MAX_ROWS="${SCAN_MAX_ROWS:-200}"

# Activate venv if present
source .venv/bin/activate 2>/dev/null || true

bold "🧾 Using config: $CONFIG_PATH"

# Read config.yaml and compute SHEET_URL.
# Priority:
#  1) sheets.sheet_url  OR sheets.daily_intake_sheet_url
#  2) google.sheet_url  (legacy)
#  3) Build from any of *_sheet_id under sheets (daily_intake_sheet_id / holdings_sheet_id / control_sheet_id)
SHEET_URL="$(python3 - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    print("", end=""); sys.exit(0)

def pick_url(c):
    sheets = (c.get("sheets") or {})
    url = sheets.get("sheet_url") or sheets.get("daily_intake_sheet_url")
    if url: return url
    g = (c.get("google") or {})
    url = g.get("sheet_url")
    if url: return url
    sid = sheets.get("daily_intake_sheet_id") or sheets.get("holdings_sheet_id") or sheets.get("control_sheet_id")
    return f"https://docs.google.com/spreadsheets/d/{sid}/edit" if sid else ""

print(pick_url(cfg), end="")
PY
)"

export SHEET_URL

# Soft info printout (don’t fail the run if YAML preview has an issue)
{
  python3 - <<'PY' || true
import yaml, os
p = os.environ.get("CONFIG_PATH","./config.yaml")
sheet_url = os.environ.get("SHEET_URL","(missing)")
try:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    s = (cfg.get("sheets") or {})
    open_tab   = s.get("open_positions_tab","Open_Positions")
    signals    = s.get("signals_tab","Signals")
    out_dir    = s.get("output_dir","./output")
    print(f"• Google Sheet: {sheet_url if sheet_url else '(missing)'}")
    print(f"• Open Positions tab: {open_tab}")
    print(f"• Signals tab:        {signals}")
    print(f"• Output dir:         {out_dir}")
except Exception as e:
    print(f"(info) Skipping YAML preview: {e}")
PY
} 2>/dev/null

# 1) Build the Google Sheets dashboard tabs (NO --config here)
bold "📊 Building portfolio dashboard (Sheets)…"
python3 build_performance_dashboard.py

# 2) Generate the “current” weekly email (portfolio summary + per-position)
if [[ -z "${SHEET_URL}" ]]; then
  red "SHEET_URL not found in YAML. Set sheets.sheet_url OR sheets.daily_intake_sheet_id."
  exit 1
fi

COMBINED_HTML="output/combined_weekly_email.html"
SCAN_CSV="output/scan_${SCAN_UNIVERSE}.csv"
SCAN_HTML="output/scan_${SCAN_UNIVERSE}.html"

bold "📰 Generating Weinstein Weekly (portfolio) report…"
WEEKLY_FLAGS_STR="${WEEKLY_FLAGS:---email --attach-html}"   # your current default
set +e
python3 weinstein_report_weekly.py \
  --sheet-url "${SHEET_URL}" \
  --write ${WEEKLY_FLAGS_STR}
PORTFOLIO_STEP_RC=$?
set -e

# 3) Classic Weinstein scan (S&P 500) – only if watcher exists AND enabled
if [[ "${ENABLE_SCAN}" == "1" ]]; then
  if [[ -f "weinstein_intraday_watcher.py" ]]; then
    bold "🔎 Running classic Weinstein scan (${SCAN_UNIVERSE}, bench ${SCAN_BENCHMARK})…"
    set +e
    python3 weinstein_intraday_watcher.py \
      --universe "${SCAN_UNIVERSE}" \
      --benchmark "${SCAN_BENCHMARK}" \
      --write-csv "${SCAN_CSV}" \
      --write-html "${SCAN_HTML}" \
      --quiet \
      --max-rows "${SCAN_MAX_ROWS}"
    SCAN_STEP_RC=$?
    set -e
    if [[ "${SCAN_STEP_RC}" -ne 0 ]]; then
      yellow "Scanner step failed (rc=${SCAN_STEP_RC}). Continuing without classic scan…"
      SCAN_HTML="" ; SCAN_CSV=""
    fi
  else
    yellow "weinstein_intraday_watcher.py not found. Skipping classic scan."
    SCAN_HTML="" ; SCAN_CSV=""
  fi
else
  yellow "Classic scan disabled via ENABLE_SCAN=0."
  SCAN_HTML="" ; SCAN_CSV=""
fi

# 4) Build a combined HTML (portfolio + scan) for archival and as fallback email body
bold "🧩 Assembling combined weekly HTML…"
python3 - <<PY
import os, io, datetime, sys
out_path = "${COMBINED_HTML}"
scan_html = "${SCAN_HTML}"
parts = []

# Always try to include the tabular “Weekly_Report” from the report script (it writes into Sheets; our script also
# prints summary to stdout — here we just build a simple wrapper with links/files we produced).
parts.append("<h1>Weinstein Weekly — Combined</h1>")
parts.append(f"<p><em>Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>")

# If classic scan HTML exists, embed it below a divider
if scan_html and os.path.exists(scan_html):
    parts.append("<hr/><h2>Classic Weinstein Scan (Top results)</h2>")
    try:
        with open(scan_html, "r", encoding="utf-8") as f:
            parts.append(f.read())
    except Exception as e:
        parts.append(f"<p>Could not embed scan HTML: {e}</p>")
else:
    parts.append("<p><em>No classic scan available in this run.</em></p>")

html = "\n".join(parts)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"✅ Combined weekly report written: {out_path}")
PY

# 5) Final email fallback if the report script didn’t email successfully.
# We detect “success” loosely by its return code. If it failed, or if you want to always force the fallback,
# set FORCE_FALLBACK_MAIL=1 in the environment.
FORCE_FALLBACK_MAIL="${FORCE_FALLBACK_MAIL:-0}"
if [[ "${PORTFOLIO_STEP_RC}" -ne 0 || "${FORCE_FALLBACK_MAIL}" == "1" ]]; then
  yellow "Primary email step did not complete (rc=${PORTFOLIO_STEP_RC}) or forced. Attempting fallback email sender…"
  set +e
  python3 - "$CONFIG_PATH" "$COMBINED_HTML" "$SCAN_HTML" "$SCAN_CSV" <<'PY'
import sys, os, yaml

cfg_path, combined_html, scan_html, scan_csv = sys.argv[1:5]

def read_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

cfg = read_yaml(cfg_path)
email_cfg = (cfg.get("email") or {})
to_   = email_cfg.get("to")   or []
cc_   = email_cfg.get("cc")   or []
bcc_  = email_cfg.get("bcc")  or []
subj  = email_cfg.get("subject") or "Weinstein Weekly"

# Read HTML body
try:
    with open(combined_html, "r", encoding="utf-8") as f:
        body_html = f.read()
except Exception as e:
    print(f"Fallback email aborted: cannot read {combined_html}: {e}")
    sys.exit(0)

# Collect attachments if present
atts = [p for p in [combined_html, scan_html, scan_csv] if p and os.path.exists(p)]

# Try the minimal sender
try:
    import WeinsteinMinimalEmailSender as W
except Exception as e:
    print(f"Fallback email not sent (sender import error): {e}")
    sys.exit(0)

if not to_:
    print("Fallback email not sent: no recipients under email.to in config.yaml")
    sys.exit(0)

try:
    W.send_html_email(
        subject=subj,
        html_body=body_html,
        to_addrs=to_,
        cc_addrs=cc_,
        bcc_addrs=bcc_,
        attachments=atts
    )
    print("📬 Fallback email sent via WeinsteinMinimalEmailSender.")
except Exception as e:
    print(f"Fallback email failed: {e}")
PY
  set -e
else
  green "📧 Primary report script indicated email step completed."
fi

green "✅ Weekly pipeline finished (Sheets + combined report)."
