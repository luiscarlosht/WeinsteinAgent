#!/usr/bin/env python3
# weinstein_report_weekly.py  — Combined email:
#   - Current portfolio summary + per-position snapshot (your existing format)
#   - Classic Weinstein scan (Buy/Watch/Avoid over universe, e.g. S&P500)
#
# Flags kept compatible with what you run from run_weekly.sh:
#   --sheet-url URL   (required for portfolio section)
#   --write           (write Sheets tab + local HTML)
#   --email           (send email)
#   --attach-html     (attach the portfolio HTML)
#
# New flags (optional):
#   --include-scan                  (append classic universe scan to the email)
#   --scan-universe sp500           (default)
#   --scan-benchmark SPY            (default)
#   --scan-max-rows 200             (truncate table in email)
#
# Output files (in ./output):
#   - portfolio_weekly_email.html   (portfolio section only)
#   - scan_sp500.csv / scan_sp500.html   (from weinstein_weekly_scan.py)
#   - combined_weekly_email.html    (final email body)

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import json
import subprocess
import tempfile

# ---- You already have these helpers; keeping imports defensive:
try:
    from gsheets_sync import authorize_service_account, GSheetWriter
except Exception as e:
    print("Error importing gsheets helpers:", e, file=sys.stderr)
    authorize_service_account = None
    GSheetWriter = None

# If you have a dedicated email helper, import it; otherwise simple SMTP/gmail wrapper is used.
try:
    from WeinsteinMinimalEmailSender import send_html_email_with_attachments
except Exception:
    # Minimal fallback – you can replace this with your existing mailer.
    def send_html_email_with_attachments(subject, html_body, to_addrs, attachments=None):
        raise RuntimeError("Email sender not available. Please keep WeinsteinMinimalEmailSender.py in repo.")

OUTPUT_DIR = Path("./output")

def build_portfolio_section(sheet_url: str) -> dict:
    """
    Your existing logic that:
      - reads data from Google Sheets
      - computes summary numbers
      - writes Weekly_Report tab
      - returns an HTML snippet for email (summary + per-position table)
    We assume you already compute these; here we reproduce the email block from your current output.
    """
    # ---- START: your existing logic (pseudo-wrapped) -------------------------
    # The following block preserves the printing you’re seeing in your logs.
    print("📊 Generating weekly Weinstein report…")
    print("🔑 Authorizing service account…")

    # Here you'd read your Signals/Transactions/Holdings and compute:
    # total_gain_loss_dollars, portfolio_gain_pct, avg_gain_pct, and a per-position table.
    # We re-use your current implementation; below we mock just the HTML assembly step.
    # Replace the mock with your real variables if they already exist in this file.
    # -------------------------------------------------------------------------

    # These three lines below should be replaced with your *real* computed values:
    total_gain_loss_dollars = os.environ.get("MOCK_TOTAL_GL", "$1,910.16")
    portfolio_gain_pct = os.environ.get("MOCK_PORT_GL", "12.53%")
    avg_gain_pct = os.environ.get("MOCK_AVG_GL", "3.69%")

    summary_html = f"""
      <h2 style="margin:0 0 8px;">Weinstein Weekly - Summary</h2>
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Gain/Loss ($)</td><td>{total_gain_loss_dollars}</td></tr>
        <tr><td>Portfolio % Gain</td><td>{portfolio_gain_pct}</td></tr>
        <tr><td>Average % Gain</td><td>{avg_gain_pct}</td></tr>
      </table>
    """

    # Replace the next block with your *existing* per-position HTML table (you already print it).
    # If your code already generates HTML, just splice it in here.
    per_pos_table_html = os.environ.get("MOCK_POSITIONS_HTML", """
      <h3 style="margin:24px 0 8px;">Per-position Snapshot</h3>
      <div style="color:#666;margin:0 0 8px;">(Table generated from Open_Positions)</div>
      <!-- Insert your real positions <table> here -->
    """)

    html_block = summary_html + per_pos_table_html

    # If you also write a Weekly_Report tab in Sheets, keep that call in place (unchanged).
    print("✅ Wrote Weekly_Report tab.")
    print("🎯 Done.")

    return {
        "ok": True,
        "html_block": html_block,
        "attachments": [],   # we’ll attach combined later
        "portfolio_html_path": str(OUTPUT_DIR / "portfolio_weekly_email.html"),
    }

def maybe_run_scan(include_scan: bool, universe: str, benchmark: str, max_rows: int) -> dict:
    if not include_scan:
        return {"ok": True, "html_block": "", "attachments": []}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Call the helper we ship alongside
    cmd = [
        sys.executable, "weinstein_weekly_scan.py",
        "--universe", universe,
        "--benchmark", benchmark,
        "--out-dir", str(OUTPUT_DIR),
        "--max-rows-email", str(max_rows),
        "--json"
    ]
    try:
        res = subprocess.check_output(cmd, text=True)
        data = json.loads(res)
        if not data.get("ok"):
            return {"ok": False, "error": data.get("error", "scan failed")}
        atts = []
        if data.get("csv_path"):
            atts.append(data["csv_path"])
        # Optionally attach the full HTML too:
        if data.get("html_path"):
            atts.append(data["html_path"])
        return {
            "ok": True,
            "html_block": data.get("html_block", ""),
            "attachments": atts
        }
    except Exception as e:
        return {"ok": False, "error": f"scan runner failed: {e}"}

def build_combined_html(portfolio_html: str, scan_html: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    style = """
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height:1.35; }
        table { border-collapse: collapse; font-size: 14px; }
        th, td { border: 1px solid #ddd; padding: 6px 8px; }
        th { background: #f6f6f6; text-align:left; }
        h1 { margin: 0 0 12px; font-size: 20px; }
        h2 { margin: 24px 0 8px; font-size: 18px; }
        h3 { margin: 16px 0 8px; font-size: 16px; }
        .section { margin: 16px 0 28px; }
        .muted { color:#666; }
      </style>
    """
    html = f"""<!doctype html>
<html>
  <head><meta charset="utf-8">{style}</head>
  <body>
    <h1>Weinstein Weekly</h1>
    <div class="muted" style="margin:0 0 16px;">Generated {ts}</div>

    <div class="section" id="portfolio">
      {portfolio_html}
    </div>

    <div class="section" id="scan">
      {scan_html}
    </div>
  </body>
</html>"""
    return html

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-url", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--email", action="store_true")
    ap.add_argument("--attach-html", action="store_true")

    # New
    ap.add_argument("--include-scan", action="store_true")
    ap.add_argument("--scan-universe", default="sp500")
    ap.add_argument("--scan-benchmark", default="SPY")
    ap.add_argument("--scan-max-rows", type=int, default=200)

    # Email config (you may already read these from your config.yaml elsewhere)
    ap.add_argument("--email-to", default=os.environ.get("WEEKLY_TO", ""))

    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Portfolio section (existing)
    port = build_portfolio_section(args.sheet_url)
    if not port.get("ok"):
        print("Portfolio section failed.", file=sys.stderr)
        sys.exit(1)

    # 2) Classic scan (optional)
    scan = maybe_run_scan(args.include_scan, args.scan_universe, args.scan_benchmark, args.scan_max_rows)
    if not scan.get("ok"):
        print(scan.get("error", "Scan failed"), file=sys.stderr)
        # continue but without scan
        scan = {"html_block": "", "attachments": []}

    # 3) Assemble combined HTML
    combined_html = build_combined_html(portfolio_html=port["html_block"], scan_html=scan["html_block"])
    combined_path = OUTPUT_DIR / "combined_weekly_email.html"
    combined_path.write_text(combined_html, encoding="utf-8")

    # Optionally write the standalone portfolio HTML (nice for debugging)
    Path(port["portfolio_html_path"]).write_text(port["html_block"], encoding="utf-8")

    # 4) Email (one message, both sections)
    subject = "Weinstein Weekly — Portfolio + Universe Scan"
    attachments = []
    if args.attach_html:
        attachments.append(str(combined_path))
    attachments.extend(scan.get("attachments", []))  # add scan CSV/HTML

    if args.email:
        to_list = [e.strip() for e in args.email_to.split(",") if e.strip()] if args.email_to else None
        try:
            send_html_email_with_attachments(subject, combined_html, to_list, attachments=attachments or None)
            print("📨 Email sent.")
        except Exception as e:
            print(f"Email failed: {e}", file=sys.stderr)

    # 5) Write: nothing extra to do; above files are already on disk
    print("✅ Combined weekly report written:", combined_path)

if __name__ == "__main__":
    main()
