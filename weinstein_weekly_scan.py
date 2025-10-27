#!/usr/bin/env python3
# weinstein_weekly_scan.py
# Produces the classic Weinstein universe scan (e.g., S&P 500) and returns:
#  - counts summary (Buy/Watch/Avoid)
#  - a large HTML table (trimmed for email)
#  - paths to CSV/HTML artifacts for attachments / links
#
# It shells out to update_signals_and_score.py so we don’t touch your scanner logic.

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

def run_scan(universe: str, benchmark: str, out_dir: str, max_rows_email: int = 200) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, f"scan_{universe}.csv")
    html_path = os.path.join(out_dir, f"scan_{universe}.html")

    # Run your existing scanner to generate CSV + HTML
    # (If your script uses different flags, tweak these three lines only.)
    cmd = [
        sys.executable, "update_signals_and_score.py",
        "--universe", universe,
        "--benchmark", benchmark,
        "--write-csv", csv_path,
        "--write-html", html_path,
        "--quiet"
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"Scanner failed: {e}"}

    # Parse the HTML to extract the counts header if present,
    # otherwise we’ll synthesize the counts from the CSV quickly.
    summary_text = None
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        # Try to locate a line like: "Summary: ✅ Buy: 120 | 🟡 Watch: 19 | 🔴 Avoid: 366 (Total: 505)"
        # If not found, just leave summary_text = None; email builder will handle fallback.
        for line in html.splitlines():
            if "Summary:" in line and ("Buy:" in line and "Avoid:" in line):
                summary_text = line.strip()
                break
    except Exception:
        pass

    # For the email, include at most N rows so the message stays reasonable.
    # We’ll truncate the HTML table naïvely: keep everything up to <tbody>, then only first N <tr>.
    trimmed_html = None
    try:
        from bs4 import BeautifulSoup  # Optional: only if installed. If not, fall back.
        soup = BeautifulSoup(html, "html.parser")
        tbl = soup.find("table")
        if tbl:
            # Trim rows
            body = tbl.find("tbody") or tbl
            rows = body.find_all("tr")
            for r in rows[max_rows_email:]:
                r.decompose()
            trimmed_html = str(tbl)
    except Exception:
        # Fallback: simple line-based trim
        trimmed_html = html

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"""
      <h2 style="margin:24px 0 8px;">Weinstein Weekly — Benchmark: {benchmark}</h2>
      <div style="color:#666;margin:0 0 16px;">Generated {ts} — Universe: {universe}</div>
    """
    if summary_text:
        header += f'<div style="margin:8px 0 16px;">{summary_text}</div>'

    block_html = header
    if trimmed_html:
        block_html += trimmed_html

    return {
        "ok": True,
        "html_block": block_html,
        "csv_path": csv_path,
        "html_path": html_path,
        "summary_line": summary_text or ""
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--out-dir", default="./output")
    ap.add_argument("--max-rows-email", type=int, default=200)
    ap.add_argument("--json", action="store_true", help="print JSON result to stdout")
    args = ap.parse_args()

    res = run_scan(args.universe, args.benchmark, args.out_dir, args.max_rows_email)
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        if not res.get("ok"):
            print(res.get("error", "scan failed"), file=sys.stderr)
            sys.exit(1)
        print("Scan complete:", res.get("summary_line", "(no summary)"))

if __name__ == "__main__":
    main()
