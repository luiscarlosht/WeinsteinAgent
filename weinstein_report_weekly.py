#!/usr/bin/env python3
"""
weinstein_report_weekly.py

Builds a concise weekly portfolio summary (reads Google Sheet), writes the
"Weekly_Report" tab, and (optionally) emails an HTML/CSV snapshot.

CLI
---
  --write                 Write the Weekly_Report tab back to the same sheet.
  --email                 Email the summary using settings in config.yaml.
  --attach-html           Attach the generated HTML file to the email.
  --sheet-url SHEET_URL   Full Google Sheet URL to read/write.

Notes
-----
- This script purposely does NOT accept --config; the runner discovers the
  Sheet URL from YAML and passes only --sheet-url.
- Email uses WeinsteinMinimalEmailSender.send_email_with_yaml(...) and will
  pull SMTP app_password from:
      notifications.email.smtp.app_password
  inside `config.yaml` (or $CONFIG_FILE if set).
"""

import os
import sys
import csv
import io
import argparse
from datetime import datetime

# Ensure local modules (in repo) import cleanly even if run as a script
sys.path.append(os.path.dirname(__file__))

import yaml
import pandas as pd

# Try to use gsheets utilities from the repo (if present). If not, fallback to
# plain gspread auth (expects creds/service_account.json).
try:
    from gsheets_sync import (
        authorize_service_account,
        open_by_url,
        get_or_create_worksheet_by_title,
    )
    _HAVE_HELPERS = True
except Exception:
    _HAVE_HELPERS = False

try:
    import gspread
    from google.oauth2.service_account import Credentials as _Creds
except Exception as _e:
    gspread = None
    _Creds = None

def _service_account_client():
    """
    Returns an authorized gspread client.

    Prefers gsheets_sync helpers. If unavailable, uses creds/service_account.json
    directly via gspread.
    """
    if _HAVE_HELPERS:
        try:
            gc = authorize_service_account()
            return gc
        except Exception as e:
            print(f"⚠️  gsheets_sync authorize_service_account failed, falling back: {e}")

    if gspread is None or _Creds is None:
        raise RuntimeError("gspread not available and gsheets_sync helpers failed.")

    sa_path = os.path.join(os.path.dirname(__file__), "creds", "service_account.json")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = _Creds.from_service_account_file(sa_path, scopes=scopes)
    return gspread.authorize(creds)

def _open_sheet(gc, sheet_url: str):
    if _HAVE_HELPERS:
        try:
            return open_by_url(gc, sheet_url)
        except Exception as e:
            print(f"⚠️  gsheets_sync open_by_url failed, falling back: {e}")

    # fallback
    return gc.open_by_url(sheet_url)

def _get_or_create(ws, sheet, title: str):
    if _HAVE_HELPERS:
        try:
            return get_or_create_worksheet_by_title(sheet, title)
        except Exception as e:
            print(f"⚠️  gsheets_sync get_or_create_worksheet_by_title failed, falling back: {e}")

    try:
        return sheet.worksheet(title)
    except Exception:
        return sheet.add_worksheet(title=title, rows=100, cols=30)

def _df_from_worksheet(worksheet) -> pd.DataFrame:
    values = worksheet.get_all_values()
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=header)

def _format_currency(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return x

def _try_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "").replace("$", ""))
    except Exception:
        return default

def build_weekly_summary(sheet_url: str):
    """
    Pulls the Open_Positions (and optionally Signals) tabs,
    computes a compact summary table and per-position snapshot (as HTML),
    and returns (html_body, csv_bytes, df_out) where:
      - html_body: final HTML string to embed/send
      - csv_bytes: CSV of the per-position grid (for attachment)
      - df_out:    DataFrame of the per-position rows (for writing to tab)
    """
    gc = _service_account_client()
    sh = _open_sheet(gc, sheet_url)

    # Tabs
    tab_positions = os.getenv("OPEN_POSITIONS_TAB", "Open_Positions")
    tab_signals   = os.getenv("SIGNALS_TAB", "Signals")

    ws_positions = _get_or_create(gc, sh, tab_positions)
    dfp = _df_from_worksheet(ws_positions)

    if dfp.empty:
        raise RuntimeError("Open_Positions is empty or missing - cannot build report.")

    # Normalize some likely columns
    rename_map = {
        "Symbol": "Symbol",
        "Description": "Description",
        "Quantity": "Quantity",
        "Last Price": "Last Price",
        "Current Value": "Current Value",
        "Cost Basis Total": "Cost Basis Total",
        "Average Cost Basis": "Average Cost Basis",
        "Total Gain/Loss Dollar": "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent": "Total Gain/Loss Percent",
        "Recommendation": "Recommendation",
    }
    # If any expected column is missing under a slightly different name, try case-insensitive match
    cols_lower = {c.lower(): c for c in dfp.columns}
    fixed = {}
    for want in rename_map:
        if want in dfp.columns:
            fixed[want] = want
        else:
            lc = want.lower()
            if lc in cols_lower:
                fixed[want] = cols_lower[lc]
    # Filter only columns we actually found
    present = {k: v for k, v in fixed.items()}

    view_cols = [c for c in rename_map.keys() if c in present]
    dfv = dfp[[present[c] for c in view_cols]].rename(columns={present[c]: c for c in view_cols})

    # Coerce numeric
    for col in ["Quantity", "Last Price", "Current Value", "Cost Basis Total", "Average Cost Basis", "Total Gain/Loss Dollar"]:
        if col in dfv.columns:
            dfv[col] = dfv[col].apply(_try_float)

    # Compute summary
    total_gain_dollar = dfv["Total Gain/Loss Dollar"].sum() if "Total Gain/Loss Dollar" in dfv.columns else 0.0
    current_value_sum = dfv["Current Value"].sum() if "Current Value" in dfv.columns else 0.0
    cost_basis_sum    = dfv["Cost Basis Total"].sum() if "Cost Basis Total" in dfv.columns else 0.0
    portfolio_pct_gain = ((current_value_sum - cost_basis_sum) / cost_basis_sum * 100.0) if cost_basis_sum else 0.0
    avg_pct_gain = dfv["Total Gain/Loss Percent"].astype(str).str.replace("%","").astype(float).mean() if "Total Gain/Loss Percent" in dfv.columns and not dfv.empty else 0.0

    # Pretty HTML
    summary_rows = [
        ("Total Gain/Loss ($)", _format_currency(total_gain_dollar)),
        ("Portfolio % Gain",    f"{portfolio_pct_gain:.2f}%"),
        ("Average % Gain",      f"{avg_pct_gain:.2f}%"),
    ]

    summary_table_html = "".join(
        f"<tr><td style='padding:6px 12px;border:1px solid #ddd;'>{k}</td>"
        f"<td style='padding:6px 12px;border:1px solid #ddd;'>{v}</td></tr>"
        for k, v in summary_rows
    )

    # Per-position HTML
    pretty = dfv.copy()
    if "Last Price" in pretty:            pretty["Last Price"]            = pretty["Last Price"].map(_format_currency)
    if "Current Value" in pretty:         pretty["Current Value"]         = pretty["Current Value"].map(_format_currency)
    if "Cost Basis Total" in pretty:      pretty["Cost Basis Total"]      = pretty["Cost Basis Total"].map(_format_currency)
    if "Average Cost Basis" in pretty:    pretty["Average Cost Basis"]    = pretty["Average Cost Basis"].map(_format_currency)
    if "Total Gain/Loss Dollar" in pretty:pretty["Total Gain/Loss Dollar"]= pretty["Total Gain/Loss Dollar"].map(_format_currency)
    if "Total Gain/Loss Percent" in pretty:
        pretty["Total Gain/Loss Percent"] = pretty["Total Gain/Loss Percent"].astype(str)

    positions_html = pretty.to_html(index=False, border=0, classes="grid", justify="left")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_body = f"""
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; color: #222; }}
    h1, h2 {{ margin: 0.2rem 0; }}
    .card {{
      border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin: 12px 0;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    table.grid {{
      border-collapse: collapse; width: 100%;
    }}
    table.grid th, table.grid td {{
      border: 1px solid #ddd; padding: 6px 8px; text-align: left;
      white-space: nowrap;
    }}
    table.grid th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Weinstein Weekly - Summary</h1>
  <div class="card">
    <table class="grid">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {summary_table_html}
      </tbody>
    </table>
  </div>

  <h2>Per-position Snapshot</h2>
  <div class="card">
    {positions_html}
  </div>
  <div style="color:#6b7280;font-size:12px;margin-top:8px;">Generated {ts}</div>
</body>
</html>
""".strip()

    # CSV attachment
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(pretty.columns)
    for _, row in pretty.iterrows():
        writer.writerow([row.get(c, "") for c in pretty.columns])
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # DataFrame to write to Weekly_Report tab
    out_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    return html_body, csv_bytes, out_df

def write_weekly_tab(sheet_url: str, df_out: pd.DataFrame, tab_name="Weekly_Report"):
    gc = _service_account_client()
    sh = _open_sheet(gc, sheet_url)
    ws = _get_or_create(gc, sh, tab_name)
    # Clear then write
    ws.clear()
    rows = [list(df_out.columns)] + df_out.astype(str).values.tolist()
    ws.update(rows)

def _write_file(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Write Weekly_Report tab")
    ap.add_argument("--email", action="store_true", help="Send email via SMTP settings in config.yaml")
    ap.add_argument("--attach-html", action="store_true", help="Attach generated HTML to the email")
    ap.add_argument("--sheet-url", required=True, help="Google Sheet URL")
    args = ap.parse_args()

    html_body, csv_bytes, df_out = build_weekly_summary(args.sheet_url)

    # Always export an HTML file for run_weekly.sh to optionally combine
    out_dir = os.getenv("OUTPUT_DIR", "./output")
    html_path = os.path.join(out_dir, "weekly_portfolio.html")
    csv_path  = os.path.join(out_dir, "weekly_portfolio.csv")
    _write_file(html_path, html_body.encode("utf-8"))
    _write_file(csv_path, csv_bytes)

    if args.write:
        try:
            write_weekly_tab(args.sheet_url, df_out, tab_name="Weekly_Report")
            print("✅ Wrote Weekly_Report tab.")
        except Exception as e:
            print(f"⚠️ Failed writing Weekly_Report tab: {e}")

    if args.email:
        # Use the repo's lightweight sender that pulls SMTP app_password from YAML.
        try:
            from WeinsteinMinimalEmailSender import send_email_with_yaml
            cfg_path = os.environ.get("CONFIG_FILE", "./config.yaml")
            ok = send_email_with_yaml(
                subject_suffix="Portfolio Summary",
                html_body=html_body,
                attachments=[html_path] if args.attach_html else None,
                config_path=cfg_path,
            )
            if ok:
                print("📧 Email sent.")
            else:
                print("Email failed: configuration incomplete (check notifications.email in config.yaml).")
        except Exception as e:
            print("Email failed: Email sender not available. Please keep WeinsteinMinimalEmailSender.py in repo.")
            print(f"(detail: {e})")

    print("🎯 Done.")

if __name__ == "__main__":
    main()
