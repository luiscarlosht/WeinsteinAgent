#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Weekly Report

Reads current holdings from Google Sheets and produces:
- A summary metrics block (Total Gain/Loss $, Portfolio % Gain, Average % Gain)
- A per-position table with Weinstein-ish SELL/HOLD recommendation
- Writes a 'Weekly_Report' tab in the same Google Sheet
- Saves CSV/HTML snapshots to ./output
- (NEW) Optional email with attachments of the generated HTML/CSV

Usage:
  python3 weinstein_report_weekly.py [--write] [--email]
                                     [--attach-html] [--attach-csv]
                                     [--to you@example.com ...]
                                     [--sheet-url URL]
                                     [--tab-holdings Holdings]
                                     [--tab-weekly Weekly_Report]

Email settings pulled from config.yaml (see WeinsteinMinimalEmailSender.py).
"""

from __future__ import annotations

import os
import io
import argparse
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# local mailer
from WeinsteinMinimalEmailSender import send_email, load_email_config

# ─────────────────────────────
# CONFIG DEFAULTS
# ─────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# If you prefer, put the Sheet URL in config.yaml under 'sheets.weekly_url'
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"
TAB_HOLDINGS_DEFAULT = "Holdings"
TAB_WEEKLY_DEFAULT   = "Weekly_Report"

OUTPUT_DIR = "output"

# Recommendation thresholds (editable)
STRONG_HOLD_PCT = 50.0   # ≥ this % → "HOLD (Strong)"
SELL_PCT        = -10.0  # ≤ this % → "SELL"

# ─────────────────────────────
# AUTH / SHEET HELPERS
# ─────────────────────────────
def auth_gspread():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, sheet_url: str, tab: str):
    sh = gc.open_by_url(sheet_url)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=2000, cols=26)

def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.resize(rows=50, cols=10)
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(50, rows+5), cols=max(10, cols+2))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

    # chunk upload
    start = 0
    r = 1
    while start < len(data):
        end = min(start+500, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)
        r += len(block)
        start = end

# ─────────────────────────────
# DATA CALC
# ─────────────────────────────
def _to_float(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    v = str(s).replace("$", "").replace(",", "").strip()
    try:
        return float(v)
    except Exception:
        return None

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    """
    Expect Holdings columns similar to Fidelity export:
      Symbol, Description, Quantity, Last Price, Current Value, Cost Basis Total,
      Average Cost Basis, Total Gain/Loss Dollar, Total Gain/Loss Percent
    """
    if df_h.empty:
        return pd.DataFrame(columns=[
            "Symbol","Description","Quantity","Last Price","Current Value","Cost Basis Total",
            "Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent"
        ])

    cols = df_h.columns.str.strip()
    df = df_h.copy()
    df.columns = cols

    # find best-effort matches
    col_map = {}
    for key, alts in {
        "Symbol": ["Symbol","Ticker"],
        "Description": ["Description","Security Description"],
        "Quantity": ["Quantity","Qty"],
        "Last Price": ["Last Price","Price"],
        "Current Value": ["Current Value","Market Value","Value"],
        "Cost Basis Total": ["Cost Basis Total","Cost Basis","Total Cost"],
        "Average Cost Basis": ["Average Cost Basis","Avg Cost"],
        "Total Gain/Loss Dollar": ["Total Gain/Loss Dollar","Gain/Loss $","Total G/L $"],
        "Total Gain/Loss Percent": ["Total Gain/Loss Percent","Gain/Loss %","Total G/L %"],
    }.items():
        for a in alts:
            if a in df.columns:
                col_map[key] = a
                break

    # ensure all exist
    for k in [
        "Symbol","Description","Quantity","Last Price","Current Value","Cost Basis Total",
        "Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent"
    ]:
        if k not in col_map:
            df[k] = ""
            col_map[k] = k  # points to the new blank col

    out = df[list(col_map.values())].copy()
    out.columns = list(col_map.keys())

    # numeric conversions
    for c in ["Quantity","Last Price","Current Value","Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar"]:
        out[c] = out[c].map(_to_float)

    # Percent can be like "149.01%" or "149.01"
    def to_pct(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace("%","").replace(",","").strip()
        try:
            return float(s)
        except Exception:
            return None
    out["Total Gain/Loss Percent"] = out["Total Gain/Loss Percent"].map(to_pct)

    # drop cash-like rows (Symbol empty or FCASH**/SPAXX** etc.)
    out = out[ out["Symbol"].astype(str).str.strip().ne("") ]
    out = out[ ~out["Symbol"].astype(str).str.contains(r"FCASH|SPAXX|\*{2}|PENDING ACTIVITY", case=False, regex=True) ]
    out.reset_index(drop=True, inplace=True)

    return out

def weinstein_recommendation(pct_gain: float) -> str:
    if pct_gain is None:
        return "HOLD"
    if pct_gain >= STRONG_HOLD_PCT:
        return "HOLD (Strong)"
    if pct_gain <= SELL_PCT:
        return "SELL"
    return "HOLD"

def compute_summary(df: pd.DataFrame) -> Tuple[float,float,float]:
    """Total $, portfolio % gain (value-weighted), average % gain (simple mean)."""
    if df.empty:
        return 0.0, 0.0, 0.0

    total_gain = (df["Total Gain/Loss Dollar"].fillna(0.0)).sum()

    # value-weighted (current value vs cost basis total)
    cv = df["Current Value"].fillna(0.0).sum()
    cb = df["Cost Basis Total"].fillna(0.0).sum()
    portfolio_pct = (cv - cb) / cb * 100.0 if cb else 0.0

    # simple average of pct
    avg_pct = df["Total Gain/Loss Percent"].dropna()
    avg_pct = float(avg_pct.mean()) if not avg_pct.empty else 0.0
    return total_gain, portfolio_pct, avg_pct

def build_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["Recommendation"] = out["Total Gain/Loss Percent"].map(weinstein_recommendation)
    # clean floats and ordering
    cols = [
        "Symbol","Description","Quantity","Last Price","Current Value","Cost Basis Total",
        "Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent","Recommendation"
    ]
    out = out[cols]
    return out

# ─────────────────────────────
# SHEET OUTPUT SHAPE
# ─────────────────────────────
def build_weekly_sheet(view_df: pd.DataFrame, total_gain: float, portfolio_pct: float, avg_pct: float) -> pd.DataFrame:
    """A single DataFrame that has a 2-block layout: a 2-column metrics header and the table below it."""
    # Top metrics (2 columns)
    top = pd.DataFrame({
        "Metric": ["Total Gain/Loss ($)","Portfolio % Gain","Average % Gain"],
        "Value":  [f"${total_gain:,.2f}", f"{portfolio_pct:.2f}%", f"{avg_pct:.2f}%"],
    })

    # Spacer row
    spc = pd.DataFrame({"Metric":[""], "Value":[""]})

    # Bottom = main table
    bot = view_df.copy()

    # Put them together (different column counts are fine in Sheets, we’ll just write separately)
    # For Sheets convenience, return them concatenated but keep columns separate by filling missing.
    # Normalize to same columns by expanding top/spacer with blank columns to match bot.
    max_cols = max(2, bot.shape[1])
    def pad(df, width):
        if df.shape[1] >= width:
            return df
        newcols = list(df.columns) + [f"" for _ in range(width - df.shape[1])]
        df = df.copy()
        df.columns = newcols
        return df

    top = pad(top, max_cols)
    spc = pad(spc, max_cols)
    bot = pad(bot, max_cols)

    combined = pd.concat([top, spc, bot], ignore_index=True)
    return combined

# ─────────────────────────────
# FILE OUTPUT + EMAIL
# ─────────────────────────────
def save_snapshots(df_view: pd.DataFrame) -> Tuple[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(OUTPUT_DIR, f"weinstein_weekly_{ts}.csv")
    html_path = os.path.join(OUTPUT_DIR, f"weinstein_weekly_{ts}.html")

    # CSV
    df_view.to_csv(csv_path, index=False)

    # HTML
    buf = io.StringIO()
    # a quick, clean table
    styled = (
        df_view.style
            .format({
                "Quantity":"{:.2f}",
                "Last Price":"{:.2f}",
                "Current Value":"{:.2f}",
                "Cost Basis Total":"{:.2f}",
                "Average Cost Basis":"{:.2f}",
                "Total Gain/Loss Dollar":"{:.2f}",
                "Total Gain/Loss Percent":"{:.2f}%"
            }, na_rep="")
            .hide_index()
    )
    html = styled.to_html()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return csv_path, html_path

def email_report(
    *,
    subject: str,
    summary_text: str,
    html_report_path: str | None,
    csv_report_path: str | None,
    to_list: List[str] | None,
    sheet_url: str,
    also_embed_html_preview: bool = True,
) -> None:
    body_text = summary_text + f"\n\nGoogle Sheet: {sheet_url}\n"
    body_html = None
    if also_embed_html_preview and html_report_path and os.path.exists(html_report_path):
        try:
            with open(html_report_path, "r", encoding="utf-8") as f:
                html_table = f.read()
            body_html = f"""
            <html>
              <body>
                <p>{summary_text.replace('\n','<br>')}</p>
                <p><b>Google Sheet:</b> <a href="{sheet_url}">{sheet_url}</a></p>
                <hr>
                {html_table}
              </body>
            </html>
            """
        except Exception:
            body_html = None

    attachments = []
    if html_report_path and os.path.exists(html_report_path):
        attachments.append(html_report_path)
    if csv_report_path and os.path.exists(csv_report_path):
        attachments.append(csv_report_path)

    send_email(
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        to=to_list,
        attachments=attachments,
    )

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate weekly Weinstein report from Holdings tab and (optionally) email it.")
    ap.add_argument("--write", action="store_true", help="Write the 'Weekly_Report' tab in the sheet.")
    ap.add_argument("--email", action="store_true", help="Send email after building report.")
    ap.add_argument("--attach-html", action="store_true", help="Attach the generated HTML snapshot.")
    ap.add_argument("--attach-csv", action="store_true", help="Attach the generated CSV snapshot.")
    ap.add_argument("--to", nargs="*", default=None, help="Override recipient list (otherwise uses config.yaml email.to).")
    ap.add_argument("--sheet-url", default=DEFAULT_SHEET_URL, help="Google Sheet URL.")
    ap.add_argument("--tab-holdings", default=TAB_HOLDINGS_DEFAULT, help="Holdings tab name.")
    ap.add_argument("--tab-weekly", default=TAB_WEEKLY_DEFAULT, help="Weekly output tab name.")
    args = ap.parse_args()

    print("📊 Generating weekly Weinstein report…")
    print("🔑 Authorizing service account…")
    gc = auth_gspread()

    ws_h = open_ws(gc, args.sheet_url, args.tab_holdings)
    df_h = read_tab(ws_h)

    # Load holdings, compute summary & recommendations
    df_hold = load_holdings(df_h)
    total_gain, portfolio_pct, avg_pct = compute_summary(df_hold)

    print("\n=== Weinstein Weekly Report ===")
    print(f"Total Gain/Loss ($): {total_gain:,.2f}")
    print(f"Portfolio % Gain  : {portfolio_pct:.2f}%")
    print(f"Average % Gain     : {avg_pct:.2f}%\n")

    df_view = build_view(df_hold)
    if not df_view.empty:
        print("Per-position snapshot:")
        try:
            # pretty display
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print(df_view.to_string(index=False))
        except Exception:
            print(df_view.head().to_string(index=False))

    # Save snapshots (always)
    csv_path, html_path = save_snapshots(df_view)

    # Write to the sheet if requested
    if args.write:
        ws_out = open_ws(gc, args.sheet_url, args.tab_weekly)
        out_df = build_weekly_sheet(df_view, total_gain, portfolio_pct, avg_pct)
        write_tab(ws_out, out_df)
        print(f"\n✅ Wrote '{args.tab_weekly}' tab with summary and per-position details.\n")

    # Email if requested
    if args.email:
        # Subject + summary
        ts = datetime.now(timezone.utc).astimezone().strftime("%b %d %Y %I:%M %p %Z")
        subject = f"Weinstein Weekly Report — {ts}"
        summary = (
            f"=== Weinstein Weekly Report ===\n"
            f"Total Gain/Loss ($): {total_gain:,.2f}\n"
            f"Portfolio % Gain  : {portfolio_pct:.2f}%\n"
            f"Average % Gain    : {avg_pct:.2f}%"
        )
        # Which attachments?
        html_attach = html_path if args.attach_html else None
        csv_attach  = csv_path if args.attach_csv else None

        # Recipients: use overrides or config.yaml
        to_list = args.to if (args.to and len(args.to) > 0) else None

        try:
            email_report(
                subject=subject,
                summary_text=summary,
                html_report_path=html_attach,
                csv_report_path=csv_attach,
                to_list=to_list,
                sheet_url=args.sheet_url,
                also_embed_html_preview=True,
            )
            print("📧 Email sent with requested attachments.")
        except Exception as e:
            print(f"⚠️ Email failed: {e}")

    print("🎯 Done.")


if __name__ == "__main__":
    main()
