#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Weekly Report

What it does
------------
• Reads the Google Sheet tabs (Holdings, Transactions, Signals if needed)
• Computes per-position and portfolio-level P&L
• Applies simple recommendation rules (HOLD / SELL / HOLD (Strong))
• Writes a clean Weekly_Report tab (summary block + detailed table)
• Optionally emails a completion confirmation

Email (optional)
----------------
Use a Gmail App Password (recommended). Either pass flags or set env vars:

  Flags:
    --email                      Send an email on success
    --email-to you@example.com
    --email-from you@gmail.com
    --smtp-user you@gmail.com
    --smtp-pass <app_password>

  Environment variables (used if flags omitted):
    GMAIL_TO, GMAIL_FROM, GMAIL_USER, GMAIL_APP_PASSWORD

Recommendations
---------------
Default thresholds (tweak with flags):
  • SELL if Total Gain/Loss Percent <= --sell-threshold (default -7.0)
  • HOLD (Strong) if Total Gain/Loss Percent >= --strong-hold-threshold (default 50.0)
  • Else HOLD

CLI
---
  python3 weinstein_report_weekly.py --write
  python3 weinstein_report_weekly.py --write --email
  python3 weinstein_report_weekly.py --write --email --email-to x@y.com \
      --smtp-user you@gmail.com --smtp-pass <app_password>

"""

from __future__ import annotations

import os
import argparse
import smtplib
from email.mime.text import MIMEText
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_HOLDINGS      = "Holdings"
TAB_WEEKLY_REPORT = "Weekly_Report"

ROW_CHUNK = 500  # for chunked writes when needed


# ─────────────────────────────
# GSHEETS HELPERS
# ─────────────────────────────
def auth_gspread():
    print("🔑 Authorizing service account…")
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, tab: str):
    sh = gc.open_by_url(SHEET_URL)
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
    # strip strings
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def safe_float(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).replace("$", "").replace(",", "").replace("%", "").strip()
    if s == "" or s.upper() in ("N/A", "NA", "-"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


# ─────────────────────────────
# CORE LOGIC
# ─────────────────────────────
def parse_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the Holdings export into the columns we need:
      Symbol, Description, Quantity, Last Price, Current Value,
      Cost Basis Total, Average Cost Basis,
      Total Gain/Loss Dollar, Total Gain/Loss Percent
    """

    if df_h.empty:
        return pd.DataFrame(columns=[
            "Symbol","Description","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis",
            "Total Gain/Loss Dollar","Total Gain/Loss Percent"
        ])

    # Try to match typical Fidelity column names (case-insensitive contains)
    def find_col(options):
        opts = [o.lower() for o in options]
        for c in df_h.columns:
            lc = c.lower()
            for o in opts:
                if o in lc:
                    return c
        return None

    c_symbol      = find_col(["symbol"])
    c_desc        = find_col(["description"])
    c_qty         = find_col(["quantity"])
    c_last_price  = find_col(["last price"])
    c_cur_val     = find_col(["current value"])
    c_cost_total  = find_col(["cost basis total"])
    c_cost_avg    = find_col(["average cost"])
    c_gl_dollar   = find_col(["total gain/loss dollar"])
    c_gl_percent  = find_col(["total gain/loss percent"])

    # Build a view
    out = pd.DataFrame()
    out["Symbol"] = df_h.get(c_symbol, "")
    out["Description"] = df_h.get(c_desc, "")
    out["Quantity"] = df_h.get(c_qty, "").map(safe_float)
    out["Last Price"] = df_h.get(c_last_price, "").map(safe_float)
    out["Current Value"] = df_h.get(c_cur_val, "").map(safe_float)
    out["Cost Basis Total"] = df_h.get(c_cost_total, "").map(safe_float)
    out["Average Cost Basis"] = df_h.get(c_cost_avg, "").map(safe_float)
    # Gain/Loss — prefer existing columns; otherwise recompute
    gl_d = df_h.get(c_gl_dollar, "").map(safe_float)
    gl_p = df_h.get(c_gl_percent, "").map(safe_float)

    # Recompute if missing
    need_gl_d = gl_d.isna().all()
    need_gl_p = gl_p.isna().all()

    if need_gl_d:
        gl_d = out["Current Value"] - out["Cost Basis Total"]

    if need_gl_p:
        with np.errstate(divide="ignore", invalid="ignore"):
            gl_p = np.where(out["Cost Basis Total"] > 0,
                            100.0 * gl_d / out["Cost Basis Total"],
                            np.nan)

    out["Total Gain/Loss Dollar"] = gl_d
    out["Total Gain/Loss Percent"] = gl_p

    # Filter out cash/money market rows (SPAXX, FCASH, Pending, etc.)
    def is_equity_like(sym, desc):
        s = str(sym).upper()
        d = str(desc).upper()
        blacklist = ("SPAXX", "FCASH", "PENDING", "MONEY MARKET")
        return not any(tok in s or tok in d for tok in blacklist)

    out = out[out.apply(lambda r: is_equity_like(r["Symbol"], r["Description"]), axis=1)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def recommendation_for(pct_gain: float, sell_threshold: float, strong_hold_threshold: float) -> str:
    if pd.isna(pct_gain):
        return "HOLD"
    if pct_gain <= sell_threshold:
        return "SELL"
    if pct_gain >= strong_hold_threshold:
        return "HOLD (Strong)"
    return "HOLD"


def build_view_with_recos(df: pd.DataFrame,
                          sell_threshold: float,
                          strong_hold_threshold: float) -> pd.DataFrame:
    view = df.copy()
    view["Recommendation"] = view["Total Gain/Loss Percent"].map(
        lambda v: recommendation_for(v, sell_threshold, strong_hold_threshold)
    )

    # Pretty formatting for output table
    def money(x):
        return "" if pd.isna(x) else f"{x:,.2f}"

    def pct(x):
        return "" if pd.isna(x) else f"{x:.2f}%"

    pretty = pd.DataFrame({
        "Symbol": view["Symbol"],
        "Description": view["Description"],
        "Quantity": view["Quantity"].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}".rstrip('0').rstrip('.')),
        "Last Price": view["Last Price"].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}"),
        "Current Value": view["Current Value"].map(money),
        "Cost Basis Total": view["Cost Basis Total"].map(money),
        "Average Cost Basis": view["Average Cost Basis"].map(money),
        "Total Gain/Loss Dollar": view["Total Gain/Loss Dollar"].map(money),
        "Total Gain/Loss Percent": view["Total Gain/Loss Percent"].map(pct),
        "Recommendation": view["Recommendation"],
    })
    return pretty, view  # pretty for sheet, view for totals


def compute_totals(df_numeric: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns:
      total_gain_dollar, portfolio_pct_gain, average_pct_gain
    """
    total_gain = float(np.nansum(df_numeric["Total Gain/Loss Dollar"].values))
    cost_total = float(np.nansum(df_numeric["Cost Basis Total"].values))
    if cost_total > 0:
        portfolio_pct = 100.0 * total_gain / cost_total
    else:
        portfolio_pct = 0.0

    avg_pct = float(np.nanmean(df_numeric["Total Gain/Loss Percent"].values))
    return total_gain, portfolio_pct, avg_pct


# ─────────────────────────────
# WRITE WEEKLY TAB
# ─────────────────────────────
def write_weekly_tab(gc,
                     df_pretty: pd.DataFrame,
                     total_gain: float,
                     portfolio_pct: float,
                     avg_pct: float):
    ws = open_ws(gc, TAB_WEEKLY_REPORT)
    ws.clear()

    # 1) Summary block (A1:B4)
    summary_values = [
        ["Metric", "Value"],
        ["Total Gain/Loss ($)", f"${total_gain:,.2f}"],
        ["Portfolio % Gain", f"{portfolio_pct:.2f}%"],
        ["Average % Gain", f"{avg_pct:.2f}%"],
    ]
    ws.update(summary_values, range_name="A1:B4")

    # 2) Table header + rows starting at A6
    header = list(df_pretty.columns)
    rows = df_pretty.fillna("").values.tolist()
    table_values = [header] + rows

    # Resize and write
    total_rows = 6 + len(table_values)
    total_cols = max(10, len(header))
    ws.resize(rows=max(100, total_rows + 5), cols=max(26, total_cols))

    # Compute bottom-right for the table range
    top_left = gspread.utils.rowcol_to_a1(6, 1)
    bottom_right = gspread.utils.rowcol_to_a1(6 + len(table_values) - 1, len(header))
    ws.update(table_values, range_name=f"{top_left}:{bottom_right}")


# ─────────────────────────────
# EMAIL (optional)
# ─────────────────────────────
def send_email(subject: str,
               body: str,
               to_addr: str,
               from_addr: str,
               smtp_user: str,
               smtp_pass: str) -> Optional[str]:
    """
    Send a simple text email via Gmail SMTP (SSL).
    Returns an error string on failure, or None on success.
    """
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return None
    except Exception as e:
        return str(e)


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate the Weinstein Weekly Report from Google Sheets data.")
    ap.add_argument("--write", action="store_true", help="Write the Weekly_Report tab in the Google Sheet.")
    ap.add_argument("--sell-threshold", type=float, default=-7.0,
                    help="SELL if Total Gain/Loss Percent <= this (default: -7.0).")
    ap.add_argument("--strong-hold-threshold", type=float, default=50.0,
                    help="HOLD (Strong) if Total Gain/Loss Percent >= this (default: 50.0).")

    # Email flags (optional)
    ap.add_argument("--email", action="store_true", help="Send an email confirmation on success.")
    ap.add_argument("--email-to", type=str, default=os.getenv("GMAIL_TO", ""),
                    help="Email recipient (or set env GMAIL_TO).")
    ap.add_argument("--email-from", type=str, default=os.getenv("GMAIL_FROM", ""),
                    help="Email 'From' address (or set env GMAIL_FROM).")
    ap.add_argument("--smtp-user", type=str, default=os.getenv("GMAIL_USER", ""),
                    help="SMTP username (or set env GMAIL_USER).")
    ap.add_argument("--smtp-pass", type=str, default=os.getenv("GMAIL_APP_PASSWORD", ""),
                    help="SMTP app password (or set env GMAIL_APP_PASSWORD).")

    args = ap.parse_args()

    print("📊 Generating weekly Weinstein report…")
    gc = auth_gspread()

    # Load holdings
    ws_h = open_ws(gc, TAB_HOLDINGS)
    df_h = read_tab(ws_h)

    # Normalize + recos
    df_norm = parse_holdings(df_h)
    df_pretty, df_numeric = build_view_with_recos(
        df_norm,
        sell_threshold=args.sell_threshold,
        strong_hold_threshold=args.strong_hold_threshold
    )

    # Totals
    total_gain, portfolio_pct, avg_pct = compute_totals(df_numeric)

    # Console print
    print("\n=== Weinstein Weekly Report ===")
    print(f"Total Gain/Loss ($): {total_gain:,.2f}")
    print(f"Portfolio % Gain  : {portfolio_pct:.2f}%")
    print(f"Average % Gain     : {avg_pct:.2f}%\n")

    # Head of table preview
    if not df_pretty.empty:
        print("Per-position snapshot:")
        print(df_pretty.head(30).to_string(index=False))  # preview first ~30 lines
    else:
        print("(No equity-like rows found in Holdings.)")

    # Write tab if requested
    if args.write:
        write_weekly_tab(gc, df_pretty, total_gain, portfolio_pct, avg_pct)
        print("\n✅ Wrote 'Weekly_Report' tab with summary and per-position details.")

    # Optional email
    if args.email:
        to_addr   = args.email_to or ""
        from_addr = args.email_from or args.smtp_user or ""
        user      = args.smtp_user or ""
        pwd       = args.smtp_pass or ""

        if not (to_addr and from_addr and user and pwd):
            print("⚠️ Skipping email: missing --email-to/--email-from/--smtp-user/--smtp-pass "
                  "or env GMAIL_TO/GMAIL_FROM/GMAIL_USER/GMAIL_APP_PASSWORD.")
        else:
            subject = "Weinstein Weekly Report complete"
            body = (
                "Your weekly report has finished.\n\n"
                f"Total Gain/Loss ($): {total_gain:,.2f}\n"
                f"Portfolio % Gain   : {portfolio_pct:.2f}%\n"
                f"Average % Gain     : {avg_pct:.2f}%\n"
                "\nSheet tab: Weekly_Report\n"
            )
            err = send_email(subject, body, to_addr, from_addr, user, pwd)
            if err:
                print(f"⚠️ Email failed: {err}")
            else:
                print(f"📧 Email confirmation sent to {to_addr}")

    print("\n🎯 Done.")


if __name__ == "__main__":
    main()
