#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
weinstein_report_weekly.py

Generates a weekly snapshot from the Google Sheet "Holdings" tab and writes
a "Weekly_Report" tab with summary metrics and per-position rows:
- Total gain/loss ($)
- Portfolio % Gain (cumulative)
- Average % Gain (simple average across lines with valid %)
- Position-level SELL/HOLD suggestions (simple rule below)

Optional: email the report (uses WeinsteinMinimalEmailSender.py).
Flags:
  --write         Write the Weekly_Report tab to the same Google Sheet
  --email         Send an email using config.yaml (notifications/email)
  --attach-html   Attach the pretty HTML report in the email
  --sheet-url URL Override the Google Sheet URL (default from SHEET_URL const)

Env/Files:
  - creds/gcp_service_account.json     (Google service account key)
  - config.yaml                        (email config, Option A or B supported by WeinsteinMinimalEmailSender.py)
"""

from __future__ import annotations

import argparse
import os
import math
from typing import Tuple, List

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Local email helper (provided separately)
from WeinsteinMinimalEmailSender import send_email

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Default sheet URL; can be overridden with --sheet-url
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_HOLDINGS = "Holdings"
TAB_WEEKLY   = "Weekly_Report"

# Columns expected in Holdings (Fidelity export already in your sheet)
COL_SYMBOL   = "Symbol"
COL_DESC     = "Description"
COL_QTY      = "Quantity"
COL_LAST     = "Last Price"
COL_CURVAL   = "Current Value"
COL_COST     = "Cost Basis Total"
COL_AVG_COST = "Average Cost Basis"
COL_GL_DOL   = "Total Gain/Loss Dollar"
COL_GL_PCT   = "Total Gain/Loss Percent"

# ─────────────────────────────
# GOOGLE SHEETS HELPERS
# ─────────────────────────────
def auth_gspread():
    print("🔑 Authorizing service account…")
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
    # strip strings
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.resize(rows=50, cols=10)
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows+5), cols=max(min(26, cols+2), 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

    # Chunked upload
    start = 0
    r = 1
    ROW_CHUNK = 500
    while start < len(data):
        end = min(start+ROW_CHUNK, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)
        r += len(block)
        start = end

# ─────────────────────────────
# DOMAIN LOGIC
# ─────────────────────────────
def to_float(x):
    if isinstance(x, str):
        x = x.replace("$","").replace(",","").replace("%","").strip()
    try:
        return float(x)
    except Exception:
        return np.nan

def clean_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    """Normalize key numeric columns and filter out cash/SPAXX/FCASH rows."""
    if df_h.empty:
        return pd.DataFrame(columns=[
            COL_SYMBOL, COL_DESC, COL_QTY, COL_LAST, COL_CURVAL, COL_COST, COL_AVG_COST, COL_GL_DOL, COL_GL_PCT
        ])

    df = df_h.copy()

    # Coerce numerics
    for c in [COL_QTY, COL_LAST, COL_CURVAL, COL_COST, COL_AVG_COST, COL_GL_DOL, COL_GL_PCT]:
        if c in df.columns:
            df[c] = df[c].map(to_float)
        else:
            df[c] = np.nan

    # Basic filter: drop obvious cash/money market rows
    sym = df.get(COL_SYMBOL, "").astype(str).str.upper()
    bad = sym.str.contains("FCASH|SPAXX|PENDING ACTIVITY|\\*\\*", regex=True, na=False)
    df = df.loc[~bad].copy()

    # Keep key columns only (preserve order)
    keep = [COL_SYMBOL, COL_DESC, COL_QTY, COL_LAST, COL_CURVAL, COL_COST, COL_AVG_COST, COL_GL_DOL, COL_GL_PCT]
    df = df[[c for c in keep if c in df.columns]]

    # Drop rows without meaningful symbol or quantity
    df = df[(df[COL_SYMBOL].astype(str).str.len() > 0) & (df[COL_QTY] > 0)]
    df.reset_index(drop=True, inplace=True)
    return df

def recommend_row(gain_pct: float, strong_win_threshold=50.0, sell_threshold=-6.0) -> str:
    """
    Simple rule-of-thumb:
      - gain_pct >= strong_win_threshold → "HOLD (Strong)"
      - gain_pct <= sell_threshold       → "SELL"
      - else                             → "HOLD"
    """
    if pd.isna(gain_pct):
        return "HOLD"
    if gain_pct >= strong_win_threshold:
        return "HOLD (Strong)"
    if gain_pct <= sell_threshold:
        return "SELL"
    return "HOLD"

def compute_summary(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns: (total_gain_dollars, portfolio_pct_gain_cumulative, avg_pct_gain_simple)
    - cumulative % gain = (sum(Current Value) - sum(Cost Basis)) / sum(Cost Basis) * 100
    - average % gain    = simple mean of row % gains (ignoring NaN)
    """
    cur = pd.to_numeric(df[COL_CURVAL], errors="coerce").fillna(0.0).sum()
    cost = pd.to_numeric(df[COL_COST], errors="coerce").fillna(0.0).sum()
    tot_gain = cur - cost
    cum_pct = (tot_gain / cost * 100.0) if cost > 0 else 0.0

    # Average of valid row-level %s
    row_pct = pd.to_numeric(df[COL_GL_PCT], errors="coerce")
    avg_pct = row_pct[~row_pct.isna()].mean()
    if pd.isna(avg_pct):
        avg_pct = 0.0

    return tot_gain, cum_pct, avg_pct

def format_currency(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"${x:,.2f}"

def format_percent(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.2f}%"

def build_weekly_dataframe(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Build the fully formatted dataframe that will be written to the sheet.
    Also returns summary numbers.
    """
    # Compute per-row recommendation
    df_view = df_clean.copy()
    df_view["Recommendation"] = df_view[COL_GL_PCT].map(lambda p: recommend_row(p))

    # Format numerics for presentation
    df_view["Quantity"] = df_view[COL_QTY].map(lambda v: f"{v:.2f}" if isinstance(v, (int,float)) and not pd.isna(v) else "")
    df_view["Last Price"] = df_view[COL_LAST].map(format_currency)
    df_view["Current Value"] = df_view[COL_CURVAL].map(format_currency)
    df_view["Cost Basis Total"] = df_view[COL_COST].map(format_currency)
    df_view["Average Cost Basis"] = df_view[COL_AVG_COST].map(format_currency)
    df_view["Total Gain/Loss Dollar"] = df_view[COL_GL_DOL].map(format_currency)
    df_view["Total Gain/Loss Percent"] = df_view[COL_GL_PCT].map(format_percent)

    # Create final per-position sheet view
    final_cols = [
        COL_SYMBOL, COL_DESC, "Quantity", "Last Price", "Current Value",
        "Cost Basis Total", "Average Cost Basis", "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent", "Recommendation"
    ]
    df_final = df_view[final_cols].copy()

    # Compute summary
    total_gain, portfolio_pct, avg_pct = compute_summary(df_clean)

    return df_final, total_gain, portfolio_pct, avg_pct

def make_top_table(total_gain: float, portfolio_pct: float, avg_pct: float) -> pd.DataFrame:
    data = [
        ["Metric", "Value"],
        ["Total Gain/Loss ($)", format_currency(total_gain)],
        ["Portfolio % Gain",    format_percent(portfolio_pct)],
        ["Average % Gain",      format_percent(avg_pct)],
    ]
    return pd.DataFrame(data[1:], columns=data[0])

def write_weekly_report(gc, sheet_url: str, df_positions: pd.DataFrame, total_gain: float, portfolio_pct: float, avg_pct: float):
    ws_out = open_ws(gc, sheet_url, TAB_WEEKLY)
    # Construct output as a vertical stack: top summary then blank row then table header + rows
    top_df = make_top_table(total_gain, portfolio_pct, avg_pct)

    # blank spacer
    spacer = pd.DataFrame([["",""]], columns=["",""])
    # normalize spacer width to 2 columns
    spacer.columns = ["",""]

    # Build final table: header already within df_positions
    out = pd.concat([top_df, spacer], ignore_index=True)

    # Re-add a single header row before positions
    header = pd.DataFrame([list(df_positions.columns)], columns=df_positions.columns)
    out2 = pd.concat([out, header, df_positions], ignore_index=True)
    write_tab(ws_out, out2)
    print("✅ Wrote Weekly_Report tab.")

# ─────────────────────────────
# HTML & EMAIL
# ─────────────────────────────
def dataframe_to_html(df: pd.DataFrame, title: str) -> str:
    # Basic HTML wrapper
    html_head = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif;margin:20px;}"
        "h2{margin-bottom:6px}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:8px;font-size:13px;}"
        "th{background:#f5f5f5;text-align:left;}"
        "tr:nth-child(even){background:#fafafa}"
        "</style></head><body>"
    )
    html_tail = "</body></html>"

    return f"{html_head}<h2>{title}</h2>{df.to_html(index=False, escape=False)}{html_tail}"

def email_report(total_gain: float, portfolio_pct: float, avg_pct: float,
                 df_positions: pd.DataFrame,
                 attach_html: bool, out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Bodies
    text_body = (
        "=== Weinstein Weekly Report ===\n"
        f"Total Gain/Loss ($): {format_currency(total_gain)}\n"
        f"Portfolio % Gain  : {format_percent(portfolio_pct)}\n"
        f"Average % Gain     : {format_percent(avg_pct)}\n"
        "\n"
        "Per-position snapshot attached (HTML) and shown in your Weekly_Report tab."
    )

    # Build a small HTML summary + positions table
    top = pd.DataFrame(
        {
            "Metric": ["Total Gain/Loss ($)", "Portfolio % Gain", "Average % Gain"],
            "Value":  [format_currency(total_gain), format_percent(portfolio_pct), format_percent(avg_pct)]
        }
    )
    html = (
        dataframe_to_html(top, "Weinstein Weekly - Summary")
        + "<br/>"
        + dataframe_to_html(df_positions, "Per-position Snapshot")
    )

    attachments: List[str] = []
    if attach_html:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        html_path = os.path.join(out_dir, f"weinstein_weekly_{ts}.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        attachments.append(html_path)

    # Send email via minimal sender
    send_email(
        subject="Weekly Weinstein Report",
        text_body=text_body,
        html_body=html,
        attachments=attachments,
    )

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate weekly Weinstein report from Google Sheets Holdings.")
    ap.add_argument("--write", action="store_true", help="Write the Weekly_Report tab")
    ap.add_argument("--email", action="store_true", help="Send the report via email (config.yaml required)")
    ap.add_argument("--attach-html", action="store_true", help="Attach an HTML version to the email")
    ap.add_argument("--sheet-url", default=SHEET_URL, help="Override Google Sheet URL")
    args = ap.parse_args()

    print("📊 Generating weekly Weinstein report…")

    # Auth + read holdings
    gc = auth_gspread()
    ws_h = open_ws(gc, args.sheet_url, TAB_HOLDINGS)
    df_h = read_tab(ws_h)

    # Prepare/clean
    df_clean = clean_holdings(df_h)
    if df_clean.empty:
        print("No holdings found after cleaning. Nothing to report.")
        return

    # Build final positions + summary numbers
    df_positions, total_gain, portfolio_pct, avg_pct = build_weekly_dataframe(df_clean)

    # Console summary
    print(f"Total Gain/Loss ($): {format_currency(total_gain)}")
    print(f"Portfolio % Gain  : {format_percent(portfolio_pct)}")
    print(f"Average % Gain     : {format_percent(avg_pct)}\n")

    # Optional write to sheet
    if args.write:
        write_weekly_report(gc, args.sheet_url, df_positions, total_gain, portfolio_pct, avg_pct)

    # Optional email
    if args.email:
        email_report(total_gain, portfolio_pct, avg_pct, df_positions, attach_html=args.attach_html, out_dir="output")

    print("🎯 Done.")

if __name__ == "__main__":
    main()
