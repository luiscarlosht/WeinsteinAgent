#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Weekly Report

Reads current holdings from Google Sheets and produces:
- A summary metrics block (Total Gain/Loss $, Portfolio % Gain, Average % Gain)
- A per-position table with Weinstein SELL/HOLD recommendations
- Writes a 'Weekly_Report' tab in Google Sheets
- Saves CSV/HTML snapshots to ./output
- Optionally emails the report with attachments

Usage:
  python3 weinstein_report_weekly.py [--write] [--email]
                                     [--attach-html] [--attach-csv]
                                     [--to you@example.com ...]
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

# Import email sender
from WeinsteinMinimalEmailSender import send_email


# ─────────────────────────────
# CONFIG DEFAULTS
# ─────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"
TAB_HOLDINGS_DEFAULT = "Holdings"
TAB_WEEKLY_DEFAULT = "Weekly_Report"
OUTPUT_DIR = "output"

STRONG_HOLD_PCT = 50.0
SELL_PCT = -10.0


# ─────────────────────────────
# GOOGLE SHEETS HELPERS
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
        ws.update([["(empty)"]])
        return
    rows, cols = df.shape
    ws.resize(rows=max(50, rows + 5), cols=max(10, cols + 2))
    data = [df.columns.tolist()] + df.astype(str).fillna("").values.tolist()

    # chunk upload
    start = 0
    r = 1
    while start < len(data):
        end = min(start + 500, len(data))
        ws.update(data[start:end], range_name=f"A{r}")
        r += len(data[start:end])
        start = end


# ─────────────────────────────
# DATA PROCESSING
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
    if df_h.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Description",
                "Quantity",
                "Last Price",
                "Current Value",
                "Cost Basis Total",
                "Average Cost Basis",
                "Total Gain/Loss Dollar",
                "Total Gain/Loss Percent",
            ]
        )

    df = df_h.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "Symbol": ["Symbol", "Ticker"],
        "Description": ["Description"],
        "Quantity": ["Quantity", "Qty"],
        "Last Price": ["Last Price", "Price"],
        "Current Value": ["Current Value", "Market Value"],
        "Cost Basis Total": ["Cost Basis Total", "Cost Basis"],
        "Average Cost Basis": ["Average Cost Basis", "Avg Cost"],
        "Total Gain/Loss Dollar": ["Total Gain/Loss Dollar", "Gain/Loss $"],
        "Total Gain/Loss Percent": ["Total Gain/Loss Percent", "Gain/Loss %"],
    }

    col_map = {}
    for key, alts in rename_map.items():
        for a in alts:
            if a in df.columns:
                col_map[key] = a
                break

    for k in rename_map.keys():
        if k not in col_map:
            df[k] = ""
            col_map[k] = k

    out = df[list(col_map.values())].copy()
    out.columns = list(col_map.keys())

    for c in [
        "Quantity",
        "Last Price",
        "Current Value",
        "Cost Basis Total",
        "Average Cost Basis",
        "Total Gain/Loss Dollar",
    ]:
        out[c] = out[c].map(_to_float)

    def to_pct(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace("%", "").replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            return None

    out["Total Gain/Loss Percent"] = out["Total Gain/Loss Percent"].map(to_pct)
    out = out[out["Symbol"].astype(str).str.strip().ne("")]
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


def compute_summary(df: pd.DataFrame) -> Tuple[float, float, float]:
    if df.empty:
        return 0.0, 0.0, 0.0
    total_gain = df["Total Gain/Loss Dollar"].fillna(0.0).sum()
    cv = df["Current Value"].fillna(0.0).sum()
    cb = df["Cost Basis Total"].fillna(0.0).sum()
    portfolio_pct = (cv - cb) / cb * 100.0 if cb else 0.0
    avg_pct = df["Total Gain/Loss Percent"].dropna().mean() if not df.empty else 0.0
    return total_gain, portfolio_pct, avg_pct


def build_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Recommendation"] = out["Total Gain/Loss Percent"].map(weinstein_recommendation)
    cols = [
        "Symbol",
        "Description",
        "Quantity",
        "Last Price",
        "Current Value",
        "Cost Basis Total",
        "Average Cost Basis",
        "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent",
        "Recommendation",
    ]
    return out[cols]


def build_weekly_sheet(
    view_df: pd.DataFrame, total_gain: float, portfolio_pct: float, avg_pct: float
) -> pd.DataFrame:
    top = pd.DataFrame(
        {
            "Metric": ["Total Gain/Loss ($)", "Portfolio % Gain", "Average % Gain"],
            "Value": [
                f"${total_gain:,.2f}",
                f"{portfolio_pct:.2f}%",
                f"{avg_pct:.2f}%",
            ],
        }
    )
    spc = pd.DataFrame({"Metric": [""], "Value": [""]})
    bot = view_df.copy()
    top = top.reindex(columns=list(bot.columns)[:2], fill_value="")
    spc = spc.reindex(columns=list(bot.columns)[:2], fill_value="")
    return pd.concat([top, spc, bot], ignore_index=True)


# ─────────────────────────────
# SNAPSHOTS + EMAIL
# ─────────────────────────────
def save_snapshots(df_view: pd.DataFrame) -> Tuple[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(OUTPUT_DIR, f"weinstein_weekly_{ts}.csv")
    html_path = os.path.join(OUTPUT_DIR, f"weinstein_weekly_{ts}.html")
    df_view.to_csv(csv_path, index=False)
    df_view.to_html(html_path, index=False)
    return csv_path, html_path


def email_report(
    subject: str,
    summary_text: str,
    html_report_path: str | None,
    csv_report_path: str | None,
    to_list: List[str] | None,
    sheet_url: str,
):
    summary_html = summary_text.replace("\n", "<br>")
    html_table = ""
    if html_report_path and os.path.exists(html_report_path):
        with open(html_report_path, "r", encoding="utf-8") as f:
            html_table = f.read()

    body_html = f"""
    <html>
      <body>
        <p>{summary_html}</p>
        <p><b>Google Sheet:</b> <a href="{sheet_url}">{sheet_url}</a></p>
        <hr>{html_table}
      </body>
    </html>
    """

    attachments = []
    if html_report_path and os.path.exists(html_report_path):
        attachments.append(html_report_path)
    if csv_report_path and os.path.exists(csv_report_path):
        attachments.append(csv_report_path)

    send_email(
        subject=subject,
        body_text=summary_text,
        body_html=body_html,
        to=to_list,
        attachments=attachments,
    )


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--email", action="store_true")
    ap.add_argument("--attach-html", action="store_true")
    ap.add_argument("--attach-csv", action="store_true")
    ap.add_argument("--to", nargs="*", default=None)
    args = ap.parse_args()

    print("📊 Generating weekly Weinstein report…")
    gc = auth_gspread()

    ws = open_ws(gc, DEFAULT_SHEET_URL, TAB_HOLDINGS_DEFAULT)
    df = read_tab(ws)
    df_hold = load_holdings(df)
    total_gain, portfolio_pct, avg_pct = compute_summary(df_hold)

    print(f"Total Gain/Loss ($): {total_gain:,.2f}")
    print(f"Portfolio % Gain  : {portfolio_pct:.2f}%")
    print(f"Average % Gain     : {avg_pct:.2f}%\n")

    df_view = build_view(df_hold)
    csv_path, html_path = save_snapshots(df_view)

    if args.write:
        ws_out = open_ws(gc, DEFAULT_SHEET_URL, TAB_WEEKLY_DEFAULT)
        out_df = build_weekly_sheet(df_view, total_gain, portfolio_pct, avg_pct)
        write_tab(ws_out, out_df)
        print("✅ Wrote Weekly_Report tab.")

    if args.email:
        subject = f"Weinstein Weekly Report — {datetime.now().strftime('%b %d %Y %I:%M %p')}"
        summary = (
            f"=== Weinstein Weekly Report ===\n"
            f"Total Gain/Loss ($): {total_gain:,.2f}\n"
            f"Portfolio % Gain  : {portfolio_pct:.2f}%\n"
            f"Average % Gain    : {avg_pct:.2f}%"
        )
        email_report(
            subject,
            summary,
            html_path if args.attach_html else None,
            csv_path if args.attach_csv else None,
            args.to,
            DEFAULT_SHEET_URL,
        )
        print("📧 Email sent with attachments.")

    print("🎯 Done.")


if __name__ == "__main__":
    main()
