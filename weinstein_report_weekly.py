#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
weinstein_report_weekly.py

Weekly report that pulls Holdings from your Google Sheet and produces:
- Total portfolio gain/loss ($)
- Portfolio % gain (sum gain / sum cost basis)
- Average % gain across positions
- Per-position recommendation: SELL / HOLD (configurable thresholds)

Optional: write a "Weekly_Report" tab back to the sheet.

Usage examples:
  python3 weinstein_report_weekly.py
  python3 weinstein_report_weekly.py --write
  python3 weinstein_report_weekly.py --sell-thresh -7 --strong-hold 12 --write
  python3 weinstein_report_weekly.py --sheet-url "https://docs.google.com/..." --creds "creds/gcp_service_account.json"

Notes:
- Reuses the same service account method as your other scripts (gspread).
- Designed for your Fidelity-style Holdings tab columns, e.g.:
    "Symbol", "Description", "Quantity", "Last Price",
    "Current Value", "Total Gain/Loss Dollar", "Total Gain/Loss Percent",
    "Cost Basis Total", "Average Cost Basis", "Type"
- Cash rows like FCASH** / SPAXX** / "Pending activity" are ignored automatically.
"""

from __future__ import annotations

import argparse
import re
from typing import Tuple

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


# ─────────────────────────────
# DEFAULT CONFIG (matches your repo)
# ─────────────────────────────
SERVICE_ACCOUNT_FILE_DEFAULT = "creds/gcp_service_account.json"
SHEET_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_HOLDINGS = "Holdings"
TAB_WEEKLY   = "Weekly_Report"

ROW_CHUNK = 500


# ─────────────────────────────
# AUTH / IO HELPERS
# ─────────────────────────────
def auth_gspread(creds_path: str) -> gspread.Client:
    print("🔑 Authorizing service account…")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    return gspread.authorize(creds)

def open_ws(gc: gspread.Client, sheet_url: str, tab: str) -> gspread.Worksheet:
    sh = gc.open_by_url(sheet_url)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=2000, cols=26)

def read_tab(ws: gspread.Worksheet) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def write_tab(ws: gspread.Worksheet, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.resize(rows=50, cols=8)
        ws.update([["(empty)"]], range_name="A1")
        return

    # resize conservatively
    rows, cols = df.shape
    ws.resize(rows=max(100, rows + 5), cols=max(8, min(26, cols + 2)))

    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

    start = 0
    r = 1
    while start < len(data):
        end = min(start + ROW_CHUNK, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)
        r += len(block)
        start = end


# ─────────────────────────────
# PARSING HELPERS (robust to $ , % etc.)
# ─────────────────────────────
_CASHY = {"FCASH", "FCASH**", "SPAXX", "SPAXX**", "PENDING", "PENDING ACTIVITY", "PENDING ACTIVITY*", "CASH"}

def _to_float(series: pd.Series) -> pd.Series:
    def conv(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return np.nan
        s = s.replace("$", "").replace(",", "")
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.map(conv)

def _to_pct(series: pd.Series) -> pd.Series:
    def conv(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace("%", "")
        if s == "":
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.map(conv)

def _looks_like_cash(symbol: str, description: str) -> bool:
    if not symbol:
        return True
    s = symbol.strip().upper()
    if s in _CASHY:
        return True
    if "CASH" in s:
        return True
    desc = (description or "").strip().upper()
    if "HELD IN" in desc and "CASH" in desc:
        return True
    return False


# ─────────────────────────────
# ANALYSIS
# ─────────────────────────────
def prepare_holdings_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns from a Fidelity-style Holdings export.
    Returns a filtered numeric dataframe for positions (no cash/pending lines).
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Symbol","Description","Quantity","Last Price","Current Value",
            "Total Gain/Loss Dollar","Total Gain/Loss Percent","Cost Basis Total","Average Cost Basis","Type"
        ])

    # Column name resolution (case-insensitive, tolerate minor variations)
    def pick(*cands):
        for c in df.columns:
            cl = c.lower()
            for k in cands:
                if cl == k.lower():
                    return c
        # fallback partial contains
        for c in df.columns:
            cl = c.lower()
            for k in cands:
                if k.lower() in cl:
                    return c
        return None

    c_sym  = pick("Symbol")
    c_desc = pick("Description")
    c_qty  = pick("Quantity")
    c_lp   = pick("Last Price")
    c_cv   = pick("Current Value")
    c_tgld = pick("Total Gain/Loss Dollar", "Total Gain/Loss $")
    c_tglp = pick("Total Gain/Loss Percent", "Total Gain/Loss %")
    c_cb   = pick("Cost Basis Total")
    c_acb  = pick("Average Cost Basis")
    c_type = pick("Type")

    # Build a working frame with safe defaults
    out = pd.DataFrame({
        "Symbol": df.get(c_sym, ""),
        "Description": df.get(c_desc, ""),
        "Quantity": _to_float(df.get(c_qty, np.nan)),
        "Last Price": _to_float(df.get(c_lp, np.nan)),
        "Current Value": _to_float(df.get(c_cv, np.nan)),
        "Total Gain/Loss Dollar": _to_float(df.get(c_tgld, np.nan)),
        "Total Gain/Loss Percent": _to_pct(df.get(c_tglp, np.nan)),
        "Cost Basis Total": _to_float(df.get(c_cb, np.nan)),
        "Average Cost Basis": _to_float(df.get(c_acb, np.nan)),
        "Type": df.get(c_type, ""),
    })

    # Drop obvious cash/pending rows
    mask_cash = [
        _looks_like_cash(str(sym), str(desc))
        for sym, desc in zip(out["Symbol"].astype(str), out["Description"].astype(str))
    ]
    out = out.loc[~pd.Series(mask_cash, index=out.index)].copy()

    # Keep only positive-quantity or positive-value positions
    out = out[(out["Quantity"].fillna(0) > 0) | (out["Current Value"].fillna(0) > 0)]
    out.reset_index(drop=True, inplace=True)
    return out

def recommend_row(pct_gain: float, sell_thresh: float, strong_hold: float) -> str:
    """
    Simple rule:
      pct <= sell_thresh  => SELL
      pct >= strong_hold  => HOLD (Strong)
      else                => HOLD
    """
    if pd.isna(pct_gain):
        return "HOLD"
    if pct_gain <= sell_thresh:
        return "SELL"
    if pct_gain >= strong_hold:
        return "HOLD (Strong)"
    return "HOLD"

def analyze_holdings(
    df_pos: pd.DataFrame,
    sell_thresh: float,
    strong_hold: float
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Returns:
      per_position_df, total_gain_dollar, portfolio_gain_pct, avg_pct
    portfolio_gain_pct = (sum $gain) / (sum cost basis) * 100
    avg_pct            = mean of Total Gain/Loss Percent (across positions)
    """
    if df_pos.empty:
        return df_pos, 0.0, 0.0, 0.0

    # Compute missing fields if needed
    if df_pos["Total Gain/Loss Dollar"].isna().any():
        # if we have CV and CB, infer
        can = df_pos["Current Value"].notna() & df_pos["Cost Basis Total"].notna()
        df_pos.loc[can, "Total Gain/Loss Dollar"] = (
            df_pos.loc[can, "Current Value"] - df_pos.loc[can, "Cost Basis Total"]
        )

    if df_pos["Total Gain/Loss Percent"].isna().any():
        can = df_pos["Total Gain/Loss Dollar"].notna() & df_pos["Cost Basis Total"].notna() & (df_pos["Cost Basis Total"] != 0)
        df_pos.loc[can, "Total Gain/Loss Percent"] = (
            df_pos.loc[can, "Total Gain/Loss Dollar"] / df_pos.loc[can, "Cost Basis Total"] * 100.0
        )

    # Recommendations
    df_pos["Recommendation"] = df_pos["Total Gain/Loss Percent"].map(
        lambda p: recommend_row(p, sell_thresh=sell_thresh, strong_hold=strong_hold)
    )

    total_gain = float(df_pos["Total Gain/Loss Dollar"].fillna(0).sum())
    total_cost = float(df_pos["Cost Basis Total"].fillna(0).sum())
    portfolio_pct = (total_gain / total_cost * 100.0) if total_cost else 0.0
    avg_pct = float(df_pos["Total Gain/Loss Percent"].dropna().mean()) if not df_pos["Total Gain/Loss Percent"].dropna().empty else 0.0

    # Nicely rounded presentation columns (keep numeric under the hood)
    df_view = df_pos.copy()
    for col in ["Quantity", "Last Price", "Current Value", "Total Gain/Loss Dollar", "Cost Basis Total", "Average Cost Basis"]:
        df_view[col] = df_view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    df_view["Total Gain/Loss Percent"] = df_view["Total Gain/Loss Percent"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")

    return df_view, total_gain, portfolio_pct, avg_pct


# ─────────────────────────────
# SHEET OUTPUT (Weekly_Report)
# ─────────────────────────────
def build_weekly_sheet(df_view: pd.DataFrame, total_gain: float, portfolio_pct: float, avg_pct: float) -> pd.DataFrame:
    """
    Assemble a nice table with a metrics header + per-position rows for "Weekly_Report".
    """
    summary = pd.DataFrame({
        "Metric": ["Total Gain/Loss ($)", "Portfolio % Gain", "Average % Gain"],
        "Value":  [f"${total_gain:,.2f}", f"{portfolio_pct:.2f}%", f"{avg_pct:.2f}%"]
    })
    spacer = pd.DataFrame({"Metric": [""], "Value": [""]})

    # Reorder a sensible set of columns for the per-position section
    cols = [
        "Symbol", "Description", "Quantity", "Last Price", "Current Value",
        "Cost Basis Total", "Average Cost Basis",
        "Total Gain/Loss Dollar", "Total Gain/Loss Percent", "Recommendation"
    ]
    present = [c for c in cols if c in df_view.columns]
    positions = df_view[present].copy()

    # Compose into one sheet-like frame
    # We’ll return it as one DataFrame with an empty row separating sections;
    # write_tab() handles the actual upload.
    top = summary
    bot = positions

    # Convert to a common schema by padding columns
    max_cols = max(len(top.columns), len(bot.columns), len(spacer.columns))
    def pad(df, n):
        if len(df.columns) < n:
            extra = [f"" for _ in range(n - len(df.columns))]
            newcols = list(df.columns) + [f""] * (n - len(df.columns))
            df = df.copy()
            df.columns = newcols
        return df

    top = pad(top, max_cols)
    spacer = pad(spacer, max_cols)
    bot = pad(bot, max_cols)

    combined = pd.concat([top, spacer, bot], ignore_index=True)
    return combined


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Weekly Weinstein report from Google Sheets Holdings.")
    ap.add_argument("--sheet-url", default=SHEET_URL_DEFAULT, help="Google Sheet URL.")
    ap.add_argument("--creds", default=SERVICE_ACCOUNT_FILE_DEFAULT, help="Path to service account JSON.")
    ap.add_argument("--sell-thresh", type=float, default=-5.0, help="SELL if pct <= this (default: -5)")
    ap.add_argument("--strong-hold", type=float, default=10.0, help="HOLD (Strong) if pct >= this (default: 10)")
    ap.add_argument("--write", action="store_true", help=f"Write a '{TAB_WEEKLY}' tab to the sheet.")
    args = ap.parse_args()

    print("📊 Generating weekly Weinstein report…")
    gc = auth_gspread(args.creds)

    # Fetch holdings
    ws_h = open_ws(gc, args.sheet_url, TAB_HOLDINGS)
    df_hold = read_tab(ws_h)
    if df_hold.empty:
        print("⚠️ Holdings tab is empty. Nothing to report.")
        return

    # Normalize & analyze
    pos = prepare_holdings_frame(df_hold)
    df_view, total_gain, portfolio_pct, avg_pct = analyze_holdings(
        pos, sell_thresh=args.sell_thresh, strong_hold=args.strong_hold
    )

    # Console summary
    print("\n=== Weinstein Weekly Report ===")
    print(f"Total Gain/Loss ($): {total_gain:,.2f}")
    print(f"Portfolio % Gain  : {portfolio_pct:.2f}%")
    print(f"Average % Gain     : {avg_pct:.2f}%")
    print("\nPer-position snapshot:")
    if df_view.empty:
        print("(no equity positions found)")
    else:
        cols = ["Symbol","Description","Quantity","Current Value","Cost Basis Total","Total Gain/Loss Dollar","Total Gain/Loss Percent","Recommendation"]
        cols = [c for c in cols if c in df_view.columns]
        print(df_view[cols].to_string(index=False))

    # Optional write to sheet
    if args.write:
        ws_out = open_ws(gc, args.sheet_url, TAB_WEEKLY)
        out_df = build_weekly_sheet(df_view, total_gain, portfolio_pct, avg_pct)
        write_tab(ws_out, out_df)
        print(f"\n✅ Wrote '{TAB_WEEKLY}' tab with summary and per-position details.")

    print("\n🎯 Done.")


if __name__ == "__main__":
    main()
