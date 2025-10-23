#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_signals_and_score.py

What it does
------------
1) (Optional) Append one or many signals to the 'Signals_Log' sheet.
   - Use CLI flags for a single signal, or --signals-csv to append many.

2) Build per-source performance by cross-referencing:
   - Signals_Log  (Timestamp, Ticker, Source, Direction, Price, Timeframe)
   - Transactions (your Fidelity export in the 'Transactions' tab)

   Matching logic (simple, robust, explainable):
   - For a BUY signal: find the first BUY fill for that Ticker at/after the
     signal time (within --fill-window-days). That becomes entry.
     Then find the first SELL fill after that entry; compute return = (sell - buy)/buy.
   - For a SELL signal: mirror logic (first SELL after signal, then first BUY
     after that to close short). If no close yet -> "open".

   The script aggregates realized trades to produce a Source_Report:
   Columns: Source, Trades, Wins, WinRate%, AvgReturn%, MedianReturn%, OpenSignals

Assumptions
-----------
- Your Google Sheet already has tabs:
  - 'Signals_Log' (create if missing)
  - 'Transactions' (created by your upload script)
  - 'Source_Report' (will be created or overwritten by this script)
- Your Transactions sheet contains at least columns for Date/Time, Action, Symbol/Ticker, Quantity, Price.
  The script will try to auto-detect columns (case-insensitive).

Usage
-----
# Append a single signal (BUY) and then compute report
python3 update_signals_and_score.py \
  --ticker PLTR --source SuperiorStar --direction BUY --price 32.15 --timeframe short

# Append many signals from a CSV (columns: Timestamp(optional),Ticker,Source,Direction,Price,Timeframe)
python3 update_signals_and_score.py --signals-csv daily_signals.csv

# Only recompute the Source_Report (no new signals)
python3 update_signals_and_score.py --recompute-only

You can also tune matching windows:
  --fill-window-days 5    : how long after a signal we accept the entry fill
  --close-window-days 60  : how long after entry we look for a closing trade

"""

import os
import sys
import math
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG (edit to your sheet / creds)
# =========================
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS = "Signals_Log"
TAB_TXNS = "Transactions"
TAB_REPORT = "Source_Report"

# Columns we create/expect in Signals_Log
SIGNALS_COLUMNS = ["Timestamp", "Ticker", "Source", "Direction", "Price", "Timeframe"]

# Default windows
DEFAULT_FILL_WINDOW_DAYS = 5
DEFAULT_CLOSE_WINDOW_DAYS = 60

# =========================
# Google helpers
# =========================
def authorize():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, title):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=200, cols=20)

def ensure_header(ws, header):
    # If empty, set header row
    vals = ws.get_all_values()
    if not vals:
        ws.update("A1", [header])
        return
    # If header differs, replace it (we keep it simple)
    existing = vals[0]
    if [h.strip() for h in existing] != header:
        ws.clear()
        ws.update("A1", [header])

def append_rows(ws, rows):
    if not rows:
        return
    ws.append_rows(rows, value_input_option="RAW")

def read_sheet_as_df(ws):
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header = vals[0]
    data = vals[1:]
    df = pd.DataFrame(data, columns=header)
    # strip whitespace
    df.columns = [c.strip() for c in df.columns]
    return df

def write_df(ws, df):
    ws.clear()
    if df is None or df.empty:
        ws.update("A1", [["(empty)"]])
        return
    rows = [df.columns.tolist()] + df.astype(str).fillna("").values.tolist()
    ws.update("A1", rows)

# =========================
# Data normalization
# =========================
def parse_dt(x):
    if pd.isna(x) or x == "":
        return pd.NaT
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def as_float(x):
    try:
        if x in ("", None):
            return np.nan
        return float(str(x).replace("$","").replace(",",""))
    except Exception:
        return np.nan

def normalize_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Try to find common columns from Fidelity exports:
    Date/Time, Action, Symbol, Quantity, Price (per share)
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Date", "Action", "Ticker", "Quantity", "Price"])

    cols = {c.lower(): c for c in df_raw.columns}

    # Date/Time
    date_col = None
    for key in ["date", "date/time", "transaction date", "time", "trade date"]:
        if key in cols:
            date_col = cols[key]; break

    # Action
    action_col = None
    for key in ["action", "type", "transaction type"]:
        if key in cols:
            action_col = cols[key]; break

    # Ticker
    sym_col = None
    for key in ["symbol", "ticker", "security", "security symbol"]:
        if key in cols:
            sym_col = cols[key]; break

    # Quantity
    qty_col = None
    for key in ["quantity", "qty", "shares", "share quantity"]:
        if key in cols:
            qty_col = cols[key]; break

    # Price
    price_col = None
    for key in ["price", "price ($)", "price per share", "price/share"]:
        if key in cols:
            price_col = cols[key]; break

    # Build normalized frame
    out = pd.DataFrame()
    out["Date"] = df_raw[date_col].apply(parse_dt) if date_col else pd.NaT
    out["Action"] = df_raw[action_col].astype(str).str.strip().str.title() if action_col else ""
    out["Ticker"] = df_raw[sym_col].astype(str).str.strip().str.upper() if sym_col else ""
    out["Quantity"] = df_raw[qty_col].apply(as_float) if qty_col else np.nan
    out["Price"] = df_raw[price_col].apply(as_float) if price_col else np.nan

    # Keep only buys/sells with a ticker and a timestamp
    mask_basic = (~out["Ticker"].eq("")) & out["Date"].notna() & out["Action"].isin(["Buy","Bought","Sell","Sold"])
    out = out.loc[mask_basic].copy()

    # Normalize action values
    out["Action"] = out["Action"].replace({"Bought":"Buy", "Sold":"Sell"})

    # Sort in time
    out = out.sort_values("Date").reset_index(drop=True)
    return out

def normalize_signals(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=SIGNALS_COLUMNS)

    # Relaxed mapping
    cols = {c.lower(): c for c in df_raw.columns}
    def get_col(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    ts_col  = get_col("timestamp", "time", "datetime", "date")
    tic_col = get_col("ticker", "symbol")
    src_col = get_col("source")
    dir_col = get_col("direction", "side")
    prc_col = get_col("price", "signal price")
    tf_col  = get_col("timeframe", "horizon")

    out = pd.DataFrame()
    out["Timestamp"] = df_raw[ts_col].apply(parse_dt) if ts_col else pd.NaT
    out["Ticker"]    = df_raw[tic_col].astype(str).str.strip().str.upper() if tic_col else ""
    out["Source"]    = df_raw[src_col].astype(str).str.strip() if src_col else ""
    out["Direction"] = df_raw[dir_col].astype(str).str.strip().str.upper() if dir_col else ""
    out["Price"]     = df_raw[prc_col].apply(as_float) if prc_col else np.nan
    out["Timeframe"] = df_raw[tf_col].astype(str).str.strip().str.lower() if tf_col else ""

    # Default Timestamp if missing -> now (UTC)
    out.loc[out["Timestamp"].isna(), "Timestamp"] = pd.Timestamp.utcnow()

    # Filter sane rows
    dir_ok = out["Direction"].isin(["BUY","SELL"])
    tick_ok = out["Ticker"].ne("")
    out = out.loc[dir_ok & tick_ok].copy()
    out = out.sort_values("Timestamp").reset_index(drop=True)
    return out

# =========================
# Matching signals to fills
# =========================
def first_fill_after(txns_df, ticker, action, t0, within_days):
    """
    Find first transaction with given action for ticker at/after t0 and within window.
    Returns row or None.
    """
    if txns_df.empty: return None
    t1 = t0
    t2 = t0 + timedelta(days=within_days)
    m = (txns_df["Ticker"].eq(ticker)
         & txns_df["Action"].eq(action)
         & (txns_df["Date"] >= t1)
         & (txns_df["Date"] <= t2))
    hits = txns_df.loc[m].sort_values("Date")
    return hits.iloc[0] if len(hits) else None

def pair_round_trip(txns_df, entry_row, exit_action, close_window_days):
    """
    Given an entry (row) and desired exit action ('Sell' for long, 'Buy' for short),
    find the first exit within the close_window.
    Return (exit_row or None).
    """
    if entry_row is None: return None
    t0 = entry_row["Date"]
    t2 = t0 + timedelta(days=close_window_days)
    m = (txns_df["Ticker"].eq(entry_row["Ticker"])
         & txns_df["Action"].eq(exit_action)
         & (txns_df["Date"] > t0)
         & (txns_df["Date"] <= t2))
    hits = txns_df.loc[m].sort_values("Date")
    return hits.iloc[0] if len(hits) else None

def compute_return(entry_price, exit_price, direction):
    if any(pd.isna([entry_price, exit_price])):
        return np.nan
    if direction == "BUY":
        return (exit_price - entry_price) / entry_price
    else:  # SELL (short)
        return (entry_price - exit_price) / entry_price

def score_sources(signals_df: pd.DataFrame,
                  txns_df: pd.DataFrame,
                  fill_window_days: int,
                  close_window_days: int) -> pd.DataFrame:
    """
    For each signal, try to match entry+exit and compute a single-trade return.
    Aggregate by Source.
    """
    if signals_df.empty:
        return pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenSignals"])

    results = []
    for i, s in signals_df.iterrows():
        tkr   = s["Ticker"]
        src   = s["Source"] or "(unknown)"
        direc = s["Direction"]  # BUY/SELL
        ts    = s["Timestamp"]
        sig_p = s.get("Price", np.nan)

        # Map direction to first fill action
        want_action = "Buy" if direc == "BUY" else "Sell"
        entry_row = first_fill_after(txns_df, tkr, want_action, ts, within_days=fill_window_days)

        if entry_row is None:
            results.append({"Source": src, "Ticker": tkr, "Open": True, "Ret": np.nan})
            continue

        entry_price = entry_row["Price"]
        # Determine closing action
        close_action = "Sell" if direc == "BUY" else "Buy"
        exit_row = pair_round_trip(txns_df, entry_row, close_action, close_window_days)

        if exit_row is None:
            results.append({"Source": src, "Ticker": tkr, "Open": True, "Ret": np.nan})
            continue

        exit_price = exit_row["Price"]
        r = compute_return(entry_price, exit_price, direc)
        results.append({"Source": src, "Ticker": tkr, "Open": False, "Ret": r})

    res = pd.DataFrame(results)
    # Aggregate
    grouped = []
    for src, g in res.groupby("Source"):
        realized = g.loc[~g["Open"]].copy()
        trades = len(realized)
        wins = int((realized["Ret"] > 0).sum()) if trades else 0
        win_rate = (wins / trades * 100.0) if trades else 0.0
        avg_ret = (realized["Ret"].mean() * 100.0) if trades else 0.0
        med_ret = (realized["Ret"].median() * 100.0) if trades else 0.0
        open_cnt = int(g["Open"].sum())
        grouped.append({
            "Source": src,
            "Trades": trades,
            "Wins": wins,
            "WinRate%": round(win_rate, 2),
            "AvgReturn%": round(avg_ret, 2),
            "MedianReturn%": round(med_ret, 2),
            "OpenSignals": open_cnt
        })
    rep = pd.DataFrame(grouped).sort_values(["Trades","WinRate%","AvgReturn%"], ascending=[False, False, False])
    return rep

# =========================
# Signals append
# =========================
def append_single_signal(ws_signals, ticker, source, direction, price, timeframe):
    ensure_header(ws_signals, SIGNALS_COLUMNS)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    row = [now_utc, ticker.upper(), source, direction.upper(), str(price) if price is not None else "", timeframe.lower()]
    append_rows(ws_signals, [row])
    print(f"✅ Appended signal: {row}")

def append_signals_from_csv(ws_signals, csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df_raw = pd.read_csv(csv_path)
    df = normalize_signals(df_raw)
    ensure_header(ws_signals, SIGNALS_COLUMNS)

    # Convert to strings for upload
    out = []
    for _, r in df.iterrows():
        ts = r["Timestamp"]
        ts_str = pd.to_datetime(ts).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
        out.append([
            ts_str,
            r["Ticker"],
            r["Source"] or "",
            r["Direction"].upper(),
            "" if pd.isna(r["Price"]) else str(r["Price"]),
            r["Timeframe"] or ""
        ])
    append_rows(ws_signals, out)
    print(f"✅ Appended {len(out)} signals from {csv_path}")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Append signals and compute per-source performance report.")
    ap.add_argument("--ticker", help="Ticker for a single signal")
    ap.add_argument("--source", help="Source name (Sarkee, SuperiorStar, Weinstein, Bo, etc.)")
    ap.add_argument("--direction", choices=["BUY","SELL"], help="Signal direction")
    ap.add_argument("--price", type=float, help="Signal price (optional)")
    ap.add_argument("--timeframe", default="", help="Signal timeframe (e.g., short, medium, long)")

    ap.add_argument("--signals-csv", help="CSV with columns: Timestamp(optional),Ticker,Source,Direction,Price,Timeframe")
    ap.add_argument("--recompute-only", action="store_true", help="Only recompute Source_Report; do not append new signals")

    ap.add_argument("--fill-window-days", type=int, default=DEFAULT_FILL_WINDOW_DAYS, help="Days after a signal to accept first fill")
    ap.add_argument("--close-window-days", type=int, default=DEFAULT_CLOSE_WINDOW_DAYS, help="Days after entry to search for exit")

    args = ap.parse_args()

    # Auth & open sheets
    gc = authorize()
    ws_signals = open_ws(gc, TAB_SIGNALS)
    ws_txns    = open_ws(gc, TAB_TXNS)
    ws_report  = open_ws(gc, TAB_REPORT)

    # Maybe append new signals
    if not args.recompute_only:
        if args.signals_csv:
            append_signals_from_csv(ws_signals, args.signals_csv)
        elif args.ticker and args.source and args.direction:
            append_single_signal(ws_signals,
                                 ticker=args.ticker,
                                 source=args.source,
                                 direction=args.direction,
                                 price=args.price,
                                 timeframe=args.timeframe)
        else:
            print("ℹ️ No new signals appended (pass --signals-csv OR --ticker/--source/--direction).")

    # Read data back for scoring
    df_signals_raw = read_sheet_as_df(ws_signals)
    if df_signals_raw.empty:
        print("⚠️ Signals_Log is empty. Nothing to score.")
        write_df(ws_report, pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenSignals"]))
        return

    # Ensure proper columns / normalization
    # If the sheet already has our schema, use it; otherwise normalize flexibly
    if set(SIGNALS_COLUMNS).issubset(df_signals_raw.columns):
        df_signals = df_signals_raw.copy()
        df_signals["Timestamp"] = df_signals["Timestamp"].apply(parse_dt)
        df_signals["Ticker"] = df_signals["Ticker"].astype(str).str.upper()
        df_signals["Source"] = df_signals["Source"].astype(str)
        df_signals["Direction"] = df_signals["Direction"].astype(str).str.upper()
        df_signals["Price"] = df_signals["Price"].apply(as_float)
        df_signals["Timeframe"] = df_signals["Timeframe"].astype(str)
        df_signals = df_signals.sort_values("Timestamp")
    else:
        df_signals = normalize_signals(df_signals_raw)

    df_txns_raw = read_sheet_as_df(ws_txns)
    df_txns = normalize_transactions(df_txns_raw)

    # Score per source
    rep = score_sources(df_signals, df_txns, args.fill_window_days, args.close_window_days)

    # Write Source_Report
    write_df(ws_report, rep)
    print("📈 Source_Report updated.")
    if not rep.empty:
        print(rep.to_string(index=False))

if __name__ == "__main__":
    main()
