#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_fidelity_with_signals.py

Backfill missing fields in the Google Sheet 'Signals' tab using:
  1) Brokerage Transactions (preferred)
  2) Current Holdings (fallback)
Plus:
  - Default Timeframe per Source from Mapping!A:B
  - Ticker alias normalization from Mapping!D:E

Required tabs in the target Google Sheet:
  - Signals
  - Transactions
  - Holdings
  - Mapping (optional but recommended)

Columns expected on Signals:
  ['TimestampUTC','Ticker','Source','Direction','Price','Timeframe']

Run: python3 merge_fidelity_with_signals.py
"""

import os
import re
import sys
import math
import time
import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG
# =========================
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS = "Signals"
TAB_TXNS    = "Transactions"
TAB_HOLD    = "Holdings"
TAB_MAP     = "Mapping"

# Default timeframe by source (overridable via Mapping!A:B)
SOURCE_DEFAULT_TIMEFRAME = {
    "Sarkee Capital": "short",
    "SuperiorStar":   "mid",
    "Weinstein":      "long",
    "Bo Xu":          "short",
}

ROW_CHUNK = 500  # rows per update chunk


# =========================
# SHEETS UTILS
# =========================
def authorize():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, title: str):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=100, cols=26)

def ws_to_df(ws) -> pd.DataFrame:
    """Read a worksheet to DataFrame, preserving header."""
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header = [h.strip() for h in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    # Strip whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def clear_and_size(ws, n_rows: int, n_cols: int):
    ws.clear()
    ws.resize(rows=max(n_rows, 100), cols=max(n_cols, 26))

def chunked_update(ws, values, start_row=1, start_col=1):
    if not values:
        return
    total = len(values)
    i = 0
    while i < total:
        j = min(i + ROW_CHUNK, total)
        n_cols = len(values[0]) if values else 1
        tl = gspread.utils.rowcol_to_a1(start_row + i, start_col)
        br = gspread.utils.rowcol_to_a1(start_row + j - 1, start_col + n_cols - 1)
        ws.update(values[i:j], f"{tl}:{br}")  # values first (new gspread order)
        i = j

def df_to_sheet(ws, df: pd.DataFrame):
    df = df.copy()
    df = df.fillna("")
    values = [df.columns.tolist()] + df.values.tolist()
    clear_and_size(ws, len(values), len(df.columns))
    chunked_update(ws, values)


# =========================
# MAPPING
# =========================
def read_mapping_tables(gc):
    """Read Mapping!A:B (default timeframe) and Mapping!D:E (alias map)."""
    try:
        ws = open_ws(gc, TAB_MAP)
    except Exception:
        return {}, {}

    df = ws_to_df(ws)
    tf_map, alias_map = {}, {}

    if df.empty:
        return tf_map, alias_map

    # Default timeframe (A:B) columns must be named exactly 'Source','DefaultTimeframe'
    if {"Source", "DefaultTimeframe"}.issubset(df.columns):
        tmp = df[["Source", "DefaultTimeframe"]].dropna()
        for _, r in tmp.iterrows():
            s = str(r["Source"]).strip()
            t = str(r["DefaultTimeframe"]).strip()
            if s and t:
                tf_map[s] = t

    # Alias map (D:E) columns must be named 'Alias','Ticker'
    if {"Alias", "Ticker"}.issubset(df.columns):
        tmp = df[["Alias", "Ticker"]].dropna()
        for _, r in tmp.iterrows():
            a = str(r["Alias"]).strip().upper()
            t = str(r["Ticker"]).strip().upper()
            if a and t:
                alias_map[a] = t

    return tf_map, alias_map

def normalize_ticker(raw: str, alias_map: dict) -> str:
    x = (raw or "").strip().upper()
    if not x:
        return ""
    if x in alias_map:
        return alias_map[x]
    # peel common noise/suffixes and re-check
    base = re.split(r"\s|\/|\-|\(|\)|\[|\]|,|;", x)[0]
    return alias_map.get(base, base)


# =========================
# SIGNALS SHAPE
# =========================
SIG_COLS = ["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"]

def ensure_signals_header(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SIG_COLS)
    for c in SIG_COLS:
        if c not in df.columns:
            df[c] = ""
    # keep only in canonical order + any extras
    ordered = df[SIG_COLS].copy()
    # bring back any extra columns the user added
    for c in df.columns:
        if c not in ordered.columns:
            ordered[c] = df[c]
    return ordered


# =========================
# TRANSACTIONS LOOKUP
# =========================
def parse_float(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(str(x).replace(",", ""))
    except Exception:
        return None

def first_present(d: dict, keys):
    for k in keys:
        if k in d and str(d[k]).strip():
            return d[k]
    return ""

def build_txn_lookup(df_tx: pd.DataFrame) -> dict:
    """Return latest BUY/SELL per ticker: { TKR: {'Direction','Price','TimestampUTC'} }"""
    if df_tx.empty:
        return {}

    # Normalize columns (lowercase map)
    cols = {c.lower(): c for c in df_tx.columns}
    # Likely columns in Fidelity export
    c_action = cols.get("action", "Action")
    c_type   = cols.get("type", "Type")
    c_sym    = cols.get("symbol", "Symbol")
    c_desc   = cols.get("description", "Description")
    c_price  = cols.get("price ($)", "Price ($)")
    c_amt    = cols.get("amount ($)", "Amount ($)")
    c_run    = cols.get("run date", "Run Date")
    c_settle = cols.get("settlement date", "Settlement Date")

    # Consider only BUY/SELL-ish rows
    mask = (
        df_tx[c_action].str.contains("BUY|SELL", case=False, na=False) |
        df_tx[c_type].str.contains("BUY|SELL", case=False, na=False)
    )
    tx = df_tx[mask].copy()
    if tx.empty:
        return {}

    # Build normalized ticker, price, direction, timestamp
    out = {}
    for _, r in tx.iterrows():
        raw_sym = str(first_present(r, [c_sym, c_desc])).strip()
        tkr = normalize_ticker(raw_sym, {})  # no alias here (aliases mostly for Signals/holdings side)
        if not tkr:
            continue

        typ = f"{r.get(c_type,'')}".upper()
        act = f"{r.get(c_action,'')}".upper()
        direction = "SELL" if ("SELL" in typ or "SELL" in act) else "BUY"

        price = parse_float(r.get(c_price))
        # fallback: sometimes only Amount is present; avoid using Amount if it’s total cash
        # We prefer explicit price column; otherwise leave blank
        ts = str(first_present(r, [c_run, c_settle])).strip()
        if ts:
            try:
                ts = pd.to_datetime(ts).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S+0000")
            except Exception:
                # leave raw
                pass
        else:
            ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S%z")

        # Keep the most recent (by row order assume lower in file is newer; or overwrite)
        out[tkr] = {
            "Direction": direction,
            "Price": "" if price is None or not np.isfinite(price) else f"{price:.6f}".rstrip("0").rstrip("."),
            "TimestampUTC": ts,
        }

    return out


# =========================
# HOLDINGS LOOKUP (FALLBACK)
# =========================
def build_holdings_lookup(df_hold: pd.DataFrame) -> dict:
    """Return per-ticker default info from Holdings."""
    if df_hold.empty:
        return {}
    cols = {c.lower(): c for c in df_hold.columns}
    c_sym = cols.get("symbol", "Symbol")

    # prefer Last Price, then Price, then Close/Price
    price_candidates = ["last price", "price", "last price ($)", "close", "market price"]
    c_price = None
    for pc in price_candidates:
        if pc in cols:
            c_price = cols[pc]
            break

    lut = {}
    for _, r in df_hold.iterrows():
        tkr = normalize_ticker(str(r.get(c_sym, "")).strip(), {})
        if not tkr:
            continue
        p = parse_float(r.get(c_price)) if c_price else None
        price = "" if p is None or not np.isfinite(p) else f"{p:.6f}".rstrip("0").rstrip(".")
        lut[tkr] = {
            "Direction": "BUY",
            "Price": price,
            "TimestampUTC": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S%z"),
        }
    return lut


# =========================
# MAIN
# =========================
def main():
    gc = authorize()
    print("📄 Reading tabs…")
    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_tx  = open_ws(gc, TAB_TXNS)
    try:
        ws_h = open_ws(gc, TAB_HOLD)
    except Exception:
        ws_h = None

    # Load
    df_sig = ws_to_df(ws_sig)
    df_tx  = ws_to_df(ws_tx)
    df_h   = ws_to_df(ws_h) if ws_h else pd.DataFrame()

    print(f"• Signals rows: {len(df_sig)}")
    print(f"• Transactions rows: {len(df_tx)}")
    print(f"• Holdings rows: {len(df_h)}")

    # Mapping (defaults + aliases)
    tf_map, alias_map = read_mapping_tables(gc)
    if tf_map:
        SOURCE_DEFAULT_TIMEFRAME.update(tf_map)

    # Ensure Signals shape and normalize tickers via alias map
    df_sig = ensure_signals_header(df_sig)
    if not df_sig.empty:
        df_sig["Ticker"] = df_sig["Ticker"].map(lambda t: normalize_ticker(t, alias_map))

    # Build lookups
    latest_txn = build_txn_lookup(df_tx)
    hold_lut   = build_holdings_lookup(df_h)
    print(f"• Latest tickers from transactions: {len(latest_txn)}")
    print(f"• Tickers from holdings: {len(hold_lut)}")

    # Backfill
    if df_sig.empty:
        print("ℹ️ Signals is empty. Nothing to fill.")
        return

    filled = 0
    for i in range(len(df_sig)):
        tkr = str(df_sig.at[i, "Ticker"]).strip().upper()
        if not tkr:
            continue

        # Default timeframe by Source if missing
        if not str(df_sig.at[i, "Timeframe"]).strip():
            src = str(df_sig.at[i, "Source"]).strip()
            if src in SOURCE_DEFAULT_TIMEFRAME:
                df_sig.at[i, "Timeframe"] = SOURCE_DEFAULT_TIMEFRAME[src]

        # Prefer transactions; fallback to holdings
        info = latest_txn.get(tkr) or hold_lut.get(tkr)
        if not info:
            continue

        for col in ("Direction", "Price", "TimestampUTC"):
            if not str(df_sig.at[i, col]).strip():
                df_sig.at[i, col] = info.get(col, "")
                filled += 1

    if filled:
        print(f"✏️ Backfilled {filled} empty fields in Signals. Writing back…")
        df_to_sheet(ws_sig, df_sig)
        print("✅ Done. Check the Signals tab.")
    else:
        print("ℹ️ No fields needed backfilling (or no matches). Nothing to write.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
