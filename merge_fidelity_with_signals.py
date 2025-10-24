#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_fidelity_with_signals.py  v2.1

Backfills empty fields in the Google Sheet "Signals" tab:
- TimestampUTC, Direction, Price, Timeframe

Sources:
- Transactions (latest BUY/SELL price & date)
- Holdings ("Last Price ($)" or "Price ($)")
- Mapping (per-ticker or per-source default timeframe)

Only blank cells are filled. Existing values are preserved.

Usage:
  python3 merge_fidelity_with_signals.py [--verbose]
"""

from __future__ import annotations

import re
import sys
import argparse
import datetime as dt
from typing import Dict, Optional, Tuple

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ── CONFIG ───────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS      = "Signals"
TAB_TRANSACTIONS = "Transactions"
TAB_HOLDINGS     = "Holdings"
TAB_MAPPING      = "Mapping"

SOURCE_DEFAULT_TIMEFRAME = {
    "sarkee capital": "short",
    "superiorstar":   "mid",
    "weinstein":      "long",
    "bo xu":          "short",
    "bo":             "short",
}

COL_SIG_TIMESTAMP = "TimestampUTC"
COL_SIG_TICKER    = "Ticker"
COL_SIG_SOURCE    = "Source"
COL_SIG_DIR       = "Direction"
COL_SIG_PRICE     = "Price"
COL_SIG_TF        = "Timeframe"

# ── helpers ──────────────────────────────────────────────────────────────────
def auth() -> gspread.Client:
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc: gspread.Client, title: str) -> gspread.Worksheet:
    sh = gc.open_by_url(SHEET_URL)
    return sh.worksheet(title)

def ws_to_df(ws: gspread.Worksheet) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    df = pd.DataFrame(vals[1:], columns=vals[0])
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def a1_update_table(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    ws.clear()
    if df.empty:
        return
    ws.update([df.columns.tolist()] + df.values.tolist())

def norm_cols(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}

def clean_ticker(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    word = raw.strip().split()[0]
    letters = re.sub(r"[^A-Za-z]", "", word)
    return letters.upper()

def parse_float(val) -> Optional[float]:
    s = str(val).replace(",", "").strip()
    if not s or s.lower() in {"nan", "none", "--"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def latest_trade_for(tdf: pd.DataFrame, ticker: str) -> Optional[Tuple[str, str, float]]:
    if tdf.empty or not ticker:
        return None
    cols = norm_cols(tdf)
    c_run   = cols.get("run date") or cols.get("date") or cols.get("trade date")
    c_sym   = cols.get("symbol")
    c_type  = cols.get("type") or cols.get("action")
    c_price = cols.get("price ($)") or cols.get("price")
    if not (c_run and c_sym and c_type and c_price):
        return None

    work = tdf[[c_run, c_sym, c_type, c_price]].copy()
    work[c_sym] = work[c_sym].map(clean_ticker)
    work = work[work[c_sym] == ticker]
    work = work[work[c_type].str.contains("BUY|SELL", case=False, na=False)]
    if work.empty:
        return None

    ts = pd.to_datetime(work[c_run], errors="coerce", utc=True)
    work = work.assign(_ts=ts).sort_values("_ts")
    last = work.iloc[-1]
    direction = "BUY" if "buy" in str(last[c_type]).lower() else "SELL"
    price = parse_float(last[c_price])

    if pd.notna(last["_ts"]):
        iso = last["_ts"].strftime("%Y-%m-%d %H:%M:%S%z")
    else:
        iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    return (iso, direction, price if price is not None else float("nan"))

def holdings_price(hdf: pd.DataFrame, ticker: str) -> Optional[float]:
    if hdf.empty or not ticker:
        return None
    cols = norm_cols(hdf)
    c_sym = cols.get("symbol") or cols.get("ticker")
    c_lp  = cols.get("last price ($)") or cols.get("price ($)") or cols.get("last price")
    if not (c_sym and c_lp):
        return None
    work = hdf[[c_sym, c_lp]].copy()
    work[c_sym] = work[c_sym].map(clean_ticker)
    row = work[work[c_sym] == ticker]
    if row.empty:
        return None
    return parse_float(row.iloc[-1][c_lp])

def timeframe_from_mapping(mdf: pd.DataFrame, ticker: str, source: str) -> Optional[str]:
    if mdf.empty:
        return SOURCE_DEFAULT_TIMEFRAME.get(source.lower()) if source else None

    cols = norm_cols(mdf)
    c_tick = cols.get("ticker")
    c_src  = cols.get("source")
    c_tf   = cols.get("timeframe")

    if c_tick and c_tf:
        m = mdf[mdf[c_tick].map(clean_ticker) == ticker]
        if not m.empty and str(m.iloc[-1][c_tf]).strip():
            return str(m.iloc[-1][c_tf]).strip()

    if c_src and c_tf and source:
        m = mdf[(mdf[c_src].str.lower() == source.lower()) &
                ((mdf[c_tick] == "") | (mdf[c_tick].isna()) if c_tick else True)]
        if not m.empty and str(m.iloc[-1][c_tf]).strip():
            return str(m.iloc[-1][c_tf]).strip()

    return SOURCE_DEFAULT_TIMEFRAME.get(source.lower()) if source else None

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    v = args.verbose

    print("📄 Reading tabs…")
    gc = auth()
    ws_sig  = open_ws(gc, TAB_SIGNALS)
    ws_txn  = open_ws(gc, TAB_TRANSACTIONS)
    ws_hold = open_ws(gc, TAB_HOLDINGS)
    ws_map  = open_ws(gc, TAB_MAPPING)

    df_sig  = ws_to_df(ws_sig)
    df_txn  = ws_to_df(ws_txn)
    df_hold = ws_to_df(ws_hold)
    df_map  = ws_to_df(ws_map)

    print(f"• Signals rows: {len(df_sig)}")
    print(f"• Transactions rows: {len(df_txn)}")
    print(f"• Holdings rows: {len(df_hold)}")

    if df_sig.empty:
        print("⚠️ Signals tab is empty — nothing to do.")
        return

    # ensure required columns exist
    for c in [COL_SIG_TIMESTAMP, COL_SIG_TICKER, COL_SIG_SOURCE, COL_SIG_DIR, COL_SIG_PRICE, COL_SIG_TF]:
        if c not in df_sig.columns:
            df_sig[c] = ""

    df_sig["_ticker"] = df_sig[COL_SIG_TICKER].map(clean_ticker)
    df_sig["_source"] = df_sig[COL_SIG_SOURCE].astype(str).str.strip()

    filled = 0
    for idx, row in df_sig.iterrows():
        tkr = row["_ticker"]
        src = row["_source"]
        if not tkr:
            if v: print(f"  • Row {idx+2}: empty ticker → skip")
            continue

        # Current values
        ts   = str(row[COL_SIG_TIMESTAMP]).strip()
        dr   = str(row[COL_SIG_DIR]).strip()
        pr_s = str(row[COL_SIG_PRICE]).strip()
        tf   = str(row[COL_SIG_TF]).strip()

        # PRICE decision
        price_val = parse_float(pr_s)
        if price_val is None:
            txn = latest_trade_for(df_txn, tkr)
            used = None
            if txn and txn[2] is not None and not pd.isna(txn[2]) and txn[2] > 0:
                price_val = txn[2]; used = "transactions"
            else:
                hp = holdings_price(df_hold, tkr)
                if hp is not None and not pd.isna(hp) and hp > 0:
                    price_val = hp; used = "holdings"

            if price_val is not None:
                df_sig.at[idx, COL_SIG_PRICE] = f"{price_val:.4f}".rstrip("0").rstrip(".")
                filled += 1
                if v: print(f"  • {tkr}: price ← {used} ({price_val})")
            elif v:
                print(f"  • {tkr}: no price found in transactions or holdings")

        # TIMESTAMP
        if not ts:
            if 'txn' not in locals():
                txn = latest_trade_for(df_txn, tkr)
            if txn:
                df_sig.at[idx, COL_SIG_TIMESTAMP] = txn[0]; filled += 1
                if v: print(f"  • {tkr}: timestamp ← transactions ({txn[0]})")
            else:
                now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
                df_sig.at[idx, COL_SIG_TIMESTAMP] = now_iso; filled += 1
                if v: print(f"  • {tkr}: timestamp ← now ({now_iso})")

        # DIRECTION
        if not dr:
            if 'txn' not in locals():
                txn = latest_trade_for(df_txn, tkr)
            if txn:
                df_sig.at[idx, COL_SIG_DIR] = txn[1]; filled += 1
                if v: print(f"  • {tkr}: direction ← transactions ({txn[1]})")
            else:
                df_sig.at[idx, COL_SIG_DIR] = "BUY"; filled += 1
                if v: print(f"  • {tkr}: direction ← default BUY")

        # TIMEFRAME
        if not tf:
            tf_val = timeframe_from_mapping(df_map, tkr, src)
            if tf_val:
                df_sig.at[idx, COL_SIG_TF] = tf_val; filled += 1
                if v: print(f"  • {tkr}: timeframe ← mapping/default ({tf_val})")

        # clean local var for next loop
        if 'txn' in locals():
            del txn

    if filled == 0:
        print("ℹ️ No fields needed backfilling (or no matching data in transactions/holdings). Nothing to write.")
        return

    # drop helper cols & write back
    df_sig = df_sig.drop(columns=["_ticker", "_source"])
    print(f"✏️ Backfilled {filled} empty fields in Signals. Writing back…")
    a1_update_table(ws_sig, df_sig)
    print("✅ Done. Check the Signals tab.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
