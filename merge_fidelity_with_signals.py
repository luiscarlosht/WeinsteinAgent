#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_fidelity_with_signals.py

Backfills the 'Signals' tab in your Google Sheet using tickers from Holdings and Transactions.
- Keeps manually entered fields like 'Source' intact (never overwrites).
- Adds any missing tickers from Holdings/Transactions.
- Fills missing 'Price' using yfinance; if unavailable and --no-google NOT set, falls back to GOOGLEFINANCE().
- Optional 'Mapping' tab support: if a row exists with columns [Ticker, FormulaSym, TickerYF],
  we will use FormulaSym in the GOOGLEFINANCE() call and TickerYF for yfinance.

Flags:
  --debug       : verbose logging
  --strict      : skip any fuzzy merging (currently no fuzzy in use; reserved)
  --no-google   : do not fallback to GOOGLEFINANCE formulas (yfinance only)
"""

import argparse
import gspread
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials

# yfinance is optional but recommended
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS       = "Signals"
TAB_TRANSACTIONS  = "Transactions"
TAB_HOLDINGS      = "Holdings"
TAB_MAPPING       = "Mapping"

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "  # used when no Mapping.FormulaSym found

# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def auth_gspread():
    print("🔑 Authorizing service account…")
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.update([["(empty)"]], range_name="A1")
        return
    ws.update([list(df.columns)] + df.astype(str).fillna("").values.tolist())

def open_ws(gc, title: str):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=2000, cols=26)

def read_mapping(gc) -> dict:
    """Return { Ticker: {'FormulaSym': 'EXCH: TKR', 'TickerYF': 'TKR'} } if Mapping tab exists."""
    out = {}
    try:
        ws = open_ws(gc, TAB_MAPPING)
        df = read_tab(ws)
        if not df.empty and "Ticker" in df.columns:
            for _, row in df.iterrows():
                t = str(row.get("Ticker", "")).strip().upper()
                if not t:
                    continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym", "")).strip(),
                    "TickerYF": str(row.get("TickerYF", "")).strip().upper(),
                }
    except Exception:
        pass
    return out

def google_formula_for(ticker: str, mapping: dict) -> str:
    base = (ticker or "").strip().upper()
    mm = mapping.get(base, {})
    sym = mm.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), "")'

def yf_price_for(ticker: str, mapping: dict):
    if not YF_AVAILABLE:
        return None
    base = (ticker or "").strip().upper()
    mm = mapping.get(base, {})
    yf_tkr = mm.get("TickerYF") or base
    try:
        hist = yf.Ticker(yf_tkr).history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Merge Holdings/Transactions tickers into Signals; fill missing prices.")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    ap.add_argument("--strict", action="store_true", help="Reserved for future strictness (no fuzzy use currently)")
    ap.add_argument("--no-google", action="store_true", help="Do not fallback to GOOGLEFINANCE formula")
    args = ap.parse_args()
    DEBUG = args.debug

    gc = auth_gspread()
    sh = gc.open_by_url(SHEET_URL)

    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_tx  = open_ws(gc, TAB_TRANSACTIONS)
    ws_h   = open_ws(gc, TAB_HOLDINGS)

    print("📄 Reading tabs…")
    df_sig = read_tab(ws_sig)
    df_tx  = read_tab(ws_tx)
    df_h   = read_tab(ws_h)

    print(f"• Signals rows: {len(df_sig)}")
    print(f"• Transactions rows: {len(df_tx)}")
    print(f"• Holdings rows: {len(df_h)}")
    print(f"• yfinance available: {YF_AVAILABLE}")

    # Ensure Signals has the expected columns
    if df_sig.empty:
        df_sig = pd.DataFrame(columns=["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"])
    for col in ["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"]:
        if col not in df_sig.columns:
            df_sig[col] = ""

    # Normalize Signal tickers
    df_sig["Ticker"] = df_sig["Ticker"].astype(str).str.strip().str.upper()

    # Collect unique tickers from Signals, Holdings, Transactions
    sig_tickers   = set(df_sig["Ticker"].dropna().unique()) if "Ticker" in df_sig else set()
    hold_tickers  = set(df_h.get("Symbol", [])) if "Symbol" in df_h else set()
    tx_tickers    = set(df_tx.get("Symbol", [])) if "Symbol" in df_tx else set()

    all_tickers = sorted(set.union(sig_tickers, hold_tickers, tx_tickers))
    print(f"🧩 Merging tickers: found {len(all_tickers)} total unique tickers")

    # Add missing tickers to Signals (non-destructive to any existing rows)
    existing = set(df_sig["Ticker"])
    new_rows = []
    for t in all_tickers:
        t_up = (t or "").strip().upper()
        if t_up and t_up not in existing:
            # Default to "BUY" so the dashboard logic can associate
            new_rows.append({
                "TimestampUTC": "",
                "Ticker": t_up,
                "Source": "",
                "Direction": "BUY",
                "Price": "",
                "Timeframe": "",
            })
    if new_rows:
        df_sig = pd.concat([df_sig, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"➕ Added {len(new_rows)} missing tickers to Signals tab")

    # Mapping support for better GOOGLEFINANCE and yfinance tickers
    mapping = read_mapping(gc)
    if DEBUG:
        print(f"• Mapping rows: {len(mapping)}")

    # Fill missing prices:
    # 1) Try yfinance
    # 2) If still empty and --no-google is NOT set, set GOOGLEFINANCE() formula
    # Never overwrite a non-empty Price the user already filled.
    df_sig["Price"] = df_sig["Price"].replace("", np.nan)

    filled_yf = 0
    filled_gf = 0
    for t in df_sig["Ticker"].dropna().unique():
        mask = (df_sig["Ticker"] == t) & (df_sig["Price"].isna())
        if not mask.any():
            continue

        # yfinance first
        p = yf_price_for(t, mapping)
        if p is not None:
            df_sig.loc[mask, "Price"] = round(p, 2)
            filled_yf += int(mask.sum())
            if DEBUG:
                print(f"  • {t}: price ← yfinance ${round(p,2)}")
            continue

        # fallback to GOOGLEFINANCE unless --no-google
        if not args.no_google:
            formula = google_formula_for(t, mapping)
            df_sig.loc[mask, "Price"] = formula
            filled_gf += int(mask.sum())
            if DEBUG:
                print(f"  • {t}: price ← {formula}")

    print(f"✏️ Backfilled {filled_yf} price cells via yfinance and {filled_gf} via GOOGLEFINANCE.")

    # Tidy column order and write back
    keep_cols = ["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"]
    df_sig = df_sig[keep_cols]

    print(f"✅ Done. Writing back {len(df_sig)} rows to Signals.")
    write_tab(ws_sig, df_sig)
    print("🎯 Merge complete!")

if __name__ == "__main__":
    main()
