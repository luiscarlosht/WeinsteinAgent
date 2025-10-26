#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_fidelity_with_signals.py

Backfills the 'Signals' tab in your Google Sheet using tickers from Holdings and Transactions.
Keeps manually entered fields like 'Source' intact.
Fills missing 'Price' using GOOGLEFINANCE() or yfinance if available.
"""

import argparse
import gspread
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
import yfinance as yf

SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS = "Signals"
TAB_TRANSACTIONS = "Transactions"
TAB_HOLDINGS = "Holdings"

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "

def auth_gspread():
    print("🔑 Authorizing service account…")
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals: return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.update([["(empty)"]], range_name="A1")
        return
    ws.update([list(df.columns)] + df.astype(str).fillna("").values.tolist())

def google_formula(ticker: str):
    return f'=GOOGLEFINANCE("{DEFAULT_EXCHANGE_PREFIX + ticker}","price")'

def fill_price(df, tickers):
    filled = 0
    for t in tickers:
        if df.loc[df["Ticker"] == t, "Price"].isna().all():
            try:
                price = round(yf.Ticker(t).history(period="1d")["Close"].iloc[-1], 2)
                df.loc[df["Ticker"] == t, "Price"] = price
                filled += 1
                print(f"  • {t}: price ← yfinance ${price}")
            except Exception:
                df.loc[df["Ticker"] == t, "Price"] = google_formula(t)
                print(f"  • {t}: price ← GOOGLEFINANCE({t})")
    print(f"✏️ Backfilled {filled} prices via yfinance/GOOGLEFINANCE.")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Skip fuzzy ticker merging")
    args = ap.parse_args()

    gc = auth_gspread()
    sh = gc.open_by_url(SHEET_URL)
    ws_sig = sh.worksheet(TAB_SIGNALS)
    ws_tx = sh.worksheet(TAB_TRANSACTIONS)
    ws_hold = sh.worksheet(TAB_HOLDINGS)

    df_sig = read_tab(ws_sig)
    df_tx = read_tab(ws_tx)
    df_hold = read_tab(ws_hold)

    print(f"📄 Reading tabs…")
    print(f"• Signals rows: {len(df_sig)}")
    print(f"• Transactions rows: {len(df_tx)}")
    print(f"• Holdings rows: {len(df_hold)}")

    sig_tickers = set(df_sig["Ticker"].dropna().unique()) if "Ticker" in df_sig else set()
    tx_tickers = set(df_tx.get("Symbol", []))
    hold_tickers = set(df_hold.get("Symbol", []))
    all_tickers = sorted(set.union(sig_tickers, tx_tickers, hold_tickers))

    print(f"🧩 Merging tickers: found {len(all_tickers)} total unique tickers")

    if df_sig.empty:
        df_sig = pd.DataFrame(columns=["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"])

    existing = df_sig.set_index("Ticker", drop=False)
    new_rows = []
    for t in all_tickers:
        if t not in existing.index:
            new_rows.append({"Ticker": t, "Source": "", "Direction": "BUY", "Price": np.nan, "Timeframe": ""})

    if new_rows:
        df_sig = pd.concat([df_sig, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"➕ Added {len(new_rows)} missing tickers to Signals tab")

    df_sig["Ticker"] = df_sig["Ticker"].astype(str).str.strip().str.upper()
    df_sig["Price"].replace("", np.nan, inplace=True)
    df_sig = fill_price(df_sig, df_sig["Ticker"].unique())

    print(f"✅ Done. Writing back {len(df_sig)} rows to Signals.")
    write_tab(ws_sig, df_sig)
    print("🎯 Merge complete!")

if __name__ == "__main__":
    main()
