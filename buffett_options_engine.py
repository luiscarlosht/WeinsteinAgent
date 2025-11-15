# buffett_options_engine.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os

SAFE_TICKERS = [
    'AAPL', 'KO', 'JNJ', 'PEP', 'MCD', 'PG', 'MSFT', 'V', 'MA',
    'AMZN', 'COST', 'UNH', 'BRK-B', 'HD', 'WMT', 'DIS', 'TMO',
    'LLY', 'ADP', 'MRK'
]

RISK_BUFFER = 0.10  # 10% below current price
MIN_YIELD = 0.008  # 0.8% premium minimum for 30-45 DTE

def get_option_chain(ticker, max_dte=45):
    stock = yf.Ticker(ticker)
    try:
        expiry_dates = stock.options
    except Exception:
        return pd.DataFrame()
    
    today = datetime.utcnow().date()
    valid_dates = [d for d in expiry_dates if 7 <= (datetime.strptime(d, "%Y-%m-%d").date() - today).days <= max_dte]

    results = []

    for expiry in valid_dates:
        try:
            opt = stock.option_chain(expiry).puts
        except:
            continue

        current_price = stock.history(period="1d")['Close'][-1]
        buffer_price = round(current_price * (1 - RISK_BUFFER), 2)

        candidates = opt[opt['strike'] <= buffer_price].copy()
        if candidates.empty:
            continue

        dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
        candidates['ticker'] = ticker
        candidates['underlying_price'] = current_price
        candidates['dte'] = dte
        candidates['target_strike'] = buffer_price
        candidates['yield_pct'] = candidates['bid'] / current_price / dte * 365
        candidates['expiry'] = expiry

        results.append(candidates)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def scan_all():
    all_results = []
    for tkr in SAFE_TICKERS:
        print(f"Scanning {tkr}...")
        df = get_option_chain(tkr)
        if not df.empty:
            df = df[df['yield_pct'] >= MIN_YIELD]
            if not df.empty:
                all_results.append(df)
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.sort_values(['yield_pct'], ascending=False, inplace=True)
        return combined
    return pd.DataFrame()

def save_to_csv(df: pd.DataFrame):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    fpath = f"./output/Buffett_Put_Signals_{now}.csv"
    df.to_csv(fpath, index=False)
    print(f"✅ Saved: {fpath}")

if __name__ == "__main__":
    print("🚀 Running Buffett Options Engine…")
    df = scan_all()
    if df.empty:
        print("⚠️ No suitable options found.")
    else:
        print(f"✅ {len(df)} options found.")
        save_to_csv(df)
