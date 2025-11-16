# buffett_options_engine.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os

SAFE_TICKERS = [
    "AAPL", "KO", "JNJ", "PEP", "MCD", "PG", "MSFT", "V", "MA",
    "AMZN", "COST", "UNH", "BRK-B", "HD", "WMT", "DIS", "TMO",
    "LLY", "ADP", "MRK",
]

# How conservative you are vs current price
RISK_BUFFER = 0.10   # 10% below current price

# Minimum annualized yield threshold for 30–45 DTE
MIN_YIELD = 0.008    # 0.8% annualized

def get_option_chain(ticker: str, max_dte: int = 45) -> pd.DataFrame:
    """
    For a single ticker:
      - Pulls all expirations from yfinance
      - Keeps expirations between 7 and max_dte days
      - Filters PUTs with strike <= current_price * (1 - RISK_BUFFER)
      - Computes annualized yield based on bid / current_price

    Returns a DataFrame of candidate puts or empty DataFrame.
    """
    stock = yf.Ticker(ticker)

    # 1) Get expirations
    try:
        expiry_dates = stock.options
    except Exception:
        return pd.DataFrame()

    today = datetime.utcnow().date()
    valid_dates = [
        d
        for d in expiry_dates
        if 7 <= (datetime.strptime(d, "%Y-%m-%d").date() - today).days <= max_dte
    ]
    if not valid_dates:
        return pd.DataFrame()

    # 2) Get current underlying price ONCE (fixes FutureWarning)
    hist = stock.history(period="1d")
    if hist.empty or "Close" not in hist.columns:
        print(f"⚠️ No history for {ticker}, skipping.")
        return pd.DataFrame()

    current_price = hist["Close"].iloc[-1]
    if pd.isna(current_price) or current_price <= 0:
        print(f"⚠️ Invalid current price for {ticker}: {current_price}, skipping.")
        return pd.DataFrame()

    buffer_price = round(current_price * (1 - RISK_BUFFER), 2)

    results = []

    for expiry in valid_dates:
        try:
            opt = stock.option_chain(expiry).puts
        except Exception:
            continue

        if opt is None or opt.empty:
            continue

        # Only consider puts with a bid and a strike below our buffer
        puts = opt.copy()
        puts = puts[puts["bid"].notna() & (puts["bid"] > 0)]
        puts = puts[puts["strike"] <= buffer_price]

        if puts.empty:
            continue

        dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
        if dte <= 0:
            continue

        puts["ticker"] = ticker
        puts["underlying_price"] = float(current_price)
        puts["dte"] = dte
        puts["target_strike"] = buffer_price
        puts["expiry"] = expiry

        # Annualized yield approximation: (bid / underlying) * (365 / dte)
        puts["yield_pct"] = puts["bid"] / current_price / dte * 365.0

        results.append(puts)

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame()

def scan_all() -> pd.DataFrame:
    """
    Scan all SAFE_TICKERS and return a combined, filtered
    DataFrame of attractive put candidates.
    """
    all_results = []

    for tkr in SAFE_TICKERS:
        print(f"Scanning {tkr}...")
        df = get_option_chain(tkr)
        if df.empty:
            continue

        # Filter by minimum yield
        df = df[df["yield_pct"] >= MIN_YIELD]

        if not df.empty:
            all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined.sort_values(["yield_pct"], ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

def save_to_csv(df: pd.DataFrame) -> None:
    """
    Save the candidate puts to ./output/Buffett_Put_Signals_<timestamp>.csv
    """
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
