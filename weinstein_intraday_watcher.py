#!/usr/bin/env python3
"""
weinstein_intraday_watcher.py

Used by the intraday/weekly flows to stitch in “classic” Weinstein scan results.
This version makes the 'stage' column lookup robust to naming changes.

- Looks for output/scan_sp500.csv (if present) and can build signal highlights.
- If the stage column is missing or mismatched, it will **not** crash;
  it simply skips the Stage 1/2 filter and continues.

This is intentionally minimal for weekly combination; your intraday schedule
may supply additional args/behavior.
"""

import os
import sys
import pandas as pd

def pick_col(df: pd.DataFrame, *candidates):
    """
    Return a column name from df matching any candidate, case-insensitive.
    Accepts both exact and lower-case matches.
    """
    cols = list(df.columns)
    lc = {c.lower(): c for c in cols}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lc:
            return lc[name.lower()]
    return None

def load_scan_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def run():
    out_dir = os.getenv("OUTPUT_DIR", "./output")
    csv_path = os.path.join(out_dir, "scan_sp500.csv")
    if not os.path.exists(csv_path):
        print(f"ℹ️  Classic scan CSV not found at {csv_path}. Skipping classic merge.")
        return 0

    weekly_df = load_scan_csv(csv_path)

    # Robust column selection
    col_stage = pick_col(weekly_df, "stage", "Stage", "weinst_stage", "stage_name")
    col_ticker = pick_col(weekly_df, "ticker", "symbol", "Ticker", "Symbol")
    col_signal = pick_col(weekly_df, "Buy Signal", "buy_signal", "signal", "Signal")

    if not col_ticker:
        print("⚠️ classic scan: 'ticker/symbol' column not found; no ticker-based join possible.")
    if not col_signal:
        print("⚠️ classic scan: 'signal' column not found; will keep full table.")

    if not col_stage:
        print("⚠️ classic scan: 'stage' column not found; skipping basing/uptrend filter.")
        filtered = weekly_df
    else:
        ok_values = {"Stage 1 (Basing)", "Stage 2 (Uptrend)"}
        # Some scans may use short tokens like 'S1', 'S2'. Include them if present.
        short_ok = {"S1", "S2"}
        filtered = weekly_df[
            weekly_df[col_stage].astype(str).isin(ok_values.union(short_ok))
        ]

    # Write a friendlier CSV for the mailer combo step
    out_csv = os.path.join(out_dir, "scan_sp500_filtered.csv")
    filtered.to_csv(out_csv, index=False)
    print(f"✅ Classic scan filtered rows: {len(filtered)}/{len(weekly_df)} → {out_csv}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as e:
        print(f"⚠️ Classic scan failed: {e}")
        sys.exit(1)
