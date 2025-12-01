#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_signals_from_sheets.py

Bridge script:
    Google Sheets "Signals" tab  -->  ./output/signals_log.csv

We reuse helpers from build_performance_dashboard.py so we respect your
config.yaml (sheet_url, service account, tab names, etc.).

Expected columns in Sheets "Signals" tab (current design):
    - TimestampUTC
    - Ticker
    - Direction  (BUY / SELL)
    - Price
    - (Source, Timeframe, etc. are ignored for this export)

We output:
    ts,ticker,side,price,reason,near_hits,state_before,state_after

which is what weinstein_live_logic_backtest.py::load_signals() expects.
"""

from __future__ import annotations

import argparse
import os
import pandas as pd

from build_performance_dashboard import (
    load_cfg,
    resolve_sheet_url,
    resolve_service_account_file,
    resolve_tab_name,
    auth_gspread,
    open_ws,
    read_tab,
    TAB_SIGNALS,
)


def main():
    ap = argparse.ArgumentParser(description="Export Signals tab from Google Sheets to ./output/signals_log.csv")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    ap.add_argument("--output", type=str, default="./output/signals_log.csv", help="Output CSV path")
    ap.add_argument("--start", type=str, default=None, help="Filter signals from this date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="Filter signals up to this date (YYYY-MM-DD)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    sheet_url = resolve_sheet_url(cfg)
    sa_file = resolve_service_account_file(cfg)
    tab_signals = resolve_tab_name(cfg, "signals_tab", TAB_SIGNALS)

    if not sheet_url:
        raise SystemExit("No sheet_url in config (sheets.url or sheets.sheet_url).")

    print(f"🔑 Authorizing service account with {sa_file} …")
    gc = auth_gspread(sa_file)

    print(f"📄 Reading Signals tab: {tab_signals} from {sheet_url}")
    ws_sig = open_ws(gc, sheet_url, tab_signals)
    df = read_tab(ws_sig)

    if df.empty:
        print("⚠️ Signals tab is empty. Writing empty signals_log.csv with header only.")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        header_df = pd.DataFrame(columns=["ts","ticker","side","price","reason","near_hits","state_before","state_after"])
        header_df.to_csv(args.output, index=False)
        print(f"📝 Wrote: {args.output}")
        return

    # Try to discover key columns
    cols_lower = {c.lower(): c for c in df.columns}

    # Timestamp column (e.g. "TimestampUTC")
    ts_col = None
    for c in df.columns:
        if c.lower().startswith("timestamp"):
            ts_col = c
            break
    if ts_col is None:
        # fallback to first column
        ts_col = df.columns[0]

    # Ticker column
    tkr_col = None
    for name in ("ticker", "symbol"):
        if name in cols_lower:
            tkr_col = cols_lower[name]
            break
    if tkr_col is None:
        raise SystemExit("Could not find Ticker/Symbol column in Signals tab.")

    # Direction column (BUY/SELL)
    side_col = None
    for name in ("direction", "side"):
        if name in cols_lower:
            side_col = cols_lower[name]
            break
    if side_col is None:
        raise SystemExit("Could not find Direction/Side column in Signals tab.")

    # Price column (optional; default empty if missing)
    price_col = None
    for name in ("price",):
        if name in cols_lower:
            price_col = cols_lower[name]
            break

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(df[ts_col], errors="coerce")

    out["ticker"] = (
        df[tkr_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    out["side"] = (
        df[side_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if price_col:
        out["price"] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        out["price"] = ""

    # Optional: reason column if present
    reason_col = None
    for name in ("reason", "note", "notes"):
        if name in cols_lower:
            reason_col = cols_lower[name]
            break
    if reason_col:
        out["reason"] = df[reason_col].astype(str)
    else:
        out["reason"] = ""

    # The backtester expects these extra columns; we can leave them blank
    out["near_hits"] = ""
    out["state_before"] = ""
    out["state_after"] = ""

    # Basic cleaning
    out = out[out["ts"].notna() & out["ticker"].ne("") & out["side"].isin(["BUY","SELL"])]

    # Date-range filter if requested
    if args.start:
        start_dt = pd.to_datetime(args.start + " 00:00:00", utc=True, errors="coerce")
        out = out[out["ts"] >= start_dt]
    if args.end:
        end_dt = pd.to_datetime(args.end + " 23:59:59", utc=True, errors="coerce")
        out = out[out["ts"] <= end_dt]

    if out.empty:
        print("⚠️ After filtering, no signals remain. Writing header-only file.")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        header_df = pd.DataFrame(columns=["ts","ticker","side","price","reason","near_hits","state_before","state_after"])
        header_df.to_csv(args.output, index=False)
        print(f"📝 Wrote: {args.output}")
        return

    # Format ts as ISO for the backtester
    out["ts"] = out["ts"].dt.tz_convert("UTC", nonexistent="NaT", ambiguous="NaT").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Final column order
    out = out[["ts","ticker","side","price","reason","near_hits","state_before","state_after"]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"✅ Exported {len(out)} signals → {args.output}")


if __name__ == "__main__":
    main()
