#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Intraday Diagnostics (merged)
- Safe to re-run any time.
- Extracts "Near" tickers from intraday_watch_*.html into near_universe.txt
- Reads ./output/intraday_debug.csv (if present), even if previously empty
- Computes:
  * BUY near-misses (all gates green except price)
  * SELL near candidates
  * Carryover watchlist (near to pivot repeatedly)
- Writes summary text + CSVs to --outdir
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
from typing import List, Set, Tuple, Optional

import pandas as pd


NEAR_REGEX = re.compile(r"<ol><li><b>\d+\.</b>\s*<b>([A-Z]+)</b>\s*@")

def ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path) or os.path.getsize(path) < 5:
        return None
    try:
        df = pd.read_csv(path)
        # Guard for missing core columns
        core = {"ticker", "price", "pivot"}
        if not core.issubset(set(df.columns)):
            return None
        # Drop rows where these are NaN (common when file was 1-byte previously)
        df = df.dropna(subset=["ticker", "price", "pivot"])
        return df
    except Exception:
        return None


def extract_near_from_html(html_glob: str) -> Tuple[Set[str], List[str]]:
    files = sorted(glob.glob(html_glob))
    seen: Set[str] = set()
    scanned_files: List[str] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            tickers = set(NEAR_REGEX.findall(text))
            if tickers:
                scanned_files.append(f)
                seen |= tickers
        except Exception:
            pass
    return seen, scanned_files


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # elapsed_min exists in your CSV; keep a ts_min so we can aggregate last_seen
    if "elapsed_min" in df.columns:
        df["ts_min"] = df["elapsed_min"]
    else:
        df["ts_min"] = 0

    # distance to pivot in basis points (positive => price below pivot)
    df["dist_bps"] = (df["pivot"] - df["price"]) / df["pivot"] * 1e4

    # Make sure all the boolean flags we reference exist; if not, create as False
    needed_flags = [
        "cond_weekly_stage_ok",
        "cond_rs_ok",
        "cond_ma_ok",
        "cond_pivot_ok",
        "cond_buy_price_ok",
        "cond_buy_vol_ok",
        "cond_buy_confirm",
        "cond_pace_full_gate",
        "cond_near_pace_gate",
        "cond_near_now",
        "cond_sell_near_now",
        "cond_sell_price_ok",
        "cond_sell_vol_ok",
        "cond_sell_confirm",
    ]
    for c in needed_flags:
        if c not in df.columns:
            df[c] = False

    # Normalize bool-ish columns
    for c in needed_flags:
        if df[c].dtype != bool:
            df[c] = df[c].astype(bool)

    return df


def compute_buy_near_misses(df: pd.DataFrame) -> pd.DataFrame:
    """All green except price → 'near-miss' BUY"""
    buy_gates_except_price = (
        df["cond_weekly_stage_ok"]
        & df["cond_rs_ok"]
        & df["cond_ma_ok"]
        & df["cond_pivot_ok"]
        & df["cond_buy_vol_ok"]
        & df["cond_pace_full_gate"]
        & df["cond_near_pace_gate"]
        & (~df["cond_buy_price_ok"])
    )
    near = df[buy_gates_except_price].copy()
    if near.empty:
        return near

    agg = (
        near.groupby("ticker")
        .agg(
            scans=("ticker", "count"),
            min_dist_bps=("dist_bps", "min"),
            any_pace=("cond_near_pace_gate", "max"),
            weekly_ok=("cond_weekly_stage_ok", "max"),
            last_seen=("ts_min", "max"),
        )
        .sort_values(["min_dist_bps", "scans"], ascending=[True, False])
        .reset_index()
    )
    return agg


def compute_sell_near(df: pd.DataFrame) -> pd.DataFrame:
    """Tickers that were near SELL conditions."""
    sell_near = df[df["cond_sell_near_now"]].copy()
    if sell_near.empty:
        return sell_near
    agg = (
        sell_near.groupby("ticker")
        .agg(
            scans=("ticker", "count"),
            last_seen=("ts_min", "max"),
            price_ok=("cond_sell_price_ok", "max"),
            vol_ok=("cond_sell_vol_ok", "max"),
            confirmed=("cond_sell_confirm", "max"),
        )
        .sort_values(["confirmed", "scans", "last_seen"], ascending=[False, False, False])
        .reset_index()
    )
    return agg


def compute_carryover_watchlist(buy_near_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Build a near-term watchlist for next run:
    - weekly_ok True
    - min_dist_bps between 0 and 50 (within 0-0.5% of pivot)
    - at least 2 scans
    """
    if buy_near_agg is None or buy_near_agg.empty:
        return pd.DataFrame(columns=["ticker", "scans", "min_dist_bps", "last_seen"])
    mask = (
        (buy_near_agg["weekly_ok"] == True)
        & (buy_near_agg["min_dist_bps"] >= 0)
        & (buy_near_agg["min_dist_bps"] <= 50)
        & (buy_near_agg["scans"] >= 2)
    )
    watch = (
        buy_near_agg.loc[mask, ["ticker", "scans", "min_dist_bps", "last_seen"]]
        .sort_values(["min_dist_bps", "scans"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return watch


def write_list(path: str, items: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in items:
            f.write(f"{s}\n")


def main():
    ap = argparse.ArgumentParser(description="Weinstein intraday diagnostics")
    ap.add_argument("--csv", default="./output/intraday_debug.csv", help="Path to intraday_debug.csv")
    ap.add_argument("--outdir", default="./output", help="Where to write outputs")
    ap.add_argument("--html-glob", default="./output/intraday_watch_*.html", help="Glob for intraday HTML files")
    ap.add_argument("--state", default="./output/diag_state.json", help="Optional state file to store carryover")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = ts()

    # 1) Extract near-universe from HTML
    near_tickers, html_used = extract_near_from_html(args.html_glob)
    near_universe_path = os.path.join(args.outdir, "near_universe.txt")
    write_list(near_universe_path, sorted(list(near_tickers)))

    # 2) Read intraday_debug.csv
    df = safe_read_csv(args.csv)
    have_csv = df is not None
    if have_csv:
        df = add_derived_columns(df)

    # 3) Compute BUY near-misses + SELL near
    buy_near_agg = compute_buy_near_misses(df) if have_csv else pd.DataFrame()
    sell_near_agg = compute_sell_near(df) if have_csv else pd.DataFrame()

    # 4) Carryover watchlist from BUY near-misses
    carryover = compute_carryover_watchlist(buy_near_agg) if have_csv else pd.DataFrame()

    # 5) Save CSV artifacts
    saved = []
    if not buy_near_agg.empty:
        p = os.path.join(args.outdir, f"diag_buy_near_agg_{stamp}.csv")
        buy_near_agg.to_csv(p, index=False)
        saved.append(p)
    if not sell_near_agg.empty:
        p = os.path.join(args.outdir, f"diag_sell_near_agg_{stamp}.csv")
        sell_near_agg.to_csv(p, index=False)
        saved.append(p)
    if not carryover.empty:
        p = os.path.join(args.outdir, f"diag_carryover_watchlist_{stamp}.csv")
        carryover.to_csv(p, index=False)
        saved.append(p)
        # also a plain .txt for quick consumption by other scripts
        p2 = os.path.join(args.outdir, "carryover_buy_watchlist.txt")
        write_list(p2, carryover["ticker"].tolist())
        saved.append(p2)

    # 6) Optionally persist tiny state (last carryover set)
    if not carryover.empty:
        state = {
            "timestamp": stamp,
            "carryover_tickers": carryover["ticker"].tolist(),
            "counts": {
                "buy_near_rows": int(buy_near_agg["scans"].sum()) if not buy_near_agg.empty else 0,
                "sell_near_rows": int(sell_near_agg["scans"].sum()) if not sell_near_agg.empty else 0,
            },
        }
        try:
            with open(args.state, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception:
            pass

    # 7) Write human summary
    summary_path = os.path.join(args.outdir, f"diag_summary_{stamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"=== Weinstein Intraday Diagnostics @ {stamp} ===\n")
        if have_csv:
            total_rows = len(df)
            fh.write(f"intraday_debug.csv rows: {total_rows}\n")
        else:
            fh.write("No valid intraday_debug.csv found or file empty.\n")

        fh.write(f"Near-universe tickers discovered in HTML: {len(near_tickers)} (saved: near_universe.txt)\n")
        if html_used:
            fh.write(f"HTML files scanned: {len(html_used)}\n")

        if have_csv:
            fh.write("\n-- BUY Near-Miss Summary --\n")
            if buy_near_agg.empty:
                fh.write("None found.\n")
            else:
                fh.write(f"Tickers with near-miss BUYs: {len(buy_near_agg)}\n")
                fh.write(buy_near_agg.head(20).to_string(index=False))
                fh.write("\n")

            fh.write("\n-- SELL Near Summary --\n")
            if sell_near_agg.empty:
                fh.write("None found.\n")
            else:
                fh.write(f"Tickers near SELL: {len(sell_near_agg)}\n")
                fh.write(sell_near_agg.head(20).to_string(index=False))
                fh.write("\n")

            fh.write("\n-- Carryover Watchlist (BUY) --\n")
            if carryover.empty:
                fh.write("None.\n")
            else:
                fh.write(f"{len(carryover)} tickers carried over (also saved to carryover_buy_watchlist.txt)\n")
                fh.write(carryover.to_string(index=False))
                fh.write("\n")

    # 8) Final console output
    print("Diagnostics complete.")
    print(f"- Near-universe: {near_universe_path}")
    print(f"- Summary:      {summary_path}")
    for p in saved:
        print(f"- Saved:        {p}")


if __name__ == "__main__":
    main()
