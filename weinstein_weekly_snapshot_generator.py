#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_weekly_snapshot_generator.py

Builds a *historical* series of weekly Weinstein equity snapshots that your
backtest can use as "as-of" universes instead of a single, current CSV.

Output:
- One CSV per snapshot under ./data/weekly_snapshots/, named like:
    data/weekly_snapshots/weinstein_weekly_equities_YYYYMMDD.csv

Each CSV contains (at least):
    ticker, stage, rs_above_ma, ma30, rank

These are enough for weinstein_live_logic_backtest_yfinance.py, which will
pick the latest snapshot with as_of_date <= the current backtest day.

Universe:
- By default, we reuse the tickers from your latest weekly CSV:
    ./output/weinstein_weekly_equities_*.csv
- Optionally, you can provide a custom universe CSV with a 'ticker' column.

Stage logic:
- This is a *reasonable* Weinstein-style approximation for snapshot purposes:
    * 30-week moving average of weekly closes
    * "Stage 2 (Uptrend)" if:
        - price > ma30
        - ma30 rising vs 10 weeks ago
    * "Stage 4 (Downtrend)" if:
        - price < ma30
        - ma30 falling vs 10 weeks ago
    * Other cases get Stage 1/3 placeholders.
- rs_above_ma:
    * Relative strength vs benchmark (default SPY)
    * Ratio = price / benchmark_price
    * rs_above_ma = ratio > 30-week MA(ratio) at the as-of bar

This keeps your snapshot generator completely independent of the rest of the
codebase and avoids breaking existing scripts.
"""

import argparse
import os
import math
import re
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- Logging helpers ----------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, level: str = "info") -> None:
    prefix = {
        "info": "•",
        "ok": "✅",
        "step": "▶️",
        "warn": "⚠️",
        "err": "❌",
        "debug": "··",
    }.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)


# ---------- Paths / constants ----------

OUTPUT_DIR = "./data/weekly_snapshots"
LATEST_WEEKLY_DIR = "./output"
LATEST_WEEKLY_PREFIX = "weinstein_weekly_equities_"

DEFAULT_BENCHMARK = "SPY"

# We’ll use Friday weekly cadence by default.
DEFAULT_FREQ = "W-FRI"

# ---------- Helpers to find the latest weekly CSV (for universe) ----------

def _newest_weekly_csv() -> str:
    files = [
        f
        for f in os.listdir(LATEST_WEEKLY_DIR)
        if re.fullmatch(rf"{re.escape(LATEST_WEEKLY_PREFIX)}\d{{8}}(?:_\d{{4}})?\.csv", f)
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found under {LATEST_WEEKLY_DIR} with prefix "
            f"{LATEST_WEEKLY_PREFIX}. Run run_weekly.sh at least once first."
        )
    files.sort(reverse=True)
    return os.path.join(LATEST_WEEKLY_DIR, files[0])


def load_universe_from_latest_weekly() -> List[str]:
    path = _newest_weekly_csv()
    log(f"Using latest weekly CSV for universe: {path}", level="info")
    df = pd.read_csv(path)
    if "ticker" not in df.columns and "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    if "ticker" not in df.columns:
        raise RuntimeError(f"File {path} does not have a 'ticker' column.")
    tickers = sorted(set(str(t).upper() for t in df["ticker"] if str(t).strip()))
    log(f"Universe size from latest weekly: {len(tickers)} tickers.", level="info")
    return tickers


def load_universe_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns and "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    if "ticker" not in df.columns:
        raise RuntimeError(f"Universe file {path} must have a 'ticker' column.")
    tickers = sorted(set(str(t).upper() for t in df["ticker"] if str(t).strip()))
    log(f"Universe size from {path}: {len(tickers)} tickers.", level="info")
    return tickers


# ---------- Data download helpers ----------

def download_weekly_bars(
    tickers: List[str],
    benchmark: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download WEEKLY OHLCV for tickers + benchmark using yfinance.

    - Uses some padding before start to give enough history for 30-week MA.
    - Returns a MultiIndex columns DataFrame:
        index: Timestamp (week-ending)
        columns: ('Open', symbol), ('High', symbol) ...
    """
    # Pad ~80 weeks back for 30-week MAs plus some burn-in
    start_dt = datetime.fromisoformat(start)
    pad_start = (start_dt - pd.Timedelta(weeks=80)).strftime("%Y-%m-%d")
    all_symbols = sorted(set(tickers + [benchmark]))

    log(
        f"Downloading WEEKLY bars for {len(all_symbols)} symbols "
        f"({pad_start} → {end})...",
        level="step",
    )
    data = yf.download(
        all_symbols,
        start=pad_start,
        end=end,
        interval="1wk",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("No weekly data returned from yfinance.")
    log("Weekly download complete.", level="ok")
    return data


# ---------- Indicator helpers (weekly) ----------

def get_close_series(weekly_df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Returns the weekly Close series for a given symbol from a MultiIndex DataFrame.
    """
    if not isinstance(weekly_df.columns, pd.MultiIndex):
        # Single symbol fallback
        if "Close" in weekly_df.columns:
            s = weekly_df["Close"].dropna()
        else:
            s = pd.Series(dtype=float)
        return s

    try:
        s = weekly_df[("Close", symbol)].dropna()
    except KeyError:
        s = pd.Series(dtype=float)
    return s


def compute_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def classify_stage(
    price: float,
    ma30: float,
    ma30_lag: float,
) -> str:
    """
    Simple Weinstein-style stage classification:

    Stage 2 (Uptrend): price > ma30 and ma30 rising vs 10 weeks ago
    Stage 4 (Downtrend): price < ma30 and ma30 falling vs 10 weeks ago
    Others: generic Stage 1/3 placeholders, which the backtest mostly ignores.
    """
    if np.isnan(price) or np.isnan(ma30) or np.isnan(ma30_lag):
        return "Stage ?"

    rising = ma30 > ma30_lag
    falling = ma30 < ma30_lag

    if price > ma30 and rising:
        return "Stage 2 (Uptrend)"
    if price < ma30 and falling:
        return "Stage 4 (Downtrend)"

    # Simple fallback classification
    if price > ma30 and not rising:
        return "Stage 3 (Topping)"
    if price < ma30 and not falling:
        return "Stage 1 (Basing)"

    return "Stage ?"


def compute_rs_above_ma(
    ticker_close: pd.Series,
    bench_close: pd.Series,
) -> Tuple[bool, float]:
    """
    Compute "relative strength vs benchmark above its 30-week MA?"

    Returns (rs_above_ma: bool, rs_ratio: float) at the last aligned bar.
    """
    df = pd.concat(
        [ticker_close.rename("ticker"), bench_close.rename("bench")],
        axis=1,
    ).dropna()
    if df.empty:
        return False, np.nan

    df["ratio"] = df["ticker"] / df["bench"]
    df["ratio_ma"] = df["ratio"].rolling(30).mean()

    last = df.iloc[-1]
    ratio = float(last["ratio"])
    ratio_ma = float(last["ratio_ma"]) if not math.isnan(last["ratio_ma"]) else np.nan

    if math.isnan(ratio_ma):
        return False, ratio

    return bool(ratio > ratio_ma), ratio


# ---------- Snapshot builder ----------

def build_snapshot_for_date(
    weekly_df: pd.DataFrame,
    tickers: List[str],
    benchmark: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a single weekly snapshot as of a given date, for all tickers.
    """

    # Benchmark close series once
    bench_close = get_close_series(weekly_df, benchmark)
    bench_close = bench_close[bench_close.index <= as_of]
    if bench_close.empty:
        log(f"[{as_of.date()}] Benchmark {benchmark} has no data yet; snapshot empty.", level="warn")
        return pd.DataFrame(columns=["ticker", "stage", "rs_above_ma", "ma30", "rank"])

    rows = []
    rs_values: Dict[str, float] = {}

    for t in tickers:
        # Weekly closes up to as_of
        c = get_close_series(weekly_df, t)
        c = c[c.index <= as_of]
        if len(c) < 40:  # need some history for 30-week MA + lag
            continue

        price = float(c.iloc[-1])

        # 30-week MA and a 10-week lag to detect slope
        ma30 = compute_ma(c, 30)
        ma30_valid = ma30.dropna()
        if ma30_valid.empty:
            continue
        ma30_last = float(ma30_valid.iloc[-1])

        # 10-week lag (approx slope)
        ma30_lag = compute_ma(c, 30).shift(10).dropna()
        if ma30_lag.empty:
            continue
        ma30_lag_last = float(ma30_lag.iloc[-1])

        stage = classify_stage(price, ma30_last, ma30_lag_last)

        # RS vs benchmark
        rs_above, rs_ratio = compute_rs_above_ma(c, bench_close)
        rs_values[t] = rs_ratio

        rows.append(
            {
                "ticker": t,
                "stage": stage,
                "rs_above_ma": bool(rs_above),
                "ma30": ma30_last,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ticker", "stage", "rs_above_ma", "ma30", "rank"])

    df = pd.DataFrame(rows)

    # Rank: higher RS ratio gets better rank (1 = strongest)
    # Missing RS treated as worst.
    rs_series = pd.Series(rs_values)
    df["__rs"] = df["ticker"].map(rs_series).astype(float)
    df["__rs"] = df["__rs"].fillna(-1e9)
    df = df.sort_values("__rs", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df.drop(columns=["__rs"], inplace=True)

    # Add a human-readable as-of date column for reference
    df["as_of"] = as_of.date().isoformat()

    return df


def generate_snapshots(
    start: str,
    end: str,
    universe: List[str],
    benchmark: str,
    freq: str = DEFAULT_FREQ,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Build weekly grid (e.g., Fridays)
    week_dates = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    if len(week_dates) == 0:
        raise RuntimeError("No weekly dates generated; check your start/end range.")

    log(
        f"Generating snapshots from {start_ts.date()} to {end_ts.date()} "
        f"on {freq} ({len(week_dates)} snapshot dates).",
        level="info",
    )

    # Download weekly data once for all symbols
    weekly_data = download_weekly_bars(universe, benchmark, start, end)

    # Main loop
    for as_of in week_dates:
        # For each as_of date, we use all weekly bars up to that date.
        df_snap = build_snapshot_for_date(weekly_data, universe, benchmark, as_of)
        if df_snap.empty:
            log(f"[{as_of.date()}] No valid symbols for snapshot; skipping CSV.", level="warn")
            continue

        out_name = f"weinstein_weekly_equities_{as_of.strftime('%Y%m%d')}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df_snap.to_csv(out_path, index=False)
        log(f"[{as_of.date()}] Wrote snapshot → {out_path}", level="ok")


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate historical weekly Weinstein equity snapshots for backtesting."
    )
    ap.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD) for snapshot range.",
    )
    ap.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD) for snapshot range.",
    )
    ap.add_argument(
        "--universe-csv",
        type=str,
        default="",
        help="Optional CSV with a 'ticker' column. If omitted, uses latest weekly CSV.",
    )
    ap.add_argument(
        "--benchmark",
        type=str,
        default=DEFAULT_BENCHMARK,
        help=f"Benchmark symbol for RS calc (default {DEFAULT_BENCHMARK}).",
    )
    ap.add_argument(
        "--freq",
        type=str,
        default=DEFAULT_FREQ,
        help=f"Pandas weekly frequency string (default {DEFAULT_FREQ}, i.e., Fridays).",
    )

    args = ap.parse_args()

    # Basic sanity
    try:
        _ = datetime.fromisoformat(args.start)
        _ = datetime.fromisoformat(args.end)
    except ValueError:
        raise SystemExit("ERROR: --start and --end must be in YYYY-MM-DD format.")

    if args.universe_csv:
        universe = load_universe_from_csv(args.universe_csv)
    else:
        universe = load_universe_from_latest_weekly()

    if not universe:
        raise SystemExit("ERROR: Universe is empty; cannot generate snapshots.")

    generate_snapshots(
        start=args.start,
        end=args.end,
        universe=universe,
        benchmark=args.benchmark.upper(),
        freq=args.freq,
    )


if __name__ == "__main__":
    main()
