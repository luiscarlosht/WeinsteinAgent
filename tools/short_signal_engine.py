#!/usr/bin/env python3
"""
short_signal_engine.py

Companion to weinstein_short_watcher.py.

Reads a short-side diagnostics CSV (e.g. short_debug.csv) and:
- Aggregates per-ticker short state (NEAR / TRIG / IDLE/etc).
- Counts how many times a ticker was NEAR or TRIG within a recent intraday window.
- Prints a compact console summary.
- Optionally writes a summary CSV.

Intended to behave similarly to tools/signal_engine.py for the long side,
but focused on Stage 4 / short setups coming from weinstein_short_watcher.py.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _now_str() -> str:
    """Return a human-readable timestamp for logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str) -> None:
    print(f"• [{_now_str()}] {msg}")


def log_ok(msg: str) -> None:
    print(f"✅ [{_now_str()}] {msg}")


def log_warn(msg: str) -> None:
    print(f"⚠️ [{_now_str()}] {msg}")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize short-side diagnostics from weinstein_short_watcher.py"
    )
    p.add_argument(
        "--csv",
        required=True,
        help=(
            "Input CSV file produced by weinstein_short_watcher.py "
            "(e.g. ./output/short_debug.csv)"
        ),
    )
    p.add_argument(
        "--outdir",
        default="./output",
        help="Directory to write summary artifacts (default: ./output)",
    )
    p.add_argument(
        "--window-min",
        type=int,
        default=120,
        help=(
            "Only consider rows where elapsed_min <= this value. "
            "Use 0 to consider all rows. Default: 120 minutes."
        ),
    )
    p.add_argument(
        "--bps-threshold",
        type=int,
        default=0,
        help=(
            "Reserved for future use (basis-point distance filters). "
            "Currently unused, accepted for CLI symmetry."
        ),
    )
    p.add_argument(
        "--explain",
        metavar="TICKER",
        help="Print a detailed explanation for a single ticker",
    )
    return p.parse_args(argv)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        log_warn(f"CSV {path} is empty.")
    return df


def apply_window_filter(df: pd.DataFrame, window_min: int) -> Tuple[pd.DataFrame, bool]:
    """
    Filter by elapsed_min <= window_min if the column exists and window_min > 0.
    Returns (filtered_df, used_window_filter).
    """
    if window_min <= 0:
        return df, False

    if "elapsed_min" not in df.columns:
        log_warn(
            "elapsed_min column not present in CSV — unable to apply window "
            "filtering; using all rows."
        )
        return df, False

    df_win = df[df["elapsed_min"] <= window_min].copy()
    if df_win.empty and not df.empty:
        log_warn("CSV had rows, but none within the requested window.")
    return df_win, True


def infer_hits_for_ticker(df_t: pd.DataFrame) -> Tuple[int, int, str, float]:
    """
    Given all rows for a single ticker (already filtered by window if applicable),
    compute:
      - near_hits
      - trig_hits
      - last_short_state
      - last_price

    This is intentionally defensive:
    - If cond_short_near_now / cond_short_confirm exist, we use them.
    - Otherwise we fall back to counting based on short_state (NEAR/TRIG).
    """
    # Get last row chronologically as stored in CSV
    last = df_t.iloc[-1]

    last_state = str(last.get("short_state", "NA"))

    # Last price: prefer 'price', but fall back to 'close' if needed
    last_px = np.nan
    if "price" in df_t.columns:
        last_px = float(last["price"])
    elif "close" in df_t.columns:
        last_px = float(last["close"])

    # Default hits
    near_hits = 0
    trig_hits = 0

    # Preferred path: use explicit boolean flags if present
    has_near_flag = "cond_short_near_now" in df_t.columns
    has_trig_flag = "cond_short_confirm" in df_t.columns or "cond_short_trig_now" in df_t.columns

    if has_near_flag or has_trig_flag:
        if has_near_flag:
            near_hits = int(
                df_t["cond_short_near_now"].fillna(False).astype(bool).sum()
            )
        else:
            # Fallback: treat rows with short_state == "NEAR" as near hits
            near_hits = int((df_t.get("short_state", "") == "NEAR").sum())

        trig_col = (
            "cond_short_confirm"
            if "cond_short_confirm" in df_t.columns
            else "cond_short_trig_now"
        )
        if trig_col in df_t.columns:
            trig_hits = int(df_t[trig_col].fillna(False).astype(bool).sum())
        else:
            # Fallback again: short_state == "TRIG"
            trig_hits = int((df_t.get("short_state", "") == "TRIG").sum())
    else:
        # No dedicated flags – rely solely on short_state
        if "short_state" in df_t.columns:
            col = df_t["short_state"].astype(str)
            near_hits = int((col == "NEAR").sum())
            trig_hits = int((col == "TRIG").sum())

    return near_hits, trig_hits, last_state, last_px


def summarize(df: pd.DataFrame, outdir: Path, window_min: int) -> pd.DataFrame:
    """
    Produce a per-ticker summary DataFrame with:
      ticker, ShortState, NearHits, TrigHits, LastPx
    and write it out as CSV in outdir.
    """
    if df.empty:
        log_warn("No rows to summarize (after any window filtering).")
        return pd.DataFrame()

    # Make sure ticker column exists
    ticker_col = None
    for cand in ("ticker", "symbol", "Ticker", "Symbol"):
        if cand in df.columns:
            ticker_col = cand
            break
    if ticker_col is None:
        raise KeyError("Could not find a ticker/symbol column in CSV.")

    rows = []
    for t, df_t in df.groupby(ticker_col):
        near_hits, trig_hits, last_state, last_px = infer_hits_for_ticker(df_t)
        rows.append(
            {
                "ticker": t,
                "ShortState": last_state,
                "NearHits": near_hits,
                "TrigHits": trig_hits,
                "LastPx": last_px,
            }
        )

    df_summary = (
        pd.DataFrame(rows)
        .sort_values(
            by=["TrigHits", "NearHits", "ticker"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{window_min}min" if window_min > 0 else "_full"
    out_path = outdir / f"short_signal_summary{suffix}.csv"
    df_summary.to_csv(out_path, index=False)
    log_ok(f"Wrote short signal summary → {out_path}")
    return df_summary


def print_table(df_summary: pd.DataFrame) -> None:
    if df_summary.empty:
        print("No short signals found in the selected window.")
        return

    # Console table similar to tools/signal_engine.py
    header = f"{'Ticker':<6} {'ShortState':<10} {'NearHits':>8} {'TrigHits':>8} {'LastPx':>12}"
    print(header)
    print("-" * len(header))
    for _, row in df_summary.iterrows():
        print(
            f"{row['ticker']:<6} "
            f"{str(row['ShortState']):<10} "
            f"{int(row['NearHits']):>8d} "
            f"{int(row['TrigHits']):>8d} "
            f"{row['LastPx']:>12.6f}"
        )


def explain_ticker(df: pd.DataFrame, ticker: str) -> None:
    """Print a more verbose explanation for a single ticker."""
    ticker_col = None
    for cand in ("ticker", "symbol", "Ticker", "Symbol"):
        if cand in df.columns:
            ticker_col = cand
            break
    if ticker_col is None:
        log_warn("No ticker/symbol column found; cannot explain specific ticker.")
        return

    df_t = df[df[ticker_col] == ticker]
    if df_t.empty:
        log_warn(f"No rows found for ticker {ticker}.")
        return

    print()
    print(f"Details for {ticker}")
    print("=" * (10 + len(ticker)))

    cols_to_show = [ticker_col]
    for c in (
        "elapsed_min",
        "price",
        "short_state",
        "cond_short_near_now",
        "cond_short_confirm",
    ):
        if c in df_t.columns:
            cols_to_show.append(c)

    # Drop duplicates while preserving order
    seen = set()
    cols_to_show_unique = []
    for c in cols_to_show:
        if c not in seen:
            cols_to_show_unique.append(c)
            seen.add(c)

    df_view = df_t[cols_to_show_unique].copy()
    print(df_view.to_string(index=False))
    print()


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    log_info(f"Loading CSV: {csv_path}")
    try:
        df_all = load_csv(csv_path)
    except Exception as e:
        log_warn(f"Failed to load CSV: {e}")
        return 1

    if df_all.empty:
        log_warn("Aborting: CSV empty.")
        return 0

    df_win, used_window = apply_window_filter(df_all, args.window_min)
    if used_window:
        log_info(f"Applied window filter: elapsed_min <= {args.window_min} minutes")

    if df_win.empty:
        # We already logged a message inside apply_window_filter
        return 0

    df_summary = summarize(df_win, outdir, args.window_min)
    print()
    print_table(df_summary)
    print()

    # Overall stats
    total_trig = int(df_summary["TrigHits"].sum())
    total_near = int(df_summary["NearHits"].sum())
    log_ok(
        f"Done. Aggregated {len(df_summary)} tickers: "
        f"{total_trig} TRIG hits, {total_near} NEAR hits."
    )

    if args.explain:
        explain_ticker(df_win, args.explain)

    return 0


if __name__ == "__main__":
    sys.exit(main())
