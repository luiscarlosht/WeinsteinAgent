#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Intraday Diagnostics

Features
- Reads intraday_debug.csv and computes:
  * "near-miss" buy candidates (all gates green except price)
  * distances to pivot in basis points
  * per-ticker aggregates (min distance, scans, last_seen)
- Extracts "near-universe" tickers from intraday_watch_*.html
- Writes a human-readable summary file with timestamp
- Maintains a simple JSON state file (rolling per-ticker stats)
- Optional: --explain T1,T2 prints gate-by-gate reasons a ticker didn’t trigger

Exit code: 0 even if CSV is empty; summary still produced for HTML near-universe.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from glob import glob
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required. pip install pandas", file=sys.stderr)
    raise

# -----------------------
# Utils
# -----------------------

TS_FMT = "%Y%m%d_%H%M%S"

COND_COLS_ORDER = [
    "cond_weekly_stage_ok",
    "cond_rs_ok",
    "cond_ma_ok",
    "cond_pivot_ok",
    "cond_buy_vol_ok",
    "cond_pace_full_gate",
    "cond_near_pace_gate",
    "cond_buy_price_ok",
]

REQUIRED_NUMERIC = ["price", "pivot", "elapsed_min"]
REQUIRED_KEY = ["ticker"]

NEAR_TICKER_REGEX = re.compile(r"<ol><li><b>\d+\.</b>\s*<b>([A-Z]+)</b>\s*@")

# -----------------------
# HTML near-universe
# -----------------------

def extract_near_universe(html_glob: str, out_file: str) -> Set[str]:
    tickers: Set[str] = set()
    for path in sorted(glob(html_glob)):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            for m in NEAR_TICKER_REGEX.finditer(text):
                tickers.add(m.group(1))
        except Exception:
            # skip unreadable file
            pass
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for t in sorted(tickers):
            f.write(f"{t}\n")
    return tickers

# -----------------------
# CSV load / guards
# -----------------------

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path) or os.path.getsize(path) < 3:
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Create any missing boolean columns as False
    for c in COND_COLS_ORDER:
        if c not in df.columns:
            df[c] = False

    # Create required numerics if missing
    for c in REQUIRED_NUMERIC:
        if c not in df.columns:
            df[c] = pd.NA

    # Create key columns if missing
    for c in REQUIRED_KEY:
        if c not in df.columns:
            df[c] = ""

    # Drop rows without fundamental keys
    df = df.dropna(subset=["ticker", "price", "pivot", "elapsed_min"])
    return df

def add_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    # positive dist_bps => price below pivot; negative => above
    df["dist_bps"] = (df["pivot"] - df["price"]) / df["pivot"] * 1e4
    # keep a sortable timestamp column if not present
    if "ts" not in df.columns:
        # emulate using elapsed_min if provided
        df["ts"] = df["elapsed_min"]
    return df

# -----------------------
# Diagnostics
# -----------------------

def compute_near_misses(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    near_miss: all gates green EXCEPT price (buy price not yet ok)
    pass_rows: all gates green including price (for sanity / reference)
    """
    gate_all_but_price = (
        df["cond_weekly_stage_ok"]
        & df["cond_rs_ok"]
        & df["cond_ma_ok"]
        & df["cond_pivot_ok"]
        & df["cond_buy_vol_ok"]
        & df["cond_pace_full_gate"]
        & df["cond_near_pace_gate"]
    )

    near_miss = gate_all_but_price & (df["cond_buy_price_ok"] == False)
    passed    = gate_all_but_price & (df["cond_buy_price_ok"] == True)

    return df[near_miss].copy(), df[passed].copy()

def aggregate_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["scans","min_dist_bps","weekly_ok","last_seen"])
    agg = (
        df.groupby("ticker")
          .agg(scans=("ticker","count"),
               min_dist_bps=("dist_bps","min"),
               weekly_ok=("cond_weekly_stage_ok","max"),
               last_seen=("elapsed_min","max"))
          .sort_values(["min_dist_bps","scans"], ascending=[True, False])
    )
    return agg

def near_now_candidates(df: pd.DataFrame, bps_threshold: float) -> pd.DataFrame:
    """Candidates where all gates green except price AND price within threshold below pivot."""
    if df.empty:
        return df
    mask = (
        (df["dist_bps"] >= 0) &
        (df["dist_bps"] <= bps_threshold) &
        df["cond_weekly_stage_ok"] &
        df["cond_rs_ok"] &
        df["cond_ma_ok"] &
        df["cond_pivot_ok"] &
        df["cond_buy_vol_ok"] &
        df["cond_pace_full_gate"] &
        df["cond_near_pace_gate"] &
        (df["cond_buy_price_ok"] == False)
    )
    return df[mask].copy().sort_values(["dist_bps","elapsed_min"], ascending=[True, False])

# -----------------------
# State file
# -----------------------

def load_state(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"tickers": {}, "runs": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tickers": {}, "runs": []}

def update_state(state: Dict, agg: pd.DataFrame, run_ts: str) -> Dict:
    tickers = state.get("tickers", {})
    for t, row in agg.reset_index().itertuples(index=False):
        info = tickers.get(t, {})
        info["min_dist_bps"] = float(row.min_dist_bps) if pd.notna(row.min_dist_bps) else None
        info["scans"] = int(row.scans) if pd.notna(row.scans) else 0
        info["weekly_ok"] = bool(row.weekly_ok) if pd.notna(row.weekly_ok) else False
        info["last_seen"] = int(row.last_seen) if pd.notna(row.last_seen) else None
        info["last_updated"] = run_ts
        tickers[t] = info
    state["tickers"] = tickers
    state.setdefault("runs", []).append(run_ts)
    # de-dup & keep last 50 run stamps
    state["runs"] = list(dict.fromkeys(state["runs"]))[-50:]
    return state

def save_state(state: Dict, path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)

# -----------------------
# Explainer
# -----------------------

def first_failing_gates(row: pd.Series) -> List[str]:
    """Return names of gates that are False (in order), excluding cond_buy_price_ok."""
    failing = []
    for c in COND_COLS_ORDER[:-1]:
        v = row.get(c, False)
        if not bool(v):
            failing.append(c)
    # handle price gate separately at the end
    if not bool(row.get("cond_buy_price_ok", False)):
        failing.append("cond_buy_price_ok")
    return failing

def explain_tickers(df: pd.DataFrame, tickers: Iterable[str], limit_rows: int = 8) -> str:
    if df is None or df.empty:
        return "No CSV rows available to explain."
    lines: List[str] = []
    cols_to_show = ["elapsed_min","price","pivot","dist_bps"] + COND_COLS_ORDER
    for t in tickers:
        sub = df[df["ticker"] == t].copy()
        lines.append(f"\n--- Explanation for {t} ---")
        if sub.empty:
            lines.append("No rows found.")
            continue
        sub = sub.sort_values("elapsed_min").tail(limit_rows)
        for _, r in sub.iterrows():
            failing = first_failing_gates(r)
            # compact gate string
            gate_str = ", ".join([g.replace("cond_","") for g in failing]) or "ALL_PASS"
            em = int(r["elapsed_min"]) if pd.notna(r["elapsed_min"]) else -1
            price = float(r["price"]) if pd.notna(r["price"]) else float("nan")
            pivot = float(r["pivot"]) if pd.notna(r["pivot"]) else float("nan")
            dist = float(r["dist_bps"]) if pd.notna(r["dist_bps"]) else float("nan")
            lines.append(
                f"t+{em:>4}m  price={price:.2f}  pivot={pivot:.2f}  dist_bps={dist:>7.1f}  failing=[{gate_str}]"
            )
    return "\n".join(lines)

# -----------------------
# Summary writer
# -----------------------

def write_summary(
    outdir: str,
    run_ts: str,
    had_csv: bool,
    near_universe_count: int,
    near_agg: Optional[pd.DataFrame],
    near_now: Optional[pd.DataFrame],
    explained_text: Optional[str],
    outfile_hint: Optional[str] = None,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    fname = outfile_hint or f"diag_summary_{run_ts}.txt"
    path = os.path.join(outdir, fname)

    def df_head(df: Optional[pd.DataFrame], n=20) -> str:
        if df is None or df.empty:
            return "(none)"
        with pd.option_context("display.max_columns", 100, "display.width", 160):
            return df.head(n).to_string()

    lines = []
    lines.append(f"=== Weinstein Intraday Diagnostics @ {run_ts} ===")
    lines.append("CSV: " + ("OK" if had_csv else "No valid intraday_debug.csv found or file empty."))
    lines.append(f"Near-universe tickers discovered in HTML: {near_universe_count} (saved: near_universe.txt)")
    lines.append("")
    lines.append("Top near-miss by ticker (min distance, scans, weekly_ok, last_seen):")
    lines.append(df_head(near_agg))
    lines.append("")
    lines.append("Near-NOW (within bps threshold and all gates except price):")
    lines.append(df_head(near_now))
    if explained_text:
        lines.append("")
        lines.append("=== Explain Results ===")
        lines.append(explained_text)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path

# -----------------------
# Main
# -----------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose intraday 'near trigger' behavior.")
    p.add_argument("--csv", required=False, default="./output/intraday_debug.csv",
                   help="Path to intraday_debug.csv")
    p.add_argument("--outdir", required=False, default="./output",
                   help="Output directory for summary/near_universe/state")
    p.add_argument("--html-glob", required=False, default="./output/intraday_watch_*.html",
                   help="Glob for intraday_watch HTML snapshots")
    p.add_argument("--state", required=False, default="./output/diag_state.json",
                   help="JSON state file path (created/updated)")
    p.add_argument("--bps-threshold", type=float, default=25.0,
                   help="Max distance to pivot (in bps) to consider 'near-NOW'")
    p.add_argument("--min-scans", type=int, default=2,
                   help="Minimum scans for a ticker to appear in top near-miss aggregate")
    p.add_argument("--explain", type=str, default="",
                   help="Comma-separated tickers to explain (e.g. MU,DDOG)")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_ts = dt.datetime.now().strftime(TS_FMT)

    # 1) Extract near-universe tickers from HTML
    near_universe_path = os.path.join(args.outdir, "near_universe.txt")
    tickers_set = extract_near_universe(args.html_glob, near_universe_path)

    # 2) Load CSV (safe)
    df = safe_read_csv(args.csv)
    had_csv = df is not None and not df.empty

    near_miss_df = pd.DataFrame()
    near_now_df  = pd.DataFrame()
    agg_df       = pd.DataFrame()
    explained_text = ""

    if had_csv:
        # Ensure columns & compute fields
        df = ensure_columns(df)
        if not df.empty:
            df = add_computed_columns(df)

            # Compute near-miss set
            near_miss_df, passed_df = compute_near_misses(df)

            # Aggregate near-miss per ticker
            agg_df = aggregate_per_ticker(near_miss_df)
            if args.min_scans > 1 and not agg_df.empty:
                agg_df = agg_df[agg_df["scans"] >= args.min_scans].copy()

            # Find near-NOW within threshold
            near_now_df = near_now_candidates(near_miss_df, args.bps_threshold)

            # Explain tickers if requested
            if args.explain.strip():
                to_explain = [t.strip().upper() for t in args.explain.split(",") if t.strip()]
                explained_text = explain_tickers(df, to_explain, limit_rows=12)

            # Update state
            state = load_state(args.state)
            state = update_state(state, agg_df, run_ts)
            save_state(state, args.state)

    # 3) Write summary
    summary_name = f"diag_summary_{run_ts}.txt"
    summary_path = write_summary(
        outdir=args.outdir,
        run_ts=run_ts,
        had_csv=had_csv,
        near_universe_count=len(tickers_set),
        near_agg=agg_df,
        near_now=near_now_df,
        explained_text=explained_text,
        outfile_hint=summary_name,
    )

    # 4) Print final pointers
    print("Diagnostics complete.")
    print(f"- Near-universe: {near_universe_path}")
    print(f"- Summary:      {summary_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
