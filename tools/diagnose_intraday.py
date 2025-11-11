#!/usr/bin/env python3
"""
Weinstein Intraday Diagnostics (integrated)

What this does (safe to run anytime):
- Reads a lightweight intraday_debug.csv (if present) and computes "near-miss" buy events
  (all gates true except the actual buy-price cross), plus a "near-now" snapshot using
  a basis-points (bps) distance to pivot.
- Optionally scrapes your generated intraday_watch_*.html files to build a
  "near-universe" of tickers that repeatedly appeared near/pivot.
- Writes a human-readable summary file and a near_universe.txt. Optionally
  appends a row-wise log (for time-series diagnostics) and persists state across runs.

Examples:
  python3 tools/diagnose_intraday.py \
    --csv ./output/intraday_debug.csv \
    --outdir ./output \
    --html-glob "./output/intraday_watch_*.html" \
    --state ./output/diag_state.json \
    --explain MU,DDOG

Outputs:
  - <outdir>/diag_summary_YYYYmmdd_HHMMSS.txt
  - <outdir>/near_universe.txt (if --html-glob provided)
  - <outdir>/diag_state.json (if --state provided)
  - optional: <outdir>/diag_log.csv (if --log-csv provided)

CSV expected columns (missing ones are handled gracefully):
  ticker, price, pivot, elapsed_min,
  cond_weekly_stage_ok, cond_rs_ok, cond_ma_ok, cond_pivot_ok,
  cond_buy_vol_ok, cond_pace_full_gate, cond_near_pace_gate, cond_buy_price_ok

"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from glob import glob
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    print("pandas is required. pip install pandas", file=sys.stderr)
    raise

# ----------------------------
# Helpers
# ----------------------------

RE_HTML_NEAR_TICKER = re.compile(r"<ol><li><b>\\d+\.</b> <b>([A-Z\.]+)</b> @")

MANDATORY_NUMERIC = ["price", "pivot"]
OPTIONAL_BOOL_COLS = [
    "cond_weekly_stage_ok",
    "cond_rs_ok",
    "cond_ma_ok",
    "cond_pivot_ok",
    "cond_buy_vol_ok",
    "cond_pace_full_gate",
    "cond_near_pace_gate",
    "cond_buy_price_ok",
]
OPTIONAL_NUMERIC_COLS = ["elapsed_min", "dist_bps"]


@dataclass
class NearMissRec:
    ticker: str
    scans: int
    min_dist_bps: float
    weekly_ok: bool
    last_seen: Optional[float]


@dataclass
class GateStats:
    ticker: str
    rows: int
    gate_true_pct: Dict[str, float]
    min_dist_bps: Optional[float]


# ----------------------------
# Core logic
# ----------------------------

def read_csv_if_any(path: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path) or os.path.getsize(path) <= 1:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Basic sanitation
    if "ticker" not in df.columns:
        return None

    # Drop rows missing the core numeric inputs
    missing = [c for c in MANDATORY_NUMERIC if c not in df.columns]
    for c in missing:
        df[c] = None
    df = df.dropna(subset=["ticker"])  # require a ticker

    # Make sure expected columns exist
    for c in OPTIONAL_BOOL_COLS:
        if c not in df.columns:
            df[c] = False
    for c in OPTIONAL_NUMERIC_COLS:
        if c not in df.columns:
            df[c] = None

    # coerce types (best-effort)
    for c in ["price", "pivot", "elapsed_min", "dist_bps"]:
        if c in df.columns:
            with pd.option_context("mode.chained_assignment", None):
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute dist_bps if missing
    if "dist_bps" not in df.columns or df["dist_bps"].isna().all():
        with pd.option_context("mode.chained_assignment", None):
            df["dist_bps"] = (df["pivot"] - df["price"]) / df["pivot"] * 1e4

    return df


def build_near_universe_from_html(html_glob: str) -> List[str]:
    if not html_glob:
        return []
    tickers = set()
    for path in glob(html_glob):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for m in RE_HTML_NEAR_TICKER.finditer(content):
                tickers.add(m.group(1))
        except Exception:
            continue
    return sorted(tickers)


def compute_near_miss(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (near_miss_df, near_now_df)

    near-miss = all gates true except cond_buy_price_ok == False
    near-now  = within bps threshold AND gates satisfied except actual buy-price cross
               (the final filtering by threshold happens later in the caller)
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Normalize booleans
    for c in OPTIONAL_BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Use best-effort elapsed_min
    ts_col = "elapsed_min" if "elapsed_min" in df.columns else None

    buy_gates_except_price = (
        df.get("cond_weekly_stage_ok", False)
        & df.get("cond_rs_ok", False)
        & df.get("cond_ma_ok", False)
        & df.get("cond_pivot_ok", False)
        & df.get("cond_buy_vol_ok", False)
        & df.get("cond_pace_full_gate", False)
        & df.get("cond_near_pace_gate", False)
        & (~df.get("cond_buy_price_ok", False))
    )

    near = df[buy_gates_except_price].copy()
    if ts_col:
        near["ts_min"] = df[ts_col]
    else:
        near["ts_min"] = None

    # near-now candidate set (same logical gates, just leaving threshold for caller)
    near_now = near.copy()

    return near, near_now


def summarize_near_miss(near: pd.DataFrame) -> List[NearMissRec]:
    if near is None or near.empty:
        return []
    g = (
        near.groupby("ticker")
        .agg(
            scans=("ticker", "count"),
            min_dist_bps=("dist_bps", "min"),
            weekly_ok=("cond_weekly_stage_ok", "max"),
            last_seen=("ts_min", "max"),
        )
        .sort_values(["min_dist_bps", "scans"], ascending=[True, False])
    )
    out: List[NearMissRec] = []
    for t, row in g.iterrows():
        out.append(
            NearMissRec(
                ticker=str(t),
                scans=int(row["scans"]) if pd.notna(row["scans"]) else 0,
                min_dist_bps=float(row["min_dist_bps"]) if pd.notna(row["min_dist_bps"]) else float("inf"),
                weekly_ok=bool(row["weekly_ok"]) if pd.notna(row["weekly_ok"]) else False,
                last_seen=float(row["last_seen"]) if pd.notna(row["last_seen"]) else None,
            )
        )
    return out


def gate_explain(df: pd.DataFrame, tickers: List[str]) -> List[GateStats]:
    if df is None or df.empty or not tickers:
        return []
    out: List[GateStats] = []
    cols = [
        "cond_weekly_stage_ok",
        "cond_rs_ok",
        "cond_ma_ok",
        "cond_pivot_ok",
        "cond_buy_vol_ok",
        "cond_pace_full_gate",
        "cond_near_pace_gate",
        "cond_buy_price_ok",
    ]
    for t in tickers:
        sdf = df[df["ticker"] == t]
        if sdf.empty:
            out.append(GateStats(ticker=t, rows=0, gate_true_pct={}, min_dist_bps=None))
            continue
        gate_pct: Dict[str, float] = {}
        for c in cols:
            if c in sdf.columns and len(sdf[c]) > 0:
                gate_pct[c] = float(sdf[c].astype(bool).mean()) * 100.0
        min_dist = None
        if "dist_bps" in sdf.columns and not sdf["dist_bps"].isna().all():
            try:
                min_dist = float(sdf["dist_bps"].min())
            except Exception:
                min_dist = None
        out.append(GateStats(ticker=t, rows=len(sdf), gate_true_pct=gate_pct, min_dist_bps=min_dist))
    return out


# ----------------------------
# State & logging
# ----------------------------

def load_state(path: Optional[str]) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: Optional[str], state: Dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_log_csv(path: Optional[str], rows: List[Dict]):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ----------------------------
# Rendering
# ----------------------------

def write_summary(outdir: str,
                  ts: str,
                  csv_ok: bool,
                  near_universe: List[str],
                  near_miss: List[NearMissRec],
                  near_now_df: pd.DataFrame,
                  explain_stats: List[GateStats]) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"diag_summary_{ts}.txt")

    lines: List[str] = []
    lines.append(f"=== Weinstein Intraday Diagnostics @ {ts} ===")
    lines.append("CSV: " + ("OK" if csv_ok else "No valid intraday_debug.csv found or file empty."))
    lines.append(f"Near-universe tickers discovered in HTML: {len(near_universe)} (saved: near_universe.txt)")
    lines.append("")

    # Near-miss table
    lines.append("Top near-miss by ticker (min distance, scans, weekly_ok, last_seen):")
    if near_miss:
        for r in near_miss[:30]:
            last_seen = "-" if r.last_seen is None else f"{r.last_seen:.0f}m"
            lines.append(
                f"  {r.ticker:<6}  min_dist={r.min_dist_bps:>7.1f} bps  scans={r.scans:>3}  weekly_ok={str(r.weekly_ok):<5}  last_seen={last_seen}"
            )
    else:
        lines.append("(none)")
    lines.append("")

    # Near-NOW
    lines.append("Near-NOW (within bps threshold and all gates except price):")
    if near_now_df is not None and not near_now_df.empty:
        show_cols = [c for c in ["ticker", "price", "pivot", "dist_bps", "elapsed_min"] if c in near_now_df.columns]
        sample = near_now_df.sort_values("dist_bps").head(25)
        for _, row in sample.iterrows():
            t = row.get("ticker", "?")
            dist = row.get("dist_bps", None)
            em = row.get("elapsed_min", None)
            price = row.get("price", None)
            pivot = row.get("pivot", None)
            lines.append(
                f"  {t:<6} price={price!s:<8} pivot={pivot!s:<8} dist_bps={(dist if pd.notna(dist) else '?'):>7} elapsed_min={(int(em) if pd.notna(em) else '-'):<4}"
            )
    else:
        lines.append("(none)")
    lines.append("")

    # Explainers
    if explain_stats:
        lines.append("Gate explainers:")
        for st in explain_stats:
            lines.append(f"  {st.ticker}: rows={st.rows}  min_dist_bps={st.min_dist_bps if st.min_dist_bps is not None else '-'}")
            if st.gate_true_pct:
                gates = ", ".join([f"{k}={v:.0f}%" for k, v in st.gate_true_pct.items()])
                lines.append(f"    {gates}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Diagnose Weinstein intraday near-misses and near-now candidates")
    ap.add_argument("--csv", default="", help="Path to intraday_debug.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for summary and artifacts")
    ap.add_argument("--html-glob", default="", help="Glob for intraday_watch_*.html to build near_universe.txt")
    ap.add_argument("--state", default="", help="Path to JSON state for persistence across runs")
    ap.add_argument("--log-csv", default="", help="Optional: append long-form diagnostics to this CSV")
    ap.add_argument("--bps-thresh", type=float, default=25.0, help="Basis points from pivot to include in near-now")
    ap.add_argument("--min-scans", type=int, default=1, help="Minimum scans to show a ticker in near-miss table")
    ap.add_argument("--explain", default="", help="Comma-separated list of tickers to explain gates for")
    ap.add_argument("--dump-candidates", default="", help="Optional path to dump raw near-now candidates CSV")

    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Build near-universe from HTML
    near_universe: List[str] = []
    if args.html_glob:
        near_universe = build_near_universe_from_html(args.html_glob)
        # write it
        nu_path = os.path.join(args.outdir, "near_universe.txt")
        with open(nu_path, "w", encoding="utf-8") as f:
            for t in near_universe:
                f.write(t + "\n")

    # 2) Load CSV & compute near-miss + near-now
    df = read_csv_if_any(args.csv)
    csv_ok = df is not None and not df.empty

    near_miss_list: List[NearMissRec] = []
    near_now_df = pd.DataFrame()

    if csv_ok:
        near_miss_df, near_now_df_raw = compute_near_miss(df)

        # Filter near-now by threshold: price just under pivot is positive dist_bps
        if not near_now_df_raw.empty:
            near_now_df = near_now_df_raw.copy()
            with pd.option_context("mode.chained_assignment", None):
                near_now_df = near_now_df[(near_now_df["dist_bps"] >= 0) & (near_now_df["dist_bps"] <= args.bps_thresh)]

        # Aggregate near-miss
        near_miss_list = [r for r in summarize_near_miss(near_miss_df) if r.scans >= args.min_scans]

        # Optional: log rows
        if args.log_csv:
            log_rows: List[Dict] = []
            if not near_miss_df.empty:
                for _, r in near_miss_df.iterrows():
                    log_rows.append({**{c: r.get(c) for c in df.columns}, "ts": ts, "kind": "near_miss"})
            if not near_now_df.empty:
                for _, r in near_now_df.iterrows():
                    log_rows.append({**{c: r.get(c) for c in df.columns}, "ts": ts, "kind": "near_now"})
            append_log_csv(args.log_csv, log_rows)

        # Optional: dump raw near-now
        if args.dump_candidates and not near_now_df.empty:
            near_now_df.to_csv(args.dump_candidates, index=False)

    # 3) Explainers
    explain_list = [t.strip().upper() for t in args.explain.split(",") if t.strip()] if args.explain else []
    explain_stats: List[GateStats] = gate_explain(df, explain_list) if csv_ok and explain_list else []

    # 4) Persist state (best-effort; we just store latest near_miss tickers + timestamp)
    state = load_state(args.state)
    if csv_ok:
        state.setdefault("runs", []).append({
            "ts": ts,
            "near_miss": [asdict(r) for r in near_miss_list[:50]],
            "near_universe_sample": near_universe[:50],
        })
        # keep last 50
        state["runs"] = state["runs"][-50:]
        save_state(args.state, state)

    # 5) Write summary
    summary_path = write_summary(
        outdir=args.outdir,
        ts=ts,
        csv_ok=csv_ok,
        near_universe=near_universe,
        near_miss=near_miss_list,
        near_now_df=near_now_df,
        explain_stats=explain_stats,
    )

    print("Diagnostics complete.")
    if near_universe:
        print(f"- Near-universe: {os.path.join(args.outdir, 'near_universe.txt')}")
    print(f"- Summary:      {summary_path}")


if __name__ == "__main__":
    main()
