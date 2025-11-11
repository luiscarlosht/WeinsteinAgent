#!/usr/bin/env python3
"""
Weinstein Intraday Diagnostics

Features
- Reads intraday_debug.csv (current-week scans) and computes:
  * Near-miss candidates ("all green except price")
  * Closest distance to pivot in basis points
  * Last-seen (elapsed_min)
- Scrapes the "near" tickers from intraday_watch_*.html files into near_universe.txt
- Writes a human-friendly summary text file
- Optional per-ticker drill-down with --explain TICK1,TICK2 (why no BUY?)
- Persists lightweight state in a JSON file (e.g., last run params)

Return codes:
- 0 on success. Non-fatal data issues are reported in the summary.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import sys
from typing import Iterable, List, Tuple, Dict, Optional

try:
    import pandas as pd
except Exception as e:
    print(f"ERROR: pandas not available: {e}", file=sys.stderr)
    sys.exit(2)


# ---------- Utilities ----------

TS_FMT = "%Y%m%d_%H%M%S"


def now_tag() -> str:
    return dt.datetime.now().strftime(TS_FMT)


def ensure_outdir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_lines(path: str, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


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
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"WARN: failed to write state file {path}: {e}", file=sys.stderr)


# ---------- HTML scraping: near-universe ----------

NEAR_LI_TICKER_RE = re.compile(
    r"<ol><li><b>\d+\.</b>\s*<b>([A-Z]+)</b>\s*@",
    re.IGNORECASE,
)

def scrape_near_universe(html_glob: Optional[str]) -> Tuple[List[str], List[str]]:
    """Return (tickers, files_scanned)."""
    if not html_glob:
        return [], []

    files = sorted(glob.glob(html_glob))
    tickers: set[str] = set()
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            for m in NEAR_LI_TICKER_RE.finditer(text):
                t = m.group(1).upper()
                tickers.add(t)
        except Exception as e:
            print(f"WARN: failed to parse {fp}: {e}", file=sys.stderr)
    return sorted(tickers), files


# ---------- CSV diagnostics ----------

CSV_REQUIRED_COLS = [
    "elapsed_min", "ticker", "price", "pivot",
    "cond_weekly_stage_ok", "cond_rs_ok", "cond_ma_ok",
    "cond_pivot_ok", "cond_buy_vol_ok",
    "cond_pace_full_gate", "cond_near_pace_gate",
    "cond_buy_price_ok",
]

def load_intraday_csv(csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path) < 5:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"WARN: could not read {csv_path}: {e}", file=sys.stderr)
        return None

    # best-effort guardrails
    missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"WARN: CSV missing columns: {missing}", file=sys.stderr)
        # Continue if we at least have enough to compute distances
        minimal = {"ticker", "price", "pivot"}
        if not minimal.issubset(set(df.columns)):
            return None

    # drop blank rows
    keep_cols = [c for c in ["ticker", "price", "pivot"] if c in df.columns]
    df = df.dropna(subset=keep_cols)
    if "elapsed_min" in df.columns:
        df["elapsed_min"] = pd.to_numeric(df["elapsed_min"], errors="coerce").fillna(0).astype(int)

    # compute distance to pivot (bps): positive => price below pivot (near breakout)
    df["dist_bps"] = (df["pivot"] - df["price"]) / df["pivot"] * 1e4

    # cast condition columns to bool if present
    for c in CSV_REQUIRED_COLS:
        if c in df.columns and df[c].dtype != bool:
            df[c] = df[c].astype(bool, copy=False, errors="ignore")
    return df


def compute_near_misses(df: pd.DataFrame) -> pd.DataFrame:
    """
    "All green except price" = a 'near-miss'
    """
    # if any condition columns are missing, treat them as False (conservative)
    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(False, index=df.index)

    gates_except_price = (
        col("cond_weekly_stage_ok")
        & col("cond_rs_ok")
        & col("cond_ma_ok")
        & col("cond_pivot_ok")
        & col("cond_buy_vol_ok")
        & col("cond_pace_full_gate")
        & col("cond_near_pace_gate")
        & (~col("cond_buy_price_ok"))
    )

    near = df[gates_except_price].copy()
    if "elapsed_min" not in near.columns:
        near["elapsed_min"] = 0
    return near


def summarize_near(near: pd.DataFrame, topn: int = 30) -> pd.DataFrame:
    if near.empty:
        return pd.DataFrame(columns=["scans", "min_dist_bps", "weekly_ok", "last_seen"])

    grp = (near
           .groupby("ticker")
           .agg(scans=("ticker", "count"),
                min_dist_bps=("dist_bps", "min"),
                weekly_ok=("cond_weekly_stage_ok", "max"),
                last_seen=("elapsed_min", "max"))
           .sort_values(["min_dist_bps", "scans"], ascending=[True, False]))
    return grp.head(topn)


# ---------- Explain drill-down ----------

COND_COLUMNS = [
    "cond_weekly_stage_ok",
    "cond_rs_ok",
    "cond_ma_ok",
    "cond_pivot_ok",
    "cond_buy_vol_ok",
    "cond_pace_full_gate",
    "cond_near_pace_gate",
    "cond_buy_price_ok",
]

def build_failed_cols_row(row: pd.Series) -> List[str]:
    failed = []
    for c in COND_COLUMNS:
        if c in row.index:
            v = bool(row.get(c, False))
            if not v:
                failed.append(c)
    return failed


def explain_tickers(df: pd.DataFrame, tickers: List[str], outdir: str, tag: str) -> List[str]:
    """Create per-ticker CSVs (and small TXT) explaining which gates failed each scan."""
    artifacts = []
    for t in sorted(set([x.strip().upper() for x in tickers if x.strip()])):
        sub = df[df["ticker"].str.upper() == t].copy()
        if sub.empty:
            # still produce a stub
            stub = os.path.join(outdir, f"diag_explain_{t}_{tag}.txt")
            write_text(stub, f"No rows for ticker {t} in CSV.\n")
            artifacts.append(stub)
            continue

        # ordering by time if available
        if "elapsed_min" in sub.columns:
            sub = sub.sort_values("elapsed_min")
        sub["dist_bps"] = (sub["pivot"] - sub["price"]) / sub["pivot"] * 1e4

        # which gates failed on each row
        sub["failed_conditions"] = sub.apply(build_failed_cols_row, axis=1)
        # friendly 'ts' if present
        if "ts" in sub.columns:
            order_cols = ["ts", "elapsed_min", "price", "pivot", "dist_bps"] + COND_COLUMNS + ["failed_conditions"]
        else:
            order_cols = ["elapsed_min", "price", "pivot", "dist_bps"] + COND_COLUMNS + ["failed_conditions"]
        order_cols = [c for c in order_cols if c in sub.columns]

        csv_path = os.path.join(outdir, f"diag_explain_{t}_{tag}.csv")
        sub.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        artifacts.append(csv_path)

        # mini headline
        mini = os.path.join(outdir, f"diag_explain_{t}_{tag}.txt")
        tight = sub.loc[sub["dist_bps"].idxmin()]
        failed_set = set(tight["failed_conditions"])
        # NB: "cond_buy_price_ok" being False is expected for a near-miss,
        # but we show all gates anyway.
        write_text(
            mini,
            (
                f"Explain {t}\n"
                f"Rows: {len(sub)}\n"
                f"Tightest dist_bps: {tight['dist_bps']:.2f}\n"
                f"At elapsed_min: {int(tight.get('elapsed_min', 0))}\n"
                f"Failed gates @ tightest: {', '.join(sorted(failed_set)) if failed_set else '(none)'}\n"
                f"CSV: {os.path.basename(csv_path)}\n"
            )
        )
        artifacts.append(mini)
    return artifacts


# ---------- Summary formatting ----------

def df_to_text_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "(none)\n"
    # Light textual table (no external deps)
    cols = list(df.columns)
    # build widths
    widths = {c: max(len(c), *(len(f"{v}") for v in df[c].head(max_rows))) for c in cols}
    lines = []
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines.append(header)
    lines.append(sep)
    for _, row in df.head(max_rows).iterrows():
        lines.append(" | ".join(f"{row[c]}".ljust(widths[c]) for c in cols))
    return "\n".join(lines) + "\n"


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Weinstein intraday diagnostics")
    ap.add_argument("--csv", help="Path to intraday_debug.csv", required=False)
    ap.add_argument("--outdir", help="Output directory", required=True)
    ap.add_argument("--html-glob", help="Glob for intraday_watch_*.html", required=False)
    ap.add_argument("--state", help="Path to diag_state.json", required=False)
    ap.add_argument("--explain", help="Comma-separated tickers to drill down (e.g. MU,DDOG)", required=False)
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    tag = now_tag()

    # state
    state = load_state(args.state)

    # 1) Scrape HTML near-universe
    near_tickers, files_scanned = scrape_near_universe(args.html_glob)
    near_universe_path = os.path.join(args.outdir, "near_universe.txt")
    if near_tickers:
        write_lines(near_universe_path, near_tickers)
    else:
        write_lines(near_universe_path, [])
    # 2) CSV diagnostics
    df = load_intraday_csv(args.csv) if args.csv else None

    found_valid_csv = df is not None and not df.empty and {"ticker", "price", "pivot"}.issubset(df.columns)

    if found_valid_csv:
        near = compute_near_misses(df)
        grp = summarize_near(near)

        # Optional explain
        explain_artifacts: List[str] = []
        explain_list: List[str] = []
        if args.explain:
            explain_list = [x.strip().upper() for x in args.explain.split(",") if x.strip()]
        elif near_tickers:
            # Helpful default: explain any current near-universe tickers we also saw in CSV
            csv_names = set(df["ticker"].str.upper().unique())
            explain_list = [t for t in near_tickers if t in csv_names][:6]  # cap to keep it tidy
        if explain_list:
            explain_artifacts = explain_tickers(df, explain_list, args.outdir, tag)
    else:
        near = pd.DataFrame()
        grp = pd.DataFrame()
        explain_artifacts = []

    # 3) Write summary
    summary_path = os.path.join(args.outdir, f"diag_summary_{tag}.txt")
    lines = []
    lines.append(f"=== Weinstein Intraday Diagnostics @ {tag} ===")

    if near_tickers:
        lines.append(f"Near-universe tickers discovered in HTML: {len(near_tickers)} (saved: near_universe.txt)")
    else:
        if args.html_glob:
            lines.append(f"No near-universe tickers found in HTML glob: {args.html_glob}")
        else:
            lines.append("HTML glob not provided; skipping near-universe scrape.")

    if not found_valid_csv:
        lines.append("No valid intraday_debug.csv found or file empty.")
    else:
        lines.append("")
        lines.append("Top near-miss candidates (all green except price):")
        # render grp with reset index so 'ticker' shows as a column
        if not grp.empty:
            grp_out = grp.reset_index()
            lines.append(df_to_text_table(grp_out))
        else:
            lines.append("(none)\n")

        # quick counts on common blockers (if columns exist)
        def exists(c: str) -> bool:
            return c in df.columns

        blockers = []
        if exists("cond_pace_full_gate"):
            blockers.append(("pace_full_gate_failed",
                             (~df["cond_pace_full_gate"]).sum()))
        if exists("cond_near_pace_gate"):
            blockers.append(("near_pace_gate_failed",
                             (~df["cond_near_pace_gate"]).sum()))
        if exists("cond_buy_vol_ok"):
            blockers.append(("buy_volume_failed",
                             (~df["cond_buy_vol_ok"]).sum()))
        if exists("cond_pivot_ok"):
            blockers.append(("pivot_disqualified",
                             (~df["cond_pivot_ok"]).sum()))
        if blockers:
            lines.append("Common blockers (count of failed scans):")
            for k, v in blockers:
                lines.append(f"- {k}: {int(v)}")
            lines.append("")

    if explain_artifacts:
        lines.append("Explain artifacts:")
        for p in explain_artifacts:
            lines.append(f"- {os.path.basename(p)}")
    else:
        if found_valid_csv:
            lines.append("No explain artifacts generated.")

    write_text(summary_path, "\n".join(lines))

    # 4) Save state
    state.update({
        "last_run": tag,
        "last_csv": args.csv,
        "last_outdir": args.outdir,
        "last_html_glob": args.html_glob,
        "last_near_universe_count": len(near_tickers),
        "last_summary": summary_path,
        "last_explain": (args.explain or ""),
        "files_scanned_html": files_scanned,
    })
    save_state(args.state, state)

    # 5) Print final console note
    print("Diagnostics complete.")
    print(f"- Near-universe: {near_universe_path}")
    print(f"- Summary:      {summary_path}")


if __name__ == "__main__":
    main()
