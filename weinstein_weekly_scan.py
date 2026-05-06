#!/usr/bin/env python3
# weinstein_weekly_scan.py
#
# “Classic” Weinstein scan block for the weekly pipeline.
#
# Instead of shelling out to a separate scanner, this script:
#   - Finds the latest output/weinstein_weekly_equities_*.csv
#   - Ranks tickers (Stage 2 first, then by distance above MA30, then RS slope)
#   - Writes:
#       * output/scan_<universe>.csv   (top N rows)
#       * output/scan_<universe>.html  (HTML table + summary header)
#   - Returns a small JSON blob when called with --json (for run_weekly.sh)
#
# This avoids the old, broken call:
#   update_signals_and_score.py --universe ... --benchmark ... --write-...
# which does not match that script’s CLI.

import argparse
import json
import os
import glob
from datetime import datetime

import pandas as pd


def _stage_rank_value(stage) -> int:
    """Map stage label to a numeric rank: lower is better.

    Handles pandas NaN/None safely; Fidelity/Yahoo-derived weekly outputs can
    contain blank stage values, which pandas reads as float NaN.
    """
    if stage is None or pd.isna(stage):
        s = ""
    else:
        s = str(stage).strip()
    if s.startswith("Stage 2"):   # Uptrend
        return 0
    if s.startswith("Stage 1"):   # Basing
        return 1
    if s.startswith("Stage 3"):   # Topping
        return 2
    if s.startswith("Stage 4"):   # Downtrend
        return 3
    if s == "Filtered":
        return 8
    if s == "N/A":
        return 9
    return 9


def _find_latest_weekly_equities_csv(out_dir: str) -> str | None:
    """Return the path to the most recent weinstein_weekly_equities_*.csv, or None."""
    pattern = os.path.join(out_dir, "weinstein_weekly_equities_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def run_scan(universe: str, benchmark: str, out_dir: str, max_rows_email: int = 200) -> dict:
    """
    Build a “classic scan” view using the latest weekly equities CSV.

    Returns:
      {
        "ok": True/False,
        "html_block": "<html snip>" or "",
        "csv_path": ".../scan_<universe>.csv" or "",
        "html_path": ".../scan_<universe>.html" or "",
        "summary_line": "Summary: ...",
        "error": "..."  # only if ok=False
      }
    """
    os.makedirs(out_dir, exist_ok=True)

    latest_csv = _find_latest_weekly_equities_csv(out_dir)
    if latest_csv is None:
        return {
            "ok": False,
            "html_block": "",
            "csv_path": "",
            "html_path": "",
            "summary_line": "",
            "error": f"No weinstein_weekly_equities_*.csv found under {out_dir}",
        }

    try:
        df = pd.read_csv(latest_csv)
    except Exception as e:
        return {
            "ok": False,
            "html_block": "",
            "csv_path": "",
            "html_path": "",
            "summary_line": "",
            "error": f"Could not read {latest_csv}: {e}",
        }

    if df.empty or "ticker" not in df.columns:
        return {
            "ok": False,
            "html_block": "",
            "csv_path": "",
            "html_path": "",
            "summary_line": "",
            "error": f"Weekly equities CSV {latest_csv} is empty or missing 'ticker' column.",
        }

    # Prefer equities only, if asset_class exists.
    if "asset_class" in df.columns:
        eq_mask = df["asset_class"].astype(str).str.contains("Equity", case=False, na=False)
        if eq_mask.any():
            df = df.loc[eq_mask].copy()

    # Ensure needed columns exist
    for col in ["stage", "buy_signal", "dist_ma_pct", "rs_slope_per_wk"]:
        if col not in df.columns:
            df[col] = None

    # Rank: Stage 2 first, then higher distance above MA30, then RS slope
    df["__stage_rank"] = df["stage"].map(_stage_rank_value)
    df_sorted = df.sort_values(
        by=["__stage_rank", "dist_ma_pct", "rs_slope_per_wk"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    trimmed = df_sorted.head(int(max_rows_email)).copy()

    # Basic counts
    buy_signal = df["buy_signal"].fillna("").astype(str).str.upper().str.strip()
    buy_count = int((buy_signal == "BUY").sum())
    watch_count = int((buy_signal == "WATCH").sum())
    avoid_count = int((buy_signal == "AVOID").sum())
    total = int(len(df))

    summary_line = (
        f"Summary: ✅ Buy: {buy_count} | 🟡 Watch: {watch_count} | "
        f"🔴 Avoid: {avoid_count} (Total: {total})"
    )

    # Output paths
    csv_path = os.path.join(out_dir, f"scan_{universe}.csv")
    html_path = os.path.join(out_dir, f"scan_{universe}.html")

    # Save trimmed CSV
    trimmed.drop(columns=["__stage_rank"], errors="ignore").to_csv(csv_path, index=False)

    # Choose a reasonable set of columns for HTML
    cols = [
        "ticker",
        "stage",
        "buy_signal",
        "price",
        "dist_ma_pct",
        "ma_slope_per_wk",
        "rs_ma30",
        "rs_slope_per_wk",
        "short_term_state_wk",
        "notes",
    ]
    # Keep only existing columns, and in that order.
    use_cols = [c for c in cols if c in trimmed.columns]
    if not use_cols:
        use_cols = list(trimmed.columns.drop("__stage_rank"))

    tbl = trimmed.drop(columns=["__stage_rank"], errors="ignore")
    html_table = tbl[use_cols].to_html(index=False, border=0, escape=False)

    # Build HTML header
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"""
    <h2 style="margin:24px 0 4px;">Weinstein Weekly — Classic Scan</h2>
    <div style="color:#555;margin:0 0 8px;">
      Universe: {universe} &nbsp;|&nbsp; Benchmark: {benchmark} &nbsp;|&nbsp;
      Generated {ts}
    </div>
    <div style="margin:0 0 16px;">{summary_line}</div>
    """

    html_block = header + html_table

    # Write HTML to file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_block)

    return {
        "ok": True,
        "html_block": html_block,
        "csv_path": csv_path,
        "html_path": html_path,
        "summary_line": summary_line,
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--out-dir", default="./output")
    ap.add_argument("--max-rows-email", type=int, default=200)
    ap.add_argument("--json", action="store_true", help="print JSON result to stdout")
    args = ap.parse_args()

    res = run_scan(args.universe, args.benchmark, args.out_dir, args.max_rows_email)
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        if not res.get("ok"):
            print(res.get("error", "scan failed"), file=sys.stderr)
            raise SystemExit(1)
        print("Scan complete:", res.get("summary_line", "(no summary)"))


if __name__ == "__main__":
    main()
