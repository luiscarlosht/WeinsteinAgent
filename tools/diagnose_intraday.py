#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Intraday Diagnostics + Signals (merged)
------------------------------------------------
Generates a plain-text summary and helper files to understand why BUY/SELL
triggers did or did not fire, and what is currently "armed".

Inputs
------
- --csv <path>                : intraday_debug.csv snapshot (optional but recommended)
- --html-glob "<pattern>"     : glob for intraday_watch_*.html (optional; collects near-universe)
- --signals-csv <path>        : optional explicit signals log; if missing we infer BUY/SELL transitions from --csv
- --outdir <dir>              : where outputs land (summary, near_universe.txt, etc.)
- --state <path>              : JSON with rolling state across runs (armed, last buys/sells)
- --bps-threshold <int>       : "near" threshold in basis points (default: 30)
- --window-min <int>          : how far back (elapsed_min) to consider for "recent" (default: 120)
- --explain TICK1,TICK2,...   : extra per-ticker dump in the summary

Outputs
-------
- <outdir>/diag_summary_YYYYMMDD_HHMMSS.txt
- <outdir>/near_universe.txt
- <state> JSON updated

Assumptions
-----------
This is defensive about columns. If `intraday_debug.csv` is empty/missing
you’ll still get a near-universe from HTML and an empty diagnostics section.

BUY/SELL inference (when --signals-csv not provided):
- BUY: all “buy gates” true & price-gate true at some scan
- ARMED (near-NOW): all buy gates except price true AND within bps threshold from pivot
- SELL: if a column that hints a sell exists (cond_position_sell, cond_sell_trigger,
        stage4_negative_pl, etc.) we use it; otherwise we mark none.

Copyright: you
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import datetime as dt
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required (pip install pandas).")
    raise

# --------------------------- Utilities ---------------------------

NOW = lambda: dt.datetime.now()
TS = lambda d=None: (d or NOW()).strftime("%Y%m%d_%H%M%S")


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_csv(df: pd.DataFrame, path: str) -> None:
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    df.to_csv(path, mode="a", index=False, header=write_header, quoting=csv_mod.QUOTE_MINIMAL)


def bps_between(pivot: float, price: float) -> Optional[float]:
    try:
        if pivot and pivot != 0:
            return (pivot - price) / pivot * 1e4
    except Exception:
        pass
    return None


def true_series(df: pd.DataFrame, col: str) -> pd.Series:
    return (df[col].astype(str).str.lower().isin(["true", "1", "yes"])) if col in df.columns else pd.Series(False, index=df.index)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None]*len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


# --------------------------- HTML near-universe ---------------------------

NEAR_RE = re.compile(r"<ol><li><b>\d+\.</b>\s*<b>([A-Z]+)</b>\s*@")

def extract_near_universe(html_glob: Optional[str]) -> List[str]:
    if not html_glob:
        return []
    tickers = set()
    for path in glob.glob(html_glob):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            for m in NEAR_RE.finditer(text):
                tickers.add(m.group(1).strip())
        except Exception:
            continue
    return sorted(tickers)


# --------------------------- State ---------------------------

@dataclass
class SignalEvent:
    ts: str
    ticker: str
    side: str   # "BUY" or "SELL"
    reason: str

@dataclass
class RunState:
    last_run_ts: str
    armed: Dict[str, Dict]           # ticker -> info
    recent_buys: List[SignalEvent]
    recent_sells: List[SignalEvent]

    @staticmethod
    def empty() -> "RunState":
        return RunState(
            last_run_ts=TS(),
            armed={},
            recent_buys=[],
            recent_sells=[],
        )

def load_state(path: Optional[str]) -> RunState:
    if not path or not os.path.exists(path):
        return RunState.empty()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Coerce
        buys = [SignalEvent(**x) for x in data.get("recent_buys", [])]
        sells = [SignalEvent(**x) for x in data.get("recent_sells", [])]
        return RunState(
            last_run_ts=data.get("last_run_ts", TS()),
            armed=data.get("armed", {}),
            recent_buys=buys,
            recent_sells=sells,
        )
    except Exception:
        return RunState.empty()

def save_state(path: Optional[str], state: RunState) -> None:
    if not path:
        return
    payload = asdict(state)
    # Convert dataclasses inside lists
    payload["recent_buys"] = [asdict(x) for x in state.recent_buys]
    payload["recent_sells"] = [asdict(x) for x in state.recent_sells]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# --------------------------- Signals engine (embedded) ---------------------------

BUY_GATE_COLS_DEFAULT = [
    "cond_weekly_stage_ok",
    "cond_rs_ok",
    "cond_ma_ok",
    "cond_pivot_ok",
    "cond_buy_vol_ok",
    "cond_pace_full_gate",
    "cond_near_pace_gate",
]

# if a repo later adds cond_buy_price_ok, we treat it as the last gate
BUY_PRICE_COL = "cond_buy_price_ok"

# loose hints for sells if dedicated column not present
SELL_HINT_COLS = [
    "cond_position_sell",
    "position_sell",
    "cond_sell_trigger",
    "sell_trigger",
    "stage4_negative_pl",
]


@dataclass
class SignalSummary:
    armed_now: pd.DataFrame           # rows that are "armed" now
    near_miss: pd.DataFrame           # rows that were near except price (all gates except price)
    buys_detected: List[SignalEvent]
    sells_detected: List[SignalEvent]


def infer_buys_sells_from_debug(df: pd.DataFrame,
                                bps_threshold: int,
                                window_min: int) -> SignalSummary:
    """
    - Armed NOW: all buy gate cols true except price, and distance within bps_threshold.
    - Near-miss: same as above across all scans (not just latest), best min distance & counts.
    - Buys: if all buy gates including price are true at any scan (last window_min minutes).
    - Sells: if any SELL_HINT_COLS is true (or 1/yes) in last window_min minutes.
    """

    must_cols = ["ticker", "price", "pivot", "elapsed_min"]
    for c in must_cols:
        if c not in df.columns:
            # can't infer anything meaningful
            empty = pd.DataFrame(columns=["ticker"])
            return SignalSummary(empty, empty, [], [])

    # clean
    df = df.dropna(subset=["ticker", "price", "pivot"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["elapsed_min"] = pd.to_numeric(df["elapsed_min"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["pivot"] = pd.to_numeric(df["pivot"], errors="coerce")

    df["dist_bps"] = (df["pivot"] - df["price"]) / df["pivot"] * 1e4

    # Build boolean gates
    gates = {}
    for col in BUY_GATE_COLS_DEFAULT:
        gates[col] = true_series(df, col)
    buy_price_ok = true_series(df, BUY_PRICE_COL)

    # "near = all except price" through time
    all_except_price = pd.Series(True, index=df.index)
    for col, ser in gates.items():
        all_except_price = all_except_price & ser

    near_mask = all_except_price & df["dist_bps"].between(0, bps_threshold, inclusive="both")
    near = df[near_mask].copy()
    near["ts_min"] = df.loc[near.index, "elapsed_min"]

    # Aggregate per ticker (min distance, scans, last seen)
    if not near.empty:
        agg_near = (near.groupby("ticker")
                        .agg(scans=("ticker", "count"),
                             min_dist_bps=("dist_bps", "min"),
                             last_seen=("ts_min", "max"))
                        .reset_index()
                        .sort_values(["min_dist_bps", "scans"], ascending=[True, False]))
    else:
        agg_near = pd.DataFrame(columns=["ticker", "scans", "min_dist_bps", "last_seen"])

    # "armed now" = same logic but constrain to the latest bar per ticker
    # Take the maximum elapsed_min per ticker as "latest"
    latest_idx = df.groupby("ticker")["elapsed_min"].idxmax()
    latest = df.loc[latest_idx].copy()
    latest_all_except_price = pd.Series(True, index=latest.index)
    for col in BUY_GATE_COLS_DEFAULT:
        latest_all_except_price = latest_all_except_price & true_series(latest, col)

    latest["dist_bps"] = (latest["pivot"] - latest["price"]) / latest["pivot"] * 1e4
    armed_now = latest[latest_all_except_price & latest["dist_bps"].between(0, bps_threshold, inclusive="both")].copy()

    # BUYs inferred in the last window_min minutes
    window_mask = df["elapsed_min"] >= (df["elapsed_min"].max() - window_min)
    buy_all = pd.Series(True, index=df.index)
    for col in BUY_GATE_COLS_DEFAULT:
        buy_all = buy_all & true_series(df, col)
    buy_all = buy_all & buy_price_ok

    df_buy = df[window_mask & buy_all].copy()
    buys = []
    if not df_buy.empty:
        # reduce to first detection per ticker in the window (earliest elapsed_min within window)
        df_buy = df_buy.sort_values(["ticker", "elapsed_min"])
        first = df_buy.groupby("ticker").head(1)
        for _, r in first.iterrows():
            buys.append(SignalEvent(
                ts=TS(),
                ticker=str(r["ticker"]),
                side="BUY",
                reason="all buy gates incl. price true (inferred)"
            ))

    # SELLs inferred via hint columns in the last window
    sell_hint = pd.Series(False, index=df.index)
    any_hint_name = None
    for col in SELL_HINT_COLS:
        if col in df.columns:
            any_hint_name = col
            sell_hint = sell_hint | true_series(df, col)

    sells = []
    df_sell = df[window_mask & sell_hint].copy()
    if not df_sell.empty:
        df_sell = df_sell.sort_values(["ticker", "elapsed_min"])
        firsts = df_sell.groupby("ticker").head(1)
        for _, r in firsts.iterrows():
            sells.append(SignalEvent(
                ts=TS(),
                ticker=str(r["ticker"]),
                side="SELL",
                reason=f"{any_hint_name} true (inferred)" if any_hint_name else "sell hint true (inferred)"
            ))

    return SignalSummary(armed_now=armed_now,
                         near_miss=agg_near,
                         buys_detected=buys,
                         sells_detected=sells)


def read_signals_csv(path: str, recent_minutes: int = 240) -> Tuple[List[SignalEvent], List[SignalEvent]]:
    """
    Optional external signals log support.
    Expect columns: timestamp|ts, ticker, side, reason (lenient).
    """
    df = safe_read_csv(path)
    buys: List[SignalEvent] = []
    sells: List[SignalEvent] = []
    if df is None or df.empty:
        return buys, sells

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("ts") or cols.get("timestamp") or cols.get("time") or list(df.columns)[0]
    tkr_col = cols.get("ticker") or "ticker"
    side_col = cols.get("side") or "side"
    reason_col = cols.get("reason") or cols.get("msg") or None

    # Filter "recent" if ts is parseable
    def parse_ts(x):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                return dt.datetime.strptime(str(x), fmt)
            except Exception:
                continue
        return None

    df["_ts_obj"] = df[ts_col].apply(parse_ts) if ts_col in df.columns else None
    if "_ts_obj" in df.columns and df["_ts_obj"].notna().any():
        cutoff = NOW() - dt.timedelta(minutes=recent_minutes)
        df = df[df["_ts_obj"].fillna(dt.datetime(1970,1,1)) >= cutoff]

    for _, r in df.iterrows():
        side = str(r.get(side_col, "")).upper()
        ev = SignalEvent(
            ts=str(r.get(ts_col, TS())),
            ticker=str(r.get(tkr_col, "")).upper(),
            side=side,
            reason=str(r.get(reason_col, "") or "").strip(),
        )
        if side == "BUY":
            buys.append(ev)
        elif side == "SELL":
            sells.append(ev)
    return buys, sells


# --------------------------- Summary formatting ---------------------------

def fmt_table(rows: List[List[str]], colsep: str = "  ") -> str:
    if not rows:
        return ""
    widths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
    out = []
    for r in rows:
        out.append(colsep.join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def render_summary(now_ts: str,
                   csv_path: Optional[str],
                   near_universe: List[str],
                   sigs: SignalSummary,
                   state: RunState,
                   explain_list: List[str]) -> str:
    lines = []
    lines.append(f"=== Weinstein Intraday Diagnostics @ {now_ts} ===")

    if csv_path:
        df_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        lines.append(f"CSV: {'Loaded' if df_exists else 'No valid intraday_debug.csv found or file empty.'}")
    else:
        lines.append("CSV: (not provided)")

    lines.append(f"Near-universe tickers discovered in HTML: {len(near_universe)} (saved: near_universe.txt)")
    lines.append("")

    # Near-miss tickers
    lines.append("Top near-miss by ticker (min distance bps, scans, last_seen):")
    if sigs.near_miss.empty:
        lines.append("(none)")
    else:
        rows = [["Ticker", "MinDist(bps)", "Scans", "LastSeen(min)"]]
        for _, r in sigs.near_miss.iterrows():
            rows.append([
                str(r["ticker"]),
                f"{r['min_dist_bps']:.1f}",
                str(int(r["scans"])),
                str(int(r["last_seen"])),
            ])
        lines.append(fmt_table(rows))
    lines.append("")

    # Armed NOW
    lines.append("Armed now (all gates except price, within bps threshold):")
    if sigs.armed_now.empty:
        lines.append("(none)")
    else:
        rows = [["Ticker", "Price", "Pivot", "Dist(bps)", "Elapsed(min)"]]
        for _, r in sigs.armed_now.iterrows():
            rows.append([
                str(r["ticker"]),
                f"{float(r['price']):.2f}",
                f"{float(r['pivot']):.2f}",
                f"{float(r['dist_bps']):.1f}",
                str(int(r["elapsed_min"])),
            ])
        lines.append(fmt_table(rows))
    lines.append("")

    # Recent BUY/SELL
    lines.append("Recent BUY signals:")
    if not sigs.buys_detected and not state.recent_buys:
        lines.append("(none)")
    else:
        rows = [["TS", "Ticker", "Reason"]]
        for ev in (sigs.buys_detected or state.recent_buys):
            rows.append([ev.ts, ev.ticker, ev.reason or ""])
        lines.append(fmt_table(rows))
    lines.append("")

    lines.append("Recent SELL signals:")
    if not sigs.sells_detected and not state.recent_sells:
        lines.append("(none)")
    else:
        rows = [["TS", "Ticker", "Reason"]]
        for ev in (sigs.sells_detected or state.recent_sells):
            rows.append([ev.ts, ev.ticker, ev.reason or ""])
        lines.append(fmt_table(rows))
    lines.append("")

    # Explain requested tickers
    if explain_list:
        lines.append("Explain (raw gate snapshot best we can infer):")
        rows = [["Ticker", "ArmedNow?", "BestMinDist(bps)", "Scans"]]
        best = sigs.near_miss.set_index("ticker") if not sigs.near_miss.empty else pd.DataFrame()
        armed_set = set(sigs.armed_now["ticker"].unique().tolist()) if not sigs.armed_now.empty else set()
        for t in explain_list:
            t = t.upper().strip()
            m = best.loc[t] if (not best.empty and t in best.index) else None
            rows.append([
                t,
                "YES" if t in armed_set else "no",
                f"{float(m['min_dist_bps']):.1f}" if m is not None else "-",
                str(int(m["scans"])) if m is not None else "-",
            ])
        lines.append(fmt_table(rows))
        lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Weinstein intraday diagnostics + signals")
    ap.add_argument("--csv", type=str, default=None, help="Path to intraday_debug.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Directory for outputs")
    ap.add_argument("--html-glob", type=str, default=None, help="Glob for intraday_watch_*.html")
    ap.add_argument("--state", type=str, default=None, help="Path to persistent diag_state.json")
    ap.add_argument("--signals-csv", type=str, default=None, help="Optional explicit signals CSV log")
    ap.add_argument("--bps-threshold", type=int, default=30, help="Basis point threshold for 'near'")
    ap.add_argument("--window-min", type=int, default=120, help="Window (minutes) for recent signals")
    ap.add_argument("--explain", type=str, default=None, help="Comma-separated tickers to explain")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    now_ts = TS()

    # near-universe from HTML
    near_universe = extract_near_universe(args.html_glob)
    near_txt = os.path.join(args.outdir, "near_universe.txt")
    write_text(near_txt, "\n".join(near_universe) + ("\n" if near_universe else ""))

    # load state
    state = load_state(args.state)

    # load CSV
    df = safe_read_csv(args.csv) if args.csv else None

    # signals detection
    if df is None:
        # no debug csv -> only near-universe section
        sigs = SignalSummary(
            armed_now=pd.DataFrame(columns=["ticker", "price", "pivot", "dist_bps", "elapsed_min"]),
            near_miss=pd.DataFrame(columns=["ticker", "scans", "min_dist_bps", "last_seen"]),
            buys_detected=[],
            sells_detected=[]
        )
    else:
        sigs = infer_buys_sells_from_debug(df, bps_threshold=args.bps_threshold, window_min=args.window_min)

    # merge explicit signals csv (if any): explicit overrides inference
    if args.signals_csv and os.path.exists(args.signals_csv):
        buys_csv, sells_csv = read_signals_csv(args.signals_csv, recent_minutes=args.window_min)
        if buys_csv:
            sigs.buys_detected = buys_csv
        if sells_csv:
            sigs.sells_detected = sells_csv

    # update state
    state.last_run_ts = now_ts
    # armed snapshot saved as dict (ticker -> info)
    new_armed = {}
    if not sigs.armed_now.empty:
        for _, r in sigs.armed_now.iterrows():
            new_armed[str(r["ticker"])] = {
                "price": float(r["price"]),
                "pivot": float(r["pivot"]),
                "dist_bps": float(r["dist_bps"]),
                "elapsed_min": int(r["elapsed_min"]),
                "ts": now_ts,
            }
    state.armed = new_armed

    # Keep only *this run's* recent buys/sells (don’t endlessly grow)
    state.recent_buys = sigs.buys_detected
    state.recent_sells = sigs.sells_detected

    # explain list
    explain_list = []
    if args.explain:
        explain_list = [x.strip() for x in args.explain.split(",") if x.strip()]

    # summary
    summary = render_summary(
        now_ts=now_ts,
        csv_path=args.csv,
        near_universe=near_universe,
        sigs=sigs,
        state=state,
        explain_list=explain_list
    )
    out_summary = os.path.join(args.outdir, f"diag_summary_{now_ts}.txt")
    write_text(out_summary, summary)

    # persist state
    save_state(args.state, state)

    # stdout footer
    print("Diagnostics complete.")
    print(f"- Near-universe: {near_txt}")
    print(f"- Summary:      {out_summary}")


if __name__ == "__main__":
    main()
