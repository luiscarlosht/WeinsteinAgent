#!/usr/bin/env python3
"""
short_signal_engine.py

Summarizes short-side diagnostics from weinstein_short_watcher.py AND
adds full short-exit tracking:

✔ Detect short TRIGGER events
✔ Open synthetic short positions
✔ Calculate stop, target1, target2 (same formulas as watcher)
✔ Detect when shorts are READY to CLOSE (T1/T2)
✔ Detect STOP HIT (adverse move)
✔ Timestamp every event to short_exits_log.csv
✔ Persist open state in short_exits_state.json

Fully standalone and safe to run every scan.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import json
import math

import numpy as np
import pandas as pd


# ----------------------------
# Logging helpers
# ----------------------------
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_info(msg: str) -> None:
    print(f"• [{_now_str()}] {msg}")

def log_ok(msg: str) -> None:
    print(f"✅ [{_now_str()}] {msg}")

def log_warn(msg: str) -> None:
    print(f"⚠️ [{_now_str()}] {msg}")


# ----------------------------
# Short-exit tunables
# ----------------------------
SHORT_HARD_STOP_PCT   = 0.20   # 20% above entry
SHORT_TRAIL_ATR_MULT  = 2.0
SHORT_MA_GUARD_PCT    = 0.03   # 3% above MA150
SHORT_TARGET1_PCT     = 0.15   # 15% downside
SHORT_TARGET2_PCT     = 0.20   # 20% downside


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize short diagnostics AND detect ready-to-close short exits."
    )
    p.add_argument("--csv", required=True, help="short_debug.csv from watcher")
    p.add_argument("--outdir", default="./output")
    p.add_argument(
        "--window-min",
        type=int,
        default=120,
        help="Only consider rows where elapsed_min <= this (0 = all rows)",
    )
    # accept both for compatibility with existing wrapper
    p.add_argument(
        "--bps",
        "--bps-threshold",
        dest="bps",
        type=int,
        default=0,
        help="Reserved / unused (basis point threshold placeholder).",
    )
    p.add_argument("--explain", metavar="TICKER")
    return p.parse_args(argv)


# ----------------------------
# CSV loading
# ----------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        log_warn(f"CSV {path} is empty.")
    return df


def apply_window(df: pd.DataFrame, window_min: int) -> pd.DataFrame:
    if window_min <= 0:
        return df
    if "elapsed_min" not in df.columns:
        log_warn("CSV missing elapsed_min — no window filtering applied.")
        return df
    df2 = df[df["elapsed_min"] <= window_min]
    if df2.empty and not df.empty:
        log_warn("CSV had rows, but none in window.")
    return df2


# ----------------------------
# Summaries
# ----------------------------
def infer_hits(df_t: pd.DataFrame):
    """Compute near_hits, trig_hits, last_state, last_px."""
    last = df_t.iloc[-1]

    last_state = str(last.get("short_state", "NA"))
    last_px = float(last.get("price", np.nan))

    if "cond_short_near_now" in df_t.columns:
        near_hits = int(df_t["cond_short_near_now"].fillna(False).astype(bool).sum())
    else:
        near_hits = int((df_t.get("short_state", "") == "NEAR").sum())

    trig_col = None
    if "cond_short_confirm" in df_t.columns:
        trig_col = "cond_short_confirm"
    elif "cond_short_trig_now" in df_t.columns:
        trig_col = "cond_short_trig_now"

    if trig_col:
        trig_hits = int(df_t[trig_col].fillna(False).astype(bool).sum())
    else:
        trig_hits = int((df_t.get("short_state", "") == "TRIGGERED").sum())

    return near_hits, trig_hits, last_state, last_px


def summarize(df: pd.DataFrame, outdir: Path, window_min: int) -> pd.DataFrame:
    if df.empty:
        log_warn("No rows to summarize.")
        return pd.DataFrame()

    ticker_col = None
    for col in ("ticker", "symbol", "Ticker", "Symbol"):
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        raise KeyError("No ticker column found.")

    rows = []
    for t, df_t in df.groupby(ticker_col):
        near_hits, trig_hits, last_state, last_px = infer_hits(df_t)
        rows.append(
            {
                "ticker": t,
                "ShortState": last_state,
                "NearHits": near_hits,
                "TrigHits": trig_hits,
                "LastPx": last_px,
            }
        )

    df_sum = pd.DataFrame(rows).sort_values(
        ["TrigHits", "NearHits", "ticker"],
        ascending=[False, False, True],
    )
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"short_signal_summary_{window_min if window_min > 0 else 'full'}min.csv"
    df_sum.to_csv(out, index=False)
    log_ok(f"Wrote short signal summary → {out}")
    return df_sum.reset_index(drop=True)


def print_table(df_sum: pd.DataFrame):
    if df_sum.empty:
        print("No short signals.")
        return

    hdr = f"{'Ticker':<6} {'ShortState':<10} {'NearHits':>8} {'TrigHits':>8} {'LastPx':>10}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df_sum.iterrows():
        print(
            f"{r['ticker']:<6} "
            f"{r['ShortState']:<10} "
            f"{int(r['NearHits']):>8} "
            f"{int(r['TrigHits']):>8} "
            f"{r['LastPx']:>10.2f}"
        )


# ----------------------------
# NEW: Short Exit Engine
# ----------------------------
def load_exit_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}

def save_exit_state(path: Path, st: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(st, f, indent=2)

def append_exit_event(path: Path, row: dict):
    header = [
        "ts",
        "ticker",
        "event",
        "price",
        "entry",
        "stop",
        "target1",
        "target2",
        "atr",
        "ma30",
        "opened_at",
    ]
    need_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=header)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def compute_entry_stop_targets(entry_px: float, ma30: float, atr: float):
    """Same formulas as in weinstein_short_watcher."""
    if entry_px is None or math.isnan(entry_px):
        return math.nan, math.nan, math.nan, math.nan

    entry = float(entry_px)

    hard = entry * (1 + SHORT_HARD_STOP_PCT)
    atr_stop = (
        entry + SHORT_TRAIL_ATR_MULT * atr
        if atr is not None and not math.isnan(atr)
        else math.nan
    )
    ma_stop = (
        ma30 * (1 + SHORT_MA_GUARD_PCT)
        if ma30 is not None and not math.isnan(ma30)
        else math.nan
    )

    candidates = [c for c in (hard, atr_stop, ma_stop) if not math.isnan(c)]
    stop = max(candidates or [hard])

    t1 = entry * (1 - SHORT_TARGET1_PCT)
    t2 = entry * (1 - SHORT_TARGET2_PCT)

    return entry, stop, t1, t2


def track_short_exits(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return

    ticker_col = None
    for col in ("ticker", "symbol", "Ticker", "Symbol"):
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        log_warn("No ticker column in short_debug CSV — cannot track exits.")
        return

    state_path = outdir / "short_exits_state.json"
    log_path = outdir / "short_exits_log.csv"
    exit_state = load_exit_state(state_path)

    for t, df_t in df.groupby(ticker_col):
        t = str(t).upper()
        last = df_t.iloc[-1]

        px = float(last.get("price", math.nan))
        ma30 = float(last.get("ma30", math.nan)) if "ma30" in last else math.nan
        atr = float(last.get("atr", math.nan)) if "atr" in last else math.nan
        short_state = str(last.get("short_state", "NA"))
        trig_flag = bool(last.get("cond_short_confirm", False))

        st = exit_state.get(
            t,
            {
                "open": False,
                "entry": math.nan,
                "stop": math.nan,
                "t1": math.nan,
                "t2": math.nan,
                "atr": math.nan,
                "ma30": math.nan,
                "opened_at": "",
                "hit_t1": False,
                "hit_t2": False,
                "stopped": False,
            },
        )

        now = _now_str()

        # OPEN SHORT position
        if not st["open"] and (short_state == "TRIGGERED" or trig_flag):
            if not math.isnan(px):
                entry, stop, t1, t2 = compute_entry_stop_targets(px, ma30, atr)
                st.update(
                    {
                        "open": True,
                        "entry": entry,
                        "stop": stop,
                        "t1": t1,
                        "t2": t2,
                        "atr": atr,
                        "ma30": ma30,
                        "opened_at": now,
                        "hit_t1": False,
                        "hit_t2": False,
                        "stopped": False,
                    }
                )
                exit_state[t] = st
                append_exit_event(
                    log_path,
                    {
                        "ts": now,
                        "ticker": t,
                        "event": "OPEN_SHORT",
                        "price": px,
                        "entry": entry,
                        "stop": stop,
                        "target1": t1,
                        "target2": t2,
                        "atr": atr,
                        "ma30": ma30,
                        "opened_at": now,
                    },
                )
                log_ok(
                    f"{t} SHORT opened at {entry:.2f} "
                    f"(T1={t1:.2f}, T2={t2:.2f}, Stop={stop:.2f})"
                )
                continue

        # No open short or no valid price
        if not st["open"] or math.isnan(px):
            exit_state[t] = st
            continue

        # T1 hit
        if not st["hit_t1"] and px <= st["t1"]:
            st["hit_t1"] = True
            append_exit_event(
                log_path,
                {
                    "ts": now,
                    "ticker": t,
                    "event": "READY_CLOSE_T1",
                    "price": px,
                    "entry": st["entry"],
                    "stop": st["stop"],
                    "target1": st["t1"],
                    "target2": st["t2"],
                    "atr": st["atr"],
                    "ma30": st["ma30"],
                    "opened_at": st["opened_at"],
                },
            )
            log_ok(
                f"{t} SHORT ready-to-close (T1) — "
                f"entry {st['entry']:.2f}, now {px:.2f}"
            )

        # T2 hit
        if not st["hit_t2"] and px <= st["t2"]:
            st["hit_t2"] = True
            append_exit_event(
                log_path,
                {
                    "ts": now,
                    "ticker": t,
                    "event": "READY_CLOSE_T2",
                    "price": px,
                    "entry": st["entry"],
                    "stop": st["stop"],
                    "target1": st["t1"],
                    "target2": st["t2"],
                    "atr": st["atr"],
                    "ma30": st["ma30"],
                    "opened_at": st["opened_at"],
                },
            )
            log_ok(
                f"{t} SHORT ready-to-close (T2) — "
                f"entry {st['entry']:.2f}, now {px:.2f}"
            )

        # STOP hit
        if not st["stopped"] and px >= st["stop"]:
            st["stopped"] = True
            append_exit_event(
                log_path,
                {
                    "ts": now,
                    "ticker": t,
                    "event": "STOP_HIT",
                    "price": px,
                    "entry": st["entry"],
                    "stop": st["stop"],
                    "target1": st["t1"],
                    "target2": st["t2"],
                    "atr": st["atr"],
                    "ma30": st["ma30"],
                    "opened_at": st["opened_at"],
                },
            )
            log_warn(
                f"{t} SHORT STOP HIT — "
                f"entry {st['entry']:.2f}, now {px:.2f}"
            )

        # Auto-close flag when T2 or STOP
        if st["hit_t2"] or st["stopped"]:
            st["open"] = False

        exit_state[t] = st

    save_exit_state(state_path, exit_state)


# ----------------------------
# Main
# ----------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    log_info(f"Loading short debug CSV: {csv_path}")
    try:
        df_all = load_csv(csv_path)
    except Exception as e:
        log_warn(f"Failed to load CSV: {e}")
        return 1

    if df_all.empty:
        log_warn("Aborting: CSV empty.")
        return 0

    df_win = apply_window(df_all, args.window_min)
    if df_win.empty:
        return 0

    df_sum = summarize(df_win, outdir, args.window_min)
    print()
    print_table(df_sum)
    print()

    # NEW: track short exit events (OPEN / READY_CLOSE_T1 / READY_CLOSE_T2 / STOP_HIT)
    track_short_exits(df_win, outdir)

    if args.explain:
        ticker = args.explain.upper()
        log_info(f"Explaining {ticker} (raw rows within window):")
        print(df_win[df_win["ticker"] == ticker])

    return 0


if __name__ == "__main__":
    sys.exit(main())
