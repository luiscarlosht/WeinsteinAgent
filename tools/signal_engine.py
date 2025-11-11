#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Engine — promotes intraday "near" conditions into confirmed BUY/SELL signals
and logs them with stateful debouncing. Designed to run right after your intraday
watcher (every 10 minutes).

Key features
- Robust CSV read: handles empty/1-byte files and partial writes.
- Uses the same boolean gates produced by the watcher (`cond_*`) but tolerates
  missing columns by falling back to derived metrics when possible.
- Stateful near->armed->triggered progression with cooldowns (BUY & SELL).
- Appends confirmed events to a signals CSV for later analysis.
- Optionally extracts the "near universe" tickers from generated HTML so
  diagnostics and the engine share the same set.

Typical flow
1) weinstein_intraday_watcher.py runs and writes ./output/intraday_debug.csv + HTML.
2) signal_engine.py ingests that CSV, updates per-ticker state, and appends confirmed
   signals to ./output/signals_log.csv.
3) diagnose_intraday.py (optional) summarizes how close things were and what missed.

Author: ChatGPT (for Luis Carlos Hernandez)
"""

import os
import re
import csv
import json
import math
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np

# ---------------- Tunables (mirror watcher defaults) ----------------
SCAN_INTERVAL_MIN = 10

# BUY progression
NEAR_HITS_WINDOW = 6        # how many scans to keep in rolling window
NEAR_HITS_MIN = 3           # #hits in window to ARM
COOLDOWN_SCANS = 24         # after trigger, how long to cool down

# SELL progression
SELL_NEAR_HITS_WINDOW = 6
SELL_NEAR_HITS_MIN = 3
SELL_COOLDOWN_SCANS = 24

# Intrabar easing gates (only used if CSV provides elapsed pace info)
INTRABAR_CONFIRM_MIN_ELAPSED = 40
INTRABAR_VOLPACE_MIN = 1.20
SELL_INTRABAR_CONFIRM_MIN_ELAPSED = 40
SELL_INTRABAR_VOLPACE_MIN = 1.20

# CSV column expectations (we degrade gracefully if missing)
EXPECTED_BOOL_COLS = [
    "cond_weekly_stage_ok",
    "cond_rs_ok",
    "cond_ma_ok",
    "cond_pivot_ok",
    "cond_buy_vol_ok",
    "cond_pace_full_gate",
    "cond_near_pace_gate",
    "cond_buy_price_ok",
    "cond_near_now",
    "sell_near_now",
    "sell_price_ok",
    "sell_vol_ok",
    "sell_confirm",
    "buy_confirm",
]
EXPECTED_NUM_COLS = [
    "price",
    "pivot",
    "ma30",
    "elapsed_min",
    "pace_intrabar",
    "pace_full_vs50dma",
    "dist_bps",
]

# ---------------- Utilities ----------------
def ts_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, level="info"):
    prefix = {"info":"•", "ok":"✅", "step":"▶️", "warn":"⚠️", "err":"❌", "debug":"··"}.get(level, "•")
    print(f"{prefix} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_state(path: str) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(path))
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(path: str, st: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(st, f, indent=2)

def update_hits(window_arr: List[int], hit: bool, window: int) -> Tuple[List[int], int]:
    window_arr = (window_arr or [])
    window_arr.append(1 if hit else 0)
    if len(window_arr) > window:
        window_arr = window_arr[-window:]
    return window_arr, sum(window_arr)

def coerce_bool(series: pd.Series) -> pd.Series:
    # Accepts True/False, 0/1, "true"/"false", etc.
    def _co(v):
        if isinstance(v, (bool, np.bool_)): return bool(v)
        s = str(v).strip().lower()
        if s in ("1","true","t","yes","y"): return True
        if s in ("0","false","f","no","n"): return False
        return False
    return series.apply(_co)

# ---------------- HTML near-universe extraction ----------------
NEAR_UNIV_RE = re.compile(r"<ol><li><b>\d+\.</b>\s*<b>([A-Z\-\.]+)</b>\s*@", re.IGNORECASE)

def extract_near_universe_from_html(glob_patterns: List[str]) -> List[str]:
    import glob
    tickers = set()
    for pat in glob_patterns:
        for path in glob.glob(pat):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                for m in NEAR_UNIV_RE.finditer(txt):
                    sym = m.group(1).strip().upper()
                    if sym:
                        tickers.add(sym)
            except Exception:
                continue
    return sorted(tickers)

def write_near_universe(outdir: str, tickers: List[str]) -> str:
    ensure_dir(outdir)
    path = os.path.join(outdir, "near_universe.txt")
    with open(path, "w") as f:
        for t in tickers:
            f.write(t + "\n")
        f.write("\n" * 50)  # match your earlier formatting
    return path

# ---------------- CSV ingest (robust) ----------------
def load_intraday_debug(csv_path: str) -> Optional[pd.DataFrame]:
    """Return DataFrame or None if file missing/empty/unreadable."""
    if not os.path.exists(csv_path):
        log(f"CSV not found: {csv_path}", level="warn")
        return None
    if os.path.getsize(csv_path) < 5:
        # 1-byte or very small -> treat as empty
        log("CSV exists but is empty/partial; nothing to do this scan.", level="info")
        return None
    try:
        # Try fast read
        df = pd.read_csv(csv_path)
    except Exception:
        try:
            # Fallback parser
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        except Exception as e2:
            log(f"Failed to read CSV: {e2}", level="err")
            return None

    if df is None or df.empty:
        log("CSV loaded but contains no rows.", level="info")
        return None

    # Normalize expected columns if present
    df.columns = [c.strip() for c in df.columns]
    for c in EXPECTED_BOOL_COLS:
        if c in df.columns:
            df[c] = coerce_bool(df[c])
    for c in EXPECTED_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Required core columns
    for core in ["ticker", "price"]:
        if core not in df.columns:
            df[core] = np.nan

    # Drop rows without ticker or price
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["ticker"])
    return df

# ---------------- Promotion engine ----------------
def default_ticker_state() -> Dict[str, Any]:
    return {
        "buy_state": "IDLE",
        "near_hits": [],
        "cooldown": 0,
        "sell_state": "IDLE",
        "sell_hits": [],
        "sell_cooldown": 0,
        "last_price": None,
        "last_seen": None,
    }

def derive_near_flags(row: pd.Series, bps_threshold: int) -> Tuple[bool, bool]:
    """
    Try to infer near flags if watcher didn't write them.
    - BUY near: all core gates except price, and within threshold bps below pivot
    - SELL near: within SELL band around MA
    """
    # Inputs (may be NaN)
    px = safe_float(row.get("price"))
    pivot = safe_float(row.get("pivot"))
    ma = safe_float(row.get("ma30"))

    weekly_ok = bool(row.get("cond_weekly_stage_ok", False))
    rs_ok = bool(row.get("cond_rs_ok", False))
    ma_ok = bool(row.get("cond_ma_ok", False))
    pivot_ok = bool(row.get("cond_pivot_ok", False))

    near_buy = False
    if weekly_ok and rs_ok and ma_ok and pivot_ok and (not math.isnan(px)) and (not math.isnan(pivot)) and pivot > 0:
        dist_bps = (pivot - px) / pivot * 1e4
        if dist_bps >= 0 and dist_bps <= bps_threshold:
            near_buy = True

    # SELL near band: |px - ma| <= small % around MA (approx via bps_threshold as well)
    near_sell = False
    if ma_ok and (not math.isnan(px)) and (not math.isnan(ma)) and ma > 0:
        dist_bps_ma = abs(px - ma) / ma * 1e4
        if dist_bps_ma <= bps_threshold:
            near_sell = True

    return near_buy, near_sell

def should_buy_confirm(row: pd.Series) -> bool:
    """
    Confirm BUY if watcher says buy_confirm; otherwise approximate with buy_price_ok AND
    (for 60m) elapsed/pace gating if present.
    """
    if bool(row.get("buy_confirm", False)):
        return True
    if bool(row.get("cond_buy_price_ok", False)):
        # If 60m intrabar context is present, respect it; otherwise assume ok.
        elapsed = row.get("elapsed_min", np.nan)
        pace_intra = row.get("pace_intrabar", np.nan)
        if pd.notna(elapsed) and pd.notna(pace_intra):
            return (elapsed >= INTRABAR_CONFIRM_MIN_ELAPSED) and (pace_intra >= INTRABAR_VOLPACE_MIN)
        return True
    return False

def should_sell_confirm(row: pd.Series) -> bool:
    """
    Confirm SELL if watcher says sell_confirm; otherwise approximate with sell_price_ok AND
    (for 60m) elapsed/pace gating if present.
    """
    if bool(row.get("sell_confirm", False)):
        return True
    if bool(row.get("sell_price_ok", False)):
        elapsed = row.get("elapsed_min", np.nan)
        pace_intra = row.get("pace_intrabar", np.nan)
        if pd.notna(elapsed) and pd.notna(pace_intra):
            return (elapsed >= SELL_INTRABAR_CONFIRM_MIN_ELAPSED) and (pace_intra >= SELL_INTRABAR_VOLPACE_MIN)
        return True
    return False

def append_signal_row(path: str, row: Dict[str, Any], write_header: Optional[bool] = None):
    ensure_dir(os.path.dirname(path))
    header = ["ts", "ticker", "side", "price", "reason", "near_hits", "state_before", "state_after"]
    need_header = write_header if write_header is not None else (not os.path.exists(path) or os.path.getsize(path) == 0)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to intraday_debug.csv from watcher")
    ap.add_argument("--outdir", default="./output")
    ap.add_argument("--state", default="./output/diag_state.json")
    ap.add_argument("--write-signals", default="./output/signals_log.csv", help="Append confirmed signals here")
    ap.add_argument("--html-glob", default="", help='Glob(s) for intraday HTML (e.g. "./output/intraday_watch_*.html"), comma-separated')
    ap.add_argument("--bps-threshold", type=int, default=35, help="Near distance threshold (basis points) when inferring near flags")
    ap.add_argument("--window-min", type=int, default=120, help="Only consider rows within this many minutes (elapsed_min<=window)")
    ap.add_argument("--explain", default="", help="Comma list of tickers to print a one-liner status for")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        def quiet_log(*_a, **_k): pass
        globals()["log"] = lambda *_a, **_k: None  # type: ignore

    ensure_dir(args.outdir)

    # 1) Optional: refresh near-universe from HTMLs
    near_tick_file = ""
    if args.html_glob.strip():
        pats = [s.strip() for s in args.html_glob.split(",") if s.strip()]
        near_tickers = extract_near_universe_from_html(pats)
        near_tick_file = write_near_universe(args.outdir, near_tickers)

    # 2) Load CSV (robustly)
    df = load_intraday_debug(args.csv)
    if df is None:
        log("No valid rows this scan. Exiting gracefully.", level="info")
        if near_tick_file:
            log(f"Near-universe saved: {near_tick_file}", level="ok")
        return

    # Optional recent-window filter
    if "elapsed_min" in df.columns and args.window_min and args.window_min > 0:
        df = df[pd.to_numeric(df["elapsed_min"], errors="coerce").fillna(0) <= args.window_min].copy()

    if df.empty:
        log("CSV had rows, but none within the requested window.", level="info")
        return

    # 3) Promote states
    state = load_state(args.state) or {}
    buy_triggers: List[Dict[str, Any]] = []
    sell_triggers: List[Dict[str, Any]] = []

    # Process by last seen per ticker (keep the last row for each)
    # If there are many rows per ticker, sort by elapsed_min asc so the *latest bar* (largest elapsed) ends last.
    if "elapsed_min" in df.columns:
        df = df.sort_values(["ticker", "elapsed_min"]).groupby("ticker", as_index=False).last()
    else:
        df = df.groupby("ticker", as_index=False).last()

    for _, r in df.iterrows():
        t = str(r["ticker"]).upper().strip()
        if not t:
            continue

        st = state.get(t, default_ticker_state())
        prev_buy = st.get("buy_state", "IDLE")
        prev_sell = st.get("sell_state", "IDLE")

        px = safe_float(r.get("price"))
        st["last_price"] = px
        st["last_seen"]  = ts_now()

        # --- Determine NEAR flags (prefer watcher output; fallback to inference)
        near_buy = bool(r.get("cond_near_now", False))
        near_sell = bool(r.get("sell_near_now", False))
        if not near_buy or not near_sell:
            d_buy, d_sell = derive_near_flags(r, args.bps_threshold)
            near_buy = near_buy or d_buy
            near_sell = near_sell or d_sell

        # --- BUY progression
        st["near_hits"], near_count = update_hits(st.get("near_hits", []), near_buy, NEAR_HITS_WINDOW)
        st["cooldown"] = max(0, int(st.get("cooldown", 0)))
        buy_state = st.get("buy_state", "IDLE")

        if buy_state == "IDLE" and near_buy:
            buy_state = "NEAR"
        elif buy_state in ("IDLE", "NEAR") and near_count >= NEAR_HITS_MIN:
            buy_state = "ARMED"

        # Confirm BUY?
        buy_confirm = should_buy_confirm(r)
        pace_full_ok = bool(r.get("cond_pace_full_gate", True))  # default permissive if missing
        if buy_state == "ARMED" and buy_confirm and pace_full_ok:
            # TRIGGER
            new_state = "TRIGGERED"
            buy_triggers.append({
                "ts": ts_now(),
                "ticker": t,
                "side": "BUY",
                "price": px if not math.isnan(px) else "",
                "reason": "confirm_over_pivot_ma",
                "near_hits": near_count,
                "state_before": prev_buy,
                "state_after": new_state,
            })
            buy_state = "COOLDOWN"
            st["cooldown"] = COOLDOWN_SCANS
            st["near_hits"] = []  # reset
        elif buy_state == "TRIGGERED":
            # stay triggered only momentarily; move to cooldown next scan
            buy_state = "COOLDOWN"
        elif st["cooldown"] > 0 and not near_buy:
            buy_state = "COOLDOWN"
            st["cooldown"] = st["cooldown"] - 1
        elif st["cooldown"] == 0 and not near_buy and not buy_confirm:
            buy_state = "IDLE"

        st["buy_state"] = buy_state

        # --- SELL progression
        st["sell_hits"], sell_hit_count = update_hits(st.get("sell_hits", []), near_sell, SELL_NEAR_HITS_WINDOW)
        st["sell_cooldown"] = max(0, int(st.get("sell_cooldown", 0)))
        sell_state = st.get("sell_state", "IDLE")

        if sell_state == "IDLE" and near_sell:
            sell_state = "NEAR"
        elif sell_state in ("IDLE", "NEAR") and sell_hit_count >= SELL_NEAR_HITS_MIN:
            sell_state = "ARMED"

        sell_confirm = should_sell_confirm(r)
        if sell_state == "ARMED" and sell_confirm:
            new_state = "TRIGGERED"
            sell_triggers.append({
                "ts": ts_now(),
                "ticker": t,
                "side": "SELL",
                "price": px if not math.isnan(px) else "",
                "reason": "confirmed_below_ma",
                "near_hits": sell_hit_count,
                "state_before": prev_sell,
                "state_after": new_state,
            })
            sell_state = "COOLDOWN"
            st["sell_cooldown"] = SELL_COOLDOWN_SCANS
            st["sell_hits"] = []
        elif sell_state == "TRIGGERED":
            sell_state = "COOLDOWN"
        elif st["sell_cooldown"] > 0 and not near_sell:
            sell_state = "COOLDOWN"
            st["sell_cooldown"] = st["sell_cooldown"] - 1
        elif st["sell_cooldown"] == 0 and not near_sell and not sell_confirm:
            sell_state = "IDLE"

        st["sell_state"] = sell_state

        state[t] = st

    # 4) Write signals & state
    wrote_any = False
    if args.write_signals:
        for s in buy_triggers + sell_triggers:
            append_signal_row(args.write_signals, s)
            wrote_any = True

    save_state(args.state, state)

    # 5) Explain summary
    if args.explain.strip():
        want = [x.strip().upper() for x in args.explain.split(",") if x.strip()]
        hdr = f"{'Ticker':6} {'BuyState':10} {'SellState':10} {'NearHits':8} {'SellHits':8} {'LastPx':>10}"
        print(hdr)
        print("-" * len(hdr))
        for t in want:
            st = state.get(t, default_ticker_state())
            nh = sum(st.get("near_hits", []))
            sh = sum(st.get("sell_hits", []))
            lp = st.get("last_price", "")
            print(f"{t:6} {st.get('buy_state',''):10} {st.get('sell_state',''):10} {nh:8} {sh:8} {lp:>10}")

    # 6) Final line
    b, s = len(buy_triggers), len(sell_triggers)
    log(f"Done. Emitted {b} BUY and {s} SELL signals.", level="ok")
    if wrote_any:
        log(f"Signals appended → {args.write_signals}", level="ok")

if __name__ == "__main__":
    main()
