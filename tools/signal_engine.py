#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/signal_engine.py

Purpose
-------
Reconstruct and emit BUY/NEAR/SELL signals from intraday diagnostics rows
(the same booleans you log in intraday_debug.csv) using the same promotion
state machine as your intraday watcher:

  IDLE -> NEAR -> ARMED -> TRIGGERED -> COOLDOWN -> IDLE

It also:
- Computes “near-now” and “armed-now” from basis-point distance to pivot.
- Tracks SELL near/armed/trigger on MA150 breaks.
- Appends durable rows into --write-signals CSV (idempotent header).
- Persists in-flight state in --state JSON (so running every 10m works).
- Prints a concise summary and optional per-ticker “explain”.

Inputs it expects
-----------------
A CSV with the columns you already write in intraday_debug.csv (or compatible):
  ticker, price, pivot, ma30, elapsed_min, cond_weekly_stage_ok, cond_rs_ok,
  cond_ma_ok, cond_pivot_ok, cond_buy_price_ok, cond_buy_vol_ok,
  cond_pace_full_gate, cond_near_pace_gate,
  (optional) cond_near_now, cond_sell_near_now, cond_sell_price_ok, cond_sell_vol_ok,
  state, sell_state

If some columns are missing, reasonable fallbacks are applied.

Usage
-----
python3 tools/signal_engine.py \
  --csv ./output/intraday_debug.csv \
  --outdir ./output \
  --state ./output/diag_state.json \
  --write-signals ./output/signals_log.csv \
  --bps-threshold 35 \
  --window-min 120 \
  --explain MU,DDOG

"""

import os
import sys
import csv
import json
import math
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# ---------------- Defaults (mirrors your watcher) ----------------
SCAN_INTERVAL_MIN = 10
NEAR_HITS_WINDOW  = 6        # ~1 hour memory if every 10m
NEAR_HITS_MIN     = 3
COOLDOWN_SCANS    = 24       # 4 hours (10m cadence)

SELL_NEAR_HITS_WINDOW = 6
SELL_NEAR_HITS_MIN    = 3
SELL_COOLDOWN_SCANS   = 24

# Buy confirmation & gates (should mirror weinstein_intraday_watcher.py)
MIN_BREAKOUT_PCT = 0.004        # 0.4% above pivot
BUY_DIST_ABOVE_MA_MIN = 0.00
VOL_PACE_MIN = 1.30
NEAR_VOL_PACE_MIN = 1.00

# 60m easing equivalents (we can’t know bar size from CSV; we only honor the booleans)
INTRABAR_CONFIRM_MIN_ELAPSED = 40
INTRABAR_VOLPACE_MIN         = 1.20

# Sell rules (must match watcher)
SELL_NEAR_ABOVE_MA_PCT = 0.005  # within +0.5% above MA is still “sell near”
SELL_BREAK_PCT         = 0.005  # 0.5% confirmed break below MA

# ---------------- Utilities ----------------
def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, level="info"):
    prefix = {"info":"•", "ok":"✅", "step":"▶️", "warn":"⚠️", "err":"❌", "debug":"··"}.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)

def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _update_hits(window_arr, hit, window):
    window_arr = (window_arr or [])
    window_arr.append(1 if hit else 0)
    if len(window_arr) > window:
        window_arr = window_arr[-window:]
    return window_arr, sum(window_arr)

def _price_below_ma(px, ma): 
    return pd.notna(px) and pd.notna(ma) and px <= ma * (1.0 - SELL_BREAK_PCT)

def _near_sell_zone(px, ma):
    if pd.isna(px) or pd.isna(ma): return False
    return (px >= ma * (1.0 - SELL_BREAK_PCT)) and (px <= ma * (1.0 + SELL_NEAR_ABOVE_MA_PCT))

# ---------------- State IO ----------------
def load_state(path: str):
    if not path:
        return {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(path: str, state: dict):
    if not path: 
        return
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

# ---------------- Signals log (append) ----------------
SIGNALS_HEADER = [
    "ts", "ticker", "type", "price",
    "pivot", "ma30", "dist_bps",
    "buy_state", "sell_state",
    "why"
]

def append_signals_csv(path, rows: list[dict]):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIGNALS_HEADER, quoting=csv.QUOTE_MINIMAL)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SIGNALS_HEADER})

# ---------------- Core engine ----------------
def compute_from_csv(
    df: pd.DataFrame,
    *,
    bps_threshold: int = 35,
    window_min: int = 60,
    near_hits_min: int = NEAR_HITS_MIN,
    sell_near_hits_min: int = SELL_NEAR_HITS_MIN,
    cooldown_scans: int = COOLDOWN_SCANS,
    sell_cooldown_scans: int = SELL_COOLDOWN_SCANS,
    state: dict | None = None,
    explain: set[str] | None = None
):
    """
    Rehydrates signals from a CSV snapshot. Returns (summary, signals_to_append, state_out, explain_lines)
    """
    if state is None:
        state = {}

    # Basic column normalization
    need_bool = [
        "cond_weekly_stage_ok","cond_rs_ok","cond_ma_ok","cond_pivot_ok",
        "cond_buy_price_ok","cond_buy_vol_ok","cond_pace_full_gate","cond_near_pace_gate",
        "cond_sell_price_ok","cond_sell_vol_ok"
    ]
    for c in need_bool:
        if c not in df.columns:
            df[c] = False

    for c in ["ticker","price","pivot","ma30","elapsed_min"]:
        if c not in df.columns: df[c] = np.nan

    # Distance to pivot in bps (positive => below pivot)
    with np.errstate(all='ignore'):
        df["dist_bps"] = (df["pivot"] - df["price"]) / df["pivot"] * 1e4

    # We’ll label “near-now” ourselves if missing
    if "cond_near_now" not in df.columns:
        df["cond_near_now"] = False
    if "cond_sell_near_now" not in df.columns:
        df["cond_sell_near_now"] = False

    # Resolve near-now and sell-near-now from price if not present
    def _resolve_near(row):
        px, pv, ma = row["price"], row["pivot"], row["ma30"]
        stage_ok = bool(row["cond_weekly_stage_ok"])
        rs_ok    = bool(row["cond_rs_ok"])
        if not (stage_ok and rs_ok and pd.notna(px) and pd.notna(pv) and pd.notna(ma)):
            return False
        # near, if within bps_threshold below pivot OR above pivot but not confirmed
        dist_ok = pd.notna(row["dist_bps"]) and (row["dist_bps"] >= -bps_threshold)  # -ve bps => already > pivot; keep near if not confirmed
        price_ok = bool(row["cond_buy_price_ok"])
        return dist_ok or (not price_ok and px >= pv)
    df["near_resolved"] = df.apply(_resolve_near, axis=1)
    df["cond_near_now"] = df["cond_near_now"] | df["near_resolved"]

    def _resolve_sell_near(row):
        px, ma = row["price"], row["ma30"]
        if pd.isna(px) or pd.isna(ma): return False
        return _near_sell_zone(px, ma)
    df["sell_near_resolved"] = df.apply(_resolve_sell_near, axis=1)
    df["cond_sell_near_now"] = df["cond_sell_near_now"] | df["sell_near_resolved"]

    # Consolidate by the latest observation per ticker (assume csv is last-scan only OR we take last seen per ticker)
    # If you pass a multi-scan stack, we’ll keep the last row per ticker.
    if "elapsed_min" in df.columns:
        # Keep the max elapsed_min per ticker (most recent bar end within the session)
        df["_rank"] = df.groupby("ticker")["elapsed_min"].rank(method="first", ascending=False)
        df = df[df["_rank"] == 1].drop(columns=["_rank"])

    # State windows from params
    scans_window = max(1, int(round(window_min / float(SCAN_INTERVAL_MIN))))  # e.g. 120m / 10m = 12 scans
    if "near_hits_window_override" in df.columns:
        scans_window = int(df["near_hits_window_override"].dropna().iloc[0])  # optional override

    buy_out, near_out, selltrig_out = [], [], []
    signals_to_append = []
    explain_lines = []

    for _, r in df.iterrows():
        t = str(r.get("ticker","")).strip().upper()
        if not t:
            continue
        px    = float(r.get("price", np.nan))
        pv    = float(r.get("pivot", np.nan))
        ma    = float(r.get("ma30", np.nan))
        dist  = float(r.get("dist_bps", np.nan)) if pd.notna(r.get("dist_bps", np.nan)) else np.nan

        stage_ok = bool(r.get("cond_weekly_stage_ok", False))
        rs_ok    = bool(r.get("cond_rs_ok", False))
        ma_ok    = bool(r.get("cond_ma_ok", False))
        pv_ok    = bool(r.get("cond_pivot_ok", False))
        price_ok = bool(r.get("cond_buy_price_ok", False))
        vol_ok   = bool(r.get("cond_buy_vol_ok", False))
        pace_full_gate = bool(r.get("cond_pace_full_gate", False))
        near_pace_gate = bool(r.get("cond_near_pace_gate", False))

        sell_price_ok = bool(r.get("cond_sell_price_ok", False))
        sell_vol_ok   = bool(r.get("cond_sell_vol_ok", False))

        near_now      = bool(r.get("cond_near_now", False))
        sell_near_now = bool(r.get("cond_sell_near_now", False))

        # BUY confirmation (we assume booleans already reflect intrabar timing/pace in the watcher logs)
        buy_confirm = bool(price_ok and vol_ok)

        # SELL confirmation
        sell_confirm = bool(sell_price_ok and sell_vol_ok)

        # Get or init state
        st = state.get(t, {
            "state":"IDLE", "near_hits":[], "cooldown":0,
            "sell_state":"IDLE", "sell_hits":[], "sell_cooldown":0
        })

        # BUY track
        st["near_hits"], near_count = _update_hits(st.get("near_hits", []), near_now, scans_window)
        if st.get("cooldown", 0) > 0:
            st["cooldown"] = int(st["cooldown"]) - 1

        prev_state = st.get("state","IDLE")
        state_now  = prev_state

        if state_now == "IDLE" and near_now:
            state_now = "NEAR"
        elif state_now in ("IDLE","NEAR") and near_count >= int(near_hits_min):
            state_now = "ARMED"
        elif state_now == "ARMED" and buy_confirm and pace_full_gate:
            state_now = "TRIGGERED"
            st["cooldown"] = int(cooldown_scans)
        elif state_now == "TRIGGERED":
            pass
        elif st["cooldown"] > 0 and not near_now:
            state_now = "COOLDOWN"
        elif st["cooldown"] == 0 and not near_now and not buy_confirm:
            state_now = "IDLE"

        st["state"] = state_now

        # SELL track
        st["sell_hits"], sell_hit_count = _update_hits(st.get("sell_hits", []), sell_near_now, scans_window)
        if st.get("sell_cooldown", 0) > 0:
            st["sell_cooldown"] = int(st["sell_cooldown"]) - 1

        prev_sell = st.get("sell_state","IDLE")
        sell_now  = prev_sell

        if sell_now == "IDLE" and sell_near_now:
            sell_now = "NEAR"
        elif sell_now in ("IDLE","NEAR") and sell_hit_count >= int(sell_near_hits_min):
            sell_now = "ARMED"
        elif sell_now == "ARMED" and sell_confirm:
            sell_now = "TRIGGERED"
            st["sell_cooldown"] = int(sell_cooldown_scans)
        elif sell_now == "TRIGGERED":
            pass
        elif st["sell_cooldown"] > 0 and not sell_near_now:
            sell_now = "COOLDOWN"
        elif st["sell_cooldown"] == 0 and not sell_near_now and not sell_confirm:
            sell_now = "IDLE"

        st["sell_state"] = sell_now
        state[t] = st  # persist local

        # Emit
        why = []
        if state_now == "TRIGGERED" and pace_full_gate:
            buy_out.append((t, px, pv, ma, dist))
            signals_to_append.append({
                "ts": _ts(), "ticker": t, "type": "BUY",
                "price": f"{px:.4f}" if pd.notna(px) else "",
                "pivot": f"{pv:.4f}" if pd.notna(pv) else "",
                "ma30":  f"{ma:.4f}" if pd.notna(ma) else "",
                "dist_bps": f"{dist:.1f}" if pd.notna(dist) else "",
                "buy_state": state_now, "sell_state": sell_now,
                "why": "confirm over pivot & MA with pace"
            })
            # move to cooldown
            state[t]["state"] = "COOLDOWN"

        elif state_now in ("NEAR","ARMED") and near_pace_gate:
            near_out.append((t, px, pv, ma, dist))

        if sell_now == "TRIGGERED":
            selltrig_out.append((t, px, pv, ma, dist))
            signals_to_append.append({
                "ts": _ts(), "ticker": t, "type": "SELL",
                "price": f"{px:.4f}" if pd.notna(px) else "",
                "pivot": f"{pv:.4f}" if pd.notna(pv) else "",
                "ma30":  f"{ma:.4f}" if pd.notna(ma) else "",
                "dist_bps": f"{dist:.1f}" if pd.notna(dist) else "",
                "buy_state": state_now, "sell_state": sell_now,
                "why": "confirmed crack below MA150"
            })
            state[t]["sell_state"] = "COOLDOWN"

        # Explain block (best effort)
        if explain and (t in explain):
            line = (f"{t}: buy_state={prev_state}->{state_now} "
                    f"(near_now={near_now}, near_hits={sum(st.get('near_hits',[]))}/{scans_window}, "
                    f"confirm={buy_confirm}, pace_full={pace_full_gate}) | "
                    f"sell_state={prev_sell}->{sell_now} "
                    f"(sell_near_now={sell_near_now}, hits={sum(st.get('sell_hits',[]))}/{scans_window}, "
                    f"confirm={sell_confirm}) "
                    f"dist_bps={dist if pd.notna(dist) else '—'}")
            explain_lines.append(line)

    summary = {
        "buy":   sorted(buy_out),
        "near":  sorted(near_out),
        "sell":  sorted(selltrig_out)
    }
    return summary, signals_to_append, state, explain_lines

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="intraday_debug.csv (latest scan or stacked scans)")
    ap.add_argument("--outdir", default="./output")
    ap.add_argument("--state", default="./output/diag_state.json", help="state file to persist NEAR/ARMED/COOLDOWN")
    ap.add_argument("--write-signals", default="", help="append durable signals to this CSV (e.g. ./output/signals_log.csv)")
    ap.add_argument("--bps-threshold", type=int, default=35, help="near-now window below pivot in bps")
    ap.add_argument("--window-min", type=int, default=60, help="rolling memory window in minutes (for near hits)")
    ap.add_argument("--near-hits-min", type=int, default=NEAR_HITS_MIN)
    ap.add_argument("--sell-near-hits-min", type=int, default=SELL_NEAR_HITS_MIN)
    ap.add_argument("--cooldown-scans", type=int, default=COOLDOWN_SCANS)
    ap.add_argument("--sell-cooldown-scans", type=int, default=SELL_COOLDOWN_SCANS)
    ap.add_argument("--explain", type=str, default="", help="comma tickers to explain (e.g. MU,DDOG)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        log(f"Failed to read CSV: {e}", level="err")
        sys.exit(1)

    # Lightweight scrub
    if "ticker" not in df.columns:
        log("CSV missing 'ticker' column.", level="err")
        sys.exit(2)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Optional narrow to most recent scan if the file contains multiple ticks
    # If you log a timestamp column (e.g. 'ts' or 'scan_ts'), you can adapt here.
    # For now, we assume the file is "latest only" or we pick last row per ticker in compute().

    state_in = load_state(args.state)
    explain = set([s.strip().upper() for s in args.explain.split(",") if s.strip()]) if args.explain else set()

    summary, sig_rows, state_out, explain_lines = compute_from_csv(
        df,
        bps_threshold=args.bps_threshold,
        window_min=args.window_min,
        near_hits_min=args.near_hits_min,
        sell_near_hits_min=args.sell_near_hits_min,
        cooldown_scans=args.cooldown_scans,
        sell_cooldown_scans=args.sell_cooldown_scans,
        state=state_in,
        explain=explain
    )

    # Persist
    save_state(args.state, state_out)
    append_signals_csv(args.write_signals, sig_rows)

    # Print summary
    def _fmt(lst):
        if not lst: return "(none)"
        return ", ".join([f"{t}@{px:.2f}" if pd.notna(px) else t for (t,px,_,_,_) in lst])

    log("Engine summary:", level="ok")
    log(f"  BUY   : {_fmt(summary['buy'])}", level="info")
    log(f"  NEAR  : {_fmt(summary['near'])}", level="info")
    log(f"  SELL  : {_fmt(summary['sell'])}", level="info")

    if sig_rows:
        log(f"Appended {len(sig_rows)} rows to {args.write_signals}", level="ok")
    else:
        log("No new durable signals this run.", level="debug")

    if explain_lines:
        print("\nExplain:")
        for ln in explain_lines:
            print(" - " + ln)

if __name__ == "__main__":
    main()
