#!/usr/bin/env python3
"""
Minimal stateful BUY/SELL engine for Weinstein intraday scans.

Inputs (per-scan DataFrame columns expected):
  ticker, price, pivot, elapsed_min,
  cond_weekly_stage_ok, cond_rs_ok, cond_ma_ok, cond_pivot_ok,
  cond_buy_vol_ok, cond_pace_full_gate, cond_near_pace_gate, cond_buy_price_ok

Behavior:
- ARMED when all gates except price are true AND 0 <= dist_bps <= bps_near.
- BUY fires **once** when ARMED and cond_buy_price_ok flips true within arm_window_min.
- SELL fires on:
  (A) failed breakout: price below pivot - bps_fail for min_consec scans
  (B) critical gate failure (MA or RS) for min_consec scans
  (C) optional stop_loss_pct / take_profit_pct vs entry

State is persisted per day in JSON to avoid re-firing.
"""
from __future__ import annotations
import json, os, datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class Params:
    bps_near: float = 25.0          # proximity to pivot to arm
    arm_window_min: int = 40        # minutes after arming to accept a BUY cross
    bps_fail: float = 30.0          # below pivot by this much = failed breakout risk
    min_consec: int = 2             # consecutive scans to confirm failure
    stop_loss_pct: Optional[float] = None  # e.g., 0.05 for 5%
    take_profit_pct: Optional[float] = None
    single_shot: bool = True        # only one BUY per ticker per day

@dataclass
class TickerState:
    phase: str = "IDLE"             # IDLE | ARMED | LONG
    armed_at_min: Optional[float] = None
    entry_price: Optional[float] = None
    entry_pivot: Optional[float] = None
    consec_fail: int = 0
    last_seen_min: Optional[float] = None
    last_buy_min: Optional[float] = None

def _today_key() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")

def load_state(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"date": _today_key(), "tickers": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        return {"date": _today_key(), "tickers": {}}
    # reset at day boundary
    if s.get("date") != _today_key():
        return {"date": _today_key(), "tickers": {}}
    s.setdefault("tickers", {})
    return s

def save_state(path: str, state: Dict) -> None:
    if not path: 
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _row_ok(row, col):  # safe bool getter
    return bool(row.get(col, False))

def _dist_bps(row):
    price = row.get("price")
    pivot = row.get("pivot")
    if price is None or pivot in (None, 0):
        return None
    try:
        return (pivot - float(price)) / float(pivot) * 1e4
    except Exception:
        return None

def _ensure_ts_min(row):
    v = row.get("elapsed_min")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def evaluate_scan(df: pd.DataFrame, state_path: str, params: Params) -> Dict[str, List[Dict]]:
    """
    Returns dict with 'buys' and 'sells' lists of signal dicts.
    Persists updated per-ticker state to `state_path`.
    """
    state = load_state(state_path)
    tickers = state["tickers"]
    signals = {"buys": [], "sells": []}

    if df is None or df.empty:
        save_state(state_path, state)
        return signals

    # normalize types
    df = df.copy()
    for c in ["price", "pivot", "elapsed_min"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # compute dist_bps if absent
    if "dist_bps" not in df.columns or df["dist_bps"].isna().all():
        df["dist_bps"] = df.apply(_dist_bps, axis=1)

    # process per ticker using the *latest row* per ticker (by elapsed_min)
    if "elapsed_min" in df.columns:
        df = df.sort_values(["ticker", "elapsed_min"])
        latest = df.groupby("ticker").tail(1)
    else:
        latest = df.drop_duplicates("ticker", keep="last")

    for _, row in latest.iterrows():
        t = str(row.get("ticker"))
        if not t or t == "nan":
            continue

        price = row.get("price")
        pivot = row.get("pivot")
        dist = row.get("dist_bps")
        ts_min = _ensure_ts_min(row)

        # ensure state
        raw = tickers.get(t) or {}
        st = TickerState(**{**TickerState().__dict__, **raw})
        st.last_seen_min = ts_min

        # gate bundles
        all_gates_except_price = (
            _row_ok(row, "cond_weekly_stage_ok") and
            _row_ok(row, "cond_rs_ok") and
            _row_ok(row, "cond_ma_ok") and
            _row_ok(row, "cond_pivot_ok") and
            _row_ok(row, "cond_buy_vol_ok") and
            _row_ok(row, "cond_pace_full_gate") and
            _row_ok(row, "cond_near_pace_gate")
        )
        buy_cross = _row_ok(row, "cond_buy_price_ok")

        # ---------- PHASE LOGIC ----------
        if st.phase == "IDLE":
            # ARM when aligned and near pivot
            if all_gates_except_price and dist is not None and 0 <= dist <= params.bps_near:
                st.phase = "ARMED"
                st.armed_at_min = ts_min
                st.consec_fail = 0

        elif st.phase == "ARMED":
            # timeout the arming if price wanders off or time window exceeded
            if not all_gates_except_price or dist is None or dist < 0 or dist > params.bps_near:
                st.phase = "IDLE"
                st.armed_at_min = None
                st.consec_fail = 0
            else:
                within_window = (
                    st.armed_at_min is not None and
                    ts_min is not None and
                    (ts_min - st.armed_at_min) <= params.arm_window_min
                )
                # BUY only when we cross while armed and within window
                if buy_cross and within_window and (not params.single_shot or st.last_buy_min is None):
                    st.phase = "LONG"
                    st.entry_price = float(price) if price is not None else None
                    st.entry_pivot = float(pivot) if pivot is not None else None
                    st.last_buy_min = ts_min
                    st.consec_fail = 0
                    signals["buys"].append({
                        "ticker": t,
                        "price": price,
                        "pivot": pivot,
                        "ts_min": ts_min,
                        "reason": "BUY: armed->cross"
                    })

        elif st.phase == "LONG":
            fail = False
            reason = None

            # A) failed breakout vs pivot
            if pivot not in (None, 0) and price is not None:
                below_bps = (float(pivot) - float(price)) / float(pivot) * 1e4
                if below_bps >= params.bps_fail:
                    st.consec_fail += 1
                else:
                    st.consec_fail = 0
                if st.consec_fail >= params.min_consec:
                    fail = True
                    reason = f"SELL: failed_breakout {below_bps:.1f}bps"

            # B) critical gate failure
            if not fail:
                crit_ok = _row_ok(row, "cond_ma_ok") and _row_ok(row, "cond_rs_ok")
                if not crit_ok:
                    st.consec_fail += 1
                else:
                    st.consec_fail = 0
                if st.consec_fail >= params.min_consec:
                    fail = True
                    reason = "SELL: critical gate failure (MA/RS)"

            # C) stop/take
            if not fail and st.entry_price and params.stop_loss_pct:
                if price is not None and (price <= st.entry_price * (1 - params.stop_loss_pct)):
                    fail = True
                    reason = f"SELL: stop {-params.stop_loss_pct*100:.1f}%"
            if not fail and st.entry_price and params.take_profit_pct:
                if price is not None and (price >= st.entry_price * (1 + params.take_profit_pct)):
                    fail = True
                    reason = f"SELL: target {params.take_profit_pct*100:.1f}%"

            if fail:
                signals["sells"].append({
                    "ticker": t,
                    "price": price,
                    "pivot": pivot,
                    "ts_min": ts_min,
                    "reason": reason
                })
                # go back to IDLE (single-shot), or allow re-arm immediately
                st.phase = "IDLE" if params.single_shot else "IDLE"
                st.armed_at_min = None
                st.entry_price = None
                st.entry_pivot = None
                st.consec_fail = 0

        # persist
        tickers[t] = {**asdict(st)}

    state["tickers"] = tickers
    save_state(state_path, state)
    return signals
