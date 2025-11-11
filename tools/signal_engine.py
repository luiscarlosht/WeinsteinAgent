#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Engine
-------------
Single place to decide:
- when a row is "armed" (all gates except price within BPS),
- when to emit BUY / SELL,
- log signals idempotently,
- maintain rolling state for duplicates & cooldowns.

This engine is intentionally dumb, pure-Python, no pandas dependency.
It operates on dict-like "row" objects (e.g., from your intraday loop).

Compatible with diagnose_intraday.py output columns.
"""

from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Iterable, Tuple
from datetime import datetime, timezone

# -----------------------------
# Defaults (must match diagnose)
# -----------------------------
DEFAULT_BPS_THRESHOLD = 35        # “near pivot” window for ARMED
DEFAULT_WINDOW_MIN     = 120      # lookback/cooldown window
DEFAULT_MIN_SCANS      = 2        # require at least N scans before emitting BUY
DEFAULT_TZ             = timezone.utc


def now_iso() -> str:
    return datetime.now(tz=DEFAULT_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")


def bps(x: float) -> float:
    return float(x) * 1e4


@dataclass
class EngineConfig:
    bps_threshold: int = DEFAULT_BPS_THRESHOLD
    window_min: int    = DEFAULT_WINDOW_MIN
    min_scans: int     = DEFAULT_MIN_SCANS
    # optional: treat weekly stage "unknown" as pass?
    weekly_required: bool = True


@dataclass
class RowView:
    # Required columns used by the engine (strings for safety; caller converts if needed)
    ticker: str
    price: float
    pivot: float
    elapsed_min: int

    # boolean “gate” flags produced by your intraday calc
    cond_weekly_stage_ok: bool
    cond_rs_ok: bool
    cond_ma_ok: bool
    cond_pivot_ok: bool
    cond_buy_vol_ok: bool
    cond_pace_full_gate: bool    # strict pace gate
    cond_near_pace_gate: bool    # near pace gate
    cond_buy_price_ok: bool      # price >= pivot (or your exact rule)

    # SELL gates (these mirror reasons you print in HTML)
    cond_sell_stage4_negpl: bool = False      # e.g. Stage 4 + negative P/L
    cond_sell_drawdown_8: bool   = False      # drawdown <= -8%

    # Optional extra context:
    scans_for_ticker: int = 1                # running count in current day/session
    weekly_stage_label: Optional[str] = None # e.g. "Stage 2 (Uptrend)"
    debug_ts: Optional[str] = None           # ISO time

    @property
    def dist_bps(self) -> float:
        # positive → price below pivot (needs to travel up to hit pivot)
        try:
            return (self.pivot - self.price) / self.pivot * 1e4
        except ZeroDivisionError:
            return 1e9


@dataclass
class Signal:
    ts: str
    ticker: str
    side: str        # BUY or SELL
    reason: str
    price: float
    pivot: float
    dist_bps: float
    elapsed_min: int
    scans: int


class SignalEngine:
    """
    Stateless rules + lightweight state file to suppress duplicates within window.
    """

    def __init__(self,
                 cfg: EngineConfig = EngineConfig(),
                 state_path: Optional[str] = None,
                 signals_csv: Optional[str] = None) -> None:
        self.cfg = cfg
        self.state_path = state_path
        self.signals_csv = signals_csv

        self.state: Dict[str, Dict[str, int]] = {"last_buy_min": {}, "last_sell_min": {}}
        if state_path and os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    self.state = json.load(f)
            except Exception:
                pass

        # ensure CSV header
        if signals_csv and (not os.path.exists(signals_csv) or os.path.getsize(signals_csv) == 0):
            with open(signals_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts","ticker","side","reason","price","pivot","dist_bps","elapsed_min","scans"])

    # ---------- Core logic ----------

    def is_weekly_ok(self, row: RowView) -> bool:
        if not self.cfg.weekly_required:
            return True
        return bool(row.cond_weekly_stage_ok)

    def is_armed(self, row: RowView) -> bool:
        """
        “Armed” = all buy gates except price are satisfied AND within bps window.
        This mirrors diagnose_intraday.py’s “Armed now”.
        """
        gates = [
            self.is_weekly_ok(row),
            row.cond_rs_ok,
            row.cond_ma_ok,
            row.cond_pivot_ok,
            row.cond_buy_vol_ok,
            row.cond_pace_full_gate,
            row.cond_near_pace_gate,
        ]
        if not all(gates):
            return False
        return (row.dist_bps >= 0) and (row.dist_bps <= self.cfg.bps_threshold)

    def should_buy(self, row: RowView) -> Tuple[bool, str]:
        """
        BUY when “armed” AND price gate flips true (your definition:
        e.g., price ≥ pivot and pace_full_gate true).
        Add a min scans guard and cooldown window to avoid spam.
        """
        if row.scans_for_ticker < self.cfg.min_scans:
            return False, f"min_scans<{self.cfg.min_scans}"

        if not self.is_armed(row):
            return False, "not_armed"

        # final price gate
        if not row.cond_buy_price_ok:
            return False, "price_gate_false"

        # cooldown (per ticker)
        last = self.state["last_buy_min"].get(row.ticker, -10_000)
        if row.elapsed_min - last < self.cfg.window_min:
            return False, "cooldown"

        return True, "armed+price_gate"

    def should_sell(self, row: RowView) -> Tuple[bool, str]:
        """
        SELL when Stage 4 + negative P/L OR drawdown ≤ -8% (or extend with your rules).
        Cooldown applied per ticker.
        """
        if not (row.cond_sell_stage4_negpl or row.cond_sell_drawdown_8):
            return False, "no_sell_gate"

        last = self.state["last_sell_min"].get(row.ticker, -10_000)
        if row.elapsed_min - last < self.cfg.window_min:
            return False, "cooldown"

        if row.cond_sell_stage4_negpl:
            return True, "stage4_negpl"
        else:
            return True, "drawdown_8"

    # ---------- Emission helpers ----------

    def emit(self, sig: Signal) -> None:
        if self.signals_csv:
            with open(self.signals_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([sig.ts, sig.ticker, sig.side, sig.reason,
                            f"{sig.price:.4f}", f"{sig.pivot:.4f}",
                            f"{sig.dist_bps:.1f}", sig.elapsed_min, sig.scans])

    def save_state(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_path)

    # ---------- Public API ----------

    def process_row(self, row_dict: Dict) -> Optional[Signal]:
        """
        Feed one calc row (dict). Returns a Signal or None.
        """
        # normalize to RowView
        row = RowView(
            ticker=row_dict["ticker"],
            price=float(row_dict["price"]),
            pivot=float(row_dict["pivot"]),
            elapsed_min=int(row_dict["elapsed_min"]),

            cond_weekly_stage_ok=bool(row_dict.get("cond_weekly_stage_ok", False)),
            cond_rs_ok=bool(row_dict.get("cond_rs_ok", False)),
            cond_ma_ok=bool(row_dict.get("cond_ma_ok", False)),
            cond_pivot_ok=bool(row_dict.get("cond_pivot_ok", False)),
            cond_buy_vol_ok=bool(row_dict.get("cond_buy_vol_ok", False)),
            cond_pace_full_gate=bool(row_dict.get("cond_pace_full_gate", False)),
            cond_near_pace_gate=bool(row_dict.get("cond_near_pace_gate", False)),
            cond_buy_price_ok=bool(row_dict.get("cond_buy_price_ok", False)),

            cond_sell_stage4_negpl=bool(row_dict.get("cond_sell_stage4_negpl", False)),
            cond_sell_drawdown_8=bool(row_dict.get("cond_sell_drawdown_8", False)),

            scans_for_ticker=int(row_dict.get("scans_for_ticker", 1)),
            weekly_stage_label=row_dict.get("weekly_stage_label"),
            debug_ts=row_dict.get("debug_ts"),
        )

        # BUY?
        ok_buy, why_buy = self.should_buy(row)
        if ok_buy:
            sig = Signal(
                ts=row.debug_ts or now_iso(),
                ticker=row.ticker,
                side="BUY",
                reason=why_buy,
                price=row.price,
                pivot=row.pivot,
                dist_bps=row.dist_bps,
                elapsed_min=row.elapsed_min,
                scans=row.scans_for_ticker,
            )
            self.emit(sig)
            self.state["last_buy_min"][row.ticker] = row.elapsed_min
            self.save_state()
            return sig

        # SELL?
        ok_sell, why_sell = self.should_sell(row)
        if ok_sell:
            sig = Signal(
                ts=row.debug_ts or now_iso(),
                ticker=row.ticker,
                side="SELL",
                reason=why_sell,
                price=row.price,
                pivot=row.pivot,
                dist_bps=row.dist_bps,
                elapsed_min=row.elapsed_min,
                scans=row.scans_for_ticker,
            )
            self.emit(sig)
            self.state["last_sell_min"][row.ticker] = row.elapsed_min
            self.save_state()
            return sig

        return None


# ---------------- CLI demo (optional) ----------------
if __name__ == "__main__":
    # simple smoke test (no output expected)
    eng = SignalEngine(
        cfg=EngineConfig(),
        state_path=None,
        signals_csv=None,
    )
    demo = {
        "ticker": "TEST",
        "price": 100, "pivot": 100,
        "elapsed_min": 500,
        "cond_weekly_stage_ok": True,
        "cond_rs_ok": True, "cond_ma_ok": True, "cond_pivot_ok": True,
        "cond_buy_vol_ok": True, "cond_pace_full_gate": True, "cond_near_pace_gate": True,
        "cond_buy_price_ok": True,
        "scans_for_ticker": 3,
        "debug_ts": now_iso(),
    }
    sig = eng.process_row(demo)
    print("DEMO signal:", sig)
