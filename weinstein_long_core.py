#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_long_core.py

Shared LONG-side core logic for:
- Intraday PROD watchers
- Daily SIM backtester (weinstein_live_logic_backtest_yfinance.py)

Core idea:
- One place where we define:
    * thresholds (breakout %, min distance above MA, vol pace, ADX)
    * "can we enter?" decision logic
    * risk helpers (hard stop / ATR / MA guard / optional targets)
- Anything that wants to do a Weinstein Stage 2 breakout
  just calls check_long_entry(...) and inspects the result.

This module intentionally mirrors weinstein_short_core.py so that:
- Entry filters (LONG vs SHORT) live in symmetric LongEntry*/ShortEntry* APIs.
- Risk/exit helpers (long_stop_level/should_exit_long) are shared
  between PROD and SIM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# We reuse the canonical ADX_MIN from weinstein_indicators so that
# PROD, SIM, and any other tools stay in sync on the threshold.
try:
    from weinstein_indicators import ADX_MIN as DEFAULT_ADX_MIN
except ImportError:
    # Fallback, in case the module is missing or renamed.
    DEFAULT_ADX_MIN = 18.0


# -------------------------------------------------------------------
# Shared LONG-side constants
# -------------------------------------------------------------------

# Breakout / volume thresholds (aligned with backtester + intraday)
LONG_BREAK_PCT: float = 0.004   # ≈0.4% above pivot breakout
LONG_VOL_MIN: float = 1.30      # volume vs 50dma (≈ 1.3×)

# Risk / stop / target mapping (used by both intraday + backtests)
LONG_HARD_STOP_PCT: float = 0.20    # 20% below entry (disaster stop)
LONG_TRAIL_ATR_MULT: float = 2.0    # ATR cushion
LONG_MA_GUARD_PCT: float = 0.03     # 3% below MA guard (MA30 or MA150 depending on caller)

# Optional upside targets, symmetric with short core (-15% / -20% there)
LONG_TARGET1_PCT: float = 0.15      # 15% upside
LONG_TARGET2_PCT: float = 0.20      # 20% upside


# -------------------------------------------------------------------
# Parameter + result models
# -------------------------------------------------------------------

@dataclass
class LongEntryParams:
    """
    Tunable thresholds for the long entry decision.

    These are "generic" enough that both:
      - intraday logic (5m/15m) with volume pace
      - daily SIM logic (daily bars)
    can share them.

    Attributes:
        min_break_pct:
            Required % breakout above the pivot high (e.g. 0.004 = 0.4%).
        dist_above_ma_min:
            Optional extra headroom above MA (e.g. 0.01 = 1%).
            SIM currently sets this to 0.0 (just price > MA).
        vol_min:
            Minimum volume multiple vs 50dma (e.g. 1.3).
        adx_min:
            Minimum ADX to accept (e.g. 18.0).
    """
    min_break_pct: float = LONG_BREAK_PCT
    dist_above_ma_min: float = 0.0
    vol_min: float = LONG_VOL_MIN
    adx_min: float = DEFAULT_ADX_MIN


@dataclass
class LongEntryResult:
    """
    Outcome of the long entry check.

    Attributes:
        can_enter:
            True when all gates (RS, MA, pivot, vol, ADX) pass.
        reason:
            Short diagnostic string explaining the first reason for rejection
            (or "ok" if can_enter=True).
        adx_ok:
            True when ADX is either:
              - NaN / missing (we do NOT block in that case), or
              - >= adx_min
            False only when ADX is present AND < adx_min.
    """
    can_enter: bool
    reason: str
    adx_ok: bool


# -------------------------------------------------------------------
# Helper(s)
# -------------------------------------------------------------------

def _is_nan(x: float) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


# -------------------------------------------------------------------
# Core check function (shared between PROD + SIM)
# -------------------------------------------------------------------

def check_long_entry(
    *,
    price: float,
    ma_val: float,
    pivot: float,
    rs_above_ma: bool,
    vol_mult: float,
    adx_val: float,
    # Preferred interface (used by SIM backtester and intraday):
    params: Optional[LongEntryParams] = None,
    # Backward-compatible knobs (in case some caller passes explicit thresholds):
    min_break_pct: Optional[float] = None,
    dist_above_ma_min: Optional[float] = None,
    vol_min: Optional[float] = None,
    adx_min: Optional[float] = None,
) -> LongEntryResult:
    """
    Shared Weinstein Stage 2 LONG entry filter.

    Inputs:
        price:
            Current price (close / last).
        ma_val:
            MA(30) or MA150 depending on caller (trend MA).
        pivot:
            Highest close in lookback window (e.g. 50d).
        rs_above_ma:
            True if RS line is above its MA (strong RS).
        vol_mult:
            Volume multiple vs 50dma (e.g. 1.3 means 30% above).
        adx_val:
            Current ADX (same period as ADX_MIN).
        params:
            LongEntryParams object with thresholds.
        min_break_pct / dist_above_ma_min / vol_min / adx_min:
            Optional explicit overrides. If provided, they win over params.

    Returns:
        LongEntryResult(can_enter, reason, adx_ok)
    """
    # Effective thresholds
    if params is None:
        params = LongEntryParams()

    thr_break = min_break_pct if min_break_pct is not None else params.min_break_pct
    thr_dist_ma = (
        dist_above_ma_min
        if dist_above_ma_min is not None
        else params.dist_above_ma_min
    )
    thr_vol = vol_min if vol_min is not None else params.vol_min
    thr_adx = adx_min if adx_min is not None else params.adx_min

    # --- Basic NaN guards ---
    if _is_nan(price) or _is_nan(ma_val) or _is_nan(pivot):
        return LongEntryResult(
            can_enter=False,
            reason="nan_input",
            adx_ok=True,  # we don't know ADX effect if inputs are NaN
        )

    # --- RS must be above its MA (your weekly RS filter) ---
    if not rs_above_ma:
        return LongEntryResult(
            can_enter=False,
            reason="rs_not_above_ma",
            adx_ok=True,
        )

    # --- Price must be above MA with optional extra headroom ---
    # If dist_above_ma_min == 0 -> just price > MA.
    required_ma = ma_val * (1.0 + thr_dist_ma)
    if price <= required_ma:
        return LongEntryResult(
            can_enter=False,
            reason="price_not_above_ma",
            adx_ok=True,
        )

    # --- Breakout vs pivot high (e.g. 0.4% above 50-day high) ---
    required_pivot_level = pivot * (1.0 + thr_break)
    if price <= required_pivot_level:
        return LongEntryResult(
            can_enter=False,
            reason="no_breakout_vs_pivot",
            adx_ok=True,
        )

    # --- Volume pace filter (e.g. ≥ 1.3× 50dma) ---
    if _is_nan(vol_mult) or vol_mult < thr_vol:
        return LongEntryResult(
            can_enter=False,
            reason="volume_too_low",
            adx_ok=True,
        )

    # --- ADX filter (optional block) ---
    # If ADX is NaN, we *do not block* (adx_ok=True, can_enter depends on other gates).
    if not _is_nan(adx_val):
        if adx_val < thr_adx:
            return LongEntryResult(
                can_enter=False,
                reason="adx_below_min",
                adx_ok=False,
            )

    # If we got here, all gates have passed.
    return LongEntryResult(
        can_enter=True,
        reason="ok",
        adx_ok=True,
    )


# -------------------------------------------------------------------
# Shared LONG-side stop / target / exit helpers
# -------------------------------------------------------------------

def _long_entry_stop_targets(
    entry: float,
    ma_val: float,
    atr: float,
) -> Tuple[float, float, float, float]:
    """
    Convenience helper for callers that want a full "package" of levels.

    Parameters
    ----------
    entry : float
        Entry price (typically current price when the signal fires).
    ma_val : float
        Trend MA used for guard (MA30 in SIM, MA150 in intraday, etc.).
    atr : float
        ATR value for volatility-based cushion (14d in SIM).

    Returns
    -------
    (entry, stop, target1, target2)

      stop    = max(
                   entry * (1 - LONG_HARD_STOP_PCT),
                   entry - LONG_TRAIL_ATR_MULT * ATR,
                   ma_val * (1 - LONG_MA_GUARD_PCT)
                )
      target1 = entry * (1 + LONG_TARGET1_PCT)
      target2 = entry * (1 + LONG_TARGET2_PCT)
    """
    if _is_nan(entry):
        return np.nan, np.nan, np.nan, np.nan

    e = float(entry)

    hard = e * (1.0 - LONG_HARD_STOP_PCT)
    atr_stop = e - LONG_TRAIL_ATR_MULT * atr if not _is_nan(atr) else np.nan
    ma_guard = ma_val * (1.0 - LONG_MA_GUARD_PCT) if not _is_nan(ma_val) else np.nan

    cands = [c for c in (hard, atr_stop, ma_guard) if not _is_nan(c)]
    stop = max(cands) if cands else hard

    t1 = e * (1.0 + LONG_TARGET1_PCT)
    t2 = e * (1.0 + LONG_TARGET2_PCT)
    return e, stop, t1, t2


def long_stop_level(entry: float, atr: float, ma_val: float) -> float:
    """
    Compute an initial stop for a LONG position.

    Generic form used by:
      - SIM: ma_val = MA30
      - Intraday PROD: ma_val = MA150 (if you choose to reuse it)

    stop = max(
        entry * (1 - LONG_HARD_STOP_PCT),
        entry - LONG_TRAIL_ATR_MULT * ATR,
        ma_val * (1 - LONG_MA_GUARD_PCT)
    )
    """
    if _is_nan(entry):
        return np.nan

    hard = entry * (1.0 - LONG_HARD_STOP_PCT)
    atr_stop = entry - LONG_TRAIL_ATR_MULT * atr if not _is_nan(atr) else np.nan
    ma_guard = ma_val * (1.0 - LONG_MA_GUARD_PCT) if not _is_nan(ma_val) else np.nan

    cands = [c for c in (hard, atr_stop, ma_guard) if not _is_nan(c)]
    return max(cands) if cands else hard


def should_exit_long(price: float, stop: float, ma_val: float) -> bool:
    """
    Exit condition for a LONG:

      1) price <= stop  (hard/ATR/MA guard violated)
      2) price has broken under MA by ~LONG_MA_GUARD_PCT
         (extra trend-guard).

    Used directly by the live-logic backtest; you can also reuse this
    in PROD if you want a shared definition.
    """
    if _is_nan(price):
        return False

    # 1) Stop violation
    if not _is_nan(stop) and price <= stop:
        return True

    # 2) Extra guard: under MA by ~3%
    if not _is_nan(ma_val) and price <= ma_val * (1.0 - LONG_MA_GUARD_PCT):
        return True

    return False
