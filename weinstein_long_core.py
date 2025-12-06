#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_long_core.py

Shared LONG-side entry logic for:
- Intraday PROD watchers
- Daily SIM backtester (weinstein_live_logic_backtest_yfinance.py)

Core idea:
- One place where we define:
    * thresholds (breakout %, min distance above MA, vol pace, ADX)
    * "can we enter?" decision logic
- Anything that wants to do a Weinstein Stage 2 breakout
  just calls check_long_entry(...) and inspects the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# We reuse the canonical ADX_MIN from weinstein_indicators so that
# PROD, SIM, and any other tools stay in sync on the threshold.
try:
    from weinstein_indicators import ADX_MIN as DEFAULT_ADX_MIN
except ImportError:
    # Fallback, in case the module is missing or renamed.
    DEFAULT_ADX_MIN = 18.0


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
    min_break_pct: float = 0.004
    dist_above_ma_min: float = 0.0
    vol_min: float = 1.30
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
    # Preferred interface (used by SIM backtester):
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
            MA(30) (or your equivalent trend MA).
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

    # --- Price must be above MA30 with optional extra headroom ---
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
