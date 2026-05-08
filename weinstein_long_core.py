#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_long_core.py

Unified LONG-side CORE logic shared by:
- PROD intraday watchers
- SIM backtester
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from weinstein_indicators import ADX_MIN as DEFAULT_ADX_MIN
except Exception:
    DEFAULT_ADX_MIN = 18.0


LONG_BREAK_PCT = 0.004
LONG_VOL_MIN = 1.30

LONG_HARD_STOP_PCT = 0.20
LONG_TRAIL_ATR_MULT = 2.0
LONG_MA_GUARD_PCT = 0.03

LONG_TARGET1_PCT = 0.15
LONG_TARGET2_PCT = 0.20


@dataclass
class LongEntryParams:
    min_break_pct: float = LONG_BREAK_PCT
    dist_above_ma_min: float = 0.0
    vol_min: float = LONG_VOL_MIN
    adx_min: float = DEFAULT_ADX_MIN


@dataclass
class LongEntryResult:
    can_enter: bool
    reason: str
    adx_ok: bool


@dataclass
class LongSignalResult:
    signal: str
    reason: str
    can_enter: bool = False
    adx_ok: bool = True


def _is_nan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def check_long_entry(
    *,
    price: float,
    ma_val: float,
    pivot: float,
    rs_above_ma: bool,
    vol_mult: float,
    adx_val: float,
    params: Optional[LongEntryParams] = None,
) -> LongEntryResult:

    if params is None:
        params = LongEntryParams()

    if _is_nan(price) or _is_nan(ma_val) or _is_nan(pivot):
        return LongEntryResult(False, "nan_input", True)

    if not rs_above_ma:
        return LongEntryResult(False, "rs_not_above_ma", True)

    required_ma = ma_val * (1.0 + params.dist_above_ma_min)

    if price <= required_ma:
        return LongEntryResult(False, "price_not_above_ma", True)

    required_pivot = pivot * (1.0 + params.min_break_pct)

    if price <= required_pivot:
        return LongEntryResult(False, "no_breakout_vs_pivot", True)

    if _is_nan(vol_mult) or vol_mult < params.vol_min:
        return LongEntryResult(False, "volume_too_low", True)

    if not _is_nan(adx_val):
        if adx_val < params.adx_min:
            return LongEntryResult(False, "adx_below_min", False)

    return LongEntryResult(True, "ok", True)


def evaluate_long_signal(
    *,
    price: float,
    ma_val: float,
    pivot: float,
    rs_above_ma: bool = True,
    vol_mult: float = np.nan,
    adx_val: float = np.nan,
    params: Optional[LongEntryParams] = None,
    near_below_pivot_pct: float = 0.01,
    near_vol_min: float = 0.85,
) -> LongSignalResult:

    if params is None:
        params = LongEntryParams()

    buy = check_long_entry(
        price=price,
        ma_val=ma_val,
        pivot=pivot,
        rs_above_ma=rs_above_ma,
        vol_mult=vol_mult,
        adx_val=adx_val,
        params=params,
    )

    if buy.can_enter:
        return LongSignalResult(
            signal="BUY",
            reason=(
                f"BUY: px={price:.2f} pivot={pivot:.2f} "
                f"vol={vol_mult:.2f}x adx={adx_val:.1f}"
            ),
            can_enter=True,
            adx_ok=buy.adx_ok,
        )

    if buy.reason == "adx_below_min":
        return LongSignalResult(
            signal="SKIP-ADX",
            reason=f"ADX14={adx_val:.1f} < {params.adx_min}",
            can_enter=False,
            adx_ok=False,
        )

    if buy.reason == "volume_too_low":
        return LongSignalResult(
            signal="SKIP-VOL",
            reason=(
                f"vol={vol_mult:.2f}x < req={params.vol_min:.2f}x "
                f"price={price:.2f} pivot={pivot:.2f}"
            ),
            can_enter=False,
            adx_ok=buy.adx_ok,
        )

    if buy.reason == "price_not_above_ma":
        required_ma = ma_val * (1.0 + params.dist_above_ma_min)

        return LongSignalResult(
            signal="SKIP-MA",
            reason=f"price={price:.2f} <= req_ma={required_ma:.2f}",
            can_enter=False,
            adx_ok=buy.adx_ok,
        )

    if buy.reason == "rs_not_above_ma":
        return LongSignalResult(
            signal="SKIP-RS",
            reason="relative strength below MA",
            can_enter=False,
            adx_ok=buy.adx_ok,
        )

    near_level = pivot * (1.0 - near_below_pivot_pct)

    if (
        price >= near_level
        and rs_above_ma
        and (not _is_nan(vol_mult))
        and vol_mult >= near_vol_min
    ):
        return LongSignalResult(
            signal="NEAR",
            reason=(
                f"NEAR: px={price:.2f} within "
                f"{near_below_pivot_pct*100:.2f}% of pivot={pivot:.2f}; "
                f"vol={vol_mult:.2f}x"
            ),
            can_enter=False,
            adx_ok=buy.adx_ok,
        )

    return LongSignalResult(
        signal="NONE",
        reason=buy.reason,
        can_enter=False,
        adx_ok=buy.adx_ok,
    )


def long_stop_level(entry: float, atr: float, ma_val: float) -> float:

    if _is_nan(entry):
        return np.nan

    hard = entry * (1.0 - LONG_HARD_STOP_PCT)

    atr_stop = (
        entry - LONG_TRAIL_ATR_MULT * atr
        if not _is_nan(atr)
        else np.nan
    )

    ma_guard = (
        ma_val * (1.0 - LONG_MA_GUARD_PCT)
        if not _is_nan(ma_val)
        else np.nan
    )

    cands = [
        c for c in (hard, atr_stop, ma_guard)
        if not _is_nan(c)
    ]

    return max(cands) if cands else hard


def should_exit_long(price: float, stop: float, ma_val: float) -> bool:

    if _is_nan(price):
        return False

    if not _is_nan(stop) and price <= stop:
        return True

    if (
        not _is_nan(ma_val)
        and price <= ma_val * (1.0 - LONG_MA_GUARD_PCT)
    ):
        return True

    return False


if __name__ == "__main__":
    print("weinstein_long_core loaded successfully")
