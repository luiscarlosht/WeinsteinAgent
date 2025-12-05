# weinstein_long_core.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_long_core.py

Single source of truth for LONG-side breakout logic:

- Base breakout conditions (price vs MA + pivot)
- Volume gate (full-day pace vs 50dma)
- ADX filter (trend strength, with safe NaN handling)
- Config-driven environment gate (Chapter 8 regime, Breadth, Coppock)

Used by:
- weinstein_live_logic_backtest_yfinance.py  (SIM / backtest)
- weinstein_intraday_watcher.py (PROD intraday longs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LongEntryParams:
    """
    Tunables for base LONG breakout conditions.

    min_break_pct       ~ how far above pivot a breakout must be (e.g. 0.004 ≈ 0.4%)
    dist_above_ma_min   ~ how far above MA30/MA150 price must be (0.0 = “just above”)
    vol_min             ~ full-day volume pace vs 50dma (e.g. 1.30x)
    adx_min             ~ minimum ADX(n) to consider trend “strong enough”
    """
    min_break_pct: float = 0.004
    dist_above_ma_min: float = 0.0
    vol_min: float = 1.30
    adx_min: float = 18.0


@dataclass
class LongEntryCheck:
    """
    Result of evaluating base LONG breakout conditions.
    Can be used both in backtest and intraday.

    - rs_ok      : RS regime is acceptable
    - ma_ok      : MA value is present / valid
    - pivot_ok   : pivot is present / valid
    - adx_ok     : ADX filter passes (NaN → True)
    - vol_ok     : full-day volume pace passes (NaN → True)
    - price_ok   : price above MA and pivot by requested margins
    - can_enter  : all of the above are simultaneously satisfied
    """
    rs_ok: bool
    ma_ok: bool
    pivot_ok: bool
    adx_ok: bool
    vol_ok: bool
    price_ok: bool
    can_enter: bool


def _nan(x) -> bool:
    """Small helper to test for NaNs across numpy / Python floats."""
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def price_break_ok(
    price: float,
    ma_val: float,
    pivot: float,
    params: LongEntryParams,
) -> bool:
    """
    Pure price breakout rule used by both intraday & backtest.

    Conditions:
      - price >= MA * (1 + dist_above_ma_min)
      - price >= pivot * (1 + min_break_pct)
    """
    if _nan(price) or _nan(ma_val) or _nan(pivot):
        return False

    # Above MA by at least the configured headroom
    if price < ma_val * (1.0 + params.dist_above_ma_min):
        return False

    # Above pivot by the configured breakout margin
    if price < pivot * (1.0 + params.min_break_pct):
        return False

    return True


def check_long_entry(
    price: float,
    ma_val: float,
    pivot: float,
    rs_above_ma: bool,
    vol_mult: Optional[float],
    adx_val: Optional[float],
    params: LongEntryParams,
) -> LongEntryCheck:
    """
    Evaluate core LONG breakout conditions.

    This is the shared implementation used by:
      - Backtest (daily bars)
      - Intraday watcher (as the “base rule”, on top of intrabar / state logic)

    Notes:
      - ADX filter: NaN → adx_ok=True (safe fallback, don’t block)
      - Volume filter: NaN → vol_ok=True
    """
    rs_ok = bool(rs_above_ma)
    ma_ok = not _nan(ma_val)
    pivot_ok = not _nan(pivot)

    # ADX: NaN means “do not block”
    if _nan(adx_val):
        adx_ok = True
    else:
        adx_ok = float(adx_val) >= float(params.adx_min)

    # Volume: NaN means “do not block”
    if _nan(vol_mult):
        vol_ok = True
    else:
        vol_ok = float(vol_mult) >= float(params.vol_min)

    # Price breakout relative to MA + pivot
    price_ok = price_break_ok(price, ma_val, pivot, params) if (ma_ok and pivot_ok) else False

    can_enter = rs_ok and ma_ok and pivot_ok and adx_ok and vol_ok and price_ok

    return LongEntryCheck(
        rs_ok=rs_ok,
        ma_ok=ma_ok,
        pivot_ok=pivot_ok,
        adx_ok=adx_ok,
        vol_ok=vol_ok,
        price_ok=price_ok,
        can_enter=can_enter,
    )


def compute_long_env_ok(
    *,
    market_long_ok: bool,
    breadth_long_ok: bool,
    coppock_long_ok: bool = True,
    use_ch8: bool = True,
    use_breadth: bool = True,
    use_coppock: bool = True,
) -> bool:
    """
    Config-driven environment gate for NEW LONG entries.

    Parameters:
      - market_long_ok : from Chapter 8 / VIX regime (market_regime.inspect())
      - breadth_long_ok: from breadth (% above MA50) filter
      - coppock_long_ok: from Coppock curve direction (if computed)
      - use_* flags   : whether each filter is active for LONGs

    Behavior:
      - If *no* filters are enabled → returns True (no gating).
      - Otherwise → returns logical AND of enabled filters.
    """
    flags = []

    if use_ch8:
        flags.append(bool(market_long_ok))
    if use_breadth:
        flags.append(bool(breadth_long_ok))
    if use_coppock:
        flags.append(bool(coppock_long_ok))

    if not flags:
        # No filters enabled → environment gate is effectively off.
        return True

    return all(flags)
