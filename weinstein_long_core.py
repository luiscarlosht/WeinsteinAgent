#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_long_core.py

Shared "core" logic for Weinstein-style LONG entries and position management.

This module is intentionally "dumb" and stateless — it does NOT know about:
- market regime
- breadth
- ADX filters
- universe selection

Those gates are applied at the caller level (intraday watcher, backtest engine).

Here you define:
- LongCoreParams dataclass with main tunables
- simple helpers to:
    * check breakout vs pivot
    * compute initial stops
    * update trailing stops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LongCoreParams:
    """Core tunables for long entries and exits."""
    break_pct: float = 0.004    # ~0.4% above pivot
    vol_min: float = 1.30       # min full-day volume vs 50dma
    stop_hard: float = 0.20     # 20% max loss from entry
    trail_atr: float = 2.0      # ATR multiple for trailing stop
    ma_guard: float = 0.03      # guard vs MA30 (~3% cushion)


def passes_volume_filter(
    volume_ratio: float,
    params: LongCoreParams,
) -> bool:
    """
    volume_ratio: today's full-day volume / 50-day avg volume
    """
    return volume_ratio >= params.vol_min


def is_breakout(
    close_price: float,
    pivot_price: float,
    params: LongCoreParams,
) -> bool:
    """
    Basic breakout rule: close above pivot by `break_pct`.
    """
    if pivot_price <= 0:
        return False
    threshold = pivot_price * (1.0 + params.break_pct)
    return close_price >= threshold


def guard_vs_ma(
    close_price: float,
    ma_price: float,
    params: LongCoreParams,
) -> bool:
    """
    Optional extra guard that price should not be too far below MA30.
    Typically you want close >= MA30 * (1 - ma_guard).
    """
    if ma_price <= 0:
        return True
    min_allowed = ma_price * (1.0 - params.ma_guard)
    return close_price >= min_allowed


def initial_stop(
    entry_price: float,
    atr_value: float,
    params: LongCoreParams,
) -> float:
    """
    Initial stop = max( entry * (1 - stop_hard), entry - trail_atr * ATR ).
    """
    hard_stop = entry_price * (1.0 - params.stop_hard)
    atr_stop = entry_price - params.trail_atr * atr_value
    return max(hard_stop, atr_stop)


def update_trailing_stop(
    current_stop: float,
    close_price: float,
    atr_value: float,
    params: LongCoreParams,
) -> float:
    """
    Trailing stop moves up as price rises.
    New stop = max(old_stop, close - trail_atr * ATR).
    Never goes DOWN.
    """
    candidate = close_price - params.trail_atr * atr_value
    return max(current_stop, candidate)


def stop_hit(
    stop_price: float,
    low_price: float,
) -> bool:
    """
    Returns True if intra-day low would have breached the stop.
    """
    return low_price <= stop_price
