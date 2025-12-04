#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coppock.py

Coppock Curve (CO) indicator for long-side confirmation.

Classic parameters (adapted slightly for automation):
  - ROC1 period: 11 months
  - ROC2 period: 14 months
  - WMA length: 10 months

In daily data, we approximate "months" as 21 trading days each:
  roc1_period = 11 * 21 = 231
  roc2_period = 14 * 21 = 294
  wma_len     = 10 * 21 = 210

We expose:
    - compute_coppock(close: pd.Series, roc1=231, roc2=294, wma_len=210) -> pd.Series
    - is_coppock_bullish(coppock: pd.Series, lookback=3, threshold=0.0) -> bool

Usage in your backtest / live logic:
    from coppock import compute_coppock, is_coppock_bullish

    close = daily_df["Close"]
    co = compute_coppock(close)
    if is_coppock_bullish(co):
        # allow new LONG entries
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wma(series: pd.Series, length: int) -> pd.Series:
    """
    Weighted Moving Average with linearly increasing weights.
    """
    if length <= 0:
        raise ValueError("WMA length must be positive.")

    weights = np.arange(1, length + 1, dtype=float)
    weights = weights / weights.sum()

    # Use rolling apply with dot-product
    return series.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)


def compute_coppock(
    close: pd.Series,
    roc1: int = 231,
    roc2: int = 294,
    wma_len: int = 210,
) -> pd.Series:
    """
    Compute Coppock Curve from a close-price series.

    Args:
        close:    pd.Series of close prices indexed by datetime
        roc1:     First rate-of-change lookback in days (default ~11 months)
        roc2:     Second rate-of-change lookback in days (default ~14 months)
        wma_len:  WMA smoothing length in days (default ~10 months)

    Returns:
        pd.Series of Coppock values, aligned with `close` index.
    """
    close = close.astype(float)

    if roc1 <= 0 or roc2 <= 0:
        raise ValueError("roc1 and roc2 must be positive.")
    if wma_len <= 0:
        raise ValueError("wma_len must be positive.")

    # Rate of change in percent
    roc1_series = (close / close.shift(roc1) - 1.0) * 100.0
    roc2_series = (close / close.shift(roc2) - 1.0) * 100.0

    coppock_raw = roc1_series + roc2_series
    coppock_smoothed = _wma(coppock_raw, wma_len)

    return coppock_smoothed


def is_coppock_bullish(
    coppock: pd.Series,
    lookback: int = 3,
    threshold: float = 0.0,
) -> bool:
    """
    Simple bullish filter:

        - Coppock must be above `threshold` (usually 0.0)
        - Coppock must be rising over the last `lookback` bars

    Args:
        coppock:  pd.Series as returned by compute_coppock()
        lookback: how many most-recent bars to check for rising slope
        threshold: minimum Coppock value to be considered bullish

    Returns:
        bool indicating whether Coppock is bullish "now" (at the last index).
    """
    if coppock.empty:
        return False

    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    last_values = coppock.dropna().iloc[-lookback:]
    if last_values.empty:
        return False

    # Above threshold?
    if not (last_values.iloc[-1] > threshold):
        return False

    # Rising? (simple check: last value greater than the earliest in the window)
    if last_values.iloc[-1] <= last_values.iloc[0]:
        return False

    return True
