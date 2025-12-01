#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_indicators.py

Shared technical indicator helpers for Weinstein-style systems.

This module is the single source of truth for:
- ADX_WINDOW, ADX_MIN
- ADX computation helpers used by:
    * weinstein_intraday_watcher.py  (production intraday)
    * weinstein_live_logic_backtest_yfinance.py  (daily backtest)

Design:
- compute_adx_series(df, n=ADX_WINDOW):
    * df: single-ticker daily OHLC DataFrame with columns ['High','Low','Close']
    * returns: ADX(n) series aligned to df.index (NaN when not enough data)
- compute_adx_for_ticker(daily_df, ticker, n=ADX_WINDOW):
    * daily_df: multi-ticker yfinance panel (MultiIndex columns) or single-ticker df
    * ticker: symbol string
    * returns: float ADX(n) (last available), or NaN if not enough data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- ADX parameters (single source of truth) ---
ADX_WINDOW = 14
ADX_MIN = 22.0  # intraday & backtest will both use this unless overridden here


def _adx_from_hlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
) -> pd.Series:
    """
    Core ADX(n) computation from H/L/C series.

    Returns a Series of ADX values indexed like the (cleaned) input.
    This mirrors the logic previously used inside weinstein_intraday_watcher.py.
    """
    df = pd.DataFrame({"High": high, "Low": low, "Close": close}).dropna()
    if len(df) < n + 2:
        return pd.Series(index=df.index, dtype="float64")

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_n = tr.rolling(n).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(n).sum() / tr_n)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(n).sum() / tr_n)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100.0
    adx = dx.rolling(n).mean()

    return adx


def compute_adx_series(df: pd.DataFrame, n: int = ADX_WINDOW) -> pd.Series:
    """
    Compute ADX(n) for a single-ticker daily OHLC dataframe.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'High', 'Low', 'Close'.
    n : int
        ADX window length (default: ADX_WINDOW).

    Returns
    -------
    Series
        ADX values aligned to df.index (NaN where not enough data).
    """
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype="float64")

    adx_core = _adx_from_hlc(df["High"], df["Low"], df["Close"], n=n)
    # Re-align back to the full original index
    return adx_core.reindex(df.index)


def compute_adx_for_ticker(
    daily_df: pd.DataFrame,
    ticker: str,
    n: int = ADX_WINDOW,
) -> float:
    """
    Compute ADX(n) for a specific ticker from a yfinance-style DAILY panel.

    Parameters
    ----------
    daily_df : DataFrame
        Either:
        - MultiIndex columns (['Open','High','Low','Close','Adj Close','Volume'], ticker)
        - Single-ticker OHLCV with the same column names.
    ticker : str
        Symbol to slice out of a MultiIndex panel.
    n : int
        ADX window length (default: ADX_WINDOW).

    Returns
    -------
    float
        The last available ADX value, or NaN if not enough data.
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            sub = daily_df.xs(ticker, axis=1, level=1)
        except KeyError:
            return np.nan
    else:
        sub = daily_df

    required = {"High", "Low", "Close"}
    if not required.issubset(sub.columns):
        return np.nan

    adx_series = compute_adx_series(sub[["High", "Low", "Close"]], n=n)
    adx_clean = adx_series.dropna()
    if not len(adx_clean):
        return np.nan
    return float(adx_clean.iloc[-1])
