#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_indicators.py

Shared technical indicator helpers for Weinstein-style systems.

This module is the single source of truth for:
- ADX parameters + computation
- Breadth Health (% of tickers above MA50)

Used by:
    * weinstein_intraday_watcher.py   (PROD intraday)
    * weinstein_live_logic_backtest_yfinance.py   (SIM/backtest)
    * Any dashboards or performance analyzers

Functions provided:

    compute_adx_series(df)
    compute_adx_for_ticker(daily_df, ticker)

    compute_breadth_series_above_ma(close_panel, tickers, ma_window=50)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================
# ADX (Average Directional Index)
# ============================================================================

# --- ADX parameters (single source of truth) ---
ADX_WINDOW = 14
ADX_MIN = 22.0   # intraday & backtest will both use this unless overridden


def _adx_from_hlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
) -> pd.Series:
    """
    Core ADX(n) computation from H/L/C series.

    Returns a Series of ADX values indexed like the (cleaned) input.
    """
    df = pd.DataFrame({"High": high, "Low": low, "Close": close}).dropna()
    if len(df) < n + 2:
        return pd.Series(index=df.index, dtype="float64")

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    # Directional movements
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

    df must contain: ['High','Low','Close']
    """
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype="float64")

    adx_core = _adx_from_hlc(df["High"], df["Low"], df["Close"], n=n)
    return adx_core.reindex(df.index)


def compute_adx_for_ticker(
    daily_df: pd.DataFrame,
    ticker: str,
    n: int = ADX_WINDOW,
) -> float:
    """
    Compute ADX(n) for a specific ticker from yfinance-style panel.

    Supports:
        - MultiIndex: daily_df[("High", ticker)]
        - Flat OHLC dataframe: daily_df['High'], etc.
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


# ============================================================================
# BREADTH HEALTH (Advance/Decline Strength Filter)
# ============================================================================

def compute_breadth_series_above_ma(
    daily_close_panel: pd.DataFrame,
    tickers: list[str],
    ma_window: int = 50,
) -> pd.Series:
    """
    Compute % of tickers above MA `ma_window`.

    Parameters
    ----------
    daily_close_panel : DataFrame
        Either:
            - MultiIndex columns: ("Close", ticker)
            - Flat columns: ticker names containing daily close prices
    tickers : list[str]
        Universe to measure breadth on.
    ma_window : int
        MA window (default: 50)

    Returns
    -------
    Series (float)
        breadth(t) = (# tickers with Close > MA) / (# tickers with valid data)
        indexed by trading date.
    """
    if not tickers or daily_close_panel.empty:
        return pd.Series(dtype=float)

    # Normalize to flat Close-panel: date × ticker
    if isinstance(daily_close_panel.columns, pd.MultiIndex):
        if "Close" not in daily_close_panel.columns.levels[0]:
            # No Close data available
            return pd.Series(dtype=float)
        close_panel = daily_close_panel["Close"]
    else:
        close_panel = daily_close_panel

    # Keep only tickers we actually have data for
    cols = [t for t in tickers if t in close_panel.columns]
    if not cols:
        return pd.Series(dtype=float)

    close_panel = close_panel[cols]

    # Rolling MA for each ticker
    ma = close_panel.rolling(ma_window).mean()

    # Boolean matrix: True = Close > MA
    above = close_panel > ma

    # Breadth = fraction above MA
    breadth = above.sum(axis=1) / above.count(axis=1)
    breadth.name = f"breadth_above_ma{ma_window}"

    return breadth
