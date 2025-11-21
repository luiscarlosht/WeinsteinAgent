# market_regime.py
# -*- coding: utf-8 -*-
"""
Market Regime Detector (Weinstein Chapter 8)

Determines if overall market is in Stage 2 (bull uptrend) or not.
Used to filter long-trigger alerts.
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ─────────────────────────────────────────────
# Fetch weekly index data
# ─────────────────────────────────────────────
def fetch_weekly_data(ticker: str = "SPY", weeks: int = 104) -> pd.DataFrame:
    """
    Download ~2 years of weekly candles for the index (default SPY)
    """
    data = yf.download(ticker, period="2y", interval="1wk", auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    return data.tail(weeks).copy()

# ─────────────────────────────────────────────
# Classify Stage per Weinstein
# ─────────────────────────────────────────────
def detect_stage(df: pd.DataFrame, ma_len: int = 30) -> str:
    """
    Classify Stage 1/2/3/4 based on weekly price + 30w MA (Weinstein method).
    """
    if df.empty or "Close" not in df.columns:
        return "UNKNOWN"

    df = df.copy()
    df["MA"] = df["Close"].rolling(ma_len).mean()

    if df["MA"].isna().all():
        return "UNKNOWN"

    price = df["Close"].iloc[-1]
    ma_now = df["MA"].iloc[-1]
    ma_prev = df["MA"].iloc[-4]  # slope check 4 weeks ago

    # Stage logic
    if price > ma_now and ma_now > ma_prev:
        return "STAGE_2"  # Bull trend
    if price < ma_now and ma_now < ma_prev:
        return "STAGE_4"  # Bear trend
    if price < ma_now and ma_now > ma_prev:
        return "STAGE_3"  # Topping process
    if price > ma_now and ma_now < ma_prev:
        return "STAGE_1"  # Bottoming process

    return "UNKNOWN"

# ─────────────────────────────────────────────
# Public helper: Is the market bullish?
# ─────────────────────────────────────────────
def market_is_bull(ticker: str = "SPY") -> bool:
    """
    True ONLY if market index is in Stage 2.
    """
    df = fetch_weekly_data(ticker)
    stage = detect_stage(df)
    return stage == "STAGE_2"

# ─────────────────────────────────────────────
# Optional: return stage info for logging/debug
# ─────────────────────────────────────────────
def market_stage(ticker: str = "SPY") -> str:
    df = fetch_weekly_data(ticker)
    return detect_stage(df)
