#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_regime_exposure_core.py

Shared CORE regime + exposure logic for both SIM and PROD.

D architecture:
- BULL    => allow longs only, full long sizing
- BEAR    => allow shorts only, reduced short sizing
- NEUTRAL => configurable, default conservative long-only at reduced size

This module intentionally has no dependency on the backtester or watcher so both
can call the same decision path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeExposureDecision:
    regime_label: str
    allow_new_longs: bool
    allow_new_shorts: bool
    long_size_mult: float
    short_size_mult: float
    note: str = ""


def _clean_mult(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        return float(default)
    return max(0.0, min(1.0, v))


def _close_series_for_symbol(daily_df: pd.DataFrame, symbol: str) -> pd.Series:
    """Return Close series from either stacked (Date,Ticker) or yfinance wide frames."""
    if daily_df is None or daily_df.empty:
        return pd.Series(dtype="float64")
    sym = str(symbol or "SPY").upper().strip()

    # SIM wide yfinance shape: columns MultiIndex (Field, Ticker), e.g. ('Close','SPY')
    if isinstance(daily_df.columns, pd.MultiIndex):
        candidates = [("Close", sym), (sym, "Close")]
        for col in candidates:
            if col in daily_df.columns:
                return pd.to_numeric(daily_df[col], errors="coerce").dropna()

    # PROD stacked shape: index MultiIndex (Date,Ticker), columns include Close
    if isinstance(daily_df.index, pd.MultiIndex) and "Close" in daily_df.columns:
        try:
            names = list(daily_df.index.names)
            if "Ticker" in names:
                s = daily_df.xs(sym, level="Ticker")["Close"]
            else:
                s = daily_df.xs(sym, level=-1)["Close"]
            return pd.to_numeric(s, errors="coerce").dropna()
        except Exception:
            pass

    # Single-symbol fallback
    if "Close" in daily_df.columns:
        return pd.to_numeric(daily_df["Close"], errors="coerce").dropna()

    return pd.Series(dtype="float64")


def spy_regime_label(
    daily_df: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    market_cfg: Optional[Mapping] = None,
    *,
    benchmark: str = "SPY",
) -> str:
    """
    Weinstein-style broad market regime using benchmark 150d MA as 30-week proxy.

    BULL: benchmark >= MA150 and MA150 slope over 5 sessions >= long slope threshold.
    BEAR: benchmark < MA150 and MA150 slope over 5 sessions <= short slope threshold.
    NEUTRAL: mixed/transition/not enough information.
    """
    market_cfg = market_cfg or {}
    s = _close_series_for_symbol(daily_df, benchmark)
    if s.empty or len(s) < 160:
        return "UNKNOWN"

    s = s.sort_index()
    ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(s.index.max())

    # Use latest available row on/before as_of; PROD intraday may have timestamps.
    prior = s.loc[s.index <= ts]
    if prior.empty:
        return "UNKNOWN"

    ma150 = s.rolling(150, min_periods=150).mean()
    prior_ma = ma150.loc[ma150.index <= prior.index.max()]
    if prior_ma.empty or pd.isna(prior_ma.iloc[-1]):
        return "UNKNOWN"
    if len(prior_ma.dropna()) < 6:
        return "UNKNOWN"

    price = float(prior.iloc[-1])
    ma = float(prior_ma.iloc[-1])
    prev_ma = float(prior_ma.shift(5).iloc[-1])
    if not np.isfinite(price) or not np.isfinite(ma) or not np.isfinite(prev_ma):
        return "UNKNOWN"

    slope = ma - prev_ma
    long_slope_min = float(market_cfg.get("ma30_slope_min", 0.0) or 0.0)
    short_slope_max = float(market_cfg.get("ma30_slope_min_short", 0.0) or 0.0)

    if price >= ma and slope >= long_slope_min:
        return "BULL"
    if price < ma and slope <= short_slope_max:
        return "BEAR"
    return "NEUTRAL"


def regime_permissions_from_label(
    regime_label: str,
    *,
    neutral_policy: str = "long",
) -> Tuple[bool, bool]:
    label = str(regime_label or "UNKNOWN").upper().strip()
    neutral = str(neutral_policy or "long").lower().strip()

    if label == "BULL":
        return True, False
    if label == "BEAR":
        return False, True

    if neutral == "none":
        return False, False
    if neutral == "both":
        return True, True
    # default for NEUTRAL/UNKNOWN is conservative long-only.
    return True, False


def exposure_multipliers_from_label(
    regime_label: str,
    *,
    exposure_mode: str = "scaled",
    bull_long_mult: float = 1.0,
    neutral_long_mult: float = 0.50,
    bear_short_mult: float = 0.60,
    neutral_short_mult: float = 0.0,
) -> Tuple[float, float]:
    em = str(exposure_mode or "scaled").lower().strip()
    label = str(regime_label or "UNKNOWN").upper().strip()

    if em in ("off", "none", "false", "0"):
        return 1.0, 1.0

    if label == "BULL":
        return _clean_mult(bull_long_mult, 1.0), 0.0
    if label == "BEAR":
        return 0.0, _clean_mult(bear_short_mult, 0.60)
    return _clean_mult(neutral_long_mult, 0.50), _clean_mult(neutral_short_mult, 0.0)


def decide_regime_exposure(
    daily_df: pd.DataFrame,
    as_of: Optional[pd.Timestamp],
    market_cfg: Optional[Mapping] = None,
    *,
    benchmark: str = "SPY",
    regime_mode: str = "prod",
    exposure_mode: str = "scaled",
    neutral_policy: str = "long",
    bull_long_mult: float = 1.0,
    neutral_long_mult: float = 0.50,
    bear_short_mult: float = 0.60,
    neutral_short_mult: float = 0.0,
) -> RegimeExposureDecision:
    """Single shared D decision function used by SIM and PROD."""
    rm = str(regime_mode or "prod").lower().strip()

    if rm in ("off", "none", "false", "0"):
        return RegimeExposureDecision("OFF", True, True, 1.0, 1.0, "regime disabled")

    if rm == "prod":
        label = spy_regime_label(daily_df, as_of, market_cfg or {}, benchmark=benchmark)
        allow_long, allow_short = regime_permissions_from_label(label, neutral_policy=neutral_policy)
        long_mult, short_mult = exposure_multipliers_from_label(
            label,
            exposure_mode=exposure_mode,
            bull_long_mult=bull_long_mult,
            neutral_long_mult=neutral_long_mult,
            bear_short_mult=bear_short_mult,
            neutral_short_mult=neutral_short_mult,
        )
        if long_mult <= 0:
            allow_long = False
        if short_mult <= 0:
            allow_short = False
        return RegimeExposureDecision(label, allow_long, allow_short, long_mult, short_mult, "shared D regime/exposure")

    # For legacy/current mode, keep behavior permissive here; caller may still apply legacy gates.
    return RegimeExposureDecision("CURRENT", True, True, 1.0, 1.0, "legacy/current mode")


def read_d_config(cfg: Mapping, *, section: str = "backtest") -> dict:
    """Read D knobs from config.yaml with safe defaults."""
    cfg = cfg or {}
    sec = cfg.get(section, {}) or {}
    d_cfg = sec.get("regime_exposure", {}) or {}
    market_cfg = sec.get("market", {}) or {}
    app_cfg = cfg.get("app", {}) or {}
    return {
        "enabled": bool(d_cfg.get("enabled", False)),
        "benchmark": d_cfg.get("benchmark", app_cfg.get("benchmark", "SPY")),
        "regime_mode": d_cfg.get("regime_mode", "prod"),
        "exposure_mode": d_cfg.get("exposure_mode", "scaled"),
        "neutral_policy": d_cfg.get("neutral_policy", "long"),
        "bull_long_mult": float(d_cfg.get("bull_long_mult", 1.0)),
        "neutral_long_mult": float(d_cfg.get("neutral_long_mult", 0.50)),
        "bear_short_mult": float(d_cfg.get("bear_short_mult", 0.60)),
        "neutral_short_mult": float(d_cfg.get("neutral_short_mult", 0.0)),
        "market_cfg": market_cfg,
    }
