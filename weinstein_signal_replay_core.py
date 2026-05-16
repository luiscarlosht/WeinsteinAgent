#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_signal_replay_core.py

Shared signal replay engine for WeinsteinAgent.

Purpose:
- One source of truth for historical PROD-like BUY / NEAR / SELL events.
- No simulated cash, no fake portfolio, no equity curve.
- Portfolio backtests can later consume these events instead of re-deciding signals.

This module intentionally reuses the existing backtest CORE helpers so the first
refactor step is low-risk: same universe snapshots, same long/short CORE gates,
same D regime/exposure decisions, same signal-quality knobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from industry_filters import IndustryFilterConfig, enrich_with_industry_and_stats, industry_ok_from_row
from weinstein_indicators import compute_adx_series, ADX_MIN
from weinstein_long_core import LongEntryParams, check_long_entry, should_exit_long
from weinstein_short_core import ShortEntryParams, check_short_entry, should_exit_short as core_should_exit_short

# Reuse existing helper functions from the current research backtester.
# This avoids creating a disconnected signal implementation.
from weinstein_live_logic_backtest_yfinance import (
    get_panel,
    compute_atr_series_from_ohlc,
    pick_snapshot_for_date,
    _is_stage2,
    _is_stage4,
    stock_ma30_slope_ok_from_snapshot,
    short_slope_ok_from_snapshot,
    short_failed_rally_ok,
    _regime_permissions,
    _regime_exposure_multipliers,
    _get_snapshot_rs_above_ma,
    _long_signal_quality_score,
    _short_signal_quality_score,
    _long_signal_quality_strict_ok,
    _short_signal_quality_strict_ok,
    _quality_score_multiplier,
)


def _safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _pct_from_level(price: float, level: float) -> float:
    """Return percent distance from level: + means price above level."""
    try:
        price_f = float(price)
        level_f = float(level)
        if not np.isfinite(price_f) or not np.isfinite(level_f) or level_f == 0:
            return np.nan
        return (price_f / level_f - 1.0) * 100.0
    except Exception:
        return np.nan


def _yesno(v) -> str:
    return "yes" if bool(v) else "no"


def _fmt_num(x, digits=2) -> str:
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "n/a"
        return f"{xf:.{digits}f}"
    except Exception:
        return "n/a"


def _fmt_pct(x, digits=2) -> str:
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "n/a"
        return f"{xf:.{digits}f}%"
    except Exception:
        return "n/a"


def _explain_long_event(*, signal: str, price: float, pivot: float, ma30: float, ma150: float,
                        vol_mult: float, vol_min: float, adx_val: float, adx_min: float,
                        rs_above: bool, regime: str, long_mult: float, quality_score: float,
                        quality_mult: float, quality_mode: str, reason: str) -> str:
    pivot_dist = _pct_from_level(price, pivot)
    ma30_dist = _pct_from_level(price, ma30)
    ma150_dist = _pct_from_level(price, ma150)
    pieces = [
        f"{signal}: {reason}",
        f"regime={regime}",
        f"long_mult={_fmt_num(long_mult, 2)}",
        f"price={_fmt_num(price)}",
        f"pivot={_fmt_num(pivot)}",
        f"pivot_dist={_fmt_pct(pivot_dist)}",
        f"MA30_dist={_fmt_pct(ma30_dist)}",
        f"MA150_dist={_fmt_pct(ma150_dist)}",
        f"vol={_fmt_num(vol_mult, 2)}x vs min {_fmt_num(vol_min, 2)}x",
        f"ADX={_fmt_num(adx_val, 1)} vs min {_fmt_num(adx_min, 1)}",
        f"RS_above_MA={_yesno(rs_above)}",
    ]
    if str(quality_mode or "off").lower() != "off":
        pieces.append(f"quality={_fmt_num(quality_score, 1)}")
        pieces.append(f"quality_mult={_fmt_num(quality_mult, 2)}")
    return " | ".join(pieces)


def _explain_sell_event(*, price: float, ma150: float, ma30: float, sell_crack_pct: float,
                        vol_mult: float, adx_val: float, regime: str, source: str) -> str:
    ma150_dist = _pct_from_level(price, ma150)
    ma30_dist = _pct_from_level(price, ma30)
    crack_level = ma150 * (1.0 - sell_crack_pct) if np.isfinite(ma150) else np.nan
    return " | ".join([
        "SELL: sell_below_ma150_crack",
        f"source={source}",
        f"regime={regime}",
        f"price={_fmt_num(price)}",
        f"MA150={_fmt_num(ma150)}",
        f"crack_level={_fmt_num(crack_level)}",
        f"MA150_dist={_fmt_pct(ma150_dist)}",
        f"MA30_dist={_fmt_pct(ma30_dist)}",
        f"vol={_fmt_num(vol_mult, 2)}x",
        f"ADX={_fmt_num(adx_val, 1)}",
    ])


def _explain_short_event(*, price: float, pivot_low: float, ma30: float, ma150: float,
                         vol_mult: float, vol_min: float, rs_above: bool, regime: str,
                         short_mult: float, quality_score: float, quality_mult: float,
                         quality_mode: str, reason: str) -> str:
    pivot_dist = _pct_from_level(price, pivot_low)
    ma30_dist = _pct_from_level(price, ma30)
    ma150_dist = _pct_from_level(price, ma150)
    pieces = [
        f"SHORT: {reason}",
        f"regime={regime}",
        f"short_mult={_fmt_num(short_mult, 2)}",
        f"price={_fmt_num(price)}",
        f"pivot_low={_fmt_num(pivot_low)}",
        f"pivot_dist={_fmt_pct(pivot_dist)}",
        f"MA30_dist={_fmt_pct(ma30_dist)}",
        f"MA150_dist={_fmt_pct(ma150_dist)}",
        f"vol={_fmt_num(vol_mult, 2)}x vs min {_fmt_num(vol_min, 2)}x",
        f"RS_above_MA={_yesno(rs_above)}",
    ]
    if str(quality_mode or "off").lower() != "off":
        pieces.append(f"quality={_fmt_num(quality_score, 1)}")
        pieces.append(f"quality_mult={_fmt_num(quality_mult, 2)}")
    return " | ".join(pieces)


@dataclass(frozen=True)
class ReplaySignal:
    date: pd.Timestamp
    ticker: str
    side: str              # long | short | portfolio
    signal: str            # BUY | NEAR | SELL | SELL-WATCH
    reason: str
    price: float = np.nan
    pivot: float = np.nan
    ma30: float = np.nan
    ma150: float = np.nan
    atr14: float = np.nan
    vol_mult: float = np.nan
    adx14: float = np.nan
    regime: str = ""
    allow_long: bool = True
    allow_short: bool = True
    long_size_mult: float = 1.0
    short_size_mult: float = 1.0
    quality_score: float = np.nan
    quality_mult: float = 1.0
    stage: str = ""
    source: str = "replay"
    explanation: str = ""
    pivot_distance_pct: float = np.nan
    ma30_distance_pct: float = np.nan
    ma150_distance_pct: float = np.nan
    vol_min_required: float = np.nan
    adx_min_required: float = np.nan
    rs_above_ma: Optional[bool] = None
    explain_version: str = "v1"


def replay_signals(
    *,
    daily_df: pd.DataFrame,
    start: str,
    end: str,
    mode: str,
    universe_tickers: List[str],
    weekly_df: Optional[pd.DataFrame],
    weekly_snapshots: Optional[List[Tuple[date, pd.DataFrame]]],
    long_logic_cfg: Mapping,
    short_logic_cfg: Mapping,
    market_cfg: Mapping,
    industry_cfg: Mapping,
    regime_mode: str = "current",
    neutral_policy: str = "long",
    exposure_mode: str = "off",
    bull_long_mult: float = 1.0,
    neutral_long_mult: float = 0.50,
    bear_short_mult: float = 0.60,
    neutral_short_mult: float = 0.0,
    signal_quality_mode: str = "off",
    min_long_quality: float = 65.0,
    min_short_quality: float = 65.0,
    adaptive_reject_below: float = 60.0,
    adaptive_floor_mult: float = 0.40,
    adaptive_mid_mult: float = 0.65,
    adaptive_good_mult: float = 0.85,
    adaptive_elite_mult: float = 1.00,
    include_near: bool = True,
    include_raw_sell: bool = True,
    sell_scope: str = "raw",
    sell_tickers: Optional[List[str]] = None,
    near_zone_pct: float = 0.01,
    sell_crack_pct: float = 0.005,
) -> pd.DataFrame:
    """Return historical PROD-like signal events, with no portfolio simulation."""

    industry_filter_cfg = IndustryFilterConfig(**(industry_cfg or {}))

    if weekly_df is not None and not weekly_df.empty:
        weekly_df = enrich_with_industry_and_stats(weekly_df, cfg=industry_filter_cfg)
    if weekly_snapshots:
        weekly_snapshots = [(d, enrich_with_industry_and_stats(df, cfg=industry_filter_cfg)) for d, df in weekly_snapshots]

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    close_cache: Dict[str, pd.Series] = {}
    vol_cache: Dict[str, pd.Series] = {}
    ma30_cache: Dict[str, pd.Series] = {}
    ma150_cache: Dict[str, pd.Series] = {}
    atr_cache: Dict[str, pd.Series] = {}
    vol_mult_cache: Dict[str, pd.Series] = {}
    adx_cache: Dict[str, pd.Series] = {}

    for t in sorted(set(str(x).upper().strip() for x in universe_tickers if str(x).strip())):
        close = get_panel(daily_df, "Close", t)
        high = get_panel(daily_df, "High", t)
        low = get_panel(daily_df, "Low", t)
        vol = get_panel(daily_df, "Volume", t)
        if close.empty or high.empty or low.empty or vol.empty:
            continue
        close_cache[t] = close
        vol_cache[t] = vol
        ma30_cache[t] = close.rolling(30, min_periods=30).mean()
        ma150_cache[t] = close.rolling(150, min_periods=150).mean()
        atr_cache[t] = compute_atr_series_from_ohlc(high, low, close, n=14)
        v50 = vol.rolling(50, min_periods=50).mean()
        vol_mult_cache[t] = vol / v50
        try:
            adx_cache[t] = compute_adx_series(pd.DataFrame({"High": high, "Low": low, "Close": close}), n=14)
        except Exception:
            adx_cache[t] = pd.Series(index=close.index, dtype="float64")

    sh_break_pct = float(short_logic_cfg.get("break_pct", 0.006))
    sh_vol_min = float(short_logic_cfg.get("vol_min", 1.10))
    sh_pivot_lb = int(short_logic_cfg.get("pivot_lookback_days", short_logic_cfg.get("pivot_lookback", 50)))
    if sh_pivot_lb < 10:
        sh_pivot_lb = 50

    pivot_lb = int(long_logic_cfg.get("pivot_lookback_days", long_logic_cfg.get("pivot_lookback", 60)))
    long_core_params = LongEntryParams(
        min_break_pct=float(long_logic_cfg.get("break_pct", long_logic_cfg.get("min_break_pct", 0.004))),
        dist_above_ma_min=float(long_logic_cfg.get("dist_above_ma_min", 0.0)),
        vol_min=float(long_logic_cfg.get("vol_min", long_logic_cfg.get("vol_pace_min", 1.3))),
        adx_min=float(long_logic_cfg.get("adx_min_long", long_logic_cfg.get("adx_min", ADX_MIN))),
    )

    events: List[ReplaySignal] = []
    all_dates = list(pd.to_datetime(daily_df.index))
    quality_mode = str(signal_quality_mode or "off").strip().lower()
    sell_scope_norm = str(sell_scope or ("raw" if include_raw_sell else "none")).strip().lower()
    if not include_raw_sell:
        sell_scope_norm = "none"
    if sell_scope_norm not in {"none", "raw", "universe", "holdings"}:
        sell_scope_norm = "raw"
    sell_ticker_set = {str(x).upper().strip() for x in (sell_tickers or []) if str(x).strip()}
    do_longs = mode in ("long", "both", "auto")
    do_shorts = mode in ("short", "both", "auto")

    for dt in all_dates:
        if dt < start_dt or dt > end_dt:
            continue

        snap = pick_snapshot_for_date(weekly_snapshots, dt) if weekly_snapshots else None
        universe = snap[1] if snap else (weekly_snapshots[0][1] if weekly_snapshots else weekly_df)
        if universe is None or universe.empty:
            continue

        allow_long, allow_short, regime_label = _regime_permissions(
            daily_df, dt, market_cfg, regime_mode=regime_mode, neutral_policy=neutral_policy
        )
        long_mult, short_mult = _regime_exposure_multipliers(
            regime_label,
            regime_mode=regime_mode,
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

        universe_ticker_set = {
            str(r.get("ticker", r.get("Ticker", ""))).upper().strip()
            for _, r in universe.iterrows()
        }
        universe_ticker_set = {x for x in universe_ticker_set if x}

        # SELL replay is deliberately separated from BUY/NEAR universe evaluation.
        # PROD parity normally wants BUY/NEAR from the discovery universe but SELL
        # only for owned holdings.  Raw/universe SELL remains available for research.
        if sell_scope_norm != "none":
            if sell_scope_norm == "holdings":
                sell_candidates = sell_ticker_set
            else:
                sell_candidates = universe_ticker_set if sell_scope_norm == "universe" else set(close_cache.keys())
            for st in sorted(sell_candidates):
                if not st or st not in close_cache or dt not in close_cache[st].index:
                    continue
                s_price = close_cache[st].loc[dt]
                if pd.isna(s_price):
                    continue
                s_price_f = float(s_price)
                s_ma30 = ma30_cache.get(st, pd.Series(dtype="float64")).get(dt, np.nan)
                s_ma150 = ma150_cache.get(st, pd.Series(dtype="float64")).get(dt, np.nan)
                s_atr = atr_cache.get(st, pd.Series(dtype="float64")).get(dt, np.nan)
                s_vol = vol_mult_cache.get(st, pd.Series(dtype="float64")).get(dt, np.nan)
                s_adx = adx_cache.get(st, pd.Series(dtype="float64")).get(dt, np.nan)
                s_ma30_f = float(s_ma30) if pd.notna(s_ma30) else np.nan
                s_ma150_f = float(s_ma150) if pd.notna(s_ma150) else np.nan
                if np.isfinite(s_ma150_f) and s_price_f <= s_ma150_f * (1.0 - float(sell_crack_pct)):
                    sell_source = f"{sell_scope_norm}_sell_replay"
                    s_vol_f = float(s_vol) if pd.notna(s_vol) else np.nan
                    s_adx_f = float(s_adx) if pd.notna(s_adx) else np.nan
                    events.append(ReplaySignal(
                        date=dt, ticker=st, side="long", signal="SELL", reason="sell_below_ma150_crack",
                        price=s_price_f, pivot=np.nan, ma30=s_ma30_f, ma150=s_ma150_f,
                        atr14=float(s_atr) if pd.notna(s_atr) else np.nan,
                        vol_mult=s_vol_f,
                        adx14=s_adx_f,
                        regime=regime_label, allow_long=allow_long, allow_short=allow_short,
                        long_size_mult=long_mult, short_size_mult=short_mult, stage="risk",
                        source=sell_source,
                        explanation=_explain_sell_event(
                            price=s_price_f, ma150=s_ma150_f, ma30=s_ma30_f,
                            sell_crack_pct=float(sell_crack_pct), vol_mult=s_vol_f,
                            adx_val=s_adx_f, regime=regime_label, source=sell_source,
                        ),
                        ma30_distance_pct=_pct_from_level(s_price_f, s_ma30_f),
                        ma150_distance_pct=_pct_from_level(s_price_f, s_ma150_f),
                    ))

        for _, row in universe.iterrows():
            t = str(row.get("ticker", row.get("Ticker", ""))).upper().strip()
            if not t or t not in close_cache:
                continue
            if dt not in close_cache[t].index:
                continue

            price = close_cache[t].loc[dt]
            ma30 = ma30_cache.get(t, pd.Series(dtype="float64")).get(dt, np.nan)
            ma150 = ma150_cache.get(t, pd.Series(dtype="float64")).get(dt, np.nan)
            atr = atr_cache.get(t, pd.Series(dtype="float64")).get(dt, np.nan)
            vol_mult = vol_mult_cache.get(t, pd.Series(dtype="float64")).get(dt, np.nan)
            adx_val = adx_cache.get(t, pd.Series(dtype="float64")).get(dt, np.nan)

            if pd.isna(price):
                continue
            price_f = float(price)
            ma30_f = float(ma30) if pd.notna(ma30) else np.nan
            ma150_f = float(ma150) if pd.notna(ma150) else np.nan
            atr_f = float(atr) if pd.notna(atr) else np.nan
            vol_f = float(vol_mult) if pd.notna(vol_mult) else np.nan
            adx_f = float(adx_val) if pd.notna(adx_val) else np.nan

            # Long BUY / NEAR replay
            if do_longs and allow_long:
                if not _is_stage2(row):
                    pass
                elif not stock_ma30_slope_ok_from_snapshot(row, long_logic_cfg):
                    pass
                elif not industry_ok_from_row(row, cfg=industry_filter_cfg):
                    pass
                elif not np.isfinite(ma30_f) or not np.isfinite(atr_f):
                    pass
                else:
                    cs = close_cache[t]
                    prior = cs.loc[:dt]
                    if len(prior) >= (pivot_lb + 1):
                        pivot = float(prior.iloc[:-1].tail(pivot_lb).max())
                        rs_above = _get_snapshot_rs_above_ma(row)
                        if rs_above is None:
                            rs_above = bool(long_logic_cfg.get("default_rs_above_ma", True))
                        core_entry = check_long_entry(
                            price=price_f, ma_val=ma30_f, pivot=pivot, rs_above_ma=bool(rs_above),
                            vol_mult=vol_f, adx_val=adx_f, params=long_core_params,
                        )
                        q_score = np.nan
                        q_mult = 1.0
                        q_ok = True
                        if quality_mode in ("score", "strict", "adaptive"):
                            q_score = _long_signal_quality_score(
                                row, price=price_f, ma_val=ma30_f, pivot=pivot, vol_mult=vol_f,
                                adx_val=adx_f, atr_val=atr_f, rs_above_ma=bool(rs_above),
                            )
                            if quality_mode == "adaptive":
                                q_mult = _quality_score_multiplier(
                                    q_score, reject_below=adaptive_reject_below,
                                    floor_mult=adaptive_floor_mult, mid_mult=adaptive_mid_mult,
                                    good_mult=adaptive_good_mult, elite_mult=adaptive_elite_mult,
                                )
                                q_ok = q_mult > 0
                            else:
                                q_ok = q_score >= float(min_long_quality)
                                if q_ok and quality_mode == "strict":
                                    q_ok = _long_signal_quality_strict_ok(
                                        row, price=price_f, ma_val=ma30_f, pivot=pivot,
                                        vol_mult=vol_f, adx_val=adx_f, atr_val=atr_f,
                                        rs_above_ma=bool(rs_above),
                                    )
                        if core_entry.can_enter and q_ok:
                            events.append(ReplaySignal(
                                date=dt, ticker=t, side="long", signal="BUY", reason=core_entry.reason,
                                price=price_f, pivot=pivot, ma30=ma30_f, ma150=ma150_f, atr14=atr_f,
                                vol_mult=vol_f, adx14=adx_f, regime=regime_label,
                                allow_long=allow_long, allow_short=allow_short, long_size_mult=long_mult,
                                short_size_mult=short_mult, quality_score=q_score, quality_mult=q_mult,
                                stage="Stage 2", source="core_long",
                                explanation=_explain_long_event(
                                    signal="BUY", price=price_f, pivot=pivot, ma30=ma30_f, ma150=ma150_f,
                                    vol_mult=vol_f, vol_min=long_core_params.vol_min,
                                    adx_val=adx_f, adx_min=long_core_params.adx_min,
                                    rs_above=bool(rs_above), regime=regime_label, long_mult=long_mult,
                                    quality_score=q_score, quality_mult=q_mult, quality_mode=quality_mode,
                                    reason=core_entry.reason,
                                ),
                                pivot_distance_pct=_pct_from_level(price_f, pivot),
                                ma30_distance_pct=_pct_from_level(price_f, ma30_f),
                                ma150_distance_pct=_pct_from_level(price_f, ma150_f),
                                vol_min_required=long_core_params.vol_min,
                                adx_min_required=long_core_params.adx_min,
                                rs_above_ma=bool(rs_above),
                            ))
                        elif include_near:
                            # PROD-like early watchlist: structurally valid Stage 2 near pivot but not a full BUY.
                            near_zone = float(near_zone_pct)
                            if pivot > 0 and price_f >= pivot * (1.0 - near_zone) and bool(rs_above) and price_f > ma30_f:
                                events.append(ReplaySignal(
                                    date=dt, ticker=t, side="long", signal="NEAR", reason=core_entry.reason,
                                    price=price_f, pivot=pivot, ma30=ma30_f, ma150=ma150_f, atr14=atr_f,
                                    vol_mult=vol_f, adx14=adx_f, regime=regime_label,
                                    allow_long=allow_long, allow_short=allow_short, long_size_mult=long_mult,
                                    short_size_mult=short_mult, quality_score=q_score, quality_mult=q_mult,
                                    stage="Stage 2", source="core_long_near",
                                    explanation=_explain_long_event(
                                        signal="NEAR", price=price_f, pivot=pivot, ma30=ma30_f, ma150=ma150_f,
                                        vol_mult=vol_f, vol_min=long_core_params.vol_min,
                                        adx_val=adx_f, adx_min=long_core_params.adx_min,
                                        rs_above=bool(rs_above), regime=regime_label, long_mult=long_mult,
                                        quality_score=q_score, quality_mult=q_mult, quality_mode=quality_mode,
                                        reason=core_entry.reason,
                                    ),
                                    pivot_distance_pct=_pct_from_level(price_f, pivot),
                                    ma30_distance_pct=_pct_from_level(price_f, ma30_f),
                                    ma150_distance_pct=_pct_from_level(price_f, ma150_f),
                                    vol_min_required=long_core_params.vol_min,
                                    adx_min_required=long_core_params.adx_min,
                                    rs_above_ma=bool(rs_above),
                                ))

            # Short BUY equivalent in replay: SHORT signal when PROD/research short CORE would enter.
            if do_shorts and allow_short:
                if not _is_stage4(row):
                    continue
                if not short_slope_ok_from_snapshot(row, short_logic_cfg):
                    continue
                if not industry_ok_from_row(row, cfg=industry_filter_cfg):
                    continue
                if t not in ma30_cache or not np.isfinite(ma30_f) or not np.isfinite(atr_f):
                    continue
                if not short_failed_rally_ok(close_cache[t], dt, short_logic_cfg, ticker=t):
                    continue
                prior = close_cache[t].loc[:dt]
                if len(prior) < (sh_pivot_lb + 1):
                    continue
                pivot_low = float(prior.iloc[:-1].tail(sh_pivot_lb).min())
                rs_above = _get_snapshot_rs_above_ma(row)
                if rs_above is None:
                    rs_above = False
                short_res = check_short_entry(
                    price=price_f, ma_val=ma30_f, pivot_low=pivot_low, rs_above_ma=bool(rs_above),
                    vol_mult=vol_f, params=ShortEntryParams(min_break_pct=sh_break_pct, vol_min=sh_vol_min),
                )
                if not short_res.can_enter:
                    continue
                q_score = np.nan
                q_mult = 1.0
                q_ok = True
                if quality_mode in ("score", "strict", "adaptive"):
                    q_score = _short_signal_quality_score(
                        row, price=price_f, ma_val=ma30_f, pivot_low=pivot_low,
                        vol_mult=vol_f, atr_val=atr_f, rs_above_ma=bool(rs_above),
                    )
                    if quality_mode == "adaptive":
                        q_mult = _quality_score_multiplier(
                            q_score, reject_below=adaptive_reject_below,
                            floor_mult=adaptive_floor_mult, mid_mult=adaptive_mid_mult,
                            good_mult=adaptive_good_mult, elite_mult=adaptive_elite_mult,
                        )
                        q_ok = q_mult > 0
                    else:
                        q_ok = q_score >= float(min_short_quality)
                        if q_ok and quality_mode == "strict":
                            q_ok = _short_signal_quality_strict_ok(
                                row, price=price_f, ma_val=ma30_f, pivot_low=pivot_low,
                                vol_mult=vol_f, atr_val=atr_f, rs_above_ma=bool(rs_above),
                            )
                if q_ok:
                    events.append(ReplaySignal(
                        date=dt, ticker=t, side="short", signal="SHORT", reason=short_res.reason,
                        price=price_f, pivot=pivot_low, ma30=ma30_f, ma150=ma150_f, atr14=atr_f,
                        vol_mult=vol_f, adx14=adx_f, regime=regime_label,
                        allow_long=allow_long, allow_short=allow_short, long_size_mult=long_mult,
                        short_size_mult=short_mult, quality_score=q_score, quality_mult=q_mult,
                        stage="Stage 4", source="core_short",
                        explanation=_explain_short_event(
                            price=price_f, pivot_low=pivot_low, ma30=ma30_f, ma150=ma150_f,
                            vol_mult=vol_f, vol_min=sh_vol_min, rs_above=bool(rs_above),
                            regime=regime_label, short_mult=short_mult, quality_score=q_score,
                            quality_mult=q_mult, quality_mode=quality_mode, reason=short_res.reason,
                        ),
                        pivot_distance_pct=_pct_from_level(price_f, pivot_low),
                        ma30_distance_pct=_pct_from_level(price_f, ma30_f),
                        ma150_distance_pct=_pct_from_level(price_f, ma150_f),
                        vol_min_required=sh_vol_min,
                        rs_above_ma=bool(rs_above),
                    ))

    return replay_events_to_df(events)


def replay_events_to_df(events: List[ReplaySignal]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append({
            "date": pd.Timestamp(e.date).strftime("%Y-%m-%d"),
            "ticker": e.ticker,
            "side": e.side,
            "signal": e.signal,
            "reason": e.reason,
            "price": e.price,
            "pivot": e.pivot,
            "ma30": e.ma30,
            "ma150": e.ma150,
            "atr14": e.atr14,
            "vol_mult": e.vol_mult,
            "adx14": e.adx14,
            "regime": e.regime,
            "allow_long": e.allow_long,
            "allow_short": e.allow_short,
            "long_size_mult": e.long_size_mult,
            "short_size_mult": e.short_size_mult,
            "quality_score": e.quality_score,
            "quality_mult": e.quality_mult,
            "stage": e.stage,
            "source": e.source,
            "explanation": e.explanation,
            "pivot_distance_pct": e.pivot_distance_pct,
            "ma30_distance_pct": e.ma30_distance_pct,
            "ma150_distance_pct": e.ma150_distance_pct,
            "vol_min_required": e.vol_min_required,
            "adx_min_required": e.adx_min_required,
            "rs_above_ma": e.rs_above_ma,
            "explain_version": e.explain_version,
        })
    return pd.DataFrame(rows)


def replay_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["year", "month", "signal", "count", "unique_tickers"])
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["year"] = x["date"].dt.year
    x["month"] = x["date"].dt.to_period("M").astype(str)
    return (
        x.groupby(["year", "month", "signal"], dropna=False)
        .agg(count=("ticker", "size"), unique_tickers=("ticker", "nunique"))
        .reset_index()
        .sort_values(["year", "month", "signal"])
    )



def replay_explainability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact summary of why replay events fired, for 10-year audit reports."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["signal", "reason", "source", "count", "unique_tickers"])
    x = df.copy()
    return (
        x.groupby(["signal", "reason", "source"], dropna=False)
        .agg(
            count=("ticker", "size"),
            unique_tickers=("ticker", "nunique"),
            avg_pivot_distance_pct=("pivot_distance_pct", "mean"),
            avg_ma150_distance_pct=("ma150_distance_pct", "mean"),
            avg_vol_mult=("vol_mult", "mean"),
            avg_adx14=("adx14", "mean"),
        )
        .reset_index()
        .sort_values(["signal", "count"], ascending=[True, False])
    )
