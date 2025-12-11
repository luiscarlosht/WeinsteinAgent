#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Live Logic Backtest (daily approximation of intraday watchers)

Goals:
- Approximate your *production* Weinstein long + short logic
  (Stage 2 breakouts / Stage 4 breakdowns with volume gates)
  in a backtest so you can inspect:
    * Monthly returns & win-rate
    * Equity curve
    * Behavior of long + short sides

Key points:
- DEFAULT: Uses the latest weekly scan CSV: ./output/weinstein_weekly_equities_*.csv
- OPTIONAL (when available): can use a directory of *historical* weekly snapshots,
  one CSV per as-of date, under:
      ./data/weekly_snapshots/
  and chooses the latest snapshot with date <= current backtest date.
- Stage 2 (Uptrend) universe for LONG side
- Stage 4 (Downtrend) universe for SHORT side
- Uses DAILY bars (yfinance, auto_adjust=True)
- Entry rules:
    * Long: price above MA30, RS strong, breakout > prior 50-day high,
      daily volume ≥ ~1.3× 50-day avg, ADX filter
    * Short: price below MA30, RS weak, breakdown < prior 50-day low,
      daily volume ≥ ~1.3× 50-day avg + ADX filter
- Risk sizing:
    * Risk per trade = equity * risk_per_trade / per-share-risk
    * Stops use ATR and MA30 guard, similar to your intraday logic
- Outputs:
    * Trade log CSV
    * Equity curve PNG
    * Monthly P/L CSV + printed summary

Regime + VIX (Chapter 8) — **HISTORICAL**:
- Uses market_regime.build_historical_regime_table(...) to compute a daily regime
  table for the test period, based on major indices + VIX:

    regime_table.loc[date] has:
        regime         ("bull"/"bear"/"neutral"/"unknown")
        long_ok        (Weinstein+VIX: OK for new LONGS)
        short_ok       (Weinstein+VIX: OK for new SHORTS)
        ... plus regime-only and VIX-only gates

- Backtest then gates **per day** using:
    --use-regime-long / backtest.regime.use_long
    --use-regime-short / backtest.regime.use_short
  and/or AUTO mode (see below).

Coppock (benchmark) gates:
    --benchmark SPY
        Benchmark symbol to compute Coppock (default: SPY).
    --use-coppock-long
        Gate NEW LONG entries: only when Coppock(benchmark) > 0.
    --use-coppock-short
        Gate NEW SHORT entries: only when Coppock(benchmark) < 0.

  Coppock is computed from benchmark daily closes → monthly, with:
      CC = WMA_10( ROC_14 + ROC_11 ), classic settings,
  then forward-filled back to the daily index so each day reuses the latest
  monthly Coppock value.

Shared LONG-side core:
    * price / MA / pivot breakout
    * RS must be strong
    * volume vs 50dma
    * ADX filter (NaN → no block)
    * long stop + exit logic
  via weinstein_long_core.check_long_entry / LongEntryParams /
       long_stop_level / should_exit_long.

Config-driven backtest behavior (Option C via config.yaml.backtest):
    * snapshot_mode: static | historical | auto
    * regime.use_long / regime.use_short gates
    * coppock.use_long / coppock.use_short gates
    * breadth.enabled / breadth.ma_window / breadth.min_long
    * market.* filters (SPY MA30 slope + VIX cap)
    * industry.* filters (per-group Stage 2 + slopes)

Config-driven ADX logging noise:
    * backtest.logging.show_adx_skips: true/false
      (or use CLI --show-adx-skips)

Config-driven breadth / Coppock logging noise:
    * backtest.logging.show_breadth_skips: true/false
    * backtest.logging.show_coppock_skips: true/false

AUTO trading mode (now **historical regime-aware**):

    --mode auto
        Use the *daily* Chapter 8 + VIX regime table:
          * On a given day:
              - new longs allowed iff long_ok==True
              - new shorts allowed iff short_ok==True
          * If both False → no new entries that day (exits only)
        AUTO mode always evaluates both sides, but regime_table will zero
        out whichever side is disallowed on that date.
"""

import argparse
import os
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Tuple, Mapping

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml  # config.yaml loader

# Shared Weinstein indicators (ADX + breadth single source of truth)
from weinstein_indicators import (
    compute_adx_series,
    ADX_WINDOW,
    ADX_MIN,
    compute_breadth_series_above_ma,
)

# Shared LONG-side core (price/pivot/ADX/volume + stops/exits)
from weinstein_long_core import (
    LongEntryParams,
    check_long_entry,
    long_stop_level,
    should_exit_long,
)

# MA30 slope + other shared filters
from weinstein_filters import stock_ma30_slope_ok_from_snapshot

# Chapter 8 + VIX regime — now with historical helpers for SIM
from market_regime import (
    MarketRegimeConfig,
    build_historical_regime_table,
)


# ---------------- Logging helpers ----------------

VERBOSE = True


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, *, level: str = "info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {
        "info": "•",
        "ok": "✅",
        "step": "▶️",
        "warn": "⚠️",
        "err": "❌",
        "debug": "··",
    }.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)


# ---------------- Config helper ----------------


def load_yaml_config(path: str = "./config.yaml") -> dict:
    """
    Load YAML config shared with weekly/intraday.
    Returns {} if file is missing or invalid.
    """
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except FileNotFoundError:
        log(f"Config file {path} not found; using code defaults.", level="warn")
        return {}
    except Exception as e:
        log(f"Failed to load config {path}: {e}; using code defaults.", level="warn")
        return {}


# ---------------- File / data helpers ----------------

WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_equities_"

# Historical snapshot dir (optional, for Option A)
# Expected: many CSVs like:
#   data/weekly_snapshots/weinstein_weekly_equities_2019-01-04.csv
#   data/weekly_snapshots/weinstein_weekly_equities_20190104.csv
#   data/weekly_snapshots/weinstein_weekly_equities_20190104_1801.csv
WEEKLY_SNAPSHOT_DIR = "./data/weekly_snapshots"

_SNAPSHOT_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")


def newest_weekly_csv() -> str:
    files = [
        f
        for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found in {WEEKLY_OUTPUT_DIR}. "
            f"Run weinstein_report_weekly.py first."
        )
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report() -> pd.DataFrame:
    path = newest_weekly_csv()
    log(f"Using weekly CSV: {path}", level="info")
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    return df


def _parse_snapshot_date_from_name(fname: str) -> Optional[date]:
    """
    Try to extract an as-of date from a snapshot filename.

    Accepts:
      - YYYYMMDD
      - YYYY-MM-DD

    Examples:
      weinstein_weekly_equities_20190104_1801.csv -> 2019-01-04
      weinstein_weekly_equities_2019-01-04.csv   -> 2019-01-04
    """
    m = _SNAPSHOT_NAME_RE.search(fname)
    if not m:
        return None
    token = m.group(1)
    try:
        if len(token) == 8:
            dt_obj = datetime.strptime(token, "%Y%m%d").date()
        else:
            dt_obj = datetime.strptime(token, "%Y-%m-%d").date()
        return dt_obj
    except Exception:
        return None


def load_weekly_snapshots(snapshot_dir: str) -> List[Tuple[date, pd.DataFrame]]:
    """
    Load historical weekly equity CSV snapshots from snapshot_dir.

    Returns a list of (as_of_date, df) sorted by as_of_date.
    If the directory does not exist or nothing matches, returns [].
    """
    if not os.path.isdir(snapshot_dir):
        log(f"No snapshot dir {snapshot_dir} (skipping historical snapshots).", level="info")
        return []

    snapshots: List[Tuple[date, pd.DataFrame]] = []
    for fname in os.listdir(snapshot_dir):
        if not fname.startswith(WEEKLY_FILE_PREFIX) or not fname.endswith(".csv"):
            continue
        d = _parse_snapshot_date_from_name(fname)
        if not d:
            continue
        path = os.path.join(snapshot_dir, fname)
        try:
            df = pd.read_csv(path)
            df = df.rename(columns=str.lower)
            snapshots.append((d, df))
        except Exception as e:
            log(f"Skipping snapshot {path}: {e}", level="warn")

    snapshots.sort(key=lambda tup: tup[0])
    if snapshots:
        first, last = snapshots[0][0], snapshots[-1][0]
        log(
            f"Loaded {len(snapshots)} weekly snapshots from {snapshot_dir} "
            f"(range {first} → {last}).",
            level="info",
        )
    else:
        log(f"No weekly snapshots found under {snapshot_dir}.", level="info")
    return snapshots


def pick_snapshot_for_date(
    snapshots: List[Tuple[date, pd.DataFrame]],
    as_of_ts: pd.Timestamp,
) -> Optional[Tuple[date, pd.DataFrame]]:
    """
    Choose the most recent snapshot with as_of_date <= current date.
    If none qualifies yet (e.g. before first snapshot), returns None.
    """
    if not snapshots:
        return None
    target = as_of_ts.date()
    chosen: Optional[Tuple[date, pd.DataFrame]] = None
    for d, df in snapshots:
        if d <= target:
            chosen = (d, df)
        else:
            break
    return chosen


def download_daily_bars(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download DAILY OHLCV for tickers via yfinance, with a bit of padding
    before start to compute ATR and MAs.
    """
    start_dt = datetime.fromisoformat(start)
    pad_start = (start_dt - timedelta(days=120)).strftime("%Y-%m-%d")
    log(
        f"Downloading daily bars for {len(tickers)} symbols "
        f"({pad_start} → {end})...",
        level="step",
    )
    data = yf.download(
        tickers,
        start=pad_start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("No daily data returned from yfinance.")
    log("Download complete.", level="ok")
    return data


def compute_atr_from_df(daily_df: pd.DataFrame, ticker: str, n: int = 14) -> float:
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            sub = daily_df.xs(ticker, axis=1, level=1)
        except KeyError:
            return np.nan
    else:
        sub = daily_df
    needed = {"High", "Low", "Close"}
    if not needed.issubset(set(sub.columns)):
        return np.nan
    h, l, c = sub["High"], sub["Low"], sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    return float(atr.iloc[-1]) if len(atr.dropna()) else np.nan


# ------ volume vs 50dma helper (daily approximation of intraday pace) ------


def volume_vs_50dma(
    daily_df: pd.DataFrame, ticker: str, as_of_date: pd.Timestamp
) -> float:
    """
    daily_vol / 50-day average volume, using only data BEFORE as_of_date
    for the 50-day window.
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            v = daily_df[("Volume", ticker)]
        except KeyError:
            return np.nan
    else:
        v = daily_df["Volume"]
    if as_of_date not in v.index:
        return np.nan
    # up to and including today
    sub = v.loc[:as_of_date].dropna()
    if len(sub) < 51:
        return np.nan
    today_vol = sub.iloc[-1]
    vol50 = sub.iloc[:-1].tail(50).mean()
    if vol50 <= 0:
        return np.nan
    return float(today_vol / vol50)


# ---------------- Weinstein-universe helpers ----------------


def build_universe(weekly_df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    side = "long" -> Stage 2 universe
    side = "short" -> Stage 4 universe
    """
    if weekly_df is None or weekly_df.empty:
        return pd.DataFrame(columns=["ticker", "stage", "rs_above_ma", "ma30"])

    for miss in ["ticker", "stage", "rs_above_ma", "ma30"]:
        if miss not in weekly_df.columns:
            weekly_df[miss] = np.nan

    if "rank" in weekly_df.columns:
        weekly_df["weekly_rank"] = weekly_df["rank"]
    else:
        weekly_df["weekly_rank"] = 999999

    if side == "long":
        df = weekly_df[weekly_df["stage"].isin(["Stage 2 (Uptrend)"])].copy()
    elif side == "short":
        df = weekly_df[weekly_df["stage"].isin(["Stage 4 (Downtrend)"])].copy()
    else:
        raise ValueError("side must be 'long' or 'short'")

    df["rs_above_ma"] = df["rs_above_ma"].fillna(False).astype(bool)
    df["weekly_rank"] = (
        pd.to_numeric(df["weekly_rank"], errors="coerce").fillna(999999)
    )
    df["ma30"] = pd.to_numeric(df["ma30"], errors="coerce")
    df = df.sort_values(["weekly_rank", "ticker"])
    return df


# ---------------- Backtest data structures ----------------


@dataclass
class Position:
    ticker: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    stop: float
    atr: float
    opened: datetime


@dataclass
class Trade:
    ticker: str
    side: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float


# ---------------- Trading logic parameters ----------------

# Long side — tuned more closely to production intraday thresholds
LONG_BREAK_PCT = 0.004  # ≈0.4% above pivot breakout, default; overridden by config.backtest.long.break_pct

# Short side (mirrored)
SHORT_BREAK_PCT = 0.004  # ≈0.4% below pivot breakdown, overridden by config.backtest.short.break_pct
SHORT_STOP_HARD = 0.20
SHORT_TRAIL_ATR = 2.0
SHORT_MA_GUARD = 0.03  # extra guard above MA30 (≈3% over)

# Short-side ADX gate (overridden by config.backtest.short.adx_min)
SHORT_ADX_MIN = 22.0

# Volume filters (approximate your intraday VOL_PACE_MIN 1.3×)
# Overridden by config.backtest.long.vol_min / config.backtest.short.vol_min
LONG_VOL_MIN = 1.30
SHORT_VOL_MIN = 1.30

PIVOT_LOOKBACK_DAYS = 50  # pivot highs/lows over last ~10 weeks

# Breadth Health filter (Advance/Decline strength)
# Approximates "% of S&P500 above MA50" by using a breadth universe of tickers.
BREADTH_MA_WINDOW = 50
BREADTH_MIN_LONG = 0.60  # default; overridden by config.backtest.breadth.min_long


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def get_close_series(daily_df: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            s = daily_df[("Close", ticker)].dropna()
        except KeyError:
            return pd.Series(dtype=float)
    else:
        s = daily_df["Close"].dropna()
    return s


def get_ma_series(
    daily_df: pd.DataFrame, ticker: str, window: int = 30
) -> pd.Series:
    c = get_close_series(daily_df, ticker)
    return c.rolling(window).mean()


def get_pivot_high(
    daily_df: pd.DataFrame, ticker: str, as_of_date: pd.Timestamp
) -> float:
    """
    Last 50-day high BEFORE as_of_date (exclusive).
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            high = daily_df[("High", ticker)]
        except KeyError:
            return np.nan
    else:
        high = daily_df["High"]
    sub = high.loc[:as_of_date].iloc[:-1]  # exclude current bar
    sub = sub.dropna().tail(PIVOT_LOOKBACK_DAYS)
    return float(sub.max()) if len(sub) else np.nan


def get_pivot_low(
    daily_df: pd.DataFrame, ticker: str, as_of_date: pd.Timestamp
) -> float:
    """
    Last 50-day low BEFORE as_of_date (exclusive).
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            low = daily_df[("Low", ticker)]
        except KeyError:
            return np.nan
    else:
        low = daily_df["Low"]
    sub = low.loc[:as_of_date].iloc[:-1]
    sub = sub.dropna().tail(PIVOT_LOOKBACK_DAYS)
    return float(sub.min()) if len(sub) else np.nan


# ---------------- Coppock curve helper ----------------


def compute_coppock_from_daily(daily_df: pd.DataFrame, benchmark: str) -> pd.Series:
    """
    Classic Coppock curve on MONTHLY closes for the given benchmark:

        CC = WMA_10( ROC_14 + ROC_11 )

    where ROC_n is % rate-of-change over n months.
    Returns a DAILY series aligned to the benchmark daily index by
    forward-filling the latest monthly Coppock value.
    """
    # Get benchmark close
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            close = daily_df[("Close", benchmark)].dropna()
        except KeyError:
            log(f"Coppock: benchmark {benchmark} not found in daily data.", level="warn")
            return pd.Series(dtype="float64")
    else:
        # Single-ticker case
        close = daily_df["Close"].dropna()

    if close.empty:
        log("Coppock: close series empty; disabling Coppock filter.", level="warn")
        return pd.Series(dtype="float64")

    # Monthly closes (month-end)
    monthly_close = close.resample("ME").last()

    # 14-month and 11-month ROC (%)
    roc_14 = monthly_close.pct_change(14) * 100.0
    roc_11 = monthly_close.pct_change(11) * 100.0
    coppock_raw = roc_14 + roc_11

    # 10-period WMA
    weights = np.arange(1, 11, dtype=float)

    def _wma(x: np.ndarray) -> float:
        return float(np.sum(weights * x) / np.sum(weights))

    coppock_monthly = coppock_raw.rolling(10).apply(_wma, raw=True)

    # Map back to daily index by forward-filling monthly values
    coppock_daily = coppock_monthly.reindex(close.index, method="ffill")
    log(
        f"Coppock curve computed for benchmark {benchmark} "
        f"(monthly points={len(coppock_monthly.dropna())}).",
        level="info",
    )
    return coppock_daily


# ---------------- Market / industry filter helpers ----------------


def build_market_ok_series(
    daily_df: pd.DataFrame,
    benchmark: str,
    vix_symbol: Optional[str],
    market_cfg: Mapping,
) -> Optional[pd.Series]:
    """
    Construct a per-day boolean mask for "market is OK for NEW longs",
    based on:

      backtest.market.require_rising_ma30 (bool)
      backtest.market.ma30_slope_min     (float, default 0.0)
      backtest.market.vix_max            (float, optional)

    - MA30 rising is approximated with a 150-day MA on benchmark daily closes.
    - Slope is last_ma - prev_ma (1-day difference); compared to ma30_slope_min.
    - VIX cap blocks days where VIX close > vix_max.
    """
    require_ma = bool(market_cfg.get("require_rising_ma30", False))
    ma30_slope_min = float(market_cfg.get("ma30_slope_min", 0.0))
    vix_max_raw = market_cfg.get("vix_max", None)
    vix_max = float(vix_max_raw) if vix_max_raw is not None else None

    if not (require_ma or vix_max is not None):
        # No market-level gates configured
        return None

    if not isinstance(daily_df.columns, pd.MultiIndex) or "Close" not in daily_df.columns.levels[0]:
        log("Market filter: daily_df not in expected MultiIndex Close panel; disabling market gates.", level="warn")
        return None

    close_panel = daily_df["Close"]
    idx = close_panel.index

    # --- Benchmark MA30-slope filter ---
    if benchmark in close_panel.columns and require_ma:
        bench_close = close_panel[benchmark].dropna()
        ma = bench_close.rolling(window=150, min_periods=75).mean()
        # Slope = last - prev
        slope = ma - ma.shift(1)
        ma_ok = slope >= ma30_slope_min
    else:
        ma_ok = pd.Series(True, index=idx)

    # --- VIX cap filter ---
    if vix_max is not None and vix_symbol and vix_symbol in close_panel.columns:
        vix_close = close_panel[vix_symbol].dropna()
        vix_ok = vix_close <= vix_max
    else:
        vix_ok = pd.Series(True, index=idx)

    # Union on index, default True when missing
    market_ok = pd.Series(True, index=idx)
    market_ok.loc[ma_ok.index] &= ma_ok
    market_ok.loc[vix_ok.index] &= vix_ok

    log(
        f"Market filter active: require_ma30_rising={require_ma}, ma30_slope_min={ma30_slope_min}, "
        f"vix_max={vix_max}.",
        level="info",
    )
    return market_ok


def _row_pick_first(row: Mapping, candidates: List[str]) -> Optional[float]:
    """
    Look through possible column names and return the first non-NaN value.
    Used for industry filters from weekly snapshots.
    """
    for col in candidates:
        if col in row and not pd.isna(row[col]):
            try:
                return float(row[col])
            except Exception:
                continue
    return None


def industry_ok_from_snapshot(snapshot_row: Mapping, industry_cfg: Mapping) -> bool:
    """
    Industry / group confirmation filter based on a weekly snapshot row and
    config.backtest.industry, which may contain:

      enabled: bool
      require_stage2: bool
      min_stage2_frac: float
      require_rising_ma30: bool
      require_rising_rs: bool

    Column candidates (adjust if your weekly snapshots use different names):

      Stage:
        - 'industry_stage'
        - 'group_stage'
        - 'sector_stage'

      % in Stage 2:
        - 'industry_stage2_frac'
        - 'group_stage2_frac'

      MA30 slope:
        - 'industry_ma30_slope_per_wk'
        - 'group_ma30_slope_per_wk'

      RS slope:
        - 'industry_rs_slope_per_wk'
        - 'group_rs_slope_per_wk'
    """
    if not bool(industry_cfg.get("enabled", False)):
        return True

    require_stage2 = bool(industry_cfg.get("require_stage2", False))
    min_stage2_frac = float(industry_cfg.get("min_stage2_frac", 0.0))
    require_rising_ma30 = bool(industry_cfg.get("require_rising_ma30", False))
    require_rising_rs = bool(industry_cfg.get("require_rising_rs", False))

    # --- Stage 2 requirement ---
    if require_stage2:
        stage_val = None
        for col in ["industry_stage", "group_stage", "sector_stage"]:
            if col in snapshot_row and not pd.isna(snapshot_row[col]):
                stage_val = str(snapshot_row[col])
                break
        if stage_val is not None:
            if "Stage 2" not in stage_val:
                return False
        # If we have no stage info, don't block.

    # --- Fraction of group in Stage 2 ---
    if min_stage2_frac > 0.0:
        frac = _row_pick_first(snapshot_row, ["industry_stage2_frac", "group_stage2_frac"])
        if frac is not None and frac < min_stage2_frac:
            return False

    # --- Industry MA30 slope ---
    if require_rising_ma30:
        slope_ma = _row_pick_first(
            snapshot_row,
            ["industry_ma30_slope_per_wk", "group_ma30_slope_per_wk"],
        )
        if slope_ma is not None and slope_ma < 0.0:
            return False

    # --- Industry RS slope vs benchmark ---
    if require_rising_rs:
        slope_rs = _row_pick_first(
            snapshot_row,
            ["industry_rs_slope_per_wk", "group_rs_slope_per_wk"],
        )
        if slope_rs is not None and slope_rs < 0.0:
            return False

    return True


# ---------------- Entry / exit rules: SHORT side (local for now) ----------------


def should_enter_short(
    price: float,
    ma30_val: float,
    pivot_low: float,
    rs_above_ma: bool,
    vol_mult: float,
) -> bool:
    if np.isnan(price) or np.isnan(ma30_val) or np.isnan(pivot_low):
        return False
    # RS must be weak (NOT above its MA)
    if rs_above_ma:
        return False
    # Price must be below MA30
    if price > ma30_val:
        return False
    # Breakdown under pivot low by SHORT_BREAK_PCT
    if price > pivot_low * (1.0 - SHORT_BREAK_PCT):
        return False
    # Volume pace gate vs 50dma
    if not np.isnan(vol_mult) and vol_mult < SHORT_VOL_MIN:
        return False
    return True


def short_stop_level(entry: float, atr: float, ma30_val: float) -> float:
    if np.isnan(entry):
        return np.nan
    hard = entry * (1.0 + SHORT_STOP_HARD)
    atr_stop = entry + SHORT_TRAIL_ATR * atr if not np.isnan(atr) else np.nan
    ma_guard = ma30_val * (1.0 + SHORT_MA_GUARD) if not np.isnan(ma30_val) else np.nan
    cands = [c for c in [hard, atr_stop, ma_guard] if not np.isnan(c)]
    return min(cands) if cands else hard


def should_exit_short(price: float, stop: float, ma30_val: float) -> bool:
    if np.isnan(price):
        return False
    # 1) Stop violation
    if not np.isnan(stop) and price >= stop:
        return True
    # 2) Extra guard: reclaimed MA30 by SHORT_MA_GUARD
    if not np.isnan(ma30_val) and price >= ma30_val * (1.0 + SHORT_MA_GUARD):
        return True
    return False


# ---------------- Backtest engine ----------------


@dataclass
class Portfolio:
    cash: float  # realized P&L + unallocated capital
    positions: Dict[str, Position]
    equity: float  # cash + open P&L


def backtest(
    daily_df: pd.DataFrame,
    start: str,
    end: str,
    capital: float,
    risk_per_trade: float,
    max_long: int,
    max_short: int,
    mode: str,
    *,
    universe_tickers: List[str],
    weekly_df: Optional[pd.DataFrame] = None,
    weekly_snapshots: Optional[List[Tuple[date, pd.DataFrame]]] = None,
    regime_table: Optional[pd.DataFrame] = None,
    use_regime_long: bool = False,
    use_regime_short: bool = False,
    auto_mode: bool = False,
    benchmark: str = "SPY",
    use_coppock_long: bool = False,
    use_coppock_short: bool = False,
    coppock_series: Optional[pd.Series] = None,
    breadth_enabled: bool = True,
    show_adx_skips: bool = False,
    show_breadth_skips: bool = False,
    show_coppock_skips: bool = False,
    long_logic_cfg: Optional[Dict[str, object]] = None,
    market_cfg: Optional[Dict[str, object]] = None,
    industry_cfg: Optional[Dict[str, object]] = None,
    vix_symbol: Optional[str] = None,
) -> Dict[str, object]:
    """
    mode: "long", "short", "both", or "none"
          (AUTO mode is handled via auto_mode + regime_table; here we just
           see 'both' and use per-day regime gates to zero out sides.)

    If weekly_snapshots is provided and non-empty:
      - uses dynamic weekly universes per date (Option A).
    Else:
      - uses single weekly_df snapshot (current behavior).

    Regime settings:
      - regime_table: daily DataFrame from market_regime.build_historical_regime_table().
      - use_regime_long: if True, NEW longs allowed only when regime_table.long_ok is True.
      - use_regime_short: if True, NEW shorts allowed only when regime_table.short_ok is True.
      - auto_mode: if True, we *also* respect regime_table.long_ok / short_ok
                   even if use_regime_long/short are False; AUTO mode is
                   effectively "both sides, but day-by-day gated by long_ok/short_ok".

    Coppock gates:
      - use_coppock_long: if True, NEW longs allowed only when Coppock(benchmark) > 0.
      - use_coppock_short: if True, NEW shorts allowed only when Coppock(benchmark) < 0.

    breadth_enabled:
      - if False, breadth gate is disabled regardless of BREADTH_* constants.

    market_cfg:
      - backtest.market dict, used for MA30 slope + VIX cap on the whole market.

    industry_cfg:
      - backtest.industry dict, used for per-group confirmation from snapshots.

    long_logic_cfg:
      - config.backtest.long dict, used here for MA30 slope filters (and any
        future shared long-side filters that should match PROD).

    show_*_skips:
      - ADX / breadth / Coppock skip logging toggles.
    """
    long_logic_cfg = long_logic_cfg or {}
    market_cfg = market_cfg or {}
    industry_cfg = industry_cfg or {}

    use_snapshots = bool(weekly_snapshots)

    # Precompute static universes for fallback mode
    static_long_universe: Optional[pd.DataFrame] = None
    static_short_universe: Optional[pd.DataFrame] = None
    if not use_snapshots:
        if weekly_df is None:
            raise RuntimeError("weekly_df is required when no weekly_snapshots are provided.")
        static_long_universe = build_universe(weekly_df, side="long")
        static_short_universe = build_universe(weekly_df, side="short")

    # ----- Breadth series (approx "% of universe above MA50") -----
    breadth_series = None
    if breadth_enabled and isinstance(daily_df.columns, pd.MultiIndex) and "Close" in daily_df.columns.levels[0]:
        close_panel = daily_df["Close"]
        breadth_series = compute_breadth_series_above_ma(
            daily_close_panel=close_panel,
            tickers=sorted(universe_tickers),
            ma_window=BREADTH_MA_WINDOW,
        )
        if breadth_series is not None and not breadth_series.empty:
            log(
                f"Breadth series computed over {len(universe_tickers)} breadth tickers "
                f"(MA{BREADTH_MA_WINDOW}).",
                level="info",
            )
        else:
            log("Breadth series is empty; breadth gate will be effectively disabled.", level="warn")
            breadth_series = None
    else:
        if not breadth_enabled:
            log("Breadth gate disabled by config.", level="info")
        else:
            log(
                "Daily data not in expected MultiIndex Close panel; breadth gate disabled.",
                level="warn",
            )
        breadth_series = None

    # ----- Coppock series (benchmark) -----
    if coppock_series is None or coppock_series.empty:
        log("Coppock series is empty; Coppock gates will be effectively disabled.", level="warn")
        coppock_series = None

    # ----- Market-wide filter (SPY MA30 slope + VIX cap) -----
    market_ok_series = None
    if market_cfg:
        market_ok_series = build_market_ok_series(
            daily_df=daily_df,
            benchmark=benchmark,
            vix_symbol=vix_symbol,
            market_cfg=market_cfg,
        )

    all_dates = daily_df.index
    all_dates = [d for d in all_dates if isinstance(d, (pd.Timestamp, datetime))]
    all_dates = [pd.Timestamp(d) for d in all_dates]

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    trade_log: List[Trade] = []
    portfolio = Portfolio(cash=capital, positions={}, equity=capital)
    equity_curve = []

    # Precompute MA30, ATR, and ADX for each ticker for speed
    ma_cache: Dict[str, pd.Series] = {}
    adx_cache: Dict[str, pd.Series] = {}
    for t in universe_tickers:
        # MA30
        ma_cache[t] = get_ma_series(daily_df, t, window=30)
        # ADX series via shared helper (single source of truth)
        if isinstance(daily_df.columns, pd.MultiIndex):
            try:
                sub = daily_df.xs(t, axis=1, level=1)
            except KeyError:
                adx_cache[t] = pd.Series(dtype="float64")
                continue
        else:
            sub = daily_df
        # sub should have High/Low/Close
        if not {"High", "Low", "Close"}.issubset(sub.columns):
            adx_cache[t] = pd.Series(dtype="float64")
        else:
            adx_cache[t] = compute_adx_series(
                sub[["High", "Low", "Close"]], n=ADX_WINDOW
            )

    atr_cache: Dict[str, float] = {}
    for t in universe_tickers:
        atr_cache[t] = compute_atr_from_df(daily_df, t, n=14)

    # Shared LONG-side core parameters (aligned with existing constants)
    long_params = LongEntryParams(
        min_break_pct=LONG_BREAK_PCT,
        dist_above_ma_min=0.0,       # backtest uses "price > MA30" (no extra headroom)
        vol_min=LONG_VOL_MIN,
        adx_min=ADX_MIN,
    )

    # State for dynamic snapshots
    current_snapshot_date: Optional[date] = None
    current_long_universe: Optional[pd.DataFrame] = static_long_universe
    current_short_universe: Optional[pd.DataFrame] = static_short_universe

    # Main daily loop
    for i, dt_ in enumerate(all_dates):
        if dt_ < start_dt or dt_ > end_dt:
            continue

        # ----- choose weekly universe for this date -----
        if use_snapshots and weekly_snapshots:
            snap = pick_snapshot_for_date(weekly_snapshots, dt_)
            if snap is None:
                # Before first snapshot: no universe yet; let exits run, but no new entries
                long_universe = pd.DataFrame(columns=["ticker"])
                short_universe = pd.DataFrame(columns=["ticker"])
            else:
                snap_date, wdf = snap
                if snap_date != current_snapshot_date:
                    current_long_universe = build_universe(wdf, side="long")
                    current_short_universe = build_universe(wdf, side="short")
                    current_snapshot_date = snap_date
                long_universe = (
                    current_long_universe
                    if current_long_universe is not None
                    else pd.DataFrame(columns=["ticker"])
                )
                short_universe = (
                    current_short_universe
                    if current_short_universe is not None
                    else pd.DataFrame(columns=["ticker"])
                )
        else:
            long_universe = (
                static_long_universe
                if static_long_universe is not None
                else pd.DataFrame(columns=["ticker"])
            )
            short_universe = (
                static_short_universe
                if static_short_universe is not None
                else pd.DataFrame(columns=["ticker"])
            )

        # Build price snapshot for this day
        price_today: Dict[str, float] = {}
        if isinstance(daily_df.columns, pd.MultiIndex):
            if "Close" not in daily_df.columns.levels[0]:
                continue
            closes = daily_df["Close"]
            for t in closes.columns:
                if dt_ in closes.index:
                    price_today[t] = _safe_float(closes.loc[dt_, t])
        else:
            # Single ticker case (unlikely in your universe)
            if dt_ in daily_df.index:
                price_today["SINGLE"] = _safe_float(daily_df["Close"].loc[dt_])

        # Mark-to-market holdings: equity = cash + open P&L
        eq = portfolio.cash
        for pos in list(portfolio.positions.values()):
            t = pos.ticker
            p = price_today.get(t, np.nan)
            if np.isnan(p):
                continue
            if pos.side == "long":
                eq += pos.qty * (p - pos.entry_price)
            else:
                eq += pos.qty * (pos.entry_price - p)
        portfolio.equity = eq
        equity_curve.append({"date": dt_, "equity": eq})

        # ----- daily regime gates (Chapter 8 + VIX, historical) -----
        long_regime_ok_today = True
        short_regime_ok_today = True
        if regime_table is not None and dt_ in regime_table.index:
            r_row = regime_table.loc[dt_]
            r_long_ok = bool(r_row.get("long_ok", True))
            r_short_ok = bool(r_row.get("short_ok", True))

            # If auto_mode: always respect regime long/short OK for both sides.
            if auto_mode or use_regime_long:
                long_regime_ok_today = r_long_ok
            if auto_mode or use_regime_short:
                short_regime_ok_today = r_short_ok

        # Market-wide filter (SPY MA30 slope + VIX cap) for NEW longs
        market_ok_today = True
        if market_ok_series is not None and dt_ in market_ok_series.index:
            val = market_ok_series.loc[dt_]
            if not pd.isna(val):
                market_ok_today = bool(val)

        # Compute breadth gate for this day (for new LONG entries)
        breadth_ok = True
        breadth_val = np.nan
        if breadth_series is not None and dt_ in breadth_series.index:
            breadth_val = float(breadth_series.loc[dt_])
            if not np.isnan(breadth_val):
                breadth_ok = breadth_val >= BREADTH_MIN_LONG
            else:
                breadth_ok = True  # if NaN, don't block

        # Optional debug logging when breadth blocks new longs
        if not breadth_ok and show_breadth_skips:
            log(
                f"[SKIP-BREADTH] No new LONGs on {dt_.date()} because breadth="
                f"{breadth_val:.2%} < {BREADTH_MIN_LONG:.0%}",
                level="debug",
            )

        # Coppock gate for this day
        coppock_val = np.nan
        coppock_long_ok = True
        coppock_short_ok = True
        if coppock_series is not None and not coppock_series.empty and dt_ in coppock_series.index:
            coppock_val = float(coppock_series.loc[dt_])

        if use_coppock_long and not np.isnan(coppock_val):
            coppock_long_ok = coppock_val > 0.0
            if not coppock_long_ok and show_coppock_skips:
                log(
                    f"[SKIP-COPPOCK-LONG] No new LONGs on {dt_.date()} because "
                    f"Coppock({benchmark})={coppock_val:.2f} ≤ 0.",
                    level="debug",
                )

        if use_coppock_short and not np.isnan(coppock_val):
            coppock_short_ok = coppock_val < 0.0
            if not coppock_short_ok and show_coppock_skips:
                log(
                    f"[SKIP-COPPOCK-SHORT] No new SHORTs on {dt_.date()} because "
                    f"Coppock({benchmark})={coppock_val:.2f} ≥ 0.",
                    level="debug",
                )

        # First exits, then entries (so freed risk can be reused)
        # ------ Exits ------
        to_remove = []
        for key, pos in list(portfolio.positions.items()):
            p = price_today.get(pos.ticker, np.nan)
            if np.isnan(p):
                continue
            ma_series = ma_cache.get(pos.ticker)
            ma_val = (
                ma_series.loc[dt_]
                if ma_series is not None and dt_ in ma_series.index
                else np.nan
            )

            if pos.side == "long":
                if not should_exit_long(p, pos.stop, ma_val):
                    continue
                exit_price = p
                pnl = pos.qty * (exit_price - pos.entry_price)
            else:
                if not should_exit_short(p, pos.stop, ma_val):
                    continue
                exit_price = p
                pnl = pos.qty * (pos.entry_price - exit_price)

            pnl_pct = (
                pnl / (pos.entry_price * pos.qty) if pos.qty > 0 else 0.0
            )
            portfolio.cash += pnl  # realize P&L into cash

            trade_log.append(
                Trade(
                    ticker=pos.ticker,
                    side=pos.side,
                    entry_date=pos.opened,
                    exit_date=dt_,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    qty=pos.qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            )
            to_remove.append(key)

        for key in to_remove:
            del portfolio.positions[key]

        # ------ Entries ------
        # Determine how many new slots are available
        n_long_now = sum(1 for p in portfolio.positions.values() if p.side == "long")
        n_short_now = sum(1 for p in portfolio.positions.values() if p.side == "short")

        # LONG entries (gated by market_ok + breadth_ok + daily regime + Coppock gate)
        if (
            mode in ("long", "both")
            and n_long_now < max_long
            and market_ok_today
            and breadth_ok
            and long_regime_ok_today
            and coppock_long_ok
        ):
            for _, row in long_universe.iterrows():
                t = str(row["ticker"]).upper()
                pos_key = f"{t}_long"
                if pos_key in portfolio.positions:
                    continue

                # MA30 slope / trend filter from weekly snapshot
                if not stock_ma30_slope_ok_from_snapshot(row, long_logic_cfg):
                    continue

                # Industry / group confirmation (Stage 2, group slopes)
                if not industry_ok_from_snapshot(row, industry_cfg):
                    continue

                price = price_today.get(t, np.nan)
                if np.isnan(price):
                    continue

                ma_series = ma_cache.get(t)
                ma_val = (
                    ma_series.loc[dt_]
                    if ma_series is not None and dt_ in ma_series.index
                    else np.nan
                )
                pivot_high = get_pivot_high(daily_df, t, dt_)
                rs_above_ma = bool(row.get("rs_above_ma", False))
                vol_mult = volume_vs_50dma(daily_df, t, dt_)

                # ADX series
                adx_series = adx_cache.get(t)
                if (
                    adx_series is not None
                    and not adx_series.empty
                    and dt_ in adx_series.index
                ):
                    adx_val = float(adx_series.loc[dt_])
                else:
                    adx_val = np.nan

                # Shared LONG-side core check
                entry_check = check_long_entry(
                    price=price,
                    ma_val=ma_val,
                    pivot=pivot_high,
                    rs_above_ma=rs_above_ma,
                    vol_mult=vol_mult,
                    adx_val=adx_val,
                    params=long_params,
                )

                # Optional: keep ADX debug message gated by show_adx_skips
                if show_adx_skips and not entry_check.adx_ok and not np.isnan(adx_val):
                    log(
                        f"[SKIP-ADX] {t} because ADX{ADX_WINDOW}={adx_val:.1f} < {ADX_MIN:.1f} on {dt_.date()}",
                        level="debug",
                    )

                if not entry_check.can_enter:
                    # Price / RS / MA / pivot / volume not aligned for a breakout
                    continue

                atr = atr_cache.get(t, np.nan)
                stop = long_stop_level(price, atr, ma_val)
                if np.isnan(stop) or stop >= price:
                    continue  # invalid or non-risking stop

                # Position sizing: risk_per_trade * equity / (entry - stop)
                risk_per_pos = portfolio.equity * risk_per_trade
                per_share_risk = price - stop
                if per_share_risk <= 0:
                    continue
                qty = math.floor(risk_per_pos / per_share_risk)
                if qty <= 0:
                    continue

                portfolio.positions[pos_key] = Position(
                    ticker=t,
                    side="long",
                    qty=qty,
                    entry_price=price,
                    stop=stop,
                    atr=atr,
                    opened=dt_,
                )
                n_long_now += 1
                if n_long_now >= max_long:
                    break

        # SHORT entries (gated by daily regime + Coppock gate + ADX)
        if (
            mode in ("short", "both")
            and n_short_now < max_short
            and short_regime_ok_today
            and coppock_short_ok
        ):
            for _, row in short_universe.iterrows():
                t = str(row["ticker"]).upper()
                pos_key = f"{t}_short"
                if pos_key in portfolio.positions:
                    continue
                price = price_today.get(t, np.nan)
                if np.isnan(price):
                    continue

                ma_series = ma_cache.get(t)
                ma_val = (
                    ma_series.loc[dt_]
                    if ma_series is not None and dt_ in ma_series.index
                    else np.nan
                )
                pivot_low = get_pivot_low(daily_df, t, dt_)
                rs_above_ma = bool(row.get("rs_above_ma", False))
                vol_mult = volume_vs_50dma(daily_df, t, dt_)

                # Short-side ADX gate
                adx_series = adx_cache.get(t)
                if (
                    adx_series is not None
                    and not adx_series.empty
                    and dt_ in adx_series.index
                ):
                    adx_val = float(adx_series.loc[dt_])
                else:
                    adx_val = np.nan

                if not np.isnan(adx_val) and adx_val < SHORT_ADX_MIN:
                    if show_adx_skips and not np.isnan(adx_val):
                        log(
                            f"[SKIP-ADX-SHORT] {t} because ADX{ADX_WINDOW}="
                            f"{adx_val:.1f} < {SHORT_ADX_MIN:.1f} on {dt_.date()}",
                            level="debug",
                        )
                    continue

                if not should_enter_short(
                    price, ma_val, pivot_low, rs_above_ma, vol_mult
                ):
                    continue

                atr = atr_cache.get(t, np.nan)
                stop = short_stop_level(price, atr, ma_val)
                if np.isnan(stop) or stop <= price:
                    continue  # invalid stop

                risk_per_pos = portfolio.equity * risk_per_trade
                per_share_risk = stop - price
                if per_share_risk <= 0:
                    continue
                qty = math.floor(risk_per_pos / per_share_risk)
                if qty <= 0:
                    continue

                portfolio.positions[pos_key] = Position(
                    ticker=t,
                    side="short",
                    qty=qty,
                    entry_price=price,
                    stop=stop,
                    atr=atr,
                    opened=dt_,
                )
                n_short_now += 1
                if n_short_now >= max_short:
                    break

        # Monthly-ish progress ping (about every 20 trading days)
        if (i + 1) % 20 == 0:
            log(
                f"Progress: {dt_.date()} — equity ${portfolio.equity:,.2f}, "
                f"positions: {len(portfolio.positions)}, trades so far: {len(trade_log)}",
                level="debug",
            )

    return {
        "portfolio": portfolio,
        "trades": trade_log,
        "equity_curve": equity_curve,
    }


# ---------------- Plotting & CSV helpers ----------------


def save_trade_log(trades: List[Trade], path: str):
    if not trades:
        log("No trades to save.", level="warn")
        return
    rows = []
    for t in trades:
        rows.append(
            {
                "Ticker": t.ticker,
                "Side": t.side,
                "EntryDate": t.entry_date.strftime("%Y-%m-%d"),
                "ExitDate": t.exit_date.strftime("%Y-%m-%d"),
                "EntryPrice": t.entry_price,
                "ExitPrice": t.exit_price,
                "Qty": t.qty,
                "PnL": t.pnl,
                "PnL_pct": t.pnl_pct,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log(f"Wrote trade log → {path}", level="ok")


def save_equity_curve(equity_curve: List[Dict[str, object]], path: str):
    if not equity_curve:
        log("No equity curve to plot.", level="warn")
        return
    df = pd.DataFrame(equity_curve)
    df = df.sort_values("date")
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["equity"])
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.title("Weinstein Live Logic Backtest — Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log(f"Wrote equity curve PNG → {path}", level="ok")


def save_monthly_pnl(
    trades: List[Trade],
    equity_curve: List[Dict[str, object]],
    initial_capital: float,
    path: str,
):
    if not trades:
        log("No trades for monthly P/L.", level="warn")
        return
    # Build trades DF
    rows = []
    for t in trades:
        rows.append(
            {
                "Ticker": t.ticker,
                "Side": t.side,
                "EntryDate": t.entry_date,
                "ExitDate": t.exit_date,
                "PnL": t.pnl,
            }
        )
    df_tr = pd.DataFrame(rows)
    df_tr["ExitDate"] = pd.to_datetime(df_tr["ExitDate"])
    # Use month-end timestamps so they align with equity resample("ME")
    df_tr["Month"] = df_tr["ExitDate"].dt.to_period("M").dt.to_timestamp(how="end")

    monthly = df_tr.groupby("Month").agg(
        PnL=("PnL", "sum"),
        Trades=("PnL", "count"),
        Wins=("PnL", lambda x: (x > 0).sum()),
    )
    monthly["WinRate"] = monthly["Wins"] / monthly["Trades"]

    # Equity month-end
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date").sort_index()
    # Use "ME" (month-end) to avoid FutureWarning about "M"
    eq_monthly = eq_df.resample("ME").last().rename(columns={"equity": "Equity"})

    monthly = monthly.join(eq_monthly["Equity"], how="left")
    monthly = monthly.reset_index().rename(columns={"Month": "MonthEnd"})

    # Simple % PnL vs initial capital (not path-dependent)
    monthly["PnL_pct_of_initial"] = monthly["PnL"] / initial_capital * 100.0

    monthly.to_csv(path, index=False)
    log(f"Wrote monthly P/L breakdown → {path}", level="ok")

    # Console summary
    log("Monthly P/L summary:", level="info")
    for _, r in monthly.iterrows():
        month_str = r["MonthEnd"].strftime("%Y-%m")
        pnl = r["PnL"]
        trades_n = int(r["Trades"])
        winrate = r["WinRate"] * 100.0 if not np.isnan(r["WinRate"]) else 0.0
        eq = r["Equity"]
        log(
            f"  {month_str}: PnL=${pnl:,.2f} | Trades={trades_n} | "
            f"WinRate={winrate:5.1f}% | Equity=${eq:,.2f}",
            level="info",
        )


# ---------------- CLI ----------------


def main():
    global VERBOSE

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    ap.add_argument("--year", type=int, help="Backtest full calendar year")
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--max-long", type=int, default=10)
    ap.add_argument("--max-short", type=int, default=10)
    ap.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["long", "short", "both", "auto"],
        help=(
            "Trading side selection:\n"
            "  long  - Stage 2 longs only\n"
            "  short - Stage 4 shorts only\n"
            "  both  - trade both sides independently\n"
            "  auto  - use daily market_regime table to gate long/short per day"
        ),
    )
    ap.add_argument("--quiet", action="store_true")

    # Config path (Option C)
    ap.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to YAML config (default: ./config.yaml).",
    )

    # NEW: optional Chapter 8 + VIX regime gates (now historical)
    ap.add_argument(
        "--use-regime-long",
        action="store_true",
        help="Gate NEW long entries by daily Chapter 8 + VIX (regime_table.long_ok).",
    )
    ap.add_argument(
        "--use-regime-short",
        action="store_true",
        help="Gate NEW short entries by daily Chapter 8 + VIX (regime_table.short_ok).",
    )

    # NEW: benchmark + Coppock gates
    ap.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark symbol used for RS/breadth/Coppock filters (default: from config.app.benchmark or SPY).",
    )
    ap.add_argument(
        "--use-coppock-long",
        action="store_true",
        help="Gate NEW long entries by benchmark Coppock > 0.",
    )
    ap.add_argument(
        "--use-coppock-short",
        action="store_true",
        help="Gate NEW short entries by benchmark Coppock < 0.",
    )

    # NEW: snapshot-mode override (static | historical | auto)
    ap.add_argument(
        "--snapshot-mode",
        type=str,
        choices=["static", "historical", "auto"],
        help="Universe source: static (latest weekly), historical (snapshots), auto (prefer snapshots, fallback static).",
    )

    # NEW: ADX skip logging toggle
    ap.add_argument(
        "--show-adx-skips",
        action="store_true",
        help="Log [SKIP-ADX] debug messages for long/short entries blocked by ADX.",
    )

    args = ap.parse_args()

    VERBOSE = not args.quiet

    # ---- Load config.yaml for Option C behavior ----
    cfg = load_yaml_config(args.config)
    app_cfg = cfg.get("app", {}) or {}
    bt_cfg = cfg.get("backtest", {}) or {}

    # --- Long/short thresholds from config.backtest ---
    bt_long_cfg = bt_cfg.get("long", {}) or {}
    bt_short_cfg = bt_cfg.get("short", {}) or {}
    market_cfg = bt_cfg.get("market", {}) or {}
    industry_cfg = bt_cfg.get("industry", {}) or {}

    global LONG_BREAK_PCT, LONG_VOL_MIN
    global SHORT_BREAK_PCT, SHORT_VOL_MIN, SHORT_STOP_HARD, SHORT_TRAIL_ATR, SHORT_MA_GUARD, SHORT_ADX_MIN, ADX_MIN

    # Long side: breakout %, volume, ADX
    LONG_BREAK_PCT = float(bt_long_cfg.get("break_pct", LONG_BREAK_PCT))
    LONG_VOL_MIN = float(bt_long_cfg.get("vol_min", LONG_VOL_MIN))
    ADX_MIN = float(bt_long_cfg.get("adx_min", ADX_MIN))

    # Short side: breakout %, volume, ADX + risk block
    SHORT_BREAK_PCT = float(bt_short_cfg.get("break_pct", SHORT_BREAK_PCT))
    SHORT_VOL_MIN = float(bt_short_cfg.get("vol_min", SHORT_VOL_MIN))
    SHORT_ADX_MIN = float(bt_short_cfg.get("adx_min", SHORT_ADX_MIN))
    SHORT_STOP_HARD = float(bt_short_cfg.get("stop_hard", SHORT_STOP_HARD))
    SHORT_TRAIL_ATR = float(bt_short_cfg.get("trail_atr", SHORT_TRAIL_ATR))
    SHORT_MA_GUARD = float(bt_short_cfg.get("ma_guard", SHORT_MA_GUARD))

    # Benchmark: CLI wins, otherwise config, otherwise default "SPY"
    benchmark_cfg = (app_cfg.get("benchmark") or "SPY").upper()
    benchmark = (args.benchmark or benchmark_cfg).upper()

    # Snapshot mode: CLI override > config > default "static"
    snapshot_mode_cfg = bt_cfg.get("snapshot_mode", "static")
    snapshot_mode = args.snapshot_mode or snapshot_mode_cfg

    # Regime toggles: CLI flags OR config booleans
    regime_cfg = bt_cfg.get("regime", {}) or {}
    use_regime_long_cfg = bool(regime_cfg.get("use_long", False))
    use_regime_short_cfg = bool(regime_cfg.get("use_short", False))

    use_regime_long_effective = args.use_regime_long or use_regime_long_cfg
    use_regime_short_effective = args.use_regime_short or use_regime_short_cfg

    # Coppock toggles: CLI flags OR config
    coppock_cfg = bt_cfg.get("coppock", {}) or {}
    use_coppock_long_cfg = bool(coppock_cfg.get("use_long", False))
    use_coppock_short_cfg = bool(coppock_cfg.get("use_short", False))

    use_coppock_long_effective = args.use_coppock_long or use_coppock_long_cfg
    use_coppock_short_effective = args.use_coppock_short or use_coppock_short_cfg

    # Breadth parameters
    breadth_cfg = bt_cfg.get("breadth", {}) or {}
    breadth_enabled = bool(breadth_cfg.get("enabled", True))

    global BREADTH_MA_WINDOW, BREADTH_MIN_LONG
    BREADTH_MA_WINDOW = int(breadth_cfg.get("ma_window", BREADTH_MA_WINDOW))
    BREADTH_MIN_LONG = float(breadth_cfg.get("min_long", BREADTH_MIN_LONG))

    # Logging parameters
    logging_cfg = bt_cfg.get("logging", {}) or {}
    show_adx_cfg = bool(logging_cfg.get("show_adx_skips", False))
    show_breadth_cfg = bool(logging_cfg.get("show_breadth_skips", False))
    show_coppock_cfg = bool(logging_cfg.get("show_coppock_skips", False))

    show_adx_skips_effective = args.show_adx_skips or show_adx_cfg
    show_breadth_skips_effective = show_breadth_cfg
    show_coppock_skips_effective = show_coppock_cfg

    # Resolve start/end range
    if args.year and (args.start or args.end):
        raise SystemExit("Use either --year OR --start/--end, not both.")

    if args.year:
        start = f"{args.year}-01-01"
        end = f"{args.year}-12-31"
    else:
        if not args.start or not args.end:
            raise SystemExit("Provide both --start and --end if not using --year.")
        start, end = args.start, args.end

    # ---- Determine need for regime_table ----
    auto_mode = (args.mode == "auto")
    need_regime_table = (
        auto_mode or use_regime_long_effective or use_regime_short_effective
    )

    # Effective mode for backtest: AUTO uses both sides, but regime_table
    # will gate long/short per day.
    if auto_mode:
        effective_mode = "both"
    else:
        effective_mode = args.mode

    # ---- Weekly universe source selection (Option C via snapshot_mode) ----
    weekly_snapshots: Optional[List[Tuple[date, pd.DataFrame]]] = None
    weekly_df: Optional[pd.DataFrame] = None
    all_tickers: set[str] = set()

    if snapshot_mode == "historical":
        weekly_snapshots = load_weekly_snapshots(WEEKLY_SNAPSHOT_DIR)
        if not weekly_snapshots:
            log(
                "snapshot_mode='historical' but no snapshots found; "
                "you may want snapshot_mode='static' or 'auto'.",
                level="warn",
            )
        else:
            for _, df in weekly_snapshots:
                if "ticker" in df.columns:
                    all_tickers.update(df["ticker"].astype(str).str.upper())
            log(
                f"Using historical weekly snapshots for universe "
                f"(unique tickers={len(all_tickers)}).",
                level="info",
            )

    elif snapshot_mode == "auto":
        tmp_snapshots = load_weekly_snapshots(WEEKLY_SNAPSHOT_DIR)
        if tmp_snapshots:
            weekly_snapshots = tmp_snapshots
            for _, df in weekly_snapshots:
                if "ticker" in df.columns:
                    all_tickers.update(df["ticker"].astype(str).str.upper())
            log(
                f"[auto] Using historical weekly snapshots for universe "
                f"(unique tickers={len(all_tickers)}).",
                level="info",
            )
        else:
            weekly_df = load_weekly_report()
            all_tickers.update(weekly_df["ticker"].astype(str).str.upper())
            log(
                "[auto] No historical snapshots; using latest weekly report "
                f"for static universe ({len(all_tickers)} tickers).",
                level="info",
            )

    else:
        # snapshot_mode == "static" (Option B)
        weekly_df = load_weekly_report()
        all_tickers.update(weekly_df["ticker"].astype(str).str.upper())
        log(
            "snapshot_mode='static': using latest weekly report only "
            f"for static universe ({len(all_tickers)} tickers).",
            level="info",
        )

    # Ensure benchmark is present in daily data for Coppock computation
    all_tickers.add(benchmark)

    # If we need a regime table, also ensure the index symbols + VIX are present
    regime_cfg_defaults = MarketRegimeConfig()
    index_symbols = regime_cfg_defaults.index_symbols
    vix_symbol = regime_cfg_defaults.vix_symbol

    if need_regime_table:
        for sym in index_symbols:
            all_tickers.add(sym)
        if vix_symbol:
            all_tickers.add(vix_symbol)

    if not all_tickers:
        raise RuntimeError("Universe of tickers is empty; cannot run backtest.")

    log(
        f"Backtest range: {start} → {end} | requested_mode={args.mode}, "
        f"effective_mode={effective_mode}, capital={args.capital:,.2f}, "
        f"risk_per_trade={args.risk_per_trade:.3f}, max_long={args.max_long}, "
        f"max_short={args.max_short}",
        level="info",
    )
    log(f"Benchmark for Coppock/filters: {benchmark}", level="info")
    log(
        f"Config: snapshot_mode={snapshot_mode}, regime_long={use_regime_long_effective}, "
        f"regime_short={use_regime_short_effective}, coppock_long={use_coppock_long_effective}, "
        f"coppock_short={use_coppock_short_effective}, breadth_enabled={breadth_enabled}, "
        f"breadth_ma={BREADTH_MA_WINDOW}, breadth_min_long={BREADTH_MIN_LONG:.2f}, "
        f"show_adx_skips={show_adx_skips_effective}, show_breadth_skips={show_breadth_skips_effective}, "
        f"show_coppock_skips={show_coppock_skips_effective}, auto_mode={auto_mode}",
        level="info",
    )

    daily_df = download_daily_bars(sorted(all_tickers), start, end)

    # Compute Coppock curve for benchmark (daily series)
    coppock_series = compute_coppock_from_daily(daily_df, benchmark)

    # Build historical regime table if needed
    regime_table = None
    if need_regime_table and isinstance(daily_df.columns, pd.MultiIndex):
        closes_panel = daily_df["Close"]
        index_closes: Dict[str, pd.Series] = {}
        for sym in index_symbols:
            if sym in closes_panel.columns:
                index_closes[sym] = closes_panel[sym].dropna()

        vix_close = None
        if vix_symbol in closes_panel.columns:
            vix_close = closes_panel[vix_symbol].dropna()

        if index_closes:
            regime_cfg_obj = MarketRegimeConfig(
                index_symbols=index_symbols,
                use_vix_filter=True,
                vix_symbol=vix_symbol,
            )
            log(
                f"Building historical regime table for indices={index_symbols}, vix={vix_symbol}",
                level="info",
            )
            regime_table = build_historical_regime_table(
                index_closes=index_closes,
                vix_close=vix_close,
                cfg=regime_cfg_obj,
            )
            log(
                f"Historical regime table built with {len(regime_table)} rows.",
                level="info",
            )
        else:
            log(
                "No index close series available for regime table; "
                "regime gating will be effectively disabled.",
                level="warn",
            )

    result = backtest(
        daily_df=daily_df,
        start=start,
        end=end,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        max_long=args.max_long,
        max_short=args.max_short,
        mode=effective_mode,
        universe_tickers=sorted(all_tickers),
        weekly_df=weekly_df,
        weekly_snapshots=weekly_snapshots if weekly_snapshots else None,
        regime_table=regime_table,
        use_regime_long=use_regime_long_effective,
        use_regime_short=use_regime_short_effective,
        auto_mode=auto_mode,
        benchmark=benchmark,
        use_coppock_long=use_coppock_long_effective,
        use_coppock_short=use_coppock_short_effective,
        coppock_series=coppock_series,
        breadth_enabled=breadth_enabled,
        show_adx_skips=show_adx_skips_effective,
        show_breadth_skips=show_breadth_skips_effective,
        show_coppock_skips=show_coppock_skips_effective,
        long_logic_cfg=bt_long_cfg,
        market_cfg=market_cfg,
        industry_cfg=industry_cfg,
        vix_symbol=vix_symbol,
    )

    portfolio: Portfolio = result["portfolio"]  # type: ignore
    trades: List[Trade] = result["trades"]  # type: ignore
    equity_curve = result["equity_curve"]  # type: ignore

    # Summary
    final_eq = portfolio.equity
    pnl = final_eq - args.capital
    pnl_pct = (final_eq / args.capital - 1.0) * 100.0
    log(
        f"Backtest complete. Final equity: ${final_eq:,.2f} (P/L ${pnl:,.2f}, {pnl_pct:.2f}%) "
        f"— Trades: {len(trades)}",
        level="ok",
    )

    os.makedirs("./output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = os.path.join("./output", f"live_logic_bt_trades_{ts}.csv")
    eq_path = os.path.join("./output", f"live_logic_bt_equity_{ts}.png")
    monthly_path = os.path.join("./output", f"live_logic_bt_monthly_{ts}.csv")

    save_trade_log(trades, trades_path)
    save_equity_curve(equity_curve, eq_path)
    save_monthly_pnl(trades, equity_curve, args.capital, monthly_path)


if __name__ == "__main__":
    main()
