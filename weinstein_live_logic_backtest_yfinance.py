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
      daily volume ≥ ~1.3× 50-day avg
    * Short: price below MA30, RS weak, breakdown < prior 50-day low,
      daily volume ≥ ~1.3× 50-day avg
- Risk sizing:
    * Risk per trade = equity * risk_per_trade / per-share-risk
    * Stops use ATR and MA30 guard, similar to your intraday logic
- Outputs:
    * Trade log CSV
    * Equity curve PNG
    * Monthly P/L CSV + printed summary
"""

import argparse
import os
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared Weinstein indicators (ADX + breadth single source of truth)
from weinstein_indicators import (
    compute_adx_series,
    ADX_WINDOW,
    ADX_MIN,
    compute_breadth_series_above_ma,
)

# Shared short-side core (same module used by intraday short watcher)
from weinstein_short_core import (
    SHORT_BREAK_PCT,
    SHORT_HARD_STOP_PCT,
    SHORT_TRAIL_ATR_MULT,
    SHORT_MA_GUARD_PCT,
    VOL_PACE_MIN,
    READY_ABOVE_MA_PCT,
    _short_entry_stop_targets,
    _short_ready_to_close,
    eval_short_bar as eval_short_bar_core,
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


def load_weekly_snapshots(snapshot_dir: str) -> List[tuple[date, pd.DataFrame]]:
    """
    Load historical weekly equity CSV snapshots from snapshot_dir.

    Returns a list of (as_of_date, df) sorted by as_of_date.
    If the directory does not exist or nothing matches, returns [].
    """
    if not os.path.isdir(snapshot_dir):
        log(f"No snapshot dir {snapshot_dir} (skipping historical snapshots).", level="info")
        return []

    snapshots: List[tuple[date, pd.DataFrame]] = []
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
    snapshots: List[tuple[date, pd.DataFrame]],
    as_of_ts: pd.Timestamp,
) -> Optional[tuple[date, pd.DataFrame]]:
    """
    Choose the most recent snapshot with as_of_date <= current date.
    If none qualifies yet (e.g. before first snapshot), returns None.
    """
    if not snapshots:
        return None
    target = as_of_ts.date()
    chosen: Optional[tuple[date, pd.DataFrame]] = None
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
    log(f"{side.upper()} universe size: {len(df)} symbols.", level="info")
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
LONG_BREAK_PCT = 0.004  # ≈0.4% above pivot breakout, matching short break magnitude
LONG_STOP_HARD = 0.20  # 20% hard stop (Weinstein-style disaster stop)
LONG_TRAIL_ATR = 2.0  # ATR-based cushion
LONG_MA_GUARD = 0.03  # extra guard vs MA30 (≈3% under)

# Volume filters (approximate your intraday VOL_PACE_MIN 1.3×)
LONG_VOL_MIN = 1.30

PIVOT_LOOKBACK_DAYS = 50  # pivot highs/lows over last ~10 weeks

# Breadth Health filter (Advance/Decline strength)
# Approximates "% of S&P500 above MA50" by using a breadth universe of tickers.
BREADTH_MA_WINDOW = 50
BREADTH_MIN_LONG = 0.60  # require 60% of breadth universe above MA50 to allow new longs


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


# ---------------- Entry / exit rules (LONG side only) ----------------


def should_enter_long(
    price: float,
    ma30_val: float,
    pivot_high: float,
    rs_above_ma: bool,
    vol_mult: float,
) -> bool:
    if np.isnan(price) or np.isnan(ma30_val) or np.isnan(pivot_high):
        return False
    # RS must be strong (above its MA)
    if not rs_above_ma:
        return False
    # Price must be above MA30
    if price < ma30_val:
        return False
    # Breakout above pivot high by ≈0.4%
    if price < pivot_high * (1.0 + LONG_BREAK_PCT):
        return False
    # Volume pace gate ~1.3× 50dma, like intraday VOL_PACE_MIN
    if not np.isnan(vol_mult) and vol_mult < LONG_VOL_MIN:
        return False
    return True


def long_stop_level(entry: float, atr: float, ma30_val: float) -> float:
    if np.isnan(entry):
        return np.nan
    hard = entry * (1.0 - LONG_STOP_HARD)
    atr_stop = entry - LONG_TRAIL_ATR * atr if not np.isnan(atr) else np.nan
    ma_guard = ma30_val * (1.0 - LONG_MA_GUARD) if not np.isnan(ma30_val) else np.nan
    cands = [c for c in [hard, atr_stop, ma_guard] if not np.isnan(c)]
    return max(cands) if cands else hard


def should_exit_long(price: float, stop: float, ma30_val: float) -> bool:
    if np.isnan(price):
        return False
    # 1) Stop violation
    if not np.isnan(stop) and price <= stop:
        return True
    # 2) Extra guard: under MA30 by ~3%
    if not np.isnan(ma30_val) and price <= ma30_val * (1.0 - LONG_MA_GUARD):
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
    weekly_snapshots: Optional[List[tuple[date, pd.DataFrame]]] = None,
) -> Dict[str, object]:
    """
    mode: "long", "short", or "both"

    If weekly_snapshots is provided and non-empty:
      - uses dynamic weekly universes per date (Option A).
    Else:
      - uses single weekly_df snapshot (current behavior).
    """
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
    if isinstance(daily_df.columns, pd.MultiIndex) and "Close" in daily_df.columns.levels[0]:
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
        log(
            "Daily data not in expected MultiIndex Close panel; breadth gate disabled.",
            level="warn",
        )
        breadth_series = None

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

    # State for dynamic snapshots
    current_snapshot_date: Optional[date] = None
    current_long_universe: Optional[pd.DataFrame] = static_long_universe
    current_short_universe: Optional[pd.DataFrame] = static_short_universe

    # Short-side CORE state (per-ticker)
    short_states: Dict[str, dict] = {}

    # Main daily loop
    for i, dt in enumerate(all_dates):
        if dt < start_dt or dt > end_dt:
            continue

        # ----- choose weekly universe for this date -----
        if use_snapshots and weekly_snapshots:
            snap = pick_snapshot_for_date(weekly_snapshots, dt)
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
                    log(
                        f"Using weekly snapshot as of {snap_date} for {dt.date()} — "
                        f"long_univ={len(current_long_universe)}, short_univ={len(current_short_universe)}",
                        level="debug",
                    )
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
                if dt in closes.index:
                    price_today[t] = _safe_float(closes.loc[dt, t])
        else:
            # Single ticker case (unlikely in your universe)
            if dt in daily_df.index:
                price_today["SINGLE"] = _safe_float(daily_df["Close"].loc[dt])

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
        equity_curve.append({"date": dt, "equity": eq})

        # Compute breadth gate for this day (for new LONG entries)
        breadth_ok = True
        breadth_val = np.nan
        if breadth_series is not None and dt in breadth_series.index:
            breadth_val = float(breadth_series.loc[dt])
            if not np.isnan(breadth_val):
                breadth_ok = breadth_val >= BREADTH_MIN_LONG
            else:
                breadth_ok = True  # if NaN, don't block
        # Optional debug logging when breadth blocks new longs
        if not breadth_ok:
            log(
                f"[SKIP-BREADTH] No new LONGs on {dt.date()} because breadth="
                f"{breadth_val:.2%} < {BREADTH_MIN_LONG:.0%}",
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
                ma_series.loc[dt]
                if ma_series is not None and dt in ma_series.index
                else np.nan
            )

            if pos.side == "long":
                if not should_exit_long(p, pos.stop, ma_val):
                    continue
                exit_price = p
                pnl = pos.qty * (exit_price - pos.entry_price)
            else:
                # Short side: use CORE-style stop + READY-to-close
                hit_stop = (not np.isnan(pos.stop)) and p >= pos.stop
                ready_close = False
                if not np.isnan(ma_val):
                    ready_close = _short_ready_to_close(p, ma_val)
                if not (hit_stop or ready_close):
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
                    exit_date=dt,
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

        # LONG entries (gated by breadth_ok)
        if (
            mode in ("long", "both")
            and n_long_now < max_long
            and breadth_ok
        ):
            for _, row in long_universe.iterrows():
                t = str(row["ticker"]).upper()
                pos_key = f"{t}_long"
                if pos_key in portfolio.positions:
                    continue
                price = price_today.get(t, np.nan)
                if np.isnan(price):
                    continue

                ma_series = ma_cache.get(t)
                ma_val = (
                    ma_series.loc[dt]
                    if ma_series is not None and dt in ma_series.index
                    else np.nan
                )
                pivot_high = get_pivot_high(daily_df, t, dt)
                rs_above_ma = bool(row.get("rs_above_ma", False))
                vol_mult = volume_vs_50dma(daily_df, t, dt)

                # ADX filter (mirrors intraday: NaN → no block, real < ADX_MIN → block)
                adx_series = adx_cache.get(t)
                if (
                    adx_series is not None
                    and not adx_series.empty
                    and dt in adx_series.index
                ):
                    adx_val = float(adx_series.loc[dt])
                else:
                    adx_val = np.nan

                if np.isnan(adx_val):
                    adx_ok = True
                else:
                    adx_ok = adx_val >= ADX_MIN

                if not adx_ok:
                    # Diagnostic similar to intraday watcher
                    log(
                        f"[SKIP-ADX] {t} because ADX{ADX_WINDOW}={adx_val:.1f} < {ADX_MIN:.1f} on {dt.date()}",
                        level="debug",
                    )
                    continue

                if not should_enter_long(
                    price, ma_val, pivot_high, rs_above_ma, vol_mult
                ):
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
                    opened=dt,
                )
                n_long_now += 1
                if n_long_now >= max_long:
                    break

        # SHORT entries — now driven by CORE eval_short_bar()
        if mode in ("short", "both") and n_short_now < max_short:
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
                    ma_series.loc[dt]
                    if ma_series is not None and dt in ma_series.index
                    else np.nan
                )
                pivot_low = get_pivot_low(daily_df, t, dt)
                rs_above_ma = bool(row.get("rs_above_ma", False))
                vol_mult = volume_vs_50dma(daily_df, t, dt)

                # Require weak RS for shorts (same gating as PROD watcher)
                if rs_above_ma:
                    continue

                # Daily adaptor for CORE short eval:
                # - closes_tail: last 2 daily closes up to today
                # - intraday_interval != "60m" → CORE uses closes_tail
                price_series = get_close_series(daily_df, t)
                closes_tail = list(price_series.loc[:dt].tail(2).values)

                state_in = short_states.get(
                    t, {"short_state": "IDLE", "short_hits": [], "short_cooldown": 0}
                )

                new_state, flags = eval_short_bar_core(
                    price=price,
                    ma30=ma_val,
                    pivot_low=pivot_low,
                    pace_full=vol_mult,   # daily vol / 50dma → same as intraday pace_full
                    pace_intra=np.nan,    # not used in daily mode
                    elapsed_min=None,     # not used when intraday_interval != "60m"
                    closes_tail=closes_tail,
                    state=state_in,
                    intraday_interval="1d",  # force non-60m branch
                    test_ease=False,
                )
                short_states[t] = new_state

                if not flags.get("short_trigger_now", False):
                    continue

                # At this point, CORE has already enforced:
                # - price breakdown via _short_price_break
                # - volume pace gate via VOL_PACE_MIN
                atr = atr_cache.get(t, np.nan)
                entry, stop, t1, t2 = _short_entry_stop_targets(
                    price, ma_val, pivot_low, atr
                )
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
                    opened=dt,
                )
                n_short_now += 1
                if n_short_now >= max_short:
                    break

        if (i + 1) % 20 == 0:
            log(
                f"Progress: {dt.date()} — equity ${portfolio.equity:,.2f}, "
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
        choices=["long", "short", "both"],
        help="Enable long-only, short-only, or both",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    VERBOSE = not args.quiet

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

    log(
        f"Backtest range: {start} → {end} | mode={args.mode}, capital={args.capital:,.2f}, "
        f"risk_per_trade={args.risk_per_trade:.3f}, max_long={args.max_long}, max_short={args.max_short}",
        level="info",
    )

    # ---- Try to use historical weekly snapshots (Option A) ----
    weekly_snapshots = load_weekly_snapshots(WEEKLY_SNAPSHOT_DIR)

    weekly_df: Optional[pd.DataFrame] = None
    all_tickers: set[str] = set()

    if weekly_snapshots:
        # Union of tickers across all snapshots
        for _, df in weekly_snapshots:
            if "ticker" in df.columns:
                all_tickers.update(df["ticker"].astype(str).str.upper())
        log(
            f"Using historical weekly snapshots for universe (unique tickers={len(all_tickers)}).",
            level="info",
        )
    else:
        # Fallback: single latest weekly report (current behavior)
        weekly_df = load_weekly_report()
        all_tickers.update(weekly_df["ticker"].astype(str).str.upper())
        log(
            f"No historical snapshots; using latest weekly report for static universe "
            f"({len(all_tickers)} tickers).",
            level="info",
        )

    if not all_tickers:
        raise RuntimeError("Universe of tickers is empty; cannot run backtest.")

    daily_df = download_daily_bars(sorted(all_tickers), start, end)

    result = backtest(
        daily_df=daily_df,
        start=start,
        end=end,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        max_long=args.max_long,
        max_short=args.max_short,
        mode=args.mode,
        universe_tickers=sorted(all_tickers),
        weekly_df=weekly_df,
        weekly_snapshots=weekly_snapshots if weekly_snapshots else None,
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
