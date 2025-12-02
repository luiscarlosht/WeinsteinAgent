#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
market_regime.py

Weinstein Chapter 8 style market regime filter + VIX regime filter (v2).

Goal:
  - Use the behavior of the major indices vs. their 200-day moving averages
    to decide whether the overall environment is favorable for new LONG / SHORT entries.
  - Add a VIX-based volatility filter to prevent:
        * New LONGS when volatility is too high
        * New SHORTS when volatility is too low
  - This is a coarse "market is in gear or not" filter, not a stock-selection tool.

Core Weinstein-style ideas (Chapter 8, adapted for automation):
  - Look at major market averages (e.g. S&P 500, Nasdaq, Dow).
  - Compare price vs. a long-term moving average (e.g. 200-day).
  - Evaluate the slope of that moving average:
      * Rising MA + index above MA  → bullish.
      * Falling MA + index below MA → bearish.
      * Mixed conditions           → neutral.

VIX Regime Filter (v2):
  - Download VIX (^VIX) daily closes.
  - Apply simple hard thresholds:

      * NO new longs if VIX > 22
      * NO new shorts if VIX < 15

  - In “normal / good” years VIX spends a lot of time < 20, so
    this mainly filters out high-volatility chop and low-vol short traps.

We implement:
  - Download recent daily data for the chosen indices via yfinance.
  - Compute 200-day SMA and its "slope" over N days (default 20) for each index.
  - Classify each index as BULLISH / BEARISH / NEUTRAL.
  - Aggregate to a single market regime: BULL, BEAR, NEUTRAL, UNKNOWN.
  - Download VIX, compute last close, and derive VIX long/short gates.

Tiny helper for intraday / short watchers
-----------------------------------------
We expose a very small API that your watchers can call without having to know
about all the internals:

    from market_regime import inspect

    label, long_ok, short_ok = inspect()
    # label:   "BULL", "BEAR", "NEUTRAL", "UNKNOWN"
    # long_ok:  True if environment ok for *new longs*
    # short_ok: True if environment ok for *new shorts*

The final long_ok / short_ok are:

  1) Weinstein regime gates:
        - LONGS:
            * Allowed in BULL and NEUTRAL.
        - SHORTS:
            * Allowed ONLY in BEAR.
        - UNKNOWN:
            * Treat as NEUTRAL for longs, but block shorts.

  2) AND VIX gates:
        - LONGS:
            * Allowed only if VIX <= 22
        - SHORTS:
            * Allowed only if VIX >= 15

So:
    long_ok  = long_ok_from_regime  AND long_ok_from_vix
    short_ok = short_ok_from_regime AND short_ok_from_vix
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


# ─────────────────────────────
# ENUMS / DATA CLASSES
# ─────────────────────────────

class IndexState(Enum):
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()
    UNKNOWN = auto()


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class IndexMetrics:
    symbol: str
    state: IndexState
    last_close: float
    ma_value: float
    ma_slope_pct: float
    above_ma: bool
    ma_rising: bool
    n_days_ma_slope: int
    ma_window: int
    data_points: int


@dataclass
class MarketRegimeConfig:
    # Which indices to watch (Weinstein: major averages)
    index_symbols: List[str] = None

    # Long-term MA window (200 trading days ~ 40 weeks)
    ma_window: int = 200

    # How many days apart to compare MA for slope
    ma_slope_days: int = 20

    # Threshold: MA slope must be >= this (%) to be "rising"
    ma_slope_min_pct: float = 0.0

    # Minimum fraction of indices that must be bullish for global BULL regime
    min_bullish_fraction: float = 0.6

    # Minimum fraction of indices that must be bearish for global BEAR regime
    min_bearish_fraction: float = 0.6

    # How many calendar days of history to request from yfinance
    # (we ask more than ma_window + ma_slope_days to be safe)
    history_days: int = 400

    # If True, we print details / debugging info
    verbose: bool = False

    # ── VIX filter settings ───────────────────────────
    # Whether to apply the VIX-based long/short gates
    use_vix_filter: bool = True

    # Symbol for VIX
    vix_symbol: str = "^VIX"

    # How many days of history for VIX (usually same as history_days)
    vix_history_days: int = 400

    # Max VIX to allow new longs. If VIX > vix_long_max → block new longs.
    vix_long_max: float = 20.0

    # Min VIX to allow new shorts. If VIX < vix_short_min → block new shorts.
    vix_short_min: float = 12.0

    def __post_init__(self):
        if self.index_symbols is None:
            # Default major US indices: S&P 500, Nasdaq-100, Dow, Russell 2000
            self.index_symbols = ["^GSPC", "^NDX", "^DJI", "^RUT"]


@dataclass
class MarketRegimeSnapshot:
    regime: MarketRegime
    as_of: pd.Timestamp
    index_metrics: List[IndexMetrics]

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "as_of": self.as_of.isoformat(),
            "indices": [asdict(m) for m in self.index_metrics],
        }


# ─────────────────────────────
# CORE COMPUTATION
# ─────────────────────────────

def _download_index_data(symbols: List[str], cfg: MarketRegimeConfig) -> Dict[str, pd.Series]:
    """
    Download daily close data for each index symbol.

    Returns dict: {symbol: pd.Series of close prices (indexed by datetime)}
    """
    if yf is None:
        raise RuntimeError("yfinance is not available; cannot compute market regime.")

    uniq = sorted(set(symbols))
    period = f"{cfg.history_days}d"

    try:
        data = yf.download(
            uniq,
            period=period,
            interval="1d",
            group_by="column",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download index data via yfinance: {e}") from e

    out: Dict[str, pd.Series] = {}

    if isinstance(data.columns, pd.MultiIndex):
        # Multi-index: ('Close', symbol)
        if ("Close" not in data.columns.get_level_values(0)):
            raise RuntimeError("Downloaded data has no 'Close' column.")

        close_df = data["Close"]
        for sym in uniq:
            try:
                series = close_df[sym].dropna()
                if series.empty:
                    continue
                out[sym] = series
            except Exception:
                continue
    else:
        # Single symbol case
        if "Close" not in data.columns:
            raise RuntimeError("Downloaded data has no 'Close' column.")
        series = data["Close"].dropna()
        if not series.empty:
            out[uniq[0]] = series

    return out


def _classify_index(symbol: str, close: pd.Series, cfg: MarketRegimeConfig) -> IndexMetrics:
    """
    Compute 200d MA, slope over ma_slope_days, and classify this index.
    """
    close = close.sort_index()
    ma = close.rolling(cfg.ma_window).mean().dropna()

    if ma.empty or len(ma) <= cfg.ma_slope_days:
        # Not enough data to say anything
        last_close = float(close.iloc[-1])
        return IndexMetrics(
            symbol=symbol,
            state=IndexState.UNKNOWN,
            last_close=last_close,
            ma_value=float("nan"),
            ma_slope_pct=float("nan"),
            above_ma=False,
            ma_rising=False,
            n_days_ma_slope=cfg.ma_slope_days,
            ma_window=cfg.ma_window,
            data_points=len(close),
        )

    last_ma = float(ma.iloc[-1])
    last_close = float(close.iloc[-1])

    # Compare MA vs MA from N days ago
    past_ma_index = -1 - cfg.ma_slope_days
    try:
        past_ma = float(ma.iloc[past_ma_index])
    except IndexError:
        past_ma = float("nan")

    if past_ma > 0 and not np.isnan(past_ma):
        ma_slope_pct = (last_ma / past_ma - 1.0) * 100.0
    else:
        ma_slope_pct = float("nan")

    above_ma = last_close > last_ma
    ma_rising = ma_slope_pct >= cfg.ma_slope_min_pct if not np.isnan(ma_slope_pct) else False

    # Weinstein-style classification:
    #   - BULLISH: above MA and MA is rising.
    #   - BEARISH: below MA and MA is falling (slope clearly negative).
    #   - Otherwise NEUTRAL.
    if above_ma and ma_rising:
        state = IndexState.BULLISH
    elif (not above_ma) and (ma_slope_pct <= -cfg.ma_slope_min_pct):
        state = IndexState.BEARISH
    else:
        state = IndexState.NEUTRAL

    return IndexMetrics(
        symbol=symbol,
        state=state,
        last_close=last_close,
        ma_value=last_ma,
        ma_slope_pct=ma_slope_pct,
        above_ma=above_ma,
        ma_rising=ma_rising,
        n_days_ma_slope=cfg.ma_slope_days,
        ma_window=cfg.ma_window,
        data_points=len(close),
    )


def _aggregate_market_regime(metrics: List[IndexMetrics], cfg: MarketRegimeConfig) -> MarketRegime:
    """
    Combine per-index states into one regime.
    """
    if not metrics:
        return MarketRegime.UNKNOWN

    n = len(metrics)
    n_bull = sum(1 for m in metrics if m.state is IndexState.BULLISH)
    n_bear = sum(1 for m in metrics if m.state is IndexState.BEARISH)

    frac_bull = n_bull / n
    frac_bear = n_bear / n

    if frac_bull >= cfg.min_bullish_fraction:
        return MarketRegime.BULL
    if frac_bear >= cfg.min_bearish_fraction:
        return MarketRegime.BEAR
    return MarketRegime.NEUTRAL


def detect_market_regime(cfg: Optional[MarketRegimeConfig] = None) -> MarketRegimeSnapshot:
    """
    Main entry point.

    Returns a MarketRegimeSnapshot with:
      - overall regime (BULL / BEAR / NEUTRAL / UNKNOWN)
      - per-index metrics (state, last_close, MA, slope, etc.)

    This is what we will call from intraday / short watchers to gate new entries
    on the Weinstein side. VIX gating is applied at the `inspect()` level.
    """
    if cfg is None:
        cfg = MarketRegimeConfig()

    if yf is None:
        # yfinance missing → we can't compute regime properly
        now = pd.Timestamp.now(tz="UTC")
        return MarketRegimeSnapshot(
            regime=MarketRegime.UNKNOWN,
            as_of=now,
            index_metrics=[],
        )

    data = _download_index_data(cfg.index_symbols, cfg)
    metrics: List[IndexMetrics] = []

    for sym in cfg.index_symbols:
        series = data.get(sym)
        if series is None or series.empty:
            m = IndexMetrics(
                symbol=sym,
                state=IndexState.UNKNOWN,
                last_close=float("nan"),
                ma_value=float("nan"),
                ma_slope_pct=float("nan"),
                above_ma=False,
                ma_rising=False,
                n_days_ma_slope=cfg.ma_slope_days,
                ma_window=cfg.ma_window,
                data_points=0,
            )
        else:
            m = _classify_index(sym, series, cfg)
        metrics.append(m)

    regime = _aggregate_market_regime(metrics, cfg)
    as_of = pd.Timestamp.now(tz="UTC")

    if cfg.verbose:
        print("Market regime snapshot as of", as_of.isoformat())
        for m in metrics:
            print(
                f"  {m.symbol:6s}  state={m.state.name:8s}  "
                f"close={m.last_close:8.2f}  ma{m.ma_window}={m.ma_value:8.2f}  "
                f"slope{m.n_days_ma_slope:02d}={m.ma_slope_pct:6.2f}%  "
                f"above_ma={m.above_ma}  ma_rising={m.ma_rising}  n={m.data_points}"
            )
        print(f"→ Overall regime: {regime.value.upper()}")

    return MarketRegimeSnapshot(
        regime=regime,
        as_of=as_of,
        index_metrics=metrics,
    )


# ─────────────────────────────
# VIX FILTER HELPERS
# ─────────────────────────────

def _download_vix_series(cfg: MarketRegimeConfig) -> pd.Series:
    """
    Download daily close series for VIX (cfg.vix_symbol).
    Returns pd.Series of closes indexed by date.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available; cannot download VIX data.")

    period = f"{cfg.vix_history_days}d"
    try:
        data = yf.download(
            cfg.vix_symbol,
            period=period,
            interval="1d",
            auto_adjust=False,  # for VIX, no need to adjust
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download VIX data via yfinance: {e}") from e

    if data.empty or "Close" not in data.columns:
        raise RuntimeError("Downloaded VIX data has no 'Close' column or is empty.")

    return data["Close"].dropna()


def _compute_vix_gates(cfg: MarketRegimeConfig) -> Tuple[float, bool, bool]:
    """
    Compute VIX-based gating for longs/shorts.

    Returns:
        (vix_last, long_ok_vix, short_ok_vix)

    - If use_vix_filter is False OR yfinance missing OR download fails:
        vix_last = nan
        long_ok_vix  = True
        short_ok_vix = True
    """
    if (not cfg.use_vix_filter) or (yf is None):
        return float("nan"), True, True

    try:
        series = _download_vix_series(cfg)
        if series.empty:
            raise RuntimeError("Empty VIX series.")
        vix_last = float(series.iloc[-1])
    except Exception as e:
        if cfg.verbose:
            print(f"[VIX] Warning: could not compute VIX gates: {e}")
        return float("nan"), True, True

    long_ok_vix = vix_last <= cfg.vix_long_max
    short_ok_vix = vix_last >= cfg.vix_short_min

    if cfg.verbose:
        print(
            f"[VIX] {cfg.vix_symbol}: last={vix_last:.2f} | "
            f"long_ok_vix={long_ok_vix} (max {cfg.vix_long_max:.2f}) | "
            f"short_ok_vix={short_ok_vix} (min {cfg.vix_short_min:.2f})"
        )

    return vix_last, long_ok_vix, short_ok_vix


# ─────────────────────────────
# TINY INSPECT WRAPPER FOR WATCHERS
# ─────────────────────────────

def _compute_long_short_flags(regime: MarketRegime) -> Tuple[bool, bool]:
    """
    Map a MarketRegime → (long_ok, short_ok) according to your Chapter 8 rules.

    - LONGS:
        * Allowed in BULL and NEUTRAL.
    - SHORTS:
        * Allowed ONLY in BEAR.
    - UNKNOWN:
        * Treat as NEUTRAL for longs, but block shorts.
    """
    if regime is MarketRegime.BEAR:
        long_ok = False
        short_ok = True
    elif regime is MarketRegime.BULL:
        long_ok = True
        short_ok = False
    else:
        # NEUTRAL or UNKNOWN
        long_ok = True
        short_ok = False
    return long_ok, short_ok


def inspect(cfg: Optional[MarketRegimeConfig] = None) -> Tuple[str, bool, bool]:
    """
    Tiny helper for intraday / short watchers.

    Returns:
        (label, long_ok, short_ok)

        label:     "BULL", "BEAR", "NEUTRAL", "UNKNOWN"
        long_ok:   bool → whether new LONG entries are allowed
        short_ok:  bool → whether new SHORT entries are allowed

    Logic:
        1) Compute Weinstein regime and base long_ok/short_ok.
        2) Compute VIX gates and AND them with the regime gates.
    """
    # Use same config for both regime and VIX gating
    if cfg is None:
        cfg = MarketRegimeConfig()

    snap = detect_market_regime(cfg)
    regime = snap.regime

    # Weinstein regime gates
    long_ok_regime, short_ok_regime = _compute_long_short_flags(regime)

    # VIX gates
    _, long_ok_vix, short_ok_vix = _compute_vix_gates(cfg)

    long_ok = long_ok_regime and long_ok_vix
    short_ok = short_ok_regime and short_ok_vix

    label = regime.name.upper()  # BULL / BEAR / NEUTRAL / UNKNOWN
    return label, long_ok, short_ok


# Backwards-compatible aliases in case watchers look for a specific name
def inspect_for_intraday(cfg: Optional[MarketRegimeConfig] = None) -> Tuple[str, bool, bool]:
    """Alias for inspect()."""
    return inspect(cfg)


def inspect_market_regime(cfg: Optional[MarketRegimeConfig] = None) -> Tuple[str, bool, bool]:
    """Alias for inspect()."""
    return inspect(cfg)


# ─────────────────────────────
# SIMPLE CLI FOR MANUAL CHECKS
# ─────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Weinstein-style market regime detector (Chapter 8) + VIX filter.")
    ap.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index symbols to use (default: ^GSPC ^NDX ^DJI ^RUT)",
    )
    ap.add_argument(
        "--ma-window",
        type=int,
        default=200,
        help="Moving average window (days), default=200.",
    )
    ap.add_argument(
        "--ma-slope-days",
        type=int,
        default=20,
        help="Number of days to look back when computing MA slope, default=20.",
    )
    ap.add_argument(
        "--ma-slope-min-pct",
        type=float,
        default=0.0,
        help="Minimum MA slope (%%) to be considered rising/falling, default=0.0.",
    )
    ap.add_argument(
        "--min-bullish-fraction",
        type=float,
        default=0.6,
        help="Fraction of indices that must be bullish to declare a BULL regime, default=0.6.",
    )
    ap.add_argument(
        "--min-bearish-fraction",
        type=float,
        default=0.6,
        help="Fraction of indices that must be bearish to declare a BEAR regime, default=0.6.",
    )
    ap.add_argument(
        "--history-days",
        type=int,
        default=400,
        help="Number of calendar days of history to request from yfinance, default=400.",
    )
    # VIX-related CLI toggles (optional)
    ap.add_argument(
        "--no-vix-filter",
        action="store_true",
        help="Disable VIX filter (only Weinstein regime will be used).",
    )
    ap.add_argument(
        "--vix-symbol",
        type=str,
        default="^VIX",
        help="VIX symbol to use, default=^VIX.",
    )
    ap.add_argument(
        "--vix-long-max",
        type=float,
        default=22.0,
        help="Max VIX to allow new longs (default=22.0). If VIX > this, longs are blocked.",
    )
    ap.add_argument(
        "--vix-short-min",
        type=float,
        default=15.0,
        help="Min VIX to allow new shorts (default=15.0). If VIX < this, shorts are blocked.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-index details; only print final regime and gates.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = MarketRegimeConfig(
        index_symbols=args.indices,
        ma_window=args.ma_window,
        ma_slope_days=args.ma_slope_days,
        ma_slope_min_pct=args.ma_slope_min_pct,
        min_bullish_fraction=args.min_bullish_fraction,
        min_bearish_fraction=args.min_bearish_fraction,
        history_days=args.history_days,
        verbose=not args.quiet,
        use_vix_filter=not args.no_vix_filter,
        vix_symbol=args.vix_symbol,
        vix_long_max=args.vix_long_max,
        vix_short_min=args.vix_short_min,
        vix_history_days=args.history_days,
    )

    snap = detect_market_regime(cfg)

    # Weinstein regime gates
    long_ok_regime, short_ok_regime = _compute_long_short_flags(snap.regime)

    # VIX gates
    vix_last, long_ok_vix, short_ok_vix = _compute_vix_gates(cfg)

    # Combined gates (what inspect() would give)
    long_ok = long_ok_regime and long_ok_vix
    short_ok = short_ok_regime and short_ok_vix

    if args.quiet:
        # Just print final regime label + combined gates
        print(snap.regime.value)
        print(f"long_ok={long_ok} short_ok={short_ok}")
    else:
        print()
        print("Final regime:", snap.regime.value.upper())
        print()

        if not np.isnan(vix_last):
            print(
                f"VIX ({cfg.vix_symbol}): last={vix_last:.2f} | "
                f"long_ok_vix={long_ok_vix} (max {cfg.vix_long_max:.2f}) | "
                f"short_ok_vix={short_ok_vix} (min {cfg.vix_short_min:.2f})"
            )
        else:
            print("VIX: unavailable (filter effectively disabled).")

        print()
        print(
            f"Regime-only gates: long_ok_regime={long_ok_regime}, "
            f"short_ok_regime={short_ok_regime}"
        )
        print(
            f"Combined gates (inspect): long_ok={long_ok}, short_ok={short_ok}"
        )
        print()


if __name__ == "__main__":
    main()
