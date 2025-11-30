#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Weinstein Intraday Simulator with progress logging.

This script backtests a basic intraday breakout strategy over a full calendar
year, using a universe derived from your latest weekly report and a simple
Weinstein-style market regime classifier (BULL / NEUTRAL / BEAR).

Usage:

    python3 weinstein_intraday_sim.py --year 2025 --mode regime

Key behaviour:

- Loads ./config.yaml to get:
    app.benchmark          (e.g., SPY)
    app.ordering.account_size
    app.ordering.risk_per_trade_pct
- Loads the newest ./output/weinstein_weekly_*.csv to get your Stage 1/2 universe
- Downloads daily + 60m intraday bars for [year-1-11-01, year+1-02-01]
- Classifies each trading day for the benchmark as BULL / NEUTRAL / BEAR
  based on price vs 150-day MA and MA slope.
- Simulates a simple breakout system on 60m bars:
    LONG entry: price breaks above 10-week pivot and 150d MA, regime allows LONG
    SHORT entry: price breaks below 150d MA, regime allows SHORT
    Stops / exits are simple hard-stop and take-profit multiples.
- Logs progress to the console every ~10% of bars so you can see how far
  through the simulation it is.

NOTE: This is simpler than your full intraday watcher logic, but it's intended
to give you a realistic P/L profile and, most importantly, clear progress
feedback during long backtest runs.
"""

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

# ========= Logging helpers =========

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str, level: str = "info") -> None:
    prefix = {
        "info": "•",
        "ok": "✅",
        "step": "▶️",
        "warn": "⚠️",
        "err": "❌",
        "debug": "··",
    }.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)


# ========= Config / Weekly universe =========

BENCHMARK_DEFAULT = "SPY"
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
INTRADAY_INTERVAL = "60m"
LOOKBACK_START_MONTH = 11  # start downloads from Nov of previous year
LOOKAHEAD_END_MONTH = 2    # end downloads in Feb of next year

PIVOT_LOOKBACK_WEEKS = 10
SMA_DAYS = 150
HARD_STOP_PCT = 0.08    # 8% hard stop
TP_R_MULT = 2.0         # take-profit at 2R (2× hard-stop distance)

@dataclass
class SimConfig:
    year: int
    benchmark: str
    account_size: float
    risk_per_trade_pct: float
    mode: str  # "regime", "long_only", "short_only"


def load_config(path: str) -> Tuple[dict, str, float, float]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    app = cfg.get("app", {}) or {}
    ordering = app.get("ordering") or {}
    if not isinstance(ordering, dict):
        ordering = {}

    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    account_size = float(ordering.get("account_size", 5000.0))
    risk_pct = float(ordering.get("risk_per_trade_pct", 0.01))

    return cfg, benchmark, account_size, risk_pct


def newest_weekly_csv() -> str:
    files = [
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found in {WEEKLY_OUTPUT_DIR}. "
            "Run weinstein_report_weekly.py first."
        )
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report() -> Tuple[pd.DataFrame, str]:
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path


def build_universe(weekly_df: pd.DataFrame, benchmark: str) -> List[str]:
    w = weekly_df.rename(columns=str.lower)
    if "ticker" not in w.columns or "stage" not in w.columns:
        raise ValueError("Weekly CSV missing 'ticker' and/or 'stage' columns.")

    focus = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])].copy()
    tickers = sorted(set(focus["ticker"].dropna().astype(str).str.upper().tolist()))

    if benchmark.upper() not in tickers:
        tickers.append(benchmark.upper())

    return tickers


# ========= Data helpers =========

def download_data(universe: List[str], year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_all = datetime(year - 1, LOOKBACK_START_MONTH, 1)
    end_all = datetime(year + 1, LOOKAHEAD_END_MONTH, 1)

    log(
        f"Downloading daily + intraday for {len(universe)} tickers "
        f"({start_all.date()} → {end_all.date()})...",
        level="step",
    )

    daily = yf.download(
        universe,
        start=start_all.strftime("%Y-%m-%d"),
        end=end_all.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    intraday = yf.download(
        universe,
        start=start_all.strftime("%Y-%m-%d"),
        end=end_all.strftime("%Y-%m-%d"),
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    log("Download complete.", level="ok")
    return daily, intraday


def _get_close_series(daily: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(daily.columns, pd.MultiIndex):
        try:
            s = daily[("Close", ticker)].dropna()
        except KeyError:
            return pd.Series(dtype=float)
    else:
        s = daily["Close"].dropna()
    return s


def last_weekly_pivot_high(
    daily_df: pd.DataFrame,
    ticker: str,
    weeks: int = PIVOT_LOOKBACK_WEEKS,
    upto_date: Optional[datetime] = None,
) -> float:
    """
    Compute a "10-week pivot high" for a given ticker, restricted to data
    up to (and including) upto_date if provided.
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"].dropna()

    if upto_date is not None:
        upto_ts = pd.Timestamp(upto_date)
        highs = highs.loc[highs.index <= upto_ts]

    bars = weeks * 5  # ~5 trading days per week
    highs = highs.tail(bars)
    return float(highs.max()) if len(highs) else np.nan


def compute_sma_series(daily_df: pd.DataFrame, ticker: str, window: int) -> pd.Series:
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            s = daily_df[("Close", ticker)].dropna()
        except KeyError:
            return pd.Series(dtype=float)
    else:
        s = daily_df["Close"].dropna()
    return s.rolling(window).mean()


# ========= Regime classifier (simple Weinstein-style) =========

def classify_regime_for_benchmark(daily: pd.DataFrame, benchmark: str) -> pd.Series:
    """
    Classify each day as BULL / NEUTRAL / BEAR for the benchmark.

    BULL:    close > SMA150 and SMA150 slope (30 days) > 0
    BEAR:    close < SMA150 and SMA150 slope (30 days) < 0
    NEUTRAL: everything else
    """
    close = _get_close_series(daily, benchmark)
    if close.empty:
        raise ValueError(f"No daily close data for benchmark {benchmark}.")

    sma = close.rolling(SMA_DAYS).mean()
    slope = sma.diff(30)  # 30-day slope

    labels = []
    for dt, px in close.items():
        ma = sma.loc[dt]
        sl = slope.loc[dt]
        if pd.isna(ma) or pd.isna(sl):
            labels.append("NEUTRAL")
            continue
        if px > ma and sl > 0:
            labels.append("BULL")
        elif px < ma and sl < 0:
            labels.append("BEAR")
        else:
            labels.append("NEUTRAL")

    regime = pd.Series(labels, index=close.index, name="regime")
    return regime


def regime_flags_for_date(
    regime_series: pd.Series,
    d: date,
    mode: str,
) -> Tuple[bool, bool, str]:
    """
    Given a regime series indexed by date, return (long_ok, short_ok, label)
    for a specific calendar date and simulation mode.
    """
    ts = pd.Timestamp(d)
    subset = regime_series.loc[regime_series.index <= ts]
    if subset.empty:
        label = "NEUTRAL"
    else:
        label = str(subset.iloc[-1])

    label_upper = label.upper()

    if mode == "long_only":
        return True, False, label_upper
    if mode == "short_only":
        return False, True, label_upper

    # mode == "regime": toggle long/short based on label
    if label_upper == "BULL":
        return True, False, label_upper
    if label_upper == "BEAR":
        return False, True, label_upper
    # NEUTRAL
    return True, True, label_upper


# ========= Simple position model =========

@dataclass
class Position:
    ticker: str
    direction: str  # "long" or "short"
    entry_ts: pd.Timestamp
    entry_price: float
    qty: float
    stop_price: float
    tp_price: float


@dataclass
class Trade:
    ticker: str
    direction: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl_dollar: float
    pnl_pct: float
    regime_at_entry: str
    regime_at_exit: str


# ========= Simulation core =========

def simulate_year(sim_cfg: SimConfig, config_path: str) -> None:
    cfg, benchmark, account_size, risk_pct = load_config(config_path)
    # Override benchmark / account from CLI-configured SimConfig
    benchmark = sim_cfg.benchmark or benchmark
    account_size = sim_cfg.account_size or account_size
    risk_pct = sim_cfg.risk_per_trade_pct or risk_pct

    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Using weekly CSV: {weekly_csv_path}", level="info")

    universe = build_universe(weekly_df, benchmark)
    log(
        f"Focus universe: {len(universe)-1} symbols (Stage 1/2) + benchmark {benchmark}",
        level="info",
    )

    daily, intraday = download_data(universe, sim_cfg.year)

    # Regime series based on benchmark
    regime_series = classify_regime_for_benchmark(daily, benchmark)
    log("Computed Chapter 8-like regime time series (BULL / NEUTRAL / BEAR).", level="ok")

    # Restrict intraday bars to given calendar year
    bar_index = intraday.index
    start_year = datetime(sim_cfg.year, 1, 1)
    end_year = datetime(sim_cfg.year, 12, 31, 23, 59)
    mask_year = (bar_index >= start_year) & (bar_index <= end_year)
    bar_index = bar_index[mask_year]

    if len(bar_index) == 0:
        raise ValueError(f"No intraday bars found for year {sim_cfg.year}.")

    log(f"Intraday bars in {sim_cfg.year}: {len(bar_index)}", level="info")

    equity = float(account_size)
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []

    log(
        f"Initial account: ${equity:,.2f}, risk per trade: {risk_pct*100:.2f}% (${equity*risk_pct:,.2f})",
        level="info",
    )

    # Precompute SMA series for each ticker for speed
    sma_cache: Dict[str, pd.Series] = {}
    for t in universe:
        sma_cache[t] = compute_sma_series(daily, t, SMA_DAYS)

    n_bars = len(bar_index)
    # Milestones at each 10% of bars
    milestones = {max(1, int(n_bars * frac / 10)) for frac in range(1, 10)}

    for i, ts_bar in enumerate(bar_index, start=1):
        bar_date = ts_bar.date()
        long_ok, short_ok, regime_label = regime_flags_for_date(regime_series, bar_date, sim_cfg.mode)

        # Current bar's row (all tickers)
        row = intraday.loc[ts_bar]

        # --- 1) Check exits for existing positions ---
        to_close: List[str] = []
        for key, pos in positions.items():
            t = pos.ticker
            if ("Close", t) not in row.index:
                continue
            px = float(row[("Close", t)])

            hit_stop = (px <= pos.stop_price) if pos.direction == "long" else (px >= pos.stop_price)
            hit_tp = (px >= pos.tp_price) if pos.direction == "long" else (px <= pos.tp_price)

            if hit_stop or hit_tp:
                pnl = (px - pos.entry_price) * pos.qty if pos.direction == "long" else (pos.entry_price - px) * pos.qty
                pnl_pct = pnl / (pos.entry_price * pos.qty) * 100.0 if pos.entry_price * pos.qty != 0 else 0.0
                equity += pnl

                _, _, reg_entry = regime_flags_for_date(regime_series, pos.entry_ts.date(), sim_cfg.mode)
                _, _, reg_exit = regime_flags_for_date(regime_series, bar_date, sim_cfg.mode)

                trades.append(
                    Trade(
                        ticker=t,
                        direction=pos.direction,
                        entry_ts=pos.entry_ts,
                        exit_ts=ts_bar,
                        entry_price=pos.entry_price,
                        exit_price=px,
                        qty=pos.qty,
                        pnl_dollar=pnl,
                        pnl_pct=pnl_pct,
                        regime_at_entry=reg_entry,
                        regime_at_exit=reg_exit,
                    )
                )
                to_close.append(key)

        for key in to_close:
            del positions[key]

        # --- 2) Check entries (very simplified breakout/MA logic) ---
        risk_dollar = equity * risk_pct

        for t in universe:
            if t == benchmark:
                continue

            key_long = f"{t}_long"
            key_short = f"{t}_short"

            # skip if we already have position in that direction
            if key_long in positions or key_short in positions:
                continue

            if ("Close", t) not in row.index:
                continue
            px = float(row[("Close", t)])
            if math.isnan(px) or px <= 0:
                continue

            # Daily info up to bar date
            if isinstance(daily.columns, pd.MultiIndex):
                try:
                    dsub = daily.xs(t, axis=1, level=1).dropna()
                except KeyError:
                    continue
            else:
                dsub = daily.copy()

            dsub = dsub.loc[dsub.index <= pd.Timestamp(bar_date)]
            if dsub.empty:
                continue

            pivot = last_weekly_pivot_high(daily, t, weeks=PIVOT_LOOKBACK_WEEKS, upto_date=bar_date)
            sma = sma_cache[t].loc[sma_cache[t].index <= pd.Timestamp(bar_date)]
            if sma.empty:
                continue
            ma150 = float(sma.iloc[-1])

            # LONG entry
            if long_ok and not math.isnan(pivot) and px >= pivot and px >= ma150:
                # Hard stop below entry
                stop = px * (1.0 - HARD_STOP_PCT)
                # Take profit at 2R
                r = px - stop
                tp = px + TP_R_MULT * r
                if r <= 0:
                    continue
                qty = max(0, int(risk_dollar / r))
                if qty <= 0:
                    continue

                positions[key_long] = Position(
                    ticker=t,
                    direction="long",
                    entry_ts=ts_bar,
                    entry_price=px,
                    qty=qty,
                    stop_price=stop,
                    tp_price=tp,
                )

            # SHORT entry
            if short_ok and px <= ma150 * (1.0 - 0.01):  # 1% under MA150
                stop = px * (1.0 + HARD_STOP_PCT)
                r = stop - px
                if r <= 0:
                    continue
                tp = px - TP_R_MULT * r
                qty = max(0, int(risk_dollar / r))
                if qty <= 0:
                    continue

                positions[key_short] = Position(
                    ticker=t,
                    direction="short",
                    entry_ts=ts_bar,
                    entry_price=px,
                    qty=qty,
                    stop_price=stop,
                    tp_price=tp,
                )

        # --- Progress logging ---
        if i in milestones or i == n_bars:
            pct = i / n_bars * 100.0
            log(
                f"Simulation progress: {i}/{n_bars} bars "
                f"({pct:5.1f}%) — equity ${equity:,.2f}, open positions: {len(positions)}, "
                f"trades so far: {len(trades)}",
                level="info",
            )

    # ========= Simulation done: summary =========
    if trades:
        pnl_series = pd.Series([t.pnl_dollar for t in trades])
        wins = (pnl_series > 0).sum()
        losses = (pnl_series < 0).sum()
        win_rate = wins / len(trades) * 100.0
    else:
        wins = losses = 0
        win_rate = 0.0

    total_pnl = equity - account_size
    total_ret_pct = total_pnl / account_size * 100.0 if account_size != 0 else 0.0

    log("Simulation complete.", level="ok")
    log(
        f"Final equity: ${equity:,.2f} (P/L ${total_pnl:,.2f}, {total_ret_pct:.2f}%) — "
        f"trades: {len(trades)} (wins: {wins}, losses: {losses}, win-rate: {win_rate:.1f}%)",
        level="info",
    )

    # Save trades to CSV
    os.makedirs("./output", exist_ok=True)
    out_path = os.path.join(
        "./output",
        f"intraday_sim_{sim_cfg.year}_{sim_cfg.mode}.csv",
    )
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    trades_df.to_csv(out_path, index=False)
    log(f"Wrote trade log → {out_path}", level="ok")


# ========= CLI =========

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--year",
        type=int,
        required=True,
        help="Calendar year to simulate (e.g., 2025).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="regime",
        choices=["regime", "long_only", "short_only"],
        help="Simulation mode: 'regime' (toggle long/short by BULL/BEAR), "
             "'long_only', or 'short_only'.",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config.yaml (default: ./config.yaml).",
    )
    args = ap.parse_args()

    _, benchmark, account_size, risk_pct = load_config(args.config)
    sim_cfg = SimConfig(
        year=args.year,
        benchmark=benchmark,
        account_size=account_size,
        risk_per_trade_pct=risk_pct,
        mode=args.mode,
    )

    log(
        f"Starting simulation for year {sim_cfg.year} (mode={sim_cfg.mode}) using {args.config}",
        level="step",
    )
    try:
        simulate_year(sim_cfg, args.config)
    except Exception as e:
        log(f"Simulation error: {e}", level="err")
        raise


if __name__ == "__main__":
    main()
