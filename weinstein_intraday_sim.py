#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Weinstein Intraday Simulator with progress logging.

This script backtests a basic intraday breakout strategy over a full calendar
year, using a universe derived from your latest weekly report and a simple
Weinstein-style market regime classifier (BULL / NEUTRAL / BEAR).

Usage:

    python3 weinstein_intraday_sim.py --year 2025 --mode regime

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


# ========== Logging helpers ==========

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



# ========== Config / Weekly universe ==========

BENCHMARK_DEFAULT = "SPY"
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
INTRADAY_INTERVAL = "60m"
LOOKBACK_START_MONTH = 11  # start downloads in Nov previous year
LOOKAHEAD_END_MONTH = 2    # end downloads in Feb following year

PIVOT_LOOKBACK_WEEKS = 10
SMA_DAYS = 150
HARD_STOP_PCT = 0.08
TP_R_MULT = 2.0


@dataclass
class SimConfig:
    year: int
    benchmark: str
    account_size: float
    risk_per_trade_pct: float
    mode: str



# ========== Load config.yaml ==========

def load_config(path: str) -> Tuple[dict, str, float, float]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    app = cfg.get("app", {}) or {}
    ordering = app.get("ordering") or {}

    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    account_size = float(ordering.get("account_size", 5000.0))
    risk_pct = float(ordering.get("risk_per_trade_pct", 0.01))

    return cfg, benchmark, account_size, risk_pct



# ========== Weekly report loader ==========

def newest_weekly_csv() -> str:
    files = [
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError("No weekly CSV found in ./output")
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report() -> Tuple[pd.DataFrame, str]:
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path


def build_universe(weekly_df: pd.DataFrame, benchmark: str) -> List[str]:
    w = weekly_df.rename(columns=str.lower)
    if "ticker" not in w.columns or "stage" not in w.columns:
        raise ValueError("Weekly CSV missing 'ticker' or 'stage' columns.")

    focus = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])]
    tickers = sorted(set(focus["ticker"].dropna().str.upper().tolist()))

    if benchmark.upper() not in tickers:
        tickers.append(benchmark.upper())

    return tickers



# ========== Data download ==========

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



# ========== Helpers ==========

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
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"].dropna()

    if upto_date is not None:
        cutoff = pd.Timestamp(upto_date)
        highs = highs.loc[highs.index <= cutoff]

    highs = highs.tail(weeks * 5)
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



# ========== Regime classifier ==========

def classify_regime_for_benchmark(daily: pd.DataFrame, benchmark: str) -> pd.Series:
    close = _get_close_series(daily, benchmark)
    if close.empty:
        raise ValueError(f"No daily series for benchmark {benchmark}")

    sma = close.rolling(SMA_DAYS).mean()
    slope = sma.diff(30)

    labels = []
    for dt, px in close.items():
        ma = sma.loc[dt]
        sl = slope.loc[dt]
        if pd.isna(ma) or pd.isna(sl):
            labels.append("NEUTRAL")
        elif px > ma and sl > 0:
            labels.append("BULL")
        elif px < ma and sl < 0:
            labels.append("BEAR")
        else:
            labels.append("NEUTRAL")

    return pd.Series(labels, index=close.index, name="regime")


def regime_flags_for_date(
    regime_series: pd.Series, d: date, mode: str
) -> Tuple[bool, bool, str]:
    ts = pd.Timestamp(d)
    subset = regime_series.loc[regime_series.index <= ts]
    label = subset.iloc[-1] if not subset.empty else "NEUTRAL"
    label = label.upper()

    if mode == "long_only":
        return True, False, label
    if mode == "short_only":
        return False, True, label

    # mode=regime
    if label == "BULL":
        return True, False, label
    if label == "BEAR":
        return False, True, label
    return True, True, label



# ========== Position Models ==========

@dataclass
class Position:
    ticker: str
    direction: str
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



# ========== Main Simulation ==========

def simulate_year(sim_cfg: SimConfig, config_path: str) -> None:
    cfg, config_bench, config_acc, config_risk = load_config(config_path)

    benchmark = sim_cfg.benchmark or config_bench
    equity = float(sim_cfg.account_size or config_acc)
    risk_pct = float(sim_cfg.risk_per_trade_pct or config_risk)

    weekly_df, weekly_path = load_weekly_report()
    log(f"Using weekly CSV: {weekly_path}")

    universe = build_universe(weekly_df, benchmark)
    log(f"Focus universe: {len(universe)-1} Stage1/2 + benchmark {benchmark}")

    # Download
    daily, intraday = download_data(universe, sim_cfg.year)

    # Regime
    regime_series = classify_regime_for_benchmark(daily, benchmark)
    log("Computed regime series (BULL/NEUTRAL/BEAR)", level="ok")

    # Restrict intraday to year
    idx = intraday.index
    start = datetime(sim_cfg.year, 1, 1)
    end = datetime(sim_cfg.year, 12, 31, 23, 59)
    idx = idx[(idx >= start) & (idx <= end)]
    if len(idx) == 0:
        raise ValueError("No intraday bars for selected year.")

    log(f"Intraday bars: {len(idx)}")

    positions: Dict[str, Position] = {}
    trades: List[Trade] = []

    log(f"Initial account: ${equity:,.2f} (risk={risk_pct*100:.2f}%)")

    # Precompute daily SMA150
    sma_cache = {t: compute_sma_series(daily, t, SMA_DAYS) for t in universe}

    n_bars = len(idx)
    milestones = {max(1, int(n_bars * f / 10)) for f in range(1, 10)}

    for i, ts_bar in enumerate(idx, start=1):
        bar_date = ts_bar.date()
        row = intraday.loc[ts_bar]

        long_ok, short_ok, regime_label = regime_flags_for_date(regime_series, bar_date, sim_cfg.mode)

        # === Exit logic ===
        to_close = []
        for key, pos in positions.items():
            t = pos.ticker
            if ("Close", t) not in row:
                continue
            px = float(row[("Close", t)])
            if math.isnan(px):
                continue

            hit_stop = (px <= pos.stop_price) if pos.direction == "long" else (px >= pos.stop_price)
            hit_tp = (px >= pos.tp_price) if pos.direction == "long" else (px <= pos.tp_price)

            if hit_stop or hit_tp:
                pnl = (px - pos.entry_price) * pos.qty if pos.direction == "long" else (pos.entry_price - px) * pos.qty
                pnl_pct = pnl / (pos.entry_price * pos.qty) * 100.0 if pos.entry_price*pos.qty != 0 else 0.0
                equity += pnl

                _, _, reg_entry = regime_flags_for_date(regime_series, pos.entry_ts.date(), sim_cfg.mode)
                _, _, reg_exit = regime_flags_for_date(regime_series, bar_date, sim_cfg.mode)

                trades.append(Trade(
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
                ))
                to_close.append(key)

        for k in to_close:
            del positions[k]

        # === Entry logic ===
        risk_dollar = equity * risk_pct

        for t in universe:
            if t == benchmark:
                continue

            kl = f"{t}_long"
            ks = f"{t}_short"
            if kl in positions or ks in positions:
                continue

            if ("Close", t) not in row:
                continue
            px = float(row[("Close", t)])
            if math.isnan(px) or px <= 0:
                continue

            # Daily subset
            if isinstance(daily.columns, pd.MultiIndex):
                try:
                    ds = daily.xs(t, axis=1, level=1)
                except KeyError:
                    continue
            else:
                ds = daily.copy()
            ds = ds.loc[ds.index <= pd.Timestamp(bar_date)]
            if ds.empty:
                continue

            pivot = last_weekly_pivot_high(daily, t, upto_date=bar_date)
            sma_t = sma_cache[t].loc[sma_cache[t].index <= pd.Timestamp(bar_date)]
            if sma_t.empty:
                continue
            ma150 = float(sma_t.iloc[-1])

            # ---- Long entry ----
            if long_ok and not math.isnan(pivot) and px >= pivot and px >= ma150:
                stop = px * (1.0 - HARD_STOP_PCT)
                r = px - stop
                if r <= 0:
                    continue
                tp = px + TP_R_MULT * r
                qty = max(0, int(risk_dollar / r))
                if qty > 0:
                    positions[kl] = Position(
                        ticker=t,
                        direction="long",
                        entry_ts=ts_bar,
                        entry_price=px,
                        qty=qty,
                        stop_price=stop,
                        tp_price=tp,
                    )

            # ---- Short entry ----
            if short_ok and px <= ma150 * 0.99:  # 1% under MA150
                stop = px * (1.0 + HARD_STOP_PCT)
                r = stop - px
                if r <= 0:
                    continue
                tp = px - TP_R_MULT * r
                qty = max(0, int(risk_dollar / r))
                if qty > 0:
                    positions[ks] = Position(
                        ticker=t,
                        direction="short",
                        entry_ts=ts_bar,
                        entry_price=px,
                        qty=qty,
                        stop_price=stop,
                        tp_price=tp,
                    )

        # === Progress ===
        if i in milestones or i == n_bars:
            pct = i / n_bars * 100.0
            log(
                f"Simulation progress: {i}/{n_bars} bars "
                f"({pct:5.1f}%) — equity ${equity:,.2f}, open positions {len(positions)}, trades {len(trades)}"
            )

    # ========= Simulation complete =========
    total_pnl = equity - sim_cfg.account_size
    total_ret_pct = (total_pnl / sim_cfg.account_size) * 100.0

    wins = len([t for t in trades if t.pnl_dollar > 0])
    losses = len([t for t in trades if t.pnl_dollar < 0])
    winrate = wins / max(1, len(trades)) * 100.0

    log("Simulation complete.", level="ok")
    log(
        f"Final equity: ${equity:,.2f} (P/L ${total_pnl:,.2f}, {total_ret_pct:.2f}%) — "
        f"Trades={len(trades)} | Wins={wins} | Losses={losses} | Win-rate={winrate:.1f}%"
    )

    # Save CSV
    os.makedirs("./output", exist_ok=True)
    out_path = f"./output/intraday_sim_{sim_cfg.year}_{sim_cfg.mode}.csv"
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out_path, index=False)
    log(f"Wrote trade log → {out_path}", level="ok")



# ========== CLI driver ==========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--mode", type=str, default="regime",
                    choices=["regime", "long_only", "short_only"])
    ap.add_argument("--config", type=str, default="./config.yaml")
    args = ap.parse_args()

    _, bench, acc, risk = load_config(args.config)

    cfg = SimConfig(
        year=args.year,
        benchmark=bench,
        account_size=acc,
        risk_per_trade_pct=risk,
        mode=args.mode,
    )

    log(
        f"Starting simulation for {cfg.year} (mode={cfg.mode}) using {args.config}",
        level="step",
    )
    simulate_year(cfg, args.config)



if __name__ == "__main__":
    main()
