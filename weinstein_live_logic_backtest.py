#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Live Logic Backtest — long + short using PRODUCTION intraday cores

Goal
----
Use the *same* intraday logic as your production watchers to simulate a
simple portfolio and build an equity curve.

- Long side: uses weinstein_intraday_core.eval_long_bar(...)
- Short side: uses weinstein_short_core.eval_short_bar(...)
- Universe: Stage 2 (uptrend) for longs, Stage 4 (downtrend) for shorts,
  from the newest weekly scan CSV under ./output.
- Data: yfinance 60m bars + daily bars.
- Stops/targets: use the same param mapping as your intraday watcher cores.

Notes / Simplifications
-----------------------
- Chapter 8 market regime: **NOT** dynamically simulated here. We treat
  long_ok = short_ok = True across the whole backtest.
  (Production watcher may skip shorts in BULL, etc.)
- Volume pace (pace_full) for backtest is simplified to:
    full_day_volume / 50dma_volume
  and applied uniformly across that day's intraday bars.
- Intrabar pace is approximated as:
    bar_volume / mean(bar_volume, last 20 bars)
- Position sizing:
    * risk_per_trade = capital * risk_per_trade_pct
    * shares = floor(risk_per_trade / per-share risk)
    * per-share risk = |stop - entry|
- Targets are tracked for diagnostics, but exits are stop-based only
  (no automatic profit-taking). You can extend this if you want.

CLI
---
Example:

  python3 weinstein_live_logic_backtest.py \
      --start 2024-01-01 --end 2024-10-31 \
      --capital 100000 \
      --risk-per-trade 0.01 \
      --max-long 10 --max-short 10 \
      --mode both

Outputs
-------
- ./output/live_logic_trades.csv      (all fills)
- ./output/live_logic_equity_curve.csv
- ./output/live_logic_equity_curve.png
"""

import os
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- LONG core (ASSUMED API — adjust if needed) -----
from weinstein_intraday_core import (
    eval_long_bar,
    LONG_HARD_STOP_PCT,
    LONG_TRAIL_ATR_MULT,
    LONG_MA_GUARD_PCT,
    LONG_TARGET1_PCT,
    LONG_TARGET2_PCT,
)

# ----- SHORT core (new) -----
from weinstein_short_core import (
    eval_short_bar,
    SHORT_HARD_STOP_PCT,
    SHORT_TRAIL_ATR_MULT,
    SHORT_MA_GUARD_PCT,
    SHORT_TARGET1_PCT,
    SHORT_TARGET2_PCT,
    PIVOT_LOOKBACK_WEEKS,
    INTRADAY_AVG_VOL_WINDOW,
)

WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_equities_"
BENCHMARK_DEFAULT = "SPY"

INTRADAY_INTERVAL = "60m"
DAILY_LOOKBACK_YEARS = 3  # enough to get MA150 & 50dma for most sims


# ========= helpers =========

def newest_weekly_csv() -> str:
    files = [
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found in {WEEKLY_OUTPUT_DIR}. "
            f"Run weinstein_report_weekly.py first."
        )
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def _fmt_ts(ts) -> str:
    if isinstance(ts, str):
        return ts
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _compute_ma150_and_pivots(daily_df: pd.DataFrame):
    """
    daily_df: must have columns ["Open","High","Low","Close","Volume"] and DatetimeIndex.
    Returns:
      ma150: pd.Series
      pivot_low: pd.Series (rolling N-bar min of Low, ~10 weeks)
      pivot_high: pd.Series (rolling N-bar max of High, ~10 weeks)
      atr14: pd.Series
      vol50: pd.Series
    """
    low = daily_df["Low"]
    high = daily_df["High"]
    close = daily_df["Close"]
    vol = daily_df["Volume"]

    ma150 = close.rolling(150).mean()

    # 10 weeks * 5 trading days
    win = PIVOT_LOOKBACK_WEEKS * 5
    pivot_low = low.rolling(win).min()
    pivot_high = high.rolling(win).max()

    prev_c = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    vol50 = vol.rolling(50).mean()

    return ma150, pivot_low, pivot_high, atr14, vol50


def _compute_intraday_vol_pace(intraday_vol: pd.Series) -> pd.Series:
    """
    Approx intrabar volume pace vs avg over INTRADAY_AVG_VOL_WINDOW bars.
    """
    vavg = intraday_vol.rolling(INTRADAY_AVG_VOL_WINDOW).mean()
    return intraday_vol / vavg


def _plot_equity_curve(eq_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(eq_df["timestamp"], eq_df["equity"])
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title("Weinstein Live Logic Backtest — Equity Curve")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ========= position / portfolio structures =========

@dataclass
class Position:
    ticker: str
    side: str        # "LONG" or "SHORT"
    entry_ts: pd.Timestamp
    entry_price: float
    shares: float
    stop: float
    target1: float
    target2: float
    max_favorable_px: float = field(default_factory=float)


@dataclass
class TradeFill:
    timestamp: pd.Timestamp
    ticker: str
    side: str
    action: str      # "OPEN" or "CLOSE"
    price: float
    shares: float
    pnl: float
    equity_after: float
    reason: str


# ========= main backtest engine =========

def run_backtest(
    start_date: str,
    end_date: str,
    capital: float,
    risk_per_trade_pct: float,
    max_long_positions: int,
    max_short_positions: int,
    mode: str = "both",
    max_long_universe: int = 40,
    max_short_universe: int = 40,
    benchmark: str = BENCHMARK_DEFAULT,
):

    mode = mode.lower()
    use_long = mode in ("long", "both")
    use_short = mode in ("short", "both")

    print(f"⭐ Backtest from {start_date} to {end_date}")
    print(f"   Capital: ${capital:,.2f} | Risk per trade: {risk_per_trade_pct*100:.1f}%")
    print(f"   Max positions: long={max_long_positions}, short={max_short_positions}")
    print(f"   Mode: {mode}")

    # ---- Load weekly universe ----
    weekly_path = newest_weekly_csv()
    print(f"   Weekly universe from: {weekly_path}")
    w = pd.read_csv(weekly_path)
    wl = w.rename(columns=str.lower)

    for col in ["ticker", "stage", "rank", "ma30", "rs_above_ma"]:
        if col not in wl.columns:
            wl[col] = np.nan

    wl["rank"] = pd.to_numeric(wl["rank"], errors="coerce").fillna(999999).astype(int)

    long_universe = []
    short_universe = []

    if use_long:
        # Stage 2 uptrend names
        cand = wl[wl["stage"].str.startswith("Stage 2")].copy()
        cand = cand.sort_values("rank").head(max_long_universe)
        long_universe = cand["ticker"].dropna().astype(str).str.upper().tolist()

    if use_short:
        cand = wl[wl["stage"] == "Stage 4 (Downtrend)"].copy()
        cand = cand.sort_values("rank").head(max_short_universe)
        short_universe = cand["ticker"].dropna().astype(str).str.upper().tolist()

    tickers = sorted(set(long_universe + short_universe + [benchmark]))
    print(f"   Long universe:  {len(long_universe)} symbols")
    print(f"   Short universe: {len(short_universe)} symbols")
    print(f"   Total (incl. benchmark): {len(tickers)} symbols")

    # ---- Download data ----
    intraday = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )
    daily = yf.download(
        tickers,
        period=f"{DAILY_LOOKBACK_YEARS}y",
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    if intraday.empty:
        raise RuntimeError("No intraday data downloaded; check dates / tickers.")

    if not isinstance(intraday.columns, pd.MultiIndex):
        raise RuntimeError("Expected MultiIndex columns from yfinance intraday (OHLCV, ticker).")

    if not isinstance(daily.columns, pd.MultiIndex):
        raise RuntimeError("Expected MultiIndex columns from yfinance daily (OHLCV, ticker).")

    all_times = intraday.index
    print(f"   Intraday bars: {len(all_times)} (interval={INTRADAY_INTERVAL})")

    # ---- Precompute per-symbol daily features ----
    daily_feats = {}  # ticker -> DataFrame of daily features

    for t in tickers:
        try:
            dsub = daily.xs(t, axis=1, level=1).dropna()
        except KeyError:
            continue
        if dsub.empty or not {"High", "Low", "Close", "Volume"}.issubset(dsub.columns):
            continue

        ma150, pivot_low, pivot_high, atr14, vol50 = _compute_ma150_and_pivots(dsub)
        feats = pd.DataFrame(
            {
                "Open": dsub["Open"],
                "High": dsub["High"],
                "Low": dsub["Low"],
                "Close": dsub["Close"],
                "Volume": dsub["Volume"],
                "MA150": ma150,
                "PivotLow": pivot_low,
                "PivotHigh": pivot_high,
                "ATR14": atr14,
                "Vol50": vol50,
            }
        )
        daily_feats[t] = feats

    # ---- Precompute intraday features (volume pace etc) ----
    intraday_feats = {}  # ticker -> DataFrame with Close, Volume, pace_intra

    for t in tickers:
        try:
            isub = intraday.xs(t, axis=1, level=1)[["Close", "Volume"]].dropna()
        except KeyError:
            continue
        if isub.empty:
            continue
        pace_intra = _compute_intraday_vol_pace(isub["Volume"])
        feats = isub.copy()
        feats["pace_intra"] = pace_intra
        intraday_feats[t] = feats

    # ---- Simulation state ----
    positions: Dict[str, Position] = {}
    equity_curve: List[Dict] = []
    trades: List[TradeFill] = []

    cash = capital

    # state dicts for long/short cores (per ticker)
    long_states: Dict[str, dict] = {}
    short_states: Dict[str, dict] = {}

    # convenience to know if ticker is allowed long/short
    is_long_candidate = {t: (t in long_universe) for t in tickers}
    is_short_candidate = {t: (t in short_universe) for t in tickers}

    # ---- helper for long & short stop/targets, mirroring cores ----
    def _long_entry_stop_targets(px, ma30, pivot_high, atr):
        if px is None or math.isnan(px):
            return math.nan, math.nan, math.nan, math.nan

        entry = float(px)
        hard = entry * (1.0 - LONG_HARD_STOP_PCT)
        atr_stop = (entry - LONG_TRAIL_ATR_MULT * atr) if not math.isnan(atr) else math.nan
        ma_guard = (ma30 * (1.0 - LONG_MA_GUARD_PCT)) if not math.isnan(ma30) else math.nan

        candidates = [c for c in (hard, atr_stop, ma_guard) if not math.isnan(c)]
        stop = min(candidates) if candidates else hard

        t1 = entry * (1.0 + LONG_TARGET1_PCT)
        t2 = entry * (1.0 + LONG_TARGET2_PCT)
        return entry, stop, t1, t2

    from weinstein_short_core import _short_entry_stop_targets  # reuse exact function

    # ---- main intraday loop ----
    for ts in all_times:
        ts_date = pd.Timestamp(ts).normalize()

        # 1) price marking + exits on stops
        total_mtm = 0.0
        for t, pos in list(positions.items()):
            # find current price for this ticker at this timestamp
            try:
                px = float(intraday_feats[t].loc[ts, "Close"])
            except Exception:
                # no bar for this ticker at this time; skip marking
                px = pos.entry_price

            if pos.side == "LONG":
                mtm = (px - pos.entry_price) * pos.shares
                hit_stop = px <= pos.stop
            else:  # SHORT
                mtm = (pos.entry_price - px) * pos.shares
                hit_stop = px >= pos.stop

            total_mtm += mtm

            if hit_stop:
                # close position at px
                pnl = mtm
                cash += pos.entry_price * pos.shares + pnl
                trades.append(
                    TradeFill(
                        timestamp=ts,
                        ticker=t,
                        side=pos.side,
                        action="CLOSE",
                        price=px,
                        shares=pos.shares,
                        pnl=pnl,
                        equity_after=cash + total_mtm - pnl,  # this bar pre-close eq
                        reason="STOP",
                    )
                )
                del positions[t]

        equity = cash + total_mtm
        equity_curve.append({"timestamp": ts, "equity": equity})

        # 2) signal evaluation & entries
        # NOTE: We only open new positions after updating equity for this bar.

        # current risk-per-trade in dollars
        risk_dollars = cash * risk_per_trade_pct if cash > 0 else 0.0

        # count open positions per side
        open_longs = sum(1 for p in positions.values() if p.side == "LONG")
        open_shorts = sum(1 for p in positions.values() if p.side == "SHORT")

        # skip new entries if we don't have cash or risk budget
        if risk_dollars <= 0:
            continue

        for t in tickers:
            # skip if no intraday bar
            if t not in intraday_feats:
                continue
            if ts not in intraday_feats[t].index:
                continue

            px = float(intraday_feats[t].loc[ts, "Close"])
            vol_bar = float(intraday_feats[t].loc[ts, "Volume"])
            pace_intra = float(intraday_feats[t].loc[ts, "pace_intra"])

            # map to daily features as of this date
            if t not in daily_feats:
                continue
            dfeats = daily_feats[t]
            if ts_date not in dfeats.index:
                # before symbol started trading or before enough history
                continue

            row = dfeats.loc[ts_date]
            ma30 = float(row["MA150"]) if not math.isnan(row["MA150"]) else math.nan
            pivot_low = float(row["PivotLow"]) if not math.isnan(row["PivotLow"]) else math.nan
            pivot_high = float(row["PivotHigh"]) if not math.isnan(row["PivotHigh"]) else math.nan
            atr = float(row["ATR14"]) if not math.isnan(row["ATR14"]) else math.nan
            vol50 = float(row["Vol50"]) if not math.isnan(row["Vol50"]) else math.nan

            # full-day volume vs 50dma (simplified)
            vol_today = float(row["Volume"])
            pace_full = (vol_today / vol50) if vol50 > 0 else math.nan

            # recent intraday closes (for non-60m cores; here 60m but we keep consistent)
            # we use last 2 closes up to this timestamp
            closes_tail = (
                intraday_feats[t].loc[:ts, "Close"].tail(2).tolist()
                if ts in intraday_feats[t].index
                else []
            )

            # LONG SIDE ------------------------------------------------------
            if use_long and is_long_candidate.get(t, False):
                ls = long_states.get(t, None)
                ls, flags_long = eval_long_bar(
                    price=px,
                    ma30=ma30,
                    pivot_high=pivot_high,
                    pace_full=pace_full,
                    pace_intra=pace_intra,
                    elapsed_min=60,       # backtest bar is full 60 min
                    closes_tail=closes_tail,
                    state=ls,
                    intraday_interval=INTRADAY_INTERVAL,
                    test_ease=False,
                )
                long_states[t] = ls

                long_trigger = flags_long.get("long_trigger_now", False)

                if long_trigger and t not in positions and open_longs < max_long_positions:
                    # compute entry/stop/targets from long core-mirroring helper
                    entry, stop, t1, t2 = _long_entry_stop_targets(px, ma30, pivot_high, atr)
                    if math.isnan(stop) or stop >= entry:
                        # invalid stop => skip
                        pass
                    else:
                        per_share_risk = entry - stop
                        if per_share_risk > 0:
                            shares = math.floor(risk_dollars / per_share_risk)
                        else:
                            shares = 0
                        if shares > 0:
                            # commit capital
                            cost = entry * shares
                            if cost <= cash:
                                cash -= cost
                                pos = Position(
                                    ticker=t,
                                    side="LONG",
                                    entry_ts=ts,
                                    entry_price=entry,
                                    shares=shares,
                                    stop=stop,
                                    target1=t1,
                                    target2=t2,
                                    max_favorable_px=entry,
                                )
                                positions[t] = pos
                                open_longs += 1
                                trades.append(
                                    TradeFill(
                                        timestamp=ts,
                                        ticker=t,
                                        side="LONG",
                                        action="OPEN",
                                        price=entry,
                                        shares=shares,
                                        pnl=0.0,
                                        equity_after=cash + total_mtm,
                                        reason="LONG_TRIGGER",
                                    )
                                )

            # SHORT SIDE -----------------------------------------------------
            if use_short and is_short_candidate.get(t, False):
                ss = short_states.get(t, None)
                ss, flags_short = eval_short_bar(
                    price=px,
                    ma30=ma30,
                    pivot_low=pivot_low,
                    pace_full=pace_full,
                    pace_intra=pace_intra,
                    elapsed_min=60,       # full bar
                    closes_tail=closes_tail,
                    state=ss,
                    intraday_interval=INTRADAY_INTERVAL,
                    test_ease=False,
                )
                short_states[t] = ss

                short_trigger = flags_short.get("short_trigger_now", False)

                if short_trigger and t not in positions and open_shorts < max_short_positions:
                    entry, stop, t1, t2 = _short_entry_stop_targets(px, ma30, pivot_low, atr)
                    if math.isnan(stop) or stop <= entry:
                        # invalid stop => skip
                        pass
                    else:
                        per_share_risk = stop - entry
                        if per_share_risk > 0:
                            shares = math.floor(risk_dollars / per_share_risk)
                        else:
                            shares = 0
                        if shares > 0:
                            # for shorts, we assume same capital usage (margin abstracted)
                            cost = entry * shares
                            if cost <= cash:
                                cash -= cost
                                pos = Position(
                                    ticker=t,
                                    side="SHORT",
                                    entry_ts=ts,
                                    entry_price=entry,
                                    shares=shares,
                                    stop=stop,
                                    target1=t1,
                                    target2=t2,
                                    max_favorable_px=entry,
                                )
                                positions[t] = pos
                                open_shorts += 1
                                trades.append(
                                    TradeFill(
                                        timestamp=ts,
                                        ticker=t,
                                        side="SHORT",
                                        action="OPEN",
                                        price=entry,
                                        shares=shares,
                                        pnl=0.0,
                                        equity_after=cash + total_mtm,
                                        reason="SHORT_TRIGGER",
                                    )
                                )

    # ---- finalize outputs ----
    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame([
        {
            "timestamp": _fmt_ts(t.timestamp),
            "ticker": t.ticker,
            "side": t.side,
            "action": t.action,
            "price": t.price,
            "shares": t.shares,
            "pnl": t.pnl,
            "equity_after": t.equity_after,
            "reason": t.reason,
        }
        for t in trades
    ])

    os.makedirs("./output", exist_ok=True)
    eq_path = "./output/live_logic_equity_curve.csv"
    trades_path = "./output/live_logic_trades.csv"
    png_path = "./output/live_logic_equity_curve.png"

    eq_df.to_csv(eq_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    _plot_equity_curve(eq_df, png_path)

    print(f"\n✅ Backtest complete.")
    if not eq_df.empty:
        start_eq = float(eq_df["equity"].iloc[0])
        end_eq = float(eq_df["equity"].iloc[-1])
        ret_pct = (end_eq / start_eq - 1.0) * 100.0
        print(f"   Start equity: ${start_eq:,.2f}")
        print(f"   End equity:   ${end_eq:,.2f}")
        print(f"   Total return: {ret_pct:.2f}%")
    print(f"   Equity curve CSV:  {eq_path}")
    print(f"   Trades CSV:        {trades_path}")
    print(f"   Equity curve PNG:  {png_path}")


# ========= CLI =========

def main():
    ap = argparse.ArgumentParser(
        description="Backtest Weinstein intraday live logic (long + short)."
    )
    ap.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    ap.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.01,
        help="Fraction of current cash risked per trade (e.g. 0.01 = 1%%)",
    )
    ap.add_argument(
        "--max-long",
        type=int,
        default=10,
        help="Max simultaneous long positions",
    )
    ap.add_argument(
        "--max-short",
        type=int,
        default=10,
        help="Max simultaneous short positions",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["long", "short", "both"],
        help="Run long-only, short-only, or both.",
    )
    ap.add_argument(
        "--max-long-universe",
        type=int,
        default=40,
        help="Max Stage 2 names from weekly scan.",
    )
    ap.add_argument(
        "--max-short-universe",
        type=int,
        default=40,
        help="Max Stage 4 names from weekly scan.",
    )
    ap.add_argument(
        "--benchmark",
        type=str,
        default=BENCHMARK_DEFAULT,
        help="Benchmark ticker (included for RS, not traded).",
    )

    args = ap.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        capital=args.capital,
        risk_per_trade_pct=args.risk_per_trade,
        max_long_positions=args.max_long,
        max_short_positions=args.max_short,
        mode=args.mode,
        max_long_universe=args.max_long_universe,
        max_short_universe=args.max_short_universe,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
