#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_live_logic_backtest.py

Replays historical intraday data through your *actual* production logic.

What it does
------------
- Loads config.yaml to get account sizing / risk_per_trade_pct.
- Loads your latest weekly CSV to build the universe (Stage 1/2 + benchmark).
- Downloads daily + intraday bars from Yahoo for:
    [year-1 Nov 1]  →  [year+1 Feb 1]
- For each intraday bar in the target year:
    - Builds a price snapshot {ticker: last_price}.
    - Calls your real `compute_signals_for_snapshot(...)`.
    - Applies BUY / SELL signals to a simulated portfolio
      (cash + positions, R-based sizing using your config).
- Writes out:
    - ./output/live_logic_bt_<year>.csv  (per-trade log)
    - ./output/live_logic_bt_<year>_equity.png  (equity curve)

Usage
-----
  python3 weinstein_live_logic_backtest.py --year 2025 --config ./config.yaml
"""

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

# ---- IMPORT YOUR REAL STRATEGY HERE ----
# Adjust this import to wherever you put compute_signals_for_snapshot + LiveSignal
# from weinstein_intraday import compute_signals_for_snapshot, LiveSignal

# For now I'll re-declare a compatible LiveSignal so the file is self-contained.
# In your repo, delete this and import from your real module.
@dataclass
class LiveSignal:
    ticker: str
    action: str       # "BUY" or "SELL"
    side: str         # "long" or "short"
    price: float      # signal reference price
    stop_price: float # desired stop (if your logic has it)
    reason: str = ""


# ========== Small logging helpers ==========

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"• [{_ts()}] {msg}", flush=True)


# ========== Global tunables (same as your sim) ==========

BENCHMARK_DEFAULT = "SPY"
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
INTRADAY_INTERVAL = "60m"

LOOKBACK_START_MONTH = 11
LOOKAHEAD_END_MONTH = 2
SMA_DAYS = 150


# ========== Config + weekly universe ==========

def load_config(path: str) -> Tuple[dict, str, float, float]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    app = cfg.get("app", {}) or {}
    ordering = app.get("ordering") or {}

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
            f"Run weinstein_report_weekly.py first."
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
        raise ValueError("Weekly CSV missing 'ticker' or 'stage' columns.")

    focus = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])]
    tickers = sorted(set(focus["ticker"].dropna().str.upper().tolist()))

    bmk = benchmark.upper()
    if bmk not in tickers:
        tickers.append(bmk)

    return tickers


# ========== Data download ==========

def download_data(universe: List[str], year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_all = datetime(year - 1, LOOKBACK_START_MONTH, 1)
    end_all = datetime(year + 1, LOOKAHEAD_END_MONTH, 1)

    log(
        f"Downloading daily + intraday for {len(universe)} tickers "
        f"({start_all.date()} → {end_all.date()})..."
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

    log("Download complete.")
    return daily, intraday


# ========== Portfolio model for the sim ==========

@dataclass
class SimPosition:
    ticker: str
    side: str       # "long" or "short"
    entry_ts: pd.Timestamp
    entry_price: float
    qty: float
    stop_price: float


@dataclass
class SimTrade:
    ticker: str
    side: str
    action: str     # "OPEN" or "CLOSE"
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl_dollar: float
    pnl_pct: float
    reason: str


def apply_signals_to_portfolio(
    ts_bar: pd.Timestamp,
    prices: Dict[str, float],
    signals: List[LiveSignal],
    positions: Dict[str, SimPosition],
    equity: float,
    risk_pct: float,
) -> Tuple[float, List[SimTrade]]:
    """
    - Sells: close existing positions at current price.
    - Buys: open new positions sized by account risk_pct; if signal has stop_price,
      use that for position sizing; otherwise use a fixed 8% stop.
    """
    trades: List[SimTrade] = []
    HARD_STOP_PCT = 0.08

    # 1) Handle SELL first
    for sig in [s for s in signals if s.action.upper() == "SELL"]:
        key = f"{sig.ticker}_{sig.side.lower()}"
        pos = positions.get(key)
        px = prices.get(sig.ticker)
        if pos is None or px is None or px <= 0:
            continue

        if pos.side == "long":
            pnl = (px - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - px) * pos.qty

        pnl_pct = (
            pnl / (pos.entry_price * pos.qty) * 100.0
            if pos.entry_price * pos.qty != 0
            else 0.0
        )

        equity += pnl
        trades.append(
            SimTrade(
                ticker=pos.ticker,
                side=pos.side,
                action="CLOSE",
                entry_ts=pos.entry_ts,
                exit_ts=ts_bar,
                entry_price=pos.entry_price,
                exit_price=px,
                qty=pos.qty,
                pnl_dollar=pnl,
                pnl_pct=pnl_pct,
                reason=f"SELL: {sig.reason}",
            )
        )
        del positions[key]

    # 2) Handle BUY
    risk_dollar = equity * risk_pct

    for sig in [s for s in signals if s.action.upper() == "BUY"]:
        t = sig.ticker
        side = sig.side.lower()
        key = f"{t}_{side}"
        if key in positions:
            continue  # already in

        px = prices.get(t)
        if px is None or px <= 0:
            continue

        if sig.stop_price and sig.stop_price > 0:
            stop = sig.stop_price
        else:
            # fallback: simple % stop
            if side == "long":
                stop = px * (1.0 - HARD_STOP_PCT)
            else:
                stop = px * (1.0 + HARD_STOP_PCT)

        if side == "long":
            risk_per_share = px - stop
        else:
            risk_per_share = stop - px

        if risk_per_share <= 0:
            continue

        qty = max(0, int(risk_dollar / risk_per_share))
        if qty <= 0:
            continue

        positions[key] = SimPosition(
            ticker=t,
            side=side,
            entry_ts=ts_bar,
            entry_price=px,
            qty=qty,
            stop_price=stop,
        )

        trades.append(
            SimTrade(
                ticker=t,
                side=side,
                action="OPEN",
                entry_ts=ts_bar,
                exit_ts=ts_bar,
                entry_price=px,
                exit_price=px,
                qty=qty,
                pnl_dollar=0.0,
                pnl_pct=0.0,
                reason=f"BUY: {sig.reason}",
            )
        )

    return equity, trades


# ========== Stub that calls YOUR real logic ==========

def compute_signals_with_production_logic(
    ts_bar: pd.Timestamp,
    row: pd.Series,
    cfg: dict,
    holdings_snapshot: Dict[str, dict],
) -> List[LiveSignal]:
    """
    Build the same kind of inputs your live code uses, then delegate to
    `compute_signals_for_snapshot(...)` from your production module.

    - ts_bar: current intraday timestamp being simulated.
    - row: intraday row (multi-index columns: ('Close', 'AAPL'), etc.).
    - cfg: config.yaml dict.
    - holdings_snapshot: simulated holdings (you can map from positions dict).

    You MUST replace the body of this function with a call to your real logic.
    """

    # 1) Build a prices dict similar to what your live code uses
    prices: Dict[str, float] = {}
    if isinstance(row.index, pd.MultiIndex):
        close_cols = [c for c in row.index if c[0] == "Close"]
        for _, ticker in close_cols:
            px = float(row[("Close", ticker)])
            if px > 0:
                prices[ticker] = px
    else:
        # single-ticker case
        prices["TICKER"] = float(row["Close"])

    # 2) TODO: adapt holdings_snapshot to your internal structure if needed

    # 3) TODO: CALL YOUR REAL FUNCTION HERE, for example:
    #
    # from weinstein_intraday import compute_signals_for_snapshot
    # signals = compute_signals_for_snapshot(
    #     prices=prices,
    #     holdings=holdings_snapshot,
    #     cfg=cfg,
    #     as_of_ts=ts_bar,
    # )
    # return signals
    #
    # For now, return empty list so the script is runnable without wiring.
    return []


# ========== Main replay loop ==========

def run_backtest(year: int, config_path: str) -> None:
    cfg, bench, account_size, risk_pct = load_config(config_path)
    weekly_df, weekly_path = load_weekly_report()
    log(f"Using weekly CSV: {weekly_path}")
    universe = build_universe(weekly_df, bench)
    log(f"Focus universe: {len(universe)-1} Stage 1/2 + benchmark {bench}")

    daily, intraday = download_data(universe, year)

    # restrict intraday index to target year
    idx = intraday.index
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59)
    idx = idx[(idx >= start) & (idx <= end)]
    if len(idx) == 0:
        raise ValueError(f"No intraday bars for year {year}.")

    log(f"Intraday bars in {year}: {len(idx)}")
    equity = account_size
    positions: Dict[str, SimPosition] = {}
    trades: List[SimTrade] = []

    n_bars = len(idx)
    milestones = {max(1, int(n_bars * f / 10)) for f in range(1, 10)}

    for i, ts_bar in enumerate(idx, start=1):
        row = intraday.loc[ts_bar]

        # holdings_snapshot is what you pass back into your logic.
        # Simplest: just pass current positions in a dict keyed by ticker.
        holdings_snapshot: Dict[str, dict] = {}
        for key, pos in positions.items():
            holdings_snapshot[pos.ticker] = {
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "entry_ts": pos.entry_ts,
                "stop_price": pos.stop_price,
            }

        # 1) Get live-like signals from your real logic
        signals = compute_signals_with_production_logic(
            ts_bar=ts_bar,
            row=row,
            cfg=cfg,
            holdings_snapshot=holdings_snapshot,
        )

        # 2) Build price snapshot for the portfolio engine
        prices: Dict[str, float] = {}
        if isinstance(row.index, pd.MultiIndex):
            for (field, t) in row.index:
                if field == "Close":
                    px = float(row[(field, t)])
                    if px > 0:
                        prices[t] = px
        else:
            prices["TICKER"] = float(row["Close"])

        # 3) Apply signals to portfolio
        equity, new_trades = apply_signals_to_portfolio(
            ts_bar=ts_bar,
            prices=prices,
            signals=signals,
            positions=positions,
            equity=equity,
            risk_pct=risk_pct,
        )
        trades.extend(new_trades)

        # 4) Progress logging
        if i in milestones or i == n_bars:
            log(
                f"Progress {year}: {i}/{n_bars} bars "
                f"({i / n_bars * 100.0:5.1f}%) — "
                f"equity ${equity:,.2f}, open positions {len(positions)}, trades {len(trades)}"
            )

    # ===== Finish: summarize & save =====
    total_pnl = equity - account_size
    total_ret_pct = (total_pnl / account_size * 100.0) if account_size else 0.0

    log(f"Backtest complete for {year}.")
    log(
        f"Final equity: ${equity:,.2f} "
        f"(P/L ${total_pnl:,.2f}, {total_ret_pct:.2f}%) — "
        f"Trades={len(trades)}"
    )

    os.makedirs("./output", exist_ok=True)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    out_path = f"./output/live_logic_bt_{year}.csv"
    trades_df.to_csv(out_path, index=False)
    log(f"Wrote trade log → {out_path}")

    # optional equity curve
    if trades:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # reconstruct equity over time by replaying trades in exit order
            eq_dates: List[pd.Timestamp] = []
            eq_values: List[float] = []
            eq = account_size

            for tr in sorted(trades, key=lambda x: x.exit_ts):
                if tr.action == "CLOSE":
                    eq += tr.pnl_dollar
                    eq_dates.append(tr.exit_ts)
                    eq_values.append(eq)

            if eq_dates:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(eq_dates, eq_values)
                ax.set_title(f"Equity Curve {year} (live logic)")
                ax.set_ylabel("Equity ($)")
                ax.grid(alpha=0.3)
                fig.autofmt_xdate()
                fig.tight_layout()
                eq_path = f"./output/live_logic_bt_{year}_equity.png"
                fig.savefig(eq_path, dpi=120)
                plt.close(fig)
                log(f"Wrote equity curve PNG → {eq_path}")
        except Exception as e:
            log(f"Failed to plot equity curve: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, help="Calendar year to simulate")
    ap.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config.yaml",
    )
    args = ap.parse_args()
    run_backtest(args.year, args.config)


if __name__ == "__main__":
    main()
