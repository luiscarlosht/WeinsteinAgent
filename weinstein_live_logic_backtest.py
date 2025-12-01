#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Live-Logic Backtest
-----------------------------

Goal
----
Replay historical BUY/SELL signals from ./output/signals_log.csv and produce a
daily PnL time series that you can compare vs. your real trades using, e.g.:

  python3 weinstein_real_vs_sim_monthly.py \
    --sim-monthly ./output/live_logic_bt_monthly_YYYYMMDD_HHMMSS.csv \
    --real-trades ./data/weinstein_real_trades_2025YTD.csv \
    --initial-capital 10000 \
    --real-date-col Date \
    --real-pnl-col Realized_PnL \
    --output ./output/real_vs_sim_monthly_2025YTD.csv

What this implementation does
-----------------------------
* Reads ./output/signals_log.csv (exported from Google Sheets by
  export_signals_from_sheets.py). Expected columns (case-insensitive):

    ts, ticker, side, price, reason, near_hits, state_before, state_after

* Normalizes timestamps to **timezone-naive** UTC so we can safely compare
  against naive datetime objects from --start / --end.

* Filters signals to a given date range [--start, --end].

* Uses yfinance to fetch daily OHLC for all tickers in the range.

* Calls market_regime.inspect() ONCE at the start of the run:

    from market_regime import inspect as inspect_market_regime
    label, long_ok, short_ok = inspect_market_regime()

  and applies this as a **global gate** for the whole run:
    - If long_ok is False (i.e., BEAR regime) we do NOT open any new longs.
    - If long_ok is True (BULL / NEUTRAL / UNKNOWN), long entries are allowed.

  This is a convenience reuse of your Chapter 8 logic; it is NOT a full
  historical regime reconstruction.

* Sim logic (LONG side only, for now):
    - When we see a BUY signal for T:
        * If allow_new_longs == True AND we have fewer than max_long open longs:
            - Entry on the **next trading day** after the signal date,
              at that day's OPEN price.
            - Position size:
                qty = (current_equity * risk_per_trade) / entry_price

    - When we see a SELL signal for T:
        * If we have an open long in T:
            - Exit on the **next trading day** after the SELL signal date,
              at that day's OPEN price.
            - Realize PnL and free the slot.

* At the **end of the backtest window**, any still-open positions are
  **force-closed** at the last available daily price (prefers CLOSE, falls
  back to OPEN) on or before --end. This ensures you get realized PnL
  instead of "0 closed trades".

* We aggregate PnL by exit date and write:

    Date,Simulated_PnL

  to the CSV path passed via --save-trades.

Limitations / Notes
-------------------
* SHORT logic is not implemented yet. --mode and --max-short are accepted
  but only the LONG side is actually simulated.
* Market regime in this script is a **single snapshot at run time**:
  it does NOT reconstruct historical regimes per day.
* If there is no next trading day data for a signal (e.g. very recent):
  the signal is skipped.
* If there is no price data up to --end for a still-open position, it is
  left out of the force-close step (conservative).

You can refine this later (stops, take-profit, true historical regime
reconstruction, shorts, etc.).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# yfinance for daily bars
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# Weinstein Chapter 8 market regime helper
try:
    from market_regime import inspect as inspect_market_regime
except Exception:  # pragma: no cover
    inspect_market_regime = None


# ─────────────────────────────────────
# Data structures
# ─────────────────────────────────────

@dataclass
class Position:
    ticker: str
    entry_date: datetime
    entry_price: float
    qty: float


@dataclass
class ClosedTrade:
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    qty: float

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty


# ─────────────────────────────────────
# Helpers
# ─────────────────────────────────────

def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD into a naive datetime."""
    return datetime.strptime(s, "%Y-%m-%d")


def load_signals(csv_path: str) -> pd.DataFrame:
    """
    Load signals from CSV and normalize:
      - ts_dt: timezone-NAIVE datetime (UTC, but tz info stripped)
      - ticker: UPPERCASE, stripped
      - side: BUY/SELL only
      - drops option-style tickers starting with '-'
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"Signals CSV not found or empty: {csv_path}")

    df = pd.read_csv(csv_path)

    # Flexible column detection
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("ts") or cols.get("timestamp") or list(df.columns)[0]
    tkr_col = cols.get("ticker") or "ticker"
    side_col = cols.get("side") or "side"

    # Make timestamps tz-aware UTC, then drop tz to become naive
    dt = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df["ts_dt"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)

    df["ticker"] = df[tkr_col].astype(str).str.upper().str.strip()
    df["side"] = df[side_col].astype(str).str.upper().str.strip()

    # Keep only valid rows
    df = df[df["ts_dt"].notna() & df["ticker"].ne("") & df["side"].isin(["BUY", "SELL"])]

    # Drop option-style tickers (starting with "-")
    df = df[~df["ticker"].str.startswith("-")].copy()

    df.sort_values("ts_dt", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_daily_bars(
    tickers: List[str],
    start: datetime,
    end: datetime,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLC for each ticker (inclusive of [start, end]),
    returns dict[ticker] -> DataFrame with Date index (date-only).
    """
    if yf is None:
        raise RuntimeError("yfinance not available; install with `pip install yfinance`.")

    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return {}

    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=2)).strftime("%Y-%m-%d"),  # small cushion
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    out: Dict[str, pd.DataFrame] = {}

    # yfinance shape depends on number of tickers
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex (Ticker, Field)
        for t in tickers:
            if (t, "Open") in data.columns:
                df_t = data[(t,)].copy()
                df_t.columns = [c.lower() for c in df_t.columns]
                df_t.index = df_t.index.tz_localize(None)  # drop tz if any
                df_t["date"] = df_t.index.date
                df_t.set_index("date", inplace=True)
                out[t] = df_t
    else:
        # Single ticker case
        if not data.empty:
            t = tickers[0]
            df_t = data.copy()
            df_t.columns = [c.lower() for c in df_t.columns]
            df_t.index = df_t.index.tz_localize(None)
            df_t["date"] = df_t.index.date
            df_t.set_index("date", inplace=True)
            out[t] = df_t

    return out


def next_trading_day_open(
    bars: Dict[str, pd.DataFrame],
    ticker: str,
    signal_dt: datetime,
) -> Optional[Tuple[datetime, float]]:
    """Return (date, open_price) of the next trading day AFTER signal_dt."""
    df = bars.get(ticker)
    if df is None or df.empty:
        return None

    sig_date = signal_dt.date()
    # Find first trading date strictly after signal date
    candidates = [d for d in df.index if d > sig_date]
    if not candidates:
        return None
    d0 = min(candidates)
    row = df.loc[d0]
    op = float(row["open"]) if "open" in row.index else np.nan
    if np.isnan(op) or op <= 0:
        return None
    return datetime.combine(d0, datetime.min.time()), op


def force_close_at_end(
    bars: Dict[str, pd.DataFrame],
    open_positions: Dict[str, Position],
    end: datetime,
    quiet: bool = False,
) -> Tuple[List[ClosedTrade], float]:
    """
    Force-close any remaining open positions at the last available daily price
    on or before `end`. Prefers CLOSE, falls back to OPEN.
    Returns (list_of_closed_trades, extra_pnl).
    """
    closed: List[ClosedTrade] = []
    extra_pnl = 0.0
    end_date = end.date()

    for tkr, pos in list(open_positions.items()):
        df = bars.get(tkr)
        if df is None or df.empty:
            continue
        # any data up to end_date
        available_dates = [d for d in df.index if d <= end_date]
        if not available_dates:
            continue
        d_last = max(available_dates)
        row = df.loc[d_last]
        price = float(row.get("close", row.get("open", np.nan)))
        if np.isnan(price) or price <= 0:
            continue
        exit_dt = datetime.combine(d_last, datetime.min.time())
        trade = ClosedTrade(
            ticker=tkr,
            entry_date=pos.entry_date,
            exit_date=exit_dt,
            entry_price=pos.entry_price,
            exit_price=price,
            qty=pos.qty,
        )
        pnl = trade.pnl
        extra_pnl += pnl
        closed.append(trade)

        if not quiet:
            print(
                f"[FORCE CLOSE] {tkr} {pos.qty:.4f} @ {price:.4f} on {exit_dt.date()} "
                f"(entry {pos.entry_price:.4f} {pos.entry_date.date()}, pnl={pnl:.2f})"
            )

    return closed, extra_pnl


# ─────────────────────────────────────
# Backtest core (LONG side)
# ─────────────────────────────────────

def run_backtest_long_only(
    signals: pd.DataFrame,
    start: datetime,
    end: datetime,
    capital: float,
    risk_per_trade: float,
    max_long: int,
    quiet: bool = False,
    allow_new_longs: bool = True,
) -> Tuple[List[ClosedTrade], float]:
    """
    Very simple long-only book:
      - Open long on BUY, close on SELL (same ticker).
      - Entry/exit next trading day's OPEN.
      - Size = equity * risk_per_trade / entry_price.
      - If allow_new_longs is False (e.g. BEAR regime), no new longs are opened.
      - Any remaining open positions are force-closed at the end of the window.
    """
    # Filter to window (ts_dt is naive; start/end are naive → safe compare)
    sig = signals[(signals["ts_dt"] >= start) & (signals["ts_dt"] <= end)].copy()
    if sig.empty:
        if not quiet:
            print("No signals in the requested window; backtest empty.")
        return [], capital

    tickers = sorted(sig["ticker"].unique().tolist())

    # Fetch daily bars for all tickers (slightly expanded window)
    fetch_start = start - timedelta(days=5)
    fetch_end = end + timedelta(days=5)
    bars = fetch_daily_bars(tickers, fetch_start, fetch_end)

    equity = float(capital)
    open_positions: Dict[str, Position] = {}
    closed_trades: List[ClosedTrade] = []

    for _, row in sig.iterrows():
        tkr = row["ticker"]
        side = row["side"]
        ts_dt = row["ts_dt"]

        if side == "BUY":
            # Global market regime gate: skip new longs if not allowed
            if not allow_new_longs:
                continue

            if tkr in open_positions:
                # Already long; skip this signal
                continue
            if len(open_positions) >= max_long:
                # Slot limit reached
                continue

            nxt = next_trading_day_open(bars, tkr, ts_dt)
            if nxt is None:
                continue
            entry_date, entry_price = nxt
            if entry_date < start or entry_date > end:
                # Outside backtest window
                continue

            risk_amt = equity * risk_per_trade
            if entry_price <= 0 or risk_amt <= 0:
                continue
            qty = risk_amt / entry_price

            open_positions[tkr] = Position(
                ticker=tkr,
                entry_date=entry_date,
                entry_price=entry_price,
                qty=qty,
            )

            if not quiet:
                print(f"[OPEN] {tkr} {qty:.4f} @ {entry_price:.4f} on {entry_date.date()} (equity={equity:.2f})")

        elif side == "SELL":
            # Close any open long for this ticker
            pos = open_positions.get(tkr)
            if pos is None:
                continue

            nxt = next_trading_day_open(bars, tkr, ts_dt)
            if nxt is None:
                continue
            exit_date, exit_price = nxt
            if exit_date < start or exit_date > end:
                continue

            trade = ClosedTrade(
                ticker=tkr,
                entry_date=pos.entry_date,
                exit_date=exit_date,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                qty=pos.qty,
            )
            pnl = trade.pnl
            equity += pnl
            closed_trades.append(trade)
            del open_positions[tkr]

            if not quiet:
                print(
                    f"[CLOSE] {tkr} {pos.qty:.4f} @ {exit_price:.4f} on {exit_date.date()} "
                    f"(entry {pos.entry_price:.4f} {pos.entry_date.date()}, pnl={pnl:.2f}, equity={equity:.2f})"
                )

    # At the end of the window, force-close any remaining open positions
    if open_positions:
        fc_trades, extra_pnl = force_close_at_end(bars, open_positions, end, quiet=quiet)
        closed_trades.extend(fc_trades)
        equity += extra_pnl

    if not quiet:
        print(f"Completed backtest: {len(closed_trades)} closed trades, final equity={equity:.2f}")

    return closed_trades, equity


def trades_to_daily_pnl(trades: List[ClosedTrade]) -> pd.DataFrame:
    """Aggregate closed trades into daily PnL."""
    if not trades:
        return pd.DataFrame(columns=["Date", "Simulated_PnL"])

    rows = []
    for tr in trades:
        rows.append({
            "Date": tr.exit_date.date().strftime("%Y-%m-%d"),
            "Simulated_PnL": tr.pnl,
        })
    df = pd.DataFrame(rows)
    df = df.groupby("Date", as_index=False)["Simulated_PnL"].sum()
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────
# CLI
# ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Weinstein live-logic backtest (simplified; long-only + regime gate).")
    ap.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    ap.add_argument("--year", type=int, default=None, help="Alternative: backtest entire YEAR if start/end not provided")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01,
                    help="Fraction of equity allocated per new trade (0.01 = 1%)")
    ap.add_argument("--max-long", type=int, default=10)
    ap.add_argument("--max-short", type=int, default=10,
                    help="(accepted but not used yet)")
    ap.add_argument("--mode", choices=["long", "short", "both"], default="long",
                    help="Simulation mode (short/both not implemented yet)")
    ap.add_argument("--quiet", action="store_true", help="Less verbose")
    ap.add_argument(
        "--save-trades",
        type=str,
        default=None,
        help="If set, write daily PnL CSV here with columns: Date,Simulated_PnL",
    )
    args = ap.parse_args()

    if args.mode in ("short", "both"):
        print("⚠️ NOTE: current implementation simulates LONG side only; shorts are ignored for now.")

    # Determine date range
    if args.start and args.end:
        start = parse_date(args.start)
        end = parse_date(args.end)
    elif args.year:
        start = datetime(args.year, 1, 1)
        end = datetime(args.year, 12, 31)
    else:
        raise SystemExit("You must provide either --start/--end or --year.")

    signals_path = "./output/signals_log.csv"
    if not args.quiet:
        print(f"📥 Loading signals from {signals_path}…")
    sig = load_signals(signals_path)
    if not args.quiet:
        print(f"• Loaded {len(sig)} signals total.")

    # Market regime filter (global, single snapshot)
    allow_new_longs = True
    if inspect_market_regime is not None:
        try:
            regime_label, long_ok, short_ok = inspect_market_regime()
            allow_new_longs = bool(long_ok)
            if not args.quiet:
                print(f"📊 Market Regime: {regime_label}, long_ok={long_ok}, short_ok={short_ok}")
                if not allow_new_longs:
                    print("⛔ Regime gate: new LONG entries are disabled for this backtest run.")
        except Exception as e:
            if not args.quiet:
                print(f"⚠️ Could not compute market regime ({type(e).__name__}: {e}); proceeding with longs enabled.")
            allow_new_longs = True
    else:
        if not args.quiet:
            print("⚠️ market_regime.inspect() not available; proceeding with longs enabled.")

    trades, final_equity = run_backtest_long_only(
        signals=sig,
        start=start,
        end=end,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        max_long=args.max_long,
        quiet=args.quiet,
        allow_new_longs=allow_new_longs,
    )

    daily = trades_to_daily_pnl(trades)

    if args.save_trades:
        os.makedirs(os.path.dirname(args.save_trades), exist_ok=True)
        daily.to_csv(args.save_trades, index=False)
        if not args.quiet:
            print(f"📝 Saved simulated daily PnL to: {args.save_trades}")

    if not args.quiet:
        print("📊 Summary:")
        print(f"  • Closed trades: {len(trades)}")
        print(f"  • Final equity:  {final_equity:.2f}")
        print(f"  • Daily rows:    {len(daily)}")


if __name__ == "__main__":
    main()
