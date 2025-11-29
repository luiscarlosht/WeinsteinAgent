#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Intraday Simulator — 2025 backtest (long-only, Chapter 8 aware)

Goal:
- Replay historical 60m + daily data for 2025
- Apply simplified versions of your intraday BUY + SELL logic
- Respect Chapter 8 regime filter (BULL / BEAR / NEUTRAL => long_ok / short_ok)
- Simulate trades with position sizing & stops
- Output trades CSV + equity curve CSV for analysis

Outputs:
  ./output/sim_trades_<year>_<mode>.csv
  ./output/sim_equity_<year>_<mode>.csv

Modes:
  regime    — use Chapter 8 regime filter (BULL: long only, BEAR: short only (not used yet), NEUTRAL: both)
  long_only — ignore regime, always allow longs
  flat      — no trades, just for sanity-checking data fetch & regime series
"""

import os
import math
import json
import argparse
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

# ---------------- Tunables (aligned with your intraday watcher) ----------------

WEEKLY_OUTPUT_DIR   = "./output"
WEEKLY_FILE_PREFIX  = "weinstein_weekly_"
BENCHMARK_DEFAULT   = "SPY"
SMA_DAYS            = 150          # 150-d ≈ 30-wk MA proxy
PIVOT_LOOKBACK_WEEKS = 10
HARD_STOP_PCT       = 0.08         # 8% hard stop
TRAIL_ATR_MULT      = 2.0          # ATR-based trailing stop
PIVOT_ENTRY_BUFFER_PCT = 0.002     # +0.20% over pivot
SELL_BREAK_PCT      = 0.005        # 0.5% crack below MA150
ACCOUNT_SIZE_DEFAULT = 100000.0
RISK_PER_TRADE_PCT   = 0.01        # 1% per trade

INTRADAY_INTERVAL   = "60m"
LOOKBACK_START_DAYS = 90          # safety window before year start, if using period

VERBOSE = True

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _log(msg, level="info"):
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
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{prefix} [{ts}] {msg}", flush=True)


def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan


def _is_crypto(sym: str) -> bool:
    return (sym or "").upper().endswith("-USD")


# ---------------------------------------------------------------------------
# Config / Weekly CSV
# ---------------------------------------------------------------------------

def load_config(path="./config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}

    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")

    return cfg, benchmark, sheet_url, svc_file


def newest_weekly_csv():
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


def load_weekly_report():
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path


# ---------------------------------------------------------------------------
# Regime (Chapter 8) time series, based on SPY daily
# ---------------------------------------------------------------------------

def compute_regime_series(benchmark_daily_close: pd.Series) -> pd.DataFrame:
    """
    Compute a daily regime label + long_ok/short_ok based on benchmark (e.g., SPY).

    Simple Weinstein-style logic:
      - BULL: price > MA150 and MA150 slope > 0  => long_ok=True,  short_ok=False
      - BEAR: price < MA150 and MA150 slope < 0  => long_ok=False, short_ok=True
      - NEUTRAL: everything else                 => long_ok=True,  short_ok=True
    """
    close = benchmark_daily_close.dropna()
    ma150 = close.rolling(SMA_DAYS).mean()
    # slope over ~1 month (~20 trading days)
    slope = ma150.diff(20)

    rows = []
    for dt in close.index:
        px = close.loc[dt]
        ma = ma150.loc[dt]
        sl = slope.loc[dt]

        if pd.isna(ma):
            regime = "NEUTRAL"
            long_ok, short_ok = True, True
        else:
            if px > ma and sl > 0:
                regime = "BULL"
                long_ok, short_ok = True, False
            elif px < ma and sl < 0:
                regime = "BEAR"
                long_ok, short_ok = False, True
            else:
                regime = "NEUTRAL"
                long_ok, short_ok = True, True

        rows.append({
            "date": dt.normalize(),
            "regime_label": regime,
            "long_ok": bool(long_ok),
            "short_ok": bool(short_ok),
        })

    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Data helpers (intraday + daily, ATR, pivots, MA150)
# ---------------------------------------------------------------------------

def download_data(tickers, start_date, end_date):
    """
    Download intraday (60m) and daily bars for tickers between start_date and end_date.
    """
    uniq = list(dict.fromkeys(tickers))
    _log(f"Downloading daily + intraday for {len(uniq)} tickers ({start_date} → {end_date})...", level="step")

    intraday = yf.download(
        uniq,
        start=start_date,
        end=end_date,
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )
    daily = yf.download(
        uniq,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    _log("Download complete.", level="ok")
    return intraday, daily


def _get_daily_sub(daily_df, ticker):
    """
    Safely get daily OHLCV sub-DataFrame for one ticker from a MultiIndex yfinance frame.
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            sub = daily_df.xs(ticker, axis=1, level=1)
        except KeyError:
            return None
    else:
        sub = daily_df
    return sub


def compute_atr(daily_df, ticker, n=14):
    sub = _get_daily_sub(daily_df, ticker)
    if sub is None:
        return np.nan
    cols = set(sub.columns)
    if not {"High", "Low", "Close"}.issubset(cols):
        return np.nan

    h, l, c = sub["High"], sub["Low"], sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    atr = atr.dropna()
    if atr.empty:
        return np.nan
    return float(atr.iloc[-1])


def last_weekly_pivot_high(daily_df, ticker, weeks=10, upto_date=None):
    """
    Compute a "10-week pivot high" for a given ticker, restricted to data
    up to (and including) upto_date if provided.

    - daily_df: full daily OHLCV DataFrame (MultiIndex columns or single-ticker)
    - ticker:   symbol string
    - weeks:    lookback window in weeks (~5 trading days each)
    - upto_date: either a pandas.Timestamp, datetime.date, or None
    """
    # Select the High series for this ticker
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)].dropna().copy()
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"].dropna().copy()

    if highs.empty:
        return np.nan

    # If we were given an upto_date (date or Timestamp), restrict the series
    if upto_date is not None:
        # Key fix: make sure we compare Timestamp to Timestamp
        cutoff = pd.Timestamp(upto_date)
        highs = highs.loc[highs.index <= cutoff]
        if highs.empty:
            return np.nan

    # Use ~10 weeks of history before upto_date (or end) as the pivot window
    bars = weeks * 5  # ~5 trading days per week
    highs = highs.tail(bars)

    return float(highs.max()) if len(highs) else np.nan


def ma150_at_date(daily_df, ticker, dt: date):
    """
    Compute MA150 (30-wk proxy) using daily closes up to dt.
    """
    sub = _get_daily_sub(daily_df, ticker)
    if sub is None or "Close" not in sub.columns:
        return np.nan

    c = sub["Close"].loc[sub.index.date <= dt]
    if len(c) < SMA_DAYS:
        return np.nan
    ma = c.rolling(SMA_DAYS).mean().dropna()
    if ma.empty:
        return np.nan
    return float(ma.iloc[-1])


def price_at_bar(intraday_df, ticker, ts):
    """
    Get close price of ticker at bar timestamp ts (ts is in intraday index).
    """
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            close = intraday_df[("Close", ticker)]
        except KeyError:
            return np.nan
    else:
        close = intraday_df["Close"]

    try:
        return float(close.loc[ts])
    except KeyError:
        # Sometimes missing bar for that ticker; try last available before ts
        subset = close.loc[close.index <= ts]
        if subset.empty:
            return np.nan
        return float(subset.iloc[-1])


# ---------------------------------------------------------------------------
# Order block / stops helpers (reused from your intraday style)
# ---------------------------------------------------------------------------

def propose_entry_and_stop_for_buy(price_now, pivot, ma150, atr):
    """
    Determine suggested entry & initial stop for a new long.
    - Entry: pivot * (1 + PIVOT_ENTRY_BUFFER_PCT) if pivot exists, else current price
    - Stop:  min(
                entry * (1 - HARD_STOP_PCT),
                entry - TRAIL_ATR_MULT * atr,
                0.97 * ma150 (if MA exists)
             )
    """
    entry = None
    if pd.notna(pivot):
        entry = pivot * (1.0 + PIVOT_ENTRY_BUFFER_PCT)
    elif pd.notna(price_now):
        entry = float(price_now)

    if entry is None:
        return np.nan, np.nan

    hard = entry * (1.0 - HARD_STOP_PCT)
    atr_trail = entry - (TRAIL_ATR_MULT * atr) if pd.notna(atr) else np.nan
    ma_guard = ma150 * 0.97 if pd.notna(ma150) else np.nan

    candidates = [v for v in (hard, atr_trail, ma_guard) if pd.notna(v)]
    if not candidates:
        return entry, np.nan
    stop = min(candidates)
    return float(entry), float(stop)


# ---------------------------------------------------------------------------
# Simple BUY / SELL conditions (simplified intraday logic)
# ---------------------------------------------------------------------------

def buy_condition(px, pivot, ma150, long_ok: bool):
    """
    Simplified BUY:
      - regime: long_ok must be True
      - price >= pivot * (1 + 0.4%)
      - price >= ma150 (>= 0% over MA)
    """
    if not long_ok:
        return False
    if pd.isna(px) or pd.isna(pivot) or pd.isna(ma150):
        return False
    if px < pivot * (1.0 + 0.004):
        return False
    if px < ma150:
        return False
    return True


def sell_condition_for_long(px, ma150, entry, atr):
    """
    Simplified SELL / risk exit for an existing long:
      - Hard stop: <= entry * (1 - HARD_STOP_PCT)
      - MA crack:  <= ma150 * (1 - SELL_BREAK_PCT)
      - ATR trail: <= entry - TRAIL_ATR_MULT * atr
    """
    if pd.isna(px):
        return False

    hard = entry * (1.0 - HARD_STOP_PCT) if entry is not None else np.nan
    ma_guard = ma150 * (1.0 - SELL_BREAK_PCT) if pd.notna(ma150) else np.nan
    atr_trail = entry - (TRAIL_ATR_MULT * atr) if pd.notna(atr) else np.nan

    reasons = []
    if pd.notna(hard) and px <= hard:
        reasons.append("hard stop")
    if pd.notna(ma_guard) and px <= ma_guard:
        reasons.append("MA150 crack")
    if pd.notna(atr_trail) and px <= atr_trail:
        reasons.append("ATR trail")

    return (len(reasons) > 0), ", ".join(reasons)


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def simulate_year(year: int = 2025, mode: str = "regime", config_path="./config.yaml"):
    """
    Run a long-only backtest for a given year.

    mode:
      - "regime": use Chapter 8 regime (long_ok / short_ok from SPY)
      - "long_only": always long_ok=True
      - "flat": no trades (for sanity check)
    """
    _log(f"Starting simulation for year {year} (mode={mode}) using {config_path}", level="step")

    cfg, benchmark, _, _ = load_config(config_path)
    weekly_df, weekly_path = load_weekly_report()
    _log(f"Using weekly CSV: {weekly_path}", level="info")

    # Build focus universe: Stage 1 and Stage 2 (same as intraday watcher)
    w = weekly_df.rename(columns=str.lower)
    for col in ["ticker", "stage", "ma30", "asset_class"]:
        if col not in w.columns:
            w[col] = np.nan
    focus = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])][
        ["ticker", "stage", "ma30", "asset_class"]
    ].copy()
    if "rank" in w.columns:
        focus["weekly_rank"] = w["rank"]
    else:
        focus["weekly_rank"] = 999999

    tickers = sorted(set(focus["ticker"].tolist() + [benchmark]))
    _log(f"Focus universe: {len(focus)} symbols (Stage 1/2) + benchmark {benchmark}", level="info")

    # Data range: a little before and after target year for safety
    start_date = f"{year-1}-11-01"
    end_date   = f"{year+1}-02-01"

    intraday, daily = download_data(tickers, start_date, end_date)

    # Build regime series using benchmark daily
    if isinstance(daily.columns, pd.MultiIndex):
        try:
            bench_close = daily[("Close", benchmark)]
        except KeyError:
            raise RuntimeError(f"Benchmark {benchmark} not found in daily data.")
    else:
        bench_close = daily["Close"]
    regime_df = compute_regime_series(bench_close)
    _log("Computed Chapter 8 regime time series.", level="ok")

    # Filter intraday bars to the target year
    intr_idx = intraday.index
    if not isinstance(intr_idx, pd.DatetimeIndex):
        raise RuntimeError("Intraday index is not a DatetimeIndex.")

    intraday_year = intraday[(intr_idx.year == year)]
    _log(f"Intraday bars in {year}: {len(intraday_year)}", level="info")

    # Simulation state
    account_size = float(cfg.get("app", {}).get("ordering", {}).get("account_size", ACCOUNT_SIZE_DEFAULT))
    risk_per_trade_pct = float(cfg.get("app", {}).get("ordering", {}).get("risk_per_trade_pct", RISK_PER_TRADE_PCT))
    risk_per_trade_dollar = account_size * risk_per_trade_pct

    cash = account_size
    positions = {}    # {symbol: {"qty": int, "entry": float, "stop": float}}
    trades = []
    equity_curve = []

    _log(f"Initial account: ${account_size:,.2f}, risk per trade: {risk_per_trade_pct*100:.2f}% (${risk_per_trade_dollar:,.2f})", level="info")

    # Main bar-by-bar loop
    for ts in intraday_year.index:
        bar_date = ts.date()

        # Determine regime for this date
        reg_row = regime_df.reindex([pd.Timestamp(bar_date)]).ffill().iloc[-1]
        regime_label = reg_row["regime_label"]
        long_ok = bool(reg_row["long_ok"])
        short_ok = bool(reg_row["short_ok"])

        # Mode overrides
        if mode == "long_only":
            long_ok = True
            short_ok = False
        elif mode == "flat":
            long_ok = False
            short_ok = False

        # Step 1: mark-to-market current positions
        mtm = 0.0
        for sym, pos in list(positions.items()):
            px_now = price_at_bar(intraday, sym, ts)
            if pd.isna(px_now):
                continue
            mtm += pos["qty"] * px_now

        equity = cash + mtm
        equity_curve.append({
            "timestamp": ts,
            "equity": equity,
            "cash": cash,
            "mtm": mtm,
            "regime": regime_label,
        })

        # If we are "flat" mode, skip trades entirely
        if not long_ok and not short_ok:
            continue

        # Step 2: evaluate SELL exits first (for existing longs)
        for sym in list(positions.keys()):
            pos = positions[sym]
            px_now = price_at_bar(intraday, sym, ts)
            if pd.isna(px_now):
                continue

            # Use daily data up to bar_date
            atr = compute_atr(daily.loc[daily.index.date <= bar_date], sym, n=14)
            ma150 = ma150_at_date(daily, sym, bar_date)

            should_exit, why = sell_condition_for_long(px_now, ma150, pos["entry"], atr)
            if should_exit:
                # Close position
                cash += pos["qty"] * px_now
                pnl = (px_now - pos["entry"]) * pos["qty"]
                trades.append({
                    "timestamp": ts,
                    "ticker": sym,
                    "side": "SELL",
                    "qty": pos["qty"],
                    "price": px_now,
                    "pnl": pnl,
                    "reason": why,
                    "regime": regime_label,
                })
                del positions[sym]

        # Step 3: evaluate new BUY entries (long side only)
        if long_ok:
            for _, row in focus.iterrows():
                sym = row["ticker"]
                if sym in positions:
                    continue  # already long

                px_now = price_at_bar(intraday, sym, ts)
                if pd.isna(px_now):
                    continue

                daily_upto = daily.loc[daily.index.date <= bar_date]
                atr = compute_atr(daily_upto, sym, n=14)
                ma150 = ma150_at_date(daily, sym, bar_date)
                pivot = last_weekly_pivot_high(daily_upto, sym, weeks=PIVOT_LOOKBACK_WEEKS, upto_date=bar_date)

                if not buy_condition(px_now, pivot, ma150, long_ok=True):
                    continue

                # Determine entry & stop using same style as intraday "Order Block"
                entry, stop = propose_entry_and_stop_for_buy(px_now, pivot, ma150, atr)
                if pd.isna(entry) or pd.isna(stop):
                    continue
                if entry <= stop:
                    continue

                risk_per_share = entry - stop
                if risk_per_share <= 0:
                    continue

                qty = int(risk_per_trade_dollar / risk_per_share)
                if qty <= 0:
                    continue

                # Ensure we have enough cash
                cost = qty * entry
                if cost > cash * 1.01:  # allow a small rounding overrun
                    continue

                cash -= cost
                positions[sym] = {
                    "qty": qty,
                    "entry": entry,
                    "stop": stop,
                }
                trades.append({
                    "timestamp": ts,
                    "ticker": sym,
                    "side": "BUY",
                    "qty": qty,
                    "price": entry,
                    "pnl": 0.0,
                    "reason": "signal",
                    "regime": regime_label,
                })

    # Final mark-to-market at the end of the year
    if intraday_year.index.size > 0:
        ts_last = intraday_year.index[-1]
        bar_date = ts_last.date()
        mtm = 0.0
        for sym, pos in positions.items():
            px_now = price_at_bar(intraday, sym, ts_last)
            if pd.isna(px_now):
                continue
            mtm += pos["qty"] * px_now
        equity = cash + mtm
    else:
        equity = cash

    _log(f"Simulation finished. Final equity: ${equity:,.2f}", level="ok")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs("./output", exist_ok=True)
    trades_path = os.path.join("./output", f"sim_trades_{year}_{mode}.csv")
    equity_path = os.path.join("./output", f"sim_equity_{year}_{mode}.csv")

    pd.DataFrame(trades).to_csv(trades_path, index=False)
    pd.DataFrame(equity_curve).to_csv(equity_path, index=False)

    _log(f"Wrote trades to {trades_path}", level="ok")
    _log(f"Wrote equity curve to {equity_path}", level="ok")

    # Quick stats
    if trades:
        df_tr = pd.DataFrame(trades)
        # Only SELLs have realized P&L
        sells = df_tr[df_tr["side"] == "SELL"]
        if not sells.empty:
            total_pnl = sells["pnl"].sum()
            wins = sells[sells["pnl"] > 0]
            losses = sells[sells["pnl"] <= 0]
            win_rate = len(wins) / max(1, len(sells)) * 100.0
            avg_win = wins["pnl"].mean() if not wins.empty else 0.0
            avg_loss = losses["pnl"].mean() if not losses.empty else 0.0

            print("\n=== Simulation Summary ===")
            print(f"Year: {year}, Mode: {mode}")
            print(f"Initial equity: ${account_size:,.2f}")
            print(f"Final equity:   ${equity:,.2f}")
            print(f"Total realized P&L: ${total_pnl:,.2f}")
            print(f"Number of closed trades: {len(sells)}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Avg win:  ${avg_win:,.2f}")
            print(f"Avg loss: ${avg_loss:,.2f}")
        else:
            print("\nNo closed trades (no SELLs).")
    else:
        print("\nNo trades generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Weinstein Intraday Simulator (long-only, Chapter 8 aware)")
    ap.add_argument("--year", type=int, default=2025, help="Year to simulate (default: 2025)")
    ap.add_argument("--mode", type=str, default="regime", choices=["regime", "long_only", "flat"],
                    help="Simulation mode: 'regime', 'long_only', or 'flat'")
    ap.add_argument("--config", type=str, default="./config.yaml", help="Path to config.yaml")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")

    args = ap.parse_args()
    global VERBOSE
    VERBOSE = not args.quiet

    simulate_year(year=args.year, mode=args.mode, config_path=args.config)


if __name__ == "__main__":
    main()
