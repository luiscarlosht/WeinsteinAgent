#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_live_logic_backtest_yfinance.py

SIM / backtest runner that mirrors the live intraday + core logic as closely
as possible, using daily bars from yfinance.

- Universe comes from your weekly CSV snapshot(s).
- Applies:
    * Chapter 8 + VIX regime filter
    * Coppock curve filter (benchmark)
    * Breadth gate (% of breadth universe above MA50)
    * ADX14 gate (configurable via config.yaml)
- Uses LongCoreParams for breakout / volume / stop / trail tunables.

It prints:
- Config summary
- Breadth / ADX skip logs
- Final equity, P/L, trade count
- Equity curve PNG
- Trade log CSV
- Monthly P/L CSV
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from weinstein_long_core import LongCoreParams, is_breakout, passes_volume_filter, initial_stop, update_trailing_stop, stop_hit
import market_regime  # your existing module


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="· [%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_latest_weekly_csv(output_dir: Path) -> Path:
    """
    Finds the most recent weekly CSV in ./output named like:
        weinstein_weekly_equities_YYYYMMDD_HHmm.csv
    """
    pattern = "weinstein_weekly_equities_*.csv"
    files = sorted(output_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No weekly CSV found in {output_dir} matching {pattern}")
    return files[-1]


# --------------------------------------------------------------------------------------
# Technical indicators (ADX, ATR, moving averages, Coppock)
# --------------------------------------------------------------------------------------


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1.0 / period, adjust=False).mean() / atr

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    adx.index = df.index
    return adx


def compute_coppock_curve(close: pd.Series, w1: int = 11, w2: int = 14, w_roc: int = 10) -> pd.Series:
    """
    Basic Coppock curve on MONTHLY closes:
    Coppock = WMA( ROC(w_roc) + ROC(2*w_roc), w1+w2 )
    We can simplify and just use two WMA's or ema to approximate.
    """
    # monthly resample; in your logs you saw a FutureWarning for "M"
    monthly_close = close.resample("M").last()
    roc1 = 100 * (monthly_close / monthly_close.shift(w_roc) - 1)
    roc2 = 100 * (monthly_close / monthly_close.shift(2 * w_roc) - 1)
    coppock_raw = roc1 + roc2
    coppock = coppock_raw.ewm(span=w1 + w2, adjust=False).mean()
    coppock.index = monthly_close.index
    return coppock


def compute_breadth_series(
    prices: Dict[str, pd.DataFrame],
    ma_window: int,
) -> pd.Series:
    """
    Given a dict of per-ticker daily OHLCV, compute the breadth series:
    % of breadth universe above MA(ma_window) per day.
    """
    tickers = list(prices.keys())
    all_dates = None
    above_matrix = []

    for tkr in tickers:
        df = prices[tkr]
        close = df["Close"]
        ma = close.rolling(ma_window).mean()
        above = (close > ma).astype(int)
        above_matrix.append(above)

        if all_dates is None:
            all_dates = close.index
        else:
            all_dates = all_dates.union(close.index)

    if not above_matrix:
        return pd.Series(dtype=float)

    above_df = pd.concat(above_matrix, axis=1).fillna(0)
    above_df.columns = tickers
    pct_above = (above_df.sum(axis=1) / len(tickers)) * 100.0
    return pct_above.sort_index()


# --------------------------------------------------------------------------------------
# Backtest Engine
# --------------------------------------------------------------------------------------


class Position:
    def __init__(self, side: str, entry_date: pd.Timestamp, entry_price: float, qty: float, stop: float):
        self.side = side  # "long" only in this script
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.qty = qty
        self.stop = stop
        self.exit_date: Optional[pd.Timestamp] = None
        self.exit_price: Optional[float] = None

    def is_open(self) -> bool:
        return self.exit_date is None


def run_backtest(
    df_map: Dict[str, pd.DataFrame],
    long_universe: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    capital: float,
    risk_per_trade: float,
    max_long: int,
    use_regime_long: bool,
    use_coppock_long: bool,
    breadth_enabled: bool,
    breadth_series: pd.Series,
    breadth_min_long: float,
    adx_min_long: float,
    long_params: LongCoreParams,
    benchmark_close: pd.Series,
) -> Tuple[pd.Series, List[dict]]:
    """
    Simple long-only backtest:
    - Equity is mark-to-market daily.
    - Applies breadth & ADX gates for *new* long entries.
    - For now, no shorts (mode="both" will still be long-only here).
    """
    dates = pd.date_range(start_date, end_date, freq="B")  # business days
    equity_curve = pd.Series(index=dates, dtype=float)
    equity = capital

    open_positions: Dict[str, Position] = {}
    trades: List[dict] = []

    # Precompute ATR & ADX per ticker
    atr_map: Dict[str, pd.Series] = {}
    adx_map: Dict[str, pd.Series] = {}
    vol_ratio_map: Dict[str, pd.Series] = {}

    for tkr in long_universe:
        df = df_map[tkr].loc[start_date - pd.Timedelta(days=60) : end_date].copy()
        df["ATR14"] = compute_atr(df, period=14)
        df["ADX14"] = compute_adx(df, period=14)
        df["Vol50"] = df["Volume"].rolling(50).mean()
        df["VolRatio"] = df["Volume"] / df["Vol50"]
        df.dropna(subset=["Close"], inplace=True)
        df_map[tkr] = df
        atr_map[tkr] = df["ATR14"]
        adx_map[tkr] = df["ADX14"]
        vol_ratio_map[tkr] = df["VolRatio"]

    # Precompute Coppock on benchmark
    coppock_series = compute_coppock_curve(benchmark_close) if use_coppock_long else None

    for date in dates:
        # mark positions
        daily_value = 0.0
        for tkr, pos in list(open_positions.items()):
            df = df_map[tkr]
            if date not in df.index:
                continue
            row = df.loc[date]
            daily_value += pos.qty * row["Close"]

        equity_curve[date] = equity + daily_value

        # Determine gating flags
        allow_new_longs = True

        # Regime gate (Chapter 8 + VIX) - using market_regime helper
        if use_regime_long:
            regime = market_regime.compute_market_regime(benchmark_close.loc[:date])
            allow_new_longs = regime.long_ok

        # Coppock gate
        if allow_new_longs and use_coppock_long and coppock_series is not None:
            # map daily date to last monthly Coppock value
            coppock_val = coppock_series[coppock_series.index <= date].iloc[-1] if not coppock_series.empty else np.nan
            # basic rule: need Coppock > 0 for longs
            if not (coppock_val > 0):
                allow_new_longs = False

        # Breadth gate
        if allow_new_longs and breadth_enabled:
            if date in breadth_series.index:
                breadth_pct = breadth_series.loc[date]
                if breadth_pct < breadth_min_long * 100.0:
                    logger.info(
                        f"·· [SKIP-BREADTH] No new LONGs on {date.date()} because "
                        f"breadth={breadth_pct:.2f}% < {breadth_min_long*100:.0f}%"
                    )
                    allow_new_longs = False

        # 1) Exit logic (check stops)
        for tkr, pos in list(open_positions.items()):
            df = df_map[tkr]
            if date not in df.index or not pos.is_open():
                continue

            row = df.loc[date]
            low = row["Low"]
            close_price = row["Close"]
            atr = atr_map[tkr].get(date, np.nan)

            # stop hit?
            if stop_hit(pos.stop, low):
                pos.exit_date = date
                pos.exit_price = pos.stop
                pnl = (pos.exit_price - pos.entry_price) * pos.qty
                equity += pnl
                trades.append(
                    {
                        "ticker": tkr,
                        "side": pos.side,
                        "entry_date": pos.entry_date.date(),
                        "entry_price": pos.entry_price,
                        "exit_date": pos.exit_date.date(),
                        "exit_price": pos.exit_price,
                        "qty": pos.qty,
                        "pnl": pnl,
                    }
                )
                logger.info(
                    f"·· EXIT {tkr} on {date.date()} at {pos.exit_price:.2f} "
                    f"(stop hit, PnL={pnl:.2f})"
                )
                del open_positions[tkr]
                continue

            # trailing stop update
            if not math.isnan(atr):
                new_stop = update_trailing_stop(pos.stop, close_price, atr, long_params)
                if new_stop > pos.stop:
                    pos.stop = new_stop

        # 2) Entry logic
        if allow_new_longs:
            for tkr in long_universe:
                if tkr in open_positions:
                    continue

                df = df_map[tkr]
                if date not in df.index:
                    continue

                row = df.loc[date]
                close_price = row["Close"]
                high = row["High"]
                low = row["Low"]
                volume_ratio = vol_ratio_map[tkr].get(date, np.nan)
                atr = atr_map[tkr].get(date, np.nan)
                adx14 = adx_map[tkr].get(date, np.nan)

                if np.isnan(close_price) or np.isnan(volume_ratio) or np.isnan(atr) or np.isnan(adx14):
                    continue

                # ADX filter
                if adx14 < adx_min_long:
                    logger.info(
                        f"·· [SKIP-ADX] {tkr} because ADX14={adx14:.1f} < {adx_min_long:.1f} on {date.date()}"
                    )
                    continue

                if not passes_volume_filter(volume_ratio, long_params):
                    continue

                # simple breakout rule: today's close vs 20-day high
                hist = df.loc[:date].tail(21)
                if len(hist) < 21:
                    continue
                pivot = hist["High"].iloc[:-1].max()  # yesterday's 20-day high

                if not is_breakout(close_price, pivot, long_params):
                    continue

                # position sizing
                risk_capital = equity * risk_per_trade
                stop_price = initial_stop(close_price, atr, long_params)
                per_share_risk = max(close_price - stop_price, 0.01)
                qty = math.floor(risk_capital / per_share_risk)
                if qty <= 0:
                    continue

                pos = Position(
                    side="long",
                    entry_date=date,
                    entry_price=close_price,
                    qty=qty,
                    stop=stop_price,
                )
                open_positions[tkr] = pos
                logger.info(
                    f"·· ENTER LONG {tkr} on {date.date()} "
                    f"at {close_price:.2f} (qty={qty}, stop={stop_price:.2f}, ADX14={adx14:.1f})"
                )

    # Close all positions at end_date for reporting
    for tkr, pos in open_positions.items():
        df = df_map[tkr]
        if end_date not in df.index:
            continue
        close_price = df.loc[end_date, "Close"]
        pos.exit_date = end_date
        pos.exit_price = close_price
        pnl = (pos.exit_price - pos.entry_price) * pos.qty
        equity += pnl
        trades.append(
            {
                "ticker": tkr,
                "side": pos.side,
                "entry_date": pos.entry_date.date(),
                "entry_price": pos.entry_price,
                "exit_date": pos.exit_date.date(),
                "exit_price": pos.exit_price,
                "qty": pos.qty,
                "pnl": pnl,
            }
        )

    equity_curve.iloc[-1] = equity
    return equity_curve, trades


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weinstein live-logic backtest (yfinance).")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--mode", choices=["long", "short", "both"], default="both")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--max-long", type=int, default=10)
    p.add_argument("--max-short", type=int, default=10)
    p.add_argument("--use-regime-long", action="store_true")
    p.add_argument("--use-regime-short", action="store_true")
    p.add_argument("--benchmark", default="SPY")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    now_str = dt.datetime.now().strftime("%H:%M:%S")

    cfg = load_config(args.config)
    bt_cfg = cfg.get("backtest", {})
    app_cfg = cfg.get("app", {})
    benchmark = args.benchmark or app_cfg.get("benchmark", "SPY")

    # snapshot_mode
    snapshot_mode = bt_cfg.get("snapshot_mode", "static")

    # regime
    regime_cfg = bt_cfg.get("regime", {})
    use_regime_long = regime_cfg.get("use_long", bool(args.use_regime_long))
    use_regime_short = regime_cfg.get("use_short", bool(args.use_regime_short))

    # coppock
    coppock_cfg = bt_cfg.get("coppock", {})
    use_coppock_long = coppock_cfg.get("use_long", True)
    use_coppock_short = coppock_cfg.get("use_short", True)

    # breadth
    breadth_cfg = bt_cfg.get("breadth", {})
    breadth_enabled = bool(breadth_cfg.get("enabled", True))
    breadth_ma = int(breadth_cfg.get("ma_window", 50))
    breadth_min_long = float(breadth_cfg.get("min_long", 0.60))

    # indicators (ADX)
    shared_ind = cfg.get("indicators", {})
    bt_ind = bt_cfg.get("indicators", {})

    adx_min_long = float(bt_ind.get("adx_min_long", shared_ind.get("adx_min_long", 18.0)))
    adx_min_short = float(bt_ind.get("adx_min_short", shared_ind.get("adx_min_short", 18.0)))

    # core long / short params
    long_cfg = bt_cfg.get("long", {})
    short_cfg = bt_cfg.get("short", {})

    long_params = LongCoreParams(
        break_pct=float(long_cfg.get("break_pct", 0.004)),
        vol_min=float(long_cfg.get("vol_min", 1.30)),
        stop_hard=float(long_cfg.get("stop_hard", 0.20)),
        trail_atr=float(long_cfg.get("trail_atr", 2.0)),
        ma_guard=float(long_cfg.get("ma_guard", 0.03)),
    )

    # for now, short_params are not used (this script is long-only),
    # but we keep them for config completeness
    short_params = LongCoreParams(
        break_pct=float(short_cfg.get("break_pct", 0.004)),
        vol_min=float(short_cfg.get("vol_min", 1.30)),
        stop_hard=float(short_cfg.get("stop_hard", 0.20)),
        trail_atr=float(short_cfg.get("trail_atr", 2.0)),
        ma_guard=float(short_cfg.get("ma_guard", 0.03)),
    )

    mode = args.mode
    capital = float(args.capital)
    risk_per_trade = float(args.risk_per_trade)
    max_long = int(args.max_long)
    max_short = int(args.max_short)

    print(
        f"• [{now_str}] Backtest range: {start.date()} → {end.date()} | "
        f"mode={mode}, capital={capital:,.2f}, risk_per_trade={risk_per_trade:.3f}, "
        f"max_long={max_long}, max_short={max_short}"
    )
    print(f"• [{now_str}] Benchmark for Coppock/filters: {benchmark}")
    print(
        f"• [{now_str}] Config: snapshot_mode={snapshot_mode}, "
        f"regime_long={use_regime_long}, regime_short={use_regime_short}, "
        f"coppock_long={use_coppock_long}, coppock_short={use_coppock_short}, "
        f"breadth_enabled={breadth_enabled}, breadth_ma={breadth_ma}, breadth_min_long={breadth_min_long:.2f}, "
        f"LONG_BREAK_PCT={long_params.break_pct}, LONG_VOL_MIN={long_params.vol_min}, "
        f"SHORT_BREAK_PCT={short_params.break_pct}, SHORT_VOL_MIN={short_params.vol_min}, "
        f"ADX_MIN_LONG={adx_min_long:.1f}, ADX_MIN_SHORT={adx_min_short:.1f}"
    )

    # Weekly CSV (static universe)
    output_dir = Path(cfg.get("reporting", {}).get("output_dir", "./output"))
    weekly_csv = find_latest_weekly_csv(output_dir)
    print(f"• [{now_str}] Using weekly CSV: {weekly_csv}")

    weekly_df = pd.read_csv(weekly_csv)
    # Expect a "Ticker" column; define simple long universe = Stage 1/2
    stage_col = "stage" if "stage" in weekly_df.columns else None
    if stage_col:
        long_universe = weekly_df.loc[weekly_df[stage_col].isin([1, 2]), "Ticker"].dropna().unique().tolist()
    else:
        long_universe = weekly_df["Ticker"].dropna().unique().tolist()

    print(f"• [{now_str}] snapshot_mode='{snapshot_mode}': using latest weekly report only for static universe ({len(weekly_df)} tickers).")
    print(f"• [{now_str}] LONG universe size: {len(long_universe)} symbols.")

    # Download daily bars for all tickers + benchmark
    all_tickers = sorted(set(long_universe + [benchmark]))
    hist_start = start - pd.Timedelta(days=120)

    print(f"▶️ [{now_str}] Downloading daily bars for {len(all_tickers)} symbols ({hist_start.date()} → {end.date()})...")
    data = yf.download(
        tickers=" ".join(all_tickers),
        start=hist_start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    print("✅ Download complete.")

    df_map: Dict[str, pd.DataFrame] = {}
    for tkr in all_tickers:
        if len(all_tickers) == 1:
            df_t = data.copy()
        else:
            df_t = data[tkr].copy()
        df_t.dropna(subset=["Close"], inplace=True)
        df_map[tkr] = df_t

    benchmark_close = df_map[benchmark]["Close"]

    # breadth universe uses all tickers in long_universe for now
    breadth_prices = {tkr: df_map[tkr] for tkr in long_universe if tkr in df_map}
    breadth_series = compute_breadth_series(breadth_prices, ma_window=breadth_ma)
    if breadth_enabled:
        print(f"• [{now_str}] Breadth series computed over {len(breadth_prices)} breadth tickers (MA{breadth_ma}).")
    else:
        print(f"• [{now_str}] Breadth gate disabled by config.")

    equity_curve, trades = run_backtest(
        df_map=df_map,
        long_universe=long_universe,
        start_date=start,
        end_date=end,
        capital=capital,
        risk_per_trade=risk_per_trade,
        max_long=max_long,
        use_regime_long=use_regime_long,
        use_coppock_long=use_coppock_long,
        breadth_enabled=breadth_enabled,
        breadth_series=breadth_series,
        breadth_min_long=breadth_min_long,
        adx_min_long=adx_min_long,
        long_params=long_params,
        benchmark_close=benchmark_close,
    )

    final_equity = equity_curve.iloc[-1]
    pl = final_equity - capital
    pl_pct = (pl / capital) * 100 if capital > 0 else 0.0

    now_str2 = dt.datetime.now().strftime("%H:%M:%S")
    print(
        f"✅ [{now_str2}] Backtest complete. Final equity: ${final_equity:,.2f} "
        f"(P/L ${pl:,.2f}, {pl_pct:.2f}%) — Trades: {len(trades)}"
    )

    ts_suffix = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save trades
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = output_dir / f"live_logic_bt_trades_{ts_suffix}.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"✅ [{now_str2}] Wrote trade log → {trades_path}")
    else:
        print(f"⚠️ [{now_str2}] No trades to save.")

    # Equity curve PNG
    import matplotlib.pyplot as plt

    eq_path = output_dir / f"live_logic_bt_equity_{ts_suffix}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Backtest Equity Curve")
    plt.tight_layout()
    plt.savefig(eq_path)
    plt.close()
    print(f"✅ [{now_str2}] Wrote equity curve PNG → {eq_path}")

    # Monthly P/L
    if trades:
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        trades_df["month"] = trades_df["exit_date"].dt.to_period("M").dt.to_timestamp()
        monthly = trades_df.groupby("month")["pnl"].agg(["sum", "count"]).reset_index()
        monthly.rename(columns={"sum": "PnL", "count": "Trades"}, inplace=True)
        monthly["WinRate"] = 0.0
        # crude win-rate: trades with pnl > 0
        win_counts = trades_df.assign(win=lambda d: d["pnl"] > 0).groupby(trades_df["exit_date"].dt.to_period("M"))["win"].sum()
        for idx, row in monthly.iterrows():
            period = row["month"].to_period("M")
            wins = win_counts.get(period, 0)
            total = row["Trades"]
            monthly.loc[idx, "WinRate"] = 100.0 * wins / total if total > 0 else 0.0

        monthly_path = output_dir / f"live_logic_bt_monthly_{ts_suffix}.csv"
        monthly.to_csv(monthly_path, index=False)
        print(f"✅ [{now_str2}] Wrote monthly P/L breakdown → {monthly_path}")
        print("• Monthly P/L summary:")
        for _, row in monthly.iterrows():
            print(
                f"•   {row['month'].strftime('%Y-%m')}: PnL=${row['PnL']:.2f} | "
                f"Trades={int(row['Trades'])} | WinRate={row['WinRate']:5.1f}% "
                f"| Equity=$nan"
            )
    else:
        print(f"⚠️ [{now_str2}] No trades for monthly P/L.")


if __name__ == "__main__":
    main()
