#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Live Logic Backtest — yfinance (SIM)

- Uses config.yaml knobs:
    * backtest.snapshot_mode  (static / historical / auto)  [currently static]
    * backtest.regime.use_long / use_short
    * backtest.coppock.use_long / use_short
    * backtest.breadth.enabled / ma_window / min_long
    * backtest.long / backtest.short  (break_pct, vol_min, stops, ADX, etc.)
    * backtest.logging.show_adx_skips  (controls noisy [SKIP-ADX] lines)
- Universe from latest weekly equities CSV (static mode)
- Daily bars from yfinance

Typical run:

python3 weinstein_live_logic_backtest_yfinance.py \
  --config ./config.yaml \
  --start 2015-01-01 \
  --end   2015-12-31 \
  --mode both \
  --capital 10000 \
  --risk-per-trade 0.01 \
  --max-long 10 \
  --max-short 10 \
  --benchmark SPY
"""

import argparse
import datetime as dt
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import yaml


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"• [{now}] {msg}")


def log_step(msg: str) -> None:
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"▶️ [{now}] {msg}")


def log_sub(msg: str) -> None:
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"·· [{now}] {msg}")


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

@dataclass
class BacktestLongConfig:
    break_pct: float
    vol_min: float
    stop_hard: float
    trail_atr: float
    ma_guard: float
    adx_min: float


@dataclass
class BacktestShortConfig:
    break_pct: float
    vol_min: float
    stop_hard: float
    trail_atr: float
    ma_guard: float
    adx_min: float


@dataclass
class BacktestBreadthConfig:
    enabled: bool
    ma_window: int
    min_long: float  # 0–1 fraction


@dataclass
class BacktestRegimeConfig:
    use_long: bool
    use_short: bool


@dataclass
class BacktestCoppockConfig:
    use_long: bool
    use_short: bool


@dataclass
class BacktestGlobalConfig:
    snapshot_mode: str
    long_cfg: BacktestLongConfig
    short_cfg: BacktestShortConfig
    breadth_cfg: BacktestBreadthConfig
    regime_cfg: BacktestRegimeConfig
    coppock_cfg: BacktestCoppockConfig
    benchmark: str
    output_dir: str
    show_adx_skips: bool  # control noisy [SKIP-ADX] logs


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_bt_config(cfg_raw: Dict, benchmark_override: Optional[str]) -> BacktestGlobalConfig:
    reporting = cfg_raw.get("reporting", {})
    app = cfg_raw.get("app", {})
    backtest = cfg_raw.get("backtest", {})

    snapshot_mode = str(backtest.get("snapshot_mode", "static")).lower()

    bt_long = backtest.get("long", {})
    bt_short = backtest.get("short", {})

    # ADX thresholds — support both adx_min_long/short and adx_min (your YAML)
    adx_min_long = float(bt_long.get("adx_min_long", bt_long.get("adx_min", 18.0)))
    adx_min_short = float(bt_short.get("adx_min_short", bt_short.get("adx_min", 18.0)))

    long_cfg = BacktestLongConfig(
        break_pct=float(bt_long.get("break_pct", 0.004)),
        vol_min=float(bt_long.get("vol_min", 1.3)),
        stop_hard=float(bt_long.get("stop_hard", 0.20)),
        trail_atr=float(bt_long.get("trail_atr", 2.0)),
        ma_guard=float(bt_long.get("ma_guard", 0.03)),
        adx_min=adx_min_long,
    )

    short_cfg = BacktestShortConfig(
        break_pct=float(bt_short.get("break_pct", 0.004)),
        vol_min=float(bt_short.get("vol_min", 1.3)),
        stop_hard=float(bt_short.get("stop_hard", 0.20)),
        trail_atr=float(bt_short.get("trail_atr", 2.0)),
        ma_guard=float(bt_short.get("ma_guard", 0.03)),
        adx_min=adx_min_short,
    )

    bt_regime = backtest.get("regime", {})
    regime_cfg = BacktestRegimeConfig(
        use_long=bool(bt_regime.get("use_long", True)),
        use_short=bool(bt_regime.get("use_short", True)),
    )

    bt_coppock = backtest.get("coppock", {})
    coppock_cfg = BacktestCoppockConfig(
        use_long=bool(bt_coppock.get("use_long", True)),
        use_short=bool(bt_coppock.get("use_short", True)),
    )

    bt_breadth = backtest.get("breadth", {})
    breadth_cfg = BacktestBreadthConfig(
        enabled=bool(bt_breadth.get("enabled", True)),
        ma_window=int(bt_breadth.get("ma_window", 50)),
        min_long=float(bt_breadth.get("min_long", 0.60)),
    )

    bt_logging = backtest.get("logging", {})
    show_adx_skips = bool(bt_logging.get("show_adx_skips", False))

    benchmark = benchmark_override or app.get("benchmark", "SPY")

    return BacktestGlobalConfig(
        snapshot_mode=snapshot_mode,
        long_cfg=long_cfg,
        short_cfg=short_cfg,
        breadth_cfg=breadth_cfg,
        regime_cfg=regime_cfg,
        coppock_cfg=coppock_cfg,
        benchmark=benchmark,
        output_dir=reporting.get("output_dir", "./output"),
        show_adx_skips=show_adx_skips,
    )


# ---------------------------------------------------------------------------
# Weekly universe loader (Stage normalization)
# ---------------------------------------------------------------------------

def find_latest_weekly_csv(output_dir: str) -> str:
    pattern = os.path.join(output_dir, "weinstein_weekly_equities_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No weekly CSVs found at {pattern}")
    latest = max(files, key=os.path.getmtime)
    return latest


def normalize_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have: Ticker, Close, Stage (1..4)."""
    if df.empty:
        return df

    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    # Ticker
    ticker_col = None
    for key in ["ticker", "symbol", "sym"]:
        if key in lower:
            ticker_col = lower[key]
            break
    if ticker_col is None:
        ticker_col = cols[0]
    if ticker_col != "Ticker":
        df.rename(columns={ticker_col: "Ticker"}, inplace=True)

    # Close
    close_col = None
    for key in ["close", "price", "last", "last_price"]:
        if key in lower:
            close_col = lower[key]
            break
    if close_col is None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        close_col = numeric_cols[0] if numeric_cols else cols[0]
    if close_col != "Close":
        df.rename(columns={close_col: "Close"}, inplace=True)

    # Stage
    stage_col_raw = None
    for key in ["stage", "weinstien_stage", "stage_weinstein", "stage_num"]:
        if key in lower:
            stage_col_raw = lower[key]
            break

    if stage_col_raw is None:
        df["Stage"] = 2
    else:
        if stage_col_raw != "Stage":
            df.rename(columns={stage_col_raw: "Stage"}, inplace=True)
        stage_num = pd.to_numeric(df["Stage"], errors="coerce")
        stage_num = stage_num.fillna(2)
        stage_num = stage_num.clip(lower=1, upper=4).astype(int)
        df["Stage"] = stage_num

    return df


def load_static_universe(output_dir: str) -> pd.DataFrame:
    csv_path = find_latest_weekly_csv(output_dir)
    log(f"Using weekly CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = normalize_weekly_columns(df)
    df = df.dropna(subset=["Ticker"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["Ticker"])
    return df


# ---------------------------------------------------------------------------
# Indicators & helpers
# ---------------------------------------------------------------------------

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = low.diff().mul(-1)

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx


def compute_breadth(series_dict: Dict[str, pd.Series], ma_window: int) -> pd.Series:
    """
    series_dict: ticker -> close series
    Returns daily breadth series: % above MA(ma_window).
    """
    if not series_dict:
        return pd.Series(dtype=float)

    closes = pd.DataFrame(series_dict)  # index: date, columns: ticker
    ma = closes.rolling(ma_window).mean()
    above = (closes > ma).sum(axis=1)
    breadth = above / closes.count(axis=1)
    return breadth


def compute_coppock_curve(close: pd.Series, w1: int = 11, w2: int = 14, ema: int = 10) -> pd.Series:
    """Simple Coppock curve on monthly closes."""
    if close.empty:
        return pd.Series(dtype=float)

    monthly = close.resample("ME").last()
    roc1 = monthly.pct_change(w1)
    roc2 = monthly.pct_change(w2)
    coppock_raw = roc1 + roc2
    coppock = coppock_raw.ewm(span=ema, adjust=False).mean()
    coppock = coppock.reindex(close.index, method="ffill")
    return coppock


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_daily_bars(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
    warmup_days: int = 200,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    start_warmup = start - dt.timedelta(days=warmup_days)
    tickers_str = " ".join(sorted(set(tickers)))
    log_step(f"Downloading daily bars for {len(tickers)} symbols ({start_warmup} → {end})...")
    df = yf.download(
        tickers_str,
        start=start_warmup,
        end=end + dt.timedelta(days=1),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    log("Download complete.")

    def stack(df_raw: pd.DataFrame) -> pd.DataFrame:
        if df_raw.empty:
            return pd.DataFrame()
        if isinstance(df_raw.columns, pd.MultiIndex):
            frames = []
            for ticker in sorted(set(sym for sym, _ in df_raw.columns)):
                sub = df_raw[ticker].copy()
                sub["Ticker"] = ticker
                frames.append(sub)
            out = pd.concat(frames)
        else:
            out = df_raw.copy()
            out["Ticker"] = tickers[0]
        out.reset_index(inplace=True)
        date_col = "Date" if "Date" in out.columns else "Datetime"
        out = out.rename(columns={date_col: "Date"})
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index(["Date", "Ticker"])
        return out

    stacked = stack(df)
    return stacked.sort_index()


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class Position:
    side: str  # "long" or "short"
    ticker: str
    entry_date: dt.date
    entry_price: float
    size: int
    stop_price: float
    trail_stop: Optional[float]


def simulate_backtest(
    bt_cfg: BacktestGlobalConfig,
    weekly_df: pd.DataFrame,
    daily: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    mode: str,
    capital: float,
    risk_per_trade: float,
    max_long: int,
    max_short: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        equity_curve: DataFrame[date, equity]
        trades:       DataFrame[trade log]
    """
    # Build universes using cleaned Stage
    weekly_df = weekly_df.copy()
    if "Stage" in weekly_df.columns:
        stage_num = pd.to_numeric(weekly_df["Stage"], errors="coerce").fillna(2).astype(int)
        weekly_df["Stage"] = stage_num.clip(1, 4)
        long_universe = weekly_df.loc[weekly_df["Stage"].isin([1, 2]), "Ticker"].tolist()
        short_universe = weekly_df.loc[weekly_df["Stage"].isin([3, 4]), "Ticker"].tolist()
        if not long_universe:
            log("⚠️ LONG universe from Stage 1/2 is empty — falling back to all tickers for longs.")
            long_universe = weekly_df["Ticker"].tolist()
        if not short_universe:
            log("⚠️ SHORT universe from Stage 3/4 is empty — using empty short set.")
    else:
        long_universe = weekly_df["Ticker"].tolist()
        short_universe = weekly_df["Ticker"].tolist()

    long_universe = sorted(set(long_universe))
    short_universe = sorted(set(short_universe))

    log(f"LONG universe size: {len(long_universe)} symbols.")
    log(f"SHORT universe size: {len(short_universe)} symbols.")

    # Build close series dict for breadth and Coppock
    close_dict = {}
    for ticker in sorted(set(long_universe + short_universe)):
        try:
            d = daily.xs(ticker, level="Ticker").sort_index()
        except KeyError:
            continue
        if "Close" not in d.columns:
            continue
        close_dict[ticker] = d["Close"]

    # Breadth series
    breadth = pd.Series(dtype=float)
    if bt_cfg.breadth_cfg.enabled and close_dict:
        breadth = compute_breadth(close_dict, bt_cfg.breadth_cfg.ma_window)
        log(
            "Breadth series computed over "
            f"{len(close_dict)} breadth tickers (MA{bt_cfg.breadth_cfg.ma_window})."
        )
    else:
        log("Breadth filter disabled for backtest or no close data for breadth.")

    # Benchmark for Coppock
    try:
        bench = daily.xs(bt_cfg.benchmark, level="Ticker").sort_index()
    except KeyError:
        bench = pd.DataFrame()
    coppock = pd.Series(dtype=float)
    if not bench.empty and bt_cfg.coppock_cfg.use_long:
        coppock = compute_coppock_curve(bench["Close"])
        log(
            f"Coppock curve computed for benchmark {bt_cfg.benchmark} "
            f"(monthly points={coppock.dropna().shape[0]})."
        )
    else:
        log(f"Coppock curve disabled or benchmark {bt_cfg.benchmark} data missing.")

    # Positions and equity
    positions: Dict[Tuple[str, str], Position] = {}  # (side, ticker) -> Position
    equity = capital
    equity_curve_rows = []
    trades_rows = []

    # -------------------------------------------------------------------
    # Group daily data by calendar date (robust against .xs Date issues)
    # -------------------------------------------------------------------
    if daily.empty:
        log("⚠️ Daily price DataFrame is empty — no backtest possible.")
        return pd.DataFrame(), pd.DataFrame()

    daily_reset = daily.reset_index()  # columns: Date, Ticker, OHLCV...
    daily_reset["TradeDate"] = daily_reset["Date"].dt.date

    grouped_by_date: Dict[dt.date, pd.DataFrame] = {}
    for trade_date, g in daily_reset.groupby("TradeDate"):
        grouped_by_date[trade_date] = g.set_index("Ticker")

    all_dates = sorted(grouped_by_date.keys())
    trade_dates = [d for d in all_dates if start <= d <= end]
    if not trade_dates:
        log("⚠️ No trading dates in requested range — nothing to simulate.")
        return pd.DataFrame(), pd.DataFrame()

    total_days = len(trade_dates)

    # -------------------------------------------------------------------
    # Main backtest loop
    # -------------------------------------------------------------------
    for idx, trade_date in enumerate(trade_dates, start=1):
        day_slice = grouped_by_date[trade_date]

        # Progress log once per calendar month (on the LAST trading day of that month)
        next_is_new_month = idx < total_days and trade_dates[idx].month != trade_date.month
        is_last_of_month = (idx == total_days) or next_is_new_month
        if is_last_of_month:
            pct_done = idx / total_days * 100.0
            log_sub(
                f"Progress: {trade_date} — equity ${equity:,.2f}, "
                f"positions: {len(positions)}, trades so far: {len(trades_rows)}, "
                f"done≈{pct_done:.1f}%"
            )

        # Update existing positions (stops)
        to_close: List[Tuple[str, str, float]] = []
        for (side, ticker), pos in positions.items():
            if ticker not in day_slice.index:
                continue
            bar = day_slice.loc[ticker]
            close_price = float(bar["Close"])

            if side == "long":
                stop_price = pos.trail_stop if pos.trail_stop is not None else pos.stop_price
                if close_price <= stop_price:
                    pnl = (close_price - pos.entry_price) * pos.size
                    equity += pnl
                    trades_rows.append(
                        dict(
                            side=side,
                            ticker=ticker,
                            entry_date=pos.entry_date,
                            exit_date=trade_date,
                            entry_price=pos.entry_price,
                            exit_price=close_price,
                            size=pos.size,
                            pnl=pnl,
                        )
                    )
                    to_close.append((side, ticker, close_price))
            else:
                stop_price = pos.trail_stop if pos.trail_stop is not None else pos.stop_price
                if close_price >= stop_price:
                    pnl = (pos.entry_price - close_price) * pos.size
                    equity += pnl
                    trades_rows.append(
                        dict(
                            side=side,
                            ticker=ticker,
                            entry_date=pos.entry_date,
                            exit_date=trade_date,
                            entry_price=pos.entry_price,
                            exit_price=close_price,
                            size=pos.size,
                            pnl=pnl,
                        )
                    )
                    to_close.append((side, ticker, close_price))

        for side, ticker, _ in to_close:
            positions.pop((side, ticker), None)

        # Determine gates for new positions (LONG side for now)
        breadth_ok_long = True
        if bt_cfg.breadth_cfg.enabled and not breadth.empty and trade_date in breadth.index:
            b_val = float(breadth.loc[trade_date])
            if b_val < bt_cfg.breadth_cfg.min_long:
                breadth_ok_long = False
                log_sub(
                    f"[SKIP-BREADTH] No new LONGs on {trade_date} because "
                    f"breadth={b_val*100:.2f}% < {bt_cfg.breadth_cfg.min_long*100:.0f}%"
                )

        coppock_ok_long = True
        if not coppock.empty and bt_cfg.coppock_cfg.use_long:
            if trade_date in coppock.index:
                c_val = float(coppock.loc[trade_date])
                if c_val <= 0:
                    coppock_ok_long = False

        allow_new_longs = (
            (mode in ("long", "both"))
            and bt_cfg.regime_cfg.use_long
            and breadth_ok_long
            and coppock_ok_long
        )

        # New long entries
        if allow_new_longs:
            for ticker in long_universe:
                if ("long", ticker) in positions:
                    continue
                if ticker not in day_slice.index:
                    continue
                bar = day_slice.loc[ticker]
                if any(pd.isna(bar.get(c, np.nan)) for c in ["High", "Low", "Close", "Volume"]):
                    continue

                # Indicator history up to trade_date
                try:
                    hist = daily.xs(ticker, level="Ticker").sort_index()
                except KeyError:
                    continue
                hist = hist.loc[hist.index.get_level_values("Date") <= pd.Timestamp(trade_date)].tail(200)
                if hist.shape[0] < 60:
                    continue

                hist["MA30"] = hist["Close"].rolling(30).mean()
                hist["MA150"] = hist["Close"].rolling(150).mean()
                hist["ATR14"] = (hist["High"] - hist["Low"]).rolling(14).mean()
                hist["VolMA50"] = hist["Volume"].rolling(50).mean()
                hist["ADX14"] = compute_adx(hist["High"], hist["Low"], hist["Close"], period=14)

                last = hist.iloc[-1]
                adx14 = float(last["ADX14"]) if not pd.isna(last["ADX14"]) else np.nan

                if np.isnan(adx14) or adx14 < bt_cfg.long_cfg.adx_min:
                    if bt_cfg.show_adx_skips:
                        log_sub(
                            f"[SKIP-ADX] {ticker} because ADX14={adx14:.1f} < {bt_cfg.long_cfg.adx_min:.1f} "
                            f"on {trade_date}"
                        )
                    continue

                # Stage-like condition via MA150
                if pd.isna(last["MA150"]) or last["Close"] <= last["MA150"] * (1.0 + bt_cfg.long_cfg.ma_guard):
                    continue

                # Volume pace
                vol_ma50 = float(last["VolMA50"]) if not pd.isna(last["VolMA50"]) else np.nan
                if np.isnan(vol_ma50) or vol_ma50 <= 0:
                    continue
                vol_pace = last["Volume"] / vol_ma50
                if vol_pace < bt_cfg.long_cfg.vol_min:
                    continue

                # Breakout vs 50d high close
                pivot = hist["Close"].tail(60).max()
                trigger_price = pivot * (1.0 + bt_cfg.long_cfg.break_pct)
                if last["Close"] < trigger_price:
                    continue

                entry_price = float(last["Close"])
                if entry_price <= 0:
                    continue

                # Risk sizing
                risk_per_pos = equity * risk_per_trade
                stop_price = entry_price * (1.0 - bt_cfg.long_cfg.stop_hard)
                per_share_risk = entry_price - stop_price
                if per_share_risk <= 0:
                    continue
                size = int(risk_per_pos // per_share_risk)
                if size <= 0:
                    continue
                if len([p for p in positions.values() if p.side == "long"]) >= max_long:
                    continue

                positions[("long", ticker)] = Position(
                    side="long",
                    ticker=ticker,
                    entry_date=trade_date,
                    entry_price=entry_price,
                    size=size,
                    stop_price=stop_price,
                    trail_stop=None,
                )

        # End-of-day equity mark-to-market
        day_equity = equity
        for (side, ticker), pos in positions.items():
            if ticker not in day_slice.index:
                continue
            close_price = float(day_slice.loc[ticker]["Close"])
            if side == "long":
                mtm = (close_price - pos.entry_price) * pos.size
            else:
                mtm = (pos.entry_price - close_price) * pos.size
            day_equity += mtm

        equity_curve_rows.append(dict(Date=trade_date, Equity=day_equity))

    if not equity_curve_rows:
        log("⚠️ No equity points recorded during backtest — returning empty results.")
        return pd.DataFrame(), pd.DataFrame()

    equity_curve = pd.DataFrame(equity_curve_rows).set_index("Date")
    trades = pd.DataFrame(trades_rows)
    return equity_curve, trades


# ---------------------------------------------------------------------------
# Plotting & outputs
# ---------------------------------------------------------------------------

def save_equity_curve_png(equity: pd.DataFrame, outdir: str, stamp: str) -> None:
    if equity.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"live_logic_bt_equity_{stamp}.png")
    plt.figure()
    plt.plot(equity.index, equity["Equity"])
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Weinstein Live Logic Backtest — Equity Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log(f"Wrote equity curve PNG → {path}")


def save_trades_csv(trades: pd.DataFrame, outdir: str, stamp: str) -> Optional[str]:
    if trades.empty:
        log("No trades to save.")
        return None
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"live_logic_bt_trades_{stamp}.csv")
    trades.to_csv(path, index=False)
    log(f"Wrote trade log → {path}")
    return path


def save_monthly_pnl(trades: pd.DataFrame, outdir: str, stamp: str) -> None:
    if trades.empty:
        log("No trades for monthly P/L.")
        return
    trades = trades.copy()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades["month"] = trades["exit_date"].dt.to_period("M")
    monthly = trades.groupby("month")["pnl"].agg(["sum", "count"])
    monthly.rename(columns={"sum": "PnL", "count": "Trades"}, inplace=True)
    monthly["WinRate"] = np.nan  # left blank; can be computed if needed
    path = os.path.join(outdir, f"live_logic_bt_monthly_{stamp}.csv")
    monthly.to_csv(path)
    log(f"Wrote monthly P/L breakdown → {path}")
    log("Monthly P/L summary:")
    for idx, row in monthly.iterrows():
        print(
            f"• {idx}: PnL=${row['PnL']:.2f} | "
            f"Trades={int(row['Trades'])} | WinRate={row['WinRate']!s}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weinstein Live Logic Backtest (SIM)")
    p.add_argument("--config", type=str, default="./config.yaml", help="config.yaml path")
    p.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    p.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["long", "short", "both"],
        help="Which side(s) to trade",
    )
    p.add_argument("--capital", type=float, default=10000.0, help="Starting capital")
    p.add_argument("--risk-per-trade", type=float, default=0.01, help="Risk per trade as fraction of equity")
    p.add_argument("--max-long", type=int, default=10, help="Max concurrent long positions")
    p.add_argument("--max-short", type=int, default=10, help="Max concurrent short positions")
    p.add_argument("--benchmark", type=str, default=None, help="Override benchmark symbol")
    p.add_argument(
        "--show-adx-skips",
        action="store_true",
        help="log a debug line for every ADX-based skip",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end, "%Y-%m-%d").date()

    cfg_raw = load_yaml_config(args.config)
    bt_cfg = build_bt_config(cfg_raw, args.benchmark)

    # CLI overrides config for show_adx_skips
    if getattr(args, "show_adx_skips", False):
        bt_cfg.show_adx_skips = True

    log(
        f"Backtest range: {start} → {end} | "
        f"mode={args.mode}, capital={args.capital:,.2f}, "
        f"risk_per_trade={args.risk_per_trade:.3f}, "
        f"max_long={args.max_long}, max_short={args.max_short}"
    )

    log(
        "Config: "
        f"snapshot_mode={bt_cfg.snapshot_mode}, "
        f"regime_long={bt_cfg.regime_cfg.use_long}, regime_short={bt_cfg.regime_cfg.use_short}, "
        f"coppock_long={bt_cfg.coppock_cfg.use_long}, coppock_short={bt_cfg.coppock_cfg.use_short}, "
        f"breadth_enabled={bt_cfg.breadth_cfg.enabled}, breadth_ma={bt_cfg.breadth_cfg.ma_window}, "
        f"breadth_min_long={bt_cfg.breadth_cfg.min_long:.2f}, "
        f"LONG_BREAK_PCT={bt_cfg.long_cfg.break_pct}, LONG_VOL_MIN={bt_cfg.long_cfg.vol_min}, "
        f"SHORT_BREAK_PCT={bt_cfg.short_cfg.break_pct}, SHORT_VOL_MIN={bt_cfg.short_cfg.vol_min}, "
        f"ADX_MIN_LONG={bt_cfg.long_cfg.adx_min}, ADX_MIN_SHORT={bt_cfg.short_cfg.adx_min}, "
        f"SHOW_ADX_SKIPS={bt_cfg.show_adx_skips}"
    )

    log(f"Using weekly CSV directory: {bt_cfg.output_dir}")

    weekly_df = load_static_universe(bt_cfg.output_dir)

    # Build total symbol list (universe + benchmark)
    tickers = weekly_df["Ticker"].tolist()
    tickers.append(bt_cfg.benchmark)
    tickers = sorted(set(tickers))

    daily = fetch_daily_bars(tickers, start, end)

    # Run simulation
    equity_curve, trades = simulate_backtest(
        bt_cfg=bt_cfg,
        weekly_df=weekly_df,
        daily=daily,
        start=start,
        end=end,
        mode=args.mode,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        max_long=args.max_long,
        max_short=args.max_short,
    )

    if equity_curve.empty:
        log("Backtest produced empty equity curve.")
        return

    final_equity = float(equity_curve["Equity"].iloc[-1])
    pl = final_equity - args.capital
    pl_pct = pl / args.capital * 100.0
    log(
        f"Backtest complete. Final equity: ${final_equity:,.2f} "
        f"(P/L ${pl:,.2f}, {pl_pct:.2f}%) — Trades: {len(trades)}"
    )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_trades_csv(trades, bt_cfg.output_dir, stamp)
    save_equity_curve_png(equity_curve, bt_cfg.output_dir, stamp)
    save_monthly_pnl(trades, bt_cfg.output_dir, stamp)


if __name__ == "__main__":
    main()
