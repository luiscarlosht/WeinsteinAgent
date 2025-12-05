#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_intraday_watcher.py

PROD intraday watcher:
- Uses latest weekly equities CSV as universe (Stage 1/2).
- Downloads intraday + daily bars via yfinance.
- Applies:
    * Chapter 8 regime (market_regime.compute_market_regime)
    * Intraday breadth gate (config.intraday.breadth.*)
    * Intraday ADX14 gate (config.intraday.indicators.* / indicators.*)
- Emits diagnostics CSV + simple HTML summary.

Notes:
- BUY / NEAR logic is intentionally simple but consistent with backtest:
    * NEW BUY if:
        - regime.long_ok
        - breadth_long_ok (if enabled)
        - ADX14 >= INTR_ADX_MIN_LONG
        - current price breaks above yesterday's 20-day high by intraday.confirm_headroom_pct
        - full-day volume pace >= intraday.vol_pace_min (approx based on time-of-day)
    * NEAR if price is within intraday.near_below_pivot_pct below pivot and volume pace >= near_vol_pace_min.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from weinstein_long_core import LongCoreParams, is_breakout, passes_volume_filter
import market_regime


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="▶️ [%(asctime)s] %(message)s",
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
    pattern = "weinstein_weekly_equities_*.csv"
    files = sorted(output_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No weekly CSV found in {output_dir} matching {pattern}")
    return files[-1]


# --------------------------------------------------------------------------------------
# Indicators: ATR, ADX, breadth, volume pace
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


def compute_intraday_breadth(
    breadth_prices: Dict[str, pd.DataFrame],
    ma_window: int,
) -> float:
    """
    Intraday breadth: % of breadth universe trading above MA(ma_window) on *latest* close.
    Uses yesterday's daily close vs MA for simplicity (like daily breadth).
    """
    if not breadth_prices:
        return 0.0

    counts = 0
    n = 0
    for tkr, df in breadth_prices.items():
        if df.empty:
            continue
        close = df["Close"]
        ma = close.rolling(ma_window).mean()
        last_close = close.iloc[-1]
        last_ma = ma.iloc[-1]
        if np.isnan(last_close) or np.isnan(last_ma):
            continue
        n += 1
        if last_close > last_ma:
            counts += 1
    if n == 0:
        return 0.0
    return 100.0 * counts / n


def estimate_volume_pace(
    intraday_df: pd.DataFrame,
    daily_avg_volume: float,
) -> float:
    """
    Approximate full-day volume pace:
    (current intraday volume extrapolated to close) / 50-day avg volume.
    We approximate using fraction of regular session elapsed based on timestamp.
    """
    if daily_avg_volume <= 0 or intraday_df.empty:
        return 0.0

    last_row = intraday_df.iloc[-1]
    current_volume = intraday_df["Volume"].sum()

    ts = last_row.name
    # assume US equities regular session 09:30–16:00 Eastern (6.5h)
    # We approximate fraction of day elapsed by clock time.
    frac_elapsed = min(max(((ts.hour + ts.minute / 60.0) - 9.5) / 6.5, 0.05), 1.0)
    est_full_day = current_volume / frac_elapsed
    return est_full_day / daily_avg_volume


# --------------------------------------------------------------------------------------
# Core intraday watcher
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weinstein intraday watcher.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--log-csv", default="./output/intraday_debug.csv", help="Diagnostics CSV output path")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    cfg = load_config(args.config)
    app_cfg = cfg.get("app", {})
    intr_cfg = cfg.get("intraday", {})
    universe_cfg = cfg.get("universe", {})
    reporting_cfg = cfg.get("reporting", {})

    output_dir = Path(reporting_cfg.get("output_dir", "./output"))
    weekly_csv = find_latest_weekly_csv(output_dir)

    logger.info(f"Intraday watcher starting with config: {args.config}")
    logger.info(f"·· Weekly CSV: {weekly_csv}")

    weekly_df = pd.read_csv(weekly_csv)
    stage_col = "stage" if "stage" in weekly_df.columns else None

    # Focus universe = Stage 1/2 + SP500 mode constraints
    if stage_col:
        focus_df = weekly_df.loc[weekly_df[stage_col].isin([1, 2])]
    else:
        focus_df = weekly_df

    min_price = universe_cfg.get("min_price", 5)
    focus_df = focus_df.loc[focus_df["Close"] >= min_price]
    focus_universe = focus_df["Ticker"].dropna().unique().tolist()

    logger.info(f"• Focus universe: {len(focus_universe)} symbols (Stage 1/2).")

    # Intraday config
    vol_pace_min = float(intr_cfg.get("vol_pace_min", 1.3))
    near_vol_pace_min = float(intr_cfg.get("near_vol_pace_min", 1.0))
    sell_intrabar_vol_pace_min = float(intr_cfg.get("sell_intrabar_vol_pace_min", 1.2))
    confirm_headroom_pct = float(intr_cfg.get("confirm_headroom_pct", 0.4))
    near_below_pivot_pct = float(intr_cfg.get("near_below_pivot_pct", 0.3))
    crack_ma_pct = float(intr_cfg.get("crack_ma_pct", 0.5))
    min_elapsed_minutes = int(intr_cfg.get("min_elapsed_minutes", 40))
    ma_proxy_length = int(intr_cfg.get("ma_proxy_length", 150))

    # Breadth config
    br_cfg = intr_cfg.get("breadth", {})
    INTR_BREADTH_ENABLED = bool(br_cfg.get("enabled", True))
    INTR_BREADTH_MA = int(br_cfg.get("ma_window", 50))
    INTR_BREADTH_MIN_LONG = float(br_cfg.get("min_long", 0.60))

    # ADX config
    shared_ind = cfg.get("indicators", {})
    intr_ind = intr_cfg.get("indicators", {})

    INTR_ADX_MIN_LONG = float(intr_ind.get("adx_min_long", shared_ind.get("adx_min_long", 18.0)))
    INTR_ADX_MIN_SHORT = float(intr_ind.get("adx_min_short", shared_ind.get("adx_min_short", 18.0)))

    logger.info(
        f"• Intraday config: breadth_enabled={INTR_BREADTH_ENABLED}, breadth_ma={INTR_BREADTH_MA}, "
        f"breadth_min_long={INTR_BREADTH_MIN_LONG:.2f}, ADX_MIN_LONG={INTR_ADX_MIN_LONG:.1f}, "
        f"ADX_MIN_SHORT={INTR_ADX_MIN_SHORT:.1f}"
    )

    # Download intraday (today) + recent daily history
    tickers = focus_universe
    if not tickers:
        logger.info("⚠️ No tickers in focus universe — nothing to do.")
        return

    # Daily history for indicators
    hist_start = dt.date.today() - dt.timedelta(days=120)
    logger.info("▶️ Downloading intraday + daily bars...")
    daily_data = yf.download(
        tickers=" ".join(tickers),
        start=hist_start.strftime("%Y-%m-%d"),
        end=(dt.date.today() + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    intraday_data = yf.download(
        tickers=" ".join(tickers),
        period="1d",
        interval="5m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    logger.info("✅ Price data downloaded.")

    # Build per-ticker daily frames
    daily_map: Dict[str, pd.DataFrame] = {}
    intr_map: Dict[str, pd.DataFrame] = {}

    many = len(tickers) > 1
    for tkr in tickers:
        try:
            if many:
                ddf = daily_data[tkr].copy()
                idf = intraday_data[tkr].copy()
            else:
                ddf = daily_data.copy()
                idf = intraday_data.copy()
        except KeyError:
            continue

        ddf.dropna(subset=["Close"], inplace=True)
        idf.dropna(subset=["Close"], inplace=True)

        daily_map[tkr] = ddf
        intr_map[tkr] = idf

    # Compute breadth
    breadth_prices = daily_map
    breadth_pct = compute_intraday_breadth(breadth_prices, ma_window=INTR_BREADTH_MA)

    if INTR_BREADTH_ENABLED:
        threshold_pct = INTR_BREADTH_MIN_LONG * 100.0
        breadth_long_ok = breadth_pct >= threshold_pct
        logger.info(
            f"• Breadth Health: {breadth_pct:.1f}% of breadth universe above MA{INTR_BREADTH_MA} "
            f"→ breadth_long_ok={breadth_long_ok} (threshold {threshold_pct:.1f}%)"
        )
    else:
        breadth_long_ok = True
        logger.info(
            f"• Breadth gate DISABLED by config (intraday.breadth.enabled=false). "
            f"Computed breadth={breadth_pct:.1f}% (ignored)."
        )

    # Market regime (Chapter 8)
    benchmark_symbol = cfg.get("app", {}).get("benchmark", "SPY")
    bench_hist = yf.download(
        benchmark_symbol,
        start=hist_start.strftime("%Y-%m-%d"),
        end=(dt.date.today() + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
    )
    bench_close = bench_hist["Close"]
    regime = market_regime.compute_market_regime(bench_close)
    logger.info(
        f"• Market regime (Ch8): {regime.name} | long_ok={regime.long_ok} short_ok={regime.short_ok}"
    )

    # Prepare diagnostics
    rows: List[dict] = []
    now_ts = dt.datetime.now()
    latest_time = None

    # Evaluate candidates
    logger.info("▶️ Evaluating candidates...")
    for tkr in tickers:
        if tkr not in daily_map or tkr not in intr_map:
            continue

        ddf = daily_map[tkr]
        idf = intr_map[tkr]
        if ddf.empty or idf.empty:
            continue

        # Compute ADX on daily
        ddf = ddf.copy()
        ddf["ATR14"] = compute_atr(ddf, period=14)
        ddf["ADX14"] = compute_adx(ddf, period=14)
        ddf["Vol50"] = ddf["Volume"].rolling(50).mean()
        ddf["VolRatio"] = ddf["Volume"] / ddf["Vol50"]
        daily_map[tkr] = ddf

        last_daily = ddf.iloc[-1]
        adx14 = float(last_daily["ADX14"]) if not np.isnan(last_daily["ADX14"]) else np.nan
        vol50 = float(last_daily["Vol50"]) if not np.isnan(last_daily["Vol50"]) else 0.0

        # ADX filter
        if not np.isnan(adx14) and adx14 < INTR_ADX_MIN_LONG:
            logger.info(
                f"·· [SKIP-ADX] {tkr} because ADX14={adx14:.1f} < {INTR_ADX_MIN_LONG:.1f}"
            )
            continue

        # regime + breadth gates
        if not regime.long_ok or not breadth_long_ok:
            continue

        # Volume pace estimate
        vol_pace = estimate_volume_pace(idf, daily_avg_volume=vol50 if vol50 > 0 else 1.0)

        # simple pivot: yesterday's 20-day high
        hist = ddf.tail(21)
        if len(hist) < 21:
            continue
        pivot = hist["High"].iloc[:-1].max()

        last_intra = idf.iloc[-1]
        latest_time = last_intra.name
        last_price = float(last_intra["Close"])

        price_above_pivot_pct = (last_price / pivot - 1.0) * 100 if pivot > 0 else np.nan

        # classify BUY / NEAR / NONE
        status = "NONE"
        reason = ""

        # BUY
        if (
            not np.isnan(price_above_pivot_pct)
            and price_above_pivot_pct >= confirm_headroom_pct
            and vol_pace >= vol_pace_min
        ):
            status = "BUY"
            reason = f"price {price_above_pivot_pct:.2f}% above pivot, vol_pace={vol_pace:.2f}"
        # NEAR
        elif (
            not np.isnan(price_above_pivot_pct)
            and -near_below_pivot_pct <= price_above_pivot_pct < confirm_headroom_pct
            and vol_pace >= near_vol_pace_min
        ):
            status = "NEAR"
            reason = f"within {near_below_pivot_pct:.2f}% of pivot, vol_pace={vol_pace:.2f}"

        rows.append(
            {
                "timestamp": latest_time,
                "ticker": tkr,
                "status": status,
                "price": last_price,
                "pivot": pivot,
                "price_vs_pivot_pct": price_above_pivot_pct,
                "vol_pace": vol_pace,
                "adx14": adx14,
                "regime_long_ok": regime.long_ok,
                "breadth_long_ok": breadth_long_ok,
                "reason": reason,
            }
        )

    # Aggregate and save diagnostics
    diag_df = pd.DataFrame(rows)
    log_csv = Path(args.log_csv)

    if diag_df.empty:
        # write an empty file to keep downstream tools happy
        log_csv.write_text("")
        logger.info("• Scan done. Raw counts → BUY:0 NEAR:0")
        logger.info(f"✅ Wrote diagnostics CSV → {log_csv}")
        # simple empty HTML stub
        html_path = output_dir / f"intraday_watch_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_path.write_text("<html><body><h3>No intraday triggers.</h3></body></html>")
        logger.info(f"✅ Saved HTML → {html_path}")
        logger.info("• No BUY/NEAR triggers present — skipping email send.")
        logger.info("✅ Intraday tick complete.")
        return

    diag_df.to_csv(log_csv, index=False)

    buy_count = (diag_df["status"] == "BUY").sum()
    near_count = (diag_df["status"] == "NEAR").sum()

    logger.info(
        f"• Scan done. Raw counts → BUY:{buy_count} NEAR:{near_count}"
    )
    logger.info(f"✅ Wrote diagnostics CSV → {log_csv}")

    # Simple HTML summary
    ts_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = output_dir / f"intraday_watch_{ts_str}.html"
    html_parts = [
        "<html><body>",
        f"<h2>Intraday Watch — {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</h2>",
        f"<p>Regime: long_ok={regime.long_ok}, short_ok={regime.short_ok}</p>",
        f"<p>Breadth: {breadth_pct:.1f}% above MA{INTR_BREADTH_MA} "
        f"(gate={'ON' if INTR_BREADTH_ENABLED else 'OFF'}, "
        f"threshold={INTR_BREADTH_MIN_LONG*100:.1f}%)</p>",
        "<h3>BUY / NEAR Candidates</h3>",
        diag_df.to_html(index=False),
        "</body></html>",
    ]
    html_path.write_text("\n".join(html_parts))
    logger.info(f"✅ Saved HTML → {html_path}")

    # Email behaviour is handled by your outer runner (run_cron_short_stack.sh etc.)
    logger.info("• No direct email send from this script — caller decides based on CSV/HTML.")
    logger.info("✅ Intraday tick complete.")


if __name__ == "__main__":
    main()
