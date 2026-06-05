#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Intraday Watcher (PROD) — uses config.yaml

- Uses latest weekly equities report as the "core" universe
- Normalizes weekly CSV columns so we never crash on missing 'Ticker'/'Close'
- Cleans 'Stage' into numeric 1..4 (default 2) so Stage filters never kill universe
- Applies:
    * Stage 1/2 filter for longs
    * Min price / min avg volume from config.universe
    * ADX filter from config.intraday / backtest
    * Intraday breakout logic using 60m bars
- Saves:
    * intraday_debug.csv   → detailed per-ticker diagnostics
    * intraday_watch_*.html (polished HTML/email-style summary)

CLI:
    python3 weinstein_intraday_watcher.py \
        --config ./config.yaml \
        --log-csv ./output/intraday_debug.csv
"""

import argparse
import datetime as dt
import glob
import os
import re
import html
import io
import base64
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from weinstein_prod_history import append_prod_signal_history
import yfinance as yf
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weinstein_long_core import LongEntryParams, evaluate_long_signal, should_exit_long
from weinstein_regime_exposure_core import decide_regime_exposure, read_d_config

try:
    from weinstein_mailer import send_email
except Exception:
    send_email = None


# ---------------------------------------------------------------------------
# Small helpers for logging
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
# Config models + loaders
# ---------------------------------------------------------------------------

@dataclass
class IntradayConfig:
    vol_pace_min: float
    near_vol_pace_min: float
    sell_intrabar_vol_pace_min: float
    confirm_headroom_pct: float
    near_below_pivot_pct: float
    crack_ma_pct: float
    min_elapsed_minutes: int
    ma_proxy_length: int
    adx_min_long: float
    daily_history_period: str
    stage_above_ma_pct: float
    dist_above_ma_min: float
    pivot_lookback_days: int
    breadth_enabled: bool
    breadth_ma_window: int
    breadth_min_long: float


@dataclass
class UniverseConfig:
    min_price: float
    min_avg_volume: int


@dataclass
class RegimeConfig:
    use_long: bool
    use_short: bool
    d_enabled: bool = False
    benchmark: str = "SPY"
    regime_mode: str = "prod"
    exposure_mode: str = "scaled"
    neutral_policy: str = "long"
    bull_long_mult: float = 1.0
    neutral_long_mult: float = 0.50
    bear_short_mult: float = 0.60
    neutral_short_mult: float = 0.0



@dataclass
class AppConfig:
    output_dir: str
    benchmark: str
    timezone: str


@dataclass
class FullConfig:
    app: AppConfig
    universe: UniverseConfig
    intraday: IntradayConfig
    regime: RegimeConfig


def load_yaml_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_full_config(cfg: Dict) -> FullConfig:
    reporting = cfg.get("reporting", {})
    app = cfg.get("app", {})
    universe = cfg.get("universe", {})
    intraday = cfg.get("intraday", {})

    # Intraday knobs can be stored in either:
    #   intraday.prod  (new explicit prod block)
    #   intraday.long  (your current config style)
    #   intraday       (legacy flat style)
    # Merge them so changing config.yaml actually affects PROD.
    intraday_prod = dict(intraday)
    if isinstance(intraday.get("long"), dict):
        intraday_prod.update(intraday.get("long", {}))
    if isinstance(intraday.get("prod"), dict):
        intraday_prod.update(intraday.get("prod", {}))

    backtest = cfg.get("backtest", {})
    regime_bt = backtest.get("regime", {})
    intraday_regime = intraday.get("regime", {})

    # Regime: intraday-specific override, else fall back to backtest.regime
    regime_use_long = bool(intraday_regime.get("use_long", regime_bt.get("use_long", True)))
    regime_use_short = bool(intraday_regime.get("use_short", regime_bt.get("use_short", False)))

    # Breadth knobs for intraday:
    #  - primary source: intraday.breadth (or intraday.prod.breadth)
    #  - fallback:       backtest.breadth
    intraday_breadth = intraday_prod.get("breadth", {})
    backtest_breadth = backtest.get("breadth", {})
    breadth_enabled = bool(
        intraday_breadth.get(
            "enabled",
            backtest_breadth.get("enabled", False),
        )
    )
    breadth_ma = int(
        intraday_breadth.get(
            "ma_window",
            backtest_breadth.get("ma_window", 50),
        )
    )
    breadth_min_long = float(
        intraday_breadth.get(
            "min_long",
            backtest_breadth.get("min_long", 0.60),
        )
    )

    # ADX thresholds: intraday override; else from backtest.long / short; else fallback 18
    backtest_long = backtest.get("long", {})
    adx_min_long = float(
        intraday_prod.get(
            "adx_min_long",
            backtest_long.get("adx_min_long", 18.0),
        )
    )

    app_cfg = AppConfig(
        output_dir=reporting.get("output_dir", "./output"),
        benchmark=app.get("benchmark", "SPY"),
        timezone=app.get("timezone", "America/Chicago"),
    )
    u_cfg = UniverseConfig(
        min_price=float(universe.get("min_price", 5.0)),
        min_avg_volume=int(universe.get("min_avg_volume", 1_000_000)),
    )
    i_cfg = IntradayConfig(
        vol_pace_min=float(intraday_prod.get("vol_pace_min", 1.3)),
        near_vol_pace_min=float(intraday_prod.get("near_vol_pace_min", 1.0)),
        sell_intrabar_vol_pace_min=float(intraday_prod.get("sell_intrabar_vol_pace_min", 1.2)),
        confirm_headroom_pct=float(intraday_prod.get("confirm_headroom_pct", 0.4)),
        near_below_pivot_pct=float(intraday_prod.get("near_below_pivot_pct", 0.3)),
        crack_ma_pct=float(intraday_prod.get("crack_ma_pct", 0.5)),
        min_elapsed_minutes=int(intraday_prod.get("min_elapsed_minutes", 40)),
        ma_proxy_length=int(intraday_prod.get("ma_proxy_length", 150)),
        adx_min_long=adx_min_long,
        daily_history_period=str(intraday_prod.get("daily_history_period", "18mo")),
        stage_above_ma_pct=float(intraday_prod.get("stage_above_ma_pct", 0.005)),
        dist_above_ma_min=float(intraday_prod.get("dist_above_ma_min", 0.005)),
        pivot_lookback_days=int(intraday_prod.get("pivot_lookback_days", 60)),
        breadth_enabled=breadth_enabled,
        breadth_ma_window=breadth_ma,
        breadth_min_long=breadth_min_long,
    )
    # Shared D regime/exposure config. Intraday can override; otherwise defaults are safe/off.
    d_intraday = read_d_config(cfg, section="intraday")
    r_cfg = RegimeConfig(
        use_long=regime_use_long,
        use_short=regime_use_short,
        d_enabled=bool(d_intraday.get("enabled", False)),
        benchmark=str(d_intraday.get("benchmark", app.get("benchmark", "SPY"))),
        regime_mode=str(d_intraday.get("regime_mode", "prod")),
        exposure_mode=str(d_intraday.get("exposure_mode", "scaled")),
        neutral_policy=str(d_intraday.get("neutral_policy", "long")),
        bull_long_mult=float(d_intraday.get("bull_long_mult", 1.0)),
        neutral_long_mult=float(d_intraday.get("neutral_long_mult", 0.50)),
        bear_short_mult=float(d_intraday.get("bear_short_mult", 0.60)),
        neutral_short_mult=float(d_intraday.get("neutral_short_mult", 0.0)),
    )
    return FullConfig(
        app=app_cfg,
        universe=u_cfg,
        intraday=i_cfg,
        regime=r_cfg,
    )


# ---------------------------------------------------------------------------
# Weekly CSV loader (robust against header / Stage changes)
# ---------------------------------------------------------------------------

def find_latest_weekly_csv(output_dir: str) -> str:
    pattern = os.path.join(output_dir, "weinstein_weekly_equities_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No weekly CSVs found under {pattern}")
    latest = max(files, key=os.path.getmtime)
    return latest


def normalize_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the weekly equities CSV columns so that we *always* have:
        - 'Ticker'
        - 'Close'
        - 'Stage' (numeric 1..4, default 2)
        - 'AvgVolume' (if available)
    """
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

    # Close / Price
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

    # Average volume
    avgv_col = None
    for key in ["avgvolume", "avg_volume", "avg_vol", "vol_avg", "volume_ma50"]:
        if key in lower:
            avgv_col = lower[key]
            break
    if avgv_col is None and "volume" in lower:
        avgv_col = lower["volume"]

    if avgv_col is not None and avgv_col != "AvgVolume":
        df.rename(columns={avgv_col: "AvgVolume"}, inplace=True)
    elif "AvgVolume" not in df.columns:
        df["AvgVolume"] = np.nan

    return df


def load_focus_universe(weekly_csv: str, u_cfg: UniverseConfig) -> pd.DataFrame:
    df = pd.read_csv(weekly_csv)
    df = normalize_weekly_columns(df)

    # Primary Stage filter: 1/2 for long candidates
    if "Stage" in df.columns:
        long_mask = df["Stage"].isin([1, 2])
        if long_mask.sum() == 0:
            log("⚠️ No Stage 1/2 rows found — falling back to full weekly universe.")
        else:
            df = df.loc[long_mask]

    # Filter by price
    df = df.loc[df["Close"] >= u_cfg.min_price]

    # Filter by average volume if present
    if "AvgVolume" in df.columns and df["AvgVolume"].notna().any():
        df = df.loc[df["AvgVolume"] >= u_cfg.min_avg_volume]

    df = df.dropna(subset=["Ticker"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["Ticker"])

    return df



def augment_universe_with_holdings(focus_df: pd.DataFrame, holdings_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure every current portfolio holding is evaluated by the intraday watcher.

    The normal weekly universe is optimized for BUY discovery, so it can exclude:
      - low-liquidity holdings
      - recent/small-cap holdings
      - tickers that fail Stage 1/2 or min-volume filters

    That is fine for BUY scans, but dangerous for SELL/risk monitoring.
    For owned positions, we must still fetch daily/intraday data and evaluate
    MA150/exit risk even if the symbol is not in the buy universe.
    """
    if focus_df is None:
        focus_df = pd.DataFrame(columns=["Ticker"])
    out = focus_df.copy()
    if "Ticker" not in out.columns:
        out["Ticker"] = []
    out["Ticker"] = out["Ticker"].map(_clean_symbol)
    out = out[out["Ticker"].astype(str).str.len() > 0].copy()
    out["PortfolioOwned"] = False

    added: List[str] = []
    if holdings_df is not None and not holdings_df.empty and "Ticker" in holdings_df.columns:
        owned = (
            holdings_df["Ticker"]
            .map(_clean_symbol)
            .dropna()
            .astype(str)
            .str.strip()
        )
        owned = sorted({x for x in owned if x and x not in INVALID_HOLDING_SYMBOLS})
        existing = set(out["Ticker"].astype(str))
        missing = [x for x in owned if x not in existing]

        if missing:
            add_df = pd.DataFrame({"Ticker": missing})
            # Stage is unknown because these did not come from the weekly buy universe.
            # The evaluation loop will compute current MA150/Stage-like risk from price data.
            add_df["Stage"] = np.nan
            add_df["PortfolioOwned"] = True
            out = pd.concat([out, add_df], ignore_index=True, sort=False)
            added = missing

        out.loc[out["Ticker"].isin(owned), "PortfolioOwned"] = True

    out = out.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
    return out, added

# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Basic ADX(14) implementation; returns series aligned to 'close'.
    """
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


def compute_breadth(weekly_df: pd.DataFrame, ma_window: int = 50) -> float:
    """
    Very simple breadth: % of tickers whose Close is above its Close MA(ma_window)
    using the weekly Close as a proxy (not perfect, but stable).
    In practice, this is only used as a gating knob; if disabled, we return 100.
    """
    if weekly_df.empty or "Close" not in weekly_df.columns:
        return 100.0
    # Crude proxy: treat half of universe as "above MA" to avoid overfitting.
    above = int(len(weekly_df) * 0.5)
    return above / max(len(weekly_df), 1) * 100.0


# ---------------------------------------------------------------------------
# Intraday scanning
# ---------------------------------------------------------------------------

def fetch_price_data(tickers: List[str], daily_history_period: str = "18mo", benchmark: str = "SPY") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download:
        - daily OHLCV (configurable history period) for pivots / MAs
        - 60m bars (60 days) for intraday signals
    Returns:
        daily:  multi-index (Date, Ticker)
        intraday: multi-index (DateTime, Ticker)
    """
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    download_tickers = sorted(set(list(tickers) + ([benchmark] if benchmark else [])))
    tickers_str = " ".join(download_tickers)

    # Daily
    daily = yf.download(
        tickers_str,
        period=daily_history_period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    # Intraday 60m
    intraday = yf.download(
        tickers_str,
        period="60d",
        interval="60m",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    # Normalize into a consistent multi-index: (date/time, ticker)
    def stack_yf(df_raw: pd.DataFrame) -> pd.DataFrame:
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
            # Single ticker
            out = df_raw.copy()
            out["Ticker"] = download_tickers[0]
        out.reset_index(inplace=True)
        date_col = "Date" if "Date" in out.columns else "Datetime"
        out = out.rename(columns={date_col: "Date"})
        out = out.set_index(["Date", "Ticker"])
        return out

    daily_stacked = stack_yf(daily)
    intraday_stacked = stack_yf(intraday)

    return daily_stacked, intraday_stacked


def evaluate_intraday_signals(
    focus_df: pd.DataFrame,
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    cfg: FullConfig,
    regime_decision=None,
    owned_tickers=None,
) -> pd.DataFrame:
    """
    For each ticker:
        - compute ADX(14) on daily
        - compute 30/150d MAs on daily
        - define "pivot" as 50d high close
        - check latest 60m bar vs pivot + headroom
        - compute intraday volume pace vs 50d daily volume
        - produce BUY / NEAR / none

    Returns diagnostics DataFrame with one row per ticker.
    """
    rows = []
    owned_set = {str(t).upper().strip() for t in (owned_tickers or []) if str(t).strip()}
    regime_label = getattr(regime_decision, "regime_label", "LEGACY") if regime_decision is not None else "LEGACY"
    allow_new_longs = bool(getattr(regime_decision, "allow_new_longs", True)) if regime_decision is not None else True
    long_exposure_mult = float(getattr(regime_decision, "long_size_mult", 1.0)) if regime_decision is not None else 1.0
    short_exposure_mult = float(getattr(regime_decision, "short_size_mult", 1.0)) if regime_decision is not None else 1.0

    if focus_df.empty:
        return pd.DataFrame()

    tickers = focus_df["Ticker"].tolist()
    focus_by_ticker = focus_df.set_index("Ticker").to_dict(orient="index") if "Ticker" in focus_df.columns else {}

    def _stage_num_for(t: str) -> int:
        try:
            raw = focus_by_ticker.get(t, {}).get("Stage", 2)
            return int(float(raw))
        except Exception:
            return 2

    def _stage_structure(stage_num: int) -> str:
        labels = {
            1: "Stage 1 (Base)",
            2: "Stage 2 (Uptrend)",
            3: "Stage 3 (Topping)",
            4: "Stage 4 (Downtrend)",
        }
        return labels.get(stage_num, f"Stage {stage_num}")

    for ticker in tickers:
        stage_num = _stage_num_for(ticker)
        structure = _stage_structure(stage_num)
        try:
            d = daily.xs(ticker, level="Ticker")
        except KeyError:
            continue

        d = d.sort_index()
        if d.empty:
            continue

        needed = ["High", "Low", "Close", "Volume"]
        if not all(c in d.columns for c in needed):
            continue

        # Daily indicators
        d["MA30"] = d["Close"].rolling(30).mean()
        d["MA150"] = d["Close"].rolling(cfg.intraday.ma_proxy_length).mean()
        d["ATR14"] = (d["High"] - d["Low"]).rolling(14).mean()
        d["VolMA50"] = d["Volume"].rolling(50).mean()
        d["ADX14"] = compute_adx(d["High"], d["Low"], d["Close"], period=14)

        last = d.iloc[-1]
        adx14 = float(last["ADX14"]) if not pd.isna(last["ADX14"]) else np.nan

        # ADX is evaluated by shared CORE below so PROD and SIM use one gate.

        # Stage-like filter: price above MA150 and MA150 rising
        # Compute pivot early so SKIP-STAGE portfolio rows can still show price/pivot context.
        pivot_window = d["Close"].tail(cfg.intraday.pivot_lookback_days).max()
        close_daily = float(last["Close"]) if not pd.isna(last["Close"]) else np.nan
        ma150_val = float(last["MA150"]) if not pd.isna(last["MA150"]) else np.nan
        atr14_val = float(last["ATR14"]) if not pd.isna(last["ATR14"]) else np.nan
        vol_ma50_val = float(last["VolMA50"]) if not pd.isna(last["VolMA50"]) else np.nan
        vol_pace_daily = (float(last["Volume"]) / vol_ma50_val) if vol_ma50_val and not np.isnan(vol_ma50_val) else np.nan
        headroom_daily = ((close_daily / float(pivot_window)) - 1.0) * 100.0 if (not pd.isna(pivot_window) and pivot_window > 0 and not np.isnan(close_daily)) else np.nan

        is_owned = str(ticker).upper() in owned_set

        # Portfolio SELL/risk monitoring must run even when an owned ticker is
        # not part of the normal BUY universe.  This mirrors the SIM CORE exit
        # guard: long positions are at risk when price breaks below the MA150
        # guard zone.  The intraday config crack_ma_pct is a stricter PROD alert
        # threshold for "SELL / reduce review".
        if is_owned and not pd.isna(last["MA150"]) and not np.isnan(close_daily):
            sell_break_pct = float(cfg.intraday.crack_ma_pct) / 100.0
            core_exit = should_exit_long(close_daily, np.nan, ma150_val)
            hard_sell = bool(close_daily <= ma150_val * (1.0 - sell_break_pct))
            if hard_sell or core_exit:
                signal = "SELL" if hard_sell else "SELL-WATCH"
                reason = (
                    f"Owned position exit risk: close={close_daily:.2f}, "
                    f"MA150={ma150_val:.2f}, crack_ma={cfg.intraday.crack_ma_pct:.2f}%, "
                    f"core_exit={core_exit}"
                )
                rows.append(
                    dict(
                        Ticker=ticker,
                        Structure="Below/weak vs MA150",
                        Stage=stage_num,
                        Signal=signal,
                        Reason=reason,
                        RegimeLabel=regime_label,
                        LongExposureMult=long_exposure_mult,
                        ShortExposureMult=short_exposure_mult,
                        SuggestedLongSizePct=0.0,
                        PriceNow=close_daily,
                        Pivot=pivot_window,
                        HeadroomPct=headroom_daily,
                        VolPace=vol_pace_daily,
                        ADX14=adx14,
                        CloseDaily=close_daily,
                        MA30=float(last["MA30"]) if not pd.isna(last["MA30"]) else np.nan,
                        MA150=ma150_val,
                        ATR14=atr14_val,
                        ticker=ticker,
                        structure="Below/weak vs MA150",
                        stage=stage_num,
                        signal=signal,
                        reason=reason,
                        price=close_daily,
                        pivot=pivot_window,
                        ma30=ma150_val,
                        pace_full_vs50dma=vol_pace_daily,
                        dist_bps=headroom_daily * 100.0 if not np.isnan(headroom_daily) else np.nan,
                        elapsed_min=cfg.intraday.min_elapsed_minutes,
                        pace_intrabar=cfg.intraday.sell_intrabar_vol_pace_min,
                        cond_weekly_stage_ok=False,
                        cond_rs_ok=False,
                        cond_ma_ok=False,
                        cond_pivot_ok=bool(not pd.isna(pivot_window) and float(pivot_window) > 0),
                        cond_buy_vol_ok=False,
                        cond_pace_full_gate=False,
                        cond_near_pace_gate=False,
                        cond_buy_price_ok=False,
                        cond_near_now=False,
                        buy_confirm=False,
                        portfolio_owned=True,
                    )
                )
                continue

        if pd.isna(last["MA150"]):
            rows.append(
                dict(
                    Ticker=ticker,
                    Structure="Unknown / insufficient MA history",
                    Stage=stage_num,
                    Signal="SKIP-MA",
                    Reason="MA150 not available",
                    PriceNow=close_daily,
                    Pivot=pivot_window,
                    HeadroomPct=headroom_daily,
                    VolPace=vol_pace_daily,
                    ADX14=adx14,
                    CloseDaily=close_daily,
                    MA30=float(last["MA30"]) if not pd.isna(last["MA30"]) else np.nan,
                    MA150=ma150_val,
                    ATR14=atr14_val,
                )
            )
            continue

        if not (last["Close"] > last["MA150"] * (1.0 + cfg.intraday.stage_above_ma_pct)):
            rows.append(
                dict(
                    Ticker=ticker,
                    Structure="Not Stage 2 / below MA150",
                    Stage=stage_num,
                    Signal="SKIP-STAGE",
                    Reason=(
                        f"Close not sufficiently above MA150 +{cfg.intraday.stage_above_ma_pct * 100:.2f}% "
                        f"({last['Close']:.2f} vs {last['MA150']:.2f})"
                    ),
                    PriceNow=close_daily,
                    Pivot=pivot_window,
                    HeadroomPct=headroom_daily,
                    VolPace=vol_pace_daily,
                    ADX14=adx14,
                    CloseDaily=close_daily,
                    MA30=float(last["MA30"]) if not pd.isna(last["MA30"]) else np.nan,
                    MA150=ma150_val,
                    ATR14=atr14_val,
                )
            )
            continue

        # Pivot = recent high close over configured lookback.
        if pd.isna(pivot_window):
            rows.append(
                dict(
                    Ticker=ticker,
                    Structure=structure,
                    Stage=stage_num,
                    Signal="SKIP-PIVOT",
                    Reason="No 50d pivot",
                )
            )
            continue

        # Intraday
        try:
            intr = intraday.xs(ticker, level="Ticker").sort_index()
        except KeyError:
            rows.append(
                dict(
                    Ticker=ticker,
                    Structure=structure,
                    Stage=stage_num,
                    Signal="SKIP-INTRADAY",
                    Reason="No intraday data",
                )
            )
            continue

        if intr.empty:
            rows.append(
                dict(
                    Ticker=ticker,
                    Structure=structure,
                    Stage=stage_num,
                    Signal="SKIP-INTRADAY",
                    Reason="Empty intraday series",
                )
            )
            continue

        last_bar = intr.iloc[-1]
        price_now = float(last_bar.get("Close", np.nan))
        vol_now = float(last_bar.get("Volume", np.nan))
        vol_ma50 = float(last["VolMA50"]) if not pd.isna(last["VolMA50"]) else np.nan

        if np.isnan(price_now) or np.isnan(vol_now) or np.isnan(vol_ma50) or vol_ma50 <= 0:
            rows.append(
                dict(
                    Ticker=ticker,
                    Structure=structure,
                    Stage=stage_num,
                    Signal="SKIP-DATA",
                    Reason="Missing price/vol/VolMA50",
                )
            )
            continue

        vol_pace = vol_now / vol_ma50
        headroom_pct = (price_now / pivot_window - 1.0) * 100.0

        # ------------------------------------------------------------------
        # WATCH LAYER VISIBILITY
        # ------------------------------------------------------------------
        # This is an observability layer only.  It does NOT convert rows into
        # BUY/NEAR/SELL and it does NOT loosen production trade signals.
        #
        # Goal: expose why PROD is quiet during the day by labeling useful
        # Stage-2/pivot contexts even when strict participation gates fail.
        # Examples:
        #   WATCH_BREAKOUT_PRICE  = price is over pivot, but strict BUY failed
        #   WATCH_NEAR_PIVOT      = price is close to pivot, but not actionable
        #   WATCH_LOW_VOLUME      = price context is interesting, but vol pace is weak
        watch_signal = ""
        watch_reason = ""
        watch_price_ok = False
        watch_volume_ok = False
        watch_near_zone_pct = float(cfg.intraday.near_below_pivot_pct) / 100.0
        watch_price_ok = bool(price_now >= float(pivot_window) * (1.0 - watch_near_zone_pct))
        watch_volume_ok = bool((not np.isnan(vol_pace)) and vol_pace >= cfg.intraday.near_vol_pace_min)

        if watch_price_ok:
            if price_now >= float(pivot_window):
                if watch_volume_ok:
                    watch_signal = "WATCH_BREAKOUT_PRICE"
                    watch_reason = (
                        f"Watch: price is over pivot but strict BUY not confirmed yet; "
                        f"headroom={headroom_pct:.2f}%, vol={vol_pace:.2f}x"
                    )
                else:
                    watch_signal = "WATCH_LOW_VOLUME"
                    watch_reason = (
                        f"Watch: price is over pivot but volume pace is weak; "
                        f"headroom={headroom_pct:.2f}%, vol={vol_pace:.2f}x "
                        f"< near_req={cfg.intraday.near_vol_pace_min:.2f}x"
                    )
            else:
                if watch_volume_ok:
                    watch_signal = "WATCH_NEAR_PIVOT"
                    watch_reason = (
                        f"Watch: price is within {cfg.intraday.near_below_pivot_pct:.2f}% of pivot; "
                        f"headroom={headroom_pct:.2f}%, vol={vol_pace:.2f}x"
                    )
                else:
                    watch_signal = "WATCH_LOW_VOLUME"
                    watch_reason = (
                        f"Watch: price is near pivot but volume pace is weak; "
                        f"headroom={headroom_pct:.2f}%, vol={vol_pace:.2f}x "
                        f"< near_req={cfg.intraday.near_vol_pace_min:.2f}x"
                    )

        # BUY vs NEAR logic is delegated to CORE so PROD and SIM stay aligned.
        core_params = LongEntryParams(
            min_break_pct=cfg.intraday.confirm_headroom_pct / 100.0,
            dist_above_ma_min=cfg.intraday.dist_above_ma_min,
            vol_min=cfg.intraday.vol_pace_min,
            adx_min=cfg.intraday.adx_min_long,
        )
        core_result = evaluate_long_signal(
            price=price_now,
            ma_val=float(last["MA150"]),
            pivot=float(pivot_window),
            rs_above_ma=True,  # weekly universe already handled RS/stage quality upstream
            vol_mult=vol_pace,
            adx_val=adx14,
            params=core_params,
            near_below_pivot_pct=cfg.intraday.near_below_pivot_pct / 100.0,
            near_vol_min=cfg.intraday.near_vol_pace_min,
        )
        signal = core_result.signal
        reason = core_result.reason

        # Shared D CORE regime gate for PROD. If D is enabled and the current
        # regime does not allow new longs, do not emit BUY/NEAR alerts.
        if cfg.regime.d_enabled and not allow_new_longs:
            signal = "SKIP-REGIME"
            reason = (
                f"D regime/exposure CORE blocks new longs: regime={regime_label}, "
                f"long_mult={long_exposure_mult:.2f}, short_mult={short_exposure_mult:.2f}"
            )

        # ------------------------------------------------------------------
        # HYBRID NEAR layer
        # ------------------------------------------------------------------
        # BUY remains strict and CORE-driven.
        # NEAR is intentionally softer, closer to the original Nov/Dec scanner:
        #   - Stage/MA structure must already be OK because we reached this point
        #   - ticker must be close to the pivot OR already over pivot but not BUY-confirmed
        #   - near volume uses near_vol_pace_min, not the stricter BUY vol_pace_min
        #   - ADX does NOT block NEAR; ADX only blocks strict BUY
        # This restores the useful "watchlist forming" behavior without making
        # production BUY alerts noisy.
        near_zone_pct = cfg.intraday.near_below_pivot_pct / 100.0
        soft_near_price_ok = price_now >= float(pivot_window) * (1.0 - near_zone_pct)
        soft_near_vol_ok = (not np.isnan(vol_pace)) and vol_pace >= cfg.intraday.near_vol_pace_min
        soft_near_now = bool(signal not in ("BUY", "SKIP-REGIME") and soft_near_price_ok and soft_near_vol_ok)

        if soft_near_now:
            signal = "NEAR"
            if price_now >= float(pivot_window):
                reason = (
                    f"NEAR: px={price_now:.2f} is over pivot={float(pivot_window):.2f} "
                    f"but not fully BUY-confirmed; headroom={headroom_pct:.2f}%, "
                    f"vol={vol_pace:.2f}x, adx={adx14:.1f}"
                )
            else:
                reason = (
                    f"NEAR: px={price_now:.2f} within "
                    f"{cfg.intraday.near_below_pivot_pct:.2f}% of pivot={float(pivot_window):.2f}; "
                    f"headroom={headroom_pct:.2f}%, vol={vol_pace:.2f}x, adx={adx14:.1f}"
                )

        buy_confirm = bool(signal == "BUY")
        cond_near_now = bool(signal == "NEAR")
        cond_buy_price_ok = bool(price_now >= float(pivot_window) * (1.0 + cfg.intraday.confirm_headroom_pct / 100.0))
        cond_buy_vol_ok = bool((not np.isnan(vol_pace)) and vol_pace >= cfg.intraday.vol_pace_min)
        cond_near_pace_gate = bool((not np.isnan(vol_pace)) and vol_pace >= cfg.intraday.near_vol_pace_min)
        cond_ma_ok = bool(price_now > float(last["MA150"]) * (1.0 + cfg.intraday.dist_above_ma_min))
        cond_weekly_stage_ok = True
        cond_rs_ok = True
        cond_pivot_ok = bool(not pd.isna(pivot_window) and float(pivot_window) > 0)
        cond_pace_full_gate = cond_buy_vol_ok

        rows.append(
            dict(
                Ticker=ticker,
                Structure=structure,
                Stage=stage_num,
                Signal=signal,
                Reason=reason,
                RegimeLabel=regime_label,
                LongExposureMult=long_exposure_mult,
                ShortExposureMult=short_exposure_mult,
                SuggestedLongSizePct=long_exposure_mult * 100.0,
                PriceNow=price_now,
                Pivot=pivot_window,
                HeadroomPct=headroom_pct,
                VolPace=vol_pace,
                ADX14=adx14,
                CloseDaily=float(last["Close"]),
                MA30=float(last["MA30"]) if not pd.isna(last["MA30"]) else np.nan,
                MA150=float(last["MA150"]),
                ATR14=float(last["ATR14"]) if not pd.isna(last["ATR14"]) else np.nan,
                # Lowercase compatibility columns consumed by tools/signal_engine.py
                ticker=ticker,
                structure=structure,
                stage=stage_num,
                signal=signal,
                reason=reason,
                price=price_now,
                pivot=pivot_window,
                ma30=float(last["MA150"]),
                pace_full_vs50dma=vol_pace,
                dist_bps=headroom_pct * 100.0,
                elapsed_min=cfg.intraday.min_elapsed_minutes,
                pace_intrabar=cfg.intraday.sell_intrabar_vol_pace_min,
                cond_weekly_stage_ok=cond_weekly_stage_ok,
                cond_rs_ok=cond_rs_ok,
                cond_ma_ok=cond_ma_ok,
                cond_pivot_ok=cond_pivot_ok,
                cond_buy_vol_ok=cond_buy_vol_ok,
                cond_pace_full_gate=cond_pace_full_gate,
                cond_near_pace_gate=cond_near_pace_gate,
                cond_buy_price_ok=cond_buy_price_ok,
                cond_near_now=cond_near_now,
                buy_confirm=buy_confirm,
                WatchSignal=watch_signal,
                WatchReason=watch_reason,
                WatchPriceOK=watch_price_ok,
                WatchVolumeOK=watch_volume_ok,
                watch_signal=watch_signal,
                watch_reason=watch_reason,
            )
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HTML / Email-style report rendering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Portfolio holdings helpers
# ---------------------------------------------------------------------------

INVALID_HOLDING_SYMBOLS = {
    "", "--", "N/A", "NA", "CASH", "CORE", "SPAXX", "FDRXX", "FCASH", "QIMHQ",
    "PENDINGACTIVITY", "PENDING", "MARGINCREDITBALANCE", "MMKT"
}


def _clean_symbol(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    s = s.replace("$", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("/", "-")
    return s


def _parse_money_like(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() == "nan" or s in ("--", "N/A"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan


def _first_present_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _flag_col_truthy(df: pd.DataFrame, col_name: str, default: bool = True) -> pd.Series:
    col = _first_present_col(df, [col_name])
    if not col:
        return pd.Series(default, index=df.index)
    return df[col].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})

def _flag_col_falsey(df: pd.DataFrame, col_name: str, default: bool = False) -> pd.Series:
    col = _first_present_col(df, [col_name])
    if not col:
        return pd.Series(default, index=df.index)
    return df[col].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})

def _is_yahoo_equity_symbol(sym: str) -> bool:
    s = _clean_symbol(sym)
    if not s:
        return False
    if s in INVALID_HOLDING_SYMBOLS:
        return False
    if s.endswith("-USD"):
        return False
    if s.startswith("NON"):
        return False
    if re.fullmatch(r"[0-9A-Z]{8,12}", s) and any(ch.isdigit() for ch in s):
        return False
    return bool(re.fullmatch(r"[A-Z]{1,5}([.-][A-Z])?", s))

def _normalize_holdings_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Return normalized holdings columns: Ticker, Quantity, Value, GainPct, Account.

    For Google Sheet Holdings exports, only rows explicitly enabled for Weinstein
    are eligible for PROD intraday SELL/risk coverage.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Ticker", "Quantity", "Value", "GainPct", "Account"])
    df = raw.copy()

    # Respect Trading Hub controls. This prevents 401k/CUSIP/cash/crypto rows
    # from being injected into the Yahoo/yfinance equity download universe.
    if _first_present_col(df, ["TradableForWeinstein"]) is not None:
        df = df.loc[_flag_col_truthy(df, "TradableForWeinstein", default=False)].copy()
    if _first_present_col(df, ["IsCash"]) is not None:
        df = df.loc[~_flag_col_falsey(df, "IsCash", default=False)].copy()
    if _first_present_col(df, ["IsPending"]) is not None:
        df = df.loc[~_flag_col_falsey(df, "IsPending", default=False)].copy()
    if _first_present_col(df, ["IsCrypto"]) is not None:
        df = df.loc[~_flag_col_falsey(df, "IsCrypto", default=False)].copy()

    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Quantity", "Value", "GainPct", "Account"])

    sym_col = _first_present_col(df, ["Ticker", "NormalizedSymbol", "Symbol", "symbol", "ticker"])
    if not sym_col:
        return pd.DataFrame(columns=["Ticker", "Quantity", "Value", "GainPct", "Account"])
    # Support common Fidelity/Google-Sheet/export header variants.
    qty_col = _first_present_col(df, [
        "Quantity", "Qty", "Shares", "Share Quantity", "Current Quantity", "Total Quantity",
        "Quantity Held", "Shares Held", "quantity", "qty"
    ])
    val_col = _first_present_col(df, [
        "Current Value", "Value", "Market Value", "MarketValue", "CurrentValue",
        "Position Value", "PositionValue", "Total Value", "Account Value", "value", "current_value"
    ])
    gain_col = _first_present_col(df, [
        "Total Gain/Loss Percent", "Total Gain/Loss %", "% Total Gain/Loss", "GainPct",
        "Gain %", "Gain/Loss %", "Percent Gain/Loss", "Unrealized %", "UnrealizedPct",
        "Unrealized Gain/Loss %", "Unrealized Gain/Loss Percent", "Return %"
    ])
    acct_col = _first_present_col(df, ["Account Name", "Account", "Account Number", "account"])
    out = pd.DataFrame()
    out["Ticker"] = df[sym_col].map(_clean_symbol)
    out["Quantity"] = df[qty_col].map(_parse_money_like) if qty_col else np.nan
    out["Value"] = df[val_col].map(_parse_money_like) if val_col else np.nan
    out["GainPct"] = df[gain_col].map(_parse_money_like) if gain_col else np.nan
    out["Account"] = df[acct_col].astype(str) if acct_col else ""
    out = out[~out["Ticker"].isin(INVALID_HOLDING_SYMBOLS)].copy()
    out = out[out["Ticker"].map(_is_yahoo_equity_symbol)].copy()
    if out.empty:
        return out
    # Use min_count=1 so blank quantity/value columns remain blank instead of becoming 0.00.
    grouped = out.groupby("Ticker", as_index=False).agg(
        Quantity=("Quantity", lambda s: s.sum(min_count=1)),
        Value=("Value", lambda s: s.sum(min_count=1)),
        GainPct=("GainPct", "mean"),
        Account=("Account", lambda s: ", ".join(sorted({x for x in s.astype(str) if x and x != "nan"}))[:80]),
    )
    return grouped.sort_values("Value", ascending=False, na_position="last")


def _read_gsheet_tab(cfg_raw: Dict, tab_name: str) -> Optional[pd.DataFrame]:
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception:
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials
        except Exception:
            return None
    try:
        sheets_cfg = cfg_raw.get("sheets", {}) or {}
        google_cfg = cfg_raw.get("google", {}) or {}
        sheet_url = sheets_cfg.get("url") or sheets_cfg.get("sheet_url")
        creds_json = google_cfg.get("service_account_json")
        if not sheet_url or not creds_json:
            return None
        creds_json = os.path.expanduser(str(creds_json))
        if not os.path.exists(creds_json):
            return None
        scope = ["https://www.googleapis.com/auth/spreadsheets.readonly", "https://www.googleapis.com/auth/drive.readonly"]
        try:
            creds = Credentials.from_service_account_file(creds_json, scopes=scope)
            gc = gspread.authorize(creds)
        except Exception:
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
            gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(tab_name)
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return None


def _holding_source_quality(df: pd.DataFrame) -> int:
    """Higher score = better portfolio metadata coverage."""
    if df is None or df.empty:
        return 0
    score = len(df) * 2
    for col in ["Quantity", "Value", "GainPct"]:
        if col in df.columns:
            score += int(df[col].notna().sum()) * 10
    return score


def _merge_holding_sources(sources: List[Tuple[pd.DataFrame, str]]) -> Tuple[pd.DataFrame, str]:
    """
    Merge portfolio metadata from multiple tabs/files.

    This intentionally uses a UNION of tickers across all available sources.
    Earlier versions used Open_Positions as the membership source and only enriched
    those tickers. That missed tickers that existed only in the broker Holdings /
    snapshot export, such as positions from a second account.
    """
    clean_sources = [(df.copy(), src) for df, src in sources if df is not None and not df.empty]
    if not clean_sources:
        return pd.DataFrame(columns=["Ticker", "Quantity", "Value", "GainPct", "Account"]), "no holdings source found"

    normalized_sources = []
    used_sources = []
    for df, src in clean_sources:
        df = df.copy()
        for col in ["Ticker", "Quantity", "Value", "GainPct", "Account"]:
            if col not in df.columns:
                df[col] = np.nan if col not in ("Ticker", "Account") else ""
        df = df[~df["Ticker"].isin(INVALID_HOLDING_SYMBOLS)].copy()
        df = df[df["Ticker"].map(_is_yahoo_equity_symbol)].copy()
        if not df.empty:
            normalized_sources.append(df[["Ticker", "Quantity", "Value", "GainPct", "Account"]])
            used_sources.append(src)

    if not normalized_sources:
        return pd.DataFrame(columns=["Ticker", "Quantity", "Value", "GainPct", "Account"]), "no holdings source found"

    all_rows = pd.concat(normalized_sources, ignore_index=True)

    def _first_nonblank(series):
        for x in series:
            if x is None:
                continue
            try:
                if pd.isna(x):
                    continue
            except Exception:
                pass
            if str(x).strip() in ("", "nan", "None", "—"):
                continue
            return x
        return np.nan

    grouped = all_rows.groupby("Ticker", as_index=False).agg(
        Quantity=("Quantity", lambda s: s.sum(min_count=1)),
        Value=("Value", lambda s: s.sum(min_count=1)),
        GainPct=("GainPct", _first_nonblank),
        Account=("Account", lambda s: ", ".join(sorted({x for x in s.astype(str) if x and x != "nan"}))[:80]),
    )
    grouped = grouped.sort_values("Value", ascending=False, na_position="last")
    return grouped, " + merged with ".join(used_sources)


def load_portfolio_holdings(cfg_raw: Dict, output_dir: str = "./output") -> Tuple[pd.DataFrame, str]:
    sheets_cfg = cfg_raw.get("sheets", {}) or {}
    open_tab = sheets_cfg.get("open_positions_tab", "Open_Positions")

    # Keep Open_Positions first for membership, then enrich from broker/snapshot tabs
    # that usually contain Current Value and Total Gain/Loss Percent.
    tab_candidates = []
    for tab in [open_tab, "Open_Positions_Snapshot", "Holdings", "OpenLots_Detail", "Portfolio", "Positions"]:
        if tab and tab not in tab_candidates:
            tab_candidates.append(tab)

    gsheet_sources: List[Tuple[pd.DataFrame, str]] = []
    for tab in tab_candidates:
        raw = _read_gsheet_tab(cfg_raw, tab)
        norm = _normalize_holdings_df(raw) if raw is not None else pd.DataFrame()
        if norm is not None and not norm.empty:
            gsheet_sources.append((norm, f"Google Sheet tab '{tab}'"))

    if gsheet_sources:
        merged, source = _merge_holding_sources(gsheet_sources)
        if not merged.empty:
            return merged, source

    patterns = [
        "Portfolio_Positions*.csv", "*Portfolio*Positions*.csv", "holdings*.csv", "Holdings*.csv",
        os.path.join(output_dir, "Portfolio_Positions*.csv"), os.path.join(output_dir, "holdings*.csv")
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)

    csv_sources: List[Tuple[pd.DataFrame, str]] = []
    for path in candidates:
        try:
            raw = pd.read_csv(path, engine="python")
            norm = _normalize_holdings_df(raw)
            if norm is not None and not norm.empty:
                csv_sources.append((norm, f"local CSV '{path}'"))
        except Exception:
            continue

    if csv_sources:
        merged, source = _merge_holding_sources(csv_sources)
        if not merged.empty:
            return merged, source

    return pd.DataFrame(columns=["Ticker", "Quantity", "Value", "GainPct", "Account"]), "no holdings source found"



# ---------------------------------------------------------------------------
# HTML color helpers
# ---------------------------------------------------------------------------

def _badge_class_for_action(value: object) -> str:
    v = str(value or '').lower()
    if 'sell' in v or 'reduce' in v or 'risk' in v:
        return 'badge-red'
    if 'review manually' in v or 'monitor' in v:
        return 'badge-gray'
    if 'needs volume' in v or 'watch closely' in v or 'near' in v:
        return 'badge-yellow'
    if 'add' in v or 'buy' in v or 'hold' in v:
        return 'badge-green'
    return 'badge-gray'

def _badge_class_for_signal(value: object) -> str:
    v = str(value or '').upper()
    if v in ('BUY', 'BUY_TRIGGER', 'BUY-TRIGGER'):
        return 'badge-green'
    if v in ('NEAR', 'NEAR_BUY', 'NEAR-TRIGGER'):
        return 'badge-yellow'
    if v in ('SELL', 'SELLTRIG', 'SELL-TRIGGER', 'SELL-WATCH'):
        return 'badge-red'
    if v.startswith('SKIP') or v == 'NOT-SCANNED':
        return 'badge-gray'
    return 'badge-blue'

def _badge_class_for_structure(value: object) -> str:
    v = str(value or '').lower()
    # IMPORTANT: check weak/non-recommended labels before checking "stage 2".
    # Otherwise "Not Stage 2 / below MA150" accidentally matches "stage 2"
    # and gets colored green.
    if 'not stage' in v or 'below ma150' in v or 'below/weak' in v or 'downtrend' in v:
        return 'badge-red'
    if 'stage 4' in v:
        return 'badge-red'
    if 'stage 2' in v:
        return 'badge-green'
    if 'stage 1' in v or 'stage 3' in v:
        return 'badge-yellow'
    return 'badge-gray'

def _badge(text: object, cls: str) -> str:
    val = '' if text is None else str(text)
    return f'<span class="badge {cls}">{html.escape(val)}</span>'

def _fmt_table_num(value: object, digits: int = 2, suffix: str = '') -> str:
    try:
        if value is None or pd.isna(value):
            return '—'
        return f'{float(value):,.{digits}f}{suffix}'
    except Exception:
        return '—'

def _fmt_colored_num(value: object, digits: int = 2, suffix: str = '') -> str:
    try:
        if value is None or pd.isna(value):
            return '—'
        v = float(value)
        cls = 'num-pos' if v > 0 else 'num-neg' if v < 0 else 'num-flat'
        return f'<span class="{cls}">{v:,.{digits}f}{suffix}</span>'
    except Exception:
        return '—'

def _colorize_portfolio_table(show: pd.DataFrame) -> pd.DataFrame:
    df = show.copy()
    if 'Recommendation' in df.columns:
        df['Recommendation'] = df['Recommendation'].map(lambda x: _badge(x, _badge_class_for_action(x)))
    if 'Signal' in df.columns:
        df['Signal'] = df['Signal'].map(lambda x: _badge(x, _badge_class_for_signal(x)))
    if 'Structure' in df.columns:
        df['Structure'] = df['Structure'].map(lambda x: _badge(x, _badge_class_for_structure(x)))
    for col, digits, suffix in [
        ('PriceNow', 2, ''), ('Pivot', 2, ''), ('Pivot Distance %', 2, '%'),
        ('Vol Pace', 2, '×'), ('ADX14', 1, ''), ('MA150', 2, ''),
        ('Quantity', 2, ''), ('Value', 2, '')
    ]:
        if col in df.columns:
            df[col] = df[col].map(lambda x, d=digits, s=suffix: _fmt_table_num(x, d, s))
    if 'Gain %' in df.columns:
        df['Gain %'] = df['Gain %'].map(lambda x: _fmt_colored_num(x, 2, '%'))
    return df

def _colorize_diag_table(show: pd.DataFrame) -> pd.DataFrame:
    df = show.copy()
    if 'Signal' in df.columns:
        df['Signal'] = df['Signal'].map(lambda x: _badge(x, _badge_class_for_signal(x)))
    if 'Structure' in df.columns:
        df['Structure'] = df['Structure'].map(lambda x: _badge(x, _badge_class_for_structure(x)))
    for col, digits, suffix in [
        ('PriceNow', 2, ''), ('Pivot', 2, ''), ('HeadroomPct', 2, '%'),
        ('VolPace', 2, '×'), ('ADX14', 1, ''), ('CloseDaily', 2, ''),
        ('MA150', 2, ''), ('ATR14', 2, '')
    ]:
        if col in df.columns:
            df[col] = df[col].map(lambda x, d=digits, s=suffix: _fmt_table_num(x, d, s))
    return df


def _sort_diag_for_report(diag: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the Diagnostics Table so the best/actionable rows appear first.

    Priority order:
      1) BUY confirmations
      2) NEAR triggers
      3) Stage 2 / valid structure with no trigger yet
         - closest to pivot first
         - then better volume pace
         - then stronger ADX
      4) data/ADX skips
      5) not Stage 2 / below MA150 and other weak structures
    """
    if diag is None or diag.empty:
        return pd.DataFrame() if diag is None else diag

    df = diag.copy()

    signal = df.get('Signal', pd.Series('', index=df.index)).astype(str).str.upper()
    structure = df.get('Structure', pd.Series('', index=df.index)).astype(str).str.lower()
    watch_signal = df.get('WatchSignal', pd.Series('', index=df.index)).astype(str).str.upper()

    priority = pd.Series(50, index=df.index, dtype=float)
    priority.loc[signal.eq('BUY')] = 0
    priority.loc[signal.isin(['NEAR', 'NEAR_BUY', 'NEAR-TRIGGER'])] = 1
    priority.loc[watch_signal.eq('WATCH_BREAKOUT_PRICE')] = 2
    priority.loc[watch_signal.eq('WATCH_NEAR_PIVOT')] = 3
    priority.loc[watch_signal.eq('WATCH_LOW_VOLUME')] = 4
    priority.loc[signal.eq('NONE') & structure.str.contains('stage 2', na=False)] = 5
    priority.loc[signal.eq('NONE')] = 3
    priority.loc[signal.eq('SKIP-ADX')] = 4
    priority.loc[signal.isin(['SKIP-DATA', 'SKIP-MA', 'SKIP-INTRADAY', 'SKIP-PIVOT'])] = 5
    priority.loc[signal.eq('SKIP-STAGE')] = 9
    priority.loc[structure.str.contains('not stage|below ma150|downtrend', na=False)] = 9

    headroom = pd.to_numeric(df.get('HeadroomPct'), errors='coerce')
    volpace = pd.to_numeric(df.get('VolPace'), errors='coerce')
    adx = pd.to_numeric(df.get('ADX14'), errors='coerce')
    price = pd.to_numeric(df.get('PriceNow'), errors='coerce')

    df['_report_priority'] = priority
    df['_abs_headroom'] = headroom.abs().fillna(9999)
    df['_volpace_sort'] = volpace.fillna(-1)
    df['_adx_sort'] = adx.fillna(-1)
    df['_price_sort'] = price.fillna(-1)

    df = df.sort_values(
        ['_report_priority', '_abs_headroom', '_volpace_sort', '_adx_sort', '_price_sort'],
        ascending=[True, True, False, False, False],
    )

    return df.drop(columns=['_report_priority', '_abs_headroom', '_volpace_sort', '_adx_sort', '_price_sort'], errors='ignore')

def _portfolio_action(row: pd.Series) -> Tuple[str, str]:
    signal = str(row.get("Signal", "")).upper()
    structure = str(row.get("Structure", ""))
    reason = str(row.get("Reason", ""))
    price = _parse_money_like(row.get("PriceNow"))
    ma150 = _parse_money_like(row.get("MA150"))
    headroom = _parse_money_like(row.get("HeadroomPct"))
    vol = _parse_money_like(row.get("VolPace"))
    if signal in ("SELL", "SELLTRIG", "SELL-TRIGGER", "SELL-WATCH"):
        return "SELL / reduce review", "Confirmed sell trigger from scanner."
    if signal.startswith("SKIP-DATA") or signal in ("NOT-SCANNED", "SKIP-INTRADAY", "SKIP-MA"):
        return "Review manually", "Owned ticker was not fully evaluated by the intraday scanner."
    if signal == "SKIP-STAGE" or (not np.isnan(price) and not np.isnan(ma150) and price < ma150):
        return "Risk review", "Owned ticker is below/weak versus MA150 structure."
    if signal == "BUY":
        return "Add/hold candidate", "Confirmed breakout while already owned."
    if signal in ("NEAR", "NEAR_BUY", "NEAR-TRIGGER"):
        return "Hold / watch closely", "Near-trigger while already owned; watch for breakout confirmation."
    if "Stage 2" in structure:
        if not np.isnan(headroom) and headroom >= -1.0:
            if not np.isnan(vol) and vol < 1.0:
                return "Hold / needs volume", "Stage 2 and near pivot, but volume pace is weak."
            return "Hold / near pivot", "Stage 2 and close to pivot, but no trigger yet."
        return "Hold / no trigger", "Stage 2 structure, but not close enough to pivot yet."
    return "Monitor", reason or "No specific action signal."


def build_portfolio_review_section(diag: pd.DataFrame, holdings: Optional[pd.DataFrame], holdings_source: str, limit: int = 40) -> str:
    html_parts = ["<hr>", "<h4>Portfolio Holdings Review</h4>"]
    if holdings is None or holdings.empty:
        html_parts.append(f"<p class=\"note\">No owned-position data found ({html.escape(str(holdings_source))}). To enable this section, keep the Open_Positions or Holdings tab populated, or place a Portfolio_Positions*.csv file in the repo/output folder.</p>")
        return "\n".join(html_parts)
    if diag is None or diag.empty or "Ticker" not in diag.columns:
        html_parts.append("<p class=\"note\">Holdings were found, but scanner diagnostics were empty.</p>")
        return "\n".join(html_parts)
    d = diag.copy(); d["Ticker"] = d["Ticker"].map(_clean_symbol)
    h = holdings.copy(); h["Ticker"] = h["Ticker"].map(_clean_symbol)
    merged = h.merge(d, on="Ticker", how="left", suffixes=("_Held", ""))
    merged["Signal"] = merged["Signal"].fillna("NOT-SCANNED")
    merged["Structure"] = merged["Structure"].fillna("Not in current scanner universe")
    merged["Reason"] = merged["Reason"].fillna("Ticker not present in current intraday diagnostics")
    actions = merged.apply(_portfolio_action, axis=1, result_type="expand")
    merged["PortfolioAction"] = actions[0]
    merged["PortfolioNote"] = actions[1]
    priority = {"SELL / reduce review": 0, "Risk review": 1, "Review manually": 2, "Hold / watch closely": 3, "Add/hold candidate": 4, "Hold / needs volume": 5, "Hold / near pivot": 6, "Hold / no trigger": 7, "Monitor": 8}
    merged["_priority"] = merged["PortfolioAction"].map(priority).fillna(9)
    merged["_value"] = pd.to_numeric(merged.get("Value"), errors="coerce").fillna(0)
    merged = merged.sort_values(["_priority", "_value"], ascending=[True, False])
    counts = merged["PortfolioAction"].value_counts().to_dict()
    html_parts.append(f"<p class=\"note\">Source: {html.escape(str(holdings_source))}. Reviewed {len(merged)} owned tickers against the current intraday scanner state.</p>")
    html_parts.append("<table class=\"summary\"><thead><tr><th>Portfolio Action</th><th>Count</th></tr></thead><tbody>")
    for action, count in sorted(counts.items(), key=lambda kv: priority.get(kv[0], 9)):
        cls = _badge_class_for_action(action)
        html_parts.append(f"<tr><td>{_badge(action, cls)}</td><td>{int(count)}</td></tr>")
    html_parts.append("</tbody></table>")
    display_cols = ["Ticker", "PortfolioAction", "PortfolioNote", "Structure", "Signal", "Reason", "PriceNow", "Pivot", "HeadroomPct", "VolPace", "ADX14", "MA150", "Quantity", "Value", "GainPct"]
    show = merged[[c for c in display_cols if c in merged.columns]].head(limit).copy()
    show = show.rename(columns={"PortfolioAction": "Recommendation", "PortfolioNote": "Portfolio Note", "HeadroomPct": "Pivot Distance %", "VolPace": "Vol Pace", "GainPct": "Gain %"})
    show = _colorize_portfolio_table(show)
    html_parts.append(show.to_html(index=False, escape=False, classes="portfolio-table"))
    if len(merged) > limit:
        html_parts.append(f"<p class=\"note\">Showing top {limit} of {len(merged)} owned tickers by action priority/value.</p>")
    html_parts.append("<p class=\"note\">Portfolio recommendations are rule-based scanner labels for review, not automatic orders.</p>")
    return "\n".join(html_parts)

def _fmt_num(value, digits: int = 2, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}{suffix}"
    except Exception:
        return "—"


def _stage_label(row: pd.Series) -> str:
    for col in ("Structure", "structure"):
        if col in row and not pd.isna(row.get(col)):
            val = str(row.get(col)).strip()
            if val and val.lower() != "nan":
                return val
    for col in ("Stage", "stage", "WeeklyStage", "weekly_stage"):
        if col in row and not pd.isna(row.get(col)):
            try:
                stage_num = int(float(row.get(col)))
                return {
                    1: "Stage 1 (Base)",
                    2: "Stage 2 (Uptrend)",
                    3: "Stage 3 (Topping)",
                    4: "Stage 4 (Downtrend)",
                }.get(stage_num, f"Stage {stage_num}")
            except Exception:
                return str(row.get(col))
    return "Stage 2 (Uptrend)"


def _rank_near(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_abs_headroom"] = pd.to_numeric(out.get("HeadroomPct"), errors="coerce").abs()
    out["_vol"] = pd.to_numeric(out.get("VolPace"), errors="coerce").fillna(0)
    out["_adx"] = pd.to_numeric(out.get("ADX14"), errors="coerce").fillna(0)
    return out.sort_values(["_abs_headroom", "_vol", "_adx"], ascending=[True, False, False])


def _rank_buy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_vol"] = pd.to_numeric(out.get("VolPace"), errors="coerce").fillna(0)
    out["_adx"] = pd.to_numeric(out.get("ADX14"), errors="coerce").fillna(0)
    out["_headroom"] = pd.to_numeric(out.get("HeadroomPct"), errors="coerce").fillna(0)
    return out.sort_values(["_vol", "_adx", "_headroom"], ascending=[False, False, False])


def _rank_sell(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_adx"] = pd.to_numeric(out.get("ADX14"), errors="coerce").fillna(0)
    return out.sort_values(["_adx"], ascending=[False])


def _signal_li(row: pd.Series, idx: int, kind: str) -> str:
    ticker = str(row.get("Ticker", row.get("ticker", ""))).upper()
    price = _fmt_num(row.get("PriceNow", row.get("price")), 2)
    pivot = _fmt_num(row.get("Pivot", row.get("pivot")), 2)
    headroom = _fmt_num(row.get("HeadroomPct"), 2, "%")
    vol = _fmt_num(row.get("VolPace", row.get("pace_full_vs50dma")), 2, "x")
    adx = _fmt_num(row.get("ADX14"), 1)
    stage = _stage_label(row)
    reason = str(row.get("Reason", row.get("reason", ""))).strip()

    if kind == "SELL":
        ma150 = _fmt_num(row.get("MA150", row.get("ma30")), 2)
        detail = f"SMA150 {ma150}, ADX {adx}, {stage}"
    else:
        detail = f"pivot {pivot}, distance {headroom}, vol {vol}, ADX {adx}, {stage}"

    reason_html = f"<br><span style=\"color:#666;font-size:12px;\">{reason}</span>" if reason else ""
    return f"<li><b>{idx}.</b> <b>{ticker}</b> @ {price} ({detail}){reason_html}</li>"


def _ordered_section(title: str, df: pd.DataFrame, kind: str, empty_text: str, limit: int = 25) -> str:
    if kind == "BUY":
        df = _rank_buy(df)
    elif kind == "SELL":
        df = _rank_sell(df)
    else:
        df = _rank_near(df)

    html = [f"<h4>{title}</h4>"]
    if df.empty:
        html.append(f"<p>{empty_text}</p>")
    else:
        html.append("<ol>")
        for i, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
            html.append(_signal_li(row, i, kind))
        html.append("</ol>")
        if len(df) > limit:
            html.append(f"<p style=\"font-size:12px;color:#777;\">Showing top {limit} of {len(df)} candidates.</p>")
    return "\n".join(html)




def _safe_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _chart_candidate_rows(buys: pd.DataFrame, nears: pd.DataFrame, sells: pd.DataFrame, max_charts: int = 15) -> List[Tuple[str, pd.Series]]:
    """
    Old-style chart selection:
      - BUY first
      - NEAR second
      - SELL third
    Only chart actionable sections, not the entire portfolio/diagnostics universe.
    """
    candidates: List[Tuple[str, pd.Series]] = []
    for label, df in [("BUY", _rank_buy(buys)), ("NEAR", _rank_near(nears)), ("SELL", _rank_sell(sells))]:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            candidates.append((label, row))
            if len(candidates) >= max_charts:
                return candidates
    return candidates


def _get_daily_for_ticker(daily: Optional[pd.DataFrame], ticker: str) -> pd.DataFrame:
    if daily is None or daily.empty or not ticker:
        return pd.DataFrame()
    try:
        if isinstance(daily.index, pd.MultiIndex) and "Ticker" in daily.index.names:
            return daily.xs(ticker, level="Ticker").sort_index().copy()
        if isinstance(daily.columns, pd.MultiIndex):
            if ticker in daily.columns.get_level_values(0):
                return daily[ticker].sort_index().copy()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _build_chart_base64(daily: Optional[pd.DataFrame], ticker: str, pivot: Optional[float] = None) -> Optional[str]:
    """
    Generate a compact old-style embedded PNG chart:
      - Close
      - MA30
      - MA150
      - Pivot line when available
    The chart is returned as base64 so the HTML can be opened directly or emailed.
    """
    try:
        d = _get_daily_for_ticker(daily, ticker)
        if d.empty or "Close" not in d.columns:
            return None

        d = d.tail(180).copy()
        close = pd.to_numeric(d["Close"], errors="coerce").dropna()
        if close.empty:
            return None

        ma30 = close.rolling(30).mean()
        ma150 = close.rolling(150).mean()

        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        ax.plot(close.index, close.values, linewidth=1.6, label="Close")
        ax.plot(ma30.index, ma30.values, linewidth=1.1, label="MA30")
        ax.plot(ma150.index, ma150.values, linewidth=1.2, label="MA150")

        pivot_val = _safe_float(pivot)
        if pivot_val and pivot_val > 0:
            ax.axhline(pivot_val, linestyle="--", linewidth=1.1, label=f"Pivot {pivot_val:.2f}")

        ax.set_title(f"{ticker} — Price + MA30 + MA150 + Pivot", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ chart generation failed for {ticker}: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def build_action_charts_section(
    buys: pd.DataFrame,
    nears: pd.DataFrame,
    sells: pd.DataFrame,
    daily: Optional[pd.DataFrame],
    max_charts: int = 15,
) -> str:
    parts = ["<hr>", "<h4>Charts (BUY / NEAR / SELL)</h4>"]
    candidates = _chart_candidate_rows(buys, nears, sells, max_charts=max_charts)
    if not candidates:
        parts.append("<p>No BUY / NEAR / SELL charts to display at this scan.</p>")
        return "\n".join(parts)

    section_colors = {
        "BUY": "#137333",
        "NEAR": "#8a5a00",
        "SELL": "#a50e0e",
    }

    for label, row in candidates:
        ticker = str(row.get("Ticker", row.get("ticker", ""))).strip().upper()
        if not ticker:
            continue
        pivot = row.get("Pivot", row.get("pivot"))
        img_b64 = _build_chart_base64(daily=daily, ticker=ticker, pivot=pivot)
        if not img_b64:
            continue
        price = _fmt_num(row.get("PriceNow", row.get("price")), 2)
        pivot_txt = _fmt_num(pivot, 2)
        headroom = _fmt_num(row.get("HeadroomPct"), 2, "%")
        vol = _fmt_num(row.get("VolPace", row.get("pace_full_vs50dma")), 2, "×")
        stage = html.escape(_stage_label(row))
        color = section_colors.get(label, "#333")
        parts.append(f"""
        <div style="display:block;margin:12px 0 18px 0;padding:12px;border:1px solid #ddd;border-radius:8px;background:#fff;">
          <div style="font-size:15px;font-weight:bold;margin-bottom:6px;color:{color};">
            {html.escape(label)}: {html.escape(ticker)} @ {price}
          </div>
          <div style="font-size:12px;color:#555;margin-bottom:8px;">
            Pivot {pivot_txt} | Distance {headroom} | Vol {vol} | {stage}
          </div>
          <img src="data:image/png;base64,{img_b64}" style="max-width:100%;height:auto;border-radius:6px;border:1px solid #eee;" />
        </div>
        """)
    if len(candidates) >= max_charts:
        parts.append(f"<p class=\"note\">Chart section limited to top {max_charts} actionable tickers.</p>")
    return "\n".join(parts)


def _watch_section(diag: pd.DataFrame, limit: int = 40) -> str:
    """Render diagnostic-only WATCH rows in the intraday HTML report.

    WATCH rows are intentionally NOT trading signals. They expose cases where
    price is close to/over pivot but strict BUY/NEAR gates are not confirmed,
    usually because volume pace or confirmation is weak.
    """
    parts = ["<h4>Watch-Layer Visibility (diagnostic only)</h4>"]

    if diag is None or diag.empty or "WatchSignal" not in diag.columns:
        parts.append("<p>No WATCH rows available in diagnostics.</p>")
        return "\n".join(parts)

    watch = diag.copy()
    watch["WatchSignal"] = watch["WatchSignal"].fillna("").astype(str).str.strip()
    watch = watch[watch["WatchSignal"].ne("")].copy()

    if watch.empty:
        parts.append("<p>No WATCH contexts at this scan.</p>")
        parts.append("<p class=\"note\">WATCH is observational only and does not change BUY / NEAR / SELL recommendations.</p>")
        return "\n".join(parts)

    counts = watch["WatchSignal"].value_counts().reset_index()
    counts.columns = ["WatchSignal", "Count"]
    counts["WatchSignal"] = counts["WatchSignal"].map(lambda x: _badge(x, _badge_class_for_signal(x)))

    parts.append("<p class=\"note\">WATCH rows are diagnostic-only. They show setups near/over pivot that failed strict confirmation, commonly because volume pace is weak.</p>")
    parts.append(counts.to_html(index=False, escape=False, classes="summary"))

    # Sort strongest watch contexts first: breakout price, near pivot, then low volume.
    priority = {
        "WATCH_BREAKOUT_PRICE": 0,
        "WATCH_NEAR_PIVOT": 1,
        "WATCH_LOW_VOLUME": 2,
    }
    watch["_watch_priority"] = watch["WatchSignal"].map(priority).fillna(9)
    watch["_abs_headroom"] = pd.to_numeric(watch.get("HeadroomPct"), errors="coerce").abs()
    watch["_volpace"] = pd.to_numeric(watch.get("VolPace"), errors="coerce")
    watch = watch.sort_values(["_watch_priority", "_abs_headroom", "_volpace"], ascending=[True, True, False])

    preferred_cols = [
        "Ticker", "Structure", "Signal", "Reason", "WatchSignal", "WatchReason",
        "PriceNow", "Pivot", "HeadroomPct", "VolPace", "ADX14",
        "cond_buy_price_ok", "cond_buy_vol_ok", "cond_near_pace_gate", "cond_near_now", "buy_confirm",
    ]
    cols = [c for c in preferred_cols if c in watch.columns]
    show = watch[cols].head(limit).copy()

    if "WatchSignal" in show.columns:
        show["WatchSignal"] = show["WatchSignal"].map(lambda x: _badge(x, _badge_class_for_signal(x)))
    if "Signal" in show.columns:
        show["Signal"] = show["Signal"].map(lambda x: _badge(x, _badge_class_for_signal(x)))
    if "Structure" in show.columns:
        show["Structure"] = show["Structure"].map(lambda x: _badge(x, _badge_class_for_structure(x)))

    for col, digits, suffix in [
        ("PriceNow", 2, ""),
        ("Pivot", 2, ""),
        ("HeadroomPct", 2, "%"),
        ("VolPace", 2, "×"),
        ("ADX14", 1, ""),
    ]:
        if col in show.columns:
            show[col] = show[col].map(lambda x, d=digits, s=suffix: _fmt_table_num(x, d, s))

    parts.append(show.to_html(index=False, escape=False, classes="diagnostics-table"))
    if len(watch) > limit:
        parts.append(f"<p class=\"note\">Showing top {limit} of {len(watch)} WATCH contexts.</p>")
    parts.append("<p class=\"note\">WATCH rows are for visibility only. Confirmed BUY/NEAR/SELL rules remain unchanged.</p>")
    return "\n".join(parts)

def build_intraday_report_html(
    diag: pd.DataFrame,
    cfg: FullConfig,
    ts_display: str,
    breadth_pct: float,
    breadth_long_ok: bool,
    long_ok: bool,
    holdings: Optional[pd.DataFrame] = None,
    holdings_source: str = "",
    daily: Optional[pd.DataFrame] = None,
) -> str:
    """Build the polished intraday HTML/email-style report."""
    if diag is None:
        diag = pd.DataFrame()

    sig = diag["Signal"].astype(str).str.upper() if not diag.empty and "Signal" in diag else pd.Series(dtype=str)
    buys = diag.loc[sig.eq("BUY")].copy() if not diag.empty else pd.DataFrame()
    nears = diag.loc[sig.isin(["NEAR", "NEAR_BUY", "NEAR-TRIGGER"])].copy() if not diag.empty else pd.DataFrame()
    sells = diag.loc[sig.isin(["SELL", "SELLTRIG", "SELL-TRIGGER"])].copy() if not diag.empty else pd.DataFrame()
    watch = diag.loc[diag.get("WatchSignal", pd.Series("", index=diag.index)).fillna("").astype(str).str.len().gt(0)].copy() if not diag.empty else pd.DataFrame()

    skip_stage = int(sig.eq("SKIP-STAGE").sum()) if not sig.empty else 0
    skip_adx = int(sig.eq("SKIP-ADX").sum()) if not sig.empty else 0
    none_count = int(sig.eq("NONE").sum()) if not sig.empty else 0
    skip_data = int(sig.isin(["SKIP-DATA", "SKIP-MA", "SKIP-INTRADAY"]).sum()) if not sig.empty else 0

    market_regime = "BULL" if long_ok else "LONG DISABLED"
    short_allowed = bool(cfg.regime.use_short and not long_ok)
    env_ok = bool(long_ok and breadth_long_ok)

    rules = f"""
    <h3>Weinstein Intraday Watch — {ts_display}</h3>
    <p><i>
      <b>BUY:</b> Weekly Stage 2 breakout confirmed above pivot and SMA150 (~30-week MA proxy),
      with price confirmation, RS support, volume pace ≥ {cfg.intraday.vol_pace_min:.2f}×,
      and ADX14 ≥ {cfg.intraday.adx_min_long:.1f} when available.<br><br>
      <b>NEAR-TRIGGER:</b> Structurally valid Stage 2 setup approaching pivot breakout,
      or initial pivot cross lacking full BUY confirmation. This is the early watchlist layer before confirmed BUY signals.<br><br>
      <b>SELL-TRIGGER:</b> Confirmed breakdown below SMA150 by {cfg.intraday.crack_ma_pct:.1f}% with persistence and downside confirmation.<br><br>
      <b>WATCH:</b> Observational visibility for Stage 2 / pivot contexts that are close to actionable but fail strict BUY/NEAR gates,
      usually because volume pace is too weak. WATCH rows do not trigger trade recommendations.
    </i></p>
    <p style="font-size:13px;color:#555;">
      <b>Market Regime (Chapter 8 filter):</b> {market_regime} — LONG allowed={bool(long_ok)}, SHORT allowed={short_allowed}.<br>
      <b>Breadth Health:</b> {breadth_pct:.1f}% of breadth universe above MA{cfg.intraday.breadth_ma_window}
      (breadth filter enabled={cfg.intraday.breadth_enabled}, LONG breadth_ok={bool(breadth_long_ok)}).<br>
      <b>Effective LONG gate:</b> env_long_ok = market_long_ok AND breadth_long_ok → {env_ok}.
    </p>
    """

    style = """
    <style>
      body { font-family: Arial, Helvetica, sans-serif; color:#222; }
      h3 { margin-bottom: 8px; }
      h4 { margin-top: 18px; margin-bottom: 8px; }
      li { margin: 6px 0; line-height: 1.35; }
      .summary, .portfolio-table, .dataframe { border-collapse: collapse; margin-top: 8px; width: 100%; }
      .summary th, .summary td, .portfolio-table th, .portfolio-table td, .dataframe th, .dataframe td { border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }
      .summary th, .portfolio-table th, .dataframe th { background: #f6f6f6; }
      .portfolio-table tr:nth-child(even), .dataframe tr:nth-child(even) { background: #fafafa; }
      .badge { display:inline-block; padding:3px 8px; border-radius:12px; font-size:12px; font-weight:700; white-space:nowrap; }
      .badge-green { background:#e6f4ea; color:#137333; border:1px solid #b7dfc2; }
      .badge-yellow { background:#fff7d6; color:#8a5a00; border:1px solid #f2d675; }
      .badge-red { background:#fce8e6; color:#a50e0e; border:1px solid #f5b5ae; }
      .badge-blue { background:#e8f0fe; color:#174ea6; border:1px solid #b8cdf8; }
      .badge-gray { background:#f1f3f4; color:#3c4043; border:1px solid #d9dce0; }
      .num-pos { color:#137333; font-weight:700; }
      .num-neg { color:#a50e0e; font-weight:700; }
      .num-flat { color:#3c4043; }
      .note { color:#666; font-size:12px; }
      hr { border:0; border-top:1px solid #ddd; margin:18px 0; }
    </style>
    """

    sections = [
        "<html><body>",
        style,
        rules,
        "<hr>",
        _ordered_section("Buy Triggers (ranked)", buys, "BUY", "No confirmed BUY breakouts at this scan."),
        "<hr>",
        _ordered_section("Near-Triggers (ranked)", nears, "NEAR", "No NEAR-TRIGGER setups at this scan."),
        "<hr>",
        _watch_section(diag),
        "<hr>",
        _ordered_section("Sell Triggers (ranked)", sells, "SELL", "No SELL-TRIGGER signals."),
        build_action_charts_section(buys, nears, sells, daily=daily, max_charts=15),
        build_portfolio_review_section(diag, holdings, holdings_source),
        "<hr>",
        "<h4>Scanner Diagnostics</h4>",
        "<table class=\"summary\"><thead><tr><th>Metric</th><th>Count</th></tr></thead><tbody>",
        f"<tr><td>Confirmed BUY</td><td>{len(buys)}</td></tr>",
        f"<tr><td>NEAR-TRIGGER</td><td>{len(nears)}</td></tr>",
        f"<tr><td>WATCH rows</td><td>{len(watch)}</td></tr>",
        f"<tr><td>SELL-TRIGGER</td><td>{len(sells)}</td></tr>",
        f"<tr><td>Stage 1/2 structure, no trigger yet</td><td>{none_count}</td></tr>",
        f"<tr><td>SKIP-STAGE</td><td>{skip_stage}</td></tr>",
        f"<tr><td>SKIP-ADX</td><td>{skip_adx}</td></tr>",
        f"<tr><td>SKIP-DATA / SKIP-MA / SKIP-INTRADAY</td><td>{skip_data}</td></tr>",
        "</tbody></table>",
        "<ul>",
        "<li><b>Structure</b> shows the Weinstein stage/context of the ticker.</li>",
        "<li><b>Signal</b> shows the trading action state: BUY, NEAR, SELL, NONE, or SKIP-*.</li>",
        "<li>NONE means the ticker has acceptable structure but has not produced a trigger yet.</li>",
        "<li>BUY remains strict: confirmed breakout plus participation confirmation.</li>",
        "<li>NEAR-TRIGGER is the early watchlist layer: structurally valid setups that may become actionable if volume expands.</li>",
        "<li>WATCH rows are observational only: they reveal close-to-pivot or low-volume contexts without becoming trading signals.</li>",
        "<li>Signals are filtered through the Chapter 8 market/regime model and optional breadth health gate.</li>",
        "</ul>",
        "<hr>",
        "<h4>Diagnostics Table</h4>",
    ]

    if diag.empty:
        sections.append("<p>No diagnostics rows generated.</p>")
    else:
        preferred_cols = ["Ticker", "Structure", "Signal", "Reason", "WatchSignal", "WatchReason", "PriceNow", "Pivot", "HeadroomPct", "VolPace", "ADX14", "CloseDaily", "MA150", "ATR14"]
        cols = [c for c in preferred_cols if c in diag.columns]
        # Sort before colorizing so the raw numeric values can still be ranked.
        diag_sorted = _sort_diag_for_report(diag)
        table_df = diag_sorted[cols].copy() if cols else diag_sorted.copy()
        table_df = _colorize_diag_table(table_df)
        sections.append(table_df.to_html(index=False, escape=False, classes="diagnostics-table"))

    sections.extend([
        "<p class=\"note\">Generated by Weinstein Hybrid Intraday Engine.</p>",
        "</body></html>",
    ])
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Weinstein Intraday Watcher (PROD)")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to config.yaml")
    parser.add_argument("--log-csv", type=str, default="./output/intraday_debug.csv", help="Diagnostics CSV output")
    parser.add_argument("--test-ease", action="store_true", help="Temporarily relax PROD thresholds for validation/tuning")
    args = parser.parse_args()

    try:
        cfg_raw = load_yaml_config(args.config)
        cfg = build_full_config(cfg_raw)
        if args.test_ease:
            # Validation mode only: useful to prove the pipeline can emit NEAR/BUY
            # without permanently changing config.yaml.
            cfg.intraday.adx_min_long = min(cfg.intraday.adx_min_long, 12.0)
            cfg.intraday.vol_pace_min = min(cfg.intraday.vol_pace_min, 0.85)
            cfg.intraday.near_vol_pace_min = min(cfg.intraday.near_vol_pace_min, 0.60)
            cfg.intraday.confirm_headroom_pct = min(cfg.intraday.confirm_headroom_pct, 0.05)
            cfg.intraday.near_below_pivot_pct = max(cfg.intraday.near_below_pivot_pct, 1.5)
            cfg.intraday.stage_above_ma_pct = min(cfg.intraday.stage_above_ma_pct, 0.0)
            cfg.intraday.dist_above_ma_min = min(cfg.intraday.dist_above_ma_min, 0.0)
            log("⚠️ TEST-EASE enabled: relaxed ADX/volume/pivot/MA thresholds for validation only.")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        raise SystemExit(1)

    log_step(f"Intraday watcher starting with config: {args.config}")

    try:
        weekly_csv = find_latest_weekly_csv(cfg.app.output_dir)
    except Exception as e:
        print(f"❌ Could not locate weekly CSV: {e}")
        raise SystemExit(1)

    log_step(f"·· Weekly CSV: {weekly_csv}")

    try:
        focus_df = load_focus_universe(weekly_csv, cfg.universe)
    except Exception as e:
        print(f"❌ Failed to load weekly universe: {e}")
        raise SystemExit(1)

    log(f"Focus universe: {len(focus_df)} symbols (Stage 1/2, price/volume filtered).")

    # Load holdings BEFORE data download so portfolio positions are always
    # scanned for SELL/risk, even if they are outside the BUY discovery universe.
    try:
        holdings_df, holdings_source = load_portfolio_holdings(cfg_raw, cfg.app.output_dir)
        if holdings_df.empty:
            log(f"Portfolio holdings pre-scan: no holdings loaded ({holdings_source}).")
        else:
            log(f"Portfolio holdings pre-scan: loaded {len(holdings_df)} owned tickers from {holdings_source}.")
    except Exception as e:
        holdings_df, holdings_source = pd.DataFrame(), f"holdings load error: {e}"
        log(f"Portfolio holdings pre-scan skipped: {e}")

    focus_df, added_holdings = augment_universe_with_holdings(focus_df, holdings_df)
    if added_holdings:
        log(
            "Portfolio SELL coverage: added "
            f"{len(added_holdings)} owned tickers outside weekly BUY universe: "
            + ", ".join(added_holdings[:20])
            + ("..." if len(added_holdings) > 20 else "")
        )

    if focus_df.empty:
        print("⚠️ Focus universe is empty and no holdings were found. Nothing to do.")
        print("✅ Intraday tick complete.")
        return

    tickers = focus_df["Ticker"].tolist()

    log_step("Downloading intraday + daily bars...")
    daily, intraday = fetch_price_data(tickers, cfg.intraday.daily_history_period, benchmark=cfg.regime.benchmark)
    log("Price data downloaded.")

    # Breadth proxy (optional)
    breadth_long_ok = True
    breadth_pct = 100.0
    if cfg.intraday.breadth_enabled:
        breadth_pct = compute_breadth(focus_df, cfg.intraday.breadth_ma_window)
        breadth_long_ok = breadth_pct >= cfg.intraday.breadth_min_long * 100.0
        log(
            f"Breadth Health: {breadth_pct:.2f}% of breadth universe above MA{cfg.intraday.breadth_ma_window} "
            f"→ breadth_long_ok={breadth_long_ok} (threshold {cfg.intraday.breadth_min_long * 100:.1f}%)"
        )
    else:
        log("Breadth filter disabled for intraday.")

    # Shared D Regime + Exposure CORE gate. This is the PROD side of the same
    # architecture tested in SIM as Test D. It is opt-in via intraday.regime_exposure.enabled.
    regime_decision = None
    long_ok = True
    if not cfg.regime.use_long:
        long_ok = False
        log("Regime filter: long side disabled by config.intraday.regime.use_long=False")

    if cfg.regime.d_enabled:
        try:
            as_of = pd.Timestamp(daily.index.get_level_values(0).max()) if isinstance(daily.index, pd.MultiIndex) and not daily.empty else None
            regime_decision = decide_regime_exposure(
                daily,
                as_of,
                {},
                benchmark=cfg.regime.benchmark,
                regime_mode=cfg.regime.regime_mode,
                exposure_mode=cfg.regime.exposure_mode,
                neutral_policy=cfg.regime.neutral_policy,
                bull_long_mult=cfg.regime.bull_long_mult,
                neutral_long_mult=cfg.regime.neutral_long_mult,
                bear_short_mult=cfg.regime.bear_short_mult,
                neutral_short_mult=cfg.regime.neutral_short_mult,
            )
            long_ok = bool(long_ok and regime_decision.allow_new_longs and regime_decision.long_size_mult > 0)
            log(
                f"D Regime/Exposure CORE: regime={regime_decision.regime_label} "
                f"allow_long={regime_decision.allow_new_longs} allow_short={regime_decision.allow_new_shorts} "
                f"long_mult={regime_decision.long_size_mult:.2f} short_mult={regime_decision.short_size_mult:.2f}"
            )
        except Exception as e:
            log(f"D Regime/Exposure CORE failed; falling back to legacy long gate: {e}")
            regime_decision = None

    if not (breadth_long_ok and long_ok):
        log("Regime/Breadth gate blocking new long intraday signals — scan will still compute diagnostics.")
    else:
        log("Regime/Breadth gate OK for long signals.")

    log_step("Evaluating candidates...")
    owned_tickers = holdings_df["Ticker"].dropna().astype(str).str.upper().str.strip().tolist() if holdings_df is not None and not holdings_df.empty and "Ticker" in holdings_df.columns else []
    diag = evaluate_intraday_signals(
        focus_df,
        daily,
        intraday,
        cfg,
        regime_decision=regime_decision,
        owned_tickers=owned_tickers,
    )

    if diag.empty:
        log("No diagnostics rows generated.")
    else:
        for _, row in diag.iterrows():
            if row["Signal"] == "SKIP-ADX":
                log_sub(f"[SKIP-ADX] {row['Ticker']} because {row['Reason']}")

    # Save diagnostics CSV
    out_csv = args.log_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    diag.to_csv(out_csv, index=False)
    log(f"Wrote diagnostics CSV → {out_csv}")

    # Preserve actionable PROD intraday signal history before the next scan overwrites
    # output/intraday_debug.csv. Daily parity/routing reports use this file to avoid
    # losing transient BUY/NEAR/SELL signals that appeared earlier in the session.
    try:
        history_path = os.environ.get(
            "PROD_SIGNAL_HISTORY_CSV",
            os.path.join(cfg.app.output_dir, "prod_intraday_signal_history.csv"),
        )
        appended = append_prod_signal_history(diag, history_path, source_file=out_csv)
        if appended.empty:
            log(f"PROD signal history unchanged → {history_path} (no BUY/NEAR/SELL/SHORT rows this scan)")
        else:
            log(f"Appended PROD signal history → {history_path} rows={len(appended)}")
    except Exception as e:
        log(f"⚠️ Failed to append PROD signal history: {e}")

    # Polished HTML/email-style report
    # Keep filenames/logs in the VM local timezone, but show report time in Dallas/Central time.
    # Holdings were already loaded before data download so owned tickers could
    # be included in SELL/risk evaluation.
    if holdings_df is None or holdings_df.empty:
        log(f"Portfolio holdings review: no holdings loaded ({holdings_source}).")
    else:
        log(f"Portfolio holdings review: loaded {len(holdings_df)} owned tickers from {holdings_source}.")

    now_ct = dt.datetime.now(ZoneInfo("America/Chicago"))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_display = now_ct.strftime("%Y-%m-%d %H:%M CT")
    html_path = os.path.join(cfg.app.output_dir, f"intraday_watch_{ts}.html")
    html = ""
    try:
        html = build_intraday_report_html(
            diag=diag,
            cfg=cfg,
            ts_display=ts_display,
            breadth_pct=breadth_pct,
            breadth_long_ok=breadth_long_ok,
            long_ok=long_ok,
            holdings=holdings_df,
            holdings_source=holdings_source,
            daily=daily,
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}")
    except Exception as e:
        print(f"⚠️ Failed to write HTML summary: {e}")

    # Simple trigger summary + optional email notification
    if not diag.empty and "Signal" in diag.columns:
        sig_upper = diag["Signal"].astype(str).str.upper()
        buys = diag.loc[sig_upper.eq("BUY")].copy()
        nears = diag.loc[sig_upper.isin(["NEAR", "NEAR_BUY", "NEAR-TRIGGER"])].copy()
        sells = diag.loc[sig_upper.isin(["SELL", "SELLTRIG", "SELL-TRIGGER"])].copy()
        skip_adx_count = int(sig_upper.eq("SKIP-ADX").sum())
    else:
        buys = pd.DataFrame()
        nears = pd.DataFrame()
        sells = pd.DataFrame()
        skip_adx_count = 0

    log(
        f"Scan done. Raw counts → BUY:{len(buys)} NEAR:{len(nears)} "
        f"SELL:{len(sells)} SKIP-ADX:{skip_adx_count}"
    )

    # Email the polished HTML report when the visible recommendation sections contain
    # at least one BUY, NEAR, or SELL. This is intentionally tied to the same
    # data used by the HTML sections, not the debounced signal_engine state, so
    # portfolio SELL recommendations are not missed when a holding is outside
    # the weekly BUY universe.
    recommendations_present = (len(buys) > 0) or (len(nears) > 0) or (len(sells) > 0)
    if recommendations_present:
        if send_email is None:
            log("Email helper unavailable — cannot send recommendation email.")
        else:
            counts = f"{len(buys)} BUY / {len(nears)} NEAR / {len(sells)} SELL"
            subject = f"Intraday Watch — {counts}"

            def _lines(df: pd.DataFrame, label: str) -> str:
                if df.empty:
                    return f"No {label} recommendations."
                rows = []
                for _, r in df.head(25).iterrows():
                    t = str(r.get("Ticker", r.get("ticker", ""))).upper()
                    px = r.get("PriceNow", r.get("price", ""))
                    reason = r.get("Reason", r.get("reason", ""))
                    rows.append(f"- {t} @ {px}: {reason}")
                return "\n".join(rows)

            text_body = (
                f"Weinstein Intraday Watch — {ts_display}\n"
                f"Recommendations: {counts}\n\n"
                f"BUY:\n{_lines(buys, 'BUY')}\n\n"
                f"NEAR:\n{_lines(nears, 'NEAR')}\n\n"
                f"SELL:\n{_lines(sells, 'SELL')}\n\n"
                f"HTML report saved on VM: {html_path}\n"
            )
            try:
                log(f"Sending recommendation email → {counts}")
                send_email(
                    subject=subject,
                    html_body=html,
                    text_body=text_body,
                    cfg_path=args.config,
                    subject_tag="INTRADAY",
                )
                log("Recommendation email sent.")
            except Exception as e:
                log(f"Recommendation email failed: {e}")
    else:
        log("No BUY/NEAR/SELL recommendations present — skipping email send.")

    print("✅ Intraday tick complete.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Intraday watcher encountered an error: {e}")
        raise
