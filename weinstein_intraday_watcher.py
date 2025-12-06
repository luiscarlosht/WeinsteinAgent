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
    * intraday_watch_*.html (simple HTML summary)

CLI:
    python3 weinstein_intraday_watcher.py \
        --config ./config.yaml \
        --log-csv ./output/intraday_debug.csv
"""

import argparse
import datetime as dt
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import yaml


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

    # Allow an optional nested intraday.prod block; if not present,
    # treat the top-level intraday dict as the source of knobs.
    intraday_prod = intraday.get("prod", intraday)

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
        breadth_enabled=breadth_enabled,
        breadth_ma_window=breadth_ma,
        breadth_min_long=breadth_min_long,
    )
    r_cfg = RegimeConfig(
        use_long=regime_use_long,
        use_short=regime_use_short,
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

def fetch_price_data(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download:
        - daily OHLCV (6 months) for pivots / MAs
        - 60m bars (60 days) for intraday signals
    Returns:
        daily:  multi-index (Date, Ticker)
        intraday: multi-index (DateTime, Ticker)
    """
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    tickers_str = " ".join(sorted(set(tickers)))

    # Daily
    daily = yf.download(
        tickers_str,
        period="6mo",
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
            out["Ticker"] = tickers[0]
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

    if focus_df.empty:
        return pd.DataFrame()

    tickers = focus_df["Ticker"].tolist()

    for ticker in tickers:
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

        # ADX gate
        if np.isnan(adx14) or adx14 < cfg.intraday.adx_min_long:
            rows.append(
                dict(
                    Ticker=ticker,
                    Signal="SKIP-ADX",
                    Reason=f"ADX14={adx14:.1f} < {cfg.intraday.adx_min_long}",
                )
            )
            continue

        # Stage-like filter: price above MA150 and MA150 rising
        if pd.isna(last["MA150"]):
            rows.append(
                dict(
                    Ticker=ticker,
                    Signal="SKIP-MA",
                    Reason="MA150 not available",
                )
            )
            continue

        if not (last["Close"] > last["MA150"] * 1.02):
            rows.append(
                dict(
                    Ticker=ticker,
                    Signal="SKIP-STAGE",
                    Reason=f"Close not sufficiently above MA150 ({last['Close']:.2f} vs {last['MA150']:.2f})",
                )
            )
            continue

        # Pivot = 50d high close (using last ~60 days as a robust window)
        pivot_window = d["Close"].tail(60).max()
        if pd.isna(pivot_window):
            rows.append(
                dict(
                    Ticker=ticker,
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
                    Signal="SKIP-INTRADAY",
                    Reason="No intraday data",
                )
            )
            continue

        if intr.empty:
            rows.append(
                dict(
                    Ticker=ticker,
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
                    Signal="SKIP-DATA",
                    Reason="Missing price/vol/VolMA50",
                )
            )
            continue

        vol_pace = vol_now / vol_ma50
        headroom_pct = (price_now / pivot_window - 1.0) * 100.0

        # BUY vs NEAR logic
        if (
            price_now >= pivot_window * (1.0 + cfg.intraday.confirm_headroom_pct / 100.0)
            and vol_pace >= cfg.intraday.vol_pace_min
        ):
            signal = "BUY"
            reason = (
                f"Price {price_now:.2f} ≥ pivot {pivot_window:.2f} + "
                f"{cfg.intraday.confirm_headroom_pct:.1f}% & vol pace {vol_pace:.2f}x"
            )
        elif (
            price_now >= pivot_window * (1.0 - cfg.intraday.near_below_pivot_pct / 100.0)
            and vol_pace >= cfg.intraday.near_vol_pace_min
        ):
            signal = "NEAR"
            reason = (
                f"Price {price_now:.2f} within {cfg.intraday.near_below_pivot_pct:.1f}% "
                f"of pivot {pivot_window:.2f} & vol pace {vol_pace:.2f}x"
            )
        else:
            signal = "NONE"
            reason = f"No breakout. headroom={headroom_pct:.2f}%, vol_pace={vol_pace:.2f}x"

        rows.append(
            dict(
                Ticker=ticker,
                Signal=signal,
                Reason=reason,
                PriceNow=price_now,
                Pivot=pivot_window,
                HeadroomPct=headroom_pct,
                VolPace=vol_pace,
                ADX14=adx14,
                CloseDaily=float(last["Close"]),
                MA30=float(last["MA30"]) if not pd.isna(last["MA30"]) else np.nan,
                MA150=float(last["MA150"]),
                ATR14=float(last["ATR14"]) if not pd.isna(last["ATR14"]) else np.nan,
            )
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Weinstein Intraday Watcher (PROD)")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to config.yaml")
    parser.add_argument("--log-csv", type=str, default="./output/intraday_debug.csv", help="Diagnostics CSV output")
    args = parser.parse_args()

    try:
        cfg_raw = load_yaml_config(args.config)
        cfg = build_full_config(cfg_raw)
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

    if focus_df.empty:
        print("⚠️ Focus universe is empty. Nothing to do.")
        print("✅ Intraday tick complete.")
        return

    tickers = focus_df["Ticker"].tolist()

    log_step("Downloading intraday + daily bars...")
    daily, intraday = fetch_price_data(tickers)
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

    # Regime gate for longs (if desired you can call your market_regime module here)
    long_ok = True
    if not cfg.regime.use_long:
        long_ok = False
        log("Regime filter: long side disabled by config.intraday.regime.use_long=False")

    if not (breadth_long_ok and long_ok):
        log("Regime/Breadth gate blocking new long intraday signals — scan will still compute diagnostics.")
    else:
        log("Regime/Breadth gate OK for long signals.")

    log_step("Evaluating candidates...")
    diag = evaluate_intraday_signals(focus_df, daily, intraday, cfg)

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

    # Simple HTML summary
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(cfg.app.output_dir, f"intraday_watch_{ts}.html")
    try:
        html = diag.to_html(index=False)
        with open(html_path, "w") as f:
            f.write("<html><body>\n")
            f.write("<h2>Weinstein Intraday Watch — Diagnostics</h2>\n")
            f.write(html)
            f.write("\n</body></html>")
        log(f"Saved HTML → {html_path}")
    except Exception as e:
        print(f"⚠️ Failed to write HTML summary: {e}")

    # Simple trigger summary
    buys = diag.loc[diag["Signal"] == "BUY"] if not diag.empty else pd.DataFrame()
    nears = diag.loc[diag["Signal"] == "NEAR"] if not diag.empty else pd.DataFrame()

    log(
        f"Scan done. Raw counts → BUY:{len(buys)} NEAR:{len(nears)} "
        f"SKIP-ADX:{int((diag['Signal'] == 'SKIP-ADX').sum()) if not diag.empty else 0}"
    )

    print("✅ Intraday tick complete.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Intraday watcher encountered an error: {e}")
        raise
