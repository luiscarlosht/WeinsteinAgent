#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Intraday Watcher (slim core-driven version)

- Uses shared LONG core: weinstein_long_core.check_long_entry / LongEntryParams
  so SIM (backtest) + PROD (this script) share the same Stage 2 breakout logic.
- Keeps:
    * Weekly Stage 1/2 + RS-above-MA universe
    * 10-week pivot breakout
    * Volume pace vs 50dma
    * ADX trend-strength filter
    * Market regime (Chapter 8) gate
    * Breadth Health filter (%% above MA50)
    * Intraday NEAR/ARMED/TRIGGERED state machine
    * SELL triggers (MA150 crack) + SELL/Risk from holdings
- Sends email only if there are BUY/NEAR/SELL triggers.

This is intentionally trimmed down: no tiny charts, order block tables,
alert level CSV, or heavy HTML/CSS. Focus is on signals.
"""

import os
import json
import math
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from weinstein_mailer import send_email

# Shared indicators: ADX + Breadth
from weinstein_indicators import (
    compute_adx_for_ticker,
    compute_breadth_series_above_ma,
    ADX_WINDOW,
    ADX_MIN,
)

# Shared LONG-side core (used by backtest + intraday)
from weinstein_long_core import LongEntryParams, check_long_entry

# Market regime (Chapter 8)
try:
    from market_regime import inspect as inspect_market_regime
except ImportError:
    def inspect_market_regime():
        return "NEUTRAL (no market_regime.py)", True, True


# ---------------- Tunables ----------------
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
BENCHMARK_DEFAULT = "SPY"
CRYPTO_BENCHMARK  = "BTC-USD"   # used only to avoid mixing crypto as breadth universe

INTRADAY_INTERVAL = "60m"       # '60m' or '30m'
LOOKBACK_DAYS = 60
PIVOT_LOOKBACK_WEEKS = 10

# BUY side (aligned with long core)
VOL_PACE_MIN = 1.30             # daily projected volume vs 50dma
BUY_DIST_ABOVE_MA_MIN = 0.00    # min distance over MA proxy (same param as core)
MIN_BREAKOUT_PCT = 0.004        # ≈0.4% over pivot

# NEAR logic
NEAR_BELOW_PIVOT_PCT = 0.003    # within ~0.3% below pivot counts as "near"
NEAR_VOL_PACE_MIN = 1.00

# Intraday confirmation for 60m
INTRABAR_CONFIRM_MIN_ELAPSED = 40   # minutes elapsed in current bar
INTRABAR_VOLPACE_MIN = 1.20         # intrabar volume pace vs intraday avg

# SELL (MA150 crack)
SELL_BREAK_PCT = 0.005             # ~0.5% crack below MA150
SELL_NEAR_ABOVE_MA_PCT = 0.005     # "near" zone around MA150
SELL_INTRABAR_CONFIRM_MIN_ELAPSED = 40
SELL_INTRABAR_VOLPACE_MIN = 1.20

# Risk-style stop (for holdings)
HARD_STOP_PCT = 0.08              # 8% disaster stop vs entry
TRAIL_ATR_MULT = 2.0              # ATR trail vs price

# Breadth Health filter
BREADTH_MA_WINDOW = 50
BREADTH_MIN_LONG = 0.60           # require 60% of breadth universe above MA50

# Trigger statefulness
INTRADAY_STATE_FILE = "./state/intraday_triggers.json"
SCAN_INTERVAL_MIN = 10            # used only for interpretation
NEAR_HITS_WINDOW = 6
NEAR_HITS_MIN = 3
COOLDOWN_SCANS = 24

SELL_NEAR_HITS_WINDOW = 6
SELL_NEAR_HITS_MIN = 3
SELL_COOLDOWN_SCANS = 24

# Positions state
STATE_FILE = "./state/positions.json"

# Open positions CSV (for SELL / Risk from positions)
OPEN_POSITIONS_CSV_CANDIDATES = [
    "./output/Open_Positions.csv",
    "./output/open_positions.csv",
]

# Output
INTRADAY_HTML_PREFIX = "./output/intraday_watch_"

VERBOSE = True


# ---------------- Small helpers ----------------
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, *, level: str = "info"):
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
    print(f"{prefix} [{_ts()}] {msg}", flush=True)


def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan


def _is_crypto(sym: str) -> bool:
    return (sym or "").upper().endswith("-USD")


# ---------------- Config / IO ----------------
def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}
    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")
    return cfg, benchmark, sheet_url, svc_file


def newest_weekly_csv() -> str:
    files = [
        f
        for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            "No weekly CSV found in ./output. Run weinstein_report_weekly.py first."
        )
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report():
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path


def load_positions_state():
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"positions": {}}
    return {"positions": {}}


def _load_intraday_state():
    """
    Robust load of trigger-state JSON. If corrupted, back it up and reset.
    """
    path = INTRADAY_STATE_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = f"{path}.bak_{ts}"
                os.replace(path, backup)
                log(
                    f"Corrupted intraday state JSON ({e}); backed up to {backup} and resetting.",
                    level="warn",
                )
            except Exception as e2:
                log(
                    f"Corrupted intraday state JSON and failed to backup ({e2}); "
                    "resetting in-memory.",
                    level="err",
                )
            return {}
    return {}


def _save_intraday_state(st: dict):
    with open(INTRADAY_STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


def _elapsed_in_current_bar_minutes(intraday_df, ticker: str) -> int:
    try:
        if isinstance(intraday_df.columns, pd.MultiIndex):
            ts = intraday_df[("Close", ticker)].dropna().index[-1]
        else:
            ts = intraday_df["Close"].dropna().index[-1]
        last_bar_start = pd.Timestamp(ts).to_pydatetime()
        from datetime import datetime as _dt
        return max(0, int((_dt.utcnow() - last_bar_start).total_seconds() // 60))
    except Exception:
        return 0


def _update_hits(window_arr, hit: bool, window: int):
    window_arr = (window_arr or [])
    window_arr.append(1 if hit else 0)
    if len(window_arr) > window:
        window_arr = window_arr[-window:]
    return window_arr, sum(window_arr)


# ---------------- Data helpers ----------------
def get_intraday(tickers):
    uniq = list(dict.fromkeys(tickers))
    intraday = yf.download(
        uniq,
        period=f"{LOOKBACK_DAYS}d",
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )
    daily = yf.download(
        uniq,
        period="24mo",
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )
    return intraday, daily


def compute_atr(daily_df, t, n=14):
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            sub = daily_df.xs(t, axis=1, level=1)
        except KeyError:
            return np.nan
    else:
        sub = daily_df
    if not {"High", "Low", "Close"}.issubset(sub.columns):
        return np.nan
    h, l, c = sub["High"], sub["Low"], sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    return float(atr.dropna().iloc[-1]) if len(atr.dropna()) else np.nan


def last_weekly_pivot_high(ticker, daily_df, weeks=PIVOT_LOOKBACK_WEEKS):
    bars = weeks * (7 if _is_crypto(ticker) else 5)
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)]
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"]
    highs = highs.dropna().tail(bars)
    return float(highs.max()) if len(highs) else np.nan


def volume_pace_today_vs_50dma(ticker, daily_df):
    """
    Projected full-day volume vs 50-day avg.
    For equities: 09:30–16:00 ET pacing (13:30–20:00 UTC).
    For crypto: 24h clock.
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            v = daily_df[("Volume", ticker)].copy()
        except KeyError:
            return np.nan
    else:
        v = daily_df["Volume"].copy()
    if v.empty:
        return np.nan
    v50 = v.rolling(50).mean().iloc[-2] if len(v) > 50 else np.nan
    today_vol = v.iloc[-1]
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    if _is_crypto(ticker):
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = max(0.0, (now - day_start).total_seconds())
        fraction = min(1.0, max(0.05, elapsed / (24 * 3600.0)))
    else:
        minutes = now.hour * 60 + now.minute
        start = 13 * 60 + 30
        end = 20 * 60 + 0
        if minutes <= start:
            fraction = 0.05
        elif minutes >= end:
            fraction = 1.0
        else:
            fraction = (minutes - start) / (6.5 * 60)
            fraction = min(1.0, max(0.05, fraction))
    est_full = today_vol / fraction if fraction > 0 else today_vol
    return float(_safe_div(est_full, v50)) if pd.notna(v50) and v50 > 0 else np.nan


def get_last_n_intraday_closes(intraday_df, ticker, n=2):
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            s = intraday_df[("Close", ticker)].dropna()
        except KeyError:
            return []
    else:
        s = intraday_df["Close"].dropna()
    return list(map(float, s.tail(n).values))


def get_intraday_avg_volume(intraday_df, ticker, window=20):
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            v = intraday_df[("Volume", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        v = intraday_df["Volume"].dropna()
    if len(v) < window:
        return np.nan
    return float(v.tail(window).mean())


def intrabar_volume_pace(intraday_df, ticker, avg_window=20, bar_minutes=60):
    try:
        if isinstance(intraday_df.columns, pd.MultiIndex):
            v = intraday_df[("Volume", ticker)].dropna()
        else:
            v = intraday_df["Volume"].dropna()
    except Exception:
        return np.nan
    if len(v) < max(avg_window, 2):
        return np.nan
    last_bar_vol = float(v.iloc[-1])
    avg_bar_vol = float(v.tail(avg_window).mean())
    elapsed = _elapsed_in_current_bar_minutes(intraday_df, ticker)
    frac = min(1.0, max(0.05, elapsed / float(bar_minutes)))
    est_full = last_bar_vol / frac if frac > 0 else last_bar_vol
    return float(_safe_div(est_full, avg_bar_vol))


# ---------------- Holdings helpers (for SELL / Risk) ----------------
def _coerce_numlike(series: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).replace(",", "").replace("$", "").strip()
        if s.endswith("%"):
            s = s[:-1]
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.apply(conv)


def _find_open_positions_csv() -> str | None:
    for p in OPEN_POSITIONS_CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _load_open_positions_local() -> pd.DataFrame | None:
    p = _find_open_positions_csv()
    if not p:
        return None
    try:
        df = pd.read_csv(p)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _normalize_open_positions_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {
        "Ticker": "Symbol",
        "symbol": "Symbol",
        "SYMBOL": "Symbol",
        "Qty": "Quantity",
        "Shares": "Quantity",
        "quantity": "Quantity",
        "Last": "Last Price",
        "Price": "Last Price",
        "Current Value $": "Current Value",
        "Market Value": "Current Value",
        "MarketValue": "Current Value",
        "Cost Basis": "Cost Basis Total",
        "Cost": "Cost Basis Total",
        "Avg Cost": "Average Cost Basis",
        "AvgCost": "Average Cost Basis",
        "Gain $": "Total Gain/Loss Dollar",
        "Gain": "Total Gain/Loss Dollar",
        "Gain %": "Total Gain/Loss Percent",
        "GainPct": "Total Gain/Loss Percent",
        "Name": "Description",
        "Description/Name": "Description",
    }
    out = df.rename(columns=ren).copy()
    required = [
        "Symbol",
        "Description",
        "Quantity",
        "Last Price",
        "Current Value",
        "Cost Basis Total",
        "Average Cost Basis",
        "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent",
    ]
    for c in required:
        if c not in out.columns:
            out[c] = np.nan
    num_cols = [
        "Quantity",
        "Last Price",
        "Current Value",
        "Cost Basis Total",
        "Average Cost Basis",
        "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent",
    ]
    for c in num_cols:
        out[c] = _coerce_numlike(out[c])
    out = out.dropna(how="all")
    return out


def _merge_stage_into_positions(positions: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    w = weekly_df.rename(columns=str.lower)
    need = ["ticker", "stage", "rs_above_ma", "ma30"]
    for n in need:
        if n not in w.columns:
            w[n] = np.nan
    stage_min = w[need].rename(columns={"ticker": "Symbol"})
    return positions.merge(stage_min, on="Symbol", how="left")


# ---------------- Sorting helpers ----------------
def stage_order(stage: str) -> int:
    if isinstance(stage, str):
        if stage.startswith("Stage 2"):
            return 0
        if stage.startswith("Stage 1"):
            return 1
    return 9


def buy_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    pace = item.get("pace", np.nan)
    pace = pace if pd.notna(pace) else -1e9
    px = item.get("price", np.nan)
    pivot = item.get("pivot", np.nan)
    ma = item.get("ma30", np.nan)
    ratio_pivot = (px / pivot) if (pd.notna(px) and pd.notna(pivot) and pivot != 0) else -1e9
    ratio_ma = (px / ma) if (pd.notna(px) and pd.notna(ma) and ma != 0) else -1e9
    return (wr, st, -pace, -ratio_pivot, -ratio_ma)


def near_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan)
    pivot = item.get("pivot", np.nan)
    dist = abs(px - pivot) if (pd.notna(px) and pd.notna(pivot)) else 1e9
    pace = item.get("pace", np.nan)
    pace = pace if pd.notna(pace) else -1e9
    return (wr, st, dist, -pace)


def sell_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan)
    ma = item.get("ma30", np.nan)
    dist_below = (ma - px) if (pd.notna(px) and pd.notna(ma)) else -1e9
    pace = item.get("pace", np.nan)
    pace = pace if pd.notna(pace) else -1e9
    return (wr, st, -dist_below, -pace)


# ---------------- SELL logic helpers ----------------
def _price_below_ma(px, ma):
    return pd.notna(px) and pd.notna(ma) and px <= ma * (1.0 - SELL_BREAK_PCT)


def _near_sell_zone(px, ma):
    if pd.isna(px) or pd.isna(ma):
        return False
    return (px >= ma * (1.0 - SELL_BREAK_PCT)) and (px <= ma * (1.0 + SELL_NEAR_ABOVE_MA_PCT))


# ---------------- Main logic ----------------
def run(
    _config_path: str = "./config.yaml",
    *,
    only_tickers=None,
    log_csv: str | None = None,
    log_json: str | None = None,
    dry_run: bool = False,
):
    log(f"Intraday watcher starting with config: {_config_path}", level="step")
    cfg, benchmark, sheet_url, service_account_file = load_config(_config_path)
    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # Normalize expected weekly columns
    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker", "stage", "ma30", "rs_above_ma", "asset_class", "rank"]:
        if miss not in w.columns:
            w[miss] = np.nan

    focus = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])][
        ["ticker", "stage", "ma30", "rs_above_ma", "asset_class", "rank"]
    ].copy()
    focus["weekly_rank"] = w["rank"]

    if only_tickers:
        allowed = set(t.strip().upper() for t in only_tickers)
        focus = focus[focus["ticker"].str.upper().isin(allowed)].copy()

    log(f"Focus universe: {len(focus)} symbols (Stage 1/2).", level="info")

    # Positions state
    pos_state = load_positions_state()
    held = pos_state.get("positions", {}) or {}
    if held:
        log(f"Held symbols detected: {sorted(held.keys())}", level="debug")

    # Build symbol set for data download
    needs = sorted(set(focus["ticker"].astype(str).str.upper().tolist() + [benchmark, CRYPTO_BENCHMARK]))

    log("Downloading intraday + daily bars...", level="step")
    intraday, daily = get_intraday(needs)
    log("Price data downloaded.", level="ok")

    # ---- Market regime (Chapter 8) ----
    try:
        regime_label, long_ok, short_ok = inspect_market_regime()
        market_long_ok = bool(long_ok)
        market_short_ok = bool(short_ok)
    except Exception as e:
        log(f"Market regime evaluation failed ({e}); defaulting to neutral.", level="warn")
        regime_label = "NEUTRAL (error)"
        market_long_ok = True
        market_short_ok = True

    # ---- Breadth Health filter ----
    breadth_today = np.nan
    breadth_long_ok = True
    try:
        if "asset_class" in w.columns:
            mask_eq = w["asset_class"].fillna("").astype(str).str.contains("Equity", case=False)
            breadth_universe = w.loc[mask_eq, "ticker"].astype(str).str.upper().tolist()
        else:
            breadth_universe = focus["ticker"].astype(str).str.upper().tolist()
        breadth_universe = sorted(set(breadth_universe))
        if breadth_universe:
            breadth_series = compute_breadth_series_above_ma(
                daily,
                breadth_universe,
                ma_window=BREADTH_MA_WINDOW,
            )
            breadth_clean = breadth_series.dropna()
            if len(breadth_clean):
                breadth_today = float(breadth_clean.iloc[-1])
                breadth_long_ok = breadth_today >= BREADTH_MIN_LONG
        log(
            f"Breadth Health: {breadth_today*100:.1f}% of breadth universe above MA{BREADTH_MA_WINDOW} "
            f"→ breadth_long_ok={breadth_long_ok} (threshold {BREADTH_MIN_LONG*100:.1f}%)",
            level="info",
        )
    except Exception as e:
        log(f"Failed to compute Breadth Health filter ({e}); breadth_long_ok=True.", level="warn")
        breadth_today = np.nan
        breadth_long_ok = True

    env_long_ok = market_long_ok and breadth_long_ok

    log(
        f"Market regime (Ch8): {regime_label} | long_ok={market_long_ok} short_ok={market_short_ok}",
        level="info",
    )

    # Shortcuts for last intraday prices
    if isinstance(intraday.columns, pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    def px_now(t: str) -> float:
        if hasattr(last_closes, "index") and (t in last_closes.index):
            return float(last_closes.get(t, np.nan))
        vals = getattr(last_closes, "values", [])
        return float(vals[-1]) if len(vals) else np.nan

    # Trigger state
    trigger_state = _load_intraday_state()

    # LONG core params (shared with backtest)
    long_params = LongEntryParams(
        min_break_pct=MIN_BREAKOUT_PCT,
        dist_above_ma_min=BUY_DIST_ABOVE_MA_MIN,
        vol_min=VOL_PACE_MIN,
        adx_min=ADX_MIN,
    )

    buy_signals, near_signals = [], []
    sell_triggers, sell_signals = [], []
    info_rows, debug_rows = [], []

    log("Evaluating candidates...", level="step")

    # ----- MAIN LOOP over focus universe -----
    for _, row in focus.iterrows():
        t = str(row["ticker"])
        if t in (benchmark, CRYPTO_BENCHMARK):
            continue

        px = px_now(t)
        if np.isnan(px):
            continue

        stage = str(row["stage"])
        ma30 = float(row.get("ma30", np.nan))
        rs_above = bool(row.get("rs_above_ma", False))
        weekly_rank = float(row.get("weekly_rank", np.nan))

        # 10-week pivot, volume pace, ATR
        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma(t, daily)
        atr = compute_atr(daily, t, n=14)

        # ADX via shared indicator helper
        adx = compute_adx_for_ticker(daily, t, n=ADX_WINDOW)

        # Intraday info
        closes_n = get_last_n_intraday_closes(intraday, t, n=2)
        elapsed = _elapsed_in_current_bar_minutes(intraday, t) if INTRADAY_INTERVAL == "60m" else None
        pace_intra = intrabar_volume_pace(intraday, t, avg_window=20, bar_minutes=60) if INTRADAY_INTERVAL == "60m" else None

        # ---- Shared LONG core check ----
        entry_check = check_long_entry(
            price=px,
            ma_val=ma30,
            pivot=pivot,
            rs_above_ma=rs_above,
            vol_mult=pace,
            adx_val=adx,
            params=long_params,
        )

        # If ADX is explicitly below threshold, log skip (core also reflects this)
        if not entry_check.adx_ok and pd.notna(adx):
            log(
                f"[SKIP-ADX] {t} because ADX{ADX_WINDOW}={adx:.1f} < {ADX_MIN:.1f}",
                level="debug",
            )

        # Environment and weekly/RS sanity
        weekly_stage_ok = stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)")
        ma_ok = pd.notna(ma30)
        pivot_ok = pd.notna(pivot)
        rs_ok = rs_above

        # BUY confirmation (intraday wrapper on top of core.can_enter)
        confirm = False
        buy_price_ok = False
        buy_vol_ok = False

        if env_long_ok and weekly_stage_ok and entry_check.can_enter:
            # Core says "this is a valid breakout" on daily basis.
            buy_price_ok = bool(entry_check.price_ok)
            buy_vol_ok = bool(entry_check.vol_ok)

            if INTRADAY_INTERVAL == "60m":
                # Require intrabar elapsed + intrabar pace for confirmation
                elapsed_ok = (elapsed is not None) and (elapsed >= INTRABAR_CONFIRM_MIN_ELAPSED)
                vol_intra_ok = (pd.isna(pace_intra) or pace_intra >= INTRABAR_VOLPACE_MIN)
                confirm = buy_price_ok and buy_vol_ok and entry_check.adx_ok and elapsed_ok and vol_intra_ok
            else:
                # For 30m, simplify: if core says can_enter, treat as confirmed
                confirm = buy_price_ok and buy_vol_ok and entry_check.adx_ok

        # NEAR condition (price near pivot, not fully confirmed)
        near_now = False
        if env_long_ok and weekly_stage_ok and rs_ok and ma_ok and pivot_ok and pd.notna(px):
            above_ma = px >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
            if above_ma:
                lower = pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)
                upper = pivot * (1.0 + MIN_BREAKOUT_PCT)
                if (px >= lower) and (px < upper):
                    near_now = True
                elif (px >= upper) and not confirm:
                    near_now = True

        # SELL (MA150) logic
        sell_near_now = False
        sell_confirm = False
        sell_price_ok = False
        sell_vol_ok = True
        if ma_ok and pd.notna(px):
            sell_near_now = _near_sell_zone(px, ma30)
            if INTRADAY_INTERVAL == "60m":
                sell_price_ok = _price_below_ma(px, ma30)
                sell_vol_ok = (pd.isna(pace_intra) or (pace_intra >= SELL_INTRABAR_VOLPACE_MIN))
                elapsed_ok = (elapsed is not None) and (elapsed >= SELL_INTRABAR_CONFIRM_MIN_ELAPSED)
                sell_confirm = bool(sell_price_ok and sell_vol_ok and elapsed_ok)
            else:
                # Simplified: single-bar crack for non-60m
                sell_price_ok = _price_below_ma(px, ma30)
                sell_confirm = sell_price_ok

        # ---------------- Stateful promotion (BUY / SELL) ----------------
        ts_key = t
        st = trigger_state.get(
            ts_key,
            {
                "state": "IDLE",
                "near_hits": [],
                "cooldown": 0,
                "sell_state": "IDLE",
                "sell_hits": [],
                "sell_cooldown": 0,
            },
        )

        # If environment is OFF for longs, reset BUY side state
        if not env_long_ok:
            st["state"] = "IDLE"
            st["near_hits"] = []
            st["cooldown"] = 0

        # BUY promotion
        st["near_hits"], near_count = _update_hits(
            st.get("near_hits", []),
            near_now,
            NEAR_HITS_WINDOW,
        )
        if st.get("cooldown", 0) > 0:
            st["cooldown"] = int(st["cooldown"]) - 1

        state_now = st.get("state", "IDLE")
        if state_now == "IDLE" and near_now:
            state_now = "NEAR"
        elif state_now in ("IDLE", "NEAR") and near_count >= NEAR_HITS_MIN:
            state_now = "ARMED"
        elif (
            state_now == "ARMED"
            and confirm
            and buy_vol_ok
            and entry_check.vol_ok
            and entry_check.adx_ok
        ):
            state_now = "TRIGGERED"
            st["cooldown"] = COOLDOWN_SCANS
        elif st["cooldown"] > 0 and not near_now:
            state_now = "COOLDOWN"
        elif st["cooldown"] == 0 and not near_now and not confirm:
            state_now = "IDLE"
        st["state"] = state_now

        # SELL promotion
        st["sell_hits"], sell_hit_count = _update_hits(
            st.get("sell_hits", []),
            sell_near_now,
            SELL_NEAR_HITS_WINDOW,
        )
        if st.get("sell_cooldown", 0) > 0:
            st["sell_cooldown"] = int(st["sell_cooldown"]) - 1

        sell_state = st.get("sell_state", "IDLE")
        if sell_state == "IDLE" and sell_near_now:
            sell_state = "NEAR"
        elif sell_state in ("IDLE", "NEAR") and sell_hit_count >= SELL_NEAR_HITS_MIN:
            sell_state = "ARMED"
        elif sell_state == "ARMED" and sell_confirm and sell_vol_ok:
            sell_state = "TRIGGERED"
            st["sell_cooldown"] = SELL_COOLDOWN_SCANS
        elif st["sell_cooldown"] > 0 and not sell_near_now:
            sell_state = "COOLDOWN"
        elif st["sell_cooldown"] == 0 and not sell_near_now and not sell_confirm:
            sell_state = "IDLE"
        st["sell_state"] = sell_state

        trigger_state[ts_key] = st

        # --------------- SELL risk (from positions.json) ---------------
        pos = held.get(t)
        if pos:
            entry_price = float(pos.get("entry", np.nan))
            hard_stop = float(pos.get("stop", np.nan)) if pd.notna(pos.get("stop", np.nan)) else (
                entry_price * (1.0 - HARD_STOP_PCT) if pd.notna(entry_price) else np.nan
            )
            atr_pos = atr
            trail = px - TRAIL_ATR_MULT * atr_pos if pd.notna(atr_pos) else np.nan
            breach_hard = pd.notna(hard_stop) and px <= hard_stop
            breach_ma = pd.notna(ma30) and px <= ma30 * 0.97
            breach_trail = pd.notna(trail) and px <= trail
            if breach_hard or breach_ma or breach_trail:
                reasons = []
                if breach_hard:
                    reasons.append(f"≤ hard stop ({hard_stop:.2f})")
                if breach_ma:
                    reasons.append("≤ 30-wk MA proxy (−3%)")
                if breach_trail:
                    reasons.append(f"≤ ATR trail ({TRAIL_ATR_MULT}×)")
                sell_signals.append(
                    {
                        "ticker": t,
                        "price": px,
                        "reasons": ", ".join(reasons),
                        "stage": stage,
                        "weekly_rank": weekly_rank,
                        "source": "risk",
                    }
                )

        # --------------- Emit based on state ---------------
        if (
            st["state"] == "TRIGGERED"
            and env_long_ok
            and entry_check.can_enter
        ):
            buy_signals.append(
                {
                    "ticker": t,
                    "price": px,
                    "pivot": pivot,
                    "pace": None if pd.isna(pace) else float(pace),
                    "stage": stage,
                    "ma30": ma30,
                    "weekly_rank": weekly_rank,
                }
            )
            trigger_state[ts_key]["state"] = "COOLDOWN"
        elif st["state"] in ("NEAR", "ARMED") and env_long_ok:
            if pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN:
                near_signals.append(
                    {
                        "ticker": t,
                        "price": px,
                        "pivot": pivot,
                        "pace": None if pd.isna(pace) else float(pace),
                        "stage": stage,
                        "ma30": ma30,
                        "weekly_rank": weekly_rank,
                    }
                )

        if st["sell_state"] == "TRIGGERED":
            sell_triggers.append(
                {
                    "ticker": t,
                    "price": px,
                    "ma30": ma30,
                    "stage": stage,
                    "weekly_rank": weekly_rank,
                    "pace": None if pd.isna(pace) else float(pace),
                }
            )
            trigger_state[ts_key]["sell_state"] = "COOLDOWN"

        info_rows.append(
            {
                "ticker": t,
                "stage": stage,
                "price": px,
                "ma30": ma30,
                "pivot10w": pivot,
                "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace), 2),
                "buy_state": st["state"],
                "sell_state": st["sell_state"],
                "weekly_rank": weekly_rank,
                "adx": None if pd.isna(adx) else float(adx),
            }
        )
        debug_rows.append(
            {
                "ticker": t,
                "price": px,
                "ma30": ma30,
                "pivot": pivot,
                "atr": atr,
                "pace_full_vs50dma": None if pd.isna(pace) else float(pace),
                "pace_intrabar": None if pd.isna(pace_intra) else float(pace_intra),
                "elapsed_min": elapsed,
                "stage": stage,
                "weekly_rank": weekly_rank,
                "adx": None if pd.isna(adx) else float(adx),
                "env_long_ok": env_long_ok,
                "core_can_enter": bool(entry_check.can_enter),
                "core_price_ok": bool(entry_check.price_ok),
                "core_vol_ok": bool(entry_check.vol_ok),
                "core_adx_ok": bool(entry_check.adx_ok),
                "near_now": near_now,
                "confirm": confirm,
                "sell_near_now": sell_near_now,
                "sell_confirm": sell_confirm,
                "state": st["state"],
                "sell_state": st["sell_state"],
            }
        )

    log(
        f"Scan done. Raw counts → BUY:{len(buy_signals)} NEAR:{len(near_signals)} "
        f"SELLTRIG:{len(sell_triggers)} RISK-SELL:{len(sell_signals)}",
        level="info",
    )

    # ---- SELL recommendations from holdings CSV (strategy rules) ----
    holdings_raw = _load_open_positions_local()
    if holdings_raw is not None and not holdings_raw.empty:
        pos_norm = _normalize_open_positions_columns(holdings_raw)
        merged = _merge_stage_into_positions(pos_norm, weekly_df)
        for _, r in merged.iterrows():
            sym = str(r.get("Symbol", "")).strip()
            if not sym:
                continue
            pct = r.get("Total Gain/Loss Percent", np.nan)
            stg = str(r.get("stage", ""))
            rec_reasons = []
            if pd.notna(pct) and pct <= -8.0:
                rec_reasons.append("drawdown ≤ −8%")
            if stg.startswith("Stage 4") and pd.notna(pct) and pct < 0:
                rec_reasons.append("Stage 4 + negative P/L")
            if rec_reasons:
                sell_signals.append(
                    {
                        "ticker": sym,
                        "price": np.nan,
                        "reasons": "; ".join(rec_reasons),
                        "stage": stg,
                        "weekly_rank": np.nan,
                        "source": "positions",
                    }
                )

    # -------- Sorting --------
    buy_signals.sort(key=buy_sort_key)
    near_signals.sort(key=near_sort_key)
    sell_triggers.sort(key=sell_sort_key)

    # -------- Email HTML + Text --------
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def bullets_html(items, kind):
        if not items:
            return f"<p>No {kind} signals.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "SELL":
                price_val = it.get("price", np.nan)
                price_str = f"{price_val:.2f}" if pd.notna(price_val) else "—"
                src = it.get("source", "")
                src_label = " (Position SELL)" if src == "positions" else ""
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {price_str} — "
                    f"{it.get('reasons','')} ({it.get('stage','')}, weekly {wr_str}){src_label}</li>"
                )
            elif kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(↓ MA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})</li>"
                )
            else:  # BUY / NEAR
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']}, weekly {wr_str})</li>"
                )
        return "<ol>" + "\n".join(lis) + "</ol>"

    info_df = pd.DataFrame(info_rows)
    if not info_df.empty:
        info_df["stage_rank"] = info_df["stage"].apply(stage_order)
        info_df["weekly_rank"] = pd.to_numeric(info_df["weekly_rank"], errors="coerce").fillna(999999).astype(int)
        info_df = info_df.sort_values(["weekly_rank", "stage_rank", "ticker"]).drop(columns=["stage_rank"])

    html = f"""
    <h3>Weinstein Intraday Watch — {now}</h3>
    <p style="font-size:13px;color:#555;">
      BUY: Shared LONG core (check_long_entry) → Stage 1/2, RS above MA, price ≥ pivot + {MIN_BREAKOUT_PCT*100:.1f}%,
      above MA150 proxy, volume pace ≥ {VOL_PACE_MIN:.2f}× vs 50dma, ADX{ADX_WINDOW} ≥ {ADX_MIN:.1f} when available,
      plus intraday confirmation (elapsed ≥ {INTRABAR_CONFIRM_MIN_ELAPSED}min, intrabar pace ≥ {INTRABAR_VOLPACE_MIN:.2f}×).
      <br>
      NEAR: Stage 1/2 + RS ok, above MA, price within ~{NEAR_BELOW_PIVOT_PCT*100:.1f}% below pivot or just over pivot but not fully confirmed.
      <br>
      SELL-TRIGGER: Crack below MA150 by ~{SELL_BREAK_PCT*100:.1f}% with persistence; for 60m bars,
      elapsed ≥ {SELL_INTRABAR_CONFIRM_MIN_ELAPSED}min and intrabar pace ≥ {SELL_INTRABAR_VOLPACE_MIN:.2f}×.
    </p>
    <p style="font-size:13px;color:#555;">
      <b>Market Regime (Chapter 8):</b> {regime_label} — LONG allowed={market_long_ok}, SHORT allowed={market_short_ok}.<br>
      <b>Breadth Health:</b> {breadth_today*100:.1f}% of breadth universe above MA{BREADTH_MA_WINDOW}
      (LONG breadth_ok={breadth_long_ok}, threshold {BREADTH_MIN_LONG*100:.1f}%).<br>
      <b>Effective LONG gate:</b> env_long_ok = market_long_ok AND breadth_long_ok → {env_long_ok}.
    </p>
    <h4>Buy Triggers (ranked)</h4>
    {bullets_html(buy_signals, "BUY")}
    <h4>Near-Triggers (ranked)</h4>
    {bullets_html(near_signals, "NEAR")}
    <h4>Sell Triggers (ranked)</h4>
    {bullets_html(sell_triggers, "SELLTRIG")}
    <h4>Sell / Risk (tracked positions + strategy rules)</h4>
    {bullets_html(sell_signals, "SELL")}
    <h4>Snapshot (debug)</h4>
    {info_df.to_html(index=False) if not info_df.empty else "<p>No snapshot rows.</p>"}
    """

    # Plain text body
    def _lines_text(items, kind):
        out = []
        for i, it in enumerate(items, 1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} (below MA150 {ma_str}, pace {pace_str}, "
                    f"{it.get('stage','')}, weekly {wr_str})"
                )
            elif kind == "SELL":
                price_val = it.get("price", np.nan)
                price_str = f"{price_val:.2f}" if pd.notna(price_val) else "—"
                src = it.get("source", "")
                src_label = " [Position]" if src == "positions" else ""
                out.append(
                    f"{i}. {it['ticker']} @ {price_str}{src_label} — {it.get('reasons','')} "
                    f"({it.get('stage','')}, weekly {wr_str})"
                )
            else:
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} (pivot {it['pivot']:.2f}, "
                    f"pace {pace_str}, {it['stage']}, weekly {wr_str})"
                )
        return "\n".join(out) if out else f"No {kind} signals."

    text = (
        f"Weinstein Intraday Watch — {now}\n"
        f"Market Regime (Ch8): {regime_label} | LONG allowed={market_long_ok}, SHORT allowed={market_short_ok}\n"
        f"Breadth Health: {breadth_today*100:.1f}% above MA{BREADTH_MA_WINDOW} "
        f"(LONG breadth_ok={breadth_long_ok}, threshold {BREADTH_MIN_LONG*100:.1f}%)\n"
        f"Effective LONG gate env_long_ok = market_long_ok AND breadth_long_ok → {env_long_ok}\n"
        f"ADX filter via long core: ADX{ADX_WINDOW} ≥ {ADX_MIN:.1f} when available; NaN → no ADX block.\n\n"
        f"BUY (ranked):\n{_lines_text(buy_signals, 'BUY')}\n\n"
        f"NEAR-TRIGGER (ranked):\n{_lines_text(near_signals, 'NEAR')}\n\n"
        f"SELL TRIGGERS (ranked):\n{_lines_text(sell_triggers, 'SELLTRIG')}\n\n"
        f"SELL / RISK:\n{_lines_text(sell_signals, 'SELL')}\n"
    )

    # Persist state & diagnostics
    _save_intraday_state(trigger_state)
    if log_csv:
        try:
            pd.DataFrame(debug_rows).to_csv(log_csv, index=False)
            log(f"Wrote diagnostics CSV → {log_csv}", level="ok")
        except Exception as e:
            log(f"Failed writing diagnostics CSV: {e}", level="warn")
    if log_json:
        try:
            with open(log_json, "w") as f:
                json.dump({"rows": debug_rows}, f, indent=2, default=str)
            log(f"Wrote diagnostics JSON → {log_json}", level="ok")
        except Exception as e:
            log(f"Failed writing diagnostics JSON: {e}", level="warn")

    # Save HTML snapshot
    os.makedirs("./output", exist_ok=True)
    html_path = INTRADAY_HTML_PREFIX + datetime.now().strftime("%Y%m%d_%H%M%S") + ".html"
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    # Decide whether to send email
    has_signals = bool(buy_signals or near_signals or sell_triggers)
    if not has_signals:
        log("No BUY/NEAR/SELL triggers present — skipping email send.", level="info")
        if dry_run:
            log("DRY-RUN set — no email would be sent anyway.", level="debug")
        return

    subject_counts = (
        f"{len(buy_signals)} BUY / {len(near_signals)} NEAR / {len(sell_triggers)} SELL-TRIG"
    )
    subject_tag = f"INTRADAY {regime_label} L={env_long_ok} S={market_short_ok}"
    regime_header = (
        f"Market regime (Ch8): {regime_label} | long_ok={market_long_ok} short_ok={market_short_ok} | "
        f"breadth_above_MA{BREADTH_MA_WINDOW}={breadth_today*100:.1f}% "
        f"(long_ok={breadth_long_ok}, thresh={BREADTH_MIN_LONG*100:.1f}%)"
    )

    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        log("Sending email...", level="step")
        send_email(
            subject=f"Intraday Watch — {subject_counts}",
            html_body=html,
            text_body=text,
            cfg_path=_config_path,
            subject_tag=subject_tag,
            regime_header=regime_header,
        )
        log("Email sent.", level="ok")


# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--quiet", action="store_true", help="reduce console noise")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="comma list of tickers to restrict evaluation (e.g. MU,DDOG)",
    )
    ap.add_argument("--log-csv", type=str, default="", help="path to diagnostics CSV")
    ap.add_argument("--log-json", type=str, default="", help="path to diagnostics JSON")
    ap.add_argument("--dry-run", action="store_true", help="don’t send email")
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = (
        [s.strip().upper() for s in args.only.split(",") if s.strip()]
        if args.only
        else None
    )

    try:
        run(
            _config_path=args.config,
            only_tickers=only,
            log_csv=args.log_csv or None,
            log_json=args.log_json or None,
            dry_run=args.dry_run,
        )
        log("Intraday tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
