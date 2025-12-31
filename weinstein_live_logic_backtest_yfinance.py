#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Live Logic Backtest (daily approximation of intraday watchers)

This version (LONG FIXES + MORE WEINSTEIN-ALIGNED + SHORT CORE + METRICS):
- ✅ Proper cash accounting: subtract entry cost, add exit proceeds
- ✅ Equity = cash + market value of open positions (mark-to-market)
- ✅ Risk sizing uses EQUITY
- ✅ Caps:
    - max_leverage (default 1.0)
    - max_pos_frac per position (default 0.25)
- ✅ Restores monthly progress logging
- ✅ Final summary + outputs:
    - trades CSV
    - equity curve PNG
    - monthly breakdown CSV
    - performance summary CSV (CAGR / MaxDD / Vol / Sharpe-ish)
- ✅ Keeps Industry filters (single source of truth)
- ✅ Adds Stage 2 gate (from snapshot row) for LONG entries
- ✅ Adds SHORTS (Weinstein-style, conservative):
    - Stage 4 gate (from snapshot)
    - Market gate for shorts (optional)
    - Industry confirmation
    - Cash + liability accounting for shorts
    - Stop + trailing stop skeleton for shorts (ATR/MA guard)

- ✅ Market gate (Weinstein Chapter 8-ish):
    - SPY 30-week proxy (150d) MA slope >= ma30_slope_min for longs if require_rising_ma30=True
    - SPY 30-week proxy slope <= ma30_slope_min_short for shorts if require_falling_ma30=True
    - (and SPY below MA150 proxy for shorts)
    - Optional VIX filter (longs suppressed when ^VIX > vix_max)

IMPORTANT FIX (Dec 2025):
- ✅ Shorts were not triggering because Stage4 names were being rejected by a “long slope ok” check.
  This file keeps:
    - short_slope_ok_from_snapshot(...) (optional, configurable)
    - per-month short gating diagnostics (so you can see where candidates die)

NEW (Dec 2025):
- ✅ Optional "failed rally" short entry gate:
    - require_failed_rally: True/False
    - failed_rally_lookback: e.g. 10 days
    - failed_rally_pct: e.g. 0.02 (2% below recent lookback high)

NEW (Dec 2025 - your request):
- ✅ Wire SHORT entry to use:
    - pivot low breakdown (daily proxy, last N closes)
    - vol_mult vs 50d avg volume (uses config backtest.short.vol_min)
    - weak RS gate (snapshot rs_above_ma must be False when present)
  This makes your config change `vol_min: 1.10` actually matter.

CRITICAL BUGFIX (Dec 31, 2025):
- ✅ Pivot low for breakdown must EXCLUDE today's close; otherwise pivot==px on new lows
  and breakdown can never trigger. (This was causing no_breakdown to dominate and 0 trades.)
"""

import argparse
import os
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Tuple, Mapping

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml

# =========================
# SHARED IMPORTS
# =========================

from weinstein_indicators import (
    compute_adx_series,               # (not yet fully used here; kept for future)
    ADX_WINDOW,                       # (kept)
    ADX_MIN,                          # (kept)
    compute_breadth_series_above_ma,  # (not used here yet; kept for future)
)

from weinstein_long_core import (
    LongEntryParams,     # (kept for future wiring)
    check_long_entry,    # (kept for future wiring)
    long_stop_level,
    should_exit_long,
)

from weinstein_short_core import (
    check_short_entry,
    ShortEntryParams,
)

from weinstein_filters import stock_ma30_slope_ok_from_snapshot

from market_regime import (
    MarketRegimeConfig,              # (placeholder; not used yet)
    build_historical_regime_table,   # (placeholder; not used yet)
)

# ✅ INDUSTRY FILTERS (PROD + SIM)
from industry_filters import (
    IndustryFilterConfig,
    enrich_with_industry_and_stats,
    industry_ok_from_row,
)

# =========================
# LOGGING
# =========================

VERBOSE = True


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


# =========================
# CONFIG
# =========================

def load_yaml_config(path: str = "./config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        log(f"Failed to load {path} — using defaults.", level="warn")
        return {}


# =========================
# WEEKLY / SNAPSHOT HELPERS
# =========================

WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_equities_"
WEEKLY_SNAPSHOT_DIR = "./data/weekly_snapshots"

_SNAPSHOT_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")


def newest_weekly_csv() -> str:
    files = [
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError("No weekly CSV found.")
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report() -> pd.DataFrame:
    path = newest_weekly_csv()
    log(f"Using weekly CSV: {path}", level="info")
    return pd.read_csv(path).rename(columns=str.lower)


def _parse_snapshot_date_from_name(fname: str) -> Optional[date]:
    m = _SNAPSHOT_NAME_RE.search(fname)
    if not m:
        return None
    token = m.group(1)
    try:
        return (
            datetime.strptime(token, "%Y%m%d").date()
            if len(token) == 8
            else datetime.strptime(token, "%Y-%m-%d").date()
        )
    except Exception:
        return None


def load_weekly_snapshots(snapshot_dir: str) -> List[Tuple[date, pd.DataFrame]]:
    if not os.path.isdir(snapshot_dir):
        return []

    out: List[Tuple[date, pd.DataFrame]] = []
    for fname in os.listdir(snapshot_dir):
        if not (fname.startswith(WEEKLY_FILE_PREFIX) and fname.endswith(".csv")):
            pass

    out = []
    for fname in os.listdir(snapshot_dir):
        if not (fname.startswith(WEEKLY_FILE_PREFIX) and fname.endswith(".csv")):
            continue
        d = _parse_snapshot_date_from_name(fname)
        if not d:
            continue
        df = pd.read_csv(os.path.join(snapshot_dir, fname)).rename(columns=str.lower)
        out.append((d, df))

    out.sort(key=lambda x: x[0])
    if out:
        log(f"Loaded {len(out)} weekly snapshots ({out[0][0]} → {out[-1][0]}).", level="info")
    return out


def pick_snapshot_for_date(
    snapshots: List[Tuple[date, pd.DataFrame]],
    as_of_ts: pd.Timestamp,
) -> Optional[Tuple[date, pd.DataFrame]]:
    chosen = None
    for d, df in snapshots:
        if d <= as_of_ts.date():
            chosen = (d, df)
        else:
            break
    return chosen


# =========================
# DAILY DATA HELPERS
# =========================

def download_daily_bars(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV using yfinance.
    Adds padding before start so rolling indicators have enough history.
    """
    start_dt = datetime.fromisoformat(start)
    pad_start = (start_dt - timedelta(days=365)).strftime("%Y-%m-%d")

    tickers = sorted(set([t for t in tickers if isinstance(t, str) and t.strip()]))
    log(f"Downloading daily bars for {len(tickers)} tickers ({pad_start} → {end})...", level="step")
    df = yf.download(
        tickers=tickers,
        start=pad_start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError("No daily data returned from yfinance.")

    # Ensure MultiIndex columns: (field, ticker)
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, ["SINGLE"]])

    log("Daily download complete.", level="ok")
    return df


def get_panel(daily_df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    try:
        return daily_df[(field, ticker)].dropna()
    except Exception:
        return pd.Series(dtype="float64")


def compute_atr_series_from_ohlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    return atr


# =========================
# REPORTING / OUTPUT
# =========================

OUTPUT_DIR = "./output"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_outdir(path: str = OUTPUT_DIR):
    os.makedirs(path, exist_ok=True)


def _safe_close(daily_df: pd.DataFrame, dt: pd.Timestamp, ticker: str) -> float:
    try:
        if ("Close", ticker) in daily_df.columns and dt in daily_df.index:
            v = daily_df.loc[dt, ("Close", ticker)]
            if pd.notna(v):
                return float(v)
    except Exception:
        pass
    return np.nan


def _positions_market_value(
    daily_df: pd.DataFrame,
    dt: pd.Timestamp,
    positions: Dict[str, "Position"],
) -> float:
    mv = 0.0
    for _, p in positions.items():
        px = _safe_close(daily_df, dt, p.ticker)
        if pd.notna(px):
            if p.side == "short":
                mv -= float(p.qty) * float(px)
            else:
                mv += float(p.qty) * float(px)
    return float(mv)


def _gross_exposure(
    daily_df: pd.DataFrame,
    dt: pd.Timestamp,
    positions: Dict[str, "Position"],
) -> float:
    ex = 0.0
    for _, p in positions.items():
        px = _safe_close(daily_df, dt, p.ticker)
        if pd.notna(px):
            ex += abs(float(p.qty) * float(px))
    return float(ex)


def _equity(daily_df: pd.DataFrame, dt: pd.Timestamp, cash: float, positions: Dict[str, "Position"]) -> float:
    return float(cash) + _positions_market_value(daily_df, dt, positions)


def _trades_to_df(trades: List["Trade"]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "ticker", "side", "entry_date", "exit_date",
                "entry_price", "exit_price", "qty", "pnl", "pnl_pct",
            ]
        )
    rows = []
    for t in trades:
        rows.append({
            "ticker": t.ticker,
            "side": t.side,
            "entry_date": pd.to_datetime(t.entry_date),
            "exit_date": pd.to_datetime(t.exit_date),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "qty": int(t.qty),
            "pnl": float(t.pnl),
            "pnl_pct": float(t.pnl_pct),
        })
    df = pd.DataFrame(rows).sort_values(["exit_date", "ticker"]).reset_index(drop=True)
    return df


def _equity_to_df(equity_curve: List[Tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    if not equity_curve:
        return pd.DataFrame(columns=["date", "equity"])
    df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    df["date"] = pd.to_datetime(df["date"])
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    return df


def _monthly_breakdown(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["month", "pnl", "trades", "win_rate", "equity_end"])

    tdf = trades_df.copy()
    tdf["month"] = tdf["exit_date"].dt.to_period("M").astype(str)
    g = tdf.groupby("month", dropna=False)

    out = pd.DataFrame({
        "pnl": g["pnl"].sum(),
        "trades": g.size(),
        "win_rate": (g["pnl"].apply(lambda s: float((s > 0).mean())) * 100.0),
    }).reset_index()

    equity_end_map = {}
    if not equity_df.empty:
        edf = equity_df.copy()
        edf["month"] = edf["date"].dt.to_period("M").astype(str)
        equity_end = edf.sort_values("date").groupby("month", as_index=False).tail(1)
        equity_end_map = dict(zip(equity_end["month"], equity_end["equity"]))

    out["equity_end"] = out["month"].map(equity_end_map)
    return out.sort_values("month").reset_index(drop=True)


def _performance_summary(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df is None or equity_df.empty:
        return pd.DataFrame([{
            "start_equity": np.nan,
            "end_equity": np.nan,
            "years": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "ann_vol": np.nan,
            "sharpe0": np.nan,
        }])

    df = equity_df.sort_values("date").dropna()
    if df.empty:
        return pd.DataFrame([{
            "start_equity": np.nan,
            "end_equity": np.nan,
            "years": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "ann_vol": np.nan,
            "sharpe0": np.nan,
        }])

    start_eq = float(df["equity"].iloc[0])
    end_eq = float(df["equity"].iloc[-1])

    start_dt = pd.to_datetime(df["date"].iloc[0]).to_pydatetime()
    end_dt = pd.to_datetime(df["date"].iloc[-1]).to_pydatetime()
    days = max(1, (end_dt - start_dt).days)
    years = days / 365.25

    cagr = np.nan
    if start_eq > 0 and years > 0:
        cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0

    eq = df["equity"].astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan

    rets = eq.pct_change().dropna()
    ann_vol = float(rets.std(ddof=0) * math.sqrt(252)) if len(rets) > 2 else np.nan
    sharpe0 = np.nan
    if len(rets) > 2 and rets.std(ddof=0) > 1e-12:
        sharpe0 = float((rets.mean() / rets.std(ddof=0)) * math.sqrt(252))

    return pd.DataFrame([{
        "start_equity": start_eq,
        "end_equity": end_eq,
        "years": years,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "ann_vol": ann_vol,
        "sharpe0": sharpe0,
    }])


def _write_reports(*, tag: str, trades: List["Trade"], equity_curve: List[Tuple[pd.Timestamp, float]]):
    _ensure_outdir(OUTPUT_DIR)

    trades_df = _trades_to_df(trades)
    equity_df = _equity_to_df(equity_curve)
    monthly_df = _monthly_breakdown(trades_df, equity_df)
    perf_df = _performance_summary(equity_df)

    trades_path = os.path.join(OUTPUT_DIR, f"live_logic_bt_trades_{tag}.csv")
    equity_png = os.path.join(OUTPUT_DIR, f"live_logic_bt_equity_{tag}.png")
    monthly_path = os.path.join(OUTPUT_DIR, f"live_logic_bt_monthly_{tag}.csv")
    perf_path = os.path.join(OUTPUT_DIR, f"live_logic_bt_perf_{tag}.csv")

    trades_df.to_csv(trades_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    perf_df.to_csv(perf_path, index=False)

    if not equity_df.empty:
        plt.figure()
        plt.plot(equity_df["date"], equity_df["equity"])
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(equity_png, dpi=140)
        plt.close()

    log(f"Wrote trade log → {trades_path}", level="ok")
    if not equity_df.empty:
        log(f"Wrote equity curve PNG → {equity_png}", level="ok")
    log(f"Wrote monthly P/L breakdown → {monthly_path}", level="ok")
    log(f"Wrote performance summary → {perf_path}", level="ok")

    if not monthly_df.empty:
        log("Monthly P/L summary:", level="info")
        for _, r in monthly_df.iterrows():
            m = r["month"]
            pnl = float(r["pnl"])
            tr = int(r["trades"])
            wr = float(r["win_rate"])
            eqe = r.get("equity_end", np.nan)
            eq_s = f"${float(eqe):,.2f}" if pd.notna(eqe) else "$nan"
            log(f"  {m}: PnL=${pnl:,.2f} | Trades={tr} | WinRate={wr:5.1f}% | Equity={eq_s}", level="info")

    try:
        p = perf_df.iloc[0].to_dict()
        cagr = p.get("cagr", np.nan)
        mdd = p.get("max_drawdown", np.nan)
        vol = p.get("ann_vol", np.nan)
        sh = p.get("sharpe0", np.nan)
        log(
            f"Perf: CAGR={cagr*100:,.2f}% | MaxDD={mdd*100:,.2f}% | AnnVol={vol*100:,.2f}% | Sharpe0={sh:,.2f}",
            level="info",
        )
    except Exception:
        pass


# =========================
# WEINSTEIN HELPERS
# =========================

def _stage_num(row: pd.Series) -> Optional[int]:
    v = row.get("stage", None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        fv = float(v)
        if np.isnan(fv):
            return None
        iv = int(round(fv))
        if iv in (1, 2, 3, 4):
            return iv
    except Exception:
        pass
    s = str(v).strip().lower()
    if s in ("1", "1.0", "stage1", "stage 1"):
        return 1
    if s in ("2", "2.0", "stage2", "stage 2"):
        return 2
    if s in ("3", "3.0", "stage3", "stage 3"):
        return 3
    if s in ("4", "4.0", "stage4", "stage 4"):
        return 4
    if "stage" in s:
        for k in ("1", "2", "3", "4"):
            if k in s:
                return int(k)
    return None


def _is_stage2(row: pd.Series) -> bool:
    return _stage_num(row) == 2


def _is_stage4(row: pd.Series) -> bool:
    return _stage_num(row) == 4


def short_slope_ok_from_snapshot(row: pd.Series, short_cfg: Mapping) -> bool:
    """
    Proper short slope gate (optional).

    Config (under backtest.short):
      require_ma30_falling: bool (default False)
      ma30_slope_max: float (default 0.0)  # must be <= this (e.g. 0.0 or -0.05)
    """
    require = bool(short_cfg.get("require_ma30_falling", False))
    if not require:
        return True

    ma30_slope_max = float(short_cfg.get("ma30_slope_max", 0.0))

    for col in ("ma30_slope_per_wk", "ma_slope_per_wk", "ma30_slope"):
        if col in row.index:
            v = row.get(col)
            try:
                fv = float(v)
                if np.isfinite(fv):
                    return fv <= ma30_slope_max
            except Exception:
                pass

    return False


def short_failed_rally_ok(
    close_series: pd.Series,
    dt: pd.Timestamp,
    short_cfg: Mapping,
) -> bool:
    """
    Optional short entry gate: "failed rally" (rollover) filter.

    If enabled (backtest.short.require_failed_rally=True):
      require close <= rolling_max(close, lookback) * (1 - failed_rally_pct)

    Config (under backtest.short):
      require_failed_rally: bool (default False)
      failed_rally_lookback: int (default 10)
      failed_rally_pct: float (default 0.02)  # 2% below recent lookback high
    """
    if not bool(short_cfg.get("require_failed_rally", False)):
        return True

    lookback = int(short_cfg.get("failed_rally_lookback", 10))
    pct = float(short_cfg.get("failed_rally_pct", 0.02))

    if lookback < 2 or pct <= 0:
        return False

    if close_series is None or close_series.empty:
        return False
    if dt not in close_series.index:
        return False

    roll_hi = close_series.rolling(lookback, min_periods=lookback).max()
    if dt not in roll_hi.index or pd.isna(roll_hi.loc[dt]):
        return False

    px = float(close_series.loc[dt])
    hi = float(roll_hi.loc[dt])
    if not np.isfinite(px) or not np.isfinite(hi) or hi <= 0:
        return False

    return px <= hi * (1.0 - pct)


def _market_allows_longs(daily_df: pd.DataFrame, dt: pd.Timestamp, market_cfg: Mapping) -> bool:
    require_rising = bool(market_cfg.get("require_rising_ma30", False))
    ma30_slope_min = float(market_cfg.get("ma30_slope_min", 0.0))

    vix_max = market_cfg.get("vix_max", None)
    try:
        vix_max = float(vix_max) if vix_max is not None else None
    except Exception:
        vix_max = None

    if vix_max is not None:
        if ("Close", "^VIX") not in daily_df.columns or dt not in daily_df.index:
            return False
        vix = daily_df.loc[dt, ("Close", "^VIX")]
        if pd.isna(vix):
            return False
        if float(vix) > float(vix_max):
            return False

    if require_rising:
        if ("Close", "SPY") not in daily_df.columns:
            return False
        spy = daily_df[("Close", "SPY")].dropna()
        if dt not in spy.index:
            return False

        ma150 = spy.rolling(150, min_periods=150).mean()
        if dt not in ma150.index or pd.isna(ma150.loc[dt]):
            return False

        prev = ma150.shift(5)
        if dt not in prev.index or pd.isna(prev.loc[dt]):
            return False

        slope = float(ma150.loc[dt] - prev.loc[dt])
        if slope < float(ma30_slope_min):
            return False

    return True


def _market_allows_shorts(daily_df: pd.DataFrame, dt: pd.Timestamp, market_cfg: Mapping) -> bool:
    require_falling = bool(market_cfg.get("require_falling_ma30", False))
    ma30_slope_min_short = float(market_cfg.get("ma30_slope_min_short", 0.0))

    if not require_falling:
        return True

    if ("Close", "SPY") not in daily_df.columns:
        return False

    spy = daily_df[("Close", "SPY")].dropna()
    if dt not in spy.index:
        return False

    ma150 = spy.rolling(150, min_periods=150).mean()
    if dt not in ma150.index or pd.isna(ma150.loc[dt]):
        return False

    # ✅ Require SPY below MA proxy for shorts
    if float(spy.loc[dt]) >= float(ma150.loc[dt]):
        return False

    prev = ma150.shift(5)
    if dt not in prev.index or pd.isna(prev.loc[dt]):
        return False

    slope = float(ma150.loc[dt] - prev.loc[dt])

    # ✅ “falling” means slope <= threshold (0.0 default)
    if slope > float(ma30_slope_min_short):
        return False

    return True


def _parse_boolish(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and np.isfinite(v):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def _get_snapshot_rs_above_ma(row: pd.Series) -> Optional[bool]:
    # try common column names; return None if not found / not parseable
    for col in ("rs_above_ma", "rs_above_ma30", "rs_above"):
        if col in row.index:
            b = _parse_boolish(row.get(col))
            if b is not None:
                return b
    return None


# =========================
# SHORT HELPERS
# =========================

def short_stop_level(price: float, atr: float, ma: float, *, stop_hard_pct: float, trail_atr: float, ma_guard: float) -> float:
    cands = []
    if price > 0 and stop_hard_pct > 0:
        cands.append(price * (1.0 + stop_hard_pct))
    if ma > 0 and ma_guard >= 0:
        cands.append(ma * (1.0 + ma_guard))
    if atr > 0 and trail_atr > 0:
        cands.append(price + atr * trail_atr)

    cands = [x for x in cands if np.isfinite(x) and x > price]
    if not cands:
        return np.nan
    return float(min(cands))


def should_exit_short(price: float, stop: float, ma: float) -> bool:
    if np.isfinite(stop) and price >= stop:
        return True
    if np.isfinite(ma) and price >= ma:
        return True
    return False


# =========================
# BACKTEST DATA STRUCTURES
# =========================

@dataclass
class Position:
    ticker: str
    side: str            # "long" | "short"
    qty: int
    entry_price: float
    stop: float
    atr: float
    opened: pd.Timestamp


@dataclass
class Trade:
    ticker: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    pnl_pct: float


# =========================
# BACKTEST ENGINE
# =========================

def backtest(
    *,
    daily_df: pd.DataFrame,
    start: str,
    end: str,
    capital: float,
    risk_per_trade: float,
    max_long: int,
    max_short: int,
    mode: str,
    universe_tickers: List[str],
    weekly_df: Optional[pd.DataFrame],
    weekly_snapshots: Optional[List[Tuple[date, pd.DataFrame]]],
    regime_table: Optional[pd.DataFrame],
    long_logic_cfg: Mapping,
    short_logic_cfg: Mapping,
    market_cfg: Mapping,
    industry_cfg: Mapping,
    max_leverage: float = 1.0,
    max_pos_frac: float = 0.25,
):
    industry_filter_cfg = IndustryFilterConfig(**(industry_cfg or {}))

    if weekly_df is not None and not weekly_df.empty:
        weekly_df = enrich_with_industry_and_stats(weekly_df, cfg=industry_filter_cfg)

    if weekly_snapshots:
        weekly_snapshots = [
            (d, enrich_with_industry_and_stats(df, cfg=industry_filter_cfg))
            for d, df in weekly_snapshots
        ]

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    cash = float(capital)
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []

    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    last_progress_month: Optional[str] = None

    # caches
    close_cache: Dict[str, pd.Series] = {}
    vol_cache: Dict[str, pd.Series] = {}
    ma_cache: Dict[str, pd.Series] = {}
    atr_series_cache: Dict[str, pd.Series] = {}
    vol_mult_cache: Dict[str, pd.Series] = {}

    for t in universe_tickers:
        close = get_panel(daily_df, "Close", t)
        high = get_panel(daily_df, "High", t)
        low = get_panel(daily_df, "Low", t)
        vol = get_panel(daily_df, "Volume", t)
        if close.empty or high.empty or low.empty or vol.empty:
            continue

        close_cache[t] = close
        vol_cache[t] = vol
        ma_cache[t] = close.rolling(30, min_periods=30).mean()
        atr_series_cache[t] = compute_atr_series_from_ohlc(high, low, close, n=14)

        v50 = vol.rolling(50, min_periods=50).mean()
        vol_mult_cache[t] = vol / v50

    _ = LongEntryParams(
        min_break_pct=float(long_logic_cfg.get("break_pct", 0.004)),
        dist_above_ma_min=0.0,
        vol_min=float(long_logic_cfg.get("vol_min", 1.3)),
        adx_min=float(long_logic_cfg.get("adx_min", ADX_MIN)),
    )

    sh_stop_hard = float(short_logic_cfg.get("stop_hard", short_logic_cfg.get("stop_hard_pct", 0.20)))
    sh_trail_atr = float(short_logic_cfg.get("trail_atr", 2.0))
    sh_ma_guard = float(short_logic_cfg.get("ma_guard", 0.03))

    # Short entry gates (wired)
    sh_break_pct = float(short_logic_cfg.get("break_pct", 0.006))
    sh_vol_min = float(short_logic_cfg.get("vol_min", 1.10))
    sh_pivot_lb = int(short_logic_cfg.get("pivot_lookback_days", short_logic_cfg.get("pivot_lookback", 50)))
    if sh_pivot_lb < 10:
        sh_pivot_lb = 50

    all_dates = [pd.Timestamp(d) for d in daily_df.index if isinstance(d, (pd.Timestamp, datetime))]

    # Diagnostics for why shorts don’t fire (month-to-date)
    short_diag = {
        "stage4": 0,
        "short_slope_fail": 0,
        "failed_rally_fail": 0,
        "industry_fail": 0,
        "no_bars": 0,
        "px_not_below_ma": 0,
        "no_breakdown": 0,
        "vol_too_low": 0,
        "rs_too_strong": 0,
        "sized_zero": 0,
        "entered": 0,
    }

    for dt in all_dates:
        if dt < start_dt or dt > end_dt:
            continue

        snap = pick_snapshot_for_date(weekly_snapshots, dt) if weekly_snapshots else None
        universe = snap[1] if snap else weekly_df
        if universe is None or universe.empty:
            continue

        # TRAIL STOPS
        for key, pos in list(positions.items()):
            t = pos.ticker
            if ("Close", t) not in daily_df.columns:
                continue
            if t not in ma_cache or t not in atr_series_cache:
                continue
            if dt not in ma_cache[t].index or dt not in atr_series_cache[t].index:
                continue

            px = daily_df.loc[dt, ("Close", t)]
            ma = ma_cache[t].loc[dt]
            atr = atr_series_cache[t].loc[dt]
            if pd.isna(px) or pd.isna(ma) or pd.isna(atr):
                continue

            px_f = float(px)
            ma_f = float(ma)
            atr_f = float(atr)

            if pos.side == "long":
                new_stop = long_stop_level(px_f, atr_f, ma_f)
                if np.isfinite(new_stop):
                    pos.stop = float(max(pos.stop, new_stop))
                pos.atr = atr_f
            else:
                new_stop = short_stop_level(
                    px_f, atr_f, ma_f,
                    stop_hard_pct=sh_stop_hard,
                    trail_atr=sh_trail_atr,
                    ma_guard=sh_ma_guard,
                )
                if np.isfinite(new_stop):
                    pos.stop = float(min(pos.stop, new_stop))
                pos.atr = atr_f

            positions[key] = pos

        # EXITS
        to_close: List[str] = []
        for key, pos in positions.items():
            t = pos.ticker
            if (("Close", t) not in daily_df.columns) or (t not in ma_cache):
                continue
            if dt not in ma_cache[t].index:
                continue

            price = daily_df.loc[dt, ("Close", t)]
            if pd.isna(price):
                continue
            ma_val = ma_cache[t].loc[dt]
            ma_f = float(ma_val) if pd.notna(ma_val) else np.nan
            px_f = float(price)

            if pos.side == "long":
                if should_exit_long(px_f, float(pos.stop), ma_f):
                    exit_price = px_f
                    proceeds = float(pos.qty) * exit_price
                    entry_cost = float(pos.qty) * float(pos.entry_price)

                    pnl = proceeds - entry_cost
                    pnl_pct = pnl / entry_cost if entry_cost > 0 else 0.0

                    cash += proceeds

                    trades.append(
                        Trade(
                            ticker=t,
                            side="long",
                            entry_date=pos.opened,
                            exit_date=dt,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            qty=pos.qty,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )
                    to_close.append(key)
            else:
                if should_exit_short(px_f, float(pos.stop), ma_f):
                    cover_price = px_f
                    cover_cost = float(pos.qty) * cover_price
                    entry_proceeds = float(pos.qty) * float(pos.entry_price)

                    pnl = entry_proceeds - cover_cost
                    pnl_pct = pnl / entry_proceeds if entry_proceeds > 0 else 0.0

                    cash -= cover_cost

                    trades.append(
                        Trade(
                            ticker=t,
                            side="short",
                            entry_date=pos.opened,
                            exit_date=dt,
                            entry_price=pos.entry_price,
                            exit_price=cover_price,
                            qty=pos.qty,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )
                    to_close.append(key)

        for key in to_close:
            del positions[key]

        # ENTRIES
        eq_now = _equity(daily_df, dt, cash, positions)

        allow_new_longs = _market_allows_longs(daily_df, dt, market_cfg) if market_cfg else True
        allow_new_shorts = _market_allows_shorts(daily_df, dt, market_cfg) if market_cfg else True

        gross_expo = _gross_exposure(daily_df, dt, positions)
        buying_power = max(0.0, float(max_leverage) * eq_now - gross_expo)

        do_longs = mode in ("long", "both", "auto")
        do_shorts = mode in ("short", "both", "auto")

        n_long_now = sum(1 for p in positions.values() if p.side == "long")
        n_short_now = sum(1 for p in positions.values() if p.side == "short")

        # LONG ENTRIES
        if do_longs and allow_new_longs and n_long_now < max_long and buying_power > 0:
            for _, row in universe.iterrows():
                if n_long_now >= max_long:
                    break
                t = str(row.get("ticker", "")).upper().strip()
                if not t:
                    continue
                pos_key = f"{t}_long"
                if pos_key in positions:
                    continue

                if not _is_stage2(row):
                    continue

                if not stock_ma30_slope_ok_from_snapshot(row, long_logic_cfg):
                    continue

                if not industry_ok_from_row(row, cfg=industry_filter_cfg):
                    continue

                if ("Close", t) not in daily_df.columns:
                    continue
                if t not in ma_cache or t not in atr_series_cache:
                    continue
                if dt not in ma_cache[t].index or dt not in atr_series_cache[t].index:
                    continue

                price = daily_df.loc[dt, ("Close", t)]
                ma_val = ma_cache[t].loc[dt]
                atr_val = atr_series_cache[t].loc[dt]
                if pd.isna(price) or pd.isna(ma_val) or pd.isna(atr_val):
                    continue

                price_f = float(price)
                ma_f = float(ma_val)
                atr_f = float(atr_val)

                if price_f <= ma_f:
                    continue

                stop = long_stop_level(price_f, atr_f, ma_f)
                if np.isnan(stop) or stop >= price_f:
                    continue

                per_share_risk = price_f - float(stop)
                if per_share_risk <= 0:
                    continue

                risk_amt = float(eq_now) * float(risk_per_trade)
                qty_risk = int(math.floor(risk_amt / per_share_risk))
                if qty_risk <= 0:
                    continue

                max_pos_value = float(max_pos_frac) * float(eq_now)
                qty_cap_pos = int(math.floor(max_pos_value / price_f)) if price_f > 0 else 0
                qty_cap_bp = int(math.floor(buying_power / price_f)) if price_f > 0 else 0

                qty = max(0, min(qty_risk, qty_cap_pos, qty_cap_bp))
                if qty <= 0:
                    continue

                cost = float(qty) * price_f
                if cost <= 0 or cost > buying_power + 1e-9:
                    continue

                cash -= cost
                buying_power -= cost

                positions[pos_key] = Position(
                    ticker=t,
                    side="long",
                    qty=qty,
                    entry_price=price_f,
                    stop=float(stop),
                    atr=atr_f,
                    opened=pd.Timestamp(dt),
                )
                n_long_now += 1

        # SHORT ENTRIES (now wired to pivot + vol_min + weak RS + optional failed rally)
        if do_shorts and allow_new_shorts and n_short_now < max_short and buying_power > 0:
            for _, row in universe.iterrows():
                if n_short_now >= max_short:
                    break
                t = str(row.get("ticker", "")).upper().strip()
                if not t:
                    continue
                pos_key = f"{t}_short"
                if pos_key in positions:
                    continue

                if not _is_stage4(row):
                    continue
                short_diag["stage4"] += 1

                if not short_slope_ok_from_snapshot(row, short_logic_cfg):
                    short_diag["short_slope_fail"] += 1
                    continue

                # ✅ Industry confirmation (respects backtest.industry.enabled)
                if not industry_ok_from_row(row, cfg=industry_filter_cfg):
                    short_diag["industry_fail"] += 1
                    continue

                if ("Close", t) not in daily_df.columns:
                    short_diag["no_bars"] += 1
                    continue
                if t not in ma_cache or t not in atr_series_cache or t not in close_cache or t not in vol_cache or t not in vol_mult_cache:
                    short_diag["no_bars"] += 1
                    continue
                if dt not in ma_cache[t].index or dt not in atr_series_cache[t].index or dt not in close_cache[t].index:
                    short_diag["no_bars"] += 1
                    continue

                # ✅ Optional failed-rally filter
                if not short_failed_rally_ok(close_cache[t], dt, short_logic_cfg):
                    short_diag["failed_rally_fail"] += 1
                    continue

                price = daily_df.loc[dt, ("Close", t)]
                ma_val = ma_cache[t].loc[dt]
                atr_val = atr_series_cache[t].loc[dt]
                if pd.isna(price) or pd.isna(ma_val) or pd.isna(atr_val):
                    short_diag["no_bars"] += 1
                    continue

                price_f = float(price)
                ma_f = float(ma_val)
                atr_f = float(atr_val)

                # Pivot low: last N PRIOR closes (exclude today's close!)
                cs = close_cache[t]
                hist = cs.loc[:dt].iloc[:-1]  # <-- critical fix: exclude current bar
                tail = hist.tail(sh_pivot_lb)
                if len(tail) < sh_pivot_lb:
                    short_diag["no_bars"] += 1
                    continue
                pivot_low = float(tail.min())

                # Volume multiple vs 50d avg
                vm = vol_mult_cache[t]
                if dt not in vm.index or pd.isna(vm.loc[dt]):
                    short_diag["no_bars"] += 1
                    continue
                vol_mult = float(vm.loc[dt])

                # Weak RS gate (only if available in snapshot row; else default to False == "not strong")
                rs_above_ma = _get_snapshot_rs_above_ma(row)
                if rs_above_ma is None:
                    rs_above_ma = False

                res = check_short_entry(
                    price=price_f,
                    ma_val=ma_f,
                    pivot_low=pivot_low,
                    rs_above_ma=bool(rs_above_ma),
                    vol_mult=vol_mult,
                    params=ShortEntryParams(min_break_pct=sh_break_pct, vol_min=sh_vol_min),
                )

                if not res.can_enter:
                    if res.reason == "price_not_below_ma":
                        short_diag["px_not_below_ma"] += 1
                    elif res.reason == "no_breakdown_vs_pivot":
                        short_diag["no_breakdown"] += 1
                    elif res.reason == "volume_too_low":
                        short_diag["vol_too_low"] += 1
                    elif res.reason == "rs_too_strong_for_short":
                        short_diag["rs_too_strong"] += 1
                    else:
                        short_diag["no_bars"] += 1
                    continue

                stop = short_stop_level(
                    price_f, atr_f, ma_f,
                    stop_hard_pct=sh_stop_hard,
                    trail_atr=sh_trail_atr,
                    ma_guard=sh_ma_guard,
                )
                if np.isnan(stop) or stop <= price_f:
                    short_diag["px_not_below_ma"] += 1
                    continue

                per_share_risk = float(stop) - price_f
                if per_share_risk <= 0:
                    short_diag["px_not_below_ma"] += 1
                    continue

                risk_amt = float(eq_now) * float(risk_per_trade)
                qty_risk = int(math.floor(risk_amt / per_share_risk))
                if qty_risk <= 0:
                    short_diag["sized_zero"] += 1
                    continue

                max_pos_value = float(max_pos_frac) * float(eq_now)
                qty_cap_pos = int(math.floor(max_pos_value / price_f)) if price_f > 0 else 0
                qty_cap_bp = int(math.floor(buying_power / price_f)) if price_f > 0 else 0

                qty = max(0, min(qty_risk, qty_cap_pos, qty_cap_bp))
                if qty <= 0:
                    short_diag["sized_zero"] += 1
                    continue

                proceeds = float(qty) * price_f
                if proceeds <= 0 or proceeds > buying_power + 1e-9:
                    short_diag["sized_zero"] += 1
                    continue

                cash += proceeds
                buying_power -= proceeds

                positions[pos_key] = Position(
                    ticker=t,
                    side="short",
                    qty=qty,
                    entry_price=price_f,
                    stop=float(stop),
                    atr=atr_f,
                    opened=pd.Timestamp(dt),
                )
                n_short_now += 1
                short_diag["entered"] += 1

        eq = _equity(daily_df, dt, cash, positions)
        equity_curve.append((dt, eq))

        month_key = dt.strftime("%Y-%m")
        if last_progress_month is None:
            last_progress_month = month_key
        if month_key != last_progress_month:
            last_progress_month = month_key
            log(
                f"Progress: {dt.date()} — equity ${eq:,.2f}, positions: {len(positions)} "
                f"(L={n_long_now}, S={n_short_now}), trades so far: {len(trades)}",
                level="debug",
            )
            if mode in ("short", "both", "auto"):
                log(
                    "Short diag (month-to-date): "
                    f"stage4={short_diag['stage4']} "
                    f"slope_fail={short_diag['short_slope_fail']} "
                    f"failed_rally_fail={short_diag['failed_rally_fail']} "
                    f"industry_fail={short_diag['industry_fail']} "
                    f"no_bars={short_diag['no_bars']} "
                    f"px_not_below_ma={short_diag['px_not_below_ma']} "
                    f"no_breakdown={short_diag['no_breakdown']} "
                    f"vol_low={short_diag['vol_too_low']} "
                    f"rs_strong={short_diag['rs_too_strong']} "
                    f"sized0={short_diag['sized_zero']} "
                    f"entered={short_diag['entered']}",
                    level="debug",
                )
                for k in list(short_diag.keys()):
                    short_diag[k] = 0

    final_eq = float(equity_curve[-1][1]) if equity_curve else _equity(daily_df, end_dt, cash, positions)

    return {
        "positions": positions,
        "trades": trades,
        "final_equity": final_eq,
        "equity_curve": equity_curve,
        "cash": cash,
    }


# =========================
# CLI
# =========================

def main():
    global VERBOSE

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--mode", type=str, default="auto", choices=["long", "short", "both", "auto", "none"])
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--max-long", type=int, default=10)
    ap.add_argument("--max-short", type=int, default=6)
    ap.add_argument("--snapshot-mode", type=str, choices=["static", "historical", "auto"], default="auto")
    ap.add_argument("--config", type=str, default="./config.yaml")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--max-pos-frac", type=float, default=0.25)

    args = ap.parse_args()
    VERBOSE = not args.quiet

    cfg = load_yaml_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}

    bt_long_cfg = bt_cfg.get("long", {}) or {}
    bt_short_cfg = bt_cfg.get("short", {}) or {}
    market_cfg = bt_cfg.get("market", {}) or {}
    industry_cfg = bt_cfg.get("industry", {}) or {}

    log(
        f"Industry filters enabled={industry_cfg.get('enabled', False)} "
        f"min_stage2_frac={industry_cfg.get('min_stage2_frac', 'n/a')}",
        level="info",
    )
    log(
        f"Mode={args.mode} | market: rise_ma30={market_cfg.get('require_rising_ma30', False)} "
        f"fall_ma30={market_cfg.get('require_falling_ma30', False)}",
        level="info",
    )
    if args.mode in ("short", "both", "auto"):
        log(
            f"Short entry gates: break_pct={bt_short_cfg.get('break_pct', 'n/a')} "
            f"vol_min={bt_short_cfg.get('vol_min', 'n/a')} "
            f"pivot_lb={bt_short_cfg.get('pivot_lookback_days', bt_short_cfg.get('pivot_lookback', 50))}",
            level="info",
        )

    weekly_df: Optional[pd.DataFrame] = None
    weekly_snapshots: Optional[List[Tuple[date, pd.DataFrame]]] = None
    all_tickers: set[str] = set()

    if args.snapshot_mode == "historical":
        weekly_snapshots = load_weekly_snapshots(WEEKLY_SNAPSHOT_DIR)
        if not weekly_snapshots:
            raise SystemExit("snapshot_mode=historical but no snapshots found.")
        for _, df in weekly_snapshots:
            if "ticker" in df.columns:
                all_tickers.update(df["ticker"].astype(str).str.upper())

    elif args.snapshot_mode == "auto":
        tmp = load_weekly_snapshots(WEEKLY_SNAPSHOT_DIR)
        if tmp:
            weekly_snapshots = tmp
            for _, df in weekly_snapshots:
                if "ticker" in df.columns:
                    all_tickers.update(df["ticker"].astype(str).str.upper())
            log(f"[auto] Using snapshots (unique tickers={len(all_tickers)}).", level="info")
        else:
            weekly_df = load_weekly_report()
            all_tickers.update(weekly_df["ticker"].astype(str).str.upper())
            log(f"[auto] No snapshots; using latest weekly (tickers={len(all_tickers)}).", level="info")

    else:
        weekly_df = load_weekly_report()
        all_tickers.update(weekly_df["ticker"].astype(str).str.upper())
        log(f"snapshot_mode=static: tickers={len(all_tickers)}.", level="info")

    if not all_tickers:
        raise SystemExit("No tickers found in weekly universe.")

    if bool(market_cfg.get("require_rising_ma30", False)) or bool(market_cfg.get("require_falling_ma30", False)):
        all_tickers.add("SPY")
    if market_cfg.get("vix_max", None) is not None:
        all_tickers.add("^VIX")

    daily_df = download_daily_bars(sorted(all_tickers), args.start, args.end)

    regime_table = None

    result = backtest(
        daily_df=daily_df,
        start=args.start,
        end=args.end,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        max_long=args.max_long,
        max_short=args.max_short,
        mode=args.mode,
        universe_tickers=sorted(all_tickers),
        weekly_df=weekly_df,
        weekly_snapshots=weekly_snapshots,
        regime_table=regime_table,
        long_logic_cfg=bt_long_cfg,
        short_logic_cfg=bt_short_cfg,
        market_cfg=market_cfg,
        industry_cfg=industry_cfg,
        max_leverage=float(args.max_leverage),
        max_pos_frac=float(args.max_pos_frac),
    )

    final_eq = float(result["final_equity"])
    pnl = final_eq - float(args.capital)
    pnl_pct = (pnl / float(args.capital) * 100.0) if float(args.capital) != 0 else 0.0
    trades = result.get("trades", []) or []
    equity_curve = result.get("equity_curve", []) or []

    log(
        f"Backtest complete. Final equity: ${final_eq:,.2f} "
        f"(P/L ${pnl:,.2f}, {pnl_pct:,.2f}%) — Trades: {len(trades)}",
        level="ok",
    )

    tag = _now_tag()
    _write_reports(tag=tag, trades=trades, equity_curve=equity_curve)

    log(
        f"Done. Open positions={len(result['positions'])}, trades={len(trades)}, "
        f"final_equity=${final_eq:,.2f}",
        level="ok",
    )


if __name__ == "__main__":
    main()
