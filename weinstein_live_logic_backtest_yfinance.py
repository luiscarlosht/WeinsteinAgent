#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Live Logic Backtest (daily approximation of intraday watchers)

This version (LONG FIXES):
- ✅ Proper cash accounting: subtract entry cost, add exit proceeds
- ✅ Equity = cash + market value of open positions (mark-to-market)
- ✅ Risk sizing uses EQUITY (not cash_like)
- ✅ Caps:
    - max_leverage (default 1.0)
    - max_pos_frac per position (default 0.25)
- ✅ Restores monthly progress logging
- ✅ Final summary + outputs:
    - trades CSV
    - equity curve PNG
    - monthly breakdown CSV
- ✅ Keeps Industry filters (single source of truth)

NOTE:
This still runs LONG only (no shorts yet). We’ll add SHORT after LONG is stable.
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
    compute_adx_series,
    ADX_WINDOW,
    ADX_MIN,
    compute_breadth_series_above_ma,
)

from weinstein_long_core import (
    LongEntryParams,
    check_long_entry,
    long_stop_level,
    should_exit_long,
)

from weinstein_filters import stock_ma30_slope_ok_from_snapshot

from market_regime import (
    MarketRegimeConfig,
    build_historical_regime_table,
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
        log("Failed to load config.yaml — using defaults.", level="warn")
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

    log(f"Downloading daily bars for {len(tickers)} tickers ({pad_start} → {end})...", level="step")
    df = yf.download(
        tickers=sorted(set(tickers)),
        start=pad_start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError("No daily data returned from yfinance.")

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
    """
    ATR using True Range:
      TR = max(High-Low, abs(High-prevClose), abs(Low-prevClose))
    ATR = SMA(TR, n)
    """
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


def _positions_market_value(daily_df: pd.DataFrame, dt: pd.Timestamp, positions: Dict[str, "Position"]) -> float:
    mv = 0.0
    for _, p in positions.items():
        px = _safe_close(daily_df, dt, p.ticker)
        if pd.notna(px):
            mv += float(p.qty) * float(px)
    return float(mv)


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


def _write_reports(*, tag: str, trades: List["Trade"], equity_curve: List[Tuple[pd.Timestamp, float]]):
    _ensure_outdir(OUTPUT_DIR)

    trades_df = _trades_to_df(trades)
    equity_df = _equity_to_df(equity_curve)
    monthly_df = _monthly_breakdown(trades_df, equity_df)

    trades_path = os.path.join(OUTPUT_DIR, f"live_logic_bt_trades_{tag}.csv")
    equity_png = os.path.join(OUTPUT_DIR, f"live_logic_bt_equity_{tag}.png")
    monthly_path = os.path.join(OUTPUT_DIR, f"live_logic_bt_monthly_{tag}.csv")

    trades_df.to_csv(trades_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

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

    if not monthly_df.empty:
        log("Monthly P/L summary:", level="info")
        for _, r in monthly_df.iterrows():
            m = r["month"]
            pnl = float(r["pnl"])
            tr = int(r["trades"])
            wr = float(r["win_rate"])
            eq = r.get("equity_end", np.nan)
            eq_s = f"${float(eq):,.2f}" if pd.notna(eq) else "$nan"
            log(f"  {m}: PnL=${pnl:,.2f} | Trades={tr} | WinRate={wr:5.1f}% | Equity={eq_s}", level="info")


# =========================
# BACKTEST DATA STRUCTURES
# =========================

@dataclass
class Position:
    ticker: str
    side: str
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
    mode: str,
    universe_tickers: List[str],
    weekly_df: Optional[pd.DataFrame],
    weekly_snapshots: Optional[List[Tuple[date, pd.DataFrame]]],
    regime_table: Optional[pd.DataFrame],
    long_logic_cfg: Mapping,
    market_cfg: Mapping,
    industry_cfg: Mapping,
    # NEW knobs (Weinstein-ish, safety first)
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

    ma_cache: Dict[str, pd.Series] = {}
    atr_series_cache: Dict[str, pd.Series] = {}

    for t in universe_tickers:
        close = get_panel(daily_df, "Close", t)
        high = get_panel(daily_df, "High", t)
        low = get_panel(daily_df, "Low", t)
        if close.empty or high.empty or low.empty:
            continue
        ma_cache[t] = close.rolling(30, min_periods=30).mean()
        atr_series_cache[t] = compute_atr_series_from_ohlc(high, low, close, n=14)

    _ = LongEntryParams(
        min_break_pct=float(long_logic_cfg.get("break_pct", 0.004)),
        dist_above_ma_min=0.0,
        vol_min=float(long_logic_cfg.get("vol_min", 1.3)),
        adx_min=float(long_logic_cfg.get("adx_min", ADX_MIN)),
    )

    all_dates = [pd.Timestamp(d) for d in daily_df.index if isinstance(d, (pd.Timestamp, datetime))]

    for dt in all_dates:
        if dt < start_dt or dt > end_dt:
            continue

        snap = pick_snapshot_for_date(weekly_snapshots, dt) if weekly_snapshots else None
        universe = snap[1] if snap else weekly_df
        if universe is None or universe.empty:
            continue

        # ==============
        # EXITS
        # ==============
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

            if should_exit_long(float(price), float(pos.stop), float(ma_val) if not pd.isna(ma_val) else np.nan):
                exit_price = float(price)
                proceeds = float(pos.qty) * exit_price
                entry_cost = float(pos.qty) * float(pos.entry_price)

                pnl = proceeds - entry_cost
                pnl_pct = pnl / entry_cost if entry_cost > 0 else 0.0

                # ✅ Real accounting
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

        for key in to_close:
            del positions[key]

        # ==============
        # ENTRIES (LONG only for now)
        # ==============
        if mode not in ("long", "both", "auto"):
            eq = _equity(daily_df, dt, cash, positions)
            equity_curve.append((dt, eq))
            continue

        n_long_now = sum(1 for p in positions.values() if p.side == "long")

        eq_now = _equity(daily_df, dt, cash, positions)
        # Buying power capped by leverage
        buying_power = max(0.0, float(max_leverage) * eq_now - _positions_market_value(daily_df, dt, positions))

        if n_long_now < max_long and buying_power > 0:
            for _, row in universe.iterrows():
                if n_long_now >= max_long:
                    break

                t = str(row.get("ticker", "")).upper().strip()
                if not t:
                    continue
                pos_key = f"{t}_long"
                if pos_key in positions:
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

                # Risk sizing uses equity
                risk_amt = float(eq_now) * float(risk_per_trade)

                qty_risk = int(math.floor(risk_amt / per_share_risk))
                if qty_risk <= 0:
                    continue

                # Cap per-position value
                max_pos_value = float(max_pos_frac) * float(eq_now)
                qty_cap_pos = int(math.floor(max_pos_value / price_f)) if price_f > 0 else 0

                # Cap by buying power (and cash if leverage=1)
                qty_cap_bp = int(math.floor(buying_power / price_f)) if price_f > 0 else 0

                qty = max(0, min(qty_risk, qty_cap_pos, qty_cap_bp))
                if qty <= 0:
                    continue

                cost = float(qty) * price_f
                if cost <= 0 or cost > buying_power + 1e-9:
                    continue

                # ✅ Real accounting on entry
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

        # ==============
        # EQUITY + MONTHLY PROGRESS
        # ==============
        eq = _equity(daily_df, dt, cash, positions)
        equity_curve.append((dt, eq))

        month_key = dt.strftime("%Y-%m")
        if last_progress_month is None:
            last_progress_month = month_key
        if month_key != last_progress_month:
            last_progress_month = month_key
            log(
                f"Progress: {dt.date()} — equity ${eq:,.2f}, positions: {len(positions)}, trades so far: {len(trades)}",
                level="debug",
            )

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
    ap.add_argument("--mode", type=str, default="auto", choices=["long", "both", "auto", "none"])
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--max-long", type=int, default=10)
    ap.add_argument("--snapshot-mode", type=str, choices=["static", "historical", "auto"], default="auto")
    ap.add_argument("--config", type=str, default="./config.yaml")
    ap.add_argument("--quiet", action="store_true")

    # Safety knobs
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--max-pos-frac", type=float, default=0.25)

    args = ap.parse_args()
    VERBOSE = not args.quiet

    cfg = load_yaml_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}

    bt_long_cfg = bt_cfg.get("long", {}) or {}
    market_cfg = bt_cfg.get("market", {}) or {}
    industry_cfg = bt_cfg.get("industry", {}) or {}

    log(
        f"Industry filters enabled={industry_cfg.get('enabled', False)} "
        f"min_stage2_frac={industry_cfg.get('min_stage2_frac', 'n/a')}",
        level="info",
    )

    # -------- Universe selection (static / historical / auto) --------
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

    daily_df = download_daily_bars(sorted(all_tickers), args.start, args.end)

    regime_table = None  # still placeholder

    result = backtest(
        daily_df=daily_df,
        start=args.start,
        end=args.end,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        max_long=args.max_long,
        mode=args.mode,
        universe_tickers=sorted(all_tickers),
        weekly_df=weekly_df,
        weekly_snapshots=weekly_snapshots,
        regime_table=regime_table,
        long_logic_cfg=bt_long_cfg,
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
