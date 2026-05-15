#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_replay_portfolio_backtest.py

Phase 2 refactor: portfolio research wrapper that consumes the PROD-like
signal replay stream instead of independently deciding entries.

Architecture:
  daily data + weekly snapshots
      -> weinstein_signal_replay_core.replay_signals(...)
      -> this wrapper applies portfolio accounting/sizing/exits

Purpose:
  Keep the strategy decision layer closer to PROD.  The wrapper only decides
  whether available cash/slots allow acting on a replayed signal.

Notes:
  - BUY/SHORT entries come from replay events.
  - Long exits can use replay SELL events (MA150 crack) for held tickers.
  - NEAR events are retained in the replay CSV but not traded by default.
  - This is intentionally separate from the legacy backtester while we validate
    parity before replacing the old research path.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from weinstein_live_logic_backtest_yfinance import (
    OUTPUT_DIR,
    WEEKLY_SNAPSHOT_DIR,
    build_sim_long_config,
    compute_atr_series_from_ohlc,
    download_daily_bars,
    get_panel,
    load_weekly_report,
    load_weekly_snapshots,
    load_yaml_config,
    log,
    pick_snapshot_for_date,
)
from weinstein_signal_replay_core import replay_signals, replay_summary
from weinstein_long_core import long_stop_level, should_exit_long
from weinstein_short_core import short_stop_level as core_short_stop_level, should_exit_short as core_should_exit_short


@dataclass
class Position:
    ticker: str
    side: str
    opened: pd.Timestamp
    entry_price: float
    qty: int
    stop: float
    atr: float = np.nan
    entry_reason: str = ""


@dataclass
class Trade:
    ticker: str
    side: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    pnl_pct: float
    entry_reason: str = ""
    exit_reason: str = ""


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(v, default=np.nan) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _close_at(daily_df: pd.DataFrame, dt: pd.Timestamp, ticker: str) -> float:
    try:
        v = daily_df.loc[dt, ("Close", ticker)]
        return _safe_float(v)
    except Exception:
        return np.nan


def _equity(daily_df: pd.DataFrame, dt: pd.Timestamp, cash: float, positions: Dict[str, Position]) -> float:
    eq = float(cash)
    for p in positions.values():
        px = _close_at(daily_df, dt, p.ticker)
        if not np.isfinite(px):
            px = p.entry_price
        if p.side == "long":
            eq += float(p.qty) * px
        else:
            # short equity contribution: entry proceeds minus mark-to-cover liability
            eq += float(p.qty) * (2.0 * p.entry_price - px)
    return float(eq)


def _gross_exposure(daily_df: pd.DataFrame, dt: pd.Timestamp, positions: Dict[str, Position]) -> float:
    gross = 0.0
    for p in positions.values():
        px = _close_at(daily_df, dt, p.ticker)
        if not np.isfinite(px):
            px = p.entry_price
        gross += abs(float(p.qty) * px)
    return float(gross)


def _build_indicator_caches(daily_df: pd.DataFrame, tickers: List[str]):
    close_cache, ma30_cache, ma150_cache, atr_cache = {}, {}, {}, {}
    for t in sorted(set(tickers)):
        close = get_panel(daily_df, "Close", t)
        high = get_panel(daily_df, "High", t)
        low = get_panel(daily_df, "Low", t)
        if close.empty or high.empty or low.empty:
            continue
        close_cache[t] = close
        ma30_cache[t] = close.rolling(30, min_periods=30).mean()
        ma150_cache[t] = close.rolling(150, min_periods=150).mean()
        atr_cache[t] = compute_atr_series_from_ohlc(high, low, close, n=14)
    return close_cache, ma30_cache, ma150_cache, atr_cache


def _load_universe(snapshot_mode: str):
    weekly_df = None
    weekly_snapshots = None
    all_tickers = set()

    if snapshot_mode == "historical":
        weekly_snapshots = load_weekly_snapshots(WEEKLY_SNAPSHOT_DIR)
        if not weekly_snapshots:
            raise SystemExit("snapshot_mode=historical but no snapshots found.")
        for _, df in weekly_snapshots:
            if "ticker" in df.columns:
                all_tickers.update(df["ticker"].astype(str).str.upper())
    elif snapshot_mode == "auto":
        weekly_snapshots = load_weekly_snapshots(WEEKLY_SNAPSHOT_DIR)
        if weekly_snapshots:
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

    all_tickers = {t for t in all_tickers if t and t not in {"NAN", "NONE"}}
    if not all_tickers:
        raise SystemExit("No tickers found in weekly universe.")
    return weekly_df, weekly_snapshots, all_tickers


def run_replay_portfolio(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_yaml_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}
    bt_long_cfg = build_sim_long_config(cfg)
    bt_short_cfg = bt_cfg.get("short", {}) or {}
    market_cfg = bt_cfg.get("market", {}) or {}
    industry_cfg = bt_cfg.get("industry", {}) or {}

    weekly_df, weekly_snapshots, all_tickers = _load_universe(args.snapshot_mode)
    # Regime support tickers
    all_tickers.add("SPY")
    if market_cfg.get("vix_max", None) is not None:
        all_tickers.add("^VIX")

    daily_df = download_daily_bars(sorted(all_tickers), args.start, args.end)

    # IMPORTANT: include raw SELL risk events in the signal stream, but the
    # portfolio wrapper only acts on SELL for tickers it actually holds.
    events = replay_signals(
        daily_df=daily_df,
        start=args.start,
        end=args.end,
        mode=args.mode,
        universe_tickers=sorted(all_tickers),
        weekly_df=weekly_df,
        weekly_snapshots=weekly_snapshots,
        long_logic_cfg=bt_long_cfg,
        short_logic_cfg=bt_short_cfg,
        market_cfg=market_cfg,
        industry_cfg=industry_cfg,
        regime_mode=args.regime_mode,
        neutral_policy=args.neutral_policy,
        exposure_mode=args.exposure_mode,
        bull_long_mult=args.bull_long_mult,
        neutral_long_mult=args.neutral_long_mult,
        bear_short_mult=args.bear_short_mult,
        neutral_short_mult=args.neutral_short_mult,
        signal_quality_mode=args.signal_quality_mode,
        min_long_quality=args.min_long_quality,
        min_short_quality=args.min_short_quality,
        adaptive_reject_below=args.adaptive_reject_below,
        adaptive_floor_mult=args.adaptive_floor_mult,
        adaptive_mid_mult=args.adaptive_mid_mult,
        adaptive_good_mult=args.adaptive_good_mult,
        adaptive_elite_mult=args.adaptive_elite_mult,
        include_near=args.include_near,
        include_raw_sell=True,
        near_zone_pct=args.near_zone_pct,
        sell_crack_pct=args.sell_crack_pct,
    )
    if events.empty:
        return events, pd.DataFrame(), pd.DataFrame()

    events["date"] = pd.to_datetime(events["date"])
    events = events.sort_values(["date", "signal", "ticker"]).reset_index(drop=True)

    tickers = sorted(set(events["ticker"].astype(str).str.upper()))
    _, ma30_cache, ma150_cache, atr_cache = _build_indicator_caches(daily_df, tickers)

    cash = float(args.capital)
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_rows: List[dict] = []
    dates = [d for d in pd.to_datetime(daily_df.index) if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    events_by_date = {d: x for d, x in events.groupby("date")}

    for dt in dates:
        # 1) Core stop exits first, for held names only. This preserves the
        # old risk guard while entry decisions come from replay events.
        for key, pos in list(positions.items()):
            px = _close_at(daily_df, dt, pos.ticker)
            if not np.isfinite(px):
                continue
            ma_series = ma30_cache.get(pos.ticker)
            ma150_series = ma150_cache.get(pos.ticker)
            atr_series = atr_cache.get(pos.ticker)
            ma30 = _safe_float(ma_series.get(dt, np.nan)) if ma_series is not None else np.nan
            ma150 = _safe_float(ma150_series.get(dt, np.nan)) if ma150_series is not None else np.nan
            atr = _safe_float(atr_series.get(dt, np.nan)) if atr_series is not None else np.nan
            if pos.side == "long":
                if np.isfinite(atr) and np.isfinite(ma30):
                    new_stop = long_stop_level(px, atr, ma30)
                    if np.isfinite(new_stop):
                        pos.stop = max(float(pos.stop), float(new_stop))
                if should_exit_long(px, float(pos.stop), ma30):
                    pnl = float(pos.qty) * (px - pos.entry_price)
                    cash += float(pos.qty) * px
                    trades.append(Trade(pos.ticker, pos.side, pos.opened.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d"), pos.entry_price, px, pos.qty, pnl, pnl/(pos.qty*pos.entry_price), pos.entry_reason, "core_stop_or_ma_exit"))
                    del positions[key]
            else:
                ma_exit = ma150 if np.isfinite(ma150) else ma30
                if np.isfinite(atr) and np.isfinite(ma30):
                    new_stop = core_short_stop_level(px, atr, ma30)
                    if np.isfinite(new_stop):
                        pos.stop = min(float(pos.stop), float(new_stop))
                if core_should_exit_short(px, float(pos.stop), ma_exit):
                    pnl = float(pos.qty) * (pos.entry_price - px)
                    cash -= float(pos.qty) * px
                    trades.append(Trade(pos.ticker, pos.side, pos.opened.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d"), pos.entry_price, px, pos.qty, pnl, pnl/(pos.qty*pos.entry_price), pos.entry_reason, "core_short_exit"))
                    del positions[key]

        day_events = events_by_date.get(dt)
        if day_events is not None and not day_events.empty:
            # 2) Replay SELL exits: only if currently held.
            for _, ev in day_events[day_events["signal"].eq("SELL")].iterrows():
                t = str(ev["ticker"]).upper()
                key = f"{t}_long"
                if key not in positions:
                    continue
                pos = positions[key]
                px = _safe_float(ev.get("price"), _close_at(daily_df, dt, t))
                if not np.isfinite(px):
                    continue
                pnl = float(pos.qty) * (px - pos.entry_price)
                cash += float(pos.qty) * px
                trades.append(Trade(t, "long", pos.opened.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d"), pos.entry_price, px, pos.qty, pnl, pnl/(pos.qty*pos.entry_price), pos.entry_reason, "replay_sell_event"))
                del positions[key]

            # 3) Entries from replay BUY/SHORT events only.
            eq_now = _equity(daily_df, dt, cash, positions)
            gross = _gross_exposure(daily_df, dt, positions)
            buying_power = max(0.0, float(args.max_leverage) * eq_now - gross)

            n_long = sum(1 for p in positions.values() if p.side == "long")
            n_short = sum(1 for p in positions.values() if p.side == "short")

            for _, ev in day_events.iterrows():
                sig = str(ev.get("signal", "")).upper()
                if sig not in {"BUY", "SHORT"}:
                    continue
                t = str(ev["ticker"]).upper()
                side = "long" if sig == "BUY" else "short"
                if side == "long" and n_long >= int(args.max_long):
                    continue
                if side == "short" and n_short >= int(args.max_short):
                    continue
                key = f"{t}_{side}"
                if key in positions:
                    continue
                px = _safe_float(ev.get("price"), _close_at(daily_df, dt, t))
                atr = _safe_float(ev.get("atr14"), np.nan)
                ma30 = _safe_float(ev.get("ma30"), np.nan)
                if not np.isfinite(px) or px <= 0:
                    continue
                if not np.isfinite(atr) or not np.isfinite(ma30):
                    continue

                # Risk sizing. Uses replay multipliers from the signal itself.
                if side == "long":
                    stop = long_stop_level(px, atr, ma30)
                    if not np.isfinite(stop) or stop >= px:
                        continue
                    per_share_risk = px - stop
                    size_mult = _safe_float(ev.get("long_size_mult"), 1.0) * _safe_float(ev.get("quality_mult"), 1.0)
                else:
                    stop = core_short_stop_level(px, atr, ma30)
                    if not np.isfinite(stop) or stop <= px:
                        continue
                    per_share_risk = stop - px
                    size_mult = _safe_float(ev.get("short_size_mult"), 1.0) * _safe_float(ev.get("quality_mult"), 1.0)
                if per_share_risk <= 0 or size_mult <= 0:
                    continue
                risk_amt = eq_now * float(args.risk_per_trade) * size_mult
                qty_risk = int(math.floor(risk_amt / per_share_risk))
                qty_cash = int(math.floor(min(cash if side == "long" else buying_power, buying_power) / px))
                qty = max(0, min(qty_risk, qty_cash))
                if qty <= 0:
                    continue
                if side == "long":
                    cost = qty * px
                    if cost > cash:
                        continue
                    cash -= cost
                    n_long += 1
                else:
                    cash += qty * px
                    n_short += 1
                positions[key] = Position(t, side, dt, px, qty, float(stop), atr=float(atr), entry_reason=str(ev.get("reason", "")))
                buying_power = max(0.0, float(args.max_leverage) * _equity(daily_df, dt, cash, positions) - _gross_exposure(daily_df, dt, positions))

        equity_rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "equity": _equity(daily_df, dt, cash, positions),
            "cash": cash,
            "positions": len(positions),
            "long_positions": sum(1 for p in positions.values() if p.side == "long"),
            "short_positions": sum(1 for p in positions.values() if p.side == "short"),
        })

    # Open positions are emitted as trade rows with blank exit_date and MTM pnl.
    last_dt = dates[-1] if dates else pd.Timestamp(args.end)
    for pos in positions.values():
        px = _close_at(daily_df, last_dt, pos.ticker)
        if not np.isfinite(px):
            px = pos.entry_price
        if pos.side == "long":
            pnl = pos.qty * (px - pos.entry_price)
        else:
            pnl = pos.qty * (pos.entry_price - px)
        denom = pos.qty * pos.entry_price
        trades.append(Trade(pos.ticker, pos.side, pos.opened.strftime("%Y-%m-%d"), "", pos.entry_price, px, pos.qty, pnl, pnl/denom if denom else 0.0, pos.entry_reason, "OPEN"))

    return events, pd.DataFrame([asdict(t) for t in trades]), pd.DataFrame(equity_rows)


def main():
    ap = argparse.ArgumentParser(description="Replay-first portfolio research wrapper.")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--mode", choices=["long", "short", "both", "auto"], default="both")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--max-long", type=int, default=10)
    ap.add_argument("--max-short", type=int, default=10)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--snapshot-mode", choices=["static", "historical", "auto"], default="auto")
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--regime-mode", choices=["off", "current", "prod"], default="prod")
    ap.add_argument("--neutral-policy", choices=["long", "none", "both", "current"], default="long")
    ap.add_argument("--exposure-mode", choices=["off", "scaled"], default="scaled")
    ap.add_argument("--bull-long-mult", type=float, default=1.0)
    ap.add_argument("--neutral-long-mult", type=float, default=0.50)
    ap.add_argument("--bear-short-mult", type=float, default=0.60)
    ap.add_argument("--neutral-short-mult", type=float, default=0.0)
    ap.add_argument("--signal-quality-mode", choices=["off", "score", "strict", "adaptive"], default="off")
    ap.add_argument("--min-long-quality", type=float, default=65.0)
    ap.add_argument("--min-short-quality", type=float, default=65.0)
    ap.add_argument("--adaptive-reject-below", type=float, default=60.0)
    ap.add_argument("--adaptive-floor-mult", type=float, default=0.40)
    ap.add_argument("--adaptive-mid-mult", type=float, default=0.65)
    ap.add_argument("--adaptive-good-mult", type=float, default=0.85)
    ap.add_argument("--adaptive-elite-mult", type=float, default=1.00)
    ap.add_argument("--include-near", action="store_true", help="Keep NEAR events in replay CSV, but do not trade them by default.")
    ap.add_argument("--near-zone-pct", type=float, default=0.01)
    ap.add_argument("--sell-crack-pct", type=float, default=0.005)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = _now_tag()
    events, trades, equity = run_replay_portfolio(args)

    events_path = os.path.join(OUTPUT_DIR, f"replay_core_events_{tag}.csv")
    trades_path = os.path.join(OUTPUT_DIR, f"replay_portfolio_trades_{tag}.csv")
    equity_path = os.path.join(OUTPUT_DIR, f"replay_portfolio_equity_{tag}.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"replay_portfolio_summary_{tag}.csv")

    events.to_csv(events_path, index=False)
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)

    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else float(args.capital)
    closed = trades[trades["exit_date"].astype(str).ne("")] if not trades.empty else pd.DataFrame()
    open_rows = trades[trades["exit_date"].astype(str).eq("")] if not trades.empty else pd.DataFrame()
    summary = pd.DataFrame([{
        "start": args.start,
        "end": args.end,
        "mode": args.mode,
        "capital": args.capital,
        "final_equity": final_equity,
        "total_pnl": final_equity - float(args.capital),
        "return_pct": (final_equity / float(args.capital) - 1.0) if args.capital else np.nan,
        "events": len(events),
        "buy_events": int((events["signal"] == "BUY").sum()) if not events.empty else 0,
        "near_events": int((events["signal"] == "NEAR").sum()) if not events.empty else 0,
        "sell_events_raw": int((events["signal"] == "SELL").sum()) if not events.empty else 0,
        "closed_trades": len(closed),
        "open_positions": len(open_rows),
        "realized_pnl": float(closed["pnl"].sum()) if not closed.empty else 0.0,
        "open_mtm_pnl": float(open_rows["pnl"].sum()) if not open_rows.empty else 0.0,
    }])
    summary.to_csv(summary_path, index=False)

    log(f"Replay-first portfolio complete. Final equity=${final_equity:,.2f} ({(final_equity/args.capital-1.0)*100:.2f}%)", level="ok")
    log(f"Wrote replay events → {events_path}", level="ok")
    log(f"Wrote replay portfolio trades → {trades_path}", level="ok")
    log(f"Wrote replay portfolio equity → {equity_path}", level="ok")
    log(f"Wrote replay portfolio summary → {summary_path}", level="ok")


if __name__ == "__main__":
    main()
