#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_live_logic_backtest.py

Backtest the *live intraday trigger logic* (long + short) over historical
60-minute bars, using the same tunables as weinstein_intraday_watcher.py.

- Universe: Stage 1/2 from latest weekly CSV (+ benchmark)
- Signals: same state machine (NEAR → ARMED → TRIGGERED + cooldown)
- Entries:
    * LONG:  when buy_state == TRIGGERED and long_ok == True
    * SHORT: when sell_state == TRIGGERED and short_ok == True
- Exits:
    * LONG:  when sell_state == TRIGGERED
    * SHORT: when buy_state == TRIGGERED

This is intentionally simplified on volume pacing (no intrabar/pace), but
reuses the same price/pivot/MA/near/confirm logic and the same trigger states.
"""

import os
import math
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# --- Reuse config, weekly loader, regime + tunables from the intraday watcher ---
from weinstein_intraday_watcher import (
    load_config,
    load_weekly_report,
    inspect_market_regime,
    compute_atr,
    last_weekly_pivot_high,
    _update_hits,
    _price_below_ma,
    _near_sell_zone,
    MIN_BREAKOUT_PCT,
    BUY_DIST_ABOVE_MA_MIN,
    CONFIRM_BARS,
    NEAR_BELOW_PIVOT_PCT,
    SELL_BREAK_PCT,
    NEAR_HITS_WINDOW,
    NEAR_HITS_MIN,
    COOLDOWN_SCANS,
    SELL_NEAR_HITS_WINDOW,
    SELL_NEAR_HITS_MIN,
    SELL_COOLDOWN_SCANS,
    BENCHMARK_DEFAULT,
    log,
)

INTRADAY_INTERVAL = "60m"

# --- Backtest-specific tunables (you can tweak) ---
INITIAL_EQUITY_DEFAULT = 5000.0
MAX_POSITION_FRACTION = 0.10   # max 10% of equity per trade
RISK_FRACTION = 0.01           # risk budget per trade (~1% of equity)


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _stage_ok(stage: str) -> bool:
    s = str(stage or "")
    return s.startswith("Stage 1") or s.startswith("Stage 2")


def _get_universe(config_path: str):
    """
    Load latest weekly CSV and build Stage 1/2 focus universe + benchmark.
    """
    cfg, benchmark, _, _ = load_config(config_path)
    weekly_df, weekly_path = load_weekly_report()
    log(f"Using weekly CSV: {weekly_path}", level="info")

    w = weekly_df.rename(columns=str.lower)
    for col in ["ticker", "stage", "ma30", "rs_above_ma"]:
        if col not in w.columns:
            w[col] = np.nan

    focus = w[w["stage"].apply(_stage_ok)].copy()
    if "rank" in w.columns:
        focus["weekly_rank"] = w["rank"]
    else:
        focus["weekly_rank"] = 999999

    tickers = sorted(set(focus["ticker"].tolist()))
    bench = benchmark or BENCHMARK_DEFAULT

    if bench not in tickers:
        tickers.append(bench)

    log(
        f"Focus universe: {len(focus)} Stage 1/2 + benchmark {bench}",
        level="info",
    )
    return focus, bench, weekly_df


def _download_data(tickers, year: int):
    """
    Download daily + intraday 60m bars for a window around the given year.
    We pad a bit before/after to have enough context for pivots/ATR.
    """
    start = datetime(year - 1, 11, 1)
    end = datetime(year + 1, 2, 1)

    log(
        f"Downloading daily + intraday for {len(tickers)} tickers "
        f"({start.date()} → {end.date()})...",
        level="step",
    )

    intraday = yf.download(
        tickers,
        start=start,
        end=end,
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )
    daily = yf.download(
        tickers,
        start=start.date() - timedelta(days=200),
        end=end.date(),
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    log("Download complete.", level="ok")
    return intraday, daily


def _slice_year_intraday(intraday: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Restrict intraday to bars within the requested year.
    """
    if intraday.empty:
        return intraday
    # yfinance intraday index is a DatetimeIndex
    return intraday.loc[str(year)]


def _get_close_series(intraday_year: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Returns the close series for a ticker within intraday_year.
    """
    if intraday_year.empty:
        return pd.Series(dtype=float)

    if isinstance(intraday_year.columns, pd.MultiIndex):
        try:
            s = intraday_year[("Close", ticker)]
        except KeyError:
            return pd.Series(dtype=float)
    else:
        # single ticker case
        s = intraday_year["Close"]
    return s.dropna()


def _get_close_on_bar(series: pd.Series, ts) -> float:
    """
    Get the close at or before ts from a per-ticker close series.
    """
    if series.empty:
        return math.nan
    # Up to this timestamp
    s = series.loc[:ts]
    if s.empty:
        return math.nan
    return float(s.iloc[-1])


def _daily_slice_to(ts, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    return daily.loc[:ts.date()]


def _entry_size(equity: float, price: float) -> float:
    """
    Very simple sizing: min(max_pos_fraction * equity, 5 * risk_budget).
    """
    if price <= 0 or equity <= 0:
        return 0.0
    max_pos_dollars = MAX_POSITION_FRACTION * equity
    risk_budget = RISK_FRACTION * equity
    dol = min(max_pos_dollars, 5.0 * risk_budget)
    return dol / price


def backtest_year(
    year: int,
    config_path: str,
    side: str = "both",
    initial_equity: float = INITIAL_EQUITY_DEFAULT,
    out_csv: str | None = None,
    no_regime_filter: bool = False,
):
    # --- Universe + weekly data ---
    focus, benchmark, weekly_df = _get_universe(config_path)
    cfg, _, _, _ = load_config(config_path)

    tickers = sorted(set(focus["ticker"].tolist() + [benchmark]))

    # --- Market regime (Chapter 8) ---
    label, long_ok, short_ok = inspect_market_regime()
    if no_regime_filter:
        long_ok = True
        short_ok = True
        regime_label = f"{label} (overridden: no regime filter)"
    else:
        regime_label = label

    log(
        f"Market regime (Ch8): {regime_label} | long_ok={long_ok} short_ok={short_ok}",
        level="info",
    )

    # --- Data ---
    intraday, daily = _download_data(tickers, year)
    intraday_year = _slice_year_intraday(intraday, year)
    if intraday_year.empty:
        log(f"No intraday bars found for {year}.", level="err")
        return

    bar_times = intraday_year.index.unique()
    total_bars = len(bar_times)
    log(f"Intraday bars in {year}: {total_bars}", level="info")

    # --- Prepare per-ticker close series for quick lookup ---
    close_series = {t: _get_close_series(intraday_year, t) for t in tickers}

    # --- State ---
    trigger_state = {
        t: {
            "state": "IDLE",
            "near_hits": [],
            "cooldown": 0,
            "sell_state": "IDLE",
            "sell_hits": [],
            "sell_cooldown": 0,
        }
        for t in focus["ticker"].tolist()
    }

    positions = {}  # ticker -> dict(side, entry, shares)
    cash = float(initial_equity)
    equity = float(initial_equity)
    trades = []

    # Map weekly info for quick lookup
    w = weekly_df.rename(columns=str.lower)
    for col in ["ticker", "stage", "ma30", "rs_above_ma", "rank"]:
        if col not in w.columns:
            w[col] = np.nan

    weekly_by_ticker = {
        str(r["ticker"]): {
            "stage": r["stage"],
            "ma30": r.get("ma30", np.nan),
            "rs_above_ma": bool(r.get("rs_above_ma", False)),
            "weekly_rank": r.get("rank", np.nan),
        }
        for _, r in w.iterrows()
    }

    def mark_to_market(ts):
        nonlocal equity
        eq = cash
        for t, pos in positions.items():
            s = close_series.get(t)
            if s is None or s.empty:
                continue
            px = _get_close_on_bar(s, ts)
            if math.isnan(px):
                continue
            if pos["side"] == "long":
                eq += px * pos["shares"]
            else:
                # Short P/L = (entry - current) * shares
                eq += (pos["entry"] - px) * pos["shares"]
        equity = eq

    # --- Main bar loop ---
    last_progress_pct = -1

    for idx, ts in enumerate(bar_times, start=1):
        # Progress every ~10%
        pct = int(idx * 100.0 / total_bars)
        if pct // 10 != last_progress_pct // 10:
            last_progress_pct = pct
            mark_to_market(ts)
            log(
                f"Progress {year}: {idx}/{total_bars} bars ({pct:4.1f}%) — "
                f"equity ${equity:,.2f}, open positions {len(positions)}, trades {len(trades)}",
                level="info",
            )

        # Per-ticker logic
        for t in focus["ticker"].tolist():
            s_close = close_series.get(t)
            if s_close is None or s_close.empty:
                continue

            px = _get_close_on_bar(s_close, ts)
            if math.isnan(px):
                continue

            wi = weekly_by_ticker.get(t, {})
            stage = wi.get("stage", "")
            ma30 = _safe_float(wi.get("ma30", np.nan))
            rs_ok = bool(wi.get("rs_above_ma", False))
            weekly_rank = wi.get("weekly_rank", np.nan)

            if not _stage_ok(stage):
                continue

            ma_ok = not math.isnan(ma30)
            if not ma_ok or not rs_ok:
                # In live logic we also require RS + MA
                continue

            # Daily slice up to this bar's date
            daily_to = _daily_slice_to(ts, daily)
            if daily_to.empty:
                continue

            pivot = last_weekly_pivot_high(t, daily_to)
            atr = compute_atr(daily_to, t, n=14)

            pivot_ok = not math.isnan(pivot)

            st = trigger_state.get(t)
            if st is None:
                st = {
                    "state": "IDLE",
                    "near_hits": [],
                    "cooldown": 0,
                    "sell_state": "IDLE",
                    "sell_hits": [],
                    "sell_cooldown": 0,
                }
                trigger_state[t] = st

            # If market not favorable for longs, reset buy state (like live script)
            if not long_ok:
                st["state"] = "IDLE"
                st["near_hits"] = []
                st["cooldown"] = 0

            closes_up_to_ts = s_close.loc[:ts]
            # Need at least CONFIRM_BARS for proper confirm
            if len(closes_up_to_ts) < max(CONFIRM_BARS, 2):
                closes_tail = closes_up_to_ts.tail(len(closes_up_to_ts))
            else:
                closes_tail = closes_up_to_ts.tail(CONFIRM_BARS)

            # --- BUY confirmation ---
            buy_price_ok = False
            buy_confirm = False

            if long_ok and pivot_ok and ma_ok and len(closes_tail) >= 1:
                def _price_ok(c):
                    return (
                        c >= pivot * (1.0 + MIN_BREAKOUT_PCT)
                        and c >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
                    )

                if len(closes_tail) >= CONFIRM_BARS:
                    buy_price_ok = all(_price_ok(c) for c in closes_tail)
                else:
                    buy_price_ok = _price_ok(closes_tail.iloc[-1])

                buy_confirm = buy_price_ok

            # --- BUY "near" logic (similar to live script, using current px only) ---
            near_now = False
            if long_ok and pivot_ok and ma_ok:
                above_ma = px >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
                if above_ma:
                    if (px >= pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)) and (
                        px < pivot * (1.0 + MIN_BREAKOUT_PCT)
                    ):
                        near_now = True
                    elif (px >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not buy_confirm:
                        near_now = True

            # --- SELL "near" + confirm (mirror of live) ---
            sell_near_now = False
            sell_price_ok = False
            sell_confirm = False

            if ma_ok:
                sell_near_now = _near_sell_zone(px, ma30)

                closes_tail2 = closes_up_to_ts.tail(max(CONFIRM_BARS, 2))
                if len(closes_tail2) >= 1:
                    sell_price_ok = all(
                        (c <= ma30 * (1.0 - SELL_BREAK_PCT))
                        for c in closes_tail2.tail(CONFIRM_BARS)
                    )
                    sell_confirm = sell_price_ok

            # --- Promote state: BUY side ---
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
            elif state_now == "ARMED" and buy_confirm:
                state_now = "TRIGGERED"
                st["cooldown"] = COOLDOWN_SCANS
            elif st["cooldown"] > 0 and not near_now:
                state_now = "COOLDOWN"
            elif st["cooldown"] == 0 and not near_now and not buy_confirm:
                state_now = "IDLE"
            st["state"] = state_now

            # --- Promote state: SELL side ---
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
            elif sell_state == "ARMED" and sell_confirm:
                sell_state = "TRIGGERED"
                st["sell_cooldown"] = SELL_COOLDOWN_SCANS
            elif st["sell_cooldown"] > 0 and not sell_near_now:
                sell_state = "COOLDOWN"
            elif st["sell_cooldown"] == 0 and not sell_near_now and not sell_confirm:
                sell_state = "IDLE"
            st["sell_state"] = sell_state

            trigger_state[t] = st

            # --- Trading logic (long / short) ---
            pos = positions.get(t)
            current_side = pos["side"] if pos else None

            # Exits first, then entries (avoid flip-flop on same bar)
            # LONG exit: SELL trigger
            if pos and current_side == "long" and sell_state == "TRIGGERED":
                # close long
                cash += px * pos["shares"]
                trades.append(
                    {
                        "timestamp": ts,
                        "ticker": t,
                        "side": "LONG",
                        "action": "CLOSE",
                        "price": px,
                        "shares": pos["shares"],
                    }
                )
                del positions[t]

            # SHORT exit: BUY trigger
            elif pos and current_side == "short" and state_now == "TRIGGERED":
                # buy to cover
                cash -= px * pos["shares"]
                trades.append(
                    {
                        "timestamp": ts,
                        "ticker": t,
                        "side": "SHORT",
                        "action": "CLOSE",
                        "price": px,
                        "shares": pos["shares"],
                    }
                )
                del positions[t]

            # Re-evaluate after exits
            pos = positions.get(t)
            current_side = pos["side"] if pos else None

            # LONG entry
            if (
                state_now == "TRIGGERED"
                and long_ok
                and (side in ("long", "both"))
                and current_side is None
            ):
                # basic sizing
                mark_to_market(ts)
                shares = _entry_size(equity, px)
                if shares > 0 and px * shares <= cash:
                    positions[t] = {
                        "side": "long",
                        "entry": px,
                        "shares": shares,
                    }
                    cash -= px * shares
                    trades.append(
                        {
                            "timestamp": ts,
                            "ticker": t,
                            "side": "LONG",
                            "action": "OPEN",
                            "price": px,
                            "shares": shares,
                        }
                    )
                    # Cooldown after triggering
                    st["state"] = "COOLDOWN"
                    trigger_state[t] = st

            # SHORT entry
            elif (
                sell_state == "TRIGGERED"
                and short_ok
                and (side in ("short", "both"))
                and current_side is None
            ):
                # short sale: we receive proceeds
                mark_to_market(ts)
                shares = _entry_size(equity, px)
                if shares > 0:
                    positions[t] = {
                        "side": "short",
                        "entry": px,
                        "shares": shares,
                    }
                    cash += px * shares
                    trades.append(
                        {
                            "timestamp": ts,
                            "ticker": t,
                            "side": "SHORT",
                            "action": "OPEN",
                            "price": px,
                            "shares": shares,
                        }
                    )
                    st["sell_state"] = "COOLDOWN"
                    trigger_state[t] = st

        # End per-ticker loop
        mark_to_market(ts)

    # --- Done: final stats + CSV ---
    mark_to_market(bar_times[-1])

    pl = equity - initial_equity
    pl_pct = (pl / initial_equity * 100.0) if initial_equity else 0.0

    log(
        f"Backtest complete for {year}.",
        level="ok",
    )
    log(
        f"Final equity: ${equity:,.2f} (P/L ${pl:,.2f}, {pl_pct:.2f}%) — "
        f"Trades={len(trades)}",
        level="info",
    )

    if not out_csv:
        out_csv = os.path.join("./output", f"live_logic_bt_{year}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades.sort_values("timestamp", inplace=True)
    df_trades.to_csv(out_csv, index=False)
    log(f"Wrote trade log → {out_csv}", level="ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to backtest (e.g. 2025)",
    )
    ap.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to config.yaml (same as intraday watcher)",
    )
    ap.add_argument(
        "--side",
        choices=["long", "short", "both"],
        default="both",
        help="Which side(s) to trade: long, short, or both (default).",
    )
    ap.add_argument(
        "--initial-equity",
        type=float,
        default=INITIAL_EQUITY_DEFAULT,
        help=f"Starting equity for backtest (default {INITIAL_EQUITY_DEFAULT}).",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path for trades (default ./output/live_logic_bt_<year>.csv).",
    )
    ap.add_argument(
        "--no-regime-filter",
        action="store_true",
        help="Ignore market regime filter (force long_ok=True, short_ok=True).",
    )
    args = ap.parse_args()

    out_csv = args.out_csv or None

    backtest_year(
        year=args.year,
        config_path=args.config,
        side=args.side,
        initial_equity=args.initial_equity,
        out_csv=out_csv,
        no_regime_filter=args.no_regime_filter,
    )


if __name__ == "__main__":
    main()
