#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Intraday Simulator — regime-aware with progress + equity curve + Monte Carlo

Backtests a simplified intraday breakout / breakdown system over one or more
calendar years, using:

- Universe from your latest weekly report (Stage 1/2 + benchmark)
- Chapter 8 style market regime classifier on the benchmark (BULL / NEUTRAL / BEAR)
- Long / short entries gated by regime (or forced long-only / short-only)
- Fixed R-multiple exits (hard stop + 2R take-profit)
- Per-trade and per-regime P&L
- Equity curve PNG
- Optional Monte Carlo resampling of the trade sequence

Usage examples:

  # Regime-driven long/short simulation for 2025:
  python3 weinstein_intraday_sim.py --year 2025 --mode regime

  # Long-only 2025, with 1000 Monte Carlo runs:
  python3 weinstein_intraday_sim.py --year 2025 --mode long_only --mc-runs 1000

  # Multi-year batch, 2022–2025:
  python3 weinstein_intraday_sim.py --start-year 2022 --end-year 2025 --mode regime
"""

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

# Optional plotting (equity curve)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting disabled if matplotlib not available


# ========== Logging helpers ==========

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, level: str = "info") -> None:
    prefix = {
        "info": "•",
        "ok": "✅",
        "step": "▶️",
        "warn": "⚠️",
        "err": "❌",
        "debug": "··",
    }.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)


# ========== Global tunables ==========

BENCHMARK_DEFAULT = "SPY"
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
INTRADAY_INTERVAL = "60m"

# Data window padding around target year
# We use ~6 months of extra history before Jan 1 so SMA150 is "real" early in the year.
LOOKBACK_START_MONTH = 7   # start downloads in July of previous year (was 11 = November)
LOOKAHEAD_END_MONTH = 2    # end downloads in Feb following year

PIVOT_LOOKBACK_WEEKS = 10
SMA_DAYS = 150
HARD_STOP_PCT = 0.08
TP_R_MULT = 2.0  # 2R take-profit


@dataclass
class SimConfig:
    year: int
    benchmark: str
    account_size: float
    risk_per_trade_pct: float
    mode: str
    mc_runs: int = 0


# ========== Load config.yaml ==========

def load_config(path: str) -> Tuple[dict, str, float, float]:
    """
    Reads config.yaml and returns:
      - full cfg dict
      - benchmark symbol
      - account size
      - risk-per-trade fraction
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    app = cfg.get("app", {}) or {}
    ordering = app.get("ordering") or {}

    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    account_size = float(ordering.get("account_size", 5000.0))
    risk_pct = float(ordering.get("risk_per_trade_pct", 0.01))

    return cfg, benchmark, account_size, risk_pct


# ========== Weekly report loader / universe ==========

def newest_weekly_csv() -> str:
    files = [
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found in {WEEKLY_OUTPUT_DIR}. "
            f"Run weinstein_report_weekly.py first."
        )
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report() -> Tuple[pd.DataFrame, str]:
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path


def build_universe(weekly_df: pd.DataFrame, benchmark: str) -> List[str]:
    w = weekly_df.rename(columns=str.lower)
    if "ticker" not in w.columns or "stage" not in w.columns:
        raise ValueError("Weekly CSV missing 'ticker' or 'stage' columns.")

    focus = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])]
    tickers = sorted(set(focus["ticker"].dropna().str.upper().tolist()))

    bmk = benchmark.upper()
    if bmk not in tickers:
        tickers.append(bmk)

    return tickers


# ========== Data download ==========

def download_data(universe: List[str], year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_all = datetime(year - 1, LOOKBACK_START_MONTH, 1)
    end_all = datetime(year + 1, LOOKAHEAD_END_MONTH, 1)

    log(
        f"Downloading daily + intraday for {len(universe)} tickers "
        f"({start_all.date()} → {end_all.date()})...",
        level="step",
    )

    daily = yf.download(
        universe,
        start=start_all.strftime("%Y-%m-%d"),
        end=end_all.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    intraday = yf.download(
        universe,
        start=start_all.strftime("%Y-%m-%d"),
        end=end_all.strftime("%Y-%m-%d"),
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    log("Download complete.", level="ok")
    return daily, intraday


# ========== Series helpers ==========

def _get_close_series(daily: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(daily.columns, pd.MultiIndex):
        try:
            s = daily[("Close", ticker)].dropna()
        except KeyError:
            return pd.Series(dtype=float)
    else:
        s = daily["Close"].dropna()
    return s


def last_weekly_pivot_high(
    daily_df: pd.DataFrame,
    ticker: str,
    weeks: int = PIVOT_LOOKBACK_WEEKS,
    upto_date: Optional[datetime] = None,
) -> float:
    """
    10-week pivot high, restricted to data up to (and including) upto_date.
    """
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"].dropna()

    if upto_date is not None:
        cutoff = pd.Timestamp(upto_date)
        highs = highs.loc[highs.index <= cutoff]

    highs = highs.tail(weeks * 5)  # ~5 trading days per week
    return float(highs.max()) if len(highs) else np.nan


def compute_sma_series(daily_df: pd.DataFrame, ticker: str, window: int) -> pd.Series:
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            s = daily_df[("Close", ticker)].dropna()
        except KeyError:
            return pd.Series(dtype=float)
    else:
        s = daily_df["Close"].dropna()
    return s.rolling(window).mean()


# ========== Regime classifier (Chapter 8 flavored) ==========

def classify_regime_for_benchmark(daily: pd.DataFrame, benchmark: str) -> pd.Series:
    """
    Rough Weinstein-style regime classification on the benchmark:

      - BULL:    Price > SMA150 and SMA150 slope (30d diff) > 0
      - BEAR:    Price < SMA150 and SMA150 slope (30d diff) < 0
      - NEUTRAL: Everything else
    """
    close = _get_close_series(daily, benchmark)
    if close.empty:
        raise ValueError(f"No daily close series for benchmark {benchmark}.")

    sma = close.rolling(SMA_DAYS).mean()
    slope = sma.diff(30)

    labels = []
    for dt, px in close.items():
        ma = sma.loc[dt]
        sl = slope.loc[dt]
        if pd.isna(ma) or pd.isna(sl):
            labels.append("NEUTRAL")
        elif px > ma and sl > 0:
            labels.append("BULL")
        elif px < ma and sl < 0:
            labels.append("BEAR")
        else:
            labels.append("NEUTRAL")

    return pd.Series(labels, index=close.index, name="regime")


def regime_flags_for_date(
    regime_series: pd.Series,
    d: date,
    mode: str,
) -> Tuple[bool, bool, str]:
    """
    For a given calendar date and mode, returns:
      - long_ok (bool)
      - short_ok (bool)
      - label ("BULL"/"NEUTRAL"/"BEAR")
    """
    ts = pd.Timestamp(d)
    subset = regime_series.loc[regime_series.index <= ts]
    label = subset.iloc[-1] if not subset.empty else "NEUTRAL"
    label = str(label).upper()

    if mode == "long_only":
        return True, False, label
    if mode == "short_only":
        return False, True, label

    # mode="regime": gate by regime
    if label == "BULL":
        return True, False, label
    if label == "BEAR":
        return False, True, label
    # NEUTRAL: allow both but you can later decide to tweak
    return True, True, label


# ========== Position & Trade models ==========

@dataclass
class Position:
    ticker: str
    direction: str   # "long" or "short"
    entry_ts: pd.Timestamp
    entry_price: float
    qty: float
    stop_price: float
    tp_price: float


@dataclass
class Trade:
    ticker: str
    direction: str   # "long" or "short"
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl_dollar: float
    pnl_pct: float
    regime_at_entry: str
    regime_at_exit: str


# ========== Core simulation per-year ==========

def simulate_year(sim_cfg: SimConfig, config_path: str) -> Dict[str, float]:
    """
    Runs the simulation for a single calendar year; returns a summary dict.
    """
    cfg, config_bench, config_acc, config_risk = load_config(config_path)

    benchmark = sim_cfg.benchmark or config_bench
    start_equity = float(sim_cfg.account_size or config_acc)
    risk_pct = float(sim_cfg.risk_per_trade_pct or config_risk)

    weekly_df, weekly_path = load_weekly_report()
    log(f"Using weekly CSV: {weekly_path}")
    universe = build_universe(weekly_df, benchmark)
    log(f"Focus universe: {len(universe)-1} Stage 1/2 + benchmark {benchmark}")

    # Download market data
    daily, intraday = download_data(universe, sim_cfg.year)

    # Regime series
    regime_series = classify_regime_for_benchmark(daily, benchmark)
    log("Computed Chapter 8 regime series (BULL/NEUTRAL/BEAR).", level="ok")

    # Restrict intraday index to target year
    idx = intraday.index
    start = datetime(sim_cfg.year, 1, 1)
    end = datetime(sim_cfg.year, 12, 31, 23, 59)
    idx = idx[(idx >= start) & (idx <= end)]
    if len(idx) == 0:
        raise ValueError(f"No intraday bars for year {sim_cfg.year}.")

    log(f"Intraday bars in {sim_cfg.year}: {len(idx)}", level="info")

    equity = start_equity
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []

    log(
        f"Initial account: ${equity:,.2f}, risk per trade: {risk_pct*100:.2f}% "
        f"(${equity*risk_pct:,.2f})",
        level="info",
    )

    # Precompute daily SMA150
    sma_cache: Dict[str, pd.Series] = {
        t: compute_sma_series(daily, t, SMA_DAYS) for t in universe
    }

    n_bars = len(idx)
    # progress milestones at ~10% increments
    milestones = {max(1, int(n_bars * f / 10)) for f in range(1, 10)}

    # ----- Main intraday loop -----
    for i, ts_bar in enumerate(idx, start=1):
        bar_date = ts_bar.date()
        row = intraday.loc[ts_bar]

        long_ok, short_ok, regime_label = regime_flags_for_date(
            regime_series, bar_date, sim_cfg.mode
        )

        # === 1) Exit logic: evaluate stops / TPs for all open positions ===
        to_close = []
        for key, pos in positions.items():
            t = pos.ticker

            if ("Close", t) not in row:
                continue
            px = float(row[("Close", t)])
            if math.isnan(px) or px <= 0:
                continue

            # direction-aware hit
            hit_stop = (px <= pos.stop_price) if pos.direction == "long" else (px >= pos.stop_price)
            hit_tp = (px >= pos.tp_price) if pos.direction == "long" else (px <= pos.tp_price)

            if hit_stop or hit_tp:
                pnl = (
                    (px - pos.entry_price) * pos.qty
                    if pos.direction == "long"
                    else (pos.entry_price - px) * pos.qty
                )
                pnl_pct = (
                    pnl / (pos.entry_price * pos.qty) * 100.0
                    if pos.entry_price * pos.qty != 0
                    else 0.0
                )
                equity += pnl

                # regime at entry/exit for segmentation
                _, _, reg_entry = regime_flags_for_date(
                    regime_series, pos.entry_ts.date(), sim_cfg.mode
                )
                _, _, reg_exit = regime_flags_for_date(
                    regime_series, bar_date, sim_cfg.mode
                )

                trades.append(
                    Trade(
                        ticker=t,
                        direction=pos.direction,
                        entry_ts=pos.entry_ts,
                        exit_ts=ts_bar,
                        entry_price=pos.entry_price,
                        exit_price=px,
                        qty=pos.qty,
                        pnl_dollar=pnl,
                        pnl_pct=pnl_pct,
                        regime_at_entry=reg_entry,
                        regime_at_exit=reg_exit,
                    )
                )
                to_close.append(key)

        # close any positions flagged for exit
        for k in to_close:
            del positions[k]

        # === 2) Entry logic: attempt new long/short positions ===
        risk_dollar = equity * risk_pct

        for t in universe:
            if t == benchmark:
                continue

            key_long = f"{t}_long"
            key_short = f"{t}_short"
            if key_long in positions or key_short in positions:
                continue  # only one position per ticker/direction

            if ("Close", t) not in row:
                continue
            px = float(row[("Close", t)])
            if math.isnan(px) or px <= 0:
                continue

            # daily subset up to current date
            if isinstance(daily.columns, pd.MultiIndex):
                try:
                    ds = daily.xs(t, axis=1, level=1)
                except KeyError:
                    continue
            else:
                ds = daily.copy()
            ds = ds.loc[ds.index <= pd.Timestamp(bar_date)]
            if ds.empty:
                continue

            pivot = last_weekly_pivot_high(daily, t, upto_date=bar_date)
            sma_t = sma_cache[t]
            sma_t = sma_t.loc[sma_t.index <= pd.Timestamp(bar_date)]
            if sma_t.empty:
                continue
            ma150 = float(sma_t.iloc[-1])

            # ---- Long entry rule: simple breakout above pivot & MA150 ----
            if long_ok and not math.isnan(pivot) and px >= pivot and px >= ma150:
                stop = px * (1.0 - HARD_STOP_PCT)
                r = px - stop
                if r > 0:
                    tp = px + TP_R_MULT * r
                    qty = max(0, int(risk_dollar / r))
                    if qty > 0:
                        positions[key_long] = Position(
                            ticker=t,
                            direction="long",
                            entry_ts=ts_bar,
                            entry_price=px,
                            qty=qty,
                            stop_price=stop,
                            tp_price=tp,
                        )

            # ---- Short entry rule: breakdown under MA150 ----
            if short_ok and px <= ma150 * 0.99:  # 1% under MA150
                stop = px * (1.0 + HARD_STOP_PCT)
                r = stop - px
                if r > 0:
                    tp = px - TP_R_MULT * r
                    qty = max(0, int(risk_dollar / r))
                    if qty > 0:
                        positions[key_short] = Position(
                            ticker=t,
                            direction="short",
                            entry_ts=ts_bar,
                            entry_price=px,
                            qty=qty,
                            stop_price=stop,
                            tp_price=tp,
                        )

        # === 3) Progress logging ===
        if i in milestones or i == n_bars:
            pct = i / n_bars * 100.0
            log(
                f"Simulation progress {sim_cfg.year}: {i}/{n_bars} bars "
                f"({pct:5.1f}%) — equity ${equity:,.2f}, "
                f"open positions {len(positions)}, trades {len(trades)}"
            )

    # ===== Simulation complete, compute summaries =====
    total_pnl = equity - start_equity
    total_ret_pct = (total_pnl / start_equity * 100.0) if start_equity else 0.0

    wins = len([t for t in trades if t.pnl_dollar > 0])
    losses = len([t for t in trades if t.pnl_dollar < 0])
    n_trades = len(trades)
    winrate = wins / n_trades * 100.0 if n_trades else 0.0

    log(f"Simulation {sim_cfg.year} complete.", level="ok")
    log(
        f"Final equity: ${equity:,.2f} "
        f"(P/L ${total_pnl:,.2f}, {total_ret_pct:.2f}%) — "
        f"Trades={n_trades}, Wins={wins}, Losses={losses}, Win-rate={winrate:.1f}%",
        level="info",
    )

    # --- Regime-segmented P&L (by regime at ENTRY & EXIT) ---
    from collections import defaultdict

    pnl_by_reg_entry = defaultdict(float)
    count_by_reg_entry = defaultdict(int)

    pnl_by_reg_exit = defaultdict(float)
    count_by_reg_exit = defaultdict(int)

    for tr in trades:
        re = tr.regime_at_entry
        rx = tr.regime_at_exit
        pnl_by_reg_entry[re] += tr.pnl_dollar
        count_by_reg_entry[re] += 1
        pnl_by_reg_exit[rx] += tr.pnl_dollar
        count_by_reg_exit[rx] += 1

    if trades:
        log("P/L by regime at ENTRY:", level="info")
        for regime_label in sorted(pnl_by_reg_entry.keys()):
            pnl = pnl_by_reg_entry[regime_label]
            cnt = count_by_reg_entry[regime_label]
            log(
                f"  - {regime_label}: P/L ${pnl:,.2f} "
                f"({cnt} trades, avg ${pnl/cnt:,.2f} each)",
                level="info",
            )

        log("P/L by regime at EXIT:", level="info")
        for regime_label in sorted(pnl_by_reg_exit.keys()):
            pnl = pnl_by_reg_exit[regime_label]
            cnt = count_by_reg_exit[regime_label]
            log(
                f"  - {regime_label}: P/L ${pnl:,.2f} "
                f"({cnt} trades, avg ${pnl/cnt:,.2f} each)",
                level="info",
            )
    else:
        log("No trades executed; regime P/L breakdown is empty.", level="warn")

    # --- Save trade log CSV (always with header) ---
    os.makedirs("./output", exist_ok=True)
    trades_cols = [
        "ticker",
        "direction",
        "entry_ts",
        "exit_ts",
        "entry_price",
        "exit_price",
        "qty",
        "pnl_dollar",
        "pnl_pct",
        "regime_at_entry",
        "regime_at_exit",
    ]
    trades_df = pd.DataFrame([t.__dict__ for t in trades], columns=trades_cols)
    out_path = f"./output/intraday_sim_{sim_cfg.year}_{sim_cfg.mode}.csv"
    trades_df.to_csv(out_path, index=False)
    log(f"Wrote trade log → {out_path}", level="ok")

    # --- Equity curve & per-trade account returns ---
    equity_curve_dates: List[pd.Timestamp] = []
    equity_curve_values: List[float] = []
    trade_returns: List[float] = []  # per-trade account-level returns

    if trades:
        eq = start_equity
        for tr in sorted(trades, key=lambda x: x.exit_ts):
            before = eq
            after = before + tr.pnl_dollar
            if before > 0:
                trade_returns.append(after / before - 1.0)
            eq = after
            equity_curve_dates.append(tr.exit_ts)
            equity_curve_values.append(eq)

        # Plot equity curve if matplotlib is available
        if plt is not None and equity_curve_dates:
            try:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(equity_curve_dates, equity_curve_values)
                ax.set_title(f"Equity Curve {sim_cfg.year} ({sim_cfg.mode})")
                ax.set_ylabel("Equity ($)")
                ax.grid(alpha=0.3)
                fig.autofmt_xdate()
                eq_path = f"./output/intraday_sim_{sim_cfg.year}_{sim_cfg.mode}_equity.png"
                fig.tight_layout()
                fig.savefig(eq_path, dpi=120)
                plt.close(fig)
                log(f"Wrote equity curve PNG → {eq_path}", level="ok")
            except Exception as e:
                log(f"Failed to plot equity curve: {e}", level="warn")
        else:
            if plt is None:
                log("matplotlib not available; skipping equity-curve plot.", level="warn")
    else:
        log("No trades; skipping equity curve / trade-return computation.", level="warn")

    # --- Monte Carlo over trade sequence (optional) ---
    mc_summary = {}
    if sim_cfg.mc_runs and trades and trade_returns:
        runs = sim_cfg.mc_runs
        final_eqs = []
        rng = np.random.default_rng()

        for _ in range(runs):
            eq = start_equity
            # resample trade returns with replacement
            sampled = rng.choice(trade_returns, size=len(trade_returns), replace=True)
            for r in sampled:
                eq *= (1.0 + r)
            final_eqs.append(eq)

        final_eqs = np.array(final_eqs, dtype=float)
        mc_median = float(np.median(final_eqs))
        mc_p5 = float(np.percentile(final_eqs, 5))
        mc_p95 = float(np.percentile(final_eqs, 95))

        mc_summary = {
            "mc_runs": runs,
            "mc_final_median": mc_median,
            "mc_final_p5": mc_p5,
            "mc_final_p95": mc_p95,
        }

        log(
            f"Monte Carlo ({runs} runs) on trade sequence — "
            f"median ${mc_median:,.2f}, 5th% ${mc_p5:,.2f}, 95th% ${mc_p95:,.2f}",
            level="info",
        )
    elif sim_cfg.mc_runs:
        log(
            f"Monte Carlo requested (--mc-runs {sim_cfg.mc_runs}) but "
            f"no trades or trade-returns available; skipping.",
            level="warn",
        )

    summary = {
        "year": sim_cfg.year,
        "mode": sim_cfg.mode,
        "start_equity": start_equity,
        "final_equity": equity,
        "total_pnl": total_pnl,
        "total_ret_pct": total_ret_pct,
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "winrate_pct": winrate,
    }
    # merge MC fields if present
    summary.update(mc_summary)
    return summary


# ========== CLI driver (single year or multi-year) ==========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="Single calendar year to simulate")
    ap.add_argument("--start-year", type=int, help="Start year (for multi-year batch)")
    ap.add_argument("--end-year", type=int, help="End year (for multi-year batch)")
    ap.add_argument(
        "--mode",
        type=str,
        default="regime",
        choices=["regime", "long_only", "short_only"],
        help="How to gate long/short trades",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config.yaml",
    )
    ap.add_argument(
        "--mc-runs",
        type=int,
        default=0,
        help="Monte Carlo runs over trade sequence (0 = disabled)",
    )
    args = ap.parse_args()

    # Determine years to simulate
    years: List[int] = []
    if args.start_year and args.end_year:
        if args.end_year < args.start_year:
            raise ValueError("end-year must be >= start-year.")
        years = list(range(args.start_year, args.end_year + 1))
    elif args.year:
        years = [args.year]
    else:
        raise SystemExit("You must specify either --year or --start-year/--end-year.")

    # Load base config once (for benchmark & sizing)
    _, bench, acc, risk = load_config(args.config)

    all_summaries = []

    for y in years:
        cfg = SimConfig(
            year=y,
            benchmark=bench,
            account_size=acc,
            risk_per_trade_pct=risk,
            mode=args.mode,
            mc_runs=args.mc_runs,
        )
        log(
            f"Starting simulation for {cfg.year} (mode={cfg.mode}, mc_runs={cfg.mc_runs}) "
            f"using {args.config}",
            level="step",
        )
        summary = simulate_year(cfg, args.config)
        all_summaries.append(summary)

    # If multi-year, write a combined CSV of yearly summaries
    if len(all_summaries) > 1:
        os.makedirs("./output", exist_ok=True)
        df = pd.DataFrame(all_summaries)
        out_path = f"./output/intraday_sim_summary_{years[0]}_{years[-1]}_{args.mode}.csv"
        df.to_csv(out_path, index=False)
        log(f"Wrote multi-year summary CSV → {out_path}", level="ok")


if __name__ == "__main__":
    main()
