#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
    "BCH-USD", "DOGE-USD", "LINK-USD", "LTC-USD", "NEAR-USD",
    "TRX-USD", "XRP-USD"
]

SMA_DAYS = 150
PIVOT_LOOKBACK_DAYS = 90
VOL_LOOKBACK_DAYS = 50


@dataclass
class ProfileParams:
    min_break_pct: float
    dist_above_ma_min: float
    vol_min: float


def profile_params(profile: str) -> ProfileParams:
    p = profile.upper()
    base = ProfileParams(min_break_pct=0.004, dist_above_ma_min=0.0, vol_min=1.20)
    if p in {"A", "B"}:
        return base
    if p == "C":
        return ProfileParams(0.006, 0.0, 1.20)
    if p == "D":
        return ProfileParams(0.008, 0.015, 1.30)
    if p == "E":
        return ProfileParams(0.012, 0.025, 1.50)
    if p == "F":
        return ProfileParams(0.008, 0.015, 1.30)
    raise ValueError(f"Unsupported profile: {profile}")


def max_drawdown(equity):
    arr = np.asarray(equity, dtype=float)
    if len(arr) == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    dd = (arr / peaks) - 1.0
    return float(dd.min())


def cagr(start_value, end_value, start_date, end_date):
    years = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25, 1e-9)
    if start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


def profit_factor(trades):
    if not trades:
        return np.nan
    gains = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def download_data(universe, start, end):
    print(f"Downloading daily crypto data: {start} -> {end}")
    data = yf.download(
        universe,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return data


def get_ticker_df(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        df = data[ticker].copy()
    else:
        df = data.copy()
    df = df.rename(columns=str.title)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df.dropna(subset=["Close"])
    return df


def prepare_features(df):
    df = df.copy()
    df["SMA150"] = df["Close"].rolling(SMA_DAYS).mean()
    df["VolAvg50"] = df["Volume"].rolling(VOL_LOOKBACK_DAYS).mean()
    df["VolPace"] = df["Volume"] / df["VolAvg50"]
    df["Pivot"] = df["High"].rolling(PIVOT_LOOKBACK_DAYS).max().shift(1)
    return df


def run_profile(data_by_ticker, profile, start, end, capital, max_pos_frac, max_exposure_frac):
    params = profile_params(profile)

    cash = float(capital)
    positions = {}
    equity_rows = []
    trades = []

    all_dates = sorted(set().union(*[set(df.index) for df in data_by_ticker.values() if not df.empty]))

    for dt in all_dates:
        if dt < pd.Timestamp(start) or dt > pd.Timestamp(end):
            continue

        # mark-to-market
        equity = cash
        for t, pos in list(positions.items()):
            df = data_by_ticker[t]
            if dt in df.index:
                px = float(df.loc[dt, "Close"])
                equity += pos["qty"] * px

        # exits first: sell if close below SMA150 by 0.4%
        for t, pos in list(positions.items()):
            df = data_by_ticker[t]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            px = float(row["Close"])
            sma = row["SMA150"]
            if pd.notna(sma) and px <= float(sma) * (1 - 0.004):
                proceeds = pos["qty"] * px
                pnl = proceeds - pos["cost"]
                cash += proceeds
                trades.append({
                    "profile": profile,
                    "ticker": t,
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "entry_price": pos["entry_price"],
                    "exit_price": px,
                    "qty": pos["qty"],
                    "pnl": pnl,
                    "return_pct": pnl / pos["cost"] if pos["cost"] else np.nan,
                    "exit_reason": "SMA150_BREAK",
                })
                del positions[t]

        # recompute equity after exits
        equity = cash
        for t, pos in positions.items():
            df = data_by_ticker[t]
            if dt in df.index:
                equity += pos["qty"] * float(df.loc[dt, "Close"])

        # entries: one position per ticker, capped
        for t, df in data_by_ticker.items():
            if t in positions or dt not in df.index:
                continue
            row = df.loc[dt]
            px = row["Close"]
            sma = row["SMA150"]
            pivot = row["Pivot"]
            vol_pace = row["VolPace"]

            if pd.isna(px) or pd.isna(sma) or pd.isna(pivot) or pd.isna(vol_pace):
                continue

            px = float(px)
            sma = float(sma)
            pivot = float(pivot)
            vol_pace = float(vol_pace)

            buy = (
                px > sma * (1 + params.dist_above_ma_min)
                and px >= pivot * (1 + params.min_break_pct)
                and vol_pace >= params.vol_min
            )

            if not buy:
                continue

            equity_now = cash + sum(
                positions[x]["qty"] * float(data_by_ticker[x].loc[dt, "Close"])
                for x in positions
                if dt in data_by_ticker[x].index
            )

            current_exposure = sum(
                positions[x]["qty"] * float(data_by_ticker[x].loc[dt, "Close"])
                for x in positions
                if dt in data_by_ticker[x].index
            )
            max_total_exposure = equity_now * max_exposure_frac
            remaining_exposure = max(0.0, max_total_exposure - current_exposure)

            invest = min(cash, equity_now * max_pos_frac, remaining_exposure)
            if invest <= 100:
                continue

            qty = invest / px
            cash -= invest
            positions[t] = {
                "qty": qty,
                "cost": invest,
                "entry_price": px,
                "entry_date": dt,
            }

        # final mtm
        equity = cash
        for t, pos in positions.items():
            df = data_by_ticker[t]
            if dt in df.index:
                equity += pos["qty"] * float(df.loc[dt, "Close"])

        equity_rows.append({
            "date": dt,
            "profile": profile,
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
        })

    # close remaining at end
    if equity_rows:
        last_date = equity_rows[-1]["date"]
        for t, pos in list(positions.items()):
            df = data_by_ticker[t]
            px = float(df.loc[last_date, "Close"])
            proceeds = pos["qty"] * px
            pnl = proceeds - pos["cost"]
            cash += proceeds
            trades.append({
                "profile": profile,
                "ticker": t,
                "entry_date": pos["entry_date"],
                "exit_date": last_date,
                "entry_price": pos["entry_price"],
                "exit_price": px,
                "qty": pos["qty"],
                "pnl": pnl,
                "return_pct": pnl / pos["cost"] if pos["cost"] else np.nan,
                "exit_reason": "END",
            })

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame(trades)

    final_equity = float(eq["equity"].iloc[-1]) if not eq.empty else capital
    summary = {
        "profile": profile,
        "start": start,
        "end": end,
        "capital": capital,
        "final_equity": final_equity,
        "net_return_pct": (final_equity / capital - 1) * 100,
        "cagr_pct": cagr(capital, final_equity, start, end) * 100,
        "max_drawdown_pct": max_drawdown(eq["equity"].values) * 100 if not eq.empty else 0,
        "trades": len(tr),
        "win_rate_pct": (float((tr["pnl"] > 0).mean()) * 100) if not tr.empty else np.nan,
        "profit_factor": profit_factor(trades),
        "avg_trade_return_pct": float(tr["return_pct"].mean() * 100) if not tr.empty else np.nan,
    }
    return summary, eq, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--capital", type=float, default=20000)
    ap.add_argument("--max-pos-frac", type=float, default=0.10)
    ap.add_argument("--max-exposure-frac", type=float, default=0.50)
    ap.add_argument("--profiles", default="A,B,C,D,E,F")
    ap.add_argument("--universe", default=",".join(UNIVERSE))
    args = ap.parse_args()

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = f"output/crypto_profile_tournament_{stamp}"
    os.makedirs(outdir, exist_ok=True)

    universe = [x.strip().upper() for x in args.universe.split(",") if x.strip()]
    profiles = [x.strip().upper() for x in args.profiles.split(",") if x.strip()]

    raw = download_data(universe, args.start, args.end)
    data_by_ticker = {}
    for t in universe:
        df = get_ticker_df(raw, t)
        if df.empty:
            print(f"WARNING no data for {t}")
            continue
        data_by_ticker[t] = prepare_features(df)

    summaries = []
    all_eq = []
    all_tr = []

    for p in profiles:
        print(f"Running crypto profile {p}")
        summary, eq, tr = run_profile(
            data_by_ticker=data_by_ticker,
            profile=p,
            start=args.start,
            end=args.end,
            capital=args.capital,
            max_pos_frac=args.max_pos_frac,
            max_exposure_frac=args.max_exposure_frac,
        )
        summaries.append(summary)
        all_eq.append(eq)
        all_tr.append(tr)

    summary_df = pd.DataFrame(summaries)
    equity_df = pd.concat(all_eq, ignore_index=True) if all_eq else pd.DataFrame()
    trades_df = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()

    summary_path = os.path.join(outdir, "crypto_profile_summary.csv")
    equity_path = os.path.join(outdir, "crypto_profile_equity.csv")
    trades_path = os.path.join(outdir, "crypto_profile_trades.csv")

    summary_df.to_csv(summary_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print()
    print(summary_df.to_string(index=False))
    print()
    print(f"Done. Results in: {outdir}")


if __name__ == "__main__":
    main()
