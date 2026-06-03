#!/usr/bin/env python3
"""
Crypto A/B/C/D/E/F Weinstein research.

Research-only. Does not change stock PROD, stock SIM, stock META F, or any cron.
Long-only by default because Fidelity Crypto should be treated as long-only unless the broker explicitly supports shorting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


DEFAULT_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD",
    "AVAX-USD", "LINK-USD", "DOGE-USD", "XRP-USD",
    "ADA-USD", "BCH-USD",
]
FIDELITY_CRYPTO_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD"]


def normalize_crypto_symbol(sym: str) -> str:
    if sym is None:
        return ""
    s = str(sym).strip().upper().replace(" ", "")
    if not s or s in {"USD***", "PENDINGACTIVITY", "CASH", "FCASH", "SPAXX", "SPAXX**"}:
        return ""
    if "PENDING" in s:
        return ""
    if "/" in s:
        base, quote = s.split("/", 1)
        if quote in {"USD", "USDT", "USDC"}:
            return f"{base}-USD"
    if s.endswith("-USD"):
        return s
    known = {"BTC", "ETH", "SOL", "LTC", "AVAX", "LINK", "DOGE", "XRP", "ADA", "BCH", "NEAR", "TON", "TRX"}
    if s in known:
        return f"{s}-USD"
    return s


@dataclass(frozen=True)
class Profile:
    name: str
    ma_days: int = 150
    pivot_lookback: int = 60
    buy_buffer_pct: float = 0.004
    near_buffer_pct: float = 0.010
    sell_crack_pct: float = 0.005
    volume_min: float = 1.30
    adx_min: float = 18.0
    use_volume: bool = True
    use_adx: bool = True
    max_positions: int = 4


PROFILES: Dict[str, Profile] = {
    "A": Profile("A", volume_min=1.00, adx_min=0.0, use_volume=False, use_adx=False, max_positions=4),
    "B": Profile("B", volume_min=1.10, adx_min=14.0, use_volume=True, use_adx=True, max_positions=4),
    "C": Profile("C", volume_min=1.20, adx_min=16.0, use_volume=True, use_adx=True, max_positions=4),
    "D": Profile("D", volume_min=1.30, adx_min=18.0, use_volume=True, use_adx=True, max_positions=4),
    "E": Profile("E", volume_min=1.50, adx_min=22.0, use_volume=True, use_adx=True, buy_buffer_pct=0.006, max_positions=4),
    "F": Profile("F", volume_min=1.30, adx_min=18.0, use_volume=True, use_adx=True, near_buffer_pct=0.015, max_positions=3),
}


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    plus_di = 100 * plus_dm.rolling(n, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.rolling(n, min_periods=n).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(n, min_periods=n).mean()


def download_crypto(universe: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed.")
    tickers = sorted(set(normalize_crypto_symbol(t) for t in universe if normalize_crypto_symbol(t)))
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No crypto data returned from yfinance.")
    return data


def get_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            d = data[ticker].copy()
        elif ticker in data.columns.get_level_values(1):
            d = data.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        d = data.copy()
    cols = {str(c).lower(): c for c in d.columns}
    if not all(c in cols for c in ["open", "high", "low", "close", "volume"]):
        return pd.DataFrame()
    out = pd.DataFrame({
        "Open": d[cols["open"]],
        "High": d[cols["high"]],
        "Low": d[cols["low"]],
        "Close": d[cols["close"]],
        "Volume": d[cols["volume"]],
    }).dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    return out


def build_indicator_frame(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    d = df.copy()
    d["MA150"] = d["Close"].rolling(profile.ma_days, min_periods=profile.ma_days).mean()
    d["AvgVol50"] = d["Volume"].rolling(50, min_periods=20).mean()
    d["VolPace"] = d["Volume"] / d["AvgVol50"].replace(0, np.nan)
    d["Pivot"] = d["Close"].shift(1).rolling(profile.pivot_lookback, min_periods=profile.pivot_lookback).max()
    d["ADX14"] = compute_adx(d["High"], d["Low"], d["Close"], 14)
    return d


def generate_events(data: pd.DataFrame, universe: List[str], profile: Profile) -> pd.DataFrame:
    rows = []
    for ticker in universe:
        t = normalize_crypto_symbol(ticker)
        if not t:
            continue
        raw = get_ticker_frame(data, t)
        if raw.empty or len(raw) < profile.ma_days + profile.pivot_lookback:
            continue
        d = build_indicator_frame(raw, profile)
        for date, r in d.iterrows():
            close = float(r.get("Close", np.nan))
            ma = float(r.get("MA150", np.nan))
            pivot = float(r.get("Pivot", np.nan))
            volpace = float(r.get("VolPace", np.nan))
            adx = float(r.get("ADX14", np.nan))
            if not all(np.isfinite(x) for x in [close, ma, pivot]):
                continue
            above_ma = close > ma * (1.0 + profile.buy_buffer_pct)
            crossed_pivot = close > pivot * (1.0 + profile.buy_buffer_pct)
            near_pivot = close >= pivot * (1.0 - profile.near_buffer_pct)
            vol_ok = True if not profile.use_volume else (np.isfinite(volpace) and volpace >= profile.volume_min)
            adx_ok = True if not profile.use_adx else (np.isfinite(adx) and adx >= profile.adx_min)
            signal = "NONE"
            reason = ""
            if close < ma * (1.0 - profile.sell_crack_pct):
                signal = "SELL"
                reason = f"Close below MA150 by {profile.sell_crack_pct:.2%}"
            elif above_ma and crossed_pivot and vol_ok and adx_ok:
                signal = "BUY"
                reason = "Stage2/pivot/MA/volume/ADX gates passed"
            elif above_ma and near_pivot:
                signal = "NEAR"
                reason = "Above MA150 and near pivot"
            if signal != "NONE":
                rows.append({
                    "date": date.date().isoformat(),
                    "ticker": t,
                    "profile": profile.name,
                    "signal": signal,
                    "close": close,
                    "ma150": ma,
                    "pivot": pivot,
                    "vol_pace": volpace if np.isfinite(volpace) else "",
                    "adx14": adx if np.isfinite(adx) else "",
                    "reason": reason,
                })
    return pd.DataFrame(rows)


def run_portfolio(events: pd.DataFrame, price_data: pd.DataFrame, universe: List[str], profile: Profile, capital: float):
    tickers = sorted(set(normalize_crypto_symbol(t) for t in universe if normalize_crypto_symbol(t)))
    frames = {t: get_ticker_frame(price_data, t) for t in tickers}
    frames = {t: f for t, f in frames.items() if not f.empty}
    all_dates = sorted(set().union(*[set(f.index.date) for f in frames.values()])) if frames else []
    ev = events.copy()
    if ev.empty:
        ev = pd.DataFrame(columns=["date", "ticker", "signal"])
    ev["date"] = ev["date"].astype(str)
    cash = float(capital)
    positions = {}
    entry_price = {}
    trades = []
    equity_rows = []

    def price_on(t, d):
        f = frames.get(t)
        if f is None:
            return None
        mask = f.index.date <= d
        if not mask.any():
            return None
        return float(f.loc[mask, "Close"].iloc[-1])

    for d in all_dates:
        ds = d.isoformat()
        day_events = ev[ev["date"].eq(ds)]
        for _, row in day_events[day_events["signal"].eq("SELL")].iterrows():
            t = row["ticker"]
            if t in positions:
                px = price_on(t, d)
                if px and px > 0:
                    qty = positions.pop(t)
                    ep = entry_price.pop(t, px)
                    cash += qty * px
                    trades.append({"date": ds, "ticker": t, "action": "SELL", "price": px, "qty": qty, "pnl": (px-ep)*qty, "profile": profile.name})
        buys = day_events[day_events["signal"].eq("BUY")].copy()
        if not buys.empty:
            for c in ["adx14", "vol_pace"]:
                if c in buys.columns:
                    buys[c] = pd.to_numeric(buys[c], errors="coerce").fillna(0.0)
            buys = buys.sort_values([c for c in ["adx14", "vol_pace"] if c in buys.columns], ascending=False)
        for _, row in buys.iterrows():
            t = row["ticker"]
            if t in positions or len(positions) >= profile.max_positions:
                continue
            px = price_on(t, d)
            if not px or px <= 0:
                continue
            slot_value = cash / max(1, profile.max_positions - len(positions))
            if slot_value <= 1:
                continue
            qty = slot_value / px
            cash -= qty * px
            positions[t] = qty
            entry_price[t] = px
            trades.append({"date": ds, "ticker": t, "action": "BUY", "price": px, "qty": qty, "pnl": 0.0, "profile": profile.name})
        mv = sum((price_on(t, d) or 0) * qty for t, qty in positions.items())
        equity_rows.append({"date": ds, "profile": profile.name, "equity": cash + mv, "cash": cash, "positions": len(positions)})
    return pd.DataFrame(equity_rows), pd.DataFrame(trades)


def summarize(equity: pd.DataFrame, trades: pd.DataFrame):
    if equity.empty:
        return {}
    e = equity.copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values("date")
    start = float(e["equity"].iloc[0])
    end = float(e["equity"].iloc[-1])
    years = max((e["date"].iloc[-1] - e["date"].iloc[0]).days / 365.25, 1e-9)
    cagr = (end / start) ** (1 / years) - 1 if start > 0 else np.nan
    peak = e["equity"].cummax()
    dd = (e["equity"] / peak - 1).min()
    closed = trades[trades["action"].eq("SELL")] if not trades.empty else pd.DataFrame()
    wins = int((closed["pnl"] > 0).sum()) if not closed.empty and "pnl" in closed.columns else 0
    losses = int((closed["pnl"] <= 0).sum()) if not closed.empty and "pnl" in closed.columns else 0
    return {
        "profile": str(e["profile"].iloc[0]),
        "start_date": e["date"].iloc[0].date().isoformat(),
        "end_date": e["date"].iloc[-1].date().isoformat(),
        "start_equity": round(start, 2),
        "final_equity": round(end, 2),
        "total_return_pct": round((end/start - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "closed_trades": int(len(closed)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / max(1, wins + losses) * 100, 2),
    }


def yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    e = equity.copy()
    e["date"] = pd.to_datetime(e["date"])
    e["year"] = e["date"].dt.year
    out = []
    for y, g in e.groupby("year"):
        g = g.sort_values("date")
        s = float(g["equity"].iloc[0])
        f = float(g["equity"].iloc[-1])
        out.append({"profile": str(g["profile"].iloc[0]), "year": int(y), "start_equity": round(s,2), "final_equity": round(f,2), "return_pct": round((f/s-1)*100,2)})
    return pd.DataFrame(out)


def make_html(summary_df, yearly_df, outdir: Path, universe):
    pivot_html = ""
    if not yearly_df.empty:
        pivot = yearly_df.pivot_table(index="year", columns="profile", values="return_pct", aggfunc="first").reset_index()
        pivot_html = pivot.to_html(index=False, escape=True)
    html = [
        "<html><body>",
        "<h2>Crypto Weinstein A/B/C/D/E/F Research</h2>",
        f"<p><b>Universe:</b> {', '.join(universe)}</p>",
        "<p>Research only. No trades are placed.</p>",
        "<h3>Profile Summary</h3>",
        summary_df.to_html(index=False, escape=True) if not summary_df.empty else "<p>No rows.</p>",
        "<h3>Year-by-Year Returns (%)</h3>",
        pivot_html or "<p>No rows.</p>",
        "</body></html>",
    ]
    p = outdir / f"crypto_research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    p.write_text("\n".join(html), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    ap.add_argument("--fidelity-only", action="store_true")
    ap.add_argument("--profiles", default="A,B,C,D,E,F")
    ap.add_argument("--outdir", default="output/crypto_research")
    args = ap.parse_args()

    universe = FIDELITY_CRYPTO_UNIVERSE if args.fidelity_only else [normalize_crypto_symbol(x) for x in args.universe.split(",")]
    universe = [x for x in universe if x]

    outdir = Path(args.outdir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Crypto Weinstein research")
    print(f"Universe: {universe}")
    print(f"Start: {args.start} End: {args.end or 'latest'}")
    print(f"Outdir: {outdir}")

    data = download_crypto(universe, args.start, args.end)
    summary_rows = []
    yearly_rows = []

    for pname in [p.strip().upper() for p in args.profiles.split(",") if p.strip()]:
        prof = PROFILES[pname]
        print(f"\n=== Profile {pname} ===")
        events = generate_events(data, universe, prof)
        events.to_csv(outdir / f"crypto_{pname}_events.csv", index=False)
        equity, trades = run_portfolio(events, data, universe, prof, args.capital)
        equity.to_csv(outdir / f"crypto_{pname}_equity.csv", index=False)
        trades.to_csv(outdir / f"crypto_{pname}_trades.csv", index=False)
        print(f"Events={len(events)} EquityRows={len(equity)} Trades={len(trades)}")
        s = summarize(equity, trades)
        if s:
            summary_rows.append(s)
        yr = yearly_returns(equity)
        if not yr.empty:
            yearly_rows.append(yr)

    summary_df = pd.DataFrame(summary_rows)
    yearly_df = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    summary_path = outdir / "crypto_profile_summary.csv"
    yearly_path = outdir / "crypto_yearly_returns.csv"
    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    html_path = make_html(summary_df, yearly_df, outdir, universe)

    print("\nDONE crypto research")
    print(f"summary: {summary_path}")
    print(f"yearly: {yearly_path}")
    print(f"html: {html_path}")


if __name__ == "__main__":
    main()
