#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_prod_sim_timing_research.py

Research report:
- Compare durable PROD intraday signal history against latest SIM D / SIM F replay outputs.
- Answer timing questions:
  * Did PROD see ticker first?
  * Did SIM see same ticker?
  * Was signal direction aligned?
  * How persistent was PROD intraday signal?
  * Which PROD signals are also SIM candidates?

This script is additive and read-only.
"""

from __future__ import annotations

import argparse
import glob
import html
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def latest_parity_dir(base: str = "output/daily_parity") -> Optional[str]:
    dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def normalize_signal(s) -> str:
    x = str(s or "").upper().strip()
    if "BUY" in x and "NEAR" not in x:
        return "BUY"
    if "NEAR" in x:
        return "NEAR"
    if "SHORT" in x:
        return "SHORT"
    if "SELL" in x:
        return "SELL"
    return x if x and x != "NAN" else ""


def normalize_ticker(s) -> str:
    return str(s or "").upper().strip()


def load_prod_history(path: str) -> pd.DataFrame:
    df = norm_cols(read_csv_safe(path))
    if df.empty:
        return df

    ticker_col = find_col(df, ["Ticker", "ticker", "Symbol", "symbol"])
    signal_col = find_col(df, ["Signal", "signal"])
    runct_col = find_col(df, ["RunCT", "run_ct", "timestamp", "Timestamp"])
    price_col = find_col(df, ["PriceNow", "price", "SignalPrice"])
    reason_col = find_col(df, ["Reason", "reason"])
    vol_col = find_col(df, ["VolPace", "pace_full_vs50dma", "vol_pace"])
    adx_col = find_col(df, ["ADX14", "adx", "ADX"])
    pivot_col = find_col(df, ["Pivot", "pivot"])

    out = pd.DataFrame()
    out["Ticker"] = df[ticker_col].map(normalize_ticker) if ticker_col else ""
    out["PROD_Signal"] = df[signal_col].map(normalize_signal) if signal_col else ""
    out["RunCT"] = pd.to_datetime(df[runct_col], errors="coerce") if runct_col else pd.NaT
    out["PROD_Price"] = pd.to_numeric(df[price_col], errors="coerce") if price_col else pd.NA
    out["PROD_Pivot"] = pd.to_numeric(df[pivot_col], errors="coerce") if pivot_col else pd.NA
    out["PROD_VolPace"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else pd.NA
    out["PROD_ADX"] = pd.to_numeric(df[adx_col], errors="coerce") if adx_col else pd.NA
    out["PROD_Reason"] = df[reason_col].astype(str) if reason_col else ""
    out = out[out["Ticker"].ne("") & out["PROD_Signal"].isin(["BUY", "NEAR", "SELL", "SHORT"])]
    return out


def summarize_prod(prod: pd.DataFrame) -> pd.DataFrame:
    if prod.empty:
        return pd.DataFrame()

    def signals_join(s):
        vals = sorted(set([x for x in s if x]))
        return ",".join(vals)

    g = prod.groupby("Ticker", dropna=False).agg(
        PROD_FirstSeen=("RunCT", "min"),
        PROD_LastSeen=("RunCT", "max"),
        PROD_SeenCount=("Ticker", "count"),
        PROD_Signals=("PROD_Signal", signals_join),
        PROD_BuyCount=("PROD_Signal", lambda s: int((s == "BUY").sum())),
        PROD_NearCount=("PROD_Signal", lambda s: int((s == "NEAR").sum())),
        PROD_SellCount=("PROD_Signal", lambda s: int((s == "SELL").sum())),
        PROD_ShortCount=("PROD_Signal", lambda s: int((s == "SHORT").sum())),
        PROD_MaxVolPace=("PROD_VolPace", "max"),
        PROD_MaxADX=("PROD_ADX", "max"),
        PROD_LastPrice=("PROD_Price", "last"),
        PROD_LastPivot=("PROD_Pivot", "last"),
        PROD_LastReason=("PROD_Reason", "last"),
    ).reset_index()

    g["PROD_DurationMin"] = (g["PROD_LastSeen"] - g["PROD_FirstSeen"]).dt.total_seconds().div(60).round(1)
    return g.sort_values(["PROD_BuyCount", "PROD_NearCount", "PROD_SeenCount"], ascending=False)


def load_sim_signals_from_comparison(parity_dir: str) -> pd.DataFrame:
    comp = latest_file(os.path.join(parity_dir, "daily_prod_sim_signal_comparison_*.csv"))
    if not comp:
        return pd.DataFrame()

    df = norm_cols(read_csv_safe(comp))
    if df.empty:
        return df

    ticker_col = find_col(df, ["Ticker", "ticker"])
    d_col = find_col(df, ["SIM_D_Signal", "SIM_D_signal"])
    f_col = find_col(df, ["SIM_F_RawSignal", "SIM_F_Signal", "SIM_F_signal"])

    out = pd.DataFrame()
    out["Ticker"] = df[ticker_col].map(normalize_ticker) if ticker_col else ""
    out["SIM_D_Signal"] = df[d_col].map(normalize_signal) if d_col else ""
    out["SIM_F_Signal"] = df[f_col].map(normalize_signal) if f_col else ""
    out = out[out["Ticker"].ne("")]
    return out


def infer_sim_first_dates_from_events(parity_dir: str) -> pd.DataFrame:
    candidates = []
    for pattern, label in [
        ("sim_D_replay_events.csv", "D"),
        ("sim_F_base_events.csv", "F"),
        ("*sim_D*events*.csv", "D"),
        ("*sim_F*events*.csv", "F"),
    ]:
        for f in glob.glob(os.path.join(parity_dir, pattern)):
            candidates.append((f, label))

    rows = []
    seen_files = set()
    for path, label in candidates:
        if path in seen_files:
            continue
        seen_files.add(path)

        df = norm_cols(read_csv_safe(path))
        if df.empty:
            continue

        ticker_col = find_col(df, ["ticker", "Ticker", "symbol", "Symbol"])
        signal_col = find_col(df, ["signal", "Signal", "event", "Event", "action", "Action"])
        date_col = find_col(df, ["date", "Date", "entry_date", "EntryDate", "timestamp", "Timestamp", "bar_date"])

        if not ticker_col:
            continue

        work = pd.DataFrame()
        work["Ticker"] = df[ticker_col].map(normalize_ticker)
        work["SIM_Profile"] = label
        work["SIM_EventSignal"] = df[signal_col].map(normalize_signal) if signal_col else ""
        work["SIM_EventDate"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
        work = work[work["Ticker"].ne("")]

        if signal_col:
            work = work[work["SIM_EventSignal"].isin(["BUY", "NEAR", "SELL", "SHORT"])]

        rows.append(work)

    if not rows:
        return pd.DataFrame()

    all_events = pd.concat(rows, ignore_index=True)
    if all_events.empty:
        return pd.DataFrame()

    summary = all_events.groupby(["Ticker", "SIM_Profile"], dropna=False).agg(
        SIM_FirstEventDate=("SIM_EventDate", "min"),
        SIM_LastEventDate=("SIM_EventDate", "max"),
        SIM_EventCount=("Ticker", "count"),
        SIM_EventSignals=("SIM_EventSignal", lambda s: ",".join(sorted(set([x for x in s if x])))),
    ).reset_index()

    d = summary[summary["SIM_Profile"] == "D"].drop(columns=["SIM_Profile"]).add_prefix("SIM_D_")
    d = d.rename(columns={"SIM_D_Ticker": "Ticker"})
    f = summary[summary["SIM_Profile"] == "F"].drop(columns=["SIM_Profile"]).add_prefix("SIM_F_")
    f = f.rename(columns={"SIM_F_Ticker": "Ticker"})

    if not d.empty and not f.empty:
        return pd.merge(d, f, on="Ticker", how="outer")
    if not d.empty:
        return d
    return f


def build_report(prod_summary: pd.DataFrame, sim_state: pd.DataFrame, sim_dates: pd.DataFrame) -> pd.DataFrame:
    out = prod_summary.copy()

    if not sim_state.empty:
        out = out.merge(sim_state, on="Ticker", how="left")

    if not sim_dates.empty:
        out = out.merge(sim_dates, on="Ticker", how="left")

    out["SIM_D_MatchedTicker"] = out.get("SIM_D_Signal", pd.Series(index=out.index, dtype=object)).fillna("").ne("")
    out["SIM_F_MatchedTicker"] = out.get("SIM_F_Signal", pd.Series(index=out.index, dtype=object)).fillna("").ne("")

    out["PROD_HasBUY"] = out["PROD_Signals"].fillna("").str.contains("BUY")
    out["SIM_D_HasBUY"] = out.get("SIM_D_Signal", pd.Series(index=out.index, dtype=object)).fillna("").str.contains("BUY")
    out["SIM_F_HasBUY"] = out.get("SIM_F_Signal", pd.Series(index=out.index, dtype=object)).fillna("").str.contains("BUY")

    out["BUY_Aligned_D"] = out["PROD_HasBUY"] & out["SIM_D_HasBUY"]
    out["BUY_Aligned_F"] = out["PROD_HasBUY"] & out["SIM_F_HasBUY"]

    if "SIM_D_SIM_FirstEventDate" in out.columns:
        prod_dt = pd.to_datetime(out["PROD_FirstSeen"], errors="coerce").dt.tz_localize(None)
        sim_dt = pd.to_datetime(out["SIM_D_SIM_FirstEventDate"], errors="coerce").dt.tz_localize(None)
        out["PROD_vs_SIM_D_FirstDeltaDays"] = (prod_dt - sim_dt).dt.total_seconds().div(86400).round(2)

    return out.sort_values(["PROD_HasBUY", "BUY_Aligned_D", "BUY_Aligned_F", "PROD_SeenCount"], ascending=False)


def html_table(df: pd.DataFrame, max_rows=200) -> str:
    if df.empty:
        return "<p><i>No rows.</i></p>"
    return df.head(max_rows).to_html(index=False, escape=True)


def build_html(report: pd.DataFrame, prod: pd.DataFrame, sim_state: pd.DataFrame, parity_dir: str, generated: str) -> str:
    kpi = {
        "PROD tickers": report["Ticker"].nunique() if not report.empty else 0,
        "PROD BUY tickers": int(report["PROD_HasBUY"].sum()) if "PROD_HasBUY" in report else 0,
        "PROD events": len(prod),
        "SIM D state rows": len(sim_state),
        "BUY aligned D": int(report["BUY_Aligned_D"].sum()) if "BUY_Aligned_D" in report else 0,
        "BUY aligned F": int(report["BUY_Aligned_F"].sum()) if "BUY_Aligned_F" in report else 0,
    }
    cards = "".join(f"<div class='card'><b>{html.escape(k)}</b><br>{v}</div>" for k, v in kpi.items())

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PROD vs SIM Timing Research</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 5px; vertical-align: top; }}
th {{ background: #f3f5f8; }}
.card {{ display:inline-block; padding:12px 16px; margin:6px; background:#f7f7f7; border:1px solid #ddd; border-radius:8px; }}
.small {{ color:#555; font-size:12px; }}
</style>
</head>
<body>
<h1>PROD vs SIM Timing Research</h1>
<p class="small">Generated: {html.escape(generated)} | Parity dir: {html.escape(parity_dir)}</p>
{cards}
<h2>Research Table</h2>
{html_table(report, 500)}
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-history", default="output/prod_intraday_signal_history.csv")
    ap.add_argument("--parity-dir", default="")
    ap.add_argument("--out-dir", default="output/prod_sim_timing_research")
    args = ap.parse_args()

    parity_dir = args.parity_dir or latest_parity_dir() or ""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prod = load_prod_history(args.prod_history)
    if prod.empty:
        raise SystemExit(f"No PROD history rows found at {args.prod_history}")

    prod_summary = summarize_prod(prod)
    sim_state = load_sim_signals_from_comparison(parity_dir) if parity_dir else pd.DataFrame()
    sim_dates = infer_sim_first_dates_from_events(parity_dir) if parity_dir else pd.DataFrame()
    report = build_report(prod_summary, sim_state, sim_dates)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_out = out_dir / f"prod_signal_history_normalized_{stamp}.csv"
    summary_out = out_dir / f"prod_signal_lifecycle_summary_{stamp}.csv"
    report_out = out_dir / f"prod_vs_sim_timing_research_{stamp}.csv"
    html_out = out_dir / f"prod_vs_sim_timing_research_{stamp}.html"

    prod.to_csv(history_out, index=False)
    prod_summary.to_csv(summary_out, index=False)
    report.to_csv(report_out, index=False)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_out.write_text(build_html(report, prod, sim_state, parity_dir, generated), encoding="utf-8")

    print("DONE PROD vs SIM timing research")
    print(f"PROD events: {len(prod)}")
    print(f"PROD tickers: {prod['Ticker'].nunique()}")
    print(f"Parity dir: {parity_dir}")
    print(f"report: {report_out}")
    print(f"html: {html_out}")

    cols = [c for c in [
        "Ticker", "PROD_Signals", "PROD_FirstSeen", "PROD_LastSeen", "PROD_SeenCount",
        "PROD_BuyCount", "PROD_NearCount", "SIM_D_Signal", "SIM_F_Signal",
        "BUY_Aligned_D", "BUY_Aligned_F"
    ] if c in report.columns]
    print(report[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
