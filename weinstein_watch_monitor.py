#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_watch_monitor.py

Operational observability layer for WeinsteinAgent.

Purpose:
- Aggregate current PROD intraday diagnostics and validation diagnostics.
- Track WATCH pressure, BUY/NEAR/SIGNAL counts, SKIP-VOL, no_breakout_vs_pivot,
  and dominant gate/reason patterns.
- Produce CSV + HTML outputs for manual review or cron delivery.

This script does NOT:
- change thresholds
- place trades
- alter CORE logic
- alter PROD cron behavior
"""

from __future__ import annotations

import argparse
import html
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_OUT_DIR = Path("output/watch_monitor")


def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def get_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        low = n.lower()
        if low in lookup:
            return lookup[low]
    return None


def infer_file_timestamp(path: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path)
    m = re.search(r"(20\d{6})[_-](\d{6})", name)
    if m:
        try:
            return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
        except Exception:
            pass
    try:
        return pd.to_datetime(datetime.fromtimestamp(os.path.getmtime(path)))
    except Exception:
        return None


def normalize_debug_frame(path: str, source: str) -> pd.DataFrame:
    raw = read_csv_safe(path)
    if raw.empty:
        return pd.DataFrame()

    ticker_col = get_col(raw, "Ticker", "ticker", "symbol")
    signal_col = get_col(raw, "Signal", "signal")
    reason_col = get_col(raw, "Reason", "reason")
    watch_col = get_col(raw, "WatchSignal", "watch_signal", "watchsignal")
    watch_reason_col = get_col(raw, "WatchReason", "watch_reason", "watchreason")
    vol_col = get_col(raw, "VolPace", "vol_pace", "pace_full_vs50dma")
    headroom_col = get_col(raw, "HeadroomPct", "headroom_pct")
    price_col = get_col(raw, "PriceNow", "price_now", "price")
    pivot_col = get_col(raw, "Pivot", "pivot")
    adx_col = get_col(raw, "ADX14", "adx14", "adx")

    out = pd.DataFrame()
    out["source"] = source
    out["source_file"] = path
    out["file_timestamp"] = infer_file_timestamp(path)
    out["ticker"] = raw[ticker_col].astype(str).str.upper().str.strip() if ticker_col else ""
    out["signal"] = raw[signal_col].astype(str).str.upper().str.strip() if signal_col else ""
    out["reason"] = raw[reason_col].fillna("").astype(str).str.strip() if reason_col else ""
    out["watch_signal"] = raw[watch_col].fillna("").astype(str).str.upper().str.strip() if watch_col else ""
    out["watch_reason"] = raw[watch_reason_col].fillna("").astype(str).str.strip() if watch_reason_col else ""
    out["vol_pace"] = pd.to_numeric(raw[vol_col], errors="coerce") if vol_col else pd.NA
    out["headroom_pct"] = pd.to_numeric(raw[headroom_col], errors="coerce") if headroom_col else pd.NA
    out["price_now"] = pd.to_numeric(raw[price_col], errors="coerce") if price_col else pd.NA
    out["pivot"] = pd.to_numeric(raw[pivot_col], errors="coerce") if pivot_col else pd.NA
    out["adx14"] = pd.to_numeric(raw[adx_col], errors="coerce") if adx_col else pd.NA

    for c in [
        "cond_weekly_stage_ok",
        "cond_rs_ok",
        "cond_ma_ok",
        "cond_pivot_ok",
        "cond_buy_price_ok",
        "cond_buy_vol_ok",
        "cond_pace_full_gate",
        "cond_near_pace_gate",
        "cond_near_now",
        "buy_confirm",
    ]:
        actual = get_col(raw, c)
        out[c] = raw[actual] if actual else pd.NA

    ts = pd.to_datetime(out["file_timestamp"], errors="coerce")
    out["date"] = ts.dt.date.astype(str)
    out["hour"] = ts.dt.strftime("%Y-%m-%d %H:00")
    out = out[out["ticker"].astype(str).str.strip().ne("")]
    return out.reset_index(drop=True)


def collect_frames(paths: Iterable[str], source: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = normalize_debug_frame(p, source)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bool_true_count(s: pd.Series) -> int:
    return int(s.fillna("").astype(str).str.lower().isin(["true", "1", "yes"]).sum())


def summarize_group(g: pd.DataFrame) -> dict:
    signal = g["signal"].fillna("").astype(str).str.upper()
    reason = g["reason"].fillna("").astype(str)
    watch = g["watch_signal"].fillna("").astype(str).str.upper()

    return {
        "rows": len(g),
        "unique_tickers": g["ticker"].nunique(),
        "buy": int((signal == "BUY").sum()),
        "near": int((signal == "NEAR").sum()),
        "sell": int(signal.str.contains("SELL", na=False).sum()),
        "none": int((signal == "NONE").sum()),
        "skip_stage": int(signal.str.contains("SKIP-STAGE", na=False).sum()),
        "skip_vol": int(signal.str.contains("SKIP-VOL", na=False).sum()),
        "no_breakout_vs_pivot": int(reason.str.contains("no_breakout_vs_pivot", case=False, na=False).sum()),
        "watch_rows": int(watch.str.startswith("WATCH", na=False).sum()),
        "watch_low_volume": int((watch == "WATCH_LOW_VOLUME").sum()),
        "watch_near_pivot": int((watch == "WATCH_NEAR_PIVOT").sum()),
        "watch_breakout_price": int((watch == "WATCH_BREAKOUT_PRICE").sum()),
        "avg_vol_pace": round(float(pd.to_numeric(g["vol_pace"], errors="coerce").mean()), 4),
        "max_vol_pace": round(float(pd.to_numeric(g["vol_pace"], errors="coerce").max()), 4),
        "min_abs_headroom_pct": round(float(pd.to_numeric(g["headroom_pct"], errors="coerce").abs().min()), 4),
        "buy_price_ok": bool_true_count(g.get("cond_buy_price_ok", pd.Series(dtype=object))),
        "buy_vol_ok": bool_true_count(g.get("cond_buy_vol_ok", pd.Series(dtype=object))),
        "near_pace_gate_ok": bool_true_count(g.get("cond_near_pace_gate", pd.Series(dtype=object))),
        "buy_confirm_ok": bool_true_count(g.get("buy_confirm", pd.Series(dtype=object))),
    }


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for (source, date), g in df.groupby(["source", "date"], dropna=False):
        d = {"source": source, "date": date}
        d.update(summarize_group(g))
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["date", "source"])


def hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for (source, hour), g in df.groupby(["source", "hour"], dropna=False):
        d = {"source": source, "hour": hour}
        d.update(summarize_group(g))
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["hour", "source"])


def top_watch_reasons(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df[df["watch_reason"].fillna("").astype(str).str.strip().ne("")]
    if x.empty:
        return pd.DataFrame()
    return (
        x.groupby(["source", "watch_signal", "watch_reason"])
        .agg(count=("ticker", "count"), tickers=("ticker", lambda s: ", ".join(sorted(set(s.astype(str)))[:25])))
        .reset_index()
        .sort_values(["count"], ascending=False)
        .head(limit)
    )


def top_candidates(df: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    x["abs_headroom_pct"] = pd.to_numeric(x["headroom_pct"], errors="coerce").abs()
    x["vol_pace_sort"] = pd.to_numeric(x["vol_pace"], errors="coerce").fillna(-1)
    x["watch_priority"] = x["watch_signal"].fillna("").astype(str).str.startswith("WATCH").map({True: 0, False: 1})
    x = x.sort_values(["watch_priority", "abs_headroom_pct", "vol_pace_sort"], ascending=[True, True, False])
    cols = [
        "source", "file_timestamp", "ticker", "signal", "reason", "watch_signal", "watch_reason",
        "price_now", "pivot", "headroom_pct", "vol_pace", "adx14",
        "cond_buy_price_ok", "cond_buy_vol_ok", "cond_near_pace_gate", "buy_confirm",
    ]
    return x[[c for c in cols if c in x.columns]].head(limit)


def html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p><i>No data.</i></p>"
    return df.head(max_rows).to_html(index=False, escape=True)


def build_html(daily: pd.DataFrame, hourly: pd.DataFrame, reasons: pd.DataFrame, candidates: pd.DataFrame, generated: str) -> str:
    latest_daily = daily.tail(10) if not daily.empty else daily
    latest_hourly = hourly.tail(20) if not hourly.empty else hourly

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Weinstein Watch Monitor</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ color: #123; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
th {{ background: #f2f4f8; }}
.small {{ color: #555; font-size: 12px; }}
</style>
</head>
<body>
<h1>Weinstein Watch Monitor</h1>
<p class="small">Generated: {html.escape(generated)}</p>
<p>This is an operational telemetry report. It does not place trades and does not change thresholds.</p>

<h2>Latest Daily Summary</h2>
{html_table(latest_daily, 30)}

<h2>Latest Hourly Summary</h2>
{html_table(latest_hourly, 40)}

<h2>Top Watch Reasons</h2>
{html_table(reasons, 50)}

<h2>Top Watch / Near-Pivot Candidates</h2>
{html_table(candidates, 100)}

</body>
</html>
"""


def send_report(html_body: str, summary_text: str) -> None:
    from weinstein_mailer import send_email
    send_email(
        subject="Weinstein Watch Monitor",
        html_body=html_body,
        text_body=summary_text,
        cfg_path="config.yaml",
        subject_tag="WATCH-Monitor",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-debug", default="output/intraday_debug.csv")
    ap.add_argument("--validation-debug", default="output/intraday_debug_validation.csv")
    ap.add_argument("--include-validation", action="store_true", default=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--send-email", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.strict_debug and os.path.exists(args.strict_debug):
        frames.append(collect_frames([args.strict_debug], "STRICT_PROD"))
    if args.include_validation and args.validation_debug and os.path.exists(args.validation_debug):
        frames.append(collect_frames([args.validation_debug], "VALIDATION_TEST_EASE"))

    all_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_df.empty:
        print("No debug data found.")
        return

    daily = daily_summary(all_df)
    hourly = hourly_summary(all_df)
    reasons = top_watch_reasons(all_df)
    candidates = top_candidates(all_df)

    detail_path = out_dir / f"watch_monitor_detail_{stamp}.csv"
    daily_path = out_dir / f"watch_monitor_daily_summary_{stamp}.csv"
    hourly_path = out_dir / f"watch_monitor_hourly_summary_{stamp}.csv"
    reasons_path = out_dir / f"watch_monitor_top_watch_reasons_{stamp}.csv"
    candidates_path = out_dir / f"watch_monitor_top_candidates_{stamp}.csv"
    html_path = out_dir / f"watch_monitor_summary_{stamp}.html"

    all_df.to_csv(detail_path, index=False)
    daily.to_csv(daily_path, index=False)
    hourly.to_csv(hourly_path, index=False)
    reasons.to_csv(reasons_path, index=False)
    candidates.to_csv(candidates_path, index=False)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_body = build_html(daily, hourly, reasons, candidates, generated)
    html_path.write_text(html_body, encoding="utf-8")

    latest = daily.tail(10).to_string(index=False) if not daily.empty else "No daily rows."
    summary_text = f"Weinstein Watch Monitor generated {generated}\n\n{latest}"

    print("DONE watch monitor")
    print(f"Rows analyzed: {len(all_df)}")
    print(f"daily: {daily_path}")
    print(f"hourly: {hourly_path}")
    print(f"reasons: {reasons_path}")
    print(f"candidates: {candidates_path}")
    print(f"html: {html_path}")

    if args.send_email:
        send_report(html_body, summary_text)
        print("Email sent.")


if __name__ == "__main__":
    main()
