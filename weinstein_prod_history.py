#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_prod_history.py

Small operational helper for preserving PROD intraday signal history.

Why this exists:
- output/intraday_debug.csv is the latest snapshot and is overwritten each scan.
- Daily parity/routing reports need to know whether PROD produced BUY/NEAR/SELL
  at any time during the trading session, not only in the final snapshot.

This module is reporting/observability only. It does not alter Weinstein CORE logic.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROD_ACTIONABLE_SIGNALS = {"BUY", "NEAR", "SELL", "SHORT"}


def norm_signal(x: object) -> str:
    s = str(x or "").strip().upper()
    if s in {"NEAR_BUY", "NEAR-TRIGGER"}:
        return "NEAR"
    if s in {"SELLTRIG", "SELL-TRIGGER", "SELL-WATCH"}:
        return "SELL"
    if s in {"BUY", "NEAR", "SELL", "SHORT"}:
        return s
    return s


def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        low = name.lower()
        if low in lookup:
            return lookup[low]
    return None


def normalize_signal_frame(df: pd.DataFrame, source: str = "PROD") -> pd.DataFrame:
    """Normalize an intraday diagnostics/history frame to common signal columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])

    out = df.copy()

    ticker_col = _first_existing_col(out, ["Ticker", "ticker", "Symbol", "symbol"])
    signal_col = _first_existing_col(out, ["Signal", "signal", "Action", "action"])
    price_col = _first_existing_col(out, ["Price", "PriceNow", "price", "Close", "close", "close_price"])
    reason_col = _first_existing_col(out, ["Reason", "reason", "Details", "detail", "why"])

    norm = pd.DataFrame()
    norm["Ticker"] = out[ticker_col].astype(str).str.upper().str.strip() if ticker_col else ""
    norm["Signal"] = out[signal_col].apply(norm_signal) if signal_col else ""
    norm["Price"] = out[price_col] if price_col else ""
    norm["Reason"] = out[reason_col] if reason_col else ""
    norm["Source"] = source

    # Preserve useful audit metadata when present.
    for c in [
        "RunUTC", "RunCT", "RunDateCT", "SourceFile", "Structure", "WatchSignal", "WatchReason",
        "Pivot", "HeadroomPct", "VolPace", "ADX14", "PriceNow",
    ]:
        actual = _first_existing_col(out, [c])
        if actual:
            norm[c] = out[actual]

    norm = norm[norm["Ticker"].astype(str).str.strip().ne("")]
    norm = norm[norm["Signal"].isin(PROD_ACTIONABLE_SIGNALS)]
    return norm.reset_index(drop=True)


def append_prod_signal_history(diag: pd.DataFrame, history_path: str | Path, source_file: str = "") -> pd.DataFrame:
    """Append actionable BUY/NEAR/SELL/SHORT rows from the current PROD scan."""
    if diag is None or diag.empty:
        return pd.DataFrame()

    now_utc = datetime.utcnow().replace(microsecond=0)
    now_ct = datetime.now(ZoneInfo("America/Chicago")).replace(microsecond=0)

    rows = diag.copy()
    if "Ticker" not in rows.columns and "ticker" in rows.columns:
        rows["Ticker"] = rows["ticker"]
    if "Signal" not in rows.columns and "signal" in rows.columns:
        rows["Signal"] = rows["signal"]

    rows["Signal"] = rows.get("Signal", "").apply(norm_signal)
    rows = rows[rows["Signal"].isin(PROD_ACTIONABLE_SIGNALS)].copy()
    if rows.empty:
        return pd.DataFrame()

    rows.insert(0, "RunUTC", now_utc.isoformat())
    rows.insert(1, "RunCT", now_ct.isoformat())
    rows.insert(2, "RunDateCT", now_ct.strftime("%Y-%m-%d"))
    rows.insert(3, "SourceFile", source_file)

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not history_path.exists() or history_path.stat().st_size == 0
    rows.to_csv(history_path, mode="a", header=write_header, index=False)
    return rows


def read_prod_history_for_date(history_path: str | Path, date_ct: str | None = None) -> pd.DataFrame:
    """Read actionable PROD history for a Central-time trading date.

    If date_ct is not supplied, the latest RunDateCT in the file is used.
    """
    p = Path(history_path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])
    try:
        raw = pd.read_csv(p)
    except Exception:
        raw = pd.read_csv(p, engine="python", on_bad_lines="skip")

    if raw.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])

    if "RunDateCT" in raw.columns:
        if not date_ct:
            non_null = raw["RunDateCT"].dropna().astype(str)
            date_ct = non_null.max() if not non_null.empty else None
        if date_ct:
            raw = raw[raw["RunDateCT"].astype(str).eq(str(date_ct))].copy()

    return normalize_signal_frame(raw, source="PROD_INTRADAY_HISTORY")


def summarize_prod_history(history: pd.DataFrame) -> pd.DataFrame:
    """Collapse intraday signal history into one row per ticker/signal.

    Keeps first/last seen times and count. This prevents one recurring NEAR from
    appearing as many separate recommendations.
    """
    if history is None or history.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source", "FirstSeenCT", "LastSeenCT", "SeenCount"])

    df = history.copy()
    if "RunCT" not in df.columns:
        df["RunCT"] = ""
    if "Price" not in df.columns:
        df["Price"] = ""
    if "Reason" not in df.columns:
        df["Reason"] = ""

    grouped = []
    for (ticker, signal), g in df.groupby(["Ticker", "Signal"], dropna=False):
        g2 = g.copy()
        g2["_dt"] = pd.to_datetime(g2["RunCT"], errors="coerce")
        g2 = g2.sort_values("_dt")
        last = g2.tail(1).iloc[0]
        first_seen = g2["RunCT"].iloc[0] if "RunCT" in g2.columns else ""
        last_seen = g2["RunCT"].iloc[-1] if "RunCT" in g2.columns else ""
        grouped.append({
            "Ticker": ticker,
            "Signal": signal,
            "Price": last.get("Price", ""),
            "Reason": last.get("Reason", ""),
            "Source": "PROD_INTRADAY_HISTORY",
            "FirstSeenCT": first_seen,
            "LastSeenCT": last_seen,
            "SeenCount": len(g2),
        })
    return pd.DataFrame(grouped).sort_values(["Signal", "Ticker"]).reset_index(drop=True)
