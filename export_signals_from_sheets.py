#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_signals_from_sheets.py

Export the Google Sheets "Signals" tab into a flat CSV that mimics
signal_engine.py's ./output/signals_log.csv format:

    ts,ticker,side,price,reason,near_hits,state_before,state_after

Usage example:

  python3 export_signals_from_sheets.py \
    --config config.yaml \
    --start  2025-01-01 \
    --end    2025-11-30 \
    --output ./output/signals_log.csv

Notes:
- We expect a timestamp column in the Signals tab whose name starts with
  "Timestamp" (e.g. "TimestampUTC").
- We expect a Ticker column named "Ticker" (or "Symbol").
- We expect a Direction column named "Direction" (BUY/SELL).
- Price column "Price" is optional; if missing, price will be empty.
- We ignore option-style tickers starting with "-" (e.g. -PLTR251114C190).
- We ignore cryptos (BTC, ETH, SOL variants) to match your equity-only logic.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import yaml

# ─────────────────────────────
# CONFIG HELPERS (mirrors build_performance_dashboard.py)
# ─────────────────────────────

DEFAULT_SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TAB_SIGNALS = "Signals"


def load_cfg(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_sheet_url(cfg: dict) -> Optional[str]:
    sheets = cfg.get("sheets", {}) or {}
    return sheets.get("url") or sheets.get("sheet_url") or os.getenv("SHEET_URL")


def resolve_service_account_file(cfg: dict) -> str:
    google = cfg.get("google", {}) or {}
    return (
        google.get("service_account_json")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or DEFAULT_SERVICE_ACCOUNT_FILE
    )


def resolve_tab_name(cfg: dict, key: str, default_name: str) -> str:
    sheets = cfg.get("sheets", {}) or {}
    return sheets.get(key, default_name)


# ─────────────────────────────
# SHEETS UTILS
# ─────────────────────────────

def auth_gspread(service_account_file: str):
    print(f"🔑 Authorizing service account with {os.path.abspath(service_account_file)} …")
    creds = Credentials.from_service_account_file(service_account_file, scopes=DEFAULT_SCOPES)
    print("🔑 Authorizing service account…")
    return gspread.authorize(creds)


def open_ws(gc, sheet_url: str, tab: str):
    sh = gc.open_by_url(sheet_url)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        raise SystemExit(f"Worksheet '{tab}' not found in sheet {sheet_url}")


def strip_strings_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return out


def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    return strip_strings_df(df)


# ─────────────────────────────
# DOMAIN HELPERS
# ─────────────────────────────

def is_crypto_ticker(s: str) -> bool:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return False
    u = str(s).strip().upper()
    if not u:
        return False
    crypto_exact = {
        "BTC-USD", "ETH-USD", "SOL-USD",
        "BTC/USD", "ETH/USD", "SOL/USD",
        "BTC", "ETH", "SOL",
        "BTCUSD", "ETHUSD", "SOLUSD",
    }
    if u in crypto_exact:
        return True
    if "BTC-USD" in u or "ETH-USD" in u or "SOL-USD" in u:
        return True
    if "BTC/USD" in u or "ETH/USD" in u or "SOL/USD" in u:
        return True
    return False


# ─────────────────────────────
# CORE EXPORT LOGIC
# ─────────────────────────────

def build_signals_log(
    df_sig: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if df_sig is None or df_sig.empty:
        return pd.DataFrame(columns=[
            "ts", "ticker", "side", "price", "reason",
            "near_hits", "state_before", "state_after"
        ])

    # Identify columns
    ts_col = next(
        (c for c in df_sig.columns if c.lower().startswith("timestamp")),
        None,
    )
    tcol = next(
        (c for c in df_sig.columns if c.lower() in ("ticker", "symbol")),
        None,
    )
    dcol = next(
        (c for c in df_sig.columns if c.lower() == "direction"),
        None,
    )
    pcol = "Price" if "Price" in df_sig.columns else None

    if not ts_col or not tcol or not dcol:
        raise SystemExit(
            f"Signals tab must have columns for timestamp (starts with 'Timestamp'), "
            f"'Ticker' (or 'Symbol') and 'Direction'. Found: {list(df_sig.columns)}"
        )

    # Parse timestamp as UTC directly; no tz_convert kwargs needed
    ts = pd.to_datetime(df_sig[ts_col], errors="coerce", utc=True)

    # Normalize ticker and side
    tickers = df_sig[tcol].astype(str).str.upper().str.strip()
    side = df_sig[dcol].astype(str).str.upper().str.strip()

    # Optional price
    if pcol:
        price_raw = df_sig[pcol]
    else:
        price_raw = pd.Series([""] * len(df_sig), index=df_sig.index)

    out = pd.DataFrame({
        "ts": ts,
        "ticker": tickers,
        "side": side,
        "price": price_raw,
    })

    # Drop bad rows
    mask_valid = out["ts"].notna() & out["ticker"].ne("") & out["side"].isin(["BUY", "SELL"])
    out = out[mask_valid].copy()

    # Drop options (start with "-") and basic cryptos
    out = out[~out["ticker"].str.startswith("-")]
    out = out[~out["ticker"].map(is_crypto_ticker)].copy()

    if out.empty:
        return pd.DataFrame(columns=[
            "ts", "ticker", "side", "price", "reason",
            "near_hits", "state_before", "state_after"
        ])

    # Filter to date window (inclusive)
    start_date = start.date()
    end_date = end.date()
    date_mask = out["ts"].dt.date.between(start_date, end_date)
    out = out[date_mask].copy()

    if out.empty:
        return pd.DataFrame(columns=[
            "ts", "ticker", "side", "price", "reason",
            "near_hits", "state_before", "state_after"
        ])

    # Coerce numeric price where possible; leave as string otherwise
    def _to_price(x):
        if isinstance(x, str):
            xs = x.replace("$", "").replace(",", "").strip()
            if xs == "":
                return ""
            try:
                return float(xs)
            except Exception:
                return xs
        try:
            return float(x)
        except Exception:
            return ""

    out["price"] = out["price"].map(_to_price)

    # Final formatting: ts as ISO8601 with Z suffix (UTC)
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fill stub columns used by signal_engine / backtest but not needed here
    out["reason"] = ""
    out["near_hits"] = ""
    out["state_before"] = ""
    out["state_after"] = ""

    # Column order
    out = out[["ts", "ticker", "side", "price", "reason", "near_hits", "state_before", "state_after"]]

    # Sort by timestamp
    out.sort_values("ts", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ─────────────────────────────
# CLI
# ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Export Signals tab from Google Sheets into signals_log.csv.")
    ap.add_argument("--config", type=str, default="config.yaml", help="YAML config (with sheets.url, google.service_account_json, etc.)")
    ap.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--output", type=str, default="./output/signals_log.csv", help="Output CSV path")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    sheet_url = resolve_sheet_url(cfg)
    service_account_file = resolve_service_account_file(cfg)
    tab_signals = resolve_tab_name(cfg, "signals_tab", TAB_SIGNALS)

    if not sheet_url:
        raise SystemExit("Sheet URL not found; set sheets.url or sheets.sheet_url in config.yaml or SHEET_URL env var.")

    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)

    gc = auth_gspread(service_account_file)
    print(f"📄 Reading Signals tab: {tab_signals} from {sheet_url}")
    ws_sig = open_ws(gc, sheet_url, tab_signals)
    df_sig = read_tab(ws_sig)

    print(f"• Signals rows in sheet: {len(df_sig)}")

    out_df = build_signals_log(df_sig, start_ts, end_ts)
    print(f"• Exported {len(out_df)} rows after date/asset filters.")

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Wrote signals log CSV → {out_path}")


if __name__ == "__main__":
    main()
