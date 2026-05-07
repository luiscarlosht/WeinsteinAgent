#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_signals_from_sheets.py

Export the Google Sheets "Signals" tab into a flat CSV that mimics
signal_engine.py's ./output/signals_log.csv format:

    ts,ticker,side,price,reason,near_hits,state_before,state_after

This version is intentionally tolerant of schema drift in the Signals tab.
It supports both older intraday-style signal columns and newer weekly scan
columns, for example:

    TimestampUTC / Timestamp / Date / RunDate / Generated
    Ticker / Symbol / ticker / symbol
    Direction / Side / SignalType / Buy Signal / buy_signal / Recommendation
    AssetType / asset_class / asset_class
    Price / LastPrice / Last Price / price
    Reason / Notes / notes

Why this file exists:
- The newer weekly scan writes rows with asset_class like "Equity/ETF" and
  signal columns like "Buy Signal" or "buy_signal".
- The older exporter expected Direction and Timestamp* columns only.
- That caused: "Exported 0 rows after date/asset filters" even when the
  Signals tab had valid rows.

Usage examples:

  python3 export_signals_from_sheets.py \
    --config config.yaml \
    --start 2026-01-01 \
    --end 2026-12-31 \
    --output ./output/signals_log.csv

  python3 export_signals_from_sheets.py \
    --config config.yaml \
    --start 2026-01-01 \
    --end 2026-12-31 \
    --output ./output/signals_log.csv \
    --debug

Notes:
- Options whose ticker starts with "-" are excluded by default.
- Basic crypto tickers are excluded by default.
- Rows with WATCH/NEAR/HOLD are not exported as trades by default.
- BUY/STRONG BUY map to BUY.
- SELL/EXIT/AVOID map to SELL.
- If a timestamp column is missing or blank for a valid row, the row is dated
  with --end at 16:00 UTC. This prevents weekly rows from being dropped only
  because the weekly scan did not store a timestamp.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import yaml

# ─────────────────────────────
# CONFIG HELPERS
# ─────────────────────────────

DEFAULT_SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TAB_SIGNALS = "Signals"
OUTPUT_COLUMNS = [
    "ts", "ticker", "side", "price", "reason",
    "near_hits", "state_before", "state_after",
]


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

def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "quota exceeded" in text
        or "rate limit" in text
        or "read requests per minute" in text
    )


def _sleep_for_retry(attempt: int) -> None:
    wait = min((2 ** attempt) + 1, 60)
    print(f"⚠️ Google Sheets quota/rate limit hit. Sleeping {wait}s before retry...")
    time.sleep(wait)


def auth_gspread(service_account_file: str):
    print(f"🔑 Authorizing service account with {os.path.abspath(service_account_file)} …")
    creds = Credentials.from_service_account_file(service_account_file, scopes=DEFAULT_SCOPES)
    print("🔑 Authorizing service account…")
    return gspread.authorize(creds)


def open_ws(gc, sheet_url: str, tab: str, retries: int = 8):
    last_err = None
    for attempt in range(retries):
        try:
            sh = gc.open_by_url(sheet_url)
            return sh.worksheet(tab)
        except gspread.WorksheetNotFound:
            raise SystemExit(f"Worksheet '{tab}' not found in sheet {sheet_url}")
        except APIError as e:
            last_err = e
            if _is_quota_error(e) and attempt < retries - 1:
                _sleep_for_retry(attempt)
                continue
            raise
    raise RuntimeError(f"Failed opening worksheet '{tab}' after retries") from last_err


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
    clean_header = [str(h).strip() for h in header]
    df = pd.DataFrame(rows, columns=clean_header)
    # Drop fully empty columns that sometimes appear after Google Sheets edits.
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    return strip_strings_df(df)


# ─────────────────────────────
# DOMAIN HELPERS
# ─────────────────────────────

def _norm_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_col(df: pd.DataFrame, candidates: Iterable[str], startswith: Iterable[str] = ()) -> Optional[str]:
    """Find a column by tolerant normalized name matching."""
    if df is None or df.empty:
        return None

    normalized = {_norm_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = _norm_col_name(cand)
        if key in normalized:
            return normalized[key]

    for prefix in startswith:
        p = _norm_col_name(prefix)
        for c in df.columns:
            if _norm_col_name(c).startswith(p):
                return c

    return None


def empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


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


def is_cash_or_cusip_like_ticker(s: str) -> bool:
    """Exclude Fidelity cash/core/cusip rows that can get merged into Signals."""
    if s is None:
        return True
    u = str(s).strip().upper()
    if not u:
        return True
    if u.startswith("$"):
        return True
    if u in {"CASH", "FCASH", "SPAXX", "SPAXX**", "FCASH**", "CORE"}:
        return True
    # Common CUSIP-like rows are not tradable tickers for the sim.
    if re.fullmatch(r"[0-9A-Z]{8,12}", u) and any(ch.isdigit() for ch in u):
        return True
    return False


def normalize_signal_to_side(value: object) -> str:
    """Map several schema variants into BUY/SELL/blank."""
    v = str(value or "").strip().upper()
    if not v or v in {"NAN", "NONE", "NULL", "-"}:
        return ""

    # Remove emojis / punctuation-ish decoration while keeping words.
    v_words = re.sub(r"[^A-Z0-9 /_-]+", " ", v)
    v_words = re.sub(r"\s+", " ", v_words).strip()

    buy_values = {
        "BUY", "LONG", "ENTER", "ENTRY", "OPEN LONG",
        "STRONG BUY", "HOLD STRONG", "ACCUMULATE",
    }
    sell_values = {
        "SELL", "EXIT", "CLOSE", "CLOSE LONG", "AVOID",
        "STAGE 4", "STOP", "RISK EXIT",
    }
    skip_values = {
        "WATCH", "NEAR", "NEAR TRIGGER", "HOLD", "NO", "N/A",
        "NEUTRAL", "WAIT", "IGNORE", "FILTERED", "",
    }

    if v_words in buy_values:
        return "BUY"
    if v_words in sell_values:
        return "SELL"
    if v_words in skip_values:
        return ""

    # Contains-based fallback. Keep WATCH from accidentally becoming BUY.
    if "WATCH" in v_words or "NEAR" in v_words:
        return ""
    if "AVOID" in v_words or "SELL" in v_words or "EXIT" in v_words:
        return "SELL"
    if "BUY" in v_words or "LONG" in v_words:
        return "BUY"

    return ""


def clean_price(value: object):
    if value is None:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value) or np.isinf(value):
            return ""
        return float(value)
    s = str(value).strip()
    if not s:
        return ""
    # Do not export Google Sheets formulas as prices to the simulator.
    if s.startswith("="):
        return ""
    s2 = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(s2)
    except Exception:
        return ""


def build_reason(row: pd.Series, reason_cols: list[str]) -> str:
    parts: list[str] = []
    for c in reason_cols:
        if c in row.index:
            v = str(row.get(c, "") or "").strip()
            if v and v.lower() not in {"nan", "none"}:
                parts.append(f"{c}: {v}")
    return " | ".join(parts)


def choose_timestamp_series(df: pd.DataFrame, ts_col: Optional[str], end: pd.Timestamp) -> pd.Series:
    """Return a UTC timestamp Series.

    If the Signals sheet does not provide usable timestamps for weekly rows,
    use --end at 16:00 UTC so the rows are still available to the sim instead
    of being silently filtered out.
    """
    fallback = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
    fallback = fallback.normalize() + pd.Timedelta(hours=16)

    if ts_col:
        parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        return parsed.fillna(fallback)

    return pd.Series([fallback] * len(df), index=df.index, dtype="datetime64[ns, UTC]")


# ─────────────────────────────
# CORE EXPORT LOGIC
# ─────────────────────────────

def build_signals_log(
    df_sig: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    debug: bool = False,
) -> pd.DataFrame:
    if df_sig is None or df_sig.empty:
        return empty_output()

    df = df_sig.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = find_col(
        df,
        candidates=[
            "TimestampUTC", "Timestamp", "SignalTimeUTC", "SignalTime",
            "Date", "Run Date", "RunDate", "Generated", "GeneratedUTC",
            "CreatedAt", "CreatedUTC",
        ],
        startswith=["Timestamp"],
    )
    tcol = find_col(df, candidates=["Ticker", "Symbol", "ticker", "symbol"])
    side_col = find_col(
        df,
        candidates=[
            "Direction", "Side", "SignalType", "Signal Type", "Buy Signal",
            "buy_signal", "Recommendation", "Action", "Signal",
        ],
    )
    pcol = find_col(
        df,
        candidates=[
            "Price", "LastPrice", "Last Price", "price", "Close", "Last",
            "Current Price", "PriceNow",
        ],
    )
    asset_col = find_col(
        df,
        candidates=["AssetType", "Asset Type", "asset_class", "Asset Class", "asset_class"],
    )

    reason_cols = [
        c for c in [
            find_col(df, candidates=["Reason", "reason"]),
            find_col(df, candidates=["Notes", "notes"]),
            find_col(df, candidates=["Stage", "stage"]),
            find_col(df, candidates=["short_term_state_wk", "ShortTermState"]),
        ]
        if c
    ]

    if debug:
        print("🔎 export_signals_from_sheets debug")
        print(f"Columns: {list(df.columns)}")
        print(f"Resolved ts_col={ts_col!r}, tcol={tcol!r}, side_col={side_col!r}, pcol={pcol!r}, asset_col={asset_col!r}")
        if asset_col:
            print("Asset values:")
            print(df[asset_col].astype(str).str.strip().value_counts(dropna=False).head(20).to_string())
        if side_col:
            print("Signal/side values:")
            print(df[side_col].astype(str).str.strip().value_counts(dropna=False).head(30).to_string())
        print("Sample rows:")
        print(df.head(5).to_string(index=False))

    if not tcol:
        raise SystemExit(
            "Signals tab must have a ticker column named Ticker or Symbol. "
            f"Found: {list(df.columns)}"
        )

    if not side_col:
        raise SystemExit(
            "Signals tab must have a signal column such as Direction, Side, "
            "SignalType, Buy Signal, buy_signal, Recommendation, Action, or Signal. "
            f"Found: {list(df.columns)}"
        )

    ts = choose_timestamp_series(df, ts_col, end)
    tickers = df[tcol].astype(str).str.upper().str.strip()
    side = df[side_col].map(normalize_signal_to_side)

    if pcol:
        prices = df[pcol].map(clean_price)
    else:
        prices = pd.Series([""] * len(df), index=df.index)

    out = pd.DataFrame({
        "ts": ts,
        "ticker": tickers,
        "side": side,
        "price": prices,
    })

    # Optional equity/ETF filter. Be permissive:
    # - If asset class exists and says Equity/ETF, include.
    # - If asset class is blank, include normal-looking stock tickers.
    # - Exclude explicit crypto/option/cash/CUSIP rows later.
    if asset_col:
        asset = df[asset_col].astype(str).str.upper().str.strip()
        is_equity = (
            asset.eq("")
            | asset.str.contains("EQUITY", na=False)
            | asset.str.contains("ETF", na=False)
            | asset.str.contains("STOCK", na=False)
        )
        out = out[is_equity].copy()

    # Drop rows that cannot represent actual simulator trades.
    out = out[out["ts"].notna()].copy()
    out = out[out["ticker"].ne("")].copy()
    out = out[out["side"].isin(["BUY", "SELL"])].copy()

    if out.empty:
        if debug:
            print("After basic valid side/ticker/timestamp filters: 0 rows")
        return empty_output()

    # Drop options, cryptos, Fidelity cash/core rows, and CUSIP-like rows.
    out = out[~out["ticker"].str.startswith("-")].copy()
    out = out[~out["ticker"].map(is_crypto_ticker)].copy()
    out = out[~out["ticker"].map(is_cash_or_cusip_like_ticker)].copy()

    if out.empty:
        if debug:
            print("After asset exclusions: 0 rows")
        return empty_output()

    # Filter to date window, inclusive.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    start_date = start_ts.date()
    end_date = end_ts.date()
    date_mask = out["ts"].dt.date.between(start_date, end_date)
    out = out[date_mask].copy()

    if out.empty:
        if debug:
            print(f"After date filter {start_date} to {end_date}: 0 rows")
        return empty_output()

    # Build reason after final index subset so row alignment is preserved.
    reasons = []
    for idx in out.index:
        try:
            reasons.append(build_reason(df.loc[idx], reason_cols))
        except Exception:
            reasons.append("")
    out["reason"] = reasons

    out["near_hits"] = ""
    out["state_before"] = ""
    out["state_after"] = ""

    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out[OUTPUT_COLUMNS]
    out.sort_values(["ts", "ticker", "side"], inplace=True)
    out.drop_duplicates(subset=["ts", "ticker", "side"], keep="last", inplace=True)
    out.reset_index(drop=True, inplace=True)

    if debug:
        print(f"Export rows after all filters: {len(out)}")
        if not out.empty:
            print(out.head(20).to_string(index=False))

    return out


# ─────────────────────────────
# CLI
# ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Export Signals tab from Google Sheets into signals_log.csv.")
    ap.add_argument("--config", type=str, default="config.yaml", help="YAML config with sheets.url and google.service_account_json")
    ap.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--output", type=str, default="./output/signals_log.csv", help="Output CSV path")
    ap.add_argument("--debug", action="store_true", help="Print schema/filter diagnostics")
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

    out_df = build_signals_log(df_sig, start_ts, end_ts, debug=args.debug)
    print(f"• Exported {len(out_df)} rows after date/asset filters.")

    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Wrote signals log CSV → {out_path}")


if __name__ == "__main__":
    main()
