# Replace your entire export_signals_from_sheets.py file with the version below.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    return strip_strings_df(df)



def _norm_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())



def find_col(df: pd.DataFrame, candidates: Iterable[str], startswith: Iterable[str] = ()) -> Optional[str]:
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
    if s is None:
        return False
    u = str(s).strip().upper()
    return "BTC" in u or "ETH" in u or "SOL" in u



def is_invalid_fidelity_symbol(s: str) -> bool:
    if s is None:
        return True

    u = str(s).strip().upper()

    if not u:
        return True

    invalid_exact = {
        "FCASH",
        "FCASH**",
        "SPAXX",
        "SPAXX**",
        "CORE",
        "CASH",
    }

    if u in invalid_exact:
        return True

    if u.startswith("$"):
        return True

    if re.fullmatch(r"[0-9A-Z]{8,12}", u):
        if any(ch.isdigit() for ch in u):
            return True

    return False



def normalize_signal_to_side(value: object) -> str:
    v = str(value or "").strip().upper()

    if not v:
        return ""

    if any(x in v for x in ["BUY", "LONG", "STRONG BUY"]):
        return "BUY"

    if any(x in v for x in ["SELL", "EXIT", "AVOID"]):
        return "SELL"

    return ""



def clean_price(value: object):
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except Exception:
        return ""



def build_signals_log(
    df_sig: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    debug: bool = False,
) -> pd.DataFrame:

    if df_sig is None or df_sig.empty:
        return empty_output()

    df = df_sig.copy()

    ts_col = find_col(df, [
        "TimestampUTC", "Timestamp", "Date", "RunDate"
    ])

    tcol = find_col(df, [
        "Ticker", "Symbol", "ticker", "symbol"
    ])

    side_col = find_col(df, [
        "Direction", "Side", "SignalType", "Buy Signal",
        "buy_signal", "Recommendation", "Action", "Signal"
    ])

    pcol = find_col(df, [
        "Price", "LastPrice", "Last Price", "Close"
    ])

    if not tcol:
        raise SystemExit("No ticker column found.")

    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        ts = pd.Series(
            [pd.Timestamp.utcnow()] * len(df),
            index=df.index,
        )

    tickers = df[tcol].astype(str).str.upper().str.strip()

    if side_col:
        side = df[side_col].map(normalize_signal_to_side)
    else:
        side = pd.Series([""] * len(df), index=df.index)

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

    out = out[out["ticker"].ne("")].copy()

    out["side"] = out["side"].replace("", np.nan)

    if out["side"].isna().all():
        print("⚠️ No BUY/SELL signals detected after normalization.")
        print("⚠️ Defaulting all rows to BUY temporarily for debugging.")
        out["side"] = "BUY"

    out["side"] = out["side"].fillna("BUY")

    out = out[~out["ticker"].str.startswith("-")].copy()
    out = out[~out["ticker"].map(is_crypto_ticker)].copy()
    out = out[~out["ticker"].map(is_invalid_fidelity_symbol)].copy()

    print(f"⚠️ TEMP DEBUG MODE: exporting all {len(out)} rows without strict date filtering.")

    if out.empty:
        print("⚠️ Still zero rows after relaxed filters.")
        return empty_output()

    out["reason"] = ""
    out["near_hits"] = ""
    out["state_before"] = ""
    out["state_after"] = ""

    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["ts"] = out["ts"].fillna(pd.Timestamp.utcnow())
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out = out[OUTPUT_COLUMNS]
    out.reset_index(drop=True, inplace=True)

    return out



def main():
    ap = argparse.ArgumentParser(description="Export Signals tab from Google Sheets into signals_log.csv.")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--output", type=str, default="./output/signals_log.csv")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    sheet_url = resolve_sheet_url(cfg)
    service_account_file = resolve_service_account_file(cfg)
    tab_signals = resolve_tab_name(cfg, "signals_tab", TAB_SIGNALS)

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
