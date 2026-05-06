#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Google Sheets dashboard tabs:
- Realized_Trades
- Open_Positions
- Performance_By_Source
- Performance_By_Month
- Performance_By_Month_Source      (NEW: by Month + Kind (Equity/Option) + Source)
- OpenLots_Detail
- Open_Positions_Snapshot
- Weekly_Report
- Options_Results                  (options PnL summary)

Also writes local CSVs under ./output:
- Open_Positions.csv
- Options_Results.csv

Options behaviour:
- Signals rows whose Ticker starts with "-" are parsed as option codes in the form:
    -<UNDERLYING><YY><MM><DD><C|P><STRIKE>
  Example: -PLTR251114C190 → underlying=PLTR, expiration=2025-11-14, right=C, strike=190
- We scan Transactions (raw rows) to find BUY/SELL activity for each option contract
  and compute realized P/L, status (WIN/LOSS/OPEN/NO_TRADES), counts and first/last trade dates.
- Options_Results now also carries a Source column (from Signals), so we can
  build a monthly-by-source breakdown that separates EQUITY vs OPTION.

NOTE on shorts:
- The FIFO engine for equities is still long-oriented (BUY→SELL). We do NOT attempt
  to compute full realized P/L by separating long vs short behavior inside FIFO.
- However, we now add **open short positions** into Open_Positions by reading
  negative-quantity rows from the Holdings tab, so Open_Positions and Snapshot
  reflect both longs and shorts that exist in the broker account.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import yaml

try:
    import yfinance as yf
except Exception:
    yf = None

# ─────────────────────────────
# DEFAULTS / TAB NAMES
# ─────────────────────────────
DEFAULT_SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TAB_SIGNALS      = "Signals"
TAB_TRANSACTIONS = "Transactions"
TAB_HOLDINGS     = "Holdings"
TAB_REALIZED     = "Realized_Trades"
TAB_OPEN         = "Open_Positions"
TAB_PERF         = "Performance_By_Source"
TAB_MONTHLY      = "Performance_By_Month"          # NEW
TAB_MONTHLY_SRC  = "Performance_By_Month_Source"   # NEW (Month + Kind (Equity/Option) + Source)
TAB_OPEN_DETAIL  = "OpenLots_Detail"
TAB_SNAPSHOT     = "Open_Positions_Snapshot"
TAB_WEEKLY       = "Weekly_Report"
TAB_OPTIONS      = "Options_Results"

DEFAULT_EXCHANGE_PREFIX = "NYSE: "
ROW_CHUNK = 500

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
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
    print("🔑 Authorizing service account…")
    creds = Credentials.from_service_account_file(service_account_file, scopes=DEFAULT_SCOPES)
    return gspread.authorize(creds)

def _is_quota_error(exc: Exception) -> bool:
    """Return True for Google Sheets quota/rate-limit errors."""
    text = str(exc).lower()
    return (
        "429" in text
        or "quota exceeded" in text
        or "rate limit" in text
        or "read requests per minute" in text
    )


def _sleep_for_retry(attempt: int) -> None:
    """Exponential backoff with a small cap so weekly jobs recover from 429s."""
    wait = min((2 ** attempt) + 1, 60)
    print(f"⚠️ Google Sheets quota/rate limit hit. Sleeping {wait}s before retry...")
    time.sleep(wait)


# Cache Google Sheets objects during one script run.
# This prevents repeated gc.open_by_url() / worksheet() metadata reads,
# which were causing Sheets API 429 quota errors during the weekly pipeline.
_SPREADSHEET_CACHE = {}
_WORKSHEET_CACHE = {}


def get_spreadsheet(gc, sheet_url: str, retries: int = 8):
    """Open the spreadsheet once per process and reuse it.

    IMPORTANT: gc.open_by_url() itself calls fetch_sheet_metadata(), so if we
    call it repeatedly for every tab, we can hit the Google Sheets read quota
    even before worksheet() is reached.
    """
    if sheet_url in _SPREADSHEET_CACHE:
        return _SPREADSHEET_CACHE[sheet_url]

    last_err = None
    for attempt in range(retries):
        try:
            sh = gc.open_by_url(sheet_url)
            _SPREADSHEET_CACHE[sheet_url] = sh
            return sh
        except APIError as e:
            last_err = e
            if _is_quota_error(e) and attempt < retries - 1:
                _sleep_for_retry(attempt)
                continue
            raise

    raise RuntimeError("Failed opening spreadsheet after retries") from last_err


def open_ws(gc, sheet_url: str, tab: str, retries: int = 8):
    """Open/create a worksheet once per process and reuse it.

    This caches both the Spreadsheet object and Worksheet object. It also keeps
    retry/backoff for temporary Google Sheets 429 quota/rate-limit errors.
    """
    key = (sheet_url, tab)
    if key in _WORKSHEET_CACHE:
        return _WORKSHEET_CACHE[key]

    sh = get_spreadsheet(gc, sheet_url, retries=retries)
    last_err = None

    for attempt in range(retries):
        try:
            try:
                ws = sh.worksheet(tab)
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title=tab, rows=2000, cols=26)
            _WORKSHEET_CACHE[key] = ws
            return ws

        except APIError as e:
            last_err = e
            if _is_quota_error(e) and attempt < retries - 1:
                _sleep_for_retry(attempt)
                continue
            raise

    raise RuntimeError(f"Failed opening worksheet after retries: {tab}") from last_err

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

def _to_user_value(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        return x
    return str(x)

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.resize(rows=100, cols=8)
        ws.update([["(empty)"]], range_name="A1", value_input_option="USER_ENTERED")
        return
    rows_list = df.map(_to_user_value).values.tolist()
    header = [str(c) for c in df.columns]
    data = [header] + rows_list
    rows, cols = len(data) - 1, len(header)
    ws.resize(rows=max(100, rows + 5), cols=max(min(26, cols + 2), 8))
    start = 0
    r = 1
    while start < len(data):
        end = min(start + ROW_CHUNK, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng, value_input_option="USER_ENTERED")
        r += len(block)
        start = end

# ─────────────────────────────
# TYPE / PARSERS
# ─────────────────────────────
BLACKLIST_TOKENS = {
    "CASH","USD","INTEREST","DIVIDEND","REINVESTMENT","FEE",
    "WITHDRAWAL","DEPOSIT","TRANSFER","SWEEP","PENDING","ACTIVITY",
    "SPAXX**","FCASH**"
}

# Simple crypto detection: ignore BTC/ETH/SOL tickers and common -USD / /USD forms
def is_crypto_ticker(s: str) -> bool:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return False
    u = str(s).strip().upper()
    if not u:
        return False
    # Common explicit forms
    crypto_exact = {
        "BTC-USD","ETH-USD","SOL-USD",
        "BTC/USD","ETH/USD","SOL/USD",
        "BTC","ETH","SOL",
        "BTCUSD","ETHUSD","SOLUSD",
    }
    if u in crypto_exact:
        return True
    # Catch variants like "BTC-USD " etc.
    if "BTC-USD" in u or "ETH-USD" in u or "SOL-USD" in u:
        return True
    if "BTC/USD" in u or "ETH/USD" in u or "SOL/USD" in u:
        return True
    return False

def base_symbol_from_string(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    # Drop obvious crypto upfront
    if is_crypto_ticker(s):
        return ""
    token = s.split()[0]
    token = token.split("-")[0]
    token = token.replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token).upper()
    if not token:
        return ""
    if token in BLACKLIST_TOKENS:
        return ""
    if token.isdigit():
        return ""
    if len(token) > 8 and token.isalnum():
        return ""
    return token

# ---- Option code parsing: -PLTR251114C190
_OPT_RE = re.compile(r"^-([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d+)$")

def parse_option_code(code: str) -> Optional[dict]:
    if not isinstance(code, str):
        return None
    m = _OPT_RE.match(code.strip().upper())
    if not m:
        return None
    und, yy, mm, dd, right, strike = m.groups()
    year = 2000 + int(yy)
    month = int(mm)
    day = int(dd)
    exp = f"{year:04d}-{month:02d}-{day:02d}"
    return {
        "Code": code.strip(),
        "Underlying": und,
        "Expiration": exp,
        "Right": right,
        "Strike": float(strike),
        "IsOption": True,
    }

def occ_like_text_frag(opt: dict) -> str:
    """Text fragment to find the option in Transactions description."""
    y, m, d = opt["Expiration"].split("-")
    return f"{opt['Underlying']} {int(m):02d}/{int(d):02d}/{y} {int(opt['Strike'])} {opt['Right']}"

# ─────────────────────────────
# GOOGLEFINANCE helpers
# ─────────────────────────────
def read_mapping(gc, sheet_url: str) -> Dict[str, Dict[str, str]]:
    try:
        mws = open_ws(gc, sheet_url, "Mapping")
        dfm = read_tab(mws)
        out: Dict[str, Dict[str, str]] = {}
        if not dfm.empty and "Ticker" in dfm.columns:
            for _, row in dfm.iterrows():
                t = str(row.get("Ticker", "")).strip().upper()
                if not t:
                    continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym", "")).strip(),
                    "TickerYF": str(row.get("TickerYF", "")).strip().upper()
                }
        return out
    except Exception:
        return {}

def googlefinance_formula_for(ticker: str, row_idx: int, mapping: Dict[str, Dict[str, str]]) -> str:
    base = base_symbol_from_string(ticker)
    mapped = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    sym = mapped.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_idx},"price"), ""))'

def googlefinance_formula_for_snapshot(tkr: str, row_index: int, mapping: dict) -> str:
    return googlefinance_formula_for(tkr, row_index, mapping)

# ─────────────────────────────
# LOAD SIGNALS (+ options)
# ─────────────────────────────
def load_signals(df_sig: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      equities_df: TimestampUTC, Ticker, Source, Direction, Price, Timeframe
      options_df:  TimestampUTC, Code, Underlying, Expiration, Right, Strike,
                   Source, Direction, Price, Timeframe
    """
    if df_sig.empty:
        empty_e = pd.DataFrame(columns=["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"])
        empty_o = pd.DataFrame(columns=["TimestampUTC","Code","Underlying","Expiration","Right","Strike","Source","Direction","Price","Timeframe"])
        return empty_e, empty_o

    df = df_sig.copy()
    tcol = next((c for c in df.columns if c.lower() in ("ticker","symbol")), None)
    if not tcol:
        raise ValueError("Signals tab needs a 'Ticker' column.")
    tscol = next((c for c in df.columns if c.lower().startswith("timestamp")), None)

    # Normalize shared fields
    df["Source"]    = df.get("Source", "").fillna("").astype(str)
    df["Direction"] = df.get("Direction", "").fillna("").astype(str)
    df["Timeframe"] = df.get("Timeframe", "").fillna("").astype(str)
    df["Price"]     = df.get("Price", "").fillna("").astype(str)
    df["TimestampUTC"] = pd.to_datetime(df[tscol], errors="coerce", utc=True) if tscol else pd.NaT

    # --- Filter signals ---
    # 1) Ignore cryptos
    df["__is_crypto"] = df[tcol].map(is_crypto_ticker)
    # 2) Require BOTH timestamp and source (drop rows without either)
    df = df[
        (~df["__is_crypto"])
        & df["TimestampUTC"].notna()
        & df["Source"].str.strip().ne("")
    ].copy()
    df.drop(columns=["__is_crypto"], inplace=True, errors="ignore")

    if df.empty:
        empty_e = pd.DataFrame(columns=["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"])
        empty_o = pd.DataFrame(columns=["TimestampUTC","Code","Underlying","Expiration","Right","Strike","Source","Direction","Price","Timeframe"])
        return empty_e, empty_o

    # Split → options vs equities
    tickers = df[tcol].astype(str).fillna("")
    is_option_mask = tickers.str.startswith("-")
    df_opt_raw = df.loc[is_option_mask].copy()
    df_eq_raw  = df.loc[~is_option_mask].copy()

    # Equities
    df_eq_raw["Ticker"] = df_eq_raw[tcol].map(base_symbol_from_string)
    df_eq_raw = df_eq_raw[df_eq_raw["Ticker"].ne("")]
    equities = df_eq_raw[["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"]].sort_values(["Ticker","TimestampUTC"], ignore_index=True)

    # Options
    parsed_rows = []
    for _, r in df_opt_raw.iterrows():
        code = str(r[tcol]).strip()
        p = parse_option_code(code)
        if not p:
            continue
        parsed_rows.append({
            "TimestampUTC": r["TimestampUTC"],
            "Code": p["Code"],
            "Underlying": p["Underlying"],
            "Expiration": p["Expiration"],
            "Right": p["Right"],
            "Strike": p["Strike"],
            "Source": r["Source"],
            "Direction": r["Direction"],
            "Price": r["Price"],
            "Timeframe": r["Timeframe"],
        })
    options = pd.DataFrame(parsed_rows)
    if not options.empty:
        options.sort_values(["Underlying","Expiration","Right","Strike","TimestampUTC"], inplace=True, ignore_index=True)

    return equities, options

# ─────────────────────────────
# LOAD TRANSACTIONS (equities FIFO)
# ─────────────────────────────
_TRADE_PATT = r"\b(?:YOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"

def _safe_contains(series: pd.Series, pat: str) -> pd.Series:
    return series.astype(str).str.upper().str.contains(pat, regex=True, na=False)

def _looks_like_trade_mask(action: pd.Series, typ: pd.Series, desc: pd.Series) -> pd.Series:
    action_up = action.fillna("").astype(str)
    typ_up    = typ.fillna("").astype(str)
    desc_up   = desc.fillna("").astype(str)
    return (
        _safe_contains(action_up, _TRADE_PATT)
        | _safe_contains(typ_up, _TRADE_PATT)
        | _safe_contains(desc_up, _TRADE_PATT)
    )

def _classify_type(a: str, t: str, d: str) -> str:
    s = f"{a or ''} {t or ''} {d or ''}".upper()
    if "SOLD" in s or "SELL" in s:
        return "SELL"
    if "BOUGHT" in s or "BUY" in s:
        return "BUY"
    return ""

def _symbol_from_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    m = re.search(r"\(([A-Z][A-Z0-9\.\-]{0,7})\)", text.upper())
    return m.group(1) if m else ""

def load_transactions(df_tx: pd.DataFrame, debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_tx.empty:
        return pd.DataFrame(columns=["When","Type","Symbol","Qty","Price"]), pd.DataFrame()

    # Discover columns
    datecol = next((c for c in df_tx.columns if "run date" in c.lower() or c.lower() == "date"), None)
    actioncol = next((c for c in df_tx.columns if "action" in c.lower()), None)
    typecol   = next((c for c in df_tx.columns if c.lower() == "type"), None)
    desccol   = next((c for c in df_tx.columns if "description" in c.lower()), None)
    symcol    = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol    = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    pricecol  = next((c for c in df_tx.columns if "price" in c.lower()), None)
    amtcol    = next((c for c in df_tx.columns if "amount" in c.lower()), None)

    if not datecol:
        raise ValueError("Transactions: missing Date / Run Date column.")
    if not (actioncol or typecol or desccol):
        raise ValueError("Transactions: need at least one of Action / Type / Description.")
    if not (symcol or actioncol or desccol):
        raise ValueError("Transactions: need Symbol or parsable ticker in Action/Description.")

    df = df_tx.copy()
    action = df.get(actioncol, "")
    typ    = df.get(typecol, "")
    desc   = df.get(desccol, "")

    # --- TEMP FIX: drop explicit equity short-sale / short-cover trades ---
    combo = (
        action.astype(str).str.upper() + " "
        + typ.astype(str).str.upper() + " "
        + desc.astype(str).str.upper()
    )

    is_short_sale  = combo.str.contains("SHORT SALE",  na=False)
    is_short_cover = combo.str.contains("SHORT COVER", na=False)
    short_mask = is_short_sale | is_short_cover

    if short_mask.any() and debug:
        short_rows = df.loc[short_mask].copy()
        cols_to_show = [
            c for c in [datecol, actioncol, typecol, desccol, symcol, qtycol, pricecol, amtcol]
            if c is not None
        ]
        print("• load_transactions: dropping short trades (SHORT SALE / SHORT COVER rows):")
        try:
            print(short_rows[cols_to_show])
        except Exception:
            print(short_rows.head())

    # Remove these rows before we classify BUY/SELL and run FIFO
    df = df.loc[~short_mask].copy()

    # Rebuild views after filtering
    action = df.get(actioncol, "")
    typ    = df.get(typecol, "")
    desc   = df.get(desccol, "")

    mask_tr = _looks_like_trade_mask(action, typ, desc)

    df_tr = df.loc[mask_tr].copy()
    df_tr["When"] = pd.to_datetime(df_tr[datecol], errors="coerce", utc=True)
    df_tr["Type"] = [
        _classify_type(a, t, d) for a, t, d in zip(
            df_tr.get(actioncol, pd.Series(index=df_tr.index)),
            df_tr.get(typecol,   pd.Series(index=df_tr.index)),
            df_tr.get(desccol,   pd.Series(index=df_tr.index)),
        )
    ]
    if symcol:
        sym = df_tr[symcol].map(base_symbol_from_string)
    else:
        sym = pd.Series("", index=df_tr.index)
    sym = np.where(
        (sym == "") & df_tr.get(actioncol, "").astype(str).ne(""),
        df_tr[actioncol].map(_symbol_from_text),
        sym,
    )
    sym = np.where(
        (pd.Series(sym, index=df_tr.index) == "") & df_tr.get(desccol, "").astype(str).ne(""),
        df_tr[desccol].map(_symbol_from_text),
        sym,
    )
    df_tr["Symbol"] = pd.Series(sym, index=df_tr.index).map(base_symbol_from_string)

    def to_float(series: pd.Series) -> pd.Series:
        def conv(x):
            if isinstance(x, str):
                x = x.replace("$","").replace(",","").strip()
            try:
                return float(x)
            except Exception:
                return np.nan
        return series.map(conv)

    qty = to_float(df_tr.get(qtycol, pd.Series(np.nan, index=df_tr.index))).abs()
    price = to_float(df_tr.get(pricecol, pd.Series(np.nan, index=df_tr.index)))

    if qty.isna().all() and amtcol and pricecol:
        amt = to_float(df_tr[amtcol])
        with np.errstate(divide='ignore', invalid='ignore'):
            qty = np.where(
                (price != 0) & (~np.isnan(price)) & (~np.isnan(amt)),
                np.abs(amt) / np.abs(price),
                np.nan,
            )
        qty = pd.Series(qty, index=df_tr.index)

    df_tr["Qty"] = pd.to_numeric(qty, errors="coerce")
    df_tr["Price"] = pd.to_numeric(price, errors="coerce")

    df_tr = df_tr[
        df_tr["When"].notna()
        & df_tr["Type"].isin(["BUY","SELL"])
        & (df_tr["Qty"] > 0)
    ].copy()
    df_tr.sort_values(["When"], inplace=True)
    df_tr.reset_index(drop=True, inplace=True)

    if debug:
        print(f"• load_transactions: after cleaning → {len(df_tr)} trades")

    return df_tr[["When","Type","Symbol","Qty","Price"]], df  # second is raw df_tx (short rows removed)

# ─────────────────────────────
# LIVE PRICE & SNAPSHOT
# ─────────────────────────────
def _fetch_last_prices_yf(tickers: list[str]) -> dict:
    if not tickers or yf is None:
        return {}
    uniq = sorted({t for t in tickers if t})
    out = {}
    try:
        data = yf.download(uniq, period="5d", interval="1d", group_by="column", auto_adjust=True, progress=False)
    except Exception:
        data = None
    if data is None or (hasattr(data, "empty") and data.empty):
        for t in uniq:
            try:
                hist = yf.download(t, period="5d", interval="1d", progress=False, auto_adjust=True)
                if not hist.empty:
                    out[t] = float(hist["Close"].iloc[-1])
            except Exception:
                pass
        return out
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close" in data.columns.get_level_values(0)):
            close = data["Close"]
            for t in uniq:
                try:
                    val = close[t].dropna().iloc[-1]
                    if pd.notna(val):
                        out[t] = float(val)
                except Exception:
                    pass
    else:
        if "Close" in data.columns:
            ser = data["Close"].dropna()
            if not ser.empty:
                t = uniq[0]
                out[t] = float(ser.iloc[-1])
    return out

def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Attach GOOGLEFINANCE() formulas for PriceNow and a long/short-aware Unrealized%:

      LONG  : (PriceNow / EntryPrice - 1) * 100
      SHORT : (EntryPrice / PriceNow - 1) * 100

    Shorts are detected via OpenQty < 0.
    """
    if open_df.empty:
        return open_df

    out = open_df.copy()

    # Ensure we have the columns we expect
    if "OpenQty" not in out.columns or "EntryPrice" not in out.columns or "Ticker" not in out.columns:
        return out

    # Insert blank columns so we know final layout (Ticker, OpenQty, EntryPrice, PriceNow, Unrealized%, ...)
    out.insert(out.columns.get_loc("EntryPrice") + 1, "PriceNow", "")
    out.insert(out.columns.get_loc("PriceNow") + 1, "Unrealized%", "")

    # Column indices for A1 references (after insertion)
    col_openqty = out.columns.get_loc("OpenQty") + 1
    col_entry   = out.columns.get_loc("EntryPrice") + 1

    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        row_index = idx + 2  # row 2 is first data row in sheet

        # GOOGLEFINANCE formula for live price
        formula = googlefinance_formula_for_snapshot(tkr, row_index, mapping)
        out.at[idx, "PriceNow"] = formula

        # Build Unrealized% formula if we have a usable entry price
        try:
            epf = float(ep)
        except Exception:
            epf = float("nan")

        if not (epf > 0):
            # Leave Unrealized% blank if no valid entry
            continue

        oq_cell    = gspread.utils.rowcol_to_a1(row_index, col_openqty)
        entry_cell = gspread.utils.rowcol_to_a1(row_index, col_entry)

        # Strip leading '=' when embedding the GOOGLEFINANCE formula
        inner = formula.lstrip("=").strip()

        # If OpenQty < 0 → short: (EntryPrice / PriceNow - 1)*100
        # else long: (PriceNow / EntryPrice - 1)*100
        unreal_formula = (
            f'=IFERROR('
            f'IF({oq_cell}<0, (({entry_cell})/({inner})-1)*100, (({inner})/({entry_cell})-1)*100)'
            f',"")'
        )
        out.at[idx, "Unrealized%"] = unreal_formula

    return out

# ─────────────────────────────
# FIFO MATCHING (equities)
# ─────────────────────────────
def build_realized_and_open(
    tx: pd.DataFrame,
    sig_equities: pd.DataFrame,
    sell_cutoff: Optional[pd.Timestamp] = None,
    strict_signals: bool = False,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    sig_buy = sig_equities[(sig_equities["Direction"].str.upper() == "BUY") & sig_equities["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True, ignore_index=True)

    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append((r["TimestampUTC"], {
            "Source": r.get("Source",""),
            "Timeframe": r.get("Timeframe",""),
            "SigTime": r.get("TimestampUTC"),
            "SigPrice": r.get("Price",""),
        }))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        if not arr:
            return {"Source":"(unknown)","Timeframe":"","SigTime":pd.NaT,"SigPrice":""}
        if strict_signals:
            for t, payload in reversed(arr):
                if pd.isna(t) or pd.isna(when):
                    return {"Source":"(unknown)","Timeframe":"","SigTime":pd.NaT,"SigPrice":""}
                if t <= when:
                    return payload
            return {"Source":"(unknown)","Timeframe":"","SigTime":pd.NaT,"SigPrice":""}
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                continue
            if t <= when:
                return payload
        return arr[-1][1]

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []
    unmatched_sells = []

    for _, row in tx.iterrows():
        tkr, when, ttype, qty, price = row["Symbol"], row["When"], row["Type"], row["Qty"], row["Price"]
        if qty <= 0 or pd.isna(when) or tkr == "":
            continue

        if ttype == "SELL" and sell_cutoff is not None and when < sell_cutoff:
            if debug:
                print(f"• Ignoring SELL before cutoff: {tkr} at {when.isoformat()} qty={qty}")
            continue

        if ttype == "BUY":
            siginfo = last_signal_for(tkr, when)
            lots[tkr].append({
                "qty_left": float(qty),
                "entry_price": float(price) if not math.isnan(price) else np.nan,
                "entry_time": when,
                "source": siginfo.get("Source",""),
                "timeframe": siginfo.get("Timeframe",""),
                "sig_time": siginfo.get("SigTime"),
                "sig_price": siginfo.get("SigPrice"),
            })
        elif ttype == "SELL":
            remaining = float(qty)
            while remaining > 1e-12 and lots.get(tkr) and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                entry = lot["entry_price"] if not math.isnan(lot["entry_price"]) else np.nan
                exitp = price if not math.isnan(price) else np.nan
                ret_pct = ((exitp - entry) / entry * 100.0) if (entry and not math.isnan(entry)) else np.nan
                held_days = (when - lot["entry_time"]).days if (not pd.isna(lot["entry_time"]) and not pd.isna(when)) else ""
                realized_rows.append({
                    "Ticker": tkr,
                    "Qty": round(take, 6),
                    "EntryPrice": round(entry, 6) if not np.isnan(entry) else "",
                    "ExitPrice": round(exitp, 6) if not np.isnan(exitp) else "",
                    "Return%": round(ret_pct, 4) if not np.isnan(ret_pct) else "",
                    "HoldDays": held_days,
                    "EntryTimeUTC": lot["entry_time"],
                    "ExitTimeUTC": when,
                    "Source": lot["source"] or "(unknown)",
                    "Timeframe": lot["timeframe"],
                    "SignalTimeUTC": lot["sig_time"],
                    "SignalPrice": lot["sig_price"],
                })
                lot["qty_left"] -= take
                remaining -= take
                if lot["qty_left"] <= 1e-9:
                    lots[tkr].popleft()
            if remaining > 1e-9:
                unmatched_sells.append(f"{tkr} SELL on {when.isoformat()} qty={qty} price={price} — No prior BUY lot")

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df.sort_values("ExitTimeUTC", inplace=True, ignore_index=True)

    now_utc = pd.Timestamp.now(tz="UTC")
    open_rows = []
    for tkr, q in lots.items():
        for lot in q:
            if lot["qty_left"] <= 1e-9:
                continue
            open_rows.append({
                "Ticker": tkr,
                "OpenQty": round(lot["qty_left"], 6),
                "EntryPrice": round(lot["entry_price"], 6) if not math.isnan(lot["entry_price"]) else "",
                "EntryTimeUTC": lot["entry_time"],
                "DaysOpen": (now_utc - lot["entry_time"]).days if not pd.isna(lot["entry_time"]) else "",
                "Source": lot["source"] or "(unknown)",
                "Timeframe": lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice": lot["sig_price"],
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df.sort_values("EntryTimeUTC", inplace=True, ignore_index=True)

    return realized_df, open_df, unmatched_sells

# ─────────────────────────────
# OPTIONS: summarize vs Transactions (raw)
# ─────────────────────────────
def to_float_any(x):
    if isinstance(x, str):
        x = x.replace("$","").replace(",","").strip()
    try:
        return float(x)
    except Exception:
        return np.nan

def summarize_options_vs_transactions(options_df: pd.DataFrame, tx_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each option code from Signals, search Transactions text columns for matches,
    compute net qty and net cashflow (P/L). Outputs:
      Code, Underlying, Expiration, Right, Strike, Source, Status,
      PnL$, Buys, Sells, NetQty, FirstTrade, LastTrade

    Status ∈ {WIN, LOSS, OPEN, NO_TRADES}

    We also attach a Source (from Signals). If multiple signals exist for the same
    Code with different Sources, we pick the first non-empty Source.
    """
    base_cols = [
        "Code","Underlying","Expiration","Right","Strike",
        "Source","Status","PnL$","Buys","Sells","NetQty","FirstTrade","LastTrade"
    ]

    if options_df is None or options_df.empty or tx_raw is None or tx_raw.empty:
        return pd.DataFrame(columns=base_cols)

    # Map Code -> Source (first non-empty Source wins)
    src_map: Dict[str, str] = {}
    if "Code" in options_df.columns:
        for _, r in options_df.iterrows():
            code = str(r.get("Code","")).strip()
            if not code:
                continue
            src = str(r.get("Source","")).strip()
            if not src:
                src = "(unknown)"
            # Prefer non-(unknown) sources
            cur = src_map.get(code)
            if cur is None or cur == "(unknown)":
                src_map[code] = src

    # Find likely columns in Transactions
    desc_col = next((c for c in tx_raw.columns if "description" in c.lower()), None)
    action_col = next((c for c in tx_raw.columns if "action" in c.lower()), None)
    type_col   = next((c for c in tx_raw.columns if c.lower() == "type"), None)
    date_col   = next((c for c in tx_raw.columns if "run date" in c.lower() or c.lower() == "date"), None)
    qty_col    = next((c for c in tx_raw.columns if "quantity" in c.lower()), None)
    amt_col    = next((c for c in tx_raw.columns if "amount" in c.lower()), None)

    if not (desc_col and date_col and (action_col or type_col) and qty_col and amt_col):
        # Not enough columns to compute — return stubs
        out = options_df.copy()
        out["Source"] = out["Code"].map(lambda c: src_map.get(str(c).strip(), "(unknown)"))
        out["Status"] = "NO_TRADES"
        out["PnL$"] = 0.0
        out["Buys"] = 0
        out["Sells"] = 0
        out["NetQty"] = 0.0
        out["FirstTrade"] = ""
        out["LastTrade"] = ""
        return out[base_cols]

    # Pre-normalize
    df = tx_raw.copy()
    df["__desc"] = df[desc_col].astype(str)
    df["__type"] = df.get(type_col, "").astype(str).str.upper()
    df["__action"] = df.get(action_col, "").astype(str).str.upper()
    df["__when"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["__qty"] = df[qty_col].map(to_float_any)
    df["__amt"] = df[amt_col].map(to_float_any)  # Amount: negative for buys, positive for sells (typically)

    rows = []
    for _, r in options_df.iterrows():
        code = str(r["Code"])
        src  = src_map.get(code, "(unknown)")
        frag = occ_like_text_frag({
            "Underlying": r["Underlying"],
            "Expiration": r["Expiration"],
            "Right": r["Right"],
            "Strike": r["Strike"]
        })

        mask = df["__desc"].str.upper().str.contains(r["Underlying"].upper(), na=False)
        mask &= df["__desc"].str.contains(r"\b" + re.escape(r["Right"]) + r"\b", flags=re.IGNORECASE, regex=True, na=False)
        # Match date in either 11/21/2025 or 11/21/25 style
        y, m, d = r["Expiration"].split("-")
        md1 = f"{int(m):02d}/{int(d):02d}/{y}"
        md2 = f"{int(m):02d}/{int(d):02d}/{int(y)%100:02d}"
        mask &= (df["__desc"].str.contains(re.escape(md1)) | df["__desc"].str.contains(re.escape(md2)))
        # Strike can appear as "270", "270.00"
        strike_pat = r"\b" + re.escape(str(int(r["Strike"]))) + r"(\.0+)?\b"
        mask &= df["__desc"].str.contains(strike_pat, regex=True, na=False)

        cand = df.loc[mask].copy()
        if cand.empty:
            rows.append({
                "Code": code, "Underlying": r["Underlying"], "Expiration": r["Expiration"],
                "Right": r["Right"], "Strike": r["Strike"], "Source": src,
                "Status": "NO_TRADES",
                "PnL$": 0.0, "Buys": 0, "Sells": 0, "NetQty": 0.0,
                "FirstTrade": "", "LastTrade": ""
            })
            continue

        # Classify buy/sell via type+action fields
        def _is_buy_row(s):
            s = str(s or "")
            return ("BUY" in s) or ("BOUGHT" in s)
        def _is_sell_row(s):
            s = str(s or "")
            return ("SELL" in s) or ("SOLD" in s)

        buy_rows = cand[_is_buy_row(cand["__type"]) | _is_buy_row(cand["__action"])]
        sell_rows= cand[_is_sell_row(cand["__type"]) | _is_sell_row(cand["__action"])]

        buys  = int(len(buy_rows))
        sells = int(len(sell_rows))
        first_ts = pd.to_datetime(cand["__when"]).min()
        last_ts  = pd.to_datetime(cand["__when"]).max()

        # P/L approximation: sum of Amount (broker exports typically: buys negative, sells positive)
        pnl = float(np.nansum(cand["__amt"]))
        net_qty = float(np.nansum(
            cand["__qty"] * np.where(
                _is_buy_row(cand["__type"]) | _is_buy_row(cand["__action"]),
                1, -1
            )
        ))

        if abs(net_qty) > 1e-9:
            status = "OPEN"
        else:
            status = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT")

        rows.append({
            "Code": code, "Underlying": r["Underlying"], "Expiration": r["Expiration"],
            "Right": r["Right"], "Strike": r["Strike"], "Source": src,
            "Status": status,
            "PnL$": round(pnl, 2), "Buys": buys, "Sells": sells, "NetQty": round(net_qty, 4),
            "FirstTrade": first_ts, "LastTrade": last_ts
        })

    out = pd.DataFrame(rows)
    # Sort: closed first (wins first), then open, then no_trades
    status_rank = {"WIN":0,"FLAT":1,"LOSS":2,"OPEN":3,"NO_TRADES":4}
    out["__rank"] = out["Status"].map(lambda s: status_rank.get(s, 9))
    out.sort_values(["__rank","Underlying","Expiration","Right","Strike","Code"], inplace=True, ignore_index=True)
    out.drop(columns="__rank", inplace=True)
    return out[base_cols]

# ─────────────────────────────
# PERFORMANCE TABLES
# ─────────────────────────────
def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    if realized_df.empty:
        realized_grp = pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%"])
    else:
        tmp = realized_df.copy()
        tmp["ret"] = pd.to_numeric(tmp["Return%"], errors="coerce")
        tmp["is_win"] = tmp["ret"] > 0
        g = tmp.groupby("Source", dropna=False)
        realized_grp = pd.DataFrame({
            "Source": g.size().index,
            "Trades": g.size().values,
            "Wins": g["is_win"].sum().values,
            "WinRate%": (g["is_win"].mean().fillna(0.0).values * 100).round(2),
            "AvgReturn%": g["ret"].mean().round(2).values,
            "MedianReturn%": g["ret"].median().round(2).values,
        })

    if open_df.empty:
        open_counts = pd.DataFrame(columns=["Source","OpenLots","OpenTickers"])
    else:
        g2 = open_df.groupby("Source")
        open_counts = pd.DataFrame({
            "Source": g2.size().index,
            "OpenLots": g2.size().values,
            "OpenTickers": g2["Ticker"].nunique().values,
        })

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer")
    if perf.empty:
        perf = pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenLots","OpenTickers"])

    for c in ["Trades","Wins","OpenLots","OpenTickers"]:
        if c in perf.columns:
            perf[c] = pd.to_numeric(perf[c], errors="coerce").fillna(0).astype(int)
    for c in ["WinRate%","AvgReturn%","MedianReturn%"]:
        if c in perf.columns:
            perf[c] = pd.to_numeric(perf[c], errors="coerce").fillna(0.0)

    tot_trades = int(perf["Trades"].sum()) if "Trades" in perf else 0
    tot_wins = int(perf["Wins"].sum()) if "Wins" in perf else 0
    overall_ret_mean = 0.0
    overall_ret_median = 0.0
    if not realized_df.empty:
        r = pd.to_numeric(realized_df["Return%"], errors="coerce")
        overall_ret_mean = float(np.nanmean(r)) if len(r) else 0.0
        overall_ret_median = float(np.nanmedian(r)) if len(r) else 0.0
    tot_open_lots = int(perf["OpenLots"].sum()) if "OpenLots" in perf else 0
    tot_open_tickers = int(perf["OpenTickers"].sum()) if "OpenTickers" in perf else 0
    win_rate_total = round((tot_wins / tot_trades) * 100, 2) if tot_trades else 0.0

    totals_row = pd.DataFrame([{
        "Source":"(TOTALS)",
        "Trades": tot_trades,
        "Wins": tot_wins,
        "WinRate%": win_rate_total,
        "AvgReturn%": round(overall_ret_mean, 2),
        "MedianReturn%": round(overall_ret_median, 2),
        "OpenLots": tot_open_lots,
        "OpenTickers": tot_open_tickers,
    }])

    perf = pd.concat([totals_row, perf.sort_values(["Source"]).reset_index(drop=True)], ignore_index=True)
    return perf

def build_perf_by_month(realized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly performance (equities only, across all sources).

    Columns:
      Month (YYYY-MM), Trades, Wins, WinRate%, AvgReturn%, MedianReturn%, TotalPnL$
    """
    if realized_df is None or realized_df.empty:
        return pd.DataFrame(columns=[
            "Month","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","TotalPnL$"
        ])

    df = realized_df.copy()
    df["ExitTimeUTC"] = pd.to_datetime(df["ExitTimeUTC"], errors="coerce", utc=True)
    df["Month"] = df["ExitTimeUTC"].dt.to_period("M").astype(str)

    df["ret"] = pd.to_numeric(df["Return%"], errors="coerce")
    df["EntryPrice_num"] = pd.to_numeric(df["EntryPrice"], errors="coerce")
    df["ExitPrice_num"] = pd.to_numeric(df["ExitPrice"], errors="coerce")
    df["Qty_num"] = pd.to_numeric(df["Qty"], errors="coerce")
    df["PnL$"] = (df["ExitPrice_num"] - df["EntryPrice_num"]) * df["Qty_num"]
    df["is_win"] = df["ret"] > 0

    g = df.groupby("Month", dropna=True)

    out = pd.DataFrame({
        "Month": g.size().index,
        "Trades": g.size().values,
        "Wins": g["is_win"].sum().values,
        "WinRate%": (g["is_win"].mean().fillna(0.0).values * 100).round(2),
        "AvgReturn%": g["ret"].mean().round(2).values,
        "MedianReturn%": g["ret"].median().round(2).values,
        "TotalPnL$": g["PnL$"].sum().round(2).values,
    })

    out = out.sort_values("Month").reset_index(drop=True)

    # Overall totals row
    total_trades = int(out["Trades"].sum())
    total_wins   = int(out["Wins"].sum())
    winrate_all  = round((total_wins / total_trades) * 100, 2) if total_trades else 0.0
    overall_avg_ret = float(np.nanmean(df["ret"])) if len(df["ret"]) else 0.0
    overall_med_ret = float(np.nanmedian(df["ret"])) if len(df["ret"]) else 0.0
    total_pnl = float(np.nansum(df["PnL$"]))

    totals_row = pd.DataFrame([{
        "Month": "(ALL)",
        "Trades": total_trades,
        "Wins": total_wins,
        "WinRate%": winrate_all,
        "AvgReturn%": round(overall_avg_ret, 2),
        "MedianReturn%": round(overall_med_ret, 2),
        "TotalPnL$": round(total_pnl, 2),
    }])

    out = pd.concat([totals_row, out], ignore_index=True)
    return out

def build_perf_by_month_source_kind(
    realized_df: pd.DataFrame,
    options_results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Monthly breakdown by:
      - Month (YYYY-MM)
      - Kind: EQUITY or OPTION
      - Source

    Columns:
      Month, Kind, Source, Trades, Wins, WinRate%, TotalPnL$

    EQUITY side:
      - Uses realized_df (Realized_Trades).
      - PnL$ = Qty * (ExitPrice - EntryPrice)

    OPTION side:
      - Uses options_results_df (Options_Results).
      - Treat each row as one trade.
      - Wins = Status == 'WIN'
      - PnL$ = PnL$ column
      - Month = month of LastTrade (or FirstTrade fallback)
    """
    rows = []

    # ---- EQUITY side ----
    if realized_df is not None and not realized_df.empty:
        df = realized_df.copy()
        df["ExitTimeUTC"] = pd.to_datetime(df["ExitTimeUTC"], errors="coerce", utc=True)
        df["Month"] = df["ExitTimeUTC"].dt.to_period("M").astype(str)

        df["Source"] = df["Source"].fillna("").astype(str)
        df["Source"] = df["Source"].replace("", "(unknown)")

        df["EntryPrice_num"] = pd.to_numeric(df["EntryPrice"], errors="coerce")
        df["ExitPrice_num"]  = pd.to_numeric(df["ExitPrice"], errors="coerce")
        df["Qty_num"]        = pd.to_numeric(df["Qty"], errors="coerce")
        df["PnL$"] = (df["ExitPrice_num"] - df["EntryPrice_num"]) * df["Qty_num"]
        df["ret"] = pd.to_numeric(df["Return%"], errors="coerce")
        df["is_win"] = df["ret"] > 0

        g_eq = df.groupby(["Month","Source"], dropna=True)
        eq_part = pd.DataFrame({
            "Month": g_eq.size().index.get_level_values(0),
            "Kind": "EQUITY",
            "Source": g_eq.size().index.get_level_values(1),
            "Trades": g_eq.size().values,
            "Wins": g_eq["is_win"].sum().values,
            "WinRate%": (g_eq["is_win"].mean().fillna(0.0).values * 100).round(2),
            "TotalPnL$": g_eq["PnL$"].sum().round(2).values,
        })
        rows.append(eq_part)

    # ---- OPTION side ----
    if options_results_df is not None and not options_results_df.empty:
        opt = options_results_df.copy()
        opt["Source"] = opt["Source"].fillna("").astype(str)
        opt["Source"] = opt["Source"].replace("", "(unknown)")

        # Trade timestamp
        opt["LastTrade"]  = pd.to_datetime(opt["LastTrade"], errors="coerce", utc=True)
        opt["FirstTrade"] = pd.to_datetime(opt["FirstTrade"], errors="coerce", utc=True)
        # Month: prefer LastTrade, fallback FirstTrade
        ts = opt["LastTrade"].where(opt["LastTrade"].notna(), opt["FirstTrade"])
        opt["Month"] = ts.dt.to_period("M").astype(str)

        opt["PnL$"] = pd.to_numeric(opt["PnL$"], errors="coerce").fillna(0.0)
        opt["Status"] = opt["Status"].fillna("").astype(str).str.upper()
        opt["is_win"] = opt["Status"] == "WIN"

        g_opt = opt.groupby(["Month","Source"], dropna=True)
        opt_part = pd.DataFrame({
            "Month": g_opt.size().index.get_level_values(0),
            "Kind": "OPTION",
            "Source": g_opt.size().index.get_level_values(1),
            "Trades": g_opt.size().values,
            "Wins": g_opt["is_win"].sum().values,
            "WinRate%": (g_opt["is_win"].mean().fillna(0.0).values * 100).round(2),
            "TotalPnL$": g_opt["PnL$"].sum().round(2).values,
        })
        rows.append(opt_part)

    if not rows:
        return pd.DataFrame(columns=["Month","Kind","Source","Trades","Wins","WinRate%","TotalPnL$"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["Month","Kind","Source"]).reset_index(drop=True)

    # Overall totals across all months/kinds/sources
    total_trades = int(out["Trades"].sum())
    total_wins   = int(out["Wins"].sum())
    winrate_all  = round((total_wins / total_trades) * 100, 2) if total_trades else 0.0
    total_pnl    = float(np.nansum(out["TotalPnL$"]))

    totals_row = pd.DataFrame([{
        "Month": "(ALL)",
        "Kind": "(ALL)",
        "Source": "(ALL)",
        "Trades": total_trades,
        "Wins": total_wins,
        "WinRate%": winrate_all,
        "TotalPnL$": round(total_pnl, 2),
    }])

    out = pd.concat([totals_row, out], ignore_index=True)
    return out

# ─────────────────────────────
# SNAPSHOT / WEEKLY + SHORTS FROM HOLDINGS
# ─────────────────────────────
def build_snapshot_from_open_detail(open_detail_df: pd.DataFrame, mapping: dict | None = None, *, price_source: str = "sheets") -> pd.DataFrame:
    """
    Build Open_Positions_Snapshot.

    For price_source == "yfinance":
      - Fetch Last Price via yfinance
      - Compute long/short-aware PnL:
          LONG  : PnL$ = (Current - Cost), % = PnL$ / Cost
          SHORT : PnL$ = (Cost - Current), % = PnL$ / Cost
        where Cost = |Qty| * EntryPrice, Current = |Qty| * LastPrice.
    """
    if open_detail_df is None or open_detail_df.empty:
        return pd.DataFrame(columns=[
            "Symbol","Description","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent"
        ])

    df = open_detail_df.copy()
    df["Symbol"] = df["Ticker"].astype(str)
    df["Quantity"] = pd.to_numeric(df.get("OpenQty", np.nan), errors="coerce")
    df["Average Cost Basis"] = pd.to_numeric(df.get("EntryPrice", np.nan), errors="coerce")
    df["Description"] = ""

    if price_source == "sheets":
        # Let the sheet compute prices via GOOGLEFINANCE (no PnL here)
        last_price = []
        for i, r in df.iterrows():
            row_index = i + 2
            tkr = str(r["Symbol"])
            formula = googlefinance_formula_for_snapshot(tkr, row_index, mapping or {})
            last_price.append(formula)
        df["Last Price"] = last_price
        df["Current Value"] = ""
        df["Cost Basis Total"] = ""
        df["Total Gain/Loss Dollar"] = ""
        df["Total Gain/Loss Percent"] = ""
    else:
        # Use yfinance and compute PnL with side-awareness
        prices = _fetch_last_prices_yf(df["Symbol"].dropna().unique().tolist())
        df["Last Price"] = df["Symbol"].map(prices).astype(float)

        abs_qty = df["Quantity"].abs()
        df["Cost Basis Total"] = (abs_qty * df["Average Cost Basis"]).astype(float)
        df["Current Value"] = (abs_qty * df["Last Price"]).astype(float)

        is_short = df["Quantity"] < 0
        df["Total Gain/Loss Dollar"] = np.where(
            is_short,
            df["Cost Basis Total"] - df["Current Value"],   # SHORT: profit when price falls
            df["Current Value"] - df["Cost Basis Total"],   # LONG : profit when price rises
        ).astype(float)

        df["Total Gain/Loss Percent"] = np.where(
            df["Cost Basis Total"].fillna(0) > 0,
            df["Total Gain/Loss Dollar"] / df["Cost Basis Total"] * 100.0,
            np.nan
        )

    cols = ["Symbol","Description","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    snap = df[cols].copy()
    return snap

def write_weekly_report_tab(ws_weekly, snapshot_tab_name: str = TAB_SNAPSHOT):
    metrics = pd.DataFrame([
        ["Total Gain/Loss ($)", f"=IFERROR(SUM('{snapshot_tab_name}'!H2:H),0)"],
        ["Portfolio % Gain",    f"=IFERROR(SUM('{snapshot_tab_name}'!H2:H)/SUM('{snapshot_tab_name}'!F2:F),0)"],
        ["Average % Gain",      f"=IFERROR(AVERAGE(IF('{snapshot_tab_name}'!I2:I<>\"\",'{snapshot_tab_name}'!I2:I,)),0)"],
    ], columns=["Metric","Value"])

    ws_weekly.clear()
    ws_weekly.resize(rows=max(10, len(metrics) + 5), cols=2)
    header = ["Metric","Value"]
    data = [header] + metrics.values.tolist()
    ws_weekly.update(data, range_name=f"A1:B{len(data)}", value_input_option="USER_ENTERED")

def append_shorts_from_holdings_to_open(open_df: pd.DataFrame, holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append open short positions from Holdings into open_df.

    We detect shorts as rows where Quantity < 0 in the Holdings tab.
    We try to use:
      - Symbol/Ticker/Security column for Ticker
      - Quantity/Shares/Qty column for position size
      - Average Cost / Cost Basis per share as EntryPrice (if present)

    These synthetic rows are marked:
      Source = "(short)"
      Timeframe = ""
      SignalTimeUTC, SignalPrice left blank
      EntryTimeUTC / DaysOpen left blank (unknown)
    """
    if holdings_df is None or holdings_df.empty:
        return open_df

    df = holdings_df.copy()

    symcol = next(
        (c for c in df.columns
         if c.lower() in ("symbol", "ticker", "security", "symbol/cusip")),
        None
    )
    qtycol = next(
        (c for c in df.columns
         if "quantity" in c.lower()
         or "shares" in c.lower()
         or c.lower().startswith("qty")),
        None
    )

    if not symcol or not qtycol:
        return open_df

    df["_qty"] = pd.to_numeric(df[qtycol], errors="coerce")
    shorts = df[df["_qty"] < 0].copy()
    if shorts.empty:
        return open_df

    avgcol = next(
        (c for c in df.columns
         if "average cost" in c.lower()
         or "avg cost" in c.lower()
         or "cost basis per share" in c.lower()),
        None
    )

    rows = []
    for _, r in shorts.iterrows():
        tkr = base_symbol_from_string(r[symcol])
        if not tkr:
            continue
        qty = float(r["_qty"])
        entry_price = np.nan
        if avgcol:
            entry_price = to_float_any(r[avgcol])

        rows.append({
            "Ticker": tkr,
            "OpenQty": qty,  # keep negative to reflect short
            "EntryPrice": round(entry_price, 6) if not np.isnan(entry_price) else "",
            "EntryTimeUTC": pd.NaT,
            "DaysOpen": "",
            "Source": "(short)",
            "Timeframe": "",
            "SignalTimeUTC": pd.NaT,
            "SignalPrice": "",
        })

    if not rows:
        return open_df

    shorts_df = pd.DataFrame(rows)

    combined = open_df.copy()
    # Ensure all columns in shorts_df exist in combined
    for c in shorts_df.columns:
        if c not in combined.columns:
            combined[c] = "" if c not in ("OpenQty","EntryPrice","EntryTimeUTC","SignalTimeUTC") else np.nan

    # Reindex shorts_df to combined columns so concat is aligned
    shorts_df_reindexed = shorts_df.reindex(columns=combined.columns, fill_value="")
    combined = pd.concat([combined, shorts_df_reindexed], ignore_index=True)

    if "EntryTimeUTC" in combined.columns:
        try:
            combined.sort_values(["Ticker","EntryTimeUTC"], inplace=True, ignore_index=True)
        except Exception:
            combined.sort_values(["Ticker"], inplace=True, ignore_index=True)
    else:
        combined.sort_values(["Ticker"], inplace=True, ignore_index=True)

    return combined

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build performance dashboard tabs.")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--strict-signals", action="store_true")
    ap.add_argument("--sell-cutoff", type=str, default=None)
    ap.add_argument("--price-source", choices=["sheets","yfinance"], default="sheets")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    DEBUG = args.debug

    cfg = load_cfg(args.config)
    sheet_url = resolve_sheet_url(cfg)
    service_account_file = resolve_service_account_file(cfg)

    tab_signals  = resolve_tab_name(cfg, "signals_tab", TAB_SIGNALS)
    tab_openpos  = resolve_tab_name(cfg, "open_positions_tab", TAB_OPEN)
    tab_perf     = TAB_PERF
    tab_month    = TAB_MONTHLY
    tab_month_src= TAB_MONTHLY_SRC
    tab_real     = TAB_REALIZED
    tab_open_det = TAB_OPEN_DETAIL
    tab_tx       = TAB_TRANSACTIONS
    tab_hold     = TAB_HOLDINGS

    price_source = args.price_source

    print("📊 Building performance dashboard…")
    if not sheet_url:
        print("SHEET_URL not found in YAML. Set sheets.url OR sheets.sheet_url.")
        return

    print(f"• Google Sheet: {sheet_url}")
    gc = auth_gspread(service_account_file)

    # Read tabs
    ws_sig = open_ws(gc, sheet_url, tab_signals)
    ws_tx  = open_ws(gc, sheet_url, tab_tx)
    ws_h   = open_ws(gc, sheet_url, tab_hold)

    df_sig = read_tab(ws_sig)
    df_tx  = read_tab(ws_tx)
    df_h   = read_tab(ws_h)

    print(f"• Loaded: Signals={len(df_sig)} rows, Transactions={len(df_tx)} rows, Holdings={len(df_h)} rows")

    # Normalize signals → equities + options
    sig_equities, sig_options = load_signals(df_sig)

    # Transactions (equities FIFO) + keep raw for options scan
    tx, tx_raw = load_transactions(df_tx, debug=DEBUG)

    # Optional SELL cutoff
    sell_cutoff_ts: Optional[pd.Timestamp] = None
    if args.sell_cutoff:
        try:
            sell_cutoff_ts = pd.to_datetime(args.sell_cutoff, utc=True)
        except Exception:
            print(f"⚠️ Could not parse --sell-cutoff='{args.sell_cutoff}'. Ignoring.")

    # Realized/Open (equities)
    realized_df, open_df, unmatched_sells = build_realized_and_open(
        tx, sig_equities, sell_cutoff=sell_cutoff_ts, strict_signals=args.strict_signals, debug=DEBUG
    )

    # Add open shorts from Holdings (negative quantity rows) into Open_Positions
    open_df = append_shorts_from_holdings_to_open(open_df, df_h)

    # Live price formulas to Open_Positions (equities + shorts)
    if (not args.no_live) and (not open_df.empty) and (price_source == "sheets"):
        mapping = read_mapping(gc, sheet_url)
        open_df = add_live_price_formulas(open_df, mapping)

    # Column order
    if not realized_df.empty:
        realized_df = realized_df[[
            "Ticker","Qty","EntryPrice","ExitPrice","Return%","HoldDays",
            "EntryTimeUTC","ExitTimeUTC","Source","Timeframe","SignalTimeUTC","SignalPrice"
        ]]
    if not open_df.empty:
        cols = ["Ticker","OpenQty","EntryPrice","EntryTimeUTC","DaysOpen","Source","Timeframe","SignalTimeUTC","SignalPrice"]
        if "PriceNow" in open_df.columns and "Unrealized%" in open_df.columns:
            cols = ["Ticker","OpenQty","EntryPrice","PriceNow","Unrealized%","EntryTimeUTC","DaysOpen","Source","Timeframe","SignalTimeUTC","SignalPrice"]
        open_df = open_df[cols]

    # Performance & OpenLots_Detail (equities + shorts)
    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy())
    monthly_df = build_perf_by_month(realized_df.copy())
    open_detail_df = pd.DataFrame()
    if not open_df.empty:
        detail_cols = [c for c in ["Source","Ticker","OpenQty","EntryPrice","EntryTimeUTC","DaysOpen","Timeframe","SignalTimeUTC","SignalPrice","PriceNow","Unrealized%"] if c in open_df.columns]
        open_detail_df = open_df[detail_cols].sort_values(["Source","Ticker","EntryTimeUTC"], ignore_index=True)

    # Options results (scan against Transactions raw)
    options_results_df = summarize_options_vs_transactions(sig_options, tx_raw)

    # Monthly-by-kind-by-source (equity vs options)
    monthly_src_df = build_perf_by_month_source_kind(realized_df.copy(), options_results_df.copy())

    # --- Real-trades YTD CSV for real_vs_sim comparison ---
    try:
        if not realized_df.empty:
            daily = realized_df.copy()
            daily["ExitTimeUTC"] = pd.to_datetime(daily["ExitTimeUTC"], errors="coerce", utc=True)
            daily["EntryPrice_num"] = pd.to_numeric(daily["EntryPrice"], errors="coerce")
            daily["ExitPrice_num"]  = pd.to_numeric(daily["ExitPrice"], errors="coerce")
            daily["Qty_num"]        = pd.to_numeric(daily["Qty"], errors="coerce")
            daily["PnL$"] = (daily["ExitPrice_num"] - daily["EntryPrice_num"]) * daily["Qty_num"]

            # Map to calendar date
            daily["Date"] = daily["ExitTimeUTC"].dt.date
            daily = daily.dropna(subset=["Date"])

            # Focus on current year; fallback to all if no rows match
            current_year = pd.Timestamp.now(tz="UTC").year
            daily_year = daily[daily["ExitTimeUTC"].dt.year == current_year].copy()
            if daily_year.empty:
                daily_year = daily.copy()

            grouped = (
                daily_year
                .groupby("Date", as_index=False)["PnL$"]
                .sum()
                .rename(columns={"PnL$": "Realized_PnL"})
            )

            data_dir = ((cfg.get("reporting") or {}).get("data_dir") or "./data")
            os.makedirs(data_dir, exist_ok=True)
            ytd_name = f"weinstein_real_trades_{current_year}YTD.csv"
            real_csv_path = os.path.join(data_dir, ytd_name)
            grouped.to_csv(real_csv_path, index=False)
            print(f"📝 Wrote real-vs-sim helper CSV: {real_csv_path}")
    except Exception as e:
        print(f"⚠️ Could not generate real-trades YTD CSV: {type(e).__name__}: {e}")

    # Write tabs
    ws_real        = open_ws(gc, sheet_url, tab_real)
    ws_open        = open_ws(gc, sheet_url, tab_openpos)
    ws_perf        = open_ws(gc, sheet_url, tab_perf)
    ws_month       = open_ws(gc, sheet_url, tab_month)
    ws_month_src   = open_ws(gc, sheet_url, tab_month_src)
    ws_open_detail = open_ws(gc, sheet_url, tab_open_det)
    ws_opts        = open_ws(gc, sheet_url, TAB_OPTIONS)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_df)
    write_tab(ws_month, monthly_df)
    write_tab(ws_month_src, monthly_src_df)
    write_tab(ws_open_detail, open_detail_df)
    write_tab(ws_opts, options_results_df)

    print(f"✅ Wrote {tab_real}: {len(realized_df)} rows")
    print(f"✅ Wrote {tab_openpos}: {len(open_df)} rows")
    print(f"✅ Wrote {tab_perf}: {len(perf_df)} rows")
    print(f"✅ Wrote {tab_month}: {len(monthly_df)} rows")
    print(f"✅ Wrote {tab_month_src}: {len(monthly_src_df)} rows")
    print(f"✅ Wrote {tab_open_det}: {len(open_detail_df)} rows")
    print(f"✅ Wrote {TAB_OPTIONS}: {len(options_results_df)} rows")

    # Snapshot + Weekly + CSVs
    try:
        mapping = read_mapping(gc, sheet_url)
        snapshot_df = build_snapshot_from_open_detail(open_detail_df, mapping=mapping, price_source=price_source)
        ws_snapshot = open_ws(gc, sheet_url, TAB_SNAPSHOT)
        write_tab(ws_snapshot, snapshot_df)

        ws_weekly = open_ws(gc, sheet_url, TAB_WEEKLY)
        write_weekly_report_tab(ws_weekly, snapshot_tab_name=TAB_SNAPSHOT)

        out_dir = ((cfg.get("reporting") or {}).get("output_dir")
                   or (cfg.get("sheets") or {}).get("output_dir")
                   or "./output")
        os.makedirs(out_dir, exist_ok=True)

        # Open positions CSV (equities + shorts)
        csv_path = os.path.join(out_dir, "Open_Positions.csv")
        csv_df = snapshot_df.copy()

        # Ensure numeric for PnL calc
        csv_df["Quantity"] = pd.to_numeric(csv_df.get("Quantity", np.nan), errors="coerce")
        csv_df["Average Cost Basis"] = pd.to_numeric(csv_df.get("Average Cost Basis", np.nan), errors="coerce")

        if price_source == "sheets":
            # Re-fetch prices for CSV if snapshot was formula-based
            prices = _fetch_last_prices_yf(csv_df["Symbol"].dropna().unique().tolist())
            csv_df["Last Price"] = csv_df["Symbol"].map(prices).astype(float)
        else:
            csv_df["Last Price"] = pd.to_numeric(csv_df.get("Last Price", np.nan), errors="coerce")

        abs_qty = csv_df["Quantity"].abs()
        csv_df["Cost Basis Total"] = (abs_qty * csv_df["Average Cost Basis"]).astype(float)
        csv_df["Current Value"] = (abs_qty * csv_df["Last Price"]).astype(float)

        is_short = csv_df["Quantity"] < 0
        csv_df["Total Gain/Loss Dollar"] = np.where(
            is_short,
            csv_df["Cost Basis Total"] - csv_df["Current Value"],   # SHORT
            csv_df["Current Value"] - csv_df["Cost Basis Total"],   # LONG
        ).astype(float)

        csv_df["Total Gain/Loss Percent"] = np.where(
            csv_df["Cost Basis Total"].fillna(0) > 0,
            csv_df["Total Gain/Loss Dollar"] / csv_df["Cost Basis Total"] * 100.0,
            np.nan
        )

        csv_df.to_csv(csv_path, index=False)
        print(f"📝 Wrote local CSV for email: {csv_path}")

        # Options results CSV
        opt_csv_path = os.path.join(out_dir, "Options_Results.csv")
        options_results_df.to_csv(opt_csv_path, index=False)
        print(f"📝 Wrote local CSV for email: {opt_csv_path}")

    except Exception as e:
        print(f"⚠️ Could not refresh snapshot/weekly/options CSVs: {type(e).__name__}: {e}")

    if unmatched_sells:
        print(f"⚠️ Summary: {len(unmatched_sells)} unmatched SELL events (use --debug to print details).")

    print("🎯 Done.")

if __name__ == "__main__":
    main()
