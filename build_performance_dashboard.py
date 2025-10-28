#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Google Sheets dashboard tabs:
- Realized_Trades
- Open_Positions
- Performance_By_Source
- OpenLots_Detail
- Open_Positions_Snapshot   (NEW)
- Weekly_Report             (NEW)

Reads tabs:
  - Signals
  - Transactions
  - Holdings (optional; only context)

Features
- FIFO matching of BUY lots vs SELLs to compute realized PnL.
- Source/Timeframe is taken from the most recent Signal at/before a BUY.
  * If --strict-signals is OFF (default), falls back to the most recent
    signal for that ticker (any time) when no prior signal exists.
- Optional live price formulas for Open_Positions (disable via --no-live).
- Optional --sell-cutoff YYYY-MM-DD to ignore very old SELLs.
- Prints a summary of unmatched SELLs and any tickers with Source "(unknown)".
- Adds totals row at top of Performance_By_Source.
- Adds OpenLots_Detail tab with one row per open lot.

This version loads configuration from config.yaml:
  sheets.url or sheets.sheet_url
  google.service_account_json
  sheets.open_positions_tab, sheets.signals_tab (optional)
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import yaml

# ─────────────────────────────
# DEFAULTS (can be overridden by YAML or CLI)
# ─────────────────────────────
DEFAULT_SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Tab names (can be overridden in YAML via sheets.* if you want)
TAB_SIGNALS      = "Signals"
TAB_TRANSACTIONS = "Transactions"
TAB_HOLDINGS     = "Holdings"           # optional
TAB_REALIZED     = "Realized_Trades"
TAB_OPEN         = "Open_Positions"
TAB_PERF         = "Performance_By_Source"
TAB_OPEN_DETAIL  = "OpenLots_Detail"

# NEW tabs
TAB_SNAPSHOT     = "Open_Positions_Snapshot"
TAB_WEEKLY       = "Weekly_Report"

DEFAULT_EXCHANGE_PREFIX = "NYSE: "
ROW_CHUNK = 500

# ─────────────────────────────
# CONFIG LOADING
# ─────────────────────────────
def load_cfg(path: str) -> dict:
    """Load YAML; tolerate missing file by returning {}."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def resolve_sheet_url(cfg: dict) -> Optional[str]:
    sheets = cfg.get("sheets", {}) or {}
    # support both keys
    return sheets.get("url") or sheets.get("sheet_url") or os.getenv("SHEET_URL")

def resolve_service_account_file(cfg: dict) -> str:
    google = cfg.get("google", {}) or {}
    # env var GOOGLE_APPLICATION_CREDENTIALS also allowed
    return (
        google.get("service_account_json")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or DEFAULT_SERVICE_ACCOUNT_FILE
    )

def resolve_tab_name(cfg: dict, key: str, default_name: str) -> str:
    """Allow optional overrides of tab names in YAML under sheets.*"""
    sheets = cfg.get("sheets", {}) or {}
    return sheets.get(key, default_name)

# ─────────────────────────────
# UTILITIES
# ─────────────────────────────
def auth_gspread(service_account_file: str):
    print("🔑 Authorizing service account…")
    creds = Credentials.from_service_account_file(service_account_file, scopes=DEFAULT_SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, sheet_url: str, tab: str):
    sh = gc.open_by_url(sheet_url)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=2000, cols=26)

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

def write_tab(ws, df: pd.DataFrame):
    """
    Safe writer that always provides VALUES first, then range_name, to match the
    current gspread signature (and avoid the "Invalid value at 'data.values'" error).
    """
    ws.clear()
    if df.empty:
        ws.resize(rows=100, cols=8)
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows + 5), cols=max(min(26, cols + 2), 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

    # Chunked upload (values first, then range_name)
    start = 0
    r = 1
    while start < len(data):
        end = min(start + ROW_CHUNK, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)
        r += len(block)
        start = end

def to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def to_float(series: pd.Series) -> pd.Series:
    def conv(x):
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
        try:
            return float(x)
        except Exception:
            return np.nan
    return series.map(conv)

BLACKLIST_TOKENS = {
    "CASH", "USD", "INTEREST", "DIVIDEND", "REINVESTMENT", "FEE",
    "WITHDRAWAL", "DEPOSIT", "TRANSFER", "SWEEP", "PENDING", "ACTIVITY",
    "SPAXX**", "FCASH**"
}

def base_symbol_from_string(s) -> str:
    """Extract a base equity/ETF symbol; return '' for cash/blank/other."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    token = s.split()[0]                 # first space-delimited chunk
    token = token.split("-")[0]          # strip lot/option suffix like AAPL-12345
    token = token.replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token).upper()
    if not token:
        return ""
    if token in BLACKLIST_TOKENS:
        return ""
    if token.isdigit():
        return ""
    if len(token) > 8 and token.isalnum():  # likely account-like
        return ""
    return token

def read_mapping(gc, sheet_url: str) -> Dict[str, Dict[str, str]]:
    """Return {ticker: {'FormulaSym': 'EXCH: TKR', 'TickerYF': 'TKR'}} if Mapping tab exists."""
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

def googlefinance_symbol_for(ticker: str, mapping: Dict[str, Dict[str, str]]) -> str:
    """
    Resolve the GOOGLEFINANCE symbol string for a ticker, preferring Mapping.FormulaSym,
    else `${DEFAULT_EXCHANGE_PREFIX}${base_symbol}`.
    """
    base = base_symbol_from_string(ticker)
    mapped = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    return mapped.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)

def googlefinance_formula_for_cell(ticker: str, row_idx: int, mapping: Dict[str, Dict[str, str]]) -> str:
    """
    Build a robust GOOGLEFINANCE(price) formula:
      1) Try mapped/static exchange symbol
      2) Fallback to the value in column A (Symbol) of this row
    """
    sym = googlefinance_symbol_for(ticker, mapping)
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(A{row_idx},"price"), ""))'

# ─────────────────────────────
# LOAD SIGNALS
# ─────────────────────────────
def load_signals(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig.empty:
        return pd.DataFrame(columns=["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"])
    df = df_sig.copy()

    tcol = next((c for c in df.columns if c.lower() in ("ticker", "symbol")), None)
    if not tcol:
        raise ValueError("Signals tab needs a 'Ticker' column.")

    tscol = next((c for c in df.columns if c.lower().startswith("timestamp")), None)

    df["Ticker"] = df[tcol].map(base_symbol_from_string)
    df["Source"] = df.get("Source", "").fillna("").astype(str)
    df["Direction"] = df.get("Direction", "").fillna("").astype(str)
    df["Timeframe"] = df.get("Timeframe", "").fillna("").astype(str)
    df["TimestampUTC"] = to_dt(df[tscol]) if tscol else pd.NaT
    df["Price"] = df.get("Price", "").fillna("").astype(str)

    df = df[df["Ticker"].ne("")]
    df.sort_values(["Ticker", "TimestampUTC"], inplace=True, ignore_index=True)
    return df[["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"]]

# ─────────────────────────────
# LOAD TRANSACTIONS
# ─────────────────────────────
_TRADE_PATT = r"\b(?:YOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"

def _looks_like_trade_mask(action: pd.Series, typ: pd.Series, desc: pd.Series) -> pd.Series:
    action_up = action.fillna("").astype(str).str.upper()
    typ_up    = typ.fillna("").astype(str).str.upper()
    desc_up   = desc.fillna("").astype(str).str.upper()
    return (
        action_up.str.contains(_TRADE_PATT, regex=True, na=False)
        | typ_up.str.contains(_TRADE_PATT, regex=True, na=False)
        | desc_up.str.contains(_TRADE_PATT, regex=True, na=False)
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
        return pd.DataFrame(columns=["When", "Type", "Symbol", "Qty", "Price"]), pd.DataFrame()

    # Column discovery
    datecol = next((c for c in df_tx.columns if "run date" in c.lower() or c.lower() == "date"), None)
    actioncol = next((c for c in df_tx.columns if "action" in c.lower()), None)
    typecol   = next((c for c in df_tx.columns if c.lower() == "type"), None)
    desccol   = next((c for c in df_tx.columns if "description" in c.lower()), None)
    symcol    = next((c for c in df_tx.columns if c.lower() in ("symbol", "security", "symbol/cusip")), None)
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

    # Determine trade-like rows (N_all)
    action = df.get(actioncol, "")
    typ    = df.get(typecol, "")
    desc   = df.get(desccol, "")
    mask_tr = _looks_like_trade_mask(action, typ, desc)
    n_all = len(df)
    n_tr  = int(mask_tr.sum())
    print(f"• load_transactions: detected {n_tr} trade-like rows (of {n_all})")

    # Work only on trade-like rows (N_tr)
    df_tr = df.loc[mask_tr].copy()

    # When
    df_tr["When"] = to_dt(df_tr[datecol])

    # Type
    df_tr["Type"] = [
        _classify_type(a, t, d) for a, t, d in zip(
            df_tr.get(actioncol, pd.Series(index=df_tr.index)),
            df_tr.get(typecol,   pd.Series(index=df_tr.index)),
            df_tr.get(desccol,   pd.Series(index=df_tr.index)),
        )
    ]

    # Symbol
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

    # Quantity & Price
    qty = to_float(df_tr.get(qtycol, pd.Series(np.nan, index=df_tr.index))).abs()
    price = to_float(df_tr.get(pricecol, pd.Series(np.nan, index=df_tr.index)))

    # If no qty but have Amount & Price → reconstruct qty
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

    # Keep only valid BUY/SELL rows
    df_tr = df_tr[
        df_tr["Symbol"].ne("")
        & df_tr["When"].notna()
        & df_tr["Type"].isin(["BUY", "SELL"])
        & (df_tr["Qty"] > 0)
    ].copy()

    df_tr.sort_values(["When"], inplace=True)
    df_tr.reset_index(drop=True, inplace=True)

    if debug:
        print(f"• load_transactions: after cleaning → {len(df_tr)} trades")
        preview_cols = [c for c in ["Run Date","Account","Account Number","Action","Symbol","Description","Type","Quantity","Price ($)","Settlement Date","When","Qty","Price"] if c in df_tr.columns]
        print(df_tr[preview_cols].head(8).to_string(index=False))

    return df_tr[["When", "Type", "Symbol", "Qty", "Price"]], pd.DataFrame()

# ─────────────────────────────
# LIVE PRICE FORMULAS (Open_Positions tab)
# ─────────────────────────────
def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now = []
    unreal = []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        row_index = idx + 2  # header row is 1
        # NOTE: This formula assumes the "Ticker" is in column B on that tab.
        # It is fine for our Open_Positions layout.
        sym = googlefinance_symbol_for(tkr, mapping)
        formula = f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_index},"price"), ""))'
        price_now.append(formula)
        try:
            epf = float(ep)
            unreal.append(f'=IFERROR(( {formula} / {epf} - 1 ) * 100,"")' if epf > 0 else "")
        except Exception:
            unreal.append("")
    out.insert(out.columns.get_loc("EntryPrice") + 1, "PriceNow", price_now)
    out.insert(out.columns.get_loc("PriceNow") + 1, "Unrealized%", unreal)
    return out

# ─────────────────────────────
# FIFO MATCHING
# ─────────────────────────────
def build_realized_and_open(
    tx: pd.DataFrame,
    sig: pd.DataFrame,
    sell_cutoff: Optional[pd.Timestamp] = None,
    strict_signals: bool = False,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Index signals by (Ticker, time), only BUY signals
    sig_buy = sig[(sig["Direction"].str.upper() == "BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker", "TimestampUTC"], inplace=True, ignore_index=True)

    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append((r["TimestampUTC"], {
            "Source": r.get("Source", ""),
            "Timeframe": r.get("Timeframe", ""),
            "SigTime": r.get("TimestampUTC"),
            "SigPrice": r.get("Price", ""),
        }))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        if not arr:
            return {"Source": "(unknown)", "Timeframe": "", "SigTime": pd.NaT, "SigPrice": ""}
        if strict_signals:
            for t, payload in reversed(arr):
                if pd.isna(t) or pd.isna(when):
                    return {"Source": "(unknown)", "Timeframe": "", "SigTime": pd.NaT, "SigPrice": ""}
                if t <= when:
                    return payload
            return {"Source": "(unknown)", "Timeframe": "", "SigTime": pd.NaT, "SigPrice": ""}
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
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = row["Qty"] if not math.isnan(row["Qty"]) else 0.0
        price = row["Price"] if not math.isnan(row["Price"]) else np.nan
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
                "source": siginfo.get("Source", ""),
                "timeframe": siginfo.get("Timeframe", ""),
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
                msg = f"{tkr} SELL on {when.isoformat()} qty={qty} price={price} — No prior BUY lot in window"
                unmatched_sells.append(msg)
                if debug:
                    print(f"⚠️ Unmatched SELL: {msg}")

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df.sort_values("ExitTimeUTC", inplace=True, ignore_index=True)

    # Remaining open lots
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
# PERFORMANCE TABLE
# ─────────────────────────────
def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    if realized_df.empty:
        realized_grp = pd.DataFrame(columns=["Source", "Trades", "Wins", "WinRate%", "AvgReturn%", "MedianReturn%"])
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
        open_counts = pd.DataFrame(columns=["Source", "OpenLots", "OpenTickers"])
    else:
        g2 = open_df.groupby("Source")
        open_counts = pd.DataFrame({
            "Source": g2.size().index,
            "OpenLots": g2.size().values,
            "OpenTickers": g2["Ticker"].nunique().values,
        })

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer")
    if perf.empty:
        perf = pd.DataFrame(columns=[
            "Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenLots","OpenTickers"
        ])

    for c in ["Trades", "Wins", "OpenLots", "OpenTickers"]:
        if c in perf.columns:
            perf[c] = pd.to_numeric(perf[c], errors="coerce").fillna(0).astype(int)
    for c in ["WinRate%", "AvgReturn%", "MedianReturn%"]:
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
        "Source": "(TOTALS)",
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

# ─────────────────────────────
# REPORT HELPERS
# ─────────────────────────────
def print_unknown_sources(realized_df: pd.DataFrame, open_df: pd.DataFrame):
    unk_real = []
    if not realized_df.empty:
        unk_real = sorted(set(realized_df.loc[realized_df["Source"].eq("(unknown)"), "Ticker"]))
    unk_open = []
    if not open_df.empty:
        unk_open = sorted(set(open_df.loc[open_df["Source"].eq("(unknown)"), "Ticker"]))

    if unk_real or unk_open:
        print("🔎 Unknown Source tickers:")
        if unk_real:
            print("  • Realized:", ", ".join(unk_real))
        if unk_open:
            print("  • Open    :", ", ".join(unk_open))

def print_open_breakdown(open_df: pd.DataFrame):
    if open_df.empty:
        return
    print("🔍 Open breakdown:")
    g = open_df.groupby(["Source", "Ticker"]).size().rename("Lots").reset_index()
    for _, r in g.sort_values(["Source", "Ticker"]).iterrows():
        print(f"  - {r['Source']}: {r['Ticker']} → {int(r['Lots'])} lot(s)")

# ─────────────────────────────
# NEW: SNAPSHOT & WEEKLY BUILDERS
# ─────────────────────────────
def build_snapshot_from_open_detail(open_detail_df: pd.DataFrame, mapping: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Aggregate OpenLots_Detail to a one-row-per-ticker snapshot and attach
    GOOGLEFINANCE formulas for live price + derived P/L columns.
    Returns a DataFrame with evaluated strings (formulas kept as strings).
    """
    if open_detail_df.empty:
        return pd.DataFrame(columns=[
            "Symbol","Description","Quantity","Entry Price",
            "Last Price","Current Value","Cost Basis Total","Average Cost Basis",
            "Total Gain/Loss Dollar","Total Gain/Loss Percent"
        ])

    df = open_detail_df.copy()
    df["OpenQty"] = pd.to_numeric(df.get("OpenQty", np.nan), errors="coerce")
    df["EntryPrice"] = pd.to_numeric(df.get("EntryPrice", np.nan), errors="coerce")

    grp = df.groupby("Ticker", dropna=False)
    rows = []
    for symbol, g in grp:
        if not symbol or pd.isna(symbol):
            continue
        qty = float(pd.to_numeric(g["OpenQty"], errors="coerce").fillna(0).sum())
        # Weighted avg entry price
        ep = g[["OpenQty", "EntryPrice"]].dropna()
        if not ep.empty and float(ep["OpenQty"].sum()) > 0:
            wavg_entry = float((ep["OpenQty"] * ep["EntryPrice"]).sum()) / float(ep["OpenQty"].sum())
        else:
            wavg_entry = np.nan

        rows.append({"Symbol": symbol, "Quantity": qty, "Entry Price": wavg_entry})

    snap = pd.DataFrame(rows)
    if snap.empty:
        snap = pd.DataFrame(columns=["Symbol","Quantity","Entry Price"])

    # Round some numerics for readability
    if "Quantity" in snap.columns:
        snap["Quantity"] = snap["Quantity"].apply(lambda x: round(x, 6) if pd.notna(x) else "")
    if "Entry Price" in snap.columns:
        snap["Entry Price"] = snap["Entry Price"].apply(lambda x: round(x, 6) if pd.notna(x) else "")

    # Insert description placeholder (optional mapping could fill later)
    snap.insert(1, "Description", "")

    # Add formula columns that depend on row index
    last_price = []
    cur_val = []
    cost_total = []
    avg_cost = []
    gl_dollar = []
    gl_percent = []

    for i in range(len(snap)):
        row_index = i + 2  # header is row 1
        sym = str(snap.at[i, "Symbol"])
        price_formula = googlefinance_formula_for_cell(sym, row_index, mapping)
        last_price.append(price_formula)
        # E = Last Price, C = Quantity, D = Entry Price (after insertion positions)
        # Columns: A Symbol, B Description, C Quantity, D Entry Price, E Last Price, F Current Value, G Cost Basis Total, H Average Cost Basis, I Total Gain/Loss Dollar, J Total Gain/Loss Percent
        cur_val.append(f"=IFERROR(C{row_index}*E{row_index},)")
        cost_total.append(f"=IFERROR(C{row_index}*D{row_index},)")
        avg_cost.append(f"=D{row_index}")
        gl_dollar.append(f"=IFERROR(F{row_index}-G{row_index},)")
        gl_percent.append(f"=IFERROR(IF(G{row_index}>0,(F{row_index}/G{row_index}-1)*100,),)")

    snap["Last Price"] = last_price
    snap["Current Value"] = cur_val
    snap["Cost Basis Total"] = cost_total
    snap["Average Cost Basis"] = avg_cost
    snap["Total Gain/Loss Dollar"] = gl_dollar
    snap["Total Gain/Loss Percent"] = gl_percent

    cols = [
        "Symbol","Description","Quantity","Entry Price","Last Price","Current Value",
        "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent"
    ]
    return snap[cols]

def build_weekly_df(snapshot_tab_name: str = TAB_SNAPSHOT) -> pd.DataFrame:
    """
    Build the 2-column Weekly_Report with formulas that reference the snapshot tab.
    We keep simple, robust formulas so Sheets computes everything (even with live prices).
    """
    metrics = ["Total Gain/Loss ($)", "Portfolio % Gain", "Average % Gain"]
    # I = Total Gain/Loss Dollar, G = Cost Basis Total, J = Total Gain/Loss Percent (in snapshot)
    values = [
        f"=IFERROR(SUM('{snapshot_tab_name}'!I2:I),0)",
        f"=IFERROR(SUM('{snapshot_tab_name}'!I2:I)/SUM('{snapshot_tab_name}'!G2:G),0)",
        f"=IFERROR(AVERAGE(IF('{snapshot_tab_name}'!J2:J<>\"\",'{snapshot_tab_name}'!J2:J,)),0)"
    ]
    # For the AVERAGE with blank handling, Sheets needs array formula behavior—this works as-is.

    df = pd.DataFrame({"Metric": metrics, "Value": values})
    return df

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build performance dashboard tabs.")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--no-live", action="store_true", help="Do NOT add GOOGLEFINANCE formulas to Open_Positions.")
    ap.add_argument("--strict-signals", action="store_true", help="Only use signals at/before BUY; no fallback.")
    ap.add_argument("--sell-cutoff", type=str, default=None, help="Ignore SELLs before this date (YYYY-MM-DD).")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    args = ap.parse_args()
    DEBUG = args.debug

    cfg = load_cfg(args.config)
    sheet_url = resolve_sheet_url(cfg)
    service_account_file = resolve_service_account_file(cfg)

    # Allow tab overrides from YAML (optional)
    tab_signals = resolve_tab_name(cfg, "signals_tab", TAB_SIGNALS)
    tab_openpos = resolve_tab_name(cfg, "open_positions_tab", TAB_OPEN)
    tab_perf    = TAB_PERF
    tab_real    = TAB_REALIZED
    tab_open_det= TAB_OPEN_DETAIL
    tab_tx      = TAB_TRANSACTIONS
    tab_hold    = TAB_HOLDINGS

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

    # Normalize signals
    sig = load_signals(df_sig)

    # Transactions
    tx, _ = load_transactions(df_tx, debug=DEBUG)

    # Cutoff SELLs option
    sell_cutoff_ts: Optional[pd.Timestamp] = None
    if args.sell_cutoff:
        try:
            sell_cutoff_ts = pd.to_datetime(args.sell_cutoff, utc=True)
        except Exception:
            print(f"⚠️ Could not parse --sell-cutoff='{args.sell_cutoff}'. Ignoring.")

    # Realized & Open via FIFO
    realized_df, open_df, unmatched_sells = build_realized_and_open(
        tx, sig, sell_cutoff=sell_cutoff_ts, strict_signals=args.strict_signals, debug=DEBUG
    )

    # Optionally add live price formulas to Open_Positions
    mapping = read_mapping(gc, sheet_url)
    if not args.no_live and not open_df.empty:
        open_df = add_live_price_formulas(open_df, mapping)

    # Pretty column order
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

    # Performance & OpenLots_Detail
    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy())
    open_detail_df = pd.DataFrame()
    if not open_df.empty:
        open_detail_df = open_df.copy()
        detail_cols = [c for c in ["Source","Ticker","OpenQty","EntryPrice","EntryTimeUTC","DaysOpen","Timeframe","SignalTimeUTC","SignalPrice","PriceNow","Unrealized%"] if c in open_detail_df.columns]
        open_detail_df = open_detail_df[detail_cols].sort_values(["Source","Ticker","EntryTimeUTC"], ignore_index=True)

    # Write core tabs
    ws_real = open_ws(gc, sheet_url, tab_real)
    ws_open = open_ws(gc, sheet_url, tab_openpos)
    ws_perf = open_ws(gc, sheet_url, tab_perf)
    ws_open_detail = open_ws(gc, sheet_url, tab_open_det)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_df)
    write_tab(ws_open_detail, open_detail_df)

    print(f"✅ Wrote {tab_real}: {len(realized_df)} rows")
    print(f"✅ Wrote {tab_openpos}: {len(open_df)} rows")
    print(f"✅ Wrote {tab_perf}: {len(perf_df)} rows")
    print(f"✅ Wrote {tab_open_det}: {len(open_detail_df)} rows")

    # NEW: build Snapshot + Weekly from OpenLots_Detail
    try:
        snap_ws = open_ws(gc, sheet_url, TAB_SNAPSHOT)
        weekly_ws = open_ws(gc, sheet_url, TAB_WEEKLY)

        snapshot_df = build_snapshot_from_open_detail(open_detail_df, mapping)
        write_tab(snap_ws, snapshot_df)

        weekly_df = build_weekly_df(TAB_SNAPSHOT)
        write_tab(weekly_ws, weekly_df)
    except Exception as e:
        print(f"⚠️ Could not refresh snapshot/weekly tabs: {type(e).__name__}: {e}")

    if unmatched_sells:
        print(f"⚠️ Summary: {len(unmatched_sells)} unmatched SELL events (use --debug to print details).")
        if DEBUG:
            for line in unmatched_sells:
                print("  •", line)

    print_unknown_sources(realized_df, open_df)
    print_open_breakdown(open_df)
    print("🎯 Done.")

if __name__ == "__main__":
    main()
