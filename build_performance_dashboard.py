#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds three tabs in your Google Sheet:
- Realized_Trades
- Open_Positions
- Performance_By_Source

Key features:
- Reads "Signals", "Transactions", "Holdings"
- Robust parsing of Fidelity "History" CSV -> BUY/SELL trades
- FIFO matching of sells to buys
- Source attribution priority:
    1) Most recent Signal at/preceding trade time (by Ticker)
    2) Keyword rules from transaction Context (Account/Action/Type/Description)
    3) Account rules (e.g., '401k')
    4) Ticker default mapping (e.g., QQQM/ARKK/AAPL/VOO/CARR -> ChatGPT)
    5) "(unknown)"
- Reconciles Open positions with current "Holdings" to catch positions whose
  original BUY is outside the export window.
- Counts OpenSignals by UNIQUE TICKERS per source (not lots)
- Optional GOOGLEFINANCE() formula for live prices (disable with --no-live)
- Prints which tickers are still (unknown) so you can add one Signal row per ticker.

CLI:
    python3 build_performance_dashboard.py [--no-live] [--debug]

Requires:
    pip install pandas numpy gspread google-auth
"""

from __future__ import annotations

import math
import re
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS = "Signals"
TAB_TRANSACTIONS = "Transactions"
TAB_HOLDINGS = "Holdings"

TAB_REALIZED = "Realized_Trades"
TAB_OPEN = "Open_Positions"
TAB_PERF = "Performance_By_Source"

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "

# Keyword → Source (case-insensitive) used in fallback source inference
SOURCE_KEYWORDS = {
    "SARK": "Sarkee Capital",
    "SARKEE": "Sarkee Capital",
    "SUPERIORSTAR": "SuperiorStar",
    "ATHENA": "SuperiorStar",
    "WEINSTEIN": "Weinstein",
    "BO XU": "Bo Xu",
    "BOXU": "Bo Xu",
    "BO  XU": "Bo Xu",  # extra spaces just in case
}

# Ticker defaults → Source (used last before "(unknown)")
TICKER_DEFAULT_SOURCE = {
    # Your “ChatGPT set”
    "QQQM": "ChatGPT",
    "CARR": "ChatGPT",
    "ARKK": "ChatGPT",
    "AAPL": "ChatGPT",
    "VOO":  "ChatGPT",
    # Add any others here if you want default attribution
    # "IFBD": "Bo Xu",  # optional; Signals should already cover this
}

# ──────────────────────────────────────────────────────────────────────────────
# GSHEETS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def auth_gspread():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, tab):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=2000, cols=26)

def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # strip strings
    return df

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows + 5), cols=max(min(26, cols + 2), 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(object).where(pd.notna(df), "").astype(str).values.tolist()
    CHUNK = 500
    start = 0
    r = 1
    while start < len(data):
        end = min(start + CHUNK, len(data))
        block = data[start:end]
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        ncols = len(header)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)
        r += len(block)
        start = end

# ──────────────────────────────────────────────────────────────────────────────
# GENERAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

BLACKLIST_TOKENS = {
    "CASH", "USD", "INTEREST", "DIVIDEND", "REINVESTMENT", "FEE",
    "WITHDRAWAL", "DEPOSIT", "TRANSFER", "SWEEP"
}

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

def base_symbol_from_string(s) -> str:
    """Extract likely equity/ETF symbol, else ''."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    token = s.split()[0]
    token = token.split("-")[0]
    token = token.replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token).upper()
    if not token or token in BLACKLIST_TOKENS or token.isdigit():
        return ""
    if len(token) > 8 and token.isalnum():
        return ""
    return token

def read_mapping(gc) -> Dict[str, Dict[str, str]]:
    """Optional Mapping tab: {Ticker: {'FormulaSym': 'EXCH: TKR', 'TickerYF': 'TKR'}}"""
    out: Dict[str, Dict[str, str]] = {}
    try:
        mws = open_ws(gc, "Mapping")
        dfm = read_tab(mws)
        if not dfm.empty and "Ticker" in dfm.columns:
            for _, row in dfm.iterrows():
                t = str(row.get("Ticker", "")).strip().upper()
                if not t:
                    continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym", "")).strip(),
                    "TickerYF": str(row.get("TickerYF", "")).strip().upper()
                }
    except Exception:
        pass
    return out

def googlefinance_formula_for(ticker: str, row_idx: int, mapping: Dict[str, Dict[str, str]]) -> str:
    base = base_symbol_from_string(ticker)
    mapped = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    sym = mapped.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_idx},"price"), ""))'

# ──────────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def load_signals(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig.empty:
        return pd.DataFrame(columns=["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"])
    df = df_sig.copy()
    tcol = next((c for c in df.columns if c.lower() in ("ticker", "symbol")), None)
    if not tcol:
        raise ValueError("Signals tab needs a 'Ticker' column.")
    df["Ticker"] = df[tcol].map(base_symbol_from_string)
    df["Source"] = df.get("Source", "")
    df["Direction"] = df.get("Direction", "")
    df["Timeframe"] = df.get("Timeframe", "")
    tscol = next((c for c in df.columns if c.lower().startswith("timestamp")), None)
    df["TimestampUTC"] = to_dt(df[tscol]) if tscol else pd.NaT
    df["Price"] = df.get("Price", "")
    out = df[["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"]].copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper()
    out["Source"] = out["Source"].astype(str).str.strip()
    out.sort_values(["Ticker", "TimestampUTC"], inplace=True, ignore_index=True)
    return out

def _pick_col(df: pd.DataFrame, options: List[str]) -> str | None:
    for c in df.columns:
        if c is None:
            continue
        if c.strip().lower() in options:
            return c
    return None

def _looks_like_trade_mask(df: pd.DataFrame) -> pd.Series:
    action = df.get(_pick_col(df, ["action"]), pd.Series([""] * len(df)))
    typ = df.get(_pick_col(df, ["type"]), pd.Series([""] * len(df)))
    desc = df.get(_pick_col(df, ["description"]), pd.Series([""] * len(df)))

    action_up = action.astype(str).str.upper()
    typ_up = typ.astype(str).str.upper()
    desc_up = desc.astype(str).str.upper()
    patt = r"\b(YOU\s+)?(BOUGHT|SOLD|BUY|SELL)\b"

    mask_action = action_up.str.contains(patt, regex=True, na=False)
    mask_type = typ_up.str.contains(patt, regex=True, na=False)
    mask_desc = desc_up.str.contains(patt, regex=True, na=False)
    mask = (mask_action | mask_type | mask_desc)
    return mask

def load_transactions(df_tx: pd.DataFrame, debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (tx_df, unmatched_sells_df)
    tx_df columns: When, Type (BUY/SELL), Symbol, Qty(>0), Price, Account, Context
    """
    if df_tx.empty:
        return pd.DataFrame(columns=["When", "Type", "Symbol", "Qty", "Price", "Account", "Context"]), pd.DataFrame()

    df = df_tx.copy()

    # Slice to trade-like rows FIRST to avoid index-length mismatches
    trade_mask = _looks_like_trade_mask(df)
    if debug:
        print(f"• load_transactions: detected {int(trade_mask.sum())} trade-like rows (of {len(df)})")
    df_tr = df.loc[trade_mask].copy()

    # Pick columns on the sliced frame
    datecol = _pick_col(df_tr, ["run date"])
    symcol = _pick_col(df_tr, ["symbol", "security", "symbol/cusip"])
    typecol = _pick_col(df_tr, ["type"])
    actioncol = _pick_col(df_tr, ["action"])
    desccol = _pick_col(df_tr, ["description"])
    pricecol = next((c for c in df_tr.columns if "price" in c.lower()), None)
    qtycol = next((c for c in df_tr.columns if "quantity" in c.lower()), None)
    amtcol = next((c for c in df_tr.columns if "amount" in c.lower()), None)
    acctcol = next((c for c in df_tr.columns if c.strip().lower() == "account"), None)

    if not datecol:
        raise ValueError("Transactions need a 'Run Date' column (exact header varies).")

    when = to_dt(df_tr[datecol])
    action = df_tr[actioncol].astype(str) if actioncol else pd.Series([""] * len(df_tr), index=df_tr.index)
    typ = df_tr[typecol].astype(str) if typecol else pd.Series([""] * len(df_tr), index=df_tr.index)
    desc = df_tr[desccol].astype(str) if desccol else pd.Series([""] * len(df_tr), index=df_tr.index)
    account = df_tr[acctcol].astype(str).str.strip() if acctcol else pd.Series([""] * len(df_tr), index=df_tr.index)

    def classify(a, t, d) -> str:
        s = f"{a} {t} {d}".upper()
        if "SOLD" in s or re.search(r"\bSELL\b", s):
            return "SELL"
        if "BOUGHT" in s or re.search(r"\bBUY\b", s):
            return "BUY"
        return ""

    df_tr["Type"] = [classify(a, t, d) for a, t, d in zip(action, typ, desc)]

    # Symbol extraction
    raw_sym = df_tr[symcol] if symcol else pd.Series([""] * len(df_tr), index=df_tr.index)

    def symbol_from_action(a, d) -> str:
        s = f"{a} {d}"
        m = re.search(r"\(([A-Za-z0-9\.\-]{1,10})\)", s)
        if m:
            return base_symbol_from_string(m.group(1))
        return base_symbol_from_string(s)

    sym = raw_sym.fillna("").map(base_symbol_from_string)
    sym = np.where(sym.astype(str).eq(""),
                   [symbol_from_action(a, d) for a, d in zip(action, desc)],
                   sym)
    sym = pd.Series(sym, index=df_tr.index).astype(str).str.upper()

    # Qty: prefer explicit Quantity; otherwise derive from Amount/Price
    if qtycol:
        qty = to_float(df_tr[qtycol])
    else:
        if amtcol and pricecol:
            amt = to_float(df_tr[amtcol])
            prc = to_float(df_tr[pricecol])
            with np.errstate(divide='ignore', invalid='ignore'):
                q = np.where((prc != 0) & (~np.isnan(prc)) & (~np.isnan(amt)),
                             np.abs(amt) / np.abs(prc),
                             np.nan)
            qty = pd.Series(q, index=df_tr.index)
        else:
            qty = pd.Series(np.nan, index=df_tr.index)

    price = to_float(df_tr[pricecol]) if pricecol else pd.Series(np.nan, index=df_tr.index)

    # Normalize quantities to positive, keep Type BUY/SELL
    qty = pd.to_numeric(qty, errors="coerce").abs()

    # Build a context string for keyword matching (source inference fallback)
    context = (account.fillna("").astype(str) + " | " +
               action.fillna("").astype(str) + " | " +
               typ.fillna("").astype(str) + " | " +
               desc.fillna("").astype(str))

    tx = pd.DataFrame({
        "When": when,
        "Type": df_tr["Type"],
        "Symbol": sym,
        "Qty": qty,
        "Price": price,
        "Account": account,
        "Context": context,
    }, index=df_tr.index)

    # Valid rows
    tx = tx[tx["When"].notna() & tx["Symbol"].ne("") & tx["Type"].isin(["BUY", "SELL"])]
    tx = tx[tx["Qty"] > 0]
    tx.sort_values("When", inplace=True)
    tx.reset_index(drop=True, inplace=True)

    # Unmatched SELL diagnostics
    unmatched_rows = []
    lots: Dict[str, float] = defaultdict(float)
    for _, r in tx.iterrows():
        if r["Type"] == "BUY":
            lots[r["Symbol"]] += float(r["Qty"])
        else:
            need = float(r["Qty"])
            if lots[r["Symbol"]] <= 0:
                unmatched_rows.append({
                    "Symbol": r["Symbol"],
                    "When": r["When"],
                    "Qty": r["Qty"],
                    "Price": r["Price"],
                    "Reason": "No prior BUY lot in window"
                })
            lots[r["Symbol"]] -= need
            if lots[r["Symbol"]] < -1e-9:
                unmatched_rows.append({
                    "Symbol": r["Symbol"],
                    "When": r["When"],
                    "Qty": r["Qty"],
                    "Price": r["Price"],
                    "Reason": "SELL exceeds available BUY lots"
                })
    unmatched_df = pd.DataFrame(unmatched_rows)
    return tx, unmatched_df

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty:
        return pd.DataFrame()
    df = df_h.copy()
    symcol = _pick_col(df, ["symbol", "security", "symbol/cusip", "ticker"])
    qtycol = next((c for c in df.columns if "quantity" in c.lower() or "shares" in c.lower()), None)
    pricecol = next((c for c in df.columns if "price" in c.lower()), None)

    out = pd.DataFrame()
    if symcol:
        out["Ticker"] = df[symcol].map(base_symbol_from_string)
    if qtycol:
        out["Qty"] = to_float(df[qtycol])
    if pricecol:
        out["Price"] = to_float(df[pricecol])
    out["Ticker"] = out["Ticker"].astype(str).str.upper()
    out = out[out["Ticker"].ne("")]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE INFERENCE (fallbacks when no signal match)
# ──────────────────────────────────────────────────────────────────────────────

def infer_source_fallback(ticker: str, account: str, context: str) -> str:
    """
    Infer a Source when the signal lookup is empty.
    Order:
      1) Account rule → '401k' if '401' appears
      2) Keyword rules in context (Account/Action/Type/Description)
      3) Ticker default mapping
      4) '(unknown)'
    """
    tkr = (ticker or "").upper().strip()
    acc = (account or "").upper()
    ctx = (context or "").upper()

    # 1) Account rule for 401(k)
    if "401" in acc:
        return "401k"

    # 2) Keyword rules
    for key, src in SOURCE_KEYWORDS.items():
        if key in ctx:
            return src

    # 3) Ticker defaults
    if tkr in TICKER_DEFAULT_SOURCE:
        return TICKER_DEFAULT_SOURCE[tkr]

    # 4) Unknown
    return "(unknown)"

# ──────────────────────────────────────────────────────────────────────────────
# MATCHING / REALIZED & OPEN
# ──────────────────────────────────────────────────────────────────────────────

def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Index signals by (Ticker, time), only BUYs (if Direction provided)
    sig_buy = sig[(sig["Ticker"].ne("")) & ((sig["Direction"].astype(str).str.upper() == "BUY") | (sig["Direction"] == ""))].copy()
    sig_buy.sort_values(["Ticker", "TimestampUTC"], inplace=True)

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
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                return payload
            if t <= when:
                return payload
        return {"Source": "", "Timeframe": "", "SigTime": pd.NaT, "SigPrice": ""}

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []

    for _, row in tx.iterrows():
        tkr = row["Symbol"]
        when = row["When"]
        ttype = row["Type"]
        qty = float(row["Qty"]) if not math.isnan(row["Qty"]) else 0.0
        price = float(row["Price"]) if not math.isnan(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr == "":
            continue

        if ttype == "BUY":
            siginfo = last_signal_for(tkr, when)
            src = siginfo.get("Source", "")
            if not src:
                # Fallbacks: Account/keywords/ticker defaults
                src = infer_source_fallback(tkr, row.get("Account", ""), row.get("Context", ""))
            lots[tkr].append({
                "qty_left": qty,
                "entry_price": price,
                "entry_time": when,
                "source": src,
                "timeframe": siginfo.get("Timeframe", ""),
                "sig_time": siginfo.get("SigTime"),
                "sig_price": siginfo.get("SigPrice"),
            })
        else:  # SELL
            remaining = qty
            while remaining > 1e-12 and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                if take <= 0:
                    break
                entry = lot["entry_price"] if not math.isnan(lot["entry_price"]) else 0.0
                exitp = price if not math.isnan(price) else 0.0
                ret_pct = ((exitp - entry) / entry * 100.0) if entry else np.nan
                held_days = (when - lot["entry_time"]).days if (not pd.isna(lot["entry_time"]) and not pd.isna(when)) else ""
                realized_rows.append({
                    "Ticker": tkr,
                    "Qty": round(take, 6),
                    "EntryPrice": round(entry, 6) if entry else "",
                    "ExitPrice": round(exitp, 6) if exitp else "",
                    "Return%": round(ret_pct, 4) if not np.isnan(ret_pct) else "",
                    "HoldDays": held_days,
                    "EntryTimeUTC": lot["entry_time"],
                    "ExitTimeUTC": when,
                    "Source": lot["source"],
                    "Timeframe": lot["timeframe"],
                    "SignalTimeUTC": lot["sig_time"],
                    "SignalPrice": lot["sig_price"],
                })
                lot["qty_left"] -= take
                remaining -= take
                if lot["qty_left"] <= 1e-9:
                    lots[tkr].popleft()

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df.sort_values("ExitTimeUTC", inplace=True, ignore_index=True)

    # Remaining open lots
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
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
                "Source": lot["source"],
                "Timeframe": lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice": lot["sig_price"],
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df.sort_values("EntryTimeUTC", inplace=True, ignore_index=True)
    return realized_df, open_df

# ──────────────────────────────────────────────────────────────────────────────
# ENRICH & PERF
# ──────────────────────────────────────────────────────────────────────────────

def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str, Dict[str, str]], enable_live: bool) -> pd.DataFrame:
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now = []
    unreal = []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep = r.get("EntryPrice")
        row_index = idx + 2  # header row is 1
        if enable_live:
            formula = googlefinance_formula_for(tkr, row_index, mapping)
        else:
            formula = ""
        price_now.append(formula)
        try:
            epf = float(ep)
            unreal.append(f'=IFERROR(( {formula} / {epf} - 1 ) * 100,"")' if (enable_live and epf > 0) else "")
        except Exception:
            unreal.append("")
    insert_at = out.columns.get_loc("EntryPrice") + 1 if "EntryPrice" in out.columns else len(out.columns)
    out.insert(insert_at, "PriceNow", price_now)
    out.insert(insert_at + 1, "Unrealized%", unreal)
    return out

def reconcile_open_with_holdings(open_df: pd.DataFrame, holdings_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure any ticker that appears as a current holding also appears in open_df."""
    if holdings_df.empty:
        return open_df

    out = open_df.copy()
    dfh = holdings_df.copy()
    tcol = _pick_col(dfh, ["symbol", "security", "symbol/cusip", "ticker"])
    qcol = next((c for c in dfh.columns if "quantity" in c.lower() or "shares" in c.lower()), None)
    if not tcol:
        return out

    dfh["Ticker"] = dfh[tcol].astype(str).str.upper().str.extract(r"([A-Z0-9\.]+)")[0]
    if qcol:
        dfh["Qty"] = pd.to_numeric(dfh[qcol], errors="coerce")
    else:
        dfh["Qty"] = np.nan

    holding_tickers = set(dfh.loc[dfh["Ticker"].notna(), "Ticker"].tolist())
    open_tickers = set(out["Ticker"].astype(str).str.upper().tolist()) if not out.empty else set()
    missing = sorted(holding_tickers - open_tickers)

    if not missing:
        return out

    # Latest signal per ticker
    latest = {}
    if not signals_df.empty:
        s = signals_df.copy()
        s["Ticker"] = s["Ticker"].astype(str).str.upper()
        tscol = "TimestampUTC" if "TimestampUTC" in s.columns else next((c for c in s.columns if c.lower().startswith("timestamp")), None)
        if tscol:
            s["_ts"] = pd.to_datetime(s[tscol], errors="coerce", utc=True)
        else:
            s["_ts"] = pd.NaT
        s = s.sort_values(["Ticker", "_ts"])
        latest = s.groupby("Ticker").last()[["Source", "Timeframe"]].to_dict("index")

    add_rows = []
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    for t in missing:
        row_h = dfh[dfh["Ticker"] == t].tail(1).iloc[0]
        qty = row_h.get("Qty", "")
        info = latest.get(t, {"Source": "(unknown)", "Timeframe": ""})
        add_rows.append({
            "Ticker": t,
            "OpenQty": "" if pd.isna(qty) else round(float(qty), 6),
            "EntryPrice": "",
            "PriceNow": "",
            "Unrealized%": "",
            "EntryTimeUTC": pd.NaT,
            "DaysOpen": "",
            "Source": info.get("Source", "(unknown)") or "(unknown)",
            "Timeframe": info.get("Timeframe", ""),
            "SignalTimeUTC": pd.NaT,
            "SignalPrice": "",
        })

    if add_rows:
        out = pd.concat([out, pd.DataFrame(add_rows)], ignore_index=True)
        out["_is_real"] = out["EntryTimeUTC"].notna()
        out = out.sort_values(["Ticker", "_is_real"]).drop_duplicates(subset=["Ticker"], keep="last").drop(columns=["_is_real"]).reset_index(drop=True)

    return out

def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    # Realized stats
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

    # Open counts by UNIQUE tickers per source
    if open_df.empty:
        open_counts = pd.DataFrame(columns=["Source", "OpenSignals"])
    else:
        open_counts = (open_df.groupby("Source")["Ticker"]
                       .nunique()
                       .rename("OpenSignals")
                       .reset_index())

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer")
    if perf.empty:
        return pd.DataFrame(columns=["Source", "Trades", "Wins", "WinRate%", "AvgReturn%", "MedianReturn%", "OpenSignals"])

    # Clean types
    perf["Source"] = perf["Source"].fillna("(unknown)")
    perf["Trades"] = pd.to_numeric(perf["Trades"], errors="coerce").fillna(0).astype(int)
    perf["Wins"] = pd.to_numeric(perf["Wins"], errors="coerce").fillna(0).astype(int)
    for col in ["WinRate%", "AvgReturn%", "MedianReturn%"]:
        perf[col] = pd.to_numeric(perf[col], errors="coerce").fillna(0.0)
    perf["OpenSignals"] = pd.to_numeric(perf["OpenSignals"], errors="coerce").fillna(0).astype(int)
    perf = perf.sort_values(["Source"]).reset_index(drop=True)
    return perf

def print_unknown_source_summary(signals_df: pd.DataFrame, tx_df: pd.DataFrame, realized_df: pd.DataFrame, open_df: pd.DataFrame):
    # Universe
    from_tx = set(tx_df.get("Symbol", pd.Series([], dtype=str)).dropna().astype(str).str.upper().tolist())
    from_real = set(realized_df.get("Ticker", pd.Series([], dtype=str)).dropna().astype(str).str.upper().tolist()) if not realized_df.empty else set()
    from_open = set(open_df.get("Ticker", pd.Series([], dtype=str)).dropna().astype(str).str.upper().tolist()) if not open_df.empty else set()
    universe = sorted(from_tx | from_real | from_open)

    if signals_df.empty:
        unknowns = sorted(universe)
    else:
        s = signals_df.copy()
        s["Ticker"] = s["Ticker"].astype(str).str.upper()
        s["Source"] = s.get("Source", "").astype(str).str.strip()
        have_src = set(s.loc[s["Source"] != "", "Ticker"].unique().tolist())
        unknowns = [t for t in universe if t not in have_src]

    if unknowns:
        print("🔎 Unknown-source tickers (add one row in Signals with Source to fix):")
        print("  " + ", ".join(unknowns))
    else:
        print("✅ No unknown-source tickers.")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build Performance tabs from Signals/Transactions/Holdings.")
    ap.add_argument("--no-live", action="store_true", help="Do not insert GOOGLEFINANCE formulas in Open_Positions")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    args = ap.parse_args()
    ENABLE_LIVE = not args.no_live
    DEBUG = bool(args.debug)

    print("📊 Building performance dashboard…")
    gc = auth_gspread()

    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_tx = open_ws(gc, TAB_TRANSACTIONS)
    ws_h = open_ws(gc, TAB_HOLDINGS)

    df_sig = read_tab(ws_sig)
    df_tx = read_tab(ws_tx)
    df_h = read_tab(ws_h)

    print(f"• Loaded: Signals={len(df_sig)} rows, Transactions={len(df_tx)} rows, Holdings={len(df_h)} rows")

    sig = load_signals(df_sig)
    tx, unmatched_df = load_transactions(df_tx, debug=DEBUG)
    hold = load_holdings(df_h)

    if DEBUG:
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        if not tx.empty:
            print(tx.head(8).to_string())

    realized_df, open_df = build_realized_and_open(tx, sig)

    # Reconcile with holdings (surface still-owned tickers whose BUY is outside export window)
    open_df = reconcile_open_with_holdings(open_df, df_h, sig)

    # Live price formulas (optional)
    mapping = read_mapping(gc)
    open_df = add_live_price_formulas(open_df, mapping, enable_live=ENABLE_LIVE)

    # Pretty column order
    if not realized_df.empty:
        realized_df = realized_df[[
            "Ticker", "Qty", "EntryPrice", "ExitPrice", "Return%", "HoldDays",
            "EntryTimeUTC", "ExitTimeUTC", "Source", "Timeframe", "SignalTimeUTC", "SignalPrice"
        ]]
    if not open_df.empty:
        for col in ["PriceNow", "Unrealized%"]:
            if col not in open_df.columns:
                open_df[col] = ""
        open_df = open_df[[
            "Ticker", "OpenQty", "EntryPrice", "PriceNow", "Unrealized%",
            "EntryTimeUTC", "DaysOpen", "Source", "Timeframe", "SignalTimeUTC", "SignalPrice"
        ]]

    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy())

    # Write tabs
    ws_real = open_ws(gc, TAB_REALIZED)
    ws_open = open_ws(gc, TAB_OPEN)
    ws_perf = open_ws(gc, TAB_PERF)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_df)

    print(f"✅ Wrote {TAB_REALIZED}: {len(realized_df)} rows")
    print(f"✅ Wrote {TAB_OPEN}: {len(open_df)} rows")
    print(f"✅ Wrote {TAB_PERF}: {len(perf_df)} rows")

    # Unknown source summary
    print_unknown_source_summary(sig, tx, realized_df, open_df)

    # Unmatched SELL summary
    if not unmatched_df.empty:
        count = len(unmatched_df)
        if DEBUG:
            print("⚠️ Unmatched SELLs (no prior BUYs or over-sell):")
            for _, r in unmatched_df.iterrows():
                print(f"  • {r['Symbol']} SELL on {pd.to_datetime(r['When']).isoformat()} qty={r['Qty']} price={r['Price']} — {r['Reason']}")
        else:
            print(f"⚠️ Summary: {count} unmatched SELL events (use --debug for details).")

    print("🎯 Done.")

if __name__ == "__main__":
    main()
