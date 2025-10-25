#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds three dashboard tabs in your Google Sheet:
- Realized_Trades: FIFO-matched BUY→SELL lots with returns
- Open_Positions : Remaining open BUY lots with live price formula + unrealized P%
- Performance_By_Source: Win% / Avg% / Median% grouped by signal Source

Attribution of BUYs to a Source:
- strict: last signal at/<= buy time
- loose : best signal within ± SIGNAL_WINDOW_DAYS (prefer before; else after)
- Fallback: Mapping!Source (Ticker -> Source) if no time-aligned signal exists

Also writes a Diagnostics_Unmatched tab with any SELLs that had no available BUYs.
"""

import math
import re
import argparse
import datetime as dt
from collections import deque, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS      = "Signals"
TAB_TRANSACTIONS = "Transactions"
TAB_HOLDINGS     = "Holdings"          # optional
TAB_REALIZED     = "Realized_Trades"
TAB_OPEN         = "Open_Positions"
TAB_PERF         = "Performance_By_Source"
TAB_DIAG_UNMATCH = "Diagnostics_Unmatched"  # new

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "

# Attribution + matching behavior
ATTRIBUTION_MODE   = "loose"   # "strict" (signal at/<= buy) or "loose" (± window)
SIGNAL_WINDOW_DAYS = 30        # used when ATTRIBUTION_MODE == "loose"
IGNORE_UNMATCHED_SELLS = True  # True = warn but continue

# Optional fallback: Mapping!Source column (Ticker -> Source) if no time-aligned signal exists
USE_MAPPING_SOURCE_FALLBACK = True

# ─────────────────────────────
# UTILITIES
# ─────────────────────────────
def auth_gspread():
    print("📊 Building performance dashboard…")
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, tab):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=500, cols=26)

def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    # Strip strings safely
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def to_float(series: pd.Series) -> pd.Series:
    def conv(x):
        if isinstance(x, str):
            x = x.replace("$","").replace(",","").strip()
        try:
            return float(x)
        except Exception:
            return np.nan
    return series.map(conv)

BLACKLIST_TOKENS = {
    "CASH","USD","INTEREST","DIVIDEND","REINVESTMENT","FEE","WITHDRAWAL","DEPOSIT",
    "TRANSFER","SWEEP"
}

def base_symbol_from_string(s) -> str:
    """Robustly extract a base equity/ETF symbol; return '' for cash/blank/other."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    token = s.split()[0]                 # take first whitespace-delimited chunk
    token = token.split("-")[0]          # strip option/lot suffix like "AAPL-12345"
    token = token.replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token).upper()
    if not token:
        return ""
    if token in BLACKLIST_TOKENS:
        return ""
    if token.isdigit():
        return ""
    # very long numeric-like or account-number-ish -> ignore
    if len(token) > 8 and token.isalnum():
        return ""
    return token

def chunked_write(ws, df: pd.DataFrame):
    """Write df to ws using values-first update API to avoid deprecation warnings."""
    ws.clear()
    if df.empty:
        # still write a header row if possible to keep structure
        ws.resize(rows=100, cols=8)
        ws.update([["(empty)"]], range_name="A1:A1")
        return

    rows, cols = df.shape
    ws.resize(rows=max(rows+5, 200), cols=max(cols+2, 8))

    header = [str(c) for c in df.columns]
    data = df.astype(str).fillna("").values.tolist()
    all_values = [header] + data

    CHUNK = 500
    start = 0
    row_cursor = 1
    while start < len(all_values):
        end = min(start+CHUNK, len(all_values))
        block = all_values[start:end]
        top_left = gspread.utils.rowcol_to_a1(row_cursor, 1)
        ncols = len(header)
        bottom_right = gspread.utils.rowcol_to_a1(row_cursor + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)
        row_cursor += len(block)
        start = end

def read_mapping(gc) -> Dict[str, Dict[str,str]]:
    """
    Return {ticker: {'FormulaSym': 'EXCH: TKR', 'TickerYF': 'TKR', 'Source': 'Weinstein'}} if Mapping tab exists.
    """
    out: Dict[str, Dict[str,str]] = {}
    try:
        mws = open_ws(gc, "Mapping")
        dfm = read_tab(mws)
        if not dfm.empty and "Ticker" in dfm.columns:
            for _, row in dfm.iterrows():
                t = str(row.get("Ticker","")).strip().upper()
                if not t:
                    continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym","")).strip(),
                    "TickerYF": str(row.get("TickerYF","")).strip().upper(),
                    "Source": str(row.get("Source","")).strip()
                }
    except Exception:
        pass
    return out

def googlefinance_formula_for(ticker:str, row_idx:int, mapping:Dict[str,Dict[str,str]]) -> str:
    base = base_symbol_from_string(ticker)
    mapped = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    sym = mapped.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_idx},"price"), ""))'

# ─────────────────────────────
# LOAD SHEET DATA
# ─────────────────────────────
def load_signals(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig.empty:
        return df_sig
    df = df_sig.copy()
    tcol = next((c for c in df.columns if c.lower() in ("ticker","symbol")), None)
    if not tcol:
        raise ValueError("Signals tab needs a 'Ticker' column.")
    df["Ticker"] = df[tcol].map(base_symbol_from_string)
    df["Source"] = df.get("Source","")
    df["Direction"] = df.get("Direction","")
    df["Timeframe"] = df.get("Timeframe","")
    tscol = next((c for c in df.columns if c.lower().startswith("timestamp")), None)
    df["TimestampUTC"] = to_dt(df[tscol]) if tscol else pd.NaT
    df["Price"] = df.get("Price","")
    return df[["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"]]

def load_transactions(df_tx: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (tx, unmatched_sell_rows)
    tx columns: When, Type ('BUY'/'SELL'), Symbol, Qty(>0), Price
    """
    if df_tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Try common Fidelity headers
    symcol   = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    typecol  = next((c for c in df_tx.columns if c.lower() in ("type","action")), None)
    desc_col = next((c for c in df_tx.columns if "description" in c.lower()), None)
    pricecol = next((c for c in df_tx.columns if "price" in c.lower()), None)
    amtcol   = next((c for c in df_tx.columns if "amount" in c.lower()), None)
    qtycol   = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    datecol  = next((c for c in df_tx.columns if "run date" in c.lower()), None)
    if not (typecol and datecol):
        raise ValueError("Transactions needs 'Action/Type' and 'Run Date' columns (Fidelity).")

    df = df_tx.copy()
    action = df[typecol].fillna("").astype(str)
    desc   = df[desc_col].fillna("").astype(str) if desc_col else pd.Series([""]*len(df))
    action_up = action.str.upper()
    desc_up   = desc.str.upper()

    patt = r"\b(YOU\s+)?(BOUGHT|SOLD|BUY|SELL)\b"
    mask_action = action_up.str.contains(patt, regex=True, na=False)
    mask_desc   = desc_up.str.contains(patt,   regex=True, na=False)
    trade_mask = mask_action | mask_desc

    if debug:
        total = len(df)
        detected = trade_mask.sum()
        print(f"• load_transactions: detected {detected} trade-like rows (of {total})")

    df = df[trade_mask].copy()

    # Symbol
    if symcol:
        sym = df[symcol].map(base_symbol_from_string)
    else:
        sym = pd.Series([""]*len(df), index=df.index)

    def symbol_from_action(a):
        s = str(a)
        m = re.search(r"\(([A-Za-z0-9\.\-]{1,8})\)", s)
        if m:
            return m.group(1).upper()
        return ""

    # backfill empty symbol from action/description
    sym = sym.fillna("").astype(str)
    sym = np.where(sym == "", action.map(symbol_from_action), sym)
    sym = np.where(sym == "", desc.map(symbol_from_action), sym)
    sym = pd.Series(sym, index=df.index).map(base_symbol_from_string)

    # When
    when = to_dt(df[datecol])

    # Price & Qty
    price = to_float(df[pricecol]) if pricecol else pd.Series(np.nan, index=df.index)

    qraw = df[qtycol] if qtycol else pd.Series(np.nan, index=df.index)
    qty = pd.to_numeric(qraw, errors="coerce")

    # Normalize type
    def normalize_type(a, d):
        s = f"{a} {d}".upper()
        if "SOLD" in s or re.search(r"\bSELL\b", s):
            return "SELL"
        if "BOUGHT" in s or re.search(r"\bBUY\b", s):
            return "BUY"
        return ""
    ttype = [normalize_type(a, d) for a, d in zip(action, desc)]

    tx = pd.DataFrame({
        "When": when,
        "Type": ttype,
        "Symbol": sym,
        "QtyRaw": qty,
        "Price": price,
    })

    # Keep only BUY/SELL with symbol/time
    tx = tx[tx["Type"].isin(["BUY","SELL"]) & tx["Symbol"].ne("") & tx["When"].notna()].copy()

    # Direction: Fidelity sells appear negative qty. Fix signs → all Qty positive, Type carries direction.
    tx["Qty"] = tx["QtyRaw"].abs()

    tx = tx.drop(columns=["QtyRaw"])
    tx.sort_values("When", inplace=True)
    tx.reset_index(drop=True, inplace=True)

    if debug:
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        print(tx.head(8).to_string(index=False))

    # unmatched sells for diagnostics (we check with a quick FIFO sim per symbol)
    unmatched_rows = []
    lot_balance = defaultdict(float)
    for _, r in tx.iterrows():
        symb = r["Symbol"]; q = float(r["Qty"])
        if r["Type"] == "BUY":
            lot_balance[symb] += q
        else:
            if lot_balance[symb] + 1e-9 < q:
                unmatched_rows.append({
                    "When": r["When"], "Symbol": symb, "Qty": q, "Price": r["Price"], "Note": "SELL exceeds available BUY lots"
                })
            lot_balance[symb] = max(0.0, lot_balance[symb] - q)

    return tx, pd.DataFrame(unmatched_rows)

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty:
        return df_h
    symcol  = next((c for c in df_h.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol  = next((c for c in df_h.columns if "quantity" in c.lower()), None)
    pricecol= next((c for c in df_h.columns if "price" in c.lower()), None)
    df = pd.DataFrame()
    if symcol:   df["Ticker"] = df_h[symcol].map(base_symbol_from_string)
    if qtycol:   df["Qty"]    = to_float(df_h[qtycol])
    if pricecol: df["Price"]  = to_float(df_h[pricecol])
    return df

# ─────────────────────────────
# SIGNAL CHOOSER (STRICT/LOOSE + MAPPING FALLBACK)
# ─────────────────────────────
def get_mapping_sources(mapping: Dict[str, Dict[str, str]]) -> dict:
    out = {}
    for tkr, row in mapping.items():
        src = (row or {}).get("Source","").strip()
        if src:
            out[tkr] = src
    return out

def choose_signal_for_buy(ticker, when, sig_by_ticker, mapping_sources):
    """
    Return {'Source','Timeframe','SigTime','SigPrice'} for a BUY.
    - strict: last signal at/<= when
    - loose : best signal within ± SIGNAL_WINDOW_DAYS (prefer before; else after)
    - fallback: Mapping!Source if none found
    """
    base = {"Source":"(unknown)", "Timeframe":"", "SigTime": pd.NaT, "SigPrice": ""}
    arr = sig_by_ticker.get(ticker, [])
    if not arr:
        if USE_MAPPING_SOURCE_FALLBACK and ticker in mapping_sources:
            b = base.copy(); b["Source"] = mapping_sources[ticker]; return b
        return base

    if ATTRIBUTION_MODE == "strict":
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                break
            if t <= when:
                return payload
        if USE_MAPPING_SOURCE_FALLBACK and ticker in mapping_sources:
            b = base.copy(); b["Source"] = mapping_sources[ticker]; return b
        return base

    # loose mode
    window = pd.Timedelta(days=SIGNAL_WINDOW_DAYS)
    best_before = None
    best_after  = None
    for t, payload in arr:
        if pd.isna(t) or pd.isna(when):
            continue
        if when >= t and (when - t) <= window:
            if (best_before is None) or (t > best_before[0]):
                best_before = (t, payload)
        elif t > when and (t - when) <= window:
            if (best_after is None) or (t < best_after[0]):
                best_after = (t, payload)
    if best_before:
        return best_before[1]
    if best_after:
        return best_after[1]

    if USE_MAPPING_SOURCE_FALLBACK and ticker in mapping_sources:
        b = base.copy(); b["Source"] = mapping_sources[ticker]; return b
    return base

# ─────────────────────────────
# MATCH SIGNALS → BUYS, THEN FIFO CLOSE WITH SELLS
# ─────────────────────────────
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame, mapping: Dict[str,Dict[str,str]], debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Index signals by (Ticker, time), only BUYs
    sig_buy = sig[(sig["Direction"].str.upper()=="BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True)

    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append( (r["TimestampUTC"], {
            "Source": r.get("Source",""),
            "Timeframe": r.get("Timeframe",""),
            "SigTime": r.get("TimestampUTC"),
            "SigPrice": r.get("Price",""),
        }))

    mapping_sources = get_mapping_sources(mapping)

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []
    unmatched_sells_verbose = []

    for _, row in tx.iterrows():
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = row["Qty"] if not math.isnan(row["Qty"]) else 0.0
        price = row["Price"] if not math.isnan(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr=="":
            continue

        if ttype == "BUY":
            siginfo = choose_signal_for_buy(tkr, when, sig_by_ticker, mapping_sources)
            lots[tkr].append({
                "qty_left": qty,
                "entry_price": price,
                "entry_time": when,
                "source": siginfo.get("Source",""),
                "timeframe": siginfo.get("Timeframe",""),
                "sig_time": siginfo.get("SigTime"),
                "sig_price": siginfo.get("SigPrice"),
            })
        elif ttype == "SELL":
            remaining = qty
            consumed_any = False
            while remaining > 0 and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                if take <= 0: break
                entry = lot["entry_price"] if not math.isnan(lot["entry_price"]) else 0.0
                exitp = price if not math.isnan(price) else 0.0
                ret_pct = ((exitp - entry) / entry * 100.0) if entry else np.nan
                held_days = (when - lot["entry_time"]).days if (pd.notna(lot["entry_time"]) and pd.notna(when)) else ""
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
                consumed_any = True
                lot["qty_left"] -= take
                remaining -= take
                if lot["qty_left"] <= 1e-9:
                    lots[tkr].popleft()

            if remaining > 1e-9 and not consumed_any:
                note = f"SELL exceeds available BUY lots (remaining {remaining})"
                unmatched_sells_verbose.append({
                    "When": when, "Symbol": tkr, "Qty": qty, "Price": price, "Note": note
                })
                if not IGNORE_UNMATCHED_SELLS and debug:
                    print(f"⚠️ Unmatched SELL {tkr} {when} qty={qty} price={price}")

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df = realized_df.sort_values("ExitTimeUTC", ignore_index=True)

    # Remaining open lots
    now_utc = pd.Timestamp.utcnow()  # tz-aware already; no tz_localize
    open_rows = []
    for tkr, q in lots.items():
        for lot in q:
            if lot["qty_left"] <= 1e-9: continue
            open_rows.append({
                "Ticker": tkr,
                "OpenQty": round(lot["qty_left"], 6),
                "EntryPrice": round(lot["entry_price"], 6) if not math.isnan(lot["entry_price"]) else "",
                "EntryTimeUTC": lot["entry_time"],
                "DaysOpen": (now_utc - lot["entry_time"]).days if pd.notna(lot["entry_time"]) else "",
                "Source": lot["source"],
                "Timeframe": lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice": lot["sig_price"],
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df = open_df.sort_values("EntryTimeUTC", ignore_index=True)
    return realized_df, open_df, unmatched_sells_verbose

def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str,Dict[str,str]]) -> pd.DataFrame:
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now = []
    unreal = []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        row_index = idx + 2  # header at row 1
        formula = googlefinance_formula_for(tkr, row_index, mapping)
        price_now.append(formula)
        try:
            epf = float(ep)
            unreal.append(f'=IFERROR(( {formula} / {epf} - 1 ) * 100,"")' if epf > 0 else "")
        except Exception:
            unreal.append("")
    out.insert(out.columns.get_loc("EntryPrice")+1, "PriceNow", price_now)
    out.insert(out.columns.get_loc("PriceNow")+1, "Unrealized%", unreal)
    return out

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
            "WinRate%": (g["is_win"].mean().fillna(0.0).values*100).round(2),
            "AvgReturn%": g["ret"].mean().round(2).values,
            "MedianReturn%": g["ret"].median().round(2).values,
        })

    if open_df.empty:
        open_counts = pd.DataFrame(columns=["Source","OpenSignals"])
    else:
        open_counts = open_df.groupby("Source").size().rename("OpenSignals").reset_index()

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer")
    if perf.empty:
        return pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenSignals"])

    perf["Trades"] = pd.to_numeric(perf["Trades"], errors="coerce").fillna(0).astype(int)
    perf["Wins"] = pd.to_numeric(perf["Wins"], errors="coerce").fillna(0).astype(int)
    for col in ["WinRate%","AvgReturn%","MedianReturn%"]:
        perf[col] = pd.to_numeric(perf[col], errors="coerce").fillna(0.0)
    perf["OpenSignals"] = pd.to_numeric(perf["OpenSignals"], errors="coerce").fillna(0).astype(int)
    perf = perf.sort_values(["Source"]).reset_index(drop=True)
    return perf

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    gc = auth_gspread()

    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_tx  = open_ws(gc, TAB_TRANSACTIONS)
    ws_h   = open_ws(gc, TAB_HOLDINGS)

    df_sig = read_tab(ws_sig)
    df_tx  = read_tab(ws_tx)
    df_h   = read_tab(ws_h)

    print(f"• Loaded: Signals={len(df_sig)} rows, Transactions={len(df_tx)} rows, Holdings={len(df_h)} rows")

    sig = load_signals(df_sig)
    tx, unmatched_df = load_transactions(df_tx, debug=args.debug)
    _   = load_holdings(df_h)  # optional; currently unused in matching

    mapping = read_mapping(gc)

    realized_df, open_df, unmatched_verbose = build_realized_and_open(tx, sig, mapping, debug=args.debug)
    if args.debug:
        print(f"• realized trades: {0 if realized_df.empty else len(realized_df)} | open lots: {0 if open_df.empty else len(open_df)}")

    # Add live price formulas for open lots
    open_df = add_live_price_formulas(open_df, mapping)

    # Pretty column order
    if not realized_df.empty:
        realized_df = realized_df[[
            "Ticker","Qty","EntryPrice","ExitPrice","Return%","HoldDays",
            "EntryTimeUTC","ExitTimeUTC","Source","Timeframe","SignalTimeUTC","SignalPrice"
        ]]
    if not open_df.empty:
        open_df = open_df[[
            "Ticker","OpenQty","EntryPrice","PriceNow","Unrealized%","EntryTimeUTC","DaysOpen",
            "Source","Timeframe","SignalTimeUTC","SignalPrice"
        ]]

    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy())

    # Diagnostics sheet for unmatched sells
    diag_rows = unmatched_df.to_dict(orient="records") if not unmatched_df.empty else []
    if unmatched_verbose:
        diag_rows.extend(unmatched_verbose)
    diag_df = pd.DataFrame(diag_rows)
    if not diag_df.empty:
        diag_df = diag_df.sort_values("When", ignore_index=True)

    # Write tabs
    ws_real = open_ws(gc, TAB_REALIZED)
    ws_open = open_ws(gc, TAB_OPEN)
    ws_perf = open_ws(gc, TAB_PERF)
    ws_diag = open_ws(gc, TAB_DIAG_UNMATCH)

    chunked_write(ws_real, realized_df)
    chunked_write(ws_open, open_df)
    chunked_write(ws_perf, perf_df)
    chunked_write(ws_diag, diag_df)

    print(f"✅ Wrote {TAB_REALIZED}: {0 if realized_df.empty else len(realized_df)} rows")
    print(f"✅ Wrote {TAB_OPEN}: {0 if open_df.empty else len(open_df)} rows")
    print(f"✅ Wrote {TAB_PERF}: {0 if perf_df.empty else len(perf_df)} rows")
    if not diag_df.empty:
        print(f"⚠️ Summary: {len(diag_df)} unmatched SELL events (see {TAB_DIAG_UNMATCH}).")
    print("🎯 Done.")

if __name__ == "__main__":
    main()
