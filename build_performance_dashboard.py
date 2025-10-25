#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds three tabs from your Google Sheet data:
  - Realized_Trades   (FIFO matched from Transactions using BUY/SELL)
  - Open_Positions    (remaining FIFO lots + live GOOGLEFINANCE price)
  - Performance_By_Source (Trades/Wins/WinRate/Avg/Median Return + OpenSignals)

Key fixes:
- SELL rows no longer disappear: we take abs() of Quantity so they pass the Qty>0 filter.
- Safer regex (non-capturing groups) to avoid pandas "match groups" warnings.
- Robust ticker extraction from Symbol/Action/Description.
- Uses gspread.update(values, range_name=...) to avoid deprecation warnings.
"""

import math, re, argparse
import datetime as dt
from collections import defaultdict, deque
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

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "

# ─────────────────────────────
# UTILITIES
# ─────────────────────────────
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
    # strip strings
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        # Use new signature (values first, then range_name=...) to avoid deprecation
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows+5), cols=max(min(26, cols+2), 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

    CHUNK = 500
    start = 0
    r = 1
    while start < len(data):
        end = min(start+CHUNK, len(data))
        block = data[start:end]
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        ncols = len(header)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        ws.update(block, range_name=f"{top_left}:{bottom_right}")
        r += len(block)
        start = end

def to_dt(series: pd.Series) -> pd.Series:
    # coerce & set UTC if naive
    s = pd.to_datetime(series, errors="coerce", utc=False)
    # localize naive to UTC
    if isinstance(s, pd.Series):
        try:
            # elements may be mixed; convert naive to UTC
            s = s.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            # if already tz-aware, convert
            try:
                s = s.dt.tz_convert("UTC")
            except Exception:
                pass
    return s

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
    token = s.split()[0]                 # first whitespace-delimited chunk
    token = token.split("-")[0]          # strip option/lot suffix like "AAPL-12345"
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

def read_mapping(gc) -> Dict[str, Dict[str,str]]:
    """Return {ticker: {'FormulaSym': 'EXCH: TKR', 'TickerYF': 'TKR'}} if Mapping tab exists."""
    try:
        mws = open_ws(gc, "Mapping")
        dfm = read_tab(mws)
        out: Dict[str, Dict[str,str]] = {}
        if not dfm.empty and "Ticker" in dfm.columns:
            for _, row in dfm.iterrows():
                t = str(row.get("Ticker","")).strip().upper()
                if not t: continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym","")).strip(),
                    "TickerYF": str(row.get("TickerYF","")).strip().upper()
                }
        return out
    except Exception:
        return {}

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

def _symbol_from_text(s: str) -> str:
    """
    Try to extract (TICKER) in parentheses or the last all-caps token.
    Handles strings like:
      'YOU SOLD NEXTRACKER INC CLASS A COM (NXT) (Cash)'
      'VANGUARD S&P 500 ETF (VOO) (Cash)'
    """
    if not isinstance(s, str):
        return ""
    # Try parentheses with capital letters/numbers/.- inside
    m = re.search(r"\(([A-Z0-9\.\-]{1,8})\)", s.upper())
    if m:
        return base_symbol_from_string(m.group(1))
    # Fallback: last capitalized token
    toks = re.findall(r"[A-Z0-9\.\-]{1,8}", s.upper())
    if toks:
        return base_symbol_from_string(toks[-1])
    return ""

def load_transactions(df_tx: pd.DataFrame, debug=False) -> pd.DataFrame:
    if df_tx.empty:
        return pd.DataFrame(columns=["When","Type","Symbol","Qty","Price"])

    # Column discovery
    symcol  = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    typecol = next((c for c in df_tx.columns if c.lower() in ("type","action")), None)
    pricecol= next((c for c in df_tx.columns if "price" in c.lower()), None)
    amtcol  = next((c for c in df_tx.columns if "amount" in c.lower()), None)
    qtycol  = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    datecol = next((c for c in df_tx.columns if "run date" in c.lower()), None)
    desccol = next((c for c in df_tx.columns if "description" in c.lower()), None)
    if not datecol or not typecol:
        raise ValueError("Transactions tab must include Run Date and Type/Action columns.")
    if not symcol and not desccol and not typecol:
        raise ValueError("Transactions tab needs at least Symbol or Description/Action to extract tickers.")

    df = df_tx.copy()

    # Base fields
    when = to_dt(df[datecol])
    action_raw = df[typecol].fillna("").astype(str)
    desc_raw   = df[desccol].fillna("").astype(str) if desccol else pd.Series([""]*len(df), index=df.index)
    symbol_raw = df[symcol].fillna("").astype(str) if symcol else pd.Series([""]*len(df), index=df.index)

    # Trade-like rows: YOU BOUGHT / YOU SOLD / BUY / SELL (non-capturing groups to silence warning)
    patt = r"\b(?:YOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"
    mask_action = action_raw.str.upper().str.contains(patt, regex=True, na=False)
    mask_desc   = desc_raw.str.upper().str.contains(patt,   regex=True, na=False)
    mask_trade = (mask_action | mask_desc)

    df = df[mask_trade].copy()

    # Ticker extraction: prefer Symbol column, then (TICKER) in Action/Description
    sym = symbol_raw.map(base_symbol_from_string)
    # fill blanks from Action then Description
    sym = np.where(sym.astype(str).eq(""), action_raw.map(_symbol_from_text), sym)
    sym = np.where(pd.Series(sym).astype(str).eq(""), desc_raw.map(_symbol_from_text), sym)
    sym = pd.Series(sym, index=df.index).astype(str).fillna("")
    sym = sym.map(base_symbol_from_string)

    # Quantities & prices
    if qtycol and qtycol in df.columns:
        qty = to_float(df[qtycol]).abs()  # ← IMPORTANT: abs() so SELLs aren't dropped later
    else:
        # derive from Amount/Price if present
        if amtcol and pricecol and (amtcol in df.columns) and (pricecol in df.columns):
            amt = to_float(df[amtcol])
            prc = to_float(df[pricecol])
            with np.errstate(divide='ignore', invalid='ignore'):
                qty = np.where((prc!=0) & (~np.isnan(prc)) & (~np.isnan(amt)), np.abs(amt)/np.abs(prc), np.nan)
            qty = pd.Series(qty, index=df.index).abs()
        else:
            qty = pd.Series(np.nan, index=df.index)
    price = to_float(df[pricecol]) if (pricecol and pricecol in df.columns) else pd.Series(np.nan, index=df.index)

    tx = pd.DataFrame({
        "When": when,
        "Type": action_raw.str.upper(),
        "Symbol": sym,
        "Qty": pd.to_numeric(qty, errors="coerce"),
        "Price": price,
    }).dropna(subset=["When"])

    # Keep only rows with a symbol & positive quantity
    tx = tx[(tx["Symbol"].ne("")) & (tx["Qty"] > 0)].copy()

    # Sort
    tx.sort_values("When", inplace=True)
    tx.reset_index(drop=True, inplace=True)

    if debug:
        print(f"• load_transactions: detected {mask_trade.sum()} trade-like rows (of {len(df_tx)})")
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        print(tx.head(8))
    return tx

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty: return pd.DataFrame(columns=["Ticker","Qty","Price"])
    symcol  = next((c for c in df_h.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol  = next((c for c in df_h.columns if "quantity" in c.lower()), None)
    pricecol= next((c for c in df_h.columns if "price" in c.lower()), None)
    df = pd.DataFrame(index=df_h.index)
    df["Ticker"] = df_h[symcol].map(base_symbol_from_string) if symcol else ""
    df["Qty"]    = to_float(df_h[qtycol]) if qtycol else np.nan
    df["Price"]  = to_float(df_h[pricecol]) if pricecol else np.nan
    return df

# ─────────────────────────────
# MATCH SIGNALS → BUYS, THEN FIFO CLOSE WITH SELLS
# ─────────────────────────────
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame, debug=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tx.empty:
        if debug: print("• build_realized_and_open: no transactions → nothing to match")
        return pd.DataFrame(), pd.DataFrame()

    # Filter signals to BUYs with ticker
    sig_buy = sig[(sig["Direction"].astype(str).str.upper()=="BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True)

    # Index signals by ticker for "last signal at/<= time"
    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append( (r["TimestampUTC"], {
            "Source": r.get("Source",""),
            "Timeframe": r.get("Timeframe",""),
            "SigTime": r.get("TimestampUTC"),
            "SigPrice": r.get("Price",""),
        }))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        # from end to start to get most recent at/<= when
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                return payload
            if t <= when:
                return payload
        return {"Source":"(unknown)","Timeframe":"","SigTime": pd.NaT, "SigPrice": ""}

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []

    for _, row in tx.iterrows():
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = row["Qty"] if not math.isnan(row["Qty"]) else 0.0
        price = row["Price"] if not math.isnan(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr=="":
            continue

        is_buy  = ("BUY" in ttype) and ("SELL" not in ttype)
        is_sell = ("SELL" in ttype) or ("SOLD" in ttype)

        if is_buy:
            siginfo = last_signal_for(tkr, when)
            lots[tkr].append({
                "qty_left": qty,
                "entry_price": price,
                "entry_time": when,
                "source": siginfo.get("Source",""),
                "timeframe": siginfo.get("Timeframe",""),
                "sig_time": siginfo.get("SigTime"),
                "sig_price": siginfo.get("SigPrice"),
            })
        elif is_sell:
            remaining = qty
            while remaining > 0 and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                if take <= 0: break
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
        realized_df = realized_df.sort_values("ExitTimeUTC", ignore_index=True)

    # Remaining open lots
    now_utc = pd.Timestamp.now(tz="UTC")
    open_rows = []
    for tkr, q in lots.items():
        for lot in q:
            if lot["qty_left"] <= 1e-9: continue
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
        open_df = open_df.sort_values("EntryTimeUTC", ignore_index=True)

    if debug:
        print(f"• realized trades: {0 if realized_df.empty else len(realized_df)} | open lots: {0 if open_df.empty else len(open_df)}")

    return realized_df, open_df

def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str,Dict[str,str]]) -> pd.DataFrame:
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now = []
    unreal = []
    # row index in Sheets = dataframe index + 2 (header at row 1)
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        row_index = idx + 2
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

    # clean types
    perf["Trades"] = pd.to_numeric(perf["Trades"], errors="coerce").fillna(0).astype(int)
    perf["Wins"]   = pd.to_numeric(perf["Wins"],   errors="coerce").fillna(0).astype(int)
    for col in ["WinRate%","AvgReturn%","MedianReturn%"]:
        perf[col] = pd.to_numeric(perf[col], errors="coerce").fillna(0.0)
    perf["OpenSignals"] = pd.to_numeric(perf["OpenSignals"], errors="coerce").fillna(0).astype(int)

    perf = perf.sort_values(["Source"]).reset_index(drop=True)
    return perf

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="print debug info")
    args = ap.parse_args()

    print("📊 Building performance dashboard…")
    gc = auth_gspread()

    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_tx  = open_ws(gc, TAB_TRANSACTIONS)
    ws_h   = open_ws(gc, TAB_HOLDINGS)

    df_sig = read_tab(ws_sig)
    df_tx  = read_tab(ws_tx)
    df_h   = read_tab(ws_h)

    print(f"• Loaded: Signals={len(df_sig)} rows, Transactions={len(df_tx)} rows, Holdings={len(df_h)} rows")

    sig = load_signals(df_sig)
    tx  = load_transactions(df_tx, debug=args.debug)
    _   = load_holdings(df_h)  # optional; not used further here

    realized_df, open_df = build_realized_and_open(tx, sig, debug=args.debug)

    # Add live price formulas for open lots
    mapping = read_mapping(gc)
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

    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy(),)

    # Write tabs
    ws_real = open_ws(gc, TAB_REALIZED)
    ws_open = open_ws(gc, TAB_OPEN)
    ws_perf = open_ws(gc, TAB_PERF)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_df)

    print(f"✅ Wrote {TAB_REALIZED}: {0 if realized_df.empty else len(realized_df)} rows")
    print(f"✅ Wrote {TAB_OPEN}: {0 if open_df.empty else len(open_df)} rows")
    print(f"✅ Wrote {TAB_PERF}: {0 if perf_df.empty else len(perf_df)} rows")
    print("🎯 Done.")

if __name__ == "__main__":
    main()
