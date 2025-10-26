#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build performance dashboard tabs from Google Sheet:
- Reads:  Signals, Transactions, Holdings (optional), Mapping (optional)
- Writes: Realized_Trades, Open_Positions, Performance_By_Source

Highlights
- Robust symbol parsing
- BUY/SELL extraction from Fidelity-style exports
- FIFO realization
- Optional live price formulas with GOOGLEFINANCE (disable via --no-live)
- Clean argparse (--debug works on Python 3.11)
- Uses gspread.update(values, range_name=...) (no deprecation warnings)
"""

from __future__ import annotations
import argparse
import math
import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG (edit if needed)
# ──────────────────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS       = "Signals"
TAB_TRANSACTIONS  = "Transactions"
TAB_HOLDINGS      = "Holdings"           # optional
TAB_MAPPING       = "Mapping"            # optional: Ticker → FormulaSym / TickerYF
TAB_REALIZED      = "Realized_Trades"
TAB_OPEN          = "Open_Positions"
TAB_PERF          = "Performance_By_Source"

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "      # used if Mapping.FormulaSym missing
ROW_CHUNK = 500                           # gspread batch size

# ──────────────────────────────────────────────────────────────────────────────
# AUTH / SHEET I/O
# ──────────────────────────────────────────────────────────────────────────────
def auth_gspread():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, tab):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=100, cols=26)

def _sanitize_df_for_read(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    return _sanitize_df_for_read(df)

def chunked_update(ws, values: List[List[str]]):
    """Use new gspread signature: update(values, range_name=...)"""
    if not values:
        return
    total = len(values)
    start_row = 1
    ncols = len(values[0]) if values else 1
    while start_row <= total:
        end_row = min(start_row + ROW_CHUNK - 1, total)
        top_left = gspread.utils.rowcol_to_a1(start_row, 1)
        bottom_right = gspread.utils.rowcol_to_a1(end_row, ncols)
        ws.update(values[start_row - 1 : end_row], range_name=f"{top_left}:{bottom_right}")
        start_row = end_row + 1

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.resize(rows=100, cols=8)
        ws.update([["(empty)"]], range_name="A1:A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows+5), cols=max(min(26, cols+2), 8))
    header = [str(c) for c in df.columns]
    body = df.astype(str).fillna("").values.tolist()
    values = [header] + body
    chunked_update(ws, values)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
BLACKLIST_TOKENS = {
    "CASH", "USD", "INTEREST", "DIVIDEND", "REINVESTMENT", "FEE",
    "WITHDRAWAL", "DEPOSIT", "TRANSFER", "SWEEP"
}

def base_symbol_from_string(s) -> str:
    """Extract plausible equity/ETF symbol; return '' for cash/blank/noise."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    token = s.split()[0]
    token = token.split("-")[0]                 # AAPL-123 → AAPL
    token = token.replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token).upper()
    if not token or token in BLACKLIST_TOKENS:
        return ""
    if token.isdigit():
        return ""
    if len(token) > 8 and token.isalnum():
        return ""
    return token

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

def read_mapping(gc) -> Dict[str, Dict[str,str]]:
    out: Dict[str, Dict[str,str]] = {}
    try:
        ws = open_ws(gc, TAB_MAPPING)
        df = read_tab(ws)
        if df.empty or "Ticker" not in df.columns:
            return out
        for _, row in df.iterrows():
            t = str(row.get("Ticker","")).strip().upper()
            if not t:
                continue
            out[t] = {
                "FormulaSym": str(row.get("FormulaSym","")).strip(),
                "TickerYF":   str(row.get("TickerYF","")).strip().upper(),
            }
    except Exception:
        pass
    return out

def googlefinance_formula_for(ticker: str, row_idx: int, mapping: Dict[str,Dict[str,str]]) -> str:
    base = base_symbol_from_string(ticker)
    mapped = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    sym = mapped.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)
    # Guard with IFERROR; fall back to the value in column B{row} as a last resort.
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"),IFERROR(GOOGLEFINANCE(B{row_idx},"price"),""))'

# ──────────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────────
def load_signals(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig.empty:
        return pd.DataFrame(columns=["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"])
    df = df_sig.copy()

    tcol = next((c for c in df.columns if c.lower() in ("ticker","symbol")), None)
    if not tcol:
        raise ValueError("Signals tab needs a 'Ticker' column.")

    # Timestamp column (any col that starts with 'timestamp')
    tscol = next((c for c in df.columns if c.lower().startswith("timestamp")), None)

    df["Ticker"]      = df[tcol].map(base_symbol_from_string)
    df["Source"]      = df.get("Source","")
    df["Direction"]   = df.get("Direction","")
    df["Timeframe"]   = df.get("Timeframe","")
    df["TimestampUTC"]= to_dt(df[tscol]) if tscol else pd.NaT
    df["Price"]       = df.get("Price","")

    out = df[["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"]].copy()
    # Keep only non-empty tickers
    out = out[out["Ticker"].ne("")]
    return out

def load_transactions(df_tx: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (tx, unmatched_df)
      tx columns: When (UTC), Type (BUY/SELL), Symbol, Qty (>0), Price
      unmatched_df: SELL rows that couldn't match prior BUYs when FIFO runs (filled later)
    """
    if df_tx.empty:
        return pd.DataFrame(columns=["When","Type","Symbol","Qty","Price"]), pd.DataFrame()

    df = df_tx.copy()

    # Guess common Fidelity columns
    symcol   = next((c for c in df.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    actioncol= next((c for c in df.columns if c.lower() in ("action","type")), None)
    desccol  = next((c for c in df.columns if "description" in c.lower()), None)
    qtycol   = next((c for c in df.columns if "quantity" in c.lower()), None)
    pricecol = next((c for c in df.columns if "price" in c.lower()), None)
    amtcol   = next((c for c in df.columns if "amount" in c.lower()), None)
    datecol  = next((c for c in df.columns if "run date" in c.lower() or "date" == c.lower()), None)

    if not datecol:
        raise ValueError("Transactions tab must include a 'Run Date' or 'Date' column.")
    if not actioncol and not desccol:
        raise ValueError("Transactions need an Action/Type or Description column to identify BUY/SELL.")

    # Build “trade-like” mask from Action/Description
    patt = r"(?:\bYOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"
    action_up = df[actioncol].str.upper() if actioncol in df.columns else pd.Series("", index=df.index)
    desc_up   = df[desccol].str.upper()   if desccol   in df.columns else pd.Series("", index=df.index)
    mask_action = action_up.str.contains(patt, regex=True, na=False)
    mask_desc   = desc_up.str.contains(patt,   regex=True, na=False)
    mask_trade  = mask_action | mask_desc

    if debug:
        print(f"• load_transactions: detected {int(mask_trade.sum())} trade-like rows (of {len(df)})")

    df = df[mask_trade].copy()
    if df.empty:
        return pd.DataFrame(columns=["When","Type","Symbol","Qty","Price"]), pd.DataFrame()

    # Parse When
    df["When"] = to_dt(df[datecol])

    # Parse Symbol (fallback to extracting from Action/Description)
    raw_sym = df[symcol] if symcol else pd.Series("", index=df.index)
    sym = raw_sym.map(base_symbol_from_string).fillna("")

    def symbol_from_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        m = re.search(r"\(([A-Za-z0-9\.\-]{1,8})\)", text.upper())
        if m:
            return base_symbol_from_string(m.group(1))
        # final fallback: first ALLCAPS token up to 8 chars
        m2 = re.search(r"\b([A-Z][A-Z0-9\.\-]{0,7})\b", text.upper())
        return base_symbol_from_string(m2.group(1)) if m2 else ""

    fill_from_action = ~sym.astype(bool)
    if fill_from_action.any():
        sym.where(~fill_from_action, other=action_up.map(symbol_from_text), inplace=True)
    fill_from_desc = ~sym.astype(bool)
    if fill_from_desc.any():
        sym.where(~fill_from_desc, other=desc_up.map(symbol_from_text), inplace=True)

    df["Symbol"] = sym

    # Price
    price = to_float(df[pricecol]) if pricecol else pd.Series(np.nan, index=df.index)

    # Quantity (prefer explicit quantity; else derive from amount/price)
    if qtycol:
        qty = to_float(df[qtycol])
    else:
        if amtcol and pricecol:
            amt = to_float(df[amtcol])
            with np.errstate(divide='ignore', invalid='ignore'):
                qty = np.where((price!=0) & (~np.isnan(price)) & (~np.isnan(amt)), np.abs(amt)/np.abs(price), np.nan)
            qty = pd.Series(qty, index=df.index)
        else:
            qty = pd.Series(np.nan, index=df.index)

    # Normalize Type & sign of qty from action/desc text
    type_series = np.where(
        action_up.str.contains(r"\bSOLD|SELL\b", regex=True, na=False) |
        desc_up.str_contains(r"\bSOLD|SELL\b", regex=True, na=False),
        "SELL", "BUY"
    )
    df["Type"] = pd.Series(type_series, index=df.index)

    # Make QTY positive (we’ll use Type to decide entry/exit)
    qty = pd.to_numeric(qty, errors="coerce")
    qty = qty.abs()

    tx = pd.DataFrame({
        "When":   df["When"],
        "Type":   df["Type"],
        "Symbol": df["Symbol"],
        "Qty":    qty,
        "Price":  price,
    })

    # Clean rows
    tx = tx[tx["Symbol"].ne("") & tx["When"].notna()].copy()
    tx["Qty"] = pd.to_numeric(tx["Qty"], errors="coerce")
    tx = tx[tx["Qty"] > 0].copy()

    tx.sort_values("When", inplace=True, kind="mergesort")
    if debug:
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        print(tx.head(8).to_string(index=False))

    # unmatched placeholder (we return a real one after FIFO pass)
    return tx.reset_index(drop=True), pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# MATCHING / REALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    FIFO match SELLs vs prior BUYs per symbol.
    Also attach closest prior-or-same signal (by time) to each BUY lot for Source/Timeframe linkage.
    """
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build signal index per ticker (BUY signals only)
    sig_buy = sig[(sig["Direction"].str.upper()=="BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True, kind="mergesort")

    sig_index: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_index[r["Ticker"]].append((
            r["TimestampUTC"],
            {
                "Source":    r.get("Source",""),
                "Timeframe": r.get("Timeframe",""),
                "SigTime":   r.get("TimestampUTC"),
                "SigPrice":  r.get("Price",""),
            }
        ))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_index.get(tkr, [])
        if not arr:
            return {"Source":"(unknown)","Timeframe":"","SigTime":pd.NaT,"SigPrice":""}
        # find last t <= when
        best = {"Source":"(unknown)","Timeframe":"","SigTime":pd.NaT,"SigPrice":""}
        for t, payload in arr:
            if pd.isna(t) or pd.isna(when):
                # if we lack one of the timestamps, just take the latest we have
                best = payload
            elif t <= when:
                best = payload
        return best

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows: List[dict] = []
    unmatched_sell_rows: List[dict] = []

    for _, row in tx.iterrows():
        tkr, when, ttype, qty, price = row["Symbol"], row["When"], row["Type"], row["Qty"], row["Price"]
        if not tkr or pd.isna(when) or qty <= 0:
            continue

        if ttype == "BUY":
            siginfo = last_signal_for(tkr, when)
            lots[tkr].append({
                "qty_left":   float(qty),
                "entry_price": float(price) if not pd.isna(price) else np.nan,
                "entry_time": when,
                "source":     siginfo.get("Source",""),
                "timeframe":  siginfo.get("Timeframe",""),
                "sig_time":   siginfo.get("SigTime"),
                "sig_price":  siginfo.get("SigPrice"),
            })
        else:  # SELL
            remaining = float(qty)
            start_remaining = remaining
            while remaining > 1e-12 and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                entry = lot["entry_price"] if not pd.isna(lot["entry_price"]) else 0.0
                exitp = float(price) if not pd.isna(price) else 0.0
                ret_pct = ((exitp - entry) / entry * 100.0) if entry else np.nan
                held_days = (when - lot["entry_time"]).days if (not pd.isna(lot["entry_time"]) and not pd.isna(when)) else ""

                realized_rows.append({
                    "Ticker":        tkr,
                    "Qty":           round(take, 6),
                    "EntryPrice":    round(entry, 6) if entry else "",
                    "ExitPrice":     round(exitp, 6) if exitp else "",
                    "Return%":       round(ret_pct, 4) if not np.isnan(ret_pct) else "",
                    "HoldDays":      held_days,
                    "EntryTimeUTC":  lot["entry_time"],
                    "ExitTimeUTC":   when,
                    "Source":        lot["source"],
                    "Timeframe":     lot["timeframe"],
                    "SignalTimeUTC": lot["sig_time"],
                    "SignalPrice":   lot["sig_price"],
                })
                lot["qty_left"] -= take
                remaining -= take
                if lot["qty_left"] <= 1e-12:
                    lots[tkr].popleft()

            if remaining > 1e-12:
                unmatched_sell_rows.append({
                    "Ticker": tkr,
                    "SellWhenUTC": when,
                    "QtyRemain": round(remaining, 6),
                    "Price": float(price) if not pd.isna(price) else "",
                    "Note": "No prior BUY lot in window or SELL exceeds available lots"
                })

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df.sort_values("ExitTimeUTC", inplace=True, kind="mergesort", ignore_index=True)

    # Remaining open lots → open positions
    now_utc = pd.Timestamp.now(tz="UTC")
    open_rows: List[dict] = []
    for tkr, q in lots.items():
        for lot in q:
            if lot["qty_left"] <= 1e-12:
                continue
            open_rows.append({
                "Ticker":        tkr,
                "OpenQty":       round(lot["qty_left"], 6),
                "EntryPrice":    round(lot["entry_price"], 6) if not pd.isna(lot["entry_price"]) else "",
                "EntryTimeUTC":  lot["entry_time"],
                "DaysOpen":      (now_utc - lot["entry_time"]).days if not pd.isna(lot["entry_time"]) else "",
                "Source":        lot["source"],
                "Timeframe":     lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice":   lot["sig_price"],
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df.sort_values("EntryTimeUTC", inplace=True, kind="mergesort", ignore_index=True)

    unmatched_df = pd.DataFrame(unmatched_sell_rows)
    if debug and not unmatched_df.empty:
        print("⚠️ Unmatched SELLs (no prior BUYs or partial over-sell):")
        for _, r in unmatched_df.iterrows():
            print(f"  • {r['Ticker']} SELL on {r['SellWhenUTC'].isoformat()} qty={r['QtyRemain']} price={r['Price']} — {r['Note']}")
        print(f"⚠️ Total unmatched SELL events: {len(unmatched_df)}")

    return realized_df, open_df, unmatched_df

# ──────────────────────────────────────────────────────────────────────────────
# PRICE NOW / PERF
# ──────────────────────────────────────────────────────────────────────────────
def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str,Dict[str,str]], enabled: bool=True) -> pd.DataFrame:
    """Insert PriceNow + Unrealized% columns; optionally disable formulas."""
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now, unreal = [], []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        # Row index in the sheet (header=1, data starts at 2)
        row_index = idx + 2
        if enabled:
            formula = googlefinance_formula_for(tkr, row_index, mapping)
            price_now.append(formula)
            try:
                epf = float(ep)
                unreal.append(f'=IFERROR(({formula}/{epf}-1)*100,"")' if epf > 0 else "")
            except Exception:
                unreal.append("")
        else:
            price_now.append("")
            unreal.append("")
    out.insert(out.columns.get_loc("EntryPrice")+1, "PriceNow", price_now)
    out.insert(out.columns.get_loc("PriceNow")+1, "Unrealized%", unreal)
    return out

def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    # Realized stats
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

    # Open counts (how many open signals per source)
    if open_df.empty:
        open_counts = pd.DataFrame(columns=["Source","OpenSignals"])
    else:
        open_counts = open_df.groupby("Source").size().rename("OpenSignals").reset_index()

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer")
    if perf.empty:
        return pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenSignals"])

    perf["Trades"]       = pd.to_numeric(perf["Trades"], errors="coerce").fillna(0).astype(int)
    perf["Wins"]         = pd.to_numeric(perf["Wins"], errors="coerce").fillna(0).astype(int)
    for col in ["WinRate%","AvgReturn%","MedianReturn%"]:
        perf[col] = pd.to_numeric(perf[col], errors="coerce").fillna(0.0)
    perf["OpenSignals"]  = pd.to_numeric(perf["OpenSignals"], errors="coerce").fillna(0).astype(int)

    perf = perf.sort_values(["Source"]).reset_index(drop=True)
    return perf

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build dashboard tabs from Signals + Transactions.")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    ap.add_argument("--no-live", action="store_true", help="Do not add GOOGLEFINANCE live price formulas")
    args = ap.parse_args()
    DEBUG = bool(args.debug)
    LIVE  = not bool(args.no_live)

    print("📊 Building performance dashboard…")
    gc = auth_gspread()

    # Read tabs
    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_tx  = open_ws(gc, TAB_TRANSACTIONS)
    ws_h   = open_ws(gc, TAB_HOLDINGS)

    df_sig = read_tab(ws_sig)
    df_tx  = read_tab(ws_tx)
    df_h   = read_tab(ws_h)

    print(f"• Loaded: Signals={len(df_sig)} rows, Transactions={len(df_tx)} rows, Holdings={len(df_h)} rows")

    # Normalize inputs
    sig = load_signals(df_sig)
    tx, _ = load_transactions(df_tx, debug=DEBUG)
    _ = df_h  # reserved

    # FIFO
    realized_df, open_df, unmatched_df = build_realized_and_open(tx, sig, debug=DEBUG)

    # PriceNow formulas (optional)
    mapping = read_mapping(gc)
    open_df = add_live_price_formulas(open_df, mapping, enabled=LIVE)

    # Column order prettification
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

    # Perf by source
    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy())

    # Write outputs
    ws_real = open_ws(gc, TAB_REALIZED)
    ws_open = open_ws(gc, TAB_OPEN)
    ws_perf = open_ws(gc, TAB_PERF)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_df)

    if DEBUG and not unmatched_df.empty:
        print(f"⚠️ Summary: {len(unmatched_df)} unmatched SELL events (use --debug for details).")

    print(f"✅ Wrote {TAB_REALIZED}: {len(realized_df)} rows")
    print(f"✅ Wrote {TAB_OPEN}: {len(open_df)} rows")
    print(f"✅ Wrote {TAB_PERF}: {len(perf_df)} rows")
    print("🎯 Done.")

if __name__ == "__main__":
    main()
