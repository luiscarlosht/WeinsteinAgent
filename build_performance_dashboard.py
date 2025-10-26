#!/usr/bin/env python3
"""
Builds three Google Sheets tabs from your Signals + Fidelity data:

  • Realized_Trades      — FIFO matched BUY→SELL lots with returns
  • Open_Positions       — remaining BUY lots + live price & unrealized %
  • Performance_By_Source— win rate & returns aggregated by Source

Key improvements:
- Robust symbol extraction (no shape/broadcast crashes).
- Regex warnings removed (non-capturing patterns).
- UTC handling fixed (no tz_localize() on aware ts).
- gspread .update() uses new signature (values first, range second).
- BUY/SELL detection computed AFTER filtering to trade-like rows (fixes length mismatch).
"""

import argparse
import math
import re
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
TAB_HOLDINGS     = "Holdings"
TAB_REALIZED     = "Realized_Trades"
TAB_OPEN         = "Open_Positions"
TAB_PERF         = "Performance_By_Source"

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "

# When true (via CLI --debug), prints extra info
DEBUG = False

# ─────────────────────────────
# AUTH / SHEETS I/O
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows + 5), cols=max(min(26, cols + 2), 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

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

# ─────────────────────────────
# UTILITIES
# ─────────────────────────────
BLACKLIST_TOKENS = {
    "CASH","USD","INTEREST","DIVIDEND","REINVESTMENT","FEE","WITHDRAWAL","DEPOSIT","TRANSFER","SWEEP"
}

def base_symbol_from_string(s) -> str:
    """Extract a plausible equity/ETF symbol; returns '' for cash/blank/etc."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    token = s.split()[0]
    token = token.split("-")[0]
    token = token.replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9.\-]", "", token).upper()
    if not token:
        return ""
    if token in BLACKLIST_TOKENS:
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
            x = x.replace("$", "").replace(",", "").strip()
        try:
            return float(x)
        except Exception:
            return np.nan
    return series.map(conv)

def ensure_utc_now():
    ts = pd.Timestamp.utcnow()
    return ts if ts.tz is not None else ts.tz_localize("UTC")

# ─────────────────────────────
# MAPPING / GOOGLEFINANCE
# ─────────────────────────────
def read_mapping(gc) -> Dict[str, Dict[str, str]]:
    try:
        mws = open_ws(gc, "Mapping")
        dfm = read_tab(mws)
        out: Dict[str, Dict[str,str]] = {}
        if not dfm.empty and "Ticker" in dfm.columns:
            for _, row in dfm.iterrows():
                t = str(row.get("Ticker","")).strip().upper()
                if not t:
                    continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym","")).strip(),
                    "TickerYF": str(row.get("TickerYF","")).strip().upper()
                }
        return out
    except Exception:
        return {}

def googlefinance_formula_for(ticker: str, row_idx: int, mapping: Dict[str, Dict[str,str]]) -> str:
    base = base_symbol_from_string(ticker)
    mp = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    sym = mp.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base if base else "")
    if not sym:
        return ""
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_idx},"price"), ""))'

# ─────────────────────────────
# LOADERS
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

def load_transactions(df_tx: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses Fidelity 'Transactions' to a normalized trades DataFrame:
      Columns: When (UTC), Type ('BUY'/'SELL'), Symbol, Qty (>0), Price

    Returns: (trades_df, unmatched_sells_df_placeholder)
    """
    if df_tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Fuzzy locate typical Fidelity columns
    datecol   = next((c for c in df_tx.columns if "run date" in c.lower()), None)
    actioncol = next((c for c in df_tx.columns if c.lower() == "action"), None)
    symcol    = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    desccol   = next((c for c in df_tx.columns if c.lower() == "description"), None)
    typecol   = next((c for c in df_tx.columns if c.lower() == "type"), None)
    qtycol    = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    pricecol  = next((c for c in df_tx.columns if "price" in c.lower()), None)

    if not datecol or not actioncol:
        raise ValueError("Transactions must include 'Run Date' and 'Action' columns.")

    df_all = df_tx.copy()

    # Build masks on the unfiltered frame
    action_all = df_all[actioncol].astype(str)
    desc_all   = df_all[desccol].astype(str) if desccol else pd.Series([""]*len(df_all), index=df_all.index)

    patt = r"\b(?:YOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"
    mask_action = action_all.str.upper().str.contains(patt, regex=True, na=False)
    mask_desc   = desc_all.str.upper().str.contains(patt,   regex=True, na=False)
    trade_mask  = mask_action | mask_desc

    if debug:
        print(f"• load_transactions: detected {int(trade_mask.sum())} trade-like rows (of {len(df_all)})")

    # Filter to trade-like rows and RECOMPUTE aligned series
    df = df_all[trade_mask].copy()
    action = df[actioncol].astype(str)
    desc   = df[desccol].astype(str) if desccol else pd.Series([""]*len(df), index=df.index)
    ttype_col = df[typecol].astype(str) if typecol else pd.Series([""]*len(df), index=df.index)
    when  = to_dt(df[datecol])

    # Symbol extraction aligned to filtered df
    if symcol:
        sym = df[symcol].map(base_symbol_from_string)
    else:
        sym = pd.Series([""] * len(df), index=df.index)

    def symbol_from_text(s: str) -> str:
        m = re.search(r"\(([A-Za-z0-9.\-]{1,8})\)", str(s))
        return m.group(1).upper() if m else ""

    sym_from_action = action.map(symbol_from_text)
    sym_from_desc   = desc.map(symbol_from_text)

    sym = sym.fillna("").astype(str)
    sym = np.where(sym == "", sym_from_action, sym)
    sym = np.where(sym == "", sym_from_desc,   sym)
    sym = pd.Series(sym, index=df.index).map(base_symbol_from_string)

    # BUY/SELL detection — compute on filtered series ONLY (fixes length mismatch)
    up = (action + " " + ttype_col).str.upper()
    is_buy  = up.str.contains(r"\b(?:BOUGHT|BUY)\b",  regex=True, na=False)
    is_sell = up.str.contains(r"\b(?:SOLD|SELL)\b",   regex=True, na=False)
    ttype = pd.Series(np.where(is_sell, "SELL", np.where(is_buy, "BUY", "BUY")), index=df.index)

    # Qty & Price
    qty   = to_float(df[qtycol])   if qtycol   else pd.Series(np.nan, index=df.index)
    price = to_float(df[pricecol]) if pricecol else pd.Series(np.nan, index=df.index)

    qty = pd.to_numeric(qty, errors="coerce")
    neg_sell = qty < 0
    ttype = pd.Series(np.where(neg_sell, "SELL", ttype), index=df.index)
    qty = qty.abs()

    tx = pd.DataFrame({
        "When":  when,
        "Type":  ttype,
        "Symbol": sym,
        "Qty":   qty,
        "Price": price,
    })

    tx = tx[tx["Symbol"].ne("") & tx["When"].notna() & (tx["Qty"] > 0)].copy()
    tx.sort_values(["When", "Symbol"], inplace=True)
    tx.reset_index(drop=True, inplace=True)

    if debug and not tx.empty:
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        print(tx.head(8).to_string(index=False))

    return tx, pd.DataFrame(columns=["When","Symbol","Qty","Price","Reason"])

# ─────────────────────────────
# MATCH SIGNALS → BUYS; FIFO CLOSE WITH SELLS
# ─────────────────────────────
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sig_buy = sig[(sig["Direction"].astype(str).str.upper()=="BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True)

    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append( (r["TimestampUTC"], {
            "Source": str(r.get("Source","")),
            "Timeframe": str(r.get("Timeframe","")),
            "SigTime": r.get("TimestampUTC"),
            "SigPrice": r.get("Price",""),
        }))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                return payload
            if t <= when:
                return payload
        return {"Source":"(unknown)", "Timeframe":"", "SigTime": pd.NaT, "SigPrice": ""}

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []
    unmatched_rows = []

    for _, row in tx.iterrows():
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = float(row["Qty"]) if not math.isnan(row["Qty"]) else 0.0
        price = float(row["Price"]) if not math.isnan(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr == "":
            continue

        if ttype == "BUY":
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
        else:  # SELL
            remaining = qty
            total_avail = sum(l["qty_left"] for l in lots[tkr])
            if total_avail <= 0:
                unmatched_rows.append({
                    "When": when, "Symbol": tkr, "Qty": qty, "Price": price,
                    "Reason": "No prior BUY lot in window"
                })
            while remaining > 0 and lots[tkr]:
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

            if remaining > 0:
                unmatched_rows.append({
                    "When": when, "Symbol": tkr, "Qty": remaining, "Price": price,
                    "Reason": "SELL exceeds available BUY lots"
                })

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df.sort_values("ExitTimeUTC", inplace=True, ignore_index=True)

    now_utc = ensure_utc_now()
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

    unmatched_df = pd.DataFrame(unmatched_rows).sort_values("When", ignore_index=True) if unmatched_rows else pd.DataFrame(columns=["When","Symbol","Qty","Price","Reason"])

    if debug and not unmatched_df.empty:
        print("⚠️ Unmatched SELLs (no prior BUYs or partial over-sell):")
        for _, r in unmatched_df.iterrows():
            w = pd.to_datetime(r["When"]).isoformat()
            print(f"  • {r['Symbol']} SELL on {w} qty={r['Qty']} price={r['Price']} — {r['Reason']}")
        print(f"⚠️ Total unmatched SELL events: {len(unmatched_df)}")

    return realized_df, open_df, unmatched_df

# ─────────────────────────────
# POST-PROCESS: live price + perf
# ─────────────────────────────
def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str,Dict[str,str]]) -> pd.DataFrame:
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now = []
    unreal = []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        row_index = idx + 2  # header is row 1
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
    global DEBUG
    ap = argparse.ArgumentParser(description="Build Performance dashboard tabs.")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    args = ap.parse_args()
    DEBUG = args.debug

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
    tx, _unmatched_placeholder = load_transactions(df_tx, debug=DEBUG)
    _hold = load_holdings(df_h)

    realized_df, open_df, unmatched_df = build_realized_and_open(tx, sig, debug=DEBUG)

    mapping = read_mapping(gc)
    open_df = add_live_price_formulas(open_df, mapping)

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

    ws_real = open_ws(gc, TAB_REALIZED)
    ws_open = open_ws(gc, TAB_OPEN)
    ws_perf = open_ws(gc, TAB_PERF)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_df)

    print(f"✅ Wrote {TAB_REALIZED}: {len(realized_df)} rows")
    print(f"✅ Wrote {TAB_OPEN}: {len(open_df)} rows")
    print(f"✅ Wrote {TAB_PERF}: {len(perf_df)} rows")

    if not unmatched_df.empty:
        if not DEBUG:
            print(f"⚠️ Summary: {len(unmatched_df)} unmatched SELL events (use --debug for details).")

    print("🎯 Done.")

if __name__ == "__main__":
    main()
