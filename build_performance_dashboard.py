#!/usr/bin/env python3
import math, re, sys, datetime as dt
from collections import deque, defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
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
TAB_HOLDINGS     = "Holdings"          # optional, used only for current qty sanity checks
TAB_REALIZED     = "Realized_Trades"
TAB_OPEN         = "Open_Positions"
TAB_PERF         = "Performance_By_Source"

# If a ticker isn’t mapped in Mapping!FormulaSym, we’ll default to NASDAQ:SYMBOL for price
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
    # strip whitespace
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

def base_symbol_from_string(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # take first token, strip option-like tails
    s = s.strip()
    # many Fidelity symbols like "AAPL", or "RTX", or "CASH (3159…)" – handle those:
    s = s.split()[0]
    s = s.split("-")[0]
    s = s.replace("(", "").replace(")", "")
    return re.sub(r"[^A-Za-z0-9\.\-]", "", s).upper()

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.update("A1", [["(empty)"]])
        return
    # size sheet
    rows, cols = df.shape
    ws.resize(rows= max(100, rows+5), cols=max( min(26, cols+2), 8))
    # write header + chunked rows
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()
    # chunk to be safe
    CHUNK = 500
    start = 1
    r = 1
    while start < len(data):
        end = min(start+CHUNK-1, len(data)-1)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        ncols = len(header)
        bottom_right = gspread.utils.rowcol_to_a1(r+(end-start), ncols)
        ws.update(f"{top_left}:{bottom_right}", data[start-1:end+1])
        r += (end - start + 1)
        start = end + 1

def read_mapping(gc) -> Dict[str, Dict[str,str]]:
    """Return {ticker: {'FormulaSym': 'EXCH: TKR', 'TickerYF': 'TKR'}} if Mapping tab exists."""
    try:
        mws = open_ws(gc, "Mapping")
        dfm = read_tab(mws)
        out = {}
        if not dfm.empty and "Ticker" in dfm and ("FormulaSym" in dfm or "TickerYF" in dfm):
            for _, row in dfm.iterrows():
                t = row.get("Ticker","").strip().upper()
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
    # Put a fallback to using the raw B{row} cell too
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_idx},"price"), ""))'


# ─────────────────────────────
# LOAD SHEET DATA
# ─────────────────────────────
def load_signals(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig.empty:
        return df_sig
    colmap = {c.lower(): c for c in df_sig.columns}
    # expected headers
    tcol = next((c for c in df_sig.columns if c.lower() in ("ticker","symbol")), None)
    if not tcol: raise ValueError("Signals tab needs a 'Ticker' column.")
    df = df_sig.copy()
    df["Ticker"]      = df[tcol].map(base_symbol_from_string)
    df["Source"]      = df.get("Source","")
    df["Direction"]   = df.get("Direction","")
    df["Timeframe"]   = df.get("Timeframe","")
    # parse timestamp if present
    tscol = next((c for c in df.columns if c.lower().startswith("timestamp")), None)
    df["TimestampUTC"] = to_dt(df[tscol]) if tscol else pd.NaT
    # price may be formula string; we won’t coerce here
    df["Price"] = df.get("Price","")
    # keep only relevant
    return df[["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"]]

def load_transactions(df_tx: pd.DataFrame) -> pd.DataFrame:
    if df_tx.empty: return df_tx
    # Heuristic column names from Fidelity CSV
    symcol = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    typecol = next((c for c in df_tx.columns if c.lower() in ("type","action")), None)
    pricecol = next((c for c in df_tx.columns if "price" in c.lower()), None)
    amtcol = next((c for c in df_tx.columns if "amount" in c.lower()), None)
    qtycol = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    datecol = next((c for c in df_tx.columns if "run date" in c.lower()), None)
    if not symcol or not typecol or not datecol:
        raise ValueError("Transactions tab must include Symbol, Type/Action and Run Date columns.")

    tx = pd.DataFrame()
    tx["When"]   = to_dt(df_tx[datecol])
    tx["Type"]   = df_tx[typecol].str.upper()
    tx["Symbol"] = df_tx[symcol].map(base_symbol_from_string)
    if qtycol:
        tx["Qty"] = to_float(df_tx[qtycol])
    else:
        # derive qty from Amount/Price if both exist
        if amtcol and pricecol:
            amt = to_float(df_tx[amtcol])
            prc = to_float(df_tx[pricecol])
            with np.errstate(divide='ignore', invalid='ignore'):
                q = np.where((prc!=0) & (~np.isnan(prc)) & (~np.isnan(amt)), np.abs(amt)/np.abs(prc), np.nan)
            tx["Qty"] = q
        else:
            tx["Qty"] = np.nan
    tx["Price"] = to_float(df_tx[pricecol]) if pricecol else np.nan

    # keep only buys/sells with symbol
    m = tx["Type"].fillna("").str.contains("BUY|SOLD|SELL", regex=True)
    tx = tx[m & tx["Symbol"].ne("")].copy()
    tx.sort_values("When", inplace=True)
    tx.reset_index(drop=True, inplace=True)
    return tx

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty: return df_h
    symcol = next((c for c in df_h.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol = next((c for c in df_h.columns if "quantity" in c.lower()), None)
    pricecol = next((c for c in df_h.columns if "price" in c.lower()), None)
    df = pd.DataFrame()
    if symcol:   df["Ticker"] = df_h[symcol].map(base_symbol_from_string)
    if qtycol:   df["Qty"]    = to_float(df_h[qtycol])
    if pricecol: df["Price"]  = to_float(df_h[pricecol])
    return df


# ─────────────────────────────
# MATCH SIGNALS → BUYS, THEN FIFO CLOSE WITH SELLS
# ─────────────────────────────
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each BUY fill, attach the most recent BUY signal for that ticker at or before fill time.
    Then FIFO match SELLs to open BUY lots and compute realized P/L.
    Returns (realized_trades_df, open_lots_df).
    """
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Pre-index signals by ticker for quick lookup
    sig_buy = sig[(sig["Direction"].str.upper()=="BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True)

    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append( (r["TimestampUTC"], {
            "Source": r["Source"], "Timeframe": r["Timeframe"], "SigTime": r["TimestampUTC"], "SigPrice": r["Price"]
        }))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        # linear scan from end (lists small per ticker)
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                # if we have no times, just take the last known
                return payload
            if t <= when:
                return payload
        return {"Source":"(unknown)","Timeframe":"","SigTime": pd.NaT, "SigPrice": ""}

    # Build FIFO queues per symbol
    lots: Dict[str, deque] = defaultdict(deque)  # each item: dict with qty_left, entry_price, entry_time, source, timeframe
    realized_rows = []

    for _, row in tx.iterrows():
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = row["Qty"] if not math.isnan(row["Qty"]) else 0.0
        price = row["Price"] if not math.isnan(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr=="":
            continue

        if "BUY" in ttype and not "SELL" in ttype:
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
        elif "SELL" in ttype or "SOLD" in ttype:
            remaining = qty
            while remaining > 0 and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                if take <= 0: break
                # realized P/L on the portion
                entry = lot["entry_price"] if not math.isnan(lot["entry_price"]) else 0.0
                exitp = price if not math.isnan(price) else 0.0
                ret_pct = ((exitp - entry) / entry * 100.0) if entry else np.nan
                held_days = (when - lot["entry_time"]).days if (not pd.isna(lot["entry_time"]) and not pd.isna(when)) else None
                realized_rows.append({
                    "Ticker": tkr,
                    "Qty": round(take, 6),
                    "EntryPrice": round(entry, 6) if entry else "",
                    "ExitPrice": round(exitp, 6) if exitp else "",
                    "Return%": round(ret_pct, 4) if not np.isnan(ret_pct) else "",
                    "HoldDays": held_days if held_days is not None else "",
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
            # If sells exceed buys, leftover ignored (could log)

    realized_df = pd.DataFrame(realized_rows)
    realized_df.sort_values("ExitTimeUTC", inplace=True, ignore_index=True)

    # Remaining open lots → Open_Positions
    open_rows = []
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
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
    open_df.sort_values("EntryTimeUTC", inplace=True, ignore_index=True)
    return realized_df, open_df


def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str,Dict[str,str]]) -> pd.DataFrame:
    if open_df.empty:
        return open_df
    out = open_df.copy()
    # Create PriceNow and Unrealized% columns with GOOGLEFINANCE
    price_now = []
    unreal = []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        row_index = idx + 2  # + header row
        formula = googlefinance_formula_for(tkr, row_index, mapping)
        price_now.append(formula)
        try:
            epf = float(ep)
            if epf > 0:
                unreal.append(f'=IFERROR( ( {formula} / {epf} - 1 ) * 100 , "")')
            else:
                unreal.append("")
        except Exception:
            unreal.append("")
    out.insert(out.columns.get_loc("EntryPrice")+1, "PriceNow", price_now)
    out.insert(out.columns.get_loc("PriceNow")+1, "Unrealized%", unreal)
    return out


def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate realized returns by Source
    if realized_df.empty:
        realized_grp = pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%"])
    else:
        tmp = realized_df.copy()
        tmp["is_win"] = pd.to_numeric(tmp["Return%"], errors="coerce").fillna(0) > 0
        tmp["ret"]    = pd.to_numeric(tmp["Return%"], errors="coerce")
        g = tmp.groupby("Source", dropna=False)
        realized_grp = pd.DataFrame({
            "Source": g.size().index,
            "Trades": g.size().values,
            "Wins": g["is_win"].sum().values,
            "WinRate%": (g["is_win"].mean().fillna(0.0).values*100).round(2),
            "AvgReturn%": g["ret"].mean().round(2).values,
            "MedianReturn%": g["ret"].median().round(2).values,
        })

    # Count open signals per source (from Signals tab)
    if signals_df.empty:
        open_counts = pd.DataFrame(columns=["Source","OpenSignals"])
    else:
        # Count BUY signals that do not have a fully closed position yet.
        # Approximation: count latest BUY per ticker/source as "open" if there is any open lot with same ticker+source.
        if open_df.empty:
            open_counts = pd.DataFrame({"Source": signals_df["Source"].unique(), "OpenSignals":[0]*signals_df["Source"].nunique()})
        else:
            opens_by_source = open_df.groupby("Source").size().rename("OpenSignals").reset_index()
            open_counts = opens_by_source

    if realized_df.empty and (signals_df.empty or open_df.empty):
        perf = realized_grp.copy()
        perf["OpenSignals"] = 0
        return perf

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer").fillinplace if False else pd.merge(realized_grp, open_counts, on="Source", how="outer")
    perf["Trades"] = perf["Trades"].fillna(0).astype(int)
    perf["Wins"] = perf["Wins"].fillna(0).astype(int)
    perf["WinRate%"] = perf["WinRate%"].fillna(0)
    perf["AvgReturn%"] = perf["AvgReturn%"].fillna(0)
    perf["MedianReturn%"] = perf["MedianReturn%"].fillna(0)
    perf["OpenSignals"] = perf["OpenSignals"].fillna(0).astype(int)
    perf = perf.sort_values(["Source"]).resetear_index if False else perf.sort_values(["Source"]).reset_index(drop=True)
    return perf


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
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
    tx  = load_transactions(df_tx)
    holdings = load_holdings(df_h)  # optional

    realized_df, open_df = build_realized_and_open(tx, sig)

    # Add PriceNow/Unrealized% formulas for open lots
    mapping = read_mapping(gc)
    open_df = add_live_price_formulas(open_df, mapping)

    # Pretty columns
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

    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy(), sig)

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
    print("🎯 Done.")

if __name__ == "__main__":
    main()
