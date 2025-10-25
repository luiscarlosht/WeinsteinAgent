#!/usr/bin/env python3
import math, re, argparse
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # strip strings
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

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.update(values=[["(empty)"]], range_name="A1")
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
        ws.update(range_name=f"{top_left}:{bottom_right}", values=block)
        r += len(block)
        start = end

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

def load_transactions(df_tx: pd.DataFrame, debug=False) -> pd.DataFrame:
    """
    Parse Fidelity 'Transactions' export to trades DataFrame with cols:
    When, Type, Symbol, Qty, Price
    """
    if df_tx.empty:
        return df_tx

    # likely Fidelity column names
    symcol   = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    actioncol= next((c for c in df_tx.columns if c.lower() in ("action","type")), None)
    pricecol = next((c for c in df_tx.columns if "price" in c.lower()), None)
    qtycol   = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    amtcol   = next((c for c in df_tx.columns if "amount" in c.lower()), None)
    desc_col = next((c for c in df_tx.columns if "description" in c.lower()), None)
    datecol  = next((c for c in df_tx.columns if "run date" in c.lower()), None)
    if not actioncol or not datecol:
        raise ValueError("Transactions tab must include Action/Type and Run Date columns.")

    df = df_tx.copy()
    when = to_dt(df[datecol])
    action_raw = df[actioncol].fillna("").astype(str)
    desc_raw   = df[desc_col].fillna("").astype(str) if desc_col else pd.Series([""]*len(df))

    # detect trade-like rows
    patt = r"\b(?:YOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"
    mask_action = action_raw.str.upper().str.contains(patt, regex=True, na=False)
    mask_desc   = desc_raw.str.upper().str.contains(patt,   regex=True, na=False)
    trade_mask  = mask_action | mask_desc

    # symbol: prefer Symbol column; if blank, try to extract from Action/Description "(AAPL)"
    if symcol:
        sym = df[symcol].map(base_symbol_from_string)
    else:
        sym = pd.Series([""]*len(df))

    def symbol_from_action(s: str) -> str:
        m = re.search(r"\(([A-Za-z0-9\.\-]{1,8})\)", s or "")
        return base_symbol_from_string(m.group(1)) if m else ""

    # fill blanks from action/description
    sym_filled = sym.copy()
    need_sym = sym_filled.eq("")
    sym_filled.loc[need_sym] = action_raw[need_sym].map(symbol_from_action)
    need_sym = sym_filled.eq("")
    sym_filled.loc[need_sym] = desc_raw[need_sym].map(symbol_from_action)

    # qty & price
    if qtycol:
        qty = to_float(df[qtycol])
    else:
        if amtcol and pricecol:
            amt = to_float(df[amtcol])
            prc = to_float(df[pricecol])
            with np.errstate(divide='ignore', invalid='ignore'):
                qty = np.where((prc!=0) & (~np.isnan(prc)) & (~np.isnan(amt)), np.abs(amt)/np.abs(prc), np.nan)
            qty = pd.Series(qty, index=df.index)
        else:
            qty = pd.Series(np.nan, index=df.index)
    price = to_float(df[pricecol]) if pricecol else pd.Series(np.nan, index=df.index)

    tx = pd.DataFrame({
        "When": when,
        "Type": action_raw,
        "Symbol": sym_filled,
        "Qty": qty,
        "Price": price,
    })

    # keep only trade-like + symbol + timestamp
    tx = tx[trade_mask & tx["Symbol"].ne("") & tx["When"].notna()].copy()
    tx["Qty"] = pd.to_numeric(tx["Qty"], errors="coerce")
    tx = tx[tx["Qty"] > 0].copy()
    tx.sort_values("When", inplace=True)
    tx.reset_index(drop=True, inplace=True)

    if debug:
        detected = int(trade_mask.sum())
        print(f"• load_transactions: detected {detected} trade-like rows (of {len(df_tx)})")
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        print(tx.head(8).to_string(index=False))

    return tx

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty: return df_h
    symcol  = next((c for c in df_h.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol  = next((c for c in df_h.columns if "quantity" in c.lower()), None)
    pricecol= next((c for c in df_h.columns if "price" in c.lower()), None)
    df = pd.DataFrame()
    if symcol:   df["Ticker"] = df_h[symcol].map(base_symbol_from_string)
    if qtycol:   df["Qty"]    = to_float(df_h[qtycol])
    if pricecol: df["Price"]  = to_float(df_h[pricecol])
    return df

# ─────────────────────────────
# MATCH SIGNALS → BUYS, THEN FIFO CLOSE WITH SELLS
# ─────────────────────────────
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame, debug=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Index BUY signals by (Ticker, time)
    sig_buy = sig[(sig["Direction"].str.upper() == "BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True)

    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append((
            r["TimestampUTC"],
            {
                "Source": r.get("Source",""),
                "Timeframe": r.get("Timeframe",""),
                "SigTime": r.get("TimestampUTC"),
                "SigPrice": r.get("Price",""),
            }
        ))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        # most recent signal at/<= when
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
        ttype = str(row["Type"]).upper()
        qty   = row["Qty"] if not math.isnan(row["Qty"]) else 0.0
        price = row["Price"] if not math.isnan(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr == "":
            continue

        # Proper action detection (covers YOU BOUGHT / YOU SOLD)
        is_buy  = (("BUY"  in ttype) or ("BOUGHT" in ttype)) and not (("SELL" in ttype) or ("SOLD" in ttype))
        is_sell =  ("SELL" in ttype) or ("SOLD"   in ttype)

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
                "Source": lot["source"],
                "Timeframe": lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice": lot["sig_price"],
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df.sort_values("EntryTimeUTC", inplace=True, ignore_index=True)

    if debug:
        print(f"• realized trades: {len(realized_df)} | open lots: {len(open_df)}")

    return realized_df, open_df

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

def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
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
    ap = argparse.ArgumentParser(description="Build performance dashboard tabs from Signals + Transactions.")
    ap.add_argument("--debug", action="store_true", help="Print debug details")
    args = ap.parse_args()
    debug = args.debug

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
    tx  = load_transactions(df_tx, debug=debug)
    _   = load_holdings(df_h)  # optional; currently unused in the build

    realized_df, open_df = build_realized_and_open(tx, sig, debug=debug)

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
