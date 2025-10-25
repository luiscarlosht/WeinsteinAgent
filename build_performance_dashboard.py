#!/usr/bin/env python3
"""
Builds 3 tabs in your Google Sheet:
- Realized_Trades
- Open_Positions
- Performance_By_Source

Now includes diagnostics:
- Logs any SELL rows that could not be matched to prior BUY lots (per-ticker FIFO).
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
TAB_HOLDINGS     = "Holdings"          # optional (not required for FIFO)
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
    # Strip strings (applymap is deprecated)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def to_dt(series: pd.Series) -> pd.Series:
    # Parse and keep as timezone-aware UTC where possible
    return pd.to_datetime(series, errors="coerce", utc=True)

def to_float(series: pd.Series) -> pd.Series:
    def conv(x):
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
            x = x.replace("(", "-").replace(")", "")  # handle (84.66)
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
    token = s.split()[0]                 # take first chunk (before spaces)
    token = token.split("-")[0]          # strip option/lot suffix like "AAPL-12345"
    token = token.replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token).upper()
    if not token:
        return ""
    if token in BLACKLIST_TOKENS:
        return ""
    if token.isdigit():
        return ""
    # very long account-like strings -> ignore
    if len(token) > 8 and token.isalnum():
        return ""
    return token

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        # New gspread signature: values first, then range_name=
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
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        ws.update(block, range_name=f"{top_left}:{bottom_right}")
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

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for want in candidates:
        for k, orig in cols.items():
            if want == k:
                return orig
    return None

def load_transactions(df_tx: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    """
    Build a clean trades dataframe with:
      When (UTC), Type (upper string), Symbol, Qty (>0), Price
    BUY rows add lots; SELL/SOLD close FIFO.
    """
    if df_tx.empty:
        return df_tx

    # Typical Fidelity headings
    symcol  = _find_col(df_tx, ["symbol", "security", "symbol/cusip"])
    actionc = _find_col(df_tx, ["action"])
    descc   = _find_col(df_tx, ["description"])
    typecol = _find_col(df_tx, ["type"])  # sometimes "Cash"
    qtycol  = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    pricec  = next((c for c in df_tx.columns if "price" in c.lower()), None)
    amtc    = next((c for c in df_tx.columns if "amount" in c.lower()), None)
    datec   = _find_col(df_tx, ["run date"])
    if not datec:
        # fall back to any date-like column
        datec = next((c for c in df_tx.columns if "date" in c.lower()), None)
    if not datec:
        raise ValueError("Transactions tab must include a date column (e.g., 'Run Date').")
    if not (actionc or descc or typecol):
        raise ValueError("Transactions tab needs at least Action or Description or Type.")

    df = df_tx.copy()

    # Parse time
    when = to_dt(df[datec])

    # Uppercase text fields safely
    action = df[actionc].astype(str) if actionc else pd.Series("", index=df.index)
    desc   = df[descc].astype(str) if descc else pd.Series("", index=df.index)
    tp     = df[typecol].astype(str) if typecol else pd.Series("", index=df.index)
    action_up = action.str.upper()
    desc_up   = desc.str.upper()
    type_up   = tp.str.upper()

    # Detect trade-like rows (BUY/SELL) from either Action or Description
    patt = r"\bYOU\s+BOUGHT\b|\bYOU\s+SOLD\b|\b\bBUY\b|\bSELL\b|\bBOUGHT\b|\bSOLD\b"
    mask_action = action_up.str.contains(patt, regex=True, na=False)
    mask_desc   = desc_up.str_contains(patt, regex=True, na=False) if hasattr(desc_up, "str_contains") else desc_up.str.contains(patt, regex=True, na=False)
    trade_mask  = mask_action | mask_desc

    if debug:
        total = len(df)
        detected = int(trade_mask.sum())
        print(f"• load_transactions: detected {detected} trade-like rows (of {total})")

    # Build symbol column: prefer Symbol; if blank, try to extract from Action
    sym = df[symcol].fillna("").map(base_symbol_from_string) if symcol else pd.Series("", index=df.index)

    def symbol_from_text(s: str) -> str:
        # e.g. "YOU SOLD NEXTRACKER INC CLASS A COM (NXT) (Cash)" -> NXT
        m = re.search(r"\(([A-Za-z0-9\.\-]+)\)", s or "", flags=re.IGNORECASE)
        if m:
            return base_symbol_from_string(m.group(1))
        return ""

    sym_fallback = action.map(symbol_from_text)
    sym = np.where(sym.astype(str).eq(""), sym_fallback, sym)
    sym = pd.Series(sym, index=df.index).map(base_symbol_from_string)

    # Quantity: if quantity column exists, use abs(); otherwise derive from amount/price
    qty = pd.Series(np.nan, index=df.index)
    if qtycol:
        qty = to_float(df[qtycol]).abs()
    else:
        if amtc and pricec:
            amt = to_float(df[amtc])
            prc = to_float(df[pricec])
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where((prc != 0) & (~np.isnan(prc)) & (~np.isnan(amt)), np.abs(amt) / np.abs(prc), np.nan)
            qty = pd.Series(q, index=df.index)

    price = to_float(df[pricec]) if pricec else pd.Series(np.nan, index=df.index)

    # BUY/SELL classification (text, not sign)
    is_buy  = trade_mask & (action_up.str.contains("BOUGHT|BUY", regex=True) | desc_up.str.contains("BOUGHT|BUY", regex=True) | type_up.str.contains("BUY", regex=False))
    is_sell = trade_mask & (action_up.str.contains("SOLD|SELL", regex=True)  | desc_up.str.contains("SOLD|SELL", regex=True)  | type_up.str.contains("SELL", regex=False))

    # Build tx frame
    tx = pd.DataFrame({
        "When": when,
        "Type": np.select([is_buy, is_sell], ["BUY", "SELL"], default=""),
        "Symbol": sym,
        "Qty": pd.to_numeric(qty, errors="coerce"),
        "Price": pd.to_numeric(price, errors="coerce"),
    })

    # Filter valid rows: must be BUY or SELL, have symbol, have time, have positive qty
    tx = tx[(tx["Type"].isin(["BUY","SELL"])) & tx["Symbol"].ne("") & tx["When"].notna() & (tx["Qty"] > 0)].copy()
    tx.sort_values("When", inplace=True)
    tx.reset_index(drop=True, inplace=True)

    if debug:
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        try:
            print(tx.head(8).to_string(index=False))
        except Exception:
            print(tx.head(8))

    return tx

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty:
        return df_h
    symcol  = _find_col(df_h, ["symbol", "security", "symbol/cusip"])
    qtycol  = next((c for c in df_h.columns if "quantity" in c.lower()), None)
    pricec  = next((c for c in df_h.columns if "price" in c.lower()), None)
    df = pd.DataFrame()
    if symcol:   df["Ticker"] = df_h[symcol].map(base_symbol_from_string)
    if qtycol:   df["Qty"]    = to_float(df_h[qtycol])
    if pricec:   df["Price"]  = to_float(df_h[pricec])
    return df

# ─────────────────────────────
# MATCH SIGNALS → FIFO CLOSES
# ─────────────────────────────
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """Return realized_df, open_df, unmatched_sells(list of dicts for diagnostics)."""
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Index BUY signals by (Ticker, time), so we can attach Source/Timeframe when buys happen
    sig_buy = sig[(sig["Direction"].astype(str).str.upper()=="BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True)

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
    unmatched_sells = []  # each item: {"Ticker","When","Qty","Price","Reason"}

    for _, row in tx.iterrows():
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = row["Qty"] if not math.isnan(row["Qty"]) else 0.0
        price = row["Price"] if not math.isnan(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr=="":
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
        elif ttype == "SELL":
            remaining = qty
            if not lots[tkr]:
                unmatched_sells.append({
                    "Ticker": tkr,
                    "When": when,
                    "Qty": qty,
                    "Price": price,
                    "Reason": "No prior BUY lot in window"
                })
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
            if remaining > 1e-9:
                unmatched_sells.append({
                    "Ticker": tkr,
                    "When": when,
                    "Qty": remaining,
                    "Price": price,
                    "Reason": "SELL exceeds available BUY lots"
                })

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

    if debug and unmatched_sells:
        print("⚠️ Unmatched SELLs (no prior BUYs or partial over-sell):")
        for u in unmatched_sells:
            t = u["When"]
            ts = t.isoformat() if isinstance(t, pd.Timestamp) else str(t)
            print(f"  • {u['Ticker']} SELL on {ts} qty={u['Qty']} price={u['Price']} — {u['Reason']}")
        print(f"⚠️ Total unmatched SELL events: {len(unmatched_sells)}")

    return realized_df, open_df, unmatched_sells

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
        open_counts = open_df.groupby("Source", dropna=False).size().rename("OpenSignals").reset_index()

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
    parser = argparse.ArgumentParser(description="Build Performance dashboard tabs…")
    parser.add_argument("--debug", action="store_true", help="Print debug info & diagnostics")
    args = parser.parse_args()

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
    _   = load_holdings(df_h)  # optional; not required for FIFO matching

    realized_df, open_df, unmatched = build_realized_and_open(tx, sig, debug=args.debug)

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

    if unmatched:
        print(f"⚠️ Summary: {len(unmatched)} unmatched SELL events (use --debug for details).")
    else:
        print("👌 No unmatched SELLs.")

    print("🎯 Done.")

if __name__ == "__main__":
    main()
