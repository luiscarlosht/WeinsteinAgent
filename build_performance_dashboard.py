#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds three tabs in your Google Sheet:
  • Realized_Trades
  • Open_Positions
  • Performance_By_Source

Key improvements in this version:
  • Robust BUY/SELL parsing from Fidelity CSVs (Action/Description + Quantity sign).
  • Ignores "floating-point dust" quantities (e.g., 3.33e-16) via EPSILON_QTY.
  • Optional seeding of prior positions from Holdings so old SELLs match: --seed-from-holdings
  • Live price formulas for open lots (uses Mapping tab's FormulaSym when present).
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
TAB_MAPPING      = "Mapping"

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "

# Matching/cleanup controls
EPSILON_QTY = 1e-6       # drop / ignore quantities smaller than this
ROUND_QTY   = 6          # rounding for quantities to stabilize math

# ─────────────────────────────
# AUTH / SHEET HELPERS
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
        # Use new gspread signature: values first, then range_name
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    # Resize generously
    ws.resize(rows=max(rows + 5, 100), cols=max(cols + 2, 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

    CHUNK = 500
    start = 0
    r = 1
    while start < len(data):
        end = min(start + CHUNK, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)
        r += len(block)
        start = end

# ─────────────────────────────
# UTILITIES
# ─────────────────────────────
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
    """Extract an equity/ETF symbol; '' for cash/blank/other."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    token = s.split()[0]                 # first chunk
    token = token.split("-")[0]          # drop option/lot suffix
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

def ensure_utc_now():
    # Pandas returns tz-aware now; keep it UTC
    return pd.Timestamp.utcnow().tz_convert("UTC")

# ─────────────────────────────
# OPTIONAL MAPPING TAB
# ─────────────────────────────
def read_mapping(gc) -> Dict[str, Dict[str,str]]:
    """Return {ticker: {'FormulaSym': 'EXCH: TKR', 'TickerYF': 'TKR'}} if Mapping tab exists."""
    try:
        mws = open_ws(gc, TAB_MAPPING)
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
    # Second fallback references column B row_idx (Ticker cell)
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

def _symbol_from_action_or_desc(action_up: pd.Series, desc_up: pd.Series) -> pd.Series:
    """
    Extract symbol from Action/Description strings of form:
      "YOU BOUGHT XYZ CORPORATION (XYZ) (Cash)"
    """
    pat = r"\(([A-Z0-9\.\-]{1,8})\)"
    # Prefer Action first, then Description
    sym_a = action_up.str.extract(pat, expand=False)
    sym_d = desc_up.str.extract(pat, expand=False)
    sym = sym_a.fillna(sym_d)
    return sym.fillna("")

def load_transactions(df_tx: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (tx_df, unmatched_df placeholder (empty here; real unmatched produced later)).
    tx_df columns: When | Type ("BUY"/"SELL") | Symbol | Qty | Price
    """
    if df_tx.empty:
        return pd.DataFrame(columns=["When","Type","Symbol","Qty","Price"]), pd.DataFrame()

    df = df_tx.copy()

    # Fuzzy locate columns
    datecol = next((c for c in df.columns if "run date" in c.lower() or c.lower()=="date"), None)
    actioncol = next((c for c in df.columns if c.lower() in ("action","type")), None)
    symcol  = next((c for c in df.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    desccol = next((c for c in df.columns if c.lower().startswith("description")), None)
    qtycol  = next((c for c in df.columns if "quantity" in c.lower()), None)
    pricecol= next((c for c in df.columns if "price" in c.lower()), None)

    if not datecol or not actioncol:
        raise ValueError("Transactions must include a date column (e.g., 'Run Date') and an Action/Type column.")

    when = to_dt(df[datecol])
    action = df[actioncol].fillna("").astype(str)
    desc   = df[desccol].fillna("").astype(str) if desccol else pd.Series("", index=df.index)
    action_up = action.str.upper()
    desc_up   = desc.str.upper()

    # Trade-like rows: detect BUY/SELL in either action/description
    patt = r"\b(YOU\s+)?(BOUGHT|SOLD|BUY|SELL)\b"
    mask_action = action_up.str.contains(patt, regex=True, na=False)
    mask_desc   = desc_up.str.contains(patt,   regex=True, na=False)
    mask_trade  = mask_action | mask_desc
    detected_trades = int(mask_trade.sum())
    total_rows = len(df)
    print(f"• load_transactions: detected {detected_trades} trade-like rows (of {total_rows})")

    df = df[mask_trade].copy()

    # Symbol: prefer symbol column; if blank, extract (TICKER) from Action/Description
    if symcol:
        sym = df[symcol].map(base_symbol_from_string)
    else:
        sym = pd.Series("", index=df.index)

    sym = sym.fillna("")
    sym_fallback = _symbol_from_action_or_desc(action_up.loc[df.index], desc_up.loc[df.index])
    sym = np.where(sym == "", sym_fallback, sym)
    sym = pd.Series(sym, index=df.index).map(base_symbol_from_string)

    # Qty & Price
    qty   = to_float(df[qtycol])   if qtycol   else pd.Series(np.nan, index=df.index)
    price = to_float(df[pricecol]) if pricecol else pd.Series(np.nan, index=df.index)
    qty = pd.to_numeric(qty, errors="coerce")

    # Fidelity: negative quantity rows are SELLS; normalize
    neg_sell = qty < 0
    ttype = np.where(neg_sell, "SELL", "BUY")
    qty = qty.abs()

    # Round and drop floating dust
    qty = pd.Series(qty, index=df.index).round(ROUND_QTY)
    qty[np.abs(qty) < EPSILON_QTY] = 0.0

    tx = pd.DataFrame({
        "When":  when.loc[df.index],
        "Type":  pd.Series(ttype, index=df.index),
        "Symbol": pd.Series(sym, index=df.index),
        "Qty":   qty,
        "Price": price.loc[df.index],
    })

    # Keep valid rows only
    tx = tx[tx["Symbol"].ne("") & tx["When"].notna() & (tx["Qty"] > 0)].copy()
    tx.sort_values(["When"], inplace=True)
    tx.reset_index(drop=True, inplace=True)

    if debug:
        print(f"• load_transactions: after cleaning → {len(tx)} trades")
        try:
            print(tx.head(8).to_string(index=False))
        except Exception:
            pass

    return tx, pd.DataFrame()

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty: 
        return pd.DataFrame(columns=["Ticker","Qty","Price"])
    symcol  = next((c for c in df_h.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol  = next((c for c in df_h.columns if "quantity" in c.lower()), None)
    pricecol= next((c for c in df_h.columns if "price" in c.lower()), None)
    out = pd.DataFrame()
    if symcol:   out["Ticker"] = df_h[symcol].map(base_symbol_from_string)
    if qtycol:   out["Qty"]    = to_float(df_h[qtycol])
    if pricecol: out["Price"]  = to_float(df_h[pricecol])
    out = out[(out["Ticker"].ne("")) & (out["Qty"].fillna(0) > 0)].copy()
    return out

# ─────────────────────────────
# MATCH SIGNALS → BUYS, FIFO CLOSE WITH SELLS
# ─────────────────────────────
def build_realized_and_open(tx: pd.DataFrame,
                            sig: pd.DataFrame,
                            debug: bool=False,
                            seed_lots: Dict[str, deque] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        # from end to start to get most recent at/<= when
        for t, payload in reversed(arr):
            # If either timestamp is NaT, just return payload
            if pd.isna(t) or pd.isna(when):
                return payload
            if t <= when:
                return payload
        return {"Source":"(unknown)","Timeframe":"","SigTime": pd.NaT, "SigPrice": ""}

    lots: Dict[str, deque] = defaultdict(deque)

    # preload seeded lots (from Holdings) if provided
    if seed_lots:
        for tkr, dq in seed_lots.items():
            for lot in dq:
                lots[tkr].append(lot)

    realized_rows = []
    unmatched_rows = []

    for _, row in tx.iterrows():
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = float(row["Qty"]) if not pd.isna(row["Qty"]) else 0.0
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
            # consume from FIFO lots
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
                    "Qty": round(take, ROUND_QTY),
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

                # snap tiny values to zero so we don't carry dust
                if abs(lot["qty_left"]) < EPSILON_QTY:
                    lot["qty_left"] = 0.0
                if abs(remaining) < EPSILON_QTY:
                    remaining = 0.0

                if lot["qty_left"] <= 0:
                    lots[tkr].popleft()

            # residual SELL with no buys left → unmatched
            if remaining > EPSILON_QTY:
                unmatched_rows.append({
                    "When": when, "Symbol": tkr, "Qty": round(remaining, ROUND_QTY), "Price": price,
                    "Reason": "SELL exceeds available BUY lots"
                })

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df.sort_values("ExitTimeUTC", inplace=True, ignore_index=True)

    # Remaining open lots
    now_utc = ensure_utc_now()
    open_rows = []
    for tkr, q in lots.items():
        for lot in q:
            if lot["qty_left"] <= EPSILON_QTY:
                continue
            open_rows.append({
                "Ticker": tkr,
                "OpenQty": round(lot["qty_left"], ROUND_QTY),
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

    unmatched_df = pd.DataFrame(unmatched_rows)
    return realized_df, open_df, unmatched_df

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
    ap = argparse.ArgumentParser(description="Build performance dashboard tabs.")
    ap.add_argument("--debug", action="boolean_optional", default=False, help="Verbose debug output")
    ap.add_argument("--seed-from-holdings", action="store_true",
                    help="Treat current Holdings as starting BUY lots (useful when selling older positions).")
    args = ap.parse_args()
    DEBUG = bool(args.debug)

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
    hold_df = load_holdings(df_h)

    # Optional: seed prior positions from Holdings to avoid unmatched SELLs for older buys
    seed = None
    if args.seed_from_holdings and not hold_df.empty:
        earliest = tx["When"].min() if not tx.empty else ensure_utc_now()
        seed_time = earliest - pd.Timedelta(days=1)
        seed = defaultdict(deque)
        for _, r in hold_df.iterrows():
            tkr = str(r.get("Ticker","")).strip().upper()
            q   = r.get("Qty", np.nan)
            p   = r.get("Price", np.nan)
            if not tkr or pd.isna(q) or q <= 0:
                continue
            seed[tkr].append({
                "qty_left": float(q),
                "entry_price": float(p) if not pd.isna(p) else np.nan,
                "entry_time": seed_time,
                "source": "(seeded from Holdings)",
                "timeframe": "",
                "sig_time": pd.NaT,
                "sig_price": "",
            })

    realized_df, open_df, unmatched_df = build_realized_and_open(tx, sig, debug=DEBUG, seed_lots=seed)

    if DEBUG:
        if unmatched_df.empty:
            print("• realized trades:", len(realized_df), "| open lots:", len(open_df))
        else:
            print("⚠️ Unmatched SELLs (no prior BUYs or partial over-sell):")
            for _, r in unmatched_df.iterrows():
                print(f"  • {r['Symbol']} SELL on {r['When'].isoformat()} qty={r['Qty']} price={r['Price']} — {r['Reason']}")
            print(f"⚠️ Total unmatched SELL events: {len(unmatched_df)}")

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
    if not unmatched_df.empty and not args.seed_from_holdings:
        print(f"⚠️ Summary: {len(unmatched_df)} unmatched SELL events (add --seed-from-holdings to seed prior lots).")
    print("🎯 Done.")

if __name__ == "__main__":
    main()
