#!/usr/bin/env python3
"""
Builds three tabs in your Google Sheet:

- Realized_Trades
- Open_Positions
- Performance_By_Source

Key behaviors
-------------
• Signals['Source'] is trusted. For each BUY lot, we first try a time-match to the
  most recent signal at/earlier than the trade time (same Ticker). If that fails,
  we still backfill Source from the latest Signals row for that ticker so
  Performance_By_Source doesn't show (unknown).

• Transactions are parsed from your Fidelity export. We look for "Action"/"Type"
  fields with BUY/SELL semantics in "Action" OR "Description", then normalize to
  "BUY"/"SELL". Quantity is made positive; SELLs are matched FIFO to earlier BUYs.

• Optional live price formulas in Open_Positions:
    default: ON (uses GOOGLEFINANCE + Mapping.FormulaSym if present)
    CLI: --no-live  (disables live price columns entirely)

CLI
---
python3 build_performance_dashboard.py [--debug] [--no-live]
"""

import math
import re
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG
# =========================
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

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "  # fallback for GOOGLEFINANCE formulas

# =========================
# AUTH / SHEETS I/O
# =========================
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
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        # New gspread signature: values first, then range_name=...
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows + 5), cols=max(min(26, cols + 2), 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()
    # Upload in chunks (values first)
    CHUNK = 500
    start = 0
    r = 1
    while start < len(data):
        end = min(start + CHUNK, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        ws.update(block, range_name=f"{top_left}:{bottom_right}")
        r += len(block)
        start = end

# =========================
# UTILITIES
# =========================
BLACKLIST_TOKENS = {
    "CASH", "USD", "INTEREST", "DIVIDEND", "REINVESTMENT", "FEE",
    "WITHDRAWAL", "DEPOSIT", "TRANSFER", "SWEEP"
}

def base_symbol_from_string(s) -> str:
    """Extract a plausible base ticker; return '' if it's cash/blank/garbage."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    tok = s.split()[0]
    tok = tok.split("-")[0]
    tok = tok.replace("(", "").replace(")", "")
    tok = re.sub(r"[^A-Za-z0-9\.\-]", "", tok).upper()
    if not tok:
        return ""
    if tok in BLACKLIST_TOKENS:
        return ""
    if tok.isdigit():
        return ""
    if len(tok) > 8 and tok.isalnum():
        return ""
    return tok

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

def read_mapping(gc) -> Dict[str, Dict[str, str]]:
    """Read Mapping tab to support GOOGLEFINANCE exchange prefixes / yfinance tickers."""
    try:
        ws = open_ws(gc, "Mapping")
        dfm = read_tab(ws)
        out: Dict[str, Dict[str, str]] = {}
        if not dfm.empty and "Ticker" in dfm.columns:
            for _, row in dfm.iterrows():
                t = str(row.get("Ticker", "")).strip().upper()
                if not t:
                    continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym", "")).strip(),
                    "TickerYF": str(row.get("TickerYF", "")).strip().upper(),
                }
        return out
    except Exception:
        return {}

def googlefinance_formula_for(ticker: str, row_idx: int, mapping: Dict[str, Dict[str, str]]) -> str:
    base = base_symbol_from_string(ticker)
    mapped = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    sym = mapped.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_idx},"price"), ""))'

# ----- Signals → Source backfilling helpers
def latest_source_map_from_signals(sig_df: pd.DataFrame) -> dict:
    """Return {Ticker: Source} from the latest Signals timestamp per ticker."""
    if sig_df.empty:
        return {}
    tmp = sig_df.copy()
    tmp["Ticker"] = tmp["Ticker"].astype(str).str.strip().str.upper()
    tmp["Source"] = tmp["Source"].astype(str).str.strip()
    tmp["TimestampUTC"] = pd.to_datetime(tmp["TimestampUTC"], errors="coerce", utc=True)
    tmp = tmp[tmp["Ticker"] != ""]
    if tmp.empty:
        return {}
    tmp = tmp.sort_values(["Ticker", "TimestampUTC"])
    return tmp.groupby("Ticker")["Source"].last().to_dict()

def fill_sources_from_map(df: pd.DataFrame, src_map: dict) -> pd.DataFrame:
    """Fill empty or '(unknown)' Source using src_map by Ticker."""
    if df.empty or not src_map:
        return df
    out = df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
    cur = out.get("Source", pd.Series([""] * len(out)))
    cur = cur.astype(str).str.strip()
    need = (cur.eq("")) | (cur.eq("(unknown)"))
    out.loc[need, "Source"] = out.loc[need, "Ticker"].map(src_map).fillna(out.loc[need, "Source"])
    return out

# =========================
# LOAD TABS
# =========================
def load_signals(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig.empty:
        return df_sig
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
    return df[["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"]]

def load_transactions(df_tx: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize Fidelity transactions to trades (BUY/SELL with positive Qty).
    Returns (trades_df, unmatched_sell_df_for_debug).
    """
    if df_tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_tx.copy()

    # Find usable columns (Fidelity names vary)
    datecol = next((c for c in df.columns if "run date" in c.lower() or c.lower() == "run date"), None)
    if not datecol:
        datecol = next((c for c in df.columns if "date" in c.lower()), None)

    symcol  = next((c for c in df.columns if c.lower() in ("symbol", "security", "symbol/cusip")), None)
    actioncol = next((c for c in df.columns if "action" in c.lower()), None)
    typecol   = next((c for c in df.columns if c.lower() == "type"), None)
    desccol   = next((c for c in df.columns if "description" in c.lower()), None)
    qtycol    = next((c for c in df.columns if "quantity" in c.lower()), None)
    pricecol  = next((c for c in df.columns if "price" in c.lower()), None)

    if not datecol:
        raise ValueError("Transactions: missing a Run Date / Date column.")

    # Series on full df
    action = df[actioncol].astype(str) if actioncol else pd.Series([""] * len(df))
    typ    = df[typecol].astype(str)   if typecol   else pd.Series([""] * len(df))
    desc   = df[desccol].astype(str)   if desccol   else pd.Series([""] * len(df))

    action_up = action.str.upper()
    typ_up    = typ.str.upper()
    desc_up   = desc.str.upper()

    # Use non-capturing groups to silence the warning
    patt = r"\b(?:YOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"
    mask_action = action_up.str.contains(patt, regex=True, na=False)
    mask_type   = typ_up.str.contains(patt,    regex=True, na=False)
    mask_desc   = desc_up.str_contains(patt,   regex=True, na=False) if hasattr(desc_up, "str_contains") else desc_up.str.contains(patt, regex=True, na=False)
    mask_trades = mask_action | mask_type | mask_desc

    df_tr = df[mask_trades].copy()
    if debug:
        print(f"• load_transactions: detected {mask_trades.sum()} trade-like rows (of {len(df)})")

    # Compute normalized columns on this filtered frame ONLY
    df_tr["When"] = to_dt(df_tr[datecol])

    # Raw symbol, fallback to parsing from action/description if blank
    raw_symbol = df_tr[symcol].astype(str) if symcol else pd.Series([""] * len(df_tr))
    # Extract (AAPL) style tickers from action/description when symbol missing (on df_tr-aligned text)
    pattern_paren = re.compile(r"\(([A-Za-z0-9\.\-]{1,8})\)")
    def symbol_from_text(txt: str) -> str:
        m = pattern_paren.search(txt or "")
        return m.group(1).upper() if m else ""

    action_tr = df_tr[actioncol].astype(str) if actioncol else pd.Series([""] * len(df_tr), index=df_tr.index)
    desc_tr   = df_tr[desccol].astype(str)   if desccol   else pd.Series([""] * len(df_tr), index=df_tr.index)

    sym_fill = raw_symbol.copy()
    blank_mask = sym_fill.str.strip().eq("") | sym_fill.str.upper().isin(BLACKLIST_TOKENS)
    if blank_mask.any():
        fb = action_tr[blank_mask].map(symbol_from_text)
        fb = fb.replace("", np.nan).fillna(desc_tr[blank_mask].map(symbol_from_text))
        sym_fill.loc[blank_mask] = fb.fillna("")

    df_tr["Symbol"] = sym_fill.map(base_symbol_from_string)

    # Normalize Type to BUY/SELL using df_tr-aligned fields
    typ_tr = df_tr[typecol].astype(str) if typecol else pd.Series([""] * len(df_tr), index=df_tr.index)

    def classify_row(a: str, t: str, d: str) -> str:
        txt = f"{a} {t} {d}".upper()
        if "SOLD" in txt or re.search(r"\bSELL\b", txt):
            return "SELL"
        if "BOUGHT" in txt or re.search(r"\bBUY\b", txt):
            return "BUY"
        return ""

    df_tr["Type"] = [classify_row(a, t, d) for a, t, d in zip(action_tr, typ_tr, desc_tr)]

    # Qty / Price from df_tr
    qty_series   = to_float(df_tr[qtycol])   if qtycol   else pd.Series(np.nan, index=df_tr.index)
    price_series = to_float(df_tr[pricecol]) if pricecol else pd.Series(np.nan, index=df_tr.index)

    # Keep only valid trades with symbol + time + type in BUY/SELL
    ok = (df_tr["Symbol"].ne("")) & df_tr["When"].notna() & df_tr["Type"].isin(["BUY", "SELL"])
    df_tr = df_tr[ok].copy()

    # Make quantities positive; SELL still recorded as positive for matching
    df_tr["Qty"]   = qty_series.abs()
    df_tr["Price"] = price_series
    df_tr = df_tr.sort_values("When").reset_index(drop=True)

    # Debug snapshot
    if debug:
        print("• load_transactions: after cleaning →", len(df_tr), "trades")
        print(df_tr[["When", "Type", "Symbol", "Qty", "Price"]].head(8))

    # Nothing to match yet; unmatched SELLs computed during FIFO below
    return df_tr, pd.DataFrame()

def load_holdings(df_h: pd.DataFrame) -> pd.DataFrame:
    if df_h.empty:
        return df_h
    symcol  = next((c for c in df_h.columns if c.lower() in ("symbol", "security", "symbol/cusip")), None)
    qtycol  = next((c for c in df_h.columns if "quantity" in c.lower()), None)
    pricecol= next((c for c in df_h.columns if "price" in c.lower()), None)
    out = pd.DataFrame()
    if symcol:   out["Ticker"] = df_h[symcol].map(base_symbol_from_string)
    if qtycol:   out["Qty"]    = to_float(df_h[qtycol])
    if pricecol: out["Price"]  = to_float(df_h[pricecol])
    return out

# =========================
# MATCHING / OUTPUT TABLES
# =========================
def build_realized_and_open(tx: pd.DataFrame, sig: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    FIFO match SELLs to BUYs for each ticker. Enrich buys with Signal info.
    Returns (realized_df, open_df, unmatched_warnings)
    """
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Index BUY signals by ticker/time
    sig_buy = sig[(sig["Direction"].str.upper() == "BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker", "TimestampUTC"], inplace=True)

    sig_by_ticker = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append((
            r["TimestampUTC"],
            {
                "Source":    r.get("Source", ""),
                "Timeframe": r.get("Timeframe", ""),
                "SigTime":   r.get("TimestampUTC"),
                "SigPrice":  r.get("Price", "")
            }
        ))

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                return payload
            if t <= when:
                return payload
        return {"Source": "(unknown)", "Timeframe": "", "SigTime": pd.NaT, "SigPrice": ""}

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []
    warnings = []

    for _, row in tx.iterrows():
        tkr, when, ttype = row["Symbol"], row["When"], row["Type"]
        qty  = float(row["Qty"]) if not pd.isna(row["Qty"]) else 0.0
        prc  = float(row["Price"]) if not pd.isna(row["Price"]) else np.nan
        if qty <= 0 or pd.isna(when) or tkr == "":
            continue

        if ttype == "BUY":
            siginfo = last_signal_for(tkr, when)
            lots[tkr].append({
                "qty_left": qty,
                "entry_price": prc,
                "entry_time": when,
                "source": siginfo.get("Source", ""),
                "timeframe": siginfo.get("Timeframe", ""),
                "sig_time": siginfo.get("SigTime"),
                "sig_price": siginfo.get("SigPrice"),
            })
        elif ttype == "SELL":
            remaining = qty
            if not lots[tkr]:
                warnings.append(f"{tkr} SELL on {when.isoformat()} qty={qty} price={prc} — No prior BUY lot")
            while remaining > 0 and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                if take <= 0:
                    break
                entry = lot["entry_price"] if not math.isnan(lot["entry_price"]) else 0.0
                exitp = prc if not math.isnan(prc) else 0.0
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
                    "Source": lot["source"] if lot["source"] else "(unknown)",
                    "Timeframe": lot["timeframe"],
                    "SignalTimeUTC": lot["sig_time"],
                    "SignalPrice": lot["sig_price"],
                })
                lot["qty_left"] -= take
                remaining -= take
                if lot["qty_left"] <= 1e-9:
                    lots[tkr].popleft()
            if remaining > 1e-9:
                warnings.append(f"{tkr} SELL on {when.isoformat()} qty={qty} price={prc} — SELL exceeds available BUY lots")

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df = realized_df.sort_values("ExitTimeUTC").reset_index(drop=True)

    # anything still in lots is open
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
                "PriceNow": "",  # may be populated later
                "Unrealized%": "",
                "EntryTimeUTC": lot["entry_time"],
                "DaysOpen": (now_utc - lot["entry_time"]).days if not pd.isna(lot["entry_time"]) else "",
                "Source": lot["source"] if lot["source"] else "(unknown)",
                "Timeframe": lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice": lot["sig_price"],
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df = open_df.sort_values("EntryTimeUTC").reset_index(drop=True)

    if debug and warnings:
        print("⚠️ Unmatched SELLs (no prior BUYs or partial over-sell):")
        for w in warnings:
            print("  •", w)
        print(f"⚠️ Total unmatched SELL events: {len(warnings)}")

    return realized_df, open_df, warnings

def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Insert PriceNow + Unrealized% formulas using GOOGLEFINANCE for each open row."""
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now = []
    unreal = []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        # +2 accounts for header row in Sheets (row 1)
        row_index = idx + 2
        formula = googlefinance_formula_for(tkr, row_index, mapping)
        price_now.append(formula)
        try:
            epf = float(ep)
            unreal.append(f'=IFERROR(( {formula} / {epf} - 1 ) * 100,"")' if epf > 0 else "")
        except Exception:
            unreal.append("")
    out["PriceNow"] = price_now
    out["Unrealized%"] = unreal
    return out

def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    if realized_df.empty and open_df.empty:
        return pd.DataFrame(columns=["Source", "Trades", "Wins", "WinRate%", "AvgReturn%", "MedianReturn%", "OpenSignals"])

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

    # Open counts
    if open_df.empty:
        open_counts = pd.DataFrame(columns=["Source", "OpenSignals"])
    else:
        open_counts = open_df.groupby("Source").size().rename("OpenSignals").reset_index()

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer")
    if perf.empty:
        return pd.DataFrame(columns=["Source", "Trades", "Wins", "WinRate%", "AvgReturn%", "MedianReturn%", "OpenSignals"])

    perf["Trades"] = pd.to_numeric(perf["Trades"], errors="coerce").fillna(0).astype(int)
    perf["Wins"]   = pd.to_numeric(perf["Wins"],   errors="coerce").fillna(0).astype(int)
    for col in ["WinRate%", "AvgReturn%", "MedianReturn%"]:
        perf[col] = pd.to_numeric(perf[col], errors="coerce").fillna(0.0)
    perf["OpenSignals"] = pd.to_numeric(perf["OpenSignals"], errors="coerce").fillna(0).astype(int)
    perf = perf.sort_values(["Source"]).reset_index(drop=True)
    return perf

# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(description="Build performance dashboard tabs.")
    ap.add_argument("--debug", action="store_true", default=False, help="Verbose debug output")
    ap.add_argument("--no-live", action="store_true", default=False, help="Do not add live GOOGLEFINANCE formulas")
    args = ap.parse_args()
    DEBUG = args.debug
    USE_LIVE = not args.no_live

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
    tx, _ = load_transactions(df_tx, debug=DEBUG)
    _hold = load_holdings(df_h)

    realized_df, open_df, warnings = build_realized_and_open(tx, sig, debug=DEBUG)

    # Always backfill unknown/blank Sources using latest Signals row per ticker
    src_map = latest_source_map_from_signals(sig)
    realized_df = fill_sources_from_map(realized_df, src_map)
    open_df     = fill_sources_from_map(open_df, src_map)

    # Live price formulas for open lots (unless disabled)
    if USE_LIVE and not open_df.empty:
        mapping = read_mapping(gc)
        open_df = add_live_price_formulas(open_df, mapping)

    # Pretty column order
    if not realized_df.empty:
        realized_df = realized_df[[
            "Ticker","Qty","EntryPrice","ExitPrice","Return%","HoldDays",
            "EntryTimeUTC","ExitTimeUTC","Source","Timeframe","SignalTimeUTC","SignalPrice"
        ]]
    if not open_df.empty:
        # If live disabled, ensure PriceNow/Unrealized% exist as empty cols
        if "PriceNow" not in open_df.columns:
            open_df.insert(open_df.columns.get_loc("EntryPrice") + 1, "PriceNow", "")
            open_df.insert(open_df.columns.get_loc("PriceNow") + 1, "Unrealized%", "")
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
    if warnings:
        print(f"⚠️ Summary: {len(warnings)} unmatched SELL events (use --debug for details).")
    print("🎯 Done.")

if __name__ == "__main__":
    main()
