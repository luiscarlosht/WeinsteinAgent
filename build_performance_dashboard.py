#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Google Sheets dashboard tabs:
- Realized_Trades
- Open_Positions
- Performance_By_Source   (with totals row at TOP)
- OpenLots_Detail         (new: one row per open lot)

Reads:
  Signals        (manual or merged; you can trust 'Source' you typed)
  Transactions   (Fidelity history)
  Holdings       (optional; not used for PnL)

Notes
- FIFO matches BUY lots to SELLs to compute realized returns.
- Source/Timeframe attribution:
    * Default (fallback ON): use most-recent signal at/<= BUY time; if none, use most-recent signal for that ticker at any time.
    * --strict-signals: require a signal at or BEFORE the BUY; otherwise Source="(unknown)".
- Open_Positions can include live GOOGLEFINANCE formulas (omit with --no-live).
- Summaries printed: unmatched SELLs and any tickers still "(unknown)".
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
TAB_OPEN_DETAIL  = "OpenLots_Detail"   # new

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "
ROW_CHUNK = 500

# ─────────────────────────────
# UTILITIES
# ─────────────────────────────
def auth_gspread():
    print("🔑 Authorizing service account…")
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
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)

def write_tab(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.resize(rows=100, cols=8)
        ws.update([["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows+5), cols=max(min(26, cols+2), 8))
    header = [str(c) for c in df.columns]
    data = [header] + df.astype(str).fillna("").values.tolist()

    start = 0
    r = 1
    while start < len(data):
        end = min(start+ROW_CHUNK, len(data))
        block = data[start:end]
        ncols = len(header)
        top_left = gspread.utils.rowcol_to_a1(r, 1)
        bottom_right = gspread.utils.rowcol_to_a1(r + len(block) - 1, ncols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(block, range_name=rng)  # values first, named range second
        r += len(block)
        start = end

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
    "CASH","USD","INTEREST","DIVIDEND","REINVESTMENT","FEE","WITHDRAWAL","DEPOSIT","TRANSFER","SWEEP"
}

def base_symbol_from_string(s) -> str:
    """Extract a plausible base symbol; return '' for cash/blank/other."""
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
    if len(token) > 8 and token.isalnum():  # probably account-ish
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
        return pd.DataFrame(columns=["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"])
    df = df_sig.copy()
    tcol = next((c for c in df.columns if c.lower() in ("ticker","symbol")), None)
    if not tcol:
        raise ValueError("Signals tab needs a 'Ticker' column.")
    tscol = next((c for c in df.columns if c.lower().startswith("timestamp")), None)

    df["Ticker"] = df[tcol].map(base_symbol_from_string)
    df["Source"] = df.get("Source","").fillna("").astype(str)
    df["Direction"] = df.get("Direction","").fillna("").astype(str)
    df["Timeframe"] = df.get("Timeframe","").fillna("").astype(str)
    df["TimestampUTC"] = to_dt(df[tscol]) if tscol else pd.NaT
    df["Price"] = df.get("Price","").fillna("").astype(str)

    df = df[df["Ticker"].ne("")]
    df.sort_values(["Ticker","TimestampUTC"], inplace=True, ignore_index=True)
    return df[["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"]]

# Regex without capture groups
_TRADE_PATT = r"\b(?:YOU\s+)?(?:BOUGHT|SOLD|BUY|SELL)\b"

def _looks_like_trade_mask(action: pd.Series, typ: pd.Series, desc: pd.Series) -> pd.Series:
    action_up = action.fillna("").astype(str).str.upper()
    typ_up    = typ.fillna("").astype(str).str.upper()
    desc_up   = desc.fillna("").astype(str).str.upper()
    return (
        action_up.str.contains(_TRADE_PATT, regex=True, na=False)
        | typ_up.str.contains(_TRADE_PATT, regex=True, na=False)
        | desc_up.str.contains(_TRADE_PATT, regex=True, na=False)
    )

def _classify_type(a: str, t: str, d: str) -> str:
    s = f"{a or ''} {t or ''} {d or ''}".upper()
    if "SOLD" in s or "SELL" in s:
        return "SELL"
    if "BOUGHT" in s or "BUY" in s:
        return "BUY"
    return ""

def _symbol_from_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    m = re.search(r"\(([A-Z][A-Z0-9\.\-]{0,7})\)", text.upper())
    return m.group(1) if m else ""

def load_transactions(df_tx: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_tx.empty:
        return pd.DataFrame(columns=["When","Type","Symbol","Qty","Price"]), pd.DataFrame()

    datecol = next((c for c in df_tx.columns if "run date" in c.lower() or c.lower()=="date"), None)
    actioncol = next((c for c in df_tx.columns if "action" in c.lower()), None)
    typecol   = next((c for c in df_tx.columns if c.lower()=="type"), None)
    desccol   = next((c for c in df_tx.columns if "description" in c.lower()), None)
    symcol    = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol    = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    pricecol  = next((c for c in df_tx.columns if "price" in c.lower()), None)
    amtcol    = next((c for c in df_tx.columns if "amount" in c.lower()), None)

    if not datecol:
        raise ValueError("Transactions: missing Date / Run Date column.")
    if not (actioncol or typecol or desccol):
        raise ValueError("Transactions: need at least one of Action / Type / Description.")
    if not (symcol or actioncol or desccol):
        raise ValueError("Transactions: need Symbol or parsable ticker in Action/Description.")

    df = df_tx.copy()

    mask_tr = _looks_like_trade_mask(df.get(actioncol, ""), df.get(typecol, ""), df.get(desccol, ""))
    n_all = len(df)
    n_tr  = int(mask_tr.sum())
    print(f"• load_transactions: detected {n_tr} trade-like rows (of {n_all})")

    df_tr = df.loc[mask_tr].copy()

    df_tr["When"] = to_dt(df_tr[datecol])

    df_tr["Type"] = [
        _classify_type(a, t, d) for a, t, d in zip(
            df_tr.get(actioncol, pd.Series(index=df_tr.index)),
            df_tr.get(typecol,   pd.Series(index=df_tr.index)),
            df_tr.get(desccol,   pd.Series(index=df_tr.index)),
        )
    ]

    if symcol:
        sym = df_tr[symcol].map(base_symbol_from_string)
    else:
        sym = pd.Series("", index=df_tr.index)

    sym = np.where(
        (sym == "") & df_tr.get(actioncol, "").astype(str).ne(""),
        df_tr[actioncol].map(_symbol_from_text),
        sym,
    )
    sym = np.where(
        (pd.Series(sym, index=df_tr.index) == "") & df_tr.get(desccol, "").astype(str).ne(""),
        df_tr[desccol].map(_symbol_from_text),
        sym,
    )
    df_tr["Symbol"] = pd.Series(sym, index=df_tr.index).map(base_symbol_from_string)

    qty = to_float(df_tr.get(qtycol, pd.Series(np.nan, index=df_tr.index))).abs()
    price = to_float(df_tr.get(pricecol, pd.Series(np.nan, index=df_tr.index)))

    if qty.isna().all() and amtcol and pricecol:
        amt = to_float(df_tr[amtcol])
        with np.errstate(divide='ignore', invalid='ignore'):
            qty = np.where(
                (price != 0) & (~np.isnan(price)) & (~np.isnan(amt)),
                np.abs(amt) / np.abs(price),
                np.nan,
            )
        qty = pd.Series(qty, index=df_tr.index)

    df_tr["Qty"] = pd.to_numeric(qty, errors="coerce")
    df_tr["Price"] = pd.to_numeric(price, errors="coerce")

    df_tr = df_tr[
        df_tr["Symbol"].ne("")
        & df_tr["When"].notna()
        & df_tr["Type"].isin(["BUY","SELL"])
        & (df_tr["Qty"] > 0)
    ].copy()

    df_tr.sort_values(["When"], inplace=True)
    df_tr.reset_index(drop=True, inplace=True)

    if debug:
        print(f"• load_transactions: after cleaning → {len(df_tr)} trades")
        preview_cols = [c for c in ["Run Date","Account","Account Number","Action","Symbol","Description","Type","Quantity","Price ($)","Amount ($)","Settlement Date"] if c in df_tx.columns]
        try:
            print((df_tr.merge(df_tx[preview_cols + [datecol]], left_on=["When"], right_on=[datecol], how="left")
                      .head(8)
                      .to_string()))
        except Exception:
            print(df_tr.head(8).to_string(index=False))

    return df_tr[["When","Type","Symbol","Qty","Price"]], pd.DataFrame()

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

# ─────────────────────────────
# FIFO MATCHING (REALIZED & OPEN)
# ─────────────────────────────
def build_realized_and_open(
    tx: pd.DataFrame,
    sig: pd.DataFrame,
    debug: bool=False,
    strict_signals: bool=False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Index BUY-only signals by (Ticker, time)
    sig_buy = sig[(sig["Direction"].str.upper()=="BUY") & sig["Ticker"].ne("")].copy()
    sig_buy.sort_values(["Ticker","TimestampUTC"], inplace=True, ignore_index=True)

    sig_by_ticker: Dict[str, List[Tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, r in sig_buy.iterrows():
        sig_by_ticker[r["Ticker"]].append( (r["TimestampUTC"], {
            "Source": r.get("Source",""),
            "Timeframe": r.get("Timeframe",""),
            "SigTime": r.get("TimestampUTC"),
            "SigPrice": r.get("Price",""),
        }))

    def most_recent_signal_any(tkr: str):
        arr = sig_by_ticker.get(tkr, [])
        return arr[-1][1] if arr else {"Source":"(unknown)","Timeframe":"","SigTime": pd.NaT, "SigPrice": ""}

    def last_signal_for(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                return payload
            if t <= when:
                return payload
        # fallback only if NOT strict
        return most_recent_signal_any(tkr) if not strict_signals else {"Source":"(unknown)","Timeframe":"","SigTime": pd.NaT, "SigPrice": ""}

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []
    unmatched_sells = []

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
            while remaining > 1e-9 and lots[tkr]:
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
                    "Source": lot["source"] or "(unknown)",
                    "Timeframe": lot["timeframe"],
                    "SignalTimeUTC": lot["sig_time"],
                    "SignalPrice": lot["sig_price"],
                })
                lot["qty_left"] -= take
                remaining -= take
                if lot["qty_left"] <= 1e-9:
                    lots[tkr].popleft()
            if remaining > 1e-9:
                msg = f"{tkr} SELL on {when.isoformat()} qty={qty} price={price} — No prior BUY lot in window"
                unmatched_sells.append(msg)
                if debug:
                    print(f"⚠️ Unmatched SELL: {msg}")

    realized_df = pd.DataFrame(realized_rows)
    if not realized_df.empty:
        realized_df.sort_values("ExitTimeUTC", inplace=True, ignore_index=True)

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
                "Source": lot["source"] or "(unknown)",
                "Timeframe": lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice": lot["sig_price"],
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df.sort_values("EntryTimeUTC", inplace=True, ignore_index=True)

    return realized_df, open_df, unmatched_sells

# ─────────────────────────────
# PERFORMANCE TABLE (+ totals on TOP)
# ─────────────────────────────
def _weighted_average(series_values, series_weights):
    v = pd.to_numeric(series_values, errors="coerce")
    w = pd.to_numeric(series_weights, errors="coerce")
    mask = (~v.isna()) & (~w.isna()) & (w > 0)
    if not mask.any():
        return 0.0
    return (v[mask] * w[mask]).sum() / w[mask].sum()

def build_perf_by_source(realized_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    # realized metrics by Source
    if realized_df.empty:
        realized_grp = pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%"])
    else:
        tmp = realized_df.copy()
        tmp["ret"] = pd.to_numeric(tmp["Return%"], errors="coerce")
        tmp["is_win"] = tmp["ret"] > 0
        g = tmp.groupby("Source", dropna=False)
        realized_grp = pd.DataFrame({
            "Trades": g.size(),
            "Wins": g["is_win"].sum(),
            "WinRate%": (g["is_win"].mean()*100).round(2),
            "AvgReturn%": g["ret"].mean().round(2),
            "MedianReturn%": g["ret"].median().round(2),
        }).reset_index()

    # open lot counts by Source and distinct tickers
    if open_df.empty:
        open_counts = pd.DataFrame(columns=["Source","OpenLots","OpenTickers"])
    else:
        open_counts = (open_df
                       .groupby("Source")
                       .agg(OpenLots=("Ticker","size"),
                            OpenTickers=("Ticker", lambda s: len(set(s))))
                       .reset_index())

    perf = pd.merge(realized_grp, open_counts, on="Source", how="outer").fillna(0)
    if perf.empty:
        return pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenLots","OpenTickers"])

    # types / ordering
    perf["Trades"] = perf["Trades"].astype(int)
    perf["Wins"] = perf["Wins"].astype(int)
    perf["OpenLots"] = perf["OpenLots"].astype(int)
    perf["OpenTickers"] = perf["OpenTickers"].astype(int)
    for c in ["WinRate%","AvgReturn%","MedianReturn%"]:
        perf[c] = pd.to_numeric(perf[c], errors="coerce").fillna(0.0)

    # totals row (on TOP)
    total_trades = int(perf["Trades"].sum())
    total_wins   = int(perf["Wins"].sum())
    total_winrate = round((total_wins / total_trades * 100.0), 2) if total_trades > 0 else 0.0
    total_avg = round(_weighted_average(perf["AvgReturn%"], perf["Trades"]), 2)
    total_med = round(_weighted_average(perf["MedianReturn%"], perf["Trades"]), 2)
    total_open_lots = int(perf["OpenLots"].sum())
    total_open_tickers = int(perf["OpenTickers"].sum())

    totals_row = pd.DataFrame([{
        "Source": "(TOTALS)",
        "Trades": total_trades,
        "Wins": total_wins,
        "WinRate%": total_winrate,
        "AvgReturn%": total_avg,
        "MedianReturn%": total_med,
        "OpenLots": total_open_lots,
        "OpenTickers": total_open_tickers,
    }])

    perf = perf.sort_values(["Source"]).reset_index(drop=True)
    perf = pd.concat([totals_row, perf], ignore_index=True)
    return perf

# ─────────────────────────────
# OPEN LOTS DETAIL TAB
# ─────────────────────────────
def build_open_lots_detail(open_df: pd.DataFrame) -> pd.DataFrame:
    if open_df.empty:
        return pd.DataFrame()
    # Reorder / rename for a convenient view
    cols = [
        "Source","Ticker","OpenQty","EntryPrice","EntryTimeUTC","DaysOpen",
        "Timeframe","SignalTimeUTC","SignalPrice"
    ]
    # Ensure columns exist (when live formulas are added they're still present)
    existing = [c for c in cols if c in open_df.columns]
    detail = open_df[existing].copy()
    detail.sort_values(["Source","Ticker","EntryTimeUTC"], inplace=True, ignore_index=True)
    return detail

# ─────────────────────────────
# REPORT HELPERS
# ─────────────────────────────
def print_unknown_sources(realized_df: pd.DataFrame, open_df: pd.DataFrame):
    unk_real = []
    if not realized_df.empty:
        unk_real = sorted(set(realized_df.loc[realized_df["Source"].eq("(unknown)"), "Ticker"]))
    unk_open = []
    if not open_df.empty:
        unk_open = sorted(set(open_df.loc[open_df["Source"].eq("(unknown)"), "Ticker"]))

    if unk_real or unk_open:
        print("🔎 Unknown Source tickers:")
        if unk_real:
            print("  • Realized:", ", ".join(unk_real))
        if unk_open:
            print("  • Open    :", ", ".join(unk_open))

def print_open_breakdown(open_df: pd.DataFrame):
    if open_df.empty:
        return
    grp = open_df.groupby(["Source","Ticker"]).size().rename("Lots").reset_index()
    print("🔍 Open breakdown:")
    for _, r in grp.sort_values(["Source","Ticker"]).iterrows():
        print(f"  - {r['Source']}: {r['Ticker']} → {int(r['Lots'])} lot(s)")

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build performance dashboard tabs.")
    ap.add_argument("--no-live", action="store_true", help="Do NOT add GOOGLEFINANCE formulas to Open_Positions.")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    ap.add_argument("--strict-signals", action="store_true",
                    help="Require a signal at or BEFORE BUY to attribute Source; otherwise mark as (unknown)")
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

    # Normalize signals
    sig = load_signals(df_sig)

    # Clean transactions
    tx, _ = load_transactions(df_tx, debug=DEBUG)

    # Build realized & open (FIFO)
    realized_df, open_df, unmatched_sells = build_realized_and_open(
        tx, sig, debug=DEBUG, strict_signals=args.strict_signals
    )

    # Optionally add live price formulas (Open_Positions)
    if not args.no_live and not open_df.empty:
        mapping = read_mapping(gc)
        open_df = add_live_price_formulas(open_df, mapping)

    # Column order prettify
    if not realized_df.empty:
        realized_df = realized_df[[
            "Ticker","Qty","EntryPrice","ExitPrice","Return%","HoldDays",
            "EntryTimeUTC","ExitTimeUTC","Source","Timeframe","SignalTimeUTC","SignalPrice"
        ]]
    if not open_df.empty:
        base_cols = ["Ticker","OpenQty","EntryPrice","EntryTimeUTC","DaysOpen","Source","Timeframe","SignalTimeUTC","SignalPrice"]
        if "PriceNow" in open_df.columns and "Unrealized%" in open_df.columns:
            base_cols = ["Ticker","OpenQty","EntryPrice","PriceNow","Unrealized%","EntryTimeUTC","DaysOpen","Source","Timeframe","SignalTimeUTC","SignalPrice"]
        open_df = open_df[base_cols]

    # Performance by source
    perf_df = build_perf_by_source(realized_df.copy(), open_df.copy())

    # Open lots detail
    open_detail_df = build_open_lots_detail(open_df.copy())

    # Write tabs
    ws_real = open_ws(gc, TAB_REALIZED)
    ws_open = open_ws(gc, TAB_OPEN)
    ws_perf = open_ws(gc, TAB_PERF)
    ws_open_d = open_ws(gc, TAB_OPEN_DETAIL)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_df)
    write_tab(ws_open_d, open_detail_df)

    print(f"✅ Wrote {TAB_REALIZED}: {len(realized_df)} rows")
    print(f"✅ Wrote {TAB_OPEN}: {len(open_df)} rows")
    print(f"✅ Wrote {TAB_PERF}: {len(perf_df)} rows")
    print(f"✅ Wrote {TAB_OPEN_DETAIL}: {len(open_detail_df)} rows")

    # Summaries
    if unmatched_sells:
        print(f"⚠️ Summary: {len(unmatched_sells)} unmatched SELL events (use --debug to print details).")
        if DEBUG:
            for line in unmatched_sells:
                print("  •", line)

    print_unknown_sources(realized_df, open_df)
    if DEBUG and not open_df.empty:
        print_open_breakdown(open_df)

    print("🎯 Done.")

if __name__ == "__main__":
    main()
