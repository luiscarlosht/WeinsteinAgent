#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Google Sheets dashboard tabs:
- Realized_Trades
- Open_Positions
- Performance_By_Source (with Timeframe split and totals row)
- Performance_By_Ticker
- OpenLots_Detail
- Equity_Curve_By_Source (cumulative % gain, normalized to 100)

Reads:
  Signals        (manual or merged)
  Transactions   (Fidelity history)
  Holdings       (optional; not used in PnL, only context)

Key behavior
- FIFO match BUY lots to SELLs (fees/commissions included).
- Source/Timeframe taken from the most recent BUY-time signal.
  *Default:* falls back to most recent signal anytime for that ticker.
  *--strict-signals:* disable fallback (unknown if no prior or same-time signal).
- Optional live price formulas in Open_Positions (disable via --no-live).
- Prints unmatched SELL summary and any tickers with Source="(unknown)".

New
- Per-timeframe split in performance summary.
- Per-ticker performance tab.
- Equity curve (daily cumulative return %) per Source.
- Sell cutoff (ignore SELLs after date if no matching BUY in window).
- Account name carried into realized & open details when available.
"""

from __future__ import annotations

import math
import re
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

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
TAB_HOLDINGS     = "Holdings"                # optional
TAB_REALIZED     = "Realized_Trades"
TAB_OPEN         = "Open_Positions"
TAB_PERF         = "Performance_By_Source"
TAB_PERF_TICKER  = "Performance_By_Ticker"
TAB_OPEN_DETAIL  = "OpenLots_Detail"
TAB_EQUITY       = "Equity_Curve_By_Source"

DEFAULT_EXCHANGE_PREFIX = "NYSE: "
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

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
        ws.update(block, range_name=rng)
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
    "CASH","USD","INTEREST","DIVIDEND","REINVESTMENT","FEE","WITHDRAWAL","DEPOSIT",
    "TRANSFER","SWEEP"
}

def base_symbol_from_string(s) -> str:
    """Extract a base equity/ETF symbol; return '' for cash/blank/other."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    token = s.split()[0]
    token = token.split("-")[0]
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
        cols = ["When","Type","Symbol","Qty","Price","Account","Commission","Fees"]
        return pd.DataFrame(columns=cols), pd.DataFrame()

    datecol     = next((c for c in df_tx.columns if "run date" in c.lower() or c.lower()=="date"), None)
    actioncol   = next((c for c in df_tx.columns if "action" in c.lower()), None)
    typecol     = next((c for c in df_tx.columns if c.lower()=="type"), None)
    desccol     = next((c for c in df_tx.columns if "description" in c.lower()), None)
    symcol      = next((c for c in df_tx.columns if c.lower() in ("symbol","security","symbol/cusip")), None)
    qtycol      = next((c for c in df_tx.columns if "quantity" in c.lower()), None)
    pricecol    = next((c for c in df_tx.columns if "price" in c.lower()), None)
    amtcol      = next((c for c in df_tx.columns if "amount" in c.lower()), None)
    acctcol     = next((c for c in df_tx.columns if c.lower() in ("account","account name")), None)
    commcol     = next((c for c in df_tx.columns if "commission" in c.lower()), None)
    feescol     = next((c for c in df_tx.columns if re.search(r"\bfees?\b", c.lower() or "") ), None)

    if not datecol:
        raise ValueError("Transactions: missing Date / Run Date column.")
    if not (actioncol or typecol or desccol):
        raise ValueError("Transactions: need at least one of Action / Type / Description.")
    if not (symcol or actioncol or desccol):
        raise ValueError("Transactions: need Symbol or parsable ticker in Action/Description.")

    df = df_tx.copy()
    action = df.get(actioncol, "")
    typ    = df.get(typecol, "")
    desc   = df.get(desccol, "")
    mask_tr = _looks_like_trade_mask(action, typ, desc)
    n_all = len(df); n_tr = int(mask_tr.sum())
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

    # Symbol
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

    # Numbers
    qty = to_float(df_tr.get(qtycol, pd.Series(np.nan, index=df_tr.index))).abs()
    price = to_float(df_tr.get(pricecol, pd.Series(np.nan, index=df_tr.index)))
    comm  = to_float(df_tr.get(commcol, pd.Series(0.0, index=df_tr.index)))
    fees  = to_float(df_tr.get(feescol, pd.Series(0.0, index=df_tr.index)))
    acct  = df_tr.get(acctcol, pd.Series("", index=df_tr.index)).fillna("").astype(str)

    # Reconstruct qty if blank (Amount/Price)
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
    df_tr["Commission"] = pd.to_numeric(comm, errors="coerce").fillna(0.0)
    df_tr["Fees"] = pd.to_numeric(fees, errors="coerce").fillna(0.0)
    df_tr["Account"] = acct

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
        print(df_tr.head(8).to_string(index=False))

    cols = ["When","Type","Symbol","Qty","Price","Account","Commission","Fees"]
    return df_tr[cols], pd.DataFrame()

def add_live_price_formulas(open_df: pd.DataFrame, mapping: Dict[str,Dict[str,str]]) -> pd.DataFrame:
    if open_df.empty:
        return open_df
    out = open_df.copy()
    price_now = []
    unreal = []
    for idx, r in out.iterrows():
        tkr = r["Ticker"]
        ep  = r.get("EntryPrice")
        row_index = idx + 2  # sheet header is row 1
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
    sell_cutoff: Optional[pd.Timestamp]=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Include fees/commissions in effective prices:
      entry_effective = (price*qty + fees + comm) / qty
      exit_effective  = (price*qty - fees - comm) / qty
    """
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Index BUY signals by ticker/time
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

    def signal_for_buy(tkr: str, when: pd.Timestamp):
        arr = sig_by_ticker.get(tkr, [])
        # prefer most recent at/<= when
        best_prior = None
        for t, payload in reversed(arr):
            if pd.isna(t) or pd.isna(when):
                continue
            if t <= when:
                best_prior = payload
                break
        if best_prior:
            return best_prior
        if strict_signals:
            return {"Source":"(unknown)","Timeframe":"","SigTime": pd.NaT, "SigPrice": ""}
        # fallback to most recent anytime for that ticker
        if arr:
            return arr[-1][1]
        return {"Source":"(unknown)","Timeframe":"","SigTime": pd.NaT, "SigPrice": ""}

    lots: Dict[str, deque] = defaultdict(deque)
    realized_rows = []
    unmatched_sells = []

    for _, row in tx.iterrows():
        tkr   = row["Symbol"]
        when  = row["When"]
        ttype = row["Type"]
        qty   = row["Qty"] if not math.isnan(row["Qty"]) else 0.0
        price = row["Price"] if not math.isnan(row["Price"]) else np.nan
        comm  = float(row.get("Commission", 0.0) or 0.0)
        fees  = float(row.get("Fees", 0.0) or 0.0)
        acct  = row.get("Account","")

        if qty <= 0 or pd.isna(when) or tkr=="":
            continue

        if ttype == "BUY":
            # Effective per-share entry price including costs
            entry_ps = price
            if qty > 0 and not math.isnan(price):
                entry_ps = (price*qty + comm + fees) / qty
            siginfo = signal_for_buy(tkr, when)
            lots[tkr].append({
                "qty_left": qty,
                "entry_price": entry_ps,
                "entry_time": when,
                "account": acct,
                "source": siginfo.get("Source",""),
                "timeframe": siginfo.get("Timeframe",""),
                "sig_time": siginfo.get("SigTime"),
                "sig_price": siginfo.get("SigPrice"),
            })

        elif ttype == "SELL":
            remaining = qty
            if sell_cutoff is not None and when >= sell_cutoff:
                # If no lots available and we're post-cutoff, skip unmatched SELLs
                if not lots[tkr]:
                    if debug:
                        print(f"⏭️  Ignoring unmatched SELL after cutoff: {tkr} {when.date()} qty={qty} price={price}")
                    continue

            while remaining > 0 and lots[tkr]:
                lot = lots[tkr][0]
                take = min(remaining, lot["qty_left"])
                if take <= 0:
                    break

                entry = lot["entry_price"] if not math.isnan(lot["entry_price"]) else 0.0
                # Effective per-share exit price including costs
                exit_ps = price
                if take > 0 and not math.isnan(price):
                    # scale SELL fees/commissions per-share using the SELL's full qty
                    per_share_cost = (comm + fees) / qty if qty > 0 else 0.0
                    exit_ps = price - per_share_cost

                ret_pct = ((exit_ps - entry) / entry * 100.0) if entry else np.nan
                held_days = (when - lot["entry_time"]).days if (not pd.isna(lot["entry_time"]) and not pd.isna(when)) else ""

                realized_rows.append({
                    "Ticker": tkr,
                    "Qty": round(take, 6),
                    "EntryPrice": round(entry, 6) if entry else "",
                    "ExitPrice": round(exit_ps, 6) if not math.isnan(exit_ps) else "",
                    "Return%": round(ret_pct, 6) if not np.isnan(ret_pct) else "",
                    "HoldDays": held_days,
                    "EntryTimeUTC": lot["entry_time"],
                    "ExitTimeUTC": when,
                    "Source": lot["source"] or "(unknown)",
                    "Timeframe": lot["timeframe"],
                    "SignalTimeUTC": lot["sig_time"],
                    "SignalPrice": lot["sig_price"],
                    "Account": lot.get("account",""),
                })
                lot["qty_left"] -= take
                remaining -= take
                if lot["qty_left"] <= 1e-9:
                    lots[tkr].popleft()

            if remaining > 1e-9:
                unmatched_sells.append(
                    f"{tkr} SELL on {when.isoformat()} qty={qty} price={price} — No prior BUY lot in window"
                )
                if debug:
                    print(f"⚠️ Unmatched SELL: {unmatched_sells[-1]}")

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
                "Source": lot["source"] or "(unknown)",
                "Timeframe": lot["timeframe"],
                "SignalTimeUTC": lot["sig_time"],
                "SignalPrice": lot["sig_price"],
                "Account": lot.get("account",""),
            })
    open_df = pd.DataFrame(open_rows)
    if not open_df.empty:
        open_df.sort_values("EntryTimeUTC", inplace=True, ignore_index=True)

    return realized_df, open_df, unmatched_sells

# ─────────────────────────────
# PERFORMANCE TABLES
# ─────────────────────────────
def _totals_row(perf_like: pd.DataFrame, label="(TOTALS)") -> pd.DataFrame:
    if perf_like.empty:
        return perf_like
    # Weighted by counts for Avg/Median? We'll take simple mean of AvgReturn%/MedianReturn%,
    # and winrate weighted by trade count.
    trades = perf_like["Trades"].sum()
    wins   = perf_like["Wins"].sum()
    winrate = round((wins / trades * 100.0), 2) if trades else 0.0
    avg_ret = round(pd.to_numeric(perf_like["AvgReturn%"], errors="coerce").mean(), 2) if not perf_like.empty else 0.0
    med_ret = round(pd.to_numeric(perf_like["MedianReturn%"], errors="coerce").median(), 2) if not perf_like.empty else 0.0
    open_lots = int(perf_like["OpenLots"].sum()) if "OpenLots" in perf_like.columns else 0
    open_tks  = int(perf_like["OpenTickers"].sum()) if "OpenTickers" in perf_like.columns else 0

    extra_cols = [c for c in perf_like.columns if c not in ["Source","Timeframe","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenLots","OpenTickers"]]
    row = {
        "Source": label, "Timeframe": "",
        "Trades": trades, "Wins": wins, "WinRate%": winrate,
        "AvgReturn%": avg_ret, "MedianReturn%": med_ret,
        "OpenLots": open_lots, "OpenTickers": open_tks,
    }
    for c in extra_cols:
        row[c] = ""
    return pd.DataFrame([row])[perf_like.columns]

def build_perf_by_source_timeframe(realized_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    # Realized summary (Source, Timeframe)
    if realized_df.empty:
        realized_grp = pd.DataFrame(columns=["Source","Timeframe","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%"])
    else:
        tmp = realized_df.copy()
        tmp["ret"] = pd.to_numeric(tmp["Return%"], errors="coerce")
        tmp["is_win"] = tmp["ret"] > 0
        g = tmp.groupby(["Source","Timeframe"], dropna=False)
        realized_grp = pd.DataFrame({
            "Trades": g.size(),
            "Wins": g["is_win"].sum(),
            "WinRate%": (g["is_win"].mean()*100).round(2),
            "AvgReturn%": g["ret"].mean().round(2),
            "MedianReturn%": g["ret"].median().round(2),
        }).reset_index()

    # Open counts (Source, Timeframe)
    if open_df.empty:
        open_counts = pd.DataFrame(columns=["Source","Timeframe","OpenLots","OpenTickers"])
    else:
        # Open lots is simply count of open rows (each row is a lot)
        oc = open_df.groupby(["Source","Timeframe"]).size().rename("OpenLots").reset_index()
        # Open tickers is distinct tickers per (Source, Timeframe)
        ot = (open_df.groupby(["Source","Timeframe"])["Ticker"]
              .nunique().rename("OpenTickers").reset_index())
        open_counts = pd.merge(oc, ot, on=["Source","Timeframe"], how="outer")

    perf = pd.merge(realized_grp, open_counts, on=["Source","Timeframe"], how="outer").fillna(0)
    if perf.empty:
        return pd.DataFrame(columns=["Source","Timeframe","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenLots","OpenTickers"])

    perf["Trades"] = perf["Trades"].astype(int)
    perf["Wins"] = perf["Wins"].astype(int)
    for col in ["WinRate%","AvgReturn%","MedianReturn%"]:
        perf[col] = pd.to_numeric(perf[col], errors="coerce").fillna(0.0)
    perf["OpenLots"] = perf["OpenLots"].astype(int)
    perf["OpenTickers"] = perf["OpenTickers"].astype(int)

    perf = perf.sort_values(["Source","Timeframe"]).reset_index(drop=True)
    # Totals row at top
    totals = _totals_row(perf, label="(TOTALS)")
    perf = pd.concat([totals, perf], ignore_index=True)
    return perf

def build_perf_by_ticker(realized_df: pd.DataFrame) -> pd.DataFrame:
    if realized_df.empty:
        return pd.DataFrame(columns=["Ticker","Trades","Wins","AvgReturn%","MedianReturn%","Source","Timeframe"])
    df = realized_df.copy()
    df["ret"] = pd.to_numeric(df["Return%"], errors="coerce")
    df["is_win"] = df["ret"] > 0
    # Choose representative Source/Timeframe per ticker (most frequent)
    agg = df.groupby("Ticker").agg(
        Trades=("Ticker","size"),
        Wins=("is_win","sum"),
        AvgReturn%=("ret","mean"),
        MedianReturn%=("ret","median")
    ).reset_index()
    # mode for Source/Timeframe
    mode_src = (df.groupby("Ticker")["Source"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "(unknown)")).reset_index()
    mode_tf = (df.groupby("Ticker")["Timeframe"]
               .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")).reset_index()
    out = agg.merge(mode_src, on="Ticker").merge(mode_tf, on="Ticker")
    out.rename(columns={"Source":"Source","Timeframe":"Timeframe"}, inplace=True)
    out["AvgReturn%"] = out["AvgReturn%"].round(2)
    out["MedianReturn%"] = out["MedianReturn%"].round(2)
    # order by Ticker
    return out[["Ticker","Trades","Wins","AvgReturn%","MedianReturn%","Source","Timeframe"]].sort_values("Ticker")

# ─────────────────────────────
# EQUITY CURVE (CUMULATIVE %)
# ─────────────────────────────
def build_equity_curve_by_source(realized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative % gain per Source, normalized to 100.
    Daily step return = arithmetic mean of realized returns that day (per Source).
    Equity_t = Equity_{t-1} * (1 + daily_ret%/100)
    """
    if realized_df.empty:
        return pd.DataFrame(columns=["Date","Source","Equity"])
    df = realized_df.copy()
    df["ret"] = pd.to_numeric(df["Return%"], errors="coerce")
    df = df[df["ret"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["Date","Source","Equity"])

    df["Date"] = pd.to_datetime(df["ExitTimeUTC"]).dt.date
    daily = (df.groupby(["Date","Source"])["ret"].mean()
             .rename("DailyReturn%").reset_index())

    # build equity series per Source
    frames = []
    for src, g in daily.groupby("Source"):
        g = g.sort_values("Date").copy()
        eq = []
        cur = 100.0
        for _, r in g.iterrows():
            dr = float(r["DailyReturn%"]) if not pd.isna(r["DailyReturn%"]) else 0.0
            cur = cur * (1.0 + dr/100.0)
            eq.append(cur)
        gg = g.copy()
        gg["Equity"] = [round(x, 4) for x in eq]
        frames.append(gg[["Date","Source","Equity"]])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Date","Source","Equity"])
    out = out.sort_values(["Source","Date"]).reset_index(drop=True)
    return out

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

def print_open_unknown_breakdown(open_df: pd.DataFrame):
    if open_df.empty:
        return
    t = open_df[open_df["Source"].eq("(unknown)")]
    if t.empty:
        return
    print("🔍 Open breakdown:")
    for (src, tkr), g in t.groupby(["Source","Ticker"]):
        print(f"  - {src}: {tkr} → {len(g)} lot(s)")

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build performance dashboard tabs.")
    ap.add_argument("--no-live", action="store_true", help="Do NOT add GOOGLEFINANCE formulas to Open_Positions.")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    ap.add_argument("--strict-signals", action="store_true", help="Do NOT fallback to most-recent signal when BUY has no prior signal.")
    ap.add_argument("--sell-cutoff", type=str, default=None, help="YYYY-MM-DD — ignore unmatched SELLs on/after this date.")
    args = ap.parse_args()
    DEBUG = args.debug

    sell_cutoff = None
    if args.sell_cutoff:
        try:
            sell_cutoff = pd.to_datetime(args.sell_cutoff, utc=True)
        except Exception:
            print(f"⚠️ Could not parse --sell-cutoff '{args.sell_cutoff}'. Ignoring.")

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
        tx, sig, debug=DEBUG, strict_signals=args.strict_signals, sell_cutoff=sell_cutoff
    )

    # Optionally add live price formulas (Open_Positions)
    if not args.no_live and not open_df.empty:
        mapping = read_mapping(gc)
        open_df = add_live_price_formulas(open_df, mapping)

    # Pretty column orders
    if not realized_df.empty:
        realized_cols = [
            "Ticker","Qty","EntryPrice","ExitPrice","Return%","HoldDays",
            "EntryTimeUTC","ExitTimeUTC","Source","Timeframe","SignalTimeUTC","SignalPrice","Account"
        ]
        realized_df = realized_df[realized_cols]
    if not open_df.empty:
        cols = ["Ticker","OpenQty","EntryPrice","EntryTimeUTC","DaysOpen","Source","Timeframe","SignalTimeUTC","SignalPrice","Account"]
        if "PriceNow" in open_df.columns and "Unrealized%" in open_df.columns:
            cols = ["Ticker","OpenQty","EntryPrice","PriceNow","Unrealized%","EntryTimeUTC","DaysOpen","Source","Timeframe","SignalTimeUTC","SignalPrice","Account"]
        open_df = open_df[cols]

    # Performance tables
    perf_src_tf_df = build_perf_by_source_timeframe(realized_df.copy(), open_df.copy())
    perf_ticker_df = build_perf_by_ticker(realized_df.copy())

    # OpenLots_Detail tab
    open_detail_df = open_df.copy()
    if not open_detail_df.empty:
        # consistent order
        base_cols = ["Source","Timeframe","Ticker","OpenQty","EntryPrice","EntryTimeUTC","DaysOpen","SignalTimeUTC","SignalPrice","Account"]
        if "PriceNow" in open_detail_df.columns and "Unrealized%" in open_detail_df.columns:
            base_cols = ["Source","Timeframe","Ticker","OpenQty","EntryPrice","PriceNow","Unrealized%","EntryTimeUTC","DaysOpen","SignalTimeUTC","SignalPrice","Account"]
        open_detail_df = open_detail_df[base_cols].sort_values(["Source","Timeframe","Ticker","EntryTimeUTC"]).reset_index(drop=True)

    # Equity curve by source
    equity_df = build_equity_curve_by_source(realized_df.copy())

    # Write tabs
    ws_real = open_ws(gc, TAB_REALIZED)
    ws_open = open_ws(gc, TAB_OPEN)
    ws_perf = open_ws(gc, TAB_PERF)
    ws_pt   = open_ws(gc, TAB_PERF_TICKER)
    ws_od   = open_ws(gc, TAB_OPEN_DETAIL)
    ws_eq   = open_ws(gc, TAB_EQUITY)

    write_tab(ws_real, realized_df)
    write_tab(ws_open, open_df)
    write_tab(ws_perf, perf_src_tf_df)
    write_tab(ws_pt,   perf_ticker_df)
    write_tab(ws_od,   open_detail_df)
    write_tab(ws_eq,   equity_df)

    print(f"✅ Wrote {TAB_REALIZED}: {len(realized_df)} rows")
    print(f"✅ Wrote {TAB_OPEN}: {len(open_df)} rows")
    print(f"✅ Wrote {TAB_PERF}: {len(perf_src_tf_df)} rows")
    print(f"✅ Wrote {TAB_PERF_TICKER}: {len(perf_ticker_df)} rows")
    print(f"✅ Wrote {TAB_OPEN_DETAIL}: {len(open_detail_df)} rows")
    print(f"✅ Wrote {TAB_EQUITY}: {len(equity_df)} rows")

    # Summaries
    if unmatched_sells:
        print(f"⚠️ Summary: {len(unmatched_sells)} unmatched SELL events (use --debug to print details).")
        if DEBUG:
            for line in unmatched_sells:
                print("  •", line)

    print_unknown_sources(realized_df, open_df)
    print_open_unknown_breakdown(open_df)
    print("🎯 Done.")

if __name__ == "__main__":
    main()
