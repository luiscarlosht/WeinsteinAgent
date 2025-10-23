#!/usr/bin/env python3
"""
update_signals_and_score.py

Appends a signal (optional) to the "Signals" tab and recomputes source stats
by matching signals to fills in the "Transactions" tab.

Features:
- --backfill-days: if no fill is found at/after the signal timestamp, claim the
  most recent fill up to N days BEFORE the signal time.
- Robust parsing of Transactions: tolerant to empty/odd symbol cells like
  "Palantir Technologies Inc (PLTR)" or "PLTR - Palantir".
- Clean gspread .update(values, range_name) usage (no deprecation warnings).

Examples:
  # Append a PLTR BUY signal and recompute, allowing backfill 5 days
  python3 update_signals_and_score.py \
    --ticker PLTR --source SuperiorStar --direction BUY --price 32.15 \
    --timeframe short --backfill-days 5

  # Just recompute reports (no new signal)
  python3 update_signals_and_score.py --recompute-only --backfill-days 5
"""
import argparse
import re
import sys
import datetime as dt
from datetime import timezone, timedelta
from typing import List, Optional

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# ========= CONFIG =========
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS       = "Signals"
TAB_TRANSACTIONS  = "Transactions"
TAB_SOURCE_REPORT = "Source_Report"

# Signals header (do not change order unless you update reading logic)
SIGNALS_HEADER = ["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe"]

# Which transaction column names we try for each field
CANDIDATES = {
    "timestamp": ["Date/Time", "Date Time", "Trade Date", "Date", "Execution Time", "Timestamp"],
    "ticker":    ["Symbol", "Ticker", "Security Symbol", "Security", "Description", "Symbol/Description"],
    "action":    ["Action", "Type", "Activity", "Transaction Type"],
    "quantity":  ["Quantity", "Shares", "Qty"],
    "price":     ["Price", "Price ($)", "Fill Price", "Price Per Share"],
    "amount":    ["Amount", "Amount ($)", "Total", "Net Amount"],
}

# =============== GSPREAD BASICS ===============
def authorize():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc

def open_ws(gc, title: str):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=100, cols=26)

def get_df(ws) -> pd.DataFrame:
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    data = [r for r in data if any(c.strip() for c in r)]  # drop empty rows
    return pd.DataFrame(data, columns=header) if data else pd.DataFrame(columns=header)

def write_df(ws, df: pd.DataFrame, start_cell="A1"):
    # Clear & size
    n_rows = max(len(df) + 1, 2)
    n_cols = max(len(df.columns), 1)
    ws.clear()
    ws.resize(rows=max(n_rows, 50), cols=max(n_cols, 10))
    # Write header + data
    values = [df.columns.tolist()] + df.astype(str).fillna("").values.tolist()
    ws.update(values, start_cell)

def ensure_signals_header(ws):
    current = ws.get_all_values()
    if not current or (current and (not current[0] or current[0] != SIGNALS_HEADER)):
        ws.update([SIGNALS_HEADER], "A1")

# =============== TIME HELPERS ===============
def now_utc() -> dt.datetime:
    return dt.datetime.now(timezone.utc)

def parse_any_dt(s: str) -> Optional[dt.datetime]:
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    try:
        d = pd.to_datetime(s, utc=True, errors="raise")
        if isinstance(d, pd.Series):
            d = d.iloc[0]
        if d.tzinfo is None:
            d = d.tz_localize("UTC")
        return d.to_pydatetime()
    except Exception:
        pass
    fmts = [
        "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d"
    ]
    for f in fmts:
        try:
            d = dt.datetime.strptime(s, f)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d
        except Exception:
            continue
    return None

# =============== TRANSACTIONS PARSING ===============
def pick_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lowered = {c.lower(): c for c in cols}
    for k in keys:
        for c in cols:
            if c == k:
                return c
        for lc, orig in lowered.items():
            if k.lower() == lc or k.lower() in lc:
                return orig
    return None

_TICKER_TOKEN = re.compile(r"[A-Za-z][A-Za-z.\-]{0,9}$")  # up to 10 chars, letters/dot/dash

def extract_ticker(sym_raw: str) -> str:
    """
    Robust ticker extraction:
    - Try parenthesis first: "Company Name (PLTR)" -> PLTR
    - Then scan tokens split on whitespace/punctuation and return first token that
      looks like a ticker (A..Z with optional . or -), length <= 10.
    - Return "" if nothing reasonable.
    """
    s = (sym_raw or "").strip()
    if not s:
        return ""
    # Parentheses e.g. "Something (PLTR)"
    m = re.search(r"\(([A-Za-z.\-]{1,10})\)", s)
    if m:
        return m.group(1).upper()

    # Split by whitespace and common separators
    tokens = re.split(r"[\s/,:;|]+|-{1,}|–{1,}", s)
    for tok in tokens:
        t = re.sub(r"[^A-Za-z.\-]", "", tok).upper()
        if _TICKER_TOKEN.match(t):
            return t
    # Last resort: if the whole string looks like a ticker
    t = re.sub(r"[^A-Za-z.\-]", "", s).upper()
    if _TICKER_TOKEN.match(t):
        return t
    return ""

def normalize_action(s: str) -> str:
    if not s:
        return ""
    t = s.strip().lower()
    if "buy" in t or t == "b":
        return "BUY"
    if "sell" in t or t == "s":
        return "SELL"
    return t.upper()

def coerce_price(row, price_col: Optional[str], amount_col: Optional[str], qty_col: Optional[str]) -> Optional[float]:
    # Prefer explicit price
    if price_col and row.get(price_col, "") != "":
        try:
            return float(str(row[price_col]).replace(",", "").replace("$", ""))
        except Exception:
            pass
    # Derive from Amount / Quantity if possible
    if amount_col and qty_col:
        try:
            amt = float(str(row[amount_col]).replace(",", "").replace("$", ""))
            qty = float(str(row[qty_col]).replace(",", ""))
            if qty != 0:
                return abs(amt / qty)
        except Exception:
            pass
    return None

def load_transactions(ws_txn) -> pd.DataFrame:
    tx = get_df(ws_txn)
    if tx.empty:
        return tx

    # Identify columns
    c_ts   = pick_col(tx, CANDIDATES["timestamp"])
    c_sym  = pick_col(tx, CANDIDATES["ticker"])
    c_act  = pick_col(tx, CANDIDATES["action"])
    c_qty  = pick_col(tx, CANDIDATES["quantity"])
    c_px   = pick_col(tx, CANDIDATES["price"])
    c_amt  = pick_col(tx, CANDIDATES["amount"])

    for need, col in [("timestamp", c_ts), ("ticker", c_sym), ("action", c_act)]:
        if not col:
            raise RuntimeError(f"Transactions: could not find a '{need}' column (tried {CANDIDATES[need]})")

    # Build normalized frame
    out = []
    for _, r in tx.iterrows():
        ts = parse_any_dt(str(r.get(c_ts, "")))
        sym_raw = str(r.get(c_sym, "") or "").strip()
        ticker = extract_ticker(sym_raw)
        if not ticker:
            # Skip rows with no parseable ticker
            continue
        action = normalize_action(str(r.get(c_act, "")))
        if action not in ("BUY", "SELL"):
            continue

        qty = None
        if c_qty:
            qty_str = str(r.get(c_qty, "")).replace(",", "")
            try:
                qty = float(qty_str) if qty_str else None
            except Exception:
                qty = None

        px = coerce_price(r, c_px, c_amt, c_qty)

        if ts:
            out.append({
                "TimestampUTC": ts,
                "Ticker": ticker,
                "Action": action,
                "Quantity": qty,
                "Price": px
            })

    df = pd.DataFrame(out)
    if not df.empty:
        df.sort_values("TimestampUTC", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

# =============== SIGNALS ===============
def append_signal(ws_signals, sig_row: List[str]):
    ensure_signals_header(ws_signals)
    current = ws_signals.get_all_values()
    next_row = len(current) + 1 if current else 2
    rng = f"A{next_row}:F{next_row}"
    ws_signals.update([sig_row], rng)
    print(f"✅ Appended signal: {sig_row}")

def load_signals(ws_signals) -> pd.DataFrame:
    df = get_df(ws_signals)
    if df.empty:
        return pd.DataFrame(columns=SIGNALS_HEADER)
    for c in SIGNALS_HEADER:
        if c not in df.columns:
            df[c] = ""
    df = df[SIGNALS_HEADER].copy()
    df["TimestampUTC"] = df["TimestampUTC"].map(lambda s: parse_any_dt(s) or now_utc())
    df["Ticker"] = df["Ticker"].map(lambda s: str(s).strip().upper())
    df["Source"] = df["Source"].map(lambda s: str(s).strip())
    df["Direction"] = df["Direction"].map(lambda s: str(s).strip().upper())
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Timeframe"] = df["Timeframe"].map(lambda s: str(s).strip().lower())
    df.sort_values("TimestampUTC", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# =============== MATCHING LOGIC ===============
def find_entry_fill(tx: pd.DataFrame, ticker: str, direction: str,
                    t_sig: dt.datetime, fill_window_days: int, backfill_days: int) -> Optional[pd.Series]:
    if tx.empty:
        return None
    want_action = "BUY" if direction == "BUY" else "SELL"
    t_min = t_sig
    t_max = t_sig + timedelta(days=fill_window_days)
    mask_sym = (tx["Ticker"] == ticker) & (tx["Action"] == want_action)

    # First try: at/after signal
    cand = tx[mask_sym & (tx["TimestampUTC"] >= t_min) & (tx["TimestampUTC"] <= t_max)]
    if not cand.empty:
        return cand.iloc[0]

    # Backfill: nearest prior within window
    if backfill_days > 0:
        t_back_min = t_sig - timedelta(days=backfill_days)
        cand2 = tx[mask_sym & (tx["TimestampUTC"] < t_sig) & (tx["TimestampUTC"] >= t_back_min)]
        if not cand2.empty:
            return cand2.iloc[-1]

    return None

def find_exit_fill(tx: pd.DataFrame, ticker: str, direction: str,
                   t_entry: dt.datetime, close_window_days: int) -> Optional[pd.Series]:
    if tx.empty or t_entry is None:
        return None
    exit_action = "SELL" if direction == "BUY" else "BUY"
    t_max = t_entry + timedelta(days=close_window_days)
    cand = tx[(tx["Ticker"] == ticker) &
              (tx["Action"] == exit_action) &
              (tx["TimestampUTC"] >= t_entry) &
              (tx["TimestampUTC"] <= t_max)]
    if not cand.empty:
        return cand.iloc[0]
    return None

def compute_return_pct(direction: str, entry_price: float, exit_price: float) -> Optional[float]:
    if entry_price is None or exit_price is None:
        return None
    try:
        if direction == "BUY":
            return (exit_price - entry_price) / entry_price * 100.0
        else:
            return (entry_price - exit_price) / entry_price * 100.0
    except Exception:
        return None

# =============== REPORT ===============
def build_source_report(signals: pd.DataFrame, tx: pd.DataFrame,
                        fill_window_days: int, close_window_days: int,
                        backfill_days: int) -> pd.DataFrame:
    rows = []
    if signals.empty:
        return pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenSignals"])

    for src, sdf in signals.groupby("Source", dropna=False):
        trades = 0
        wins = 0
        rets: List[float] = []
        open_signals = 0

        for _, s in sdf.iterrows():
            ticker = s["Ticker"]
            direction = s["Direction"]
            t_sig = s["TimestampUTC"]

            ent = find_entry_fill(tx, ticker, direction, t_sig, fill_window_days, backfill_days)
            if ent is None:
                open_signals += 1
                continue

            ext = find_exit_fill(tx, ticker, direction, ent["TimestampUTC"], close_window_days)
            if ext is None:
                open_signals += 1
                continue

            trades += 1
            r = compute_return_pct(direction, ent.get("Price"), ext.get("Price"))
            if r is not None:
                rets.append(r)
                if r >= 0:
                    wins += 1

        winrate = (wins / trades * 100.0) if trades > 0 else 0.0
        avg_ret = (sum(rets) / len(rets)) if rets else 0.0
        med_ret = (float(pd.Series(rets).median()) if rets else 0.0)

        rows.append({
            "Source": src if src else "",
            "Trades": trades,
            "Wins": wins,
            "WinRate%": round(winrate, 2),
            "AvgReturn%": round(avg_ret, 2),
            "MedianReturn%": round(med_ret, 2),
            "OpenSignals": open_signals
        })

    rep = pd.DataFrame(rows).sort_values("Source").reset_index(drop=True)
    return rep

# =============== MAIN ===============
def main():
    ap = argparse.ArgumentParser(description="Append signals and recompute source report.")
    ap.add_argument("--ticker", type=str, help="Ticker (e.g., PLTR)")
    ap.add_argument("--source", type=str, help="Source (Sarkee, SuperiorStar, Weinstein, Bo, etc.)")
    ap.add_argument("--direction", type=str, choices=["BUY","SELL"], help="BUY or SELL")
    ap.add_argument("--price", type=float, help="Signal ref price (optional)")
    ap.add_argument("--timeframe", type=str, default="", help="short/mid/long or custom")
    ap.add_argument("--timestamp", type=str, default="", help="Explicit signal timestamp (UTC). If omitted, uses now.")
    ap.add_argument("--recompute-only", action="store_true", help="Skip appending a new signal; recompute stats only.")
    ap.add_argument("--fill-window-days", type=int, default=10, help="Max days after signal to look for entry fill.")
    ap.add_argument("--close-window-days", type=int, default=120, help="Max days after entry to look for exit fill.")
    ap.add_argument("--backfill-days", type=int, default=0, help="If no entry after signal, allow most recent prior fill within N days.")
    args = ap.parse_args()

    gc = authorize()
    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_txn = open_ws(gc, TAB_TRANSACTIONS)
    ws_rep = open_ws(gc, TAB_SOURCE_REPORT)

    # Append signal (unless recompute-only)
    if not args.recompute_only:
        for need, val in [("ticker", args.ticker), ("source", args.source), ("direction", args.direction)]:
            if not val:
                print(f"ERROR: --{need} is required unless --recompute-only", file=sys.stderr)
                sys.exit(1)

        t_sig = parse_any_dt(args.timestamp) if args.timestamp else now_utc()
        t_str = t_sig.strftime("%Y-%m-%d %H:%M:%S%z")
        ensure_signals_header(ws_sig)
        sig_row = [
            t_str,
            args.ticker.strip().upper(),
            args.source.strip(),
            args.direction.strip().upper(),
            "" if args.price is None else f"{args.price}",
            args.timeframe.strip().lower()
        ]
        append_signal(ws_sig, sig_row)

    # Load data
    signals = load_signals(ws_sig)
    tx = load_transactions(ws_txn)

    # Build and write report
    rep = build_source_report(signals, tx, args.fill_window_days, args.close_window_days, args.backfill_days)
    if rep.empty:
        rep = pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%","OpenSignals"])
    write_df(ws_rep, rep)
    print("📈 Source_Report updated.")
    if not rep.empty:
        print(rep.to_string(index=False))

if __name__ == "__main__":
    main()
