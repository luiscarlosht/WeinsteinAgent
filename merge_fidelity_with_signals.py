#!/usr/bin/env python3
"""
Merge Fidelity Transactions/Holdings into Signals:
- Fills missing TimestampUTC, Direction, Price from latest matching transaction
- Fills missing Timeframe using a default map per Source
- Leaves any existing values in Signals as-is
"""

import re
from datetime import timezone
import pandas as pd
import numpy as np
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

TAB_SIGNALS = "Signals"
TAB_TXNS    = "Transactions"
TAB_HOLD    = "Holdings"

# If Timeframe is blank in Signals, use this default by Source
SOURCE_DEFAULT_TIMEFRAME = {
    "Sarkee Capital": "short",
    "SuperiorStar": "mid",
    "Weinstein": "long",
    "Bo Xu": "short",
    # add your own keys as needed
}

# Acceptable headers in Signals (we’ll preserve extra columns if present)
CORE_COLS = ["TimestampUTC", "Ticker", "Source", "Direction", "Price", "Timeframe", "Notes"]


# =========================
# GSpread helpers
# =========================
def authorize():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, name):
    sh = gc.open_by_url(SHEET_URL)
    return sh.worksheet(name)

def ws_to_df(ws: gspread.Worksheet) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    # drop fully empty rows
    rows = [r for r in rows if any(c.strip() for c in r)]
    df = pd.DataFrame(rows, columns=header[:len(rows[0])]) if rows else pd.DataFrame(columns=header)
    # align width (some rows may be shorter)
    if rows:
        max_w = max(len(r) for r in rows)
        if len(header) < max_w:
            header = header + [f"__extra_{i}" for i in range(max_w - len(header))]
            df = pd.DataFrame([ (r + [""]*(max_w - len(r))) for r in rows ], columns=header)
    return df

def df_to_sheet(ws: gspread.Worksheet, df: pd.DataFrame):
    # ensure strings, fill NaNs, keep current width
    out = df.fillna("").astype(str)
    data = [list(out.columns)] + out.values.tolist()
    ws.clear()
    ws.update(range_name="A1", values=data)


# =========================
# Parsing helpers (Fidelity quirks)
# =========================
TICKER_PAREN_RE = re.compile(r"\((?P<ticker>[A-Z][A-Z0-9\.\-]{0,6})\)")

def extract_ticker(symbol: str, desc: str) -> str:
    """
    Prefer Symbol column; if empty, try to pull '(TICKER)' from Description.
    """
    sym = (symbol or "").strip()
    if sym:
        return sym.split()[0].split("-")[0].strip().upper()
    m = TICKER_PAREN_RE.search(desc or "")
    return m.group("ticker") if m else ""

def map_direction(action: str) -> str:
    a = (action or "").upper()
    if "BUY" in a:
        return "BUY"
    if "SELL" in a:
        return "SELL"
    # dividends, reinvestment, interest, etc. -> not a trade direction
    return ""

def parse_price(raw: str) -> str:
    try:
        x = float(str(raw).replace(",", ""))
        if np.isfinite(x):
            # format cleanly, strip trailing zeros
            s = f"{x:.6f}".rstrip("0").rstrip(".")
            return s
    except Exception:
        pass
    return ""

def parse_timestamp(run_date: str, settle_date: str) -> str:
    # Prefer "Run Date"; fallback to "Settlement Date"
    for cand in [run_date, settle_date]:
        if cand and cand.strip():
            try:
                dt = pd.to_datetime(cand)
                # leave naive as-is; if tz-aware, normalize to UTC ISO
                if dt.tzinfo is not None:
                    return dt.tz_convert(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
                else:
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
    return ""


# =========================
# Build latest-transaction lookup
# =========================
def build_txn_lookup(df_tx: pd.DataFrame):
    """
    From Transactions df, build a dict:
       latest_by_ticker = { "AAPL": {"Direction": "BUY", "Price": "192.33", "TimestampUTC": "..."}, ... }
    Only keeps rows with BUY/SELL.
    Uses the most recent timestamp per ticker.
    """
    if df_tx.empty:
        return {}

    # Normalize columns we might see
    cols = {c.lower(): c for c in df_tx.columns}
    col_run   = cols.get("run date", None) or cols.get("run_date", None) or "Run Date"
    col_set   = cols.get("settlement date", None) or "Settlement Date"
    col_act   = cols.get("action", None) or "Action"
    col_sym   = cols.get("symbol", None) or "Symbol"
    col_desc  = cols.get("description", None) or "Description"
    col_price = cols.get("price ($)", None) or "Price ($)"

    # Drop footer / policy lines (no dates)
    df = df_tx.copy()
    # Compute parsed fields
    df["__Ticker"]     = [extract_ticker(df.at[i, col_sym] if col_sym in df.columns else "",
                                         df.at[i, col_desc] if col_desc in df.columns else "")
                          for i in df.index]
    df["__Direction"]  = [map_direction(df.at[i, col_act] if col_act in df.columns else "")
                          for i in df.index]
    df["__Price"]      = [parse_price(df.at[i, col_price]) if col_price in df.columns else "" for i in df.index]
    df["__Timestamp"]  = [parse_timestamp(df.at[i, col_run] if col_run in df.columns else "",
                                          df.at[i, col_set] if col_set in df.columns else "")
                          for i in df.index]

    # keep only trades with direction + ticker
    df = df[(df["__Ticker"] != "") & (df["__Direction"] != "")]
    if df.empty:
        return {}

    # Use timestamp ordering to pick latest per ticker
    # (coerce to datetime where possible)
    def _to_dt(s):
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    df["__dt"] = df["__Timestamp"].map(_to_dt)
    df = df.sort_values(["__Ticker", "__dt"], ascending=[True, True])

    latest = {}
    for tkr, grp in df.groupby("__Ticker", sort=False):
        last = grp.iloc[-1]
        latest[tkr] = {
            "Direction": last["__Direction"],
            "Price": last["__Price"],
            "TimestampUTC": last["__Timestamp"],
        }
    return latest


# =========================
# Signals merge
# =========================
def ensure_signals_header(signals_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame(columns=CORE_COLS)
    # Make sure the core columns exist
    for col in CORE_COLS:
        if col not in signals_df.columns:
            signals_df[col] = ""
    # Keep any extra user columns too (preserve order: core first, then extras)
    extras = [c for c in signals_df.columns if c not in CORE_COLS]
    return signals_df[CORE_COLS + extras]

def merge_into_signals(df_sig: pd.DataFrame, latest_txn: dict) -> pd.DataFrame:
    df = df_sig.copy()
    n_rows = len(df)
    filled = 0

    for i in range(n_rows):
        tkr = (df.at[i, "Ticker"] if "Ticker" in df.columns else "").strip().upper()
        if not tkr:
            continue

        # Fill Timeframe from Source default if empty
        if "Timeframe" in df.columns:
            if not str(df.at[i, "Timeframe"]).strip():
                src = str(df.at[i, "Source"]).strip()
                if src in SOURCE_DEFAULT_TIMEFRAME:
                    df.at[i, "Timeframe"] = SOURCE_DEFAULT_TIMEFRAME[src]

        # If we have a recent txn for this ticker, backfill missing fields
        info = latest_txn.get(tkr)
        if info:
            if "Direction" in df.columns and not str(df.at[i, "Direction"]).strip():
                df.at[i, "Direction"] = info.get("Direction", "")
                filled += 1
            if "Price" in df.columns and not str(df.at[i, "Price"]).strip():
                df.at[i, "Price"] = info.get("Price", "")
                filled += 1
            if "TimestampUTC" in df.columns and not str(df.at[i, "TimestampUTC"]).strip():
                df.at[i, "TimestampUTC"] = info.get("TimestampUTC", "")
                filled += 1

    return df, filled


# =========================
# Main
# =========================
def main():
    gc = authorize()

    print("📄 Reading tabs…")
    ws_sig = open_ws(gc, TAB_SIGNALS)
    ws_tx  = open_ws(gc, TAB_TXNS)
    # holdings optional (not used for backfill yet, but available)
    try:
        ws_h = open_ws(gc, TAB_HOLD)
    except Exception:
        ws_h = None

    df_sig = ws_to_df(ws_sig)
    df_tx  = ws_to_df(ws_tx)

    df_sig = ensure_signals_header(df_sig)

    print(f"• Signals rows: {len(df_sig)}")
    print(f"• Transactions rows: {len(df_tx)}")

    latest_txn = build_txn_lookup(df_tx)
    print(f"• Latest tickers from transactions: {len(latest_txn)}")

    df_merged, filled = merge_into_signals(df_sig, latest_txn)

    if filled == 0:
        print("ℹ️ No fields needed backfilling (or no matching transactions). Nothing to write.")
        return

    print(f"✏️ Backfilled {filled} empty fields in Signals. Writing back…")
    df_to_sheet(ws_sig, df_merged)
    print("✅ Done. Check the Signals tab.")

if __name__ == "__main__":
    main()
