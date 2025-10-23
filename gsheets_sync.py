# gsheets_sync.py
import os
import re
import json
import time
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ====== CONFIG ======
SHEET_NAME = "Trading Hub"  # <-- your Google Sheet name
CREDS_JSON = os.path.expanduser("~/WeinsteinAgent/creds/gcp_service_account.json")

# Tabs
TAB_HOLDINGS = "Holdings"
TAB_TXNS = "Transactions"
TAB_SIGNALS = "Signals_Log"
TAB_WEEKLY = "Weekly_Stages"
TAB_PERF = "Performance_By_Source"

# ====== AUTH ======
def gs_client():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
    return gspread.authorize(creds)

def open_sheet(client):
    return client.open(SHEET_NAME)

# ====== HELPERS ======
def df_to_ws(ws, df: pd.DataFrame, clear=True):
    # Write DataFrame to a worksheet
    if clear:
        ws.clear()
    # Convert to strings for safety
    df_out = df.copy()
    df_out = df_out.replace({np.nan: ""})
    rows = [list(df_out.columns)] + df_out.astype(str).values.tolist()
    ws.update(rows)

def append_rows(ws, df: pd.DataFrame):
    df_out = df.copy().replace({np.nan: ""})
    rows = df_out.astype(str).values.tolist()
    ws.append_rows(rows)

def ensure_headers(ws, headers):
    values = ws.get_all_values()
    if not values:
        ws.append_row(headers)
        return
    current = values[0]
    if current != headers:
        # Simple approach: overwrite row 1 with our headers if mismatch
        ws.update([headers] + values[1:])

def parse_currency(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def parse_percent(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("%","")
    try:
        return float(s)
    except:
        return np.nan

# ====== HOLDINGS SYNC ======
def sync_holdings(positions_csv: str):
    # Load Fidelity positions export
    df = pd.read_csv(positions_csv, engine="python")
    # Keep common useful columns if present
    keep = [
        "Account Number","Account Name","Symbol","Description","Quantity",
        "Last Price","Current Value","Total Gain/Loss Dollar","Total Gain/Loss Percent",
        "Cost Basis Total","Average Cost Basis","Type"
    ]
    present = [c for c in keep if c in df.columns]
    df = df[present].copy()

    # Normalization
    for c in ["Quantity","Last Price","Current Value","Total Gain/Loss Dollar","Cost Basis Total","Average Cost Basis"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_currency)
    if "Total Gain/Loss Percent" in df.columns:
        df["Total Gain/Loss Percent"] = df["Total Gain/Loss Percent"].apply(parse_percent)

    gc = gs_client()
    sh = open_sheet(gc)

    try:
        ws = sh.worksheet(TAB_HOLDINGS)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_HOLDINGS, rows=1000, cols=50)

    df_to_ws(ws, df)

# ====== TRANSACTIONS SYNC ======
def sync_transactions(history_csv: str):
    df = pd.read_csv(history_csv, engine="python")
    # Try to keep a standard set
    keep = [
        "Run Date","Account","Account Number","Action","Symbol","Description","Type",
        "Quantity","Price ($)","Commission ($)","Fees ($)","Amount ($)"
    ]
    present = [c for c in keep if c in df.columns]
    df = df[present].copy()

    # Parse numerics
    for c in ["Quantity","Price ($)","Commission ($)","Fees ($)","Amount ($)"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_currency)

    # Parse date
    if "Run Date" in df.columns:
        def _parse_date(x):
            for fmt in ("%m/%d/%Y","%Y-%m-%d","%m/%d/%y"):
                try:
                    return datetime.strptime(str(x), fmt)
                except:
                    pass
            return pd.NaT
        df["Run Date"] = df["Run Date"].apply(_parse_date)

    gc = gs_client()
    sh = open_sheet(gc)

    try:
        ws = sh.worksheet(TAB_TXNS)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_TXNS, rows=2000, cols=50)
        ensure_headers(ws, present)

    # If sheet empty → write headers + data; else append
    existing = ws.get_all_values()
    if not existing:
        df_to_ws(ws, df)
    else:
        # Make sure headers exist/match minimal
        ensure_headers(ws, existing[0] if existing else present)
        append_rows(ws, df)

# ====== WEEKLY STAGES SYNC ======
def sync_weekly_stages(weekly_csv: str, time_frame="weekly"):
    df = pd.read_csv(weekly_csv)
    # Add rank based on your existing buy_signal + dist_ma_pct
    if "stage" in df.columns:
        stage_rank = {
            "Stage 2 (Uptrend)": 0,
            "Stage 1 (Basing)": 1,
            "Stage 3 (Topping)": 2,
            "Stage 4 (Downtrend)": 3,
            "Filtered": 8,
            "N/A": 9,
        }
        df["stage_rank"] = df["stage"].map(stage_rank).fillna(9)
    else:
        df["stage_rank"] = 9
    if "dist_ma_pct" not in df.columns:
        df["dist_ma_pct"] = np.nan

    df = df.sort_values(by=["stage_rank","dist_ma_pct"], ascending=[True, False]).reset_index(drop=True)
    df["time_frame"] = time_frame

    gc = gs_client()
    sh = open_sheet(gc)
    try:
        ws = sh.worksheet(TAB_WEEKLY)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_WEEKLY, rows=5000, cols=50)

    df_to_ws(ws, df.drop(columns=["stage_rank"]))

# ====== APPEND SIGNAL ======
def append_signal(source: str, symbol: str, instrument="stock", rationale="", stage="", time_frame="daily",
                  suggested_price=None, stop=None, target=None, note=""):
    gc = gs_client()
    sh = open_sheet(gc)

    try:
        ws = sh.worksheet(TAB_SIGNALS)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_SIGNALS, rows=5000, cols=50)
        headers = ["date_time","source","symbol","instrument","rationale","stage","time_frame",
                   "suggested_price","stop","target","note"]
        ensure_headers(ws, headers)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "date_time": now,
        "source": source,
        "symbol": symbol.upper().strip(),
        "instrument": instrument,
        "rationale": rationale,
        "stage": stage,
        "time_frame": time_frame,
        "suggested_price": suggested_price if suggested_price is not None else "",
        "stop": stop if stop is not None else "",
        "target": target if target is not None else "",
        "note": note
    }])
    append_rows(ws, row)

# ====== PERFORMANCE BY SOURCE ======
def build_performance_by_source():
    gc = gs_client()
    sh = open_sheet(gc)

    # Load Transactions + Signals_Log
    try:
        tx_ws = sh.worksheet(TAB_TXNS)
        tx = pd.DataFrame(tx_ws.get_all_records())
    except gspread.exceptions.WorksheetNotFound:
        tx = pd.DataFrame()

    try:
        sg_ws = sh.worksheet(TAB_SIGNALS)
        sg = pd.DataFrame(sg_ws.get_all_records())
    except gspread.exceptions.WorksheetNotFound:
        sg = pd.DataFrame()

    if tx.empty or sg.empty:
        out = pd.DataFrame(columns=["source","picks","closed_trades","win_rate_pct","avg_return_pct","median_return_pct"])
        try:
            ws_out = sh.worksheet(TAB_PERF)
        except gspread.exceptions.WorksheetNotFound:
            ws_out = sh.add_worksheet(title=TAB_PERF, rows=100, cols=20)
        df_to_ws(ws_out, out)
        return

    # Normalize types
    if "Run Date" in tx.columns:
        def _parse_date(x):
            try:
                return pd.to_datetime(x)
            except:
                return pd.NaT
        tx["Run Date"] = tx["Run Date"].apply(_parse_date)

    tx["Symbol"] = tx.get("Symbol","").astype(str).str.upper().str.strip()
    sg["symbol"] = sg.get("symbol","").astype(str).str.upper().str.strip()

    # Identify BUYS and SELLS in Fidelity history (Action)
    # Fidelity “Action” text varies. Catch common ones.
    action = tx.get("Action","").astype(str).str.upper()
    tx["act_type"] = np.where(action.str.contains("BUY"), "BUY",
                       np.where(action.str.contains("SELL"), "SELL", ""))

    # Pair buys to subsequent sells per symbol → simple FIFO for P/L
    # Build per-symbol event list
    results = []
    for sym, grp in tx.sort_values("Run Date").groupby("Symbol"):
        stack = []  # list of (date, qty, price, fees)
        for _, r in grp.iterrows():
            if r.get("act_type") == "BUY":
                qty = r.get("Quantity", np.nan)
                px = r.get("Price ($)", np.nan)
                if pd.isna(qty) or pd.isna(px): 
                    continue
                stack.append([r.get("Run Date"), float(qty), float(px), float(r.get("Commission ($)",0) + r.get("Fees ($)",0))])
            elif r.get("act_type") == "SELL":
                sqty = r.get("Quantity", np.nan)
                spx = r.get("Price ($)", np.nan)
                if pd.isna(sqty) or pd.isna(spx): 
                    continue
                sell_qty = float(sqty)
                sell_dt  = r.get("Run Date")
                sell_fee = float(r.get("Commission ($)",0) + r.get("Fees ($)",0))
                # Match FIFO with stack
                while sell_qty > 0 and stack:
                    bdt, bqty, bpx, bfee = stack[0]
                    used = min(bqty, sell_qty)
                    # Return% on matched lot
                    ret = (spx - bpx) / bpx if bpx else np.nan
                    results.append({
                        "symbol": sym,
                        "buy_dt": bdt,
                        "sell_dt": sell_dt,
                        "buy_px": bpx,
                        "sell_px": spx,
                        "qty": used,
                        "return_pct": ret * 100.0
                    })
                    # decrement
                    bqty -= used
                    sell_qty -= used
                    if bqty <= 1e-9:
                        stack.pop(0)
                    else:
                        stack[0][1] = bqty

    res = pd.DataFrame(results)
    if res.empty:
        out = pd.DataFrame(columns=["source","picks","closed_trades","win_rate_pct","avg_return_pct","median_return_pct"])
        try:
            ws_out = sh.worksheet(TAB_PERF)
        except gspread.exceptions.WorksheetNotFound:
            ws_out = sh.add_worksheet(title=TAB_PERF, rows=100, cols=20)
        df_to_ws(ws_out, out)
        return

    # Attribute each trade to a source: take the nearest signal BEFORE the buy_dt
    sg["date_time"] = pd.to_datetime(sg.get("date_time", pd.NaT), errors="coerce")
    sg = sg.dropna(subset=["date_time"])
    sg = sg.sort_values("date_time")

    def attach_source(row):
        sym = row["symbol"]
        bdt = row["buy_dt"]
        pool = sg[(sg["symbol"] == sym) & (sg["date_time"] <= bdt)]
        if pool.empty:
            return pd.Series({"source":"(unattributed)", "signal_dt": pd.NaT})
        last = pool.iloc[-1]
        return pd.Series({"source": last.get("source",""), "signal_dt": last.get("date_time")})

    extra = res.apply(attach_source, axis=1)
    res = pd.concat([res, extra], axis=1)

    # Aggregate by source
    def _agg(g):
        closed = len(g)
        wins = (g["return_pct"] > 0).sum()
        win_rate = (wins / closed * 100.0) if closed else 0.0
        return pd.Series({
            "picks": g["symbol"].nunique(),
            "closed_trades": closed,
            "win_rate_pct": round(win_rate, 2),
            "avg_return_pct": round(float(g["return_pct"].mean()), 2),
            "median_return_pct": round(float(g["return_pct"].median()), 2),
        })

    perf = res.groupby("source", dropna=False).apply(_agg).reset_index()
    perf = perf.sort_values(["win_rate_pct","avg_return_pct"], ascending=[False, False])

    try:
        ws_out = sh.worksheet(TAB_PERF)
    except gspread.exceptions.WorksheetNotFound:
        ws_out = sh.add_worksheet(title=TAB_PERF, rows=200, cols=20)

    df_to_ws(ws_out, perf)

# ====== CLI usage examples ======
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Google Sheets sync for Trading Hub")
    ap.add_argument("--positions_csv", help="Path to Fidelity positions export (Portfolio_Positions_*.csv)")
    ap.add_argument("--history_csv", help="Path to Fidelity transactions export (Accounts_History.csv)")
    ap.add_argument("--weekly_csv", help="Path to weekly report CSV to push to Weekly_Stages")
    ap.add_argument("--append_signal", action="store_true", help="Append a single signal row")
    ap.add_argument("--source")
    ap.add_argument("--symbol")
    ap.add_argument("--instrument", default="stock")
    ap.add_argument("--rationale", default="")
    ap.add_argument("--stage", default="")
    ap.add_argument("--time_frame", default="daily")
    ap.add_argument("--suggested_price", type=float)
    ap.add_argument("--stop", type=float)
    ap.add_argument("--target", type=float)
    ap.add_argument("--note", default="")
    ap.add_argument("--build_perf", action="store_true", help="Rebuild Performance_By_Source")
    args = ap.parse_args()

    if args.positions_csv:
        sync_holdings(args.positions_csv)
        print("Holdings synced.")

    if args.history_csv:
        sync_transactions(args.history_csv)
        print("Transactions synced.")

    if args.weekly_csv:
        sync_weekly_stages(args.weekly_csv)
        print("Weekly_Stages synced.")

    if args.append_signal:
        if not (args.source and args.symbol):
            raise SystemExit("--append_signal requires --source and --symbol")
        append_signal(
            source=args.source,
            symbol=args.symbol,
            instrument=args.instrument,
            rationale=args.rationale,
            stage=args.stage,
            time_frame=args.time_frame,
            suggested_price=args.suggested_price,
            stop=args.stop,
            target=args.target,
            note=args.note
        )
        print("Signal appended.")

    if args.build_perf:
        build_performance_by_source()
        print("Performance_By_Source rebuilt.")
