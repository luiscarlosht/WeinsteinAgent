#!/usr/bin/env python3
"""
Upload Fidelity CSVs (holdings + transactions) to Google Sheets.

- Sanitizes data for Sheets/JSON (handles NaN/inf/datetimes).
- Creates tabs if missing.
- Clears & resizes the tab.
- Uploads in chunks to avoid large request payloads.
"""

import os
import argparse
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

# Default CSV filenames (override with CLI flags if you want)
HOLDINGS_CSV = "Portfolio_Positions_Oct-23-2025.csv"
TXNS_CSV     = "Accounts_History.csv"

# Where to upload inside the Google Sheet
HOLDINGS_TAB = "Holdings"
TXNS_TAB     = "Transactions"

# Upload in chunks to keep each request payload small
ROW_CHUNK = 500   # rows per API call


# =========================
# HELPERS
# =========================
def authorize():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc

def open_ws(gc, worksheet_name: str):
    sh = gc.open_by_url(SHEET_URL)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        # create if missing
        ws = sh.add_worksheet(title=worksheet_name, rows=100, cols=26)
    return ws

def sanitize_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace values that JSON + Sheets can't handle and convert to uploadable strings.
    - Replace +/-inf -> NaN; then fill NaN with ""
    - Convert datetimes to ISO strings
    - Format floats nicely (avoid scientific notation surprises)
    """
    # Try to parse date-like columns to datetime (non-destructive if already str)
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                parsed = pd.to_datetime(df[c])  # let it raise if truly not parseable
                df[c] = parsed
            except Exception:
                # leave as-is if not parseable as dates
                pass

    # Replace infs
    df = df.replace([np.inf, -np.inf], np.nan)

    # If a column is datetime-like, stringify it
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    # For numeric columns, format floats; leave ints as-is
    def format_series(s: pd.Series) -> pd.Series:
        if pd.api.types.is_float_dtype(s):
            # round to 6 decimals, drop trailing zeros; keep empty for NaN
            return s.round(6).map(lambda x: "" if pd.isna(x) else ("{0:.6f}".format(x)).rstrip("0").rstrip("."))
        elif pd.api.types.is_integer_dtype(s):
            return s.map(lambda x: "" if pd.isna(x) else str(int(x)))
        else:
            return s

    df = df.apply(format_series)

    # Fill remaining NaN/None with empty string and ensure string type for safety
    df = df.fillna("").astype(str)

    return df

def clear_and_size(ws, n_rows: int, n_cols: int):
    # Resize to fit (Google likes knowing bounds) and clear previous contents
    ws.clear()
    ws.resize(rows=max(n_rows, 100), cols=max(n_cols, 26))

def chunked_update(ws, values):
    """
    values: list of rows (list-of-lists). We send in chunks to avoid large payloads.
    Uses the new gspread signature: update(values, range_name=...)
    """
    if not values:
        return
    # Write header + data in chunks
    total_rows = len(values)
    start_row = 1
    while start_row <= total_rows:
        end_row = min(start_row + ROW_CHUNK - 1, total_rows)
        # Compute A1 range for this chunk
        top_left = gspread.utils.rowcol_to_a1(start_row, 1)
        n_cols = len(values[0]) if values else 1
        bottom_right = gspread.utils.rowcol_to_a1(end_row, n_cols)
        rng = f"{top_left}:{bottom_right}"
        # NEW SIGNATURE (values first, then range_name)
        ws.update(values[start_row - 1 : end_row], range_name=rng)
        start_row = end_row + 1

def upload_csv_to_sheet(csv_path: str, worksheet_name: str):
    print(f"📤 Uploading '{csv_path}' -> tab '{worksheet_name}'")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV (let pandas infer)
    df = pd.read_csv(csv_path)

    # Sanitize for Sheets/JSON
    df_clean = sanitize_for_sheets(df)

    # Authorize + open tab
    gc = authorize()
    ws = open_ws(gc, worksheet_name)

    # Prepare values (header + rows)
    header = df_clean.columns.tolist()
    rows = df_clean.values.tolist()
    all_values = [header] + rows

    # Clear, size, and upload in chunks
    clear_and_size(ws, n_rows=len(all_values), n_cols=len(header))
    chunked_update(ws, all_values)
    print(f"✅ Uploaded {len(rows)} rows, {len(header)} columns to '{worksheet_name}'")

def main():
    parser = argparse.ArgumentParser(description="Upload Fidelity CSVs to Google Sheets (sanitized).")
    parser.add_argument("--holdings", default=HOLDINGS_CSV, help="Holdings CSV path")
    parser.add_argument("--txns", default=TXNS_CSV, help="Transactions CSV path")
    parser.add_argument("--holdings-tab", default=HOLDINGS_TAB, help="Sheet tab name for holdings")
    parser.add_argument("--txns-tab", default=TXNS_TAB, help="Sheet tab name for transactions")
    args = parser.parse_args()

    upload_csv_to_sheet(args.holdings, args.holdings_tab)
    upload_csv_to_sheet(args.txns, args.txns_tab)

if __name__ == "__main__":
    main()
