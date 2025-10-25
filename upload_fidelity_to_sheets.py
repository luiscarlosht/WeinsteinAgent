#!/usr/bin/env python3
"""
Upload Fidelity CSVs (holdings + transactions) to Google Sheets.

Features:
- Pass file paths via CLI flags OR omit them and you'll get interactive prompts.
- Sanitizes data for Sheets/JSON (handles NaN/inf/datetimes).
- Creates tabs if missing; clears & resizes the tab.
- Uploads in chunks to avoid large request payloads.
- Optionally override Sheet URL and service-account path from CLI.

Examples:
  python3 upload_fidelity_to_sheets.py \
    --holdings ~/Downloads/Portfolio_Positions_2025-10-24.csv \
    --txns ~/Downloads/Accounts_History_2025-10-24.csv

  # Interactive (prompts for paths if omitted):
  python3 upload_fidelity_to_sheets.py
"""

import os
import sys
import argparse
import glob
from typing import Optional, List

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────
# DEFAULT CONFIG (can be overridden by CLI)
# ─────────────────────────────
DEFAULT_SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

DEFAULT_HOLDINGS_TAB = "Holdings"
DEFAULT_TXNS_TAB = "Transactions"

ROW_CHUNK = 500   # rows per API call


# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def authorize(sa_file: str):
    creds = Credentials.from_service_account_file(sa_file, scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc

def open_ws(gc, sheet_url: str, worksheet_name: str):
    sh = gc.open_by_url(sheet_url)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=100, cols=26)
    return ws

def _try_parse_datetimes_inplace(df: pd.DataFrame) -> None:
    """Best-effort parse of 'object' columns as datetimes; leaves others intact."""
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                parsed = pd.to_datetime(df[c])
                df[c] = parsed
            except Exception:
                # Keep as-is if not parseable
                pass

def sanitize_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace values that JSON + Sheets can't handle and convert to uploadable strings.
    - Replace +/-inf -> NaN; then fill NaN with ""
    - Convert datetimes to ISO strings
    - Format floats nicely (avoid scientific notation surprises)
    """
    df = df.copy()

    # Try to parse date-like columns to datetime
    _try_parse_datetimes_inplace(df)

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
    ws.clear()
    ws.resize(rows=max(n_rows, 100), cols=max(n_cols, 26))

def chunked_update(ws, values: List[List[str]]):
    """
    values: list of rows (list-of-lists). We send in chunks to avoid large payloads.
    Uses the new gspread signature: update(values, range_name=...)
    """
    if not values:
        return
    total_rows = len(values)
    start_row = 1
    while start_row <= total_rows:
        end_row = min(start_row + ROW_CHUNK - 1, total_rows)
        # Compute A1 range for this chunk
        top_left = gspread.utils.rowcol_to_a1(start_row, 1)
        n_cols = len(values[0]) if values else 1
        bottom_right = gspread.utils.rowcol_to_a1(end_row, n_cols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(values[start_row - 1 : end_row], range_name=rng)
        start_row = end_row + 1

def upload_csv_to_sheet(csv_path: str, gc, sheet_url: str, worksheet_name: str):
    print(f"📤 Uploading '{csv_path}' -> tab '{worksheet_name}'")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV (let pandas infer)
    df = pd.read_csv(csv_path)

    # Sanitize for Sheets/JSON
    df_clean = sanitize_for_sheets(df)

    # Open tab
    ws = open_ws(gc, sheet_url, worksheet_name)

    # Prepare values (header + rows)
    header = df_clean.columns.tolist()
    rows = df_clean.values.tolist()
    all_values = [header] + rows

    # Clear, size, and upload in chunks
    clear_and_size(ws, n_rows=len(all_values), n_cols=len(header))
    chunked_update(ws, all_values)
    print(f"✅ Uploaded {len(rows)} rows, {len(header)} columns to '{worksheet_name}'")

def prompt_for_path(prompt_text: str) -> str:
    """Interactive prompt that also supports globbing after entry."""
    p = input(prompt_text).strip()
    if not p:
        return ""
    # Expand ~
    p = os.path.expanduser(p)
    # If the user typed a glob, pick the first match to be nice
    matches = glob.glob(p)
    if matches:
        return matches[0]
    return p

def resolve_path(cli_value: Optional[str], label: str) -> str:
    """Return a real file path if provided, else prompt; blank stays blank."""
    if cli_value:
        path = os.path.expanduser(cli_value)
        return path
    # Ask user (can press Enter to skip)
    return prompt_for_path(f"Enter path for {label} CSV (or press Enter to skip): ")

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Upload Fidelity CSVs to Google Sheets (sanitized).")
    parser.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT_FILE, help="Path to service account JSON")
    parser.add_argument("--sheet-url", default=DEFAULT_SHEET_URL, help="Target Google Sheet URL")

    parser.add_argument("--holdings", help="Holdings CSV path")
    parser.add_argument("--txns", help="Transactions CSV path")

    parser.add_argument("--holdings-tab", default=DEFAULT_HOLDINGS_TAB, help="Sheet tab name for holdings")
    parser.add_argument("--txns-tab", default=DEFAULT_TXNS_TAB, help="Sheet tab name for transactions")

    args = parser.parse_args()

    # Resolve inputs (CLI or prompt)
    holdings_csv = resolve_path(args.holdings, "Holdings")
    txns_csv     = resolve_path(args.txns, "Transactions")

    if not holdings_csv and not txns_csv:
        print("Nothing to do: you didn't provide Holdings or Transactions CSV paths.")
        sys.exit(0)

    # Auth once
    print("🔑 Authorizing service account…")
    gc = authorize(args.service_account)

    # Upload whichever was provided
    if holdings_csv:
        upload_csv_to_sheet(holdings_csv, gc, args.sheet_url, args.holdings_tab)
    if txns_csv:
        upload_csv_to_sheet(txns_csv, gc, args.sheet_url, args.txns_tab)

    print("🎯 Done.")

if __name__ == "__main__":
    main()
