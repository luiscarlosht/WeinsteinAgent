#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_fidelity_exports_to_google_sheet.py

Purpose
-------
Import Fidelity all-account exports into the Trading Hub Google Sheet.

Recommended workflow:
  1. Export ONE all-accounts Portfolio Positions CSV from Fidelity.
  2. Export ONE all-accounts Account History CSV from Fidelity.
  3. Dry-run this script to validate classification and preview output.
  4. Write to Google Sheets only after dry-run looks correct.

Outputs / Google Sheet tabs:
  - Holdings              all holdings, normalized + derived fields
  - CryptoHoldings        crypto-only holdings
  - Transactions          all account history, normalized + derived fields
  - CryptoTransactions    crypto-only account history

Why this exists
---------------
The parity report currently shows "Positions loaded = 0" when no positions
source is provided. This script makes Google Sheets the source of truth by
loading Fidelity exports into predictable tabs that the PROD watcher and parity
tools can read.

Examples
--------
Dry-run only:

  python3 sync_fidelity_exports_to_google_sheet.py \
    --positions-csv ./Portfolio_Positions_All_Accounts_Jun-04-2026.csv \
    --history-csv ./Accounts_History_All_Accounts.csv \
    --dry-run

Write to Google Sheets:

  python3 sync_fidelity_exports_to_google_sheet.py \
    --positions-csv ./Portfolio_Positions_All_Accounts_Jun-04-2026.csv \
    --history-csv ./Accounts_History_All_Accounts.csv \
    --write-sheet

Dependencies
------------
  pip install pandas numpy pyyaml gspread google-auth
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_HOLDINGS_TAB = "Holdings"
DEFAULT_CRYPTO_HOLDINGS_TAB = "CryptoHoldings"
DEFAULT_TRANSACTIONS_TAB = "Transactions"
DEFAULT_CRYPTO_TRANSACTIONS_TAB = "CryptoTransactions"

ROW_CHUNK = 500


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_sheet_url(cfg: dict[str, Any], cli_sheet_url: str | None) -> str | None:
    if cli_sheet_url:
        return cli_sheet_url
    sheets = cfg.get("sheets", {}) or {}
    return (
        sheets.get("url")
        or sheets.get("sheet_url")
        or os.getenv("WEINSTEIN_SHEET_URL")
        or os.getenv("SHEET_URL")
    )


def resolve_service_account(cfg: dict[str, Any], cli_service_account: str | None) -> str | None:
    candidates = []

    if cli_service_account:
        candidates.append(cli_service_account)

    google_cfg = cfg.get("google", {}) or {}
    if google_cfg.get("service_account_json"):
        candidates.append(google_cfg["service_account_json"])

    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        candidates.append(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    # Common repo-local fallback.
    candidates.append(str(repo_root() / "creds" / "gcp_service_account.json"))

    # Google VM / DigitalOcean path compatibility.
    candidates.append("/home/luiscarlosht/WeinsteinAgent/creds/gcp_service_account.json")
    candidates.append("/root/WeinsteinAgent/creds/gcp_service_account.json")

    for c in candidates:
        if c and Path(c).expanduser().exists():
            return str(Path(c).expanduser())

    return cli_service_account or google_cfg.get("service_account_json") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


# ---------------------------------------------------------------------------
# CSV reading helpers
# ---------------------------------------------------------------------------

def detect_header_row(csv_path: str | Path, required_any: Iterable[str]) -> int:
    required = {x.lower() for x in required_any}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            cells = {str(c).strip().lower() for c in row if str(c).strip()}
            if cells & required:
                return i
    return 0


def read_fidelity_positions(csv_path: str | Path) -> pd.DataFrame:
    header_row = detect_header_row(csv_path, ["Account Number", "Symbol", "Current Value"])
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        skiprows=header_row,
        index_col=False,
        engine="python",
        on_bad_lines="skip",
    )
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    df = df.dropna(how="all")

    if "Account Number" in df.columns:
        df = df[df["Account Number"].astype(str).str.strip().str.match(r"^[A-Z0-9]+$", na=False)]

    return df.reset_index(drop=True)


def read_fidelity_history(csv_path: str | Path) -> pd.DataFrame:
    header_row = detect_header_row(csv_path, ["Run Date", "Action", "Settlement Date"])
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        skiprows=header_row,
        index_col=False,
        engine="python",
        on_bad_lines="skip",
    )
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    df = df.dropna(how="all")

    if "Run Date" in df.columns:
        df = df[df["Run Date"].astype(str).str.strip().str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normalization / classification
# ---------------------------------------------------------------------------

def money_to_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "--"}:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    s = s.replace("+", "")

    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None


def symbol_clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def is_cash_symbol(symbol: str, description: str = "") -> bool:
    s = symbol_clean(symbol)
    d = str(description or "").upper()
    return (
        not s
        or s in {"USD", "USD***", "CASH", "SPAXX", "CORE"}
        or "US DOLLARS" in d
        or "FDIC INSURED DEPOSIT" in d
    )


def is_crypto_symbol(symbol: str, account_name: str = "", description: str = "") -> bool:
    s = symbol_clean(symbol)
    acct = str(account_name or "").upper()
    desc = str(description or "").upper()

    crypto_tokens = {
        "BTC", "BTC/USD", "BITCOIN",
        "ETH", "ETH/USD", "ETHEREUM",
        "SOL", "SOL/USD", "SOLANA",
        "LTC", "LTC/USD", "LITECOIN",
        "DOGE", "DOGE/USD",
        "ADA", "ADA/USD",
        "XRP", "XRP/USD",
    }

    if "CRYPTO" in acct:
        return True
    if s in crypto_tokens:
        return True
    if desc in crypto_tokens:
        return True
    if s.endswith("/USD"):
        return True
    return False


def crypto_yfinance_symbol(symbol: str) -> str:
    s = symbol_clean(symbol)
    if s in {"BTC", "BITCOIN"}:
        return "BTC-USD"
    if s in {"ETH", "ETHEREUM"}:
        return "ETH-USD"
    if s in {"SOL", "SOLANA"}:
        return "SOL-USD"
    if s in {"LTC", "LITECOIN"}:
        return "LTC-USD"
    if s.endswith("/USD"):
        return s.replace("/USD", "-USD")
    return s


def classify_account_group(account_name: str, account_number: str = "") -> str:
    name = str(account_name or "").upper()
    number = str(account_number or "").upper()

    if "CRYPTO" in name:
        return "Crypto"
    if "401" in name or "RETIRE" in name or "IRA" in name or "ROTH" in name:
        return "Retirement"
    if "INDIVIDUAL" in name or number.startswith("X") or number.startswith("Z"):
        return "Brokerage"
    return "Other"


def add_common_metadata(df: pd.DataFrame, source_file: str, source_type: str) -> pd.DataFrame:
    out = df.copy()
    out["SourceFile"] = Path(source_file).name
    out["SourceType"] = source_type
    out["ImportedAtUTC"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return out


def normalize_positions(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = add_common_metadata(df, source_file, "FidelityPositions")

    account_number = out.get("Account Number", pd.Series([""] * len(out))).astype(str)
    account_name = out.get("Account Name", pd.Series([""] * len(out))).astype(str)
    symbol = out.get("Symbol", pd.Series([""] * len(out))).astype(str)
    description = out.get("Description", pd.Series([""] * len(out))).astype(str)

    out["AccountGroup"] = [
        classify_account_group(a, n) for a, n in zip(account_name, account_number)
    ]
    out["IsCrypto"] = [
        is_crypto_symbol(s, a, d) for s, a, d in zip(symbol, account_name, description)
    ]
    out["IsCash"] = [
        is_cash_symbol(s, d) for s, d in zip(symbol, description)
    ]
    out["AssetClass"] = np.where(out["IsCrypto"], "Crypto", np.where(out["IsCash"], "Cash", "Equity"))
    out["TradableForWeinstein"] = np.where(
        (out["AssetClass"] == "Equity") & (out["AccountGroup"] == "Brokerage"),
        True,
        False,
    )
    out["TradableForCryptoWatcher"] = np.where(out["AssetClass"] == "Crypto", True, False)
    out["NormalizedSymbol"] = [
        crypto_yfinance_symbol(s) if crypto else symbol_clean(s)
        for s, crypto in zip(symbol, out["IsCrypto"])
    ]

    # Numeric normalized columns for downstream code.
    if "Quantity" in out.columns:
        out["QuantityNum"] = out["Quantity"].map(money_to_float)
    if "Current Value" in out.columns:
        out["CurrentValueNum"] = out["Current Value"].map(money_to_float)
    if "Cost Basis Total" in out.columns:
        out["CostBasisTotalNum"] = out["Cost Basis Total"].map(money_to_float)
    if "Average Cost Basis" in out.columns:
        out["AverageCostBasisNum"] = out["Average Cost Basis"].map(money_to_float)
    if "Last Price" in out.columns:
        out["LastPriceNum"] = out["Last Price"].map(money_to_float)

    return out


def normalize_history(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = add_common_metadata(df, source_file, "FidelityHistory")

    account_number = out.get("Account Number", pd.Series([""] * len(out))).astype(str)
    account_name = out.get("Account", pd.Series([""] * len(out))).astype(str)
    symbol = out.get("Symbol", pd.Series([""] * len(out))).astype(str)
    description = out.get("Description", pd.Series([""] * len(out))).astype(str)
    action = out.get("Action", pd.Series([""] * len(out))).astype(str)

    out["AccountGroup"] = [
        classify_account_group(a, n) for a, n in zip(account_name, account_number)
    ]
    out["IsCrypto"] = [
        is_crypto_symbol(s, a, d) for s, a, d in zip(symbol, account_name, description)
    ]
    out["IsCash"] = [
        is_cash_symbol(s, d) for s, d in zip(symbol, description)
    ]
    out["AssetClass"] = np.where(out["IsCrypto"], "Crypto", np.where(out["IsCash"], "Cash", "Equity"))
    out["NormalizedSymbol"] = [
        crypto_yfinance_symbol(s) if crypto else symbol_clean(s)
        for s, crypto in zip(symbol, out["IsCrypto"])
    ]

    out["TxnActionNormalized"] = action.str.upper().map(classify_transaction_action)

    if "Quantity" in out.columns:
        out["QuantityNum"] = out["Quantity"].map(money_to_float)
    if "Amount ($)" in out.columns:
        out["AmountNum"] = out["Amount ($)"].map(money_to_float)
    if "Price ($)" in out.columns:
        out["PriceNum"] = out["Price ($)"].map(money_to_float)
    if "Fees ($)" in out.columns:
        out["FeesNum"] = out["Fees ($)"].map(money_to_float)

    return out


def classify_transaction_action(action_upper: str) -> str:
    a = str(action_upper or "").upper()
    if "BOUGHT" in a or "BUY" in a:
        return "BUY"
    if "SOLD" in a or "SELL" in a:
        return "SELL"
    if "DIVIDEND" in a:
        return "DIVIDEND"
    if "INTEREST" in a:
        return "INTEREST"
    if "TRANSFER" in a:
        return "TRANSFER"
    if "DEPOSIT" in a:
        return "DEPOSIT"
    if "WITHDRAW" in a:
        return "WITHDRAWAL"
    return "OTHER"


# ---------------------------------------------------------------------------
# Google Sheets upload helpers
# ---------------------------------------------------------------------------

def authorize(service_account_file: str):
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    return gspread.authorize(creds)


def open_ws(gc, sheet_url: str, worksheet_name: str):
    import gspread

    sh = gc.open_by_url(sheet_url)
    try:
        return sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=worksheet_name, rows=100, cols=26)


def sanitize_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    def clean_cell(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (np.integer, int)):
            return str(int(v))
        if isinstance(v, (np.floating, float)):
            return ("{0:.8f}".format(float(v))).rstrip("0").rstrip(".")
        if isinstance(v, (bool, np.bool_)):
            return "TRUE" if bool(v) else "FALSE"
        return str(v)

    for c in out.columns:
        out[c] = out[c].map(clean_cell)

    return out


def chunked_update(ws, values: list[list[str]], chunk_size: int = ROW_CHUNK):
    import gspread

    if not values:
        return

    n_cols = len(values[0]) if values else 1
    total_rows = len(values)

    start_row = 1
    while start_row <= total_rows:
        end_row = min(start_row + chunk_size - 1, total_rows)
        top_left = gspread.utils.rowcol_to_a1(start_row, 1)
        bottom_right = gspread.utils.rowcol_to_a1(end_row, n_cols)
        rng = f"{top_left}:{bottom_right}"
        ws.update(values[start_row - 1:end_row], range_name=rng)
        start_row = end_row + 1


def write_df_to_sheet(gc, sheet_url: str, tab_name: str, df: pd.DataFrame):
    print(f"📤 Writing {len(df)} rows -> Google Sheet tab '{tab_name}'")
    ws = open_ws(gc, sheet_url, tab_name)
    clean = sanitize_for_sheets(df)

    values = [clean.columns.tolist()] + clean.values.tolist()

    ws.clear()
    ws.resize(rows=max(len(values), 100), cols=max(len(clean.columns), 26))
    chunked_update(ws, values)
    print(f"✅ Wrote tab '{tab_name}'")


# ---------------------------------------------------------------------------
# Reporting / preview
# ---------------------------------------------------------------------------

def print_positions_summary(df: pd.DataFrame, title: str):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Rows: {len(df)}")

    if df.empty:
        return

    if "AssetClass" in df.columns:
        print("\nBy AssetClass:")
        print(df.groupby("AssetClass").size().to_string())

    if "AccountGroup" in df.columns:
        print("\nBy AccountGroup:")
        print(df.groupby("AccountGroup").size().to_string())

    if {"Account Name", "AssetClass"}.issubset(df.columns):
        print("\nBy Account Name / AssetClass:")
        print(df.groupby(["Account Name", "AssetClass"]).size().to_string())

    if "CurrentValueNum" in df.columns and "AssetClass" in df.columns:
        print("\nCurrentValue by AssetClass:")
        print(df.groupby("AssetClass")["CurrentValueNum"].sum(min_count=1).round(2).to_string())


def print_history_summary(df: pd.DataFrame, title: str):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Rows: {len(df)}")

    if df.empty:
        return

    if "AssetClass" in df.columns:
        print("\nBy AssetClass:")
        print(df.groupby("AssetClass").size().to_string())

    if "TxnActionNormalized" in df.columns:
        print("\nBy Action:")
        print(df.groupby("TxnActionNormalized").size().to_string())

    if {"Account", "AssetClass"}.issubset(df.columns):
        print("\nBy Account / AssetClass:")
        print(df.groupby(["Account", "AssetClass"]).size().to_string())


def save_preview(output_dir: Path, name: str, df: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    df.to_csv(path, index=False)
    print(f"📝 Preview CSV -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Sync Fidelity all-account exports to Google Sheet tabs.")
    ap.add_argument("--positions-csv", help="Fidelity Portfolio Positions CSV, ideally all accounts")
    ap.add_argument("--history-csv", help="Fidelity Account History CSV, ideally all accounts")
    ap.add_argument("--config", default="config.yaml", help="config.yaml path")
    ap.add_argument("--sheet-url", help="Google Sheet URL override")
    ap.add_argument("--service-account", help="Google service account JSON override")
    ap.add_argument("--dry-run", action="store_true", help="Preview only; do not write Google Sheets")
    ap.add_argument("--write-sheet", action="store_true", help="Actually write Google Sheets")
    ap.add_argument("--preview-dir", default=None, help="Directory for preview CSVs")
    ap.add_argument("--holdings-tab", default=DEFAULT_HOLDINGS_TAB)
    ap.add_argument("--crypto-holdings-tab", default=DEFAULT_CRYPTO_HOLDINGS_TAB)
    ap.add_argument("--transactions-tab", default=DEFAULT_TRANSACTIONS_TAB)
    ap.add_argument("--crypto-transactions-tab", default=DEFAULT_CRYPTO_TRANSACTIONS_TAB)

    args = ap.parse_args()

    if not args.positions_csv and not args.history_csv:
        print("Nothing to do. Provide --positions-csv and/or --history-csv.")
        return 1

    if not args.dry_run and not args.write_sheet:
        print("Safety stop: choose --dry-run or --write-sheet.")
        return 1

    cfg = load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    preview_dir = Path(args.preview_dir) if args.preview_dir else repo_root() / "output" / "fidelity_sync_preview" / stamp

    holdings_df = pd.DataFrame()
    crypto_holdings_df = pd.DataFrame()
    history_df = pd.DataFrame()
    crypto_history_df = pd.DataFrame()

    if args.positions_csv:
        if not Path(args.positions_csv).expanduser().exists():
            raise FileNotFoundError(f"Positions CSV not found: {args.positions_csv}")

        raw_positions = read_fidelity_positions(Path(args.positions_csv).expanduser())
        holdings_df = normalize_positions(raw_positions, args.positions_csv)
        crypto_holdings_df = holdings_df[holdings_df["AssetClass"] == "Crypto"].copy()

        print_positions_summary(holdings_df, "Holdings Summary")
        print_positions_summary(crypto_holdings_df, "Crypto Holdings Summary")

        save_preview(preview_dir, "Holdings_preview.csv", holdings_df)
        save_preview(preview_dir, "CryptoHoldings_preview.csv", crypto_holdings_df)

    if args.history_csv:
        if not Path(args.history_csv).expanduser().exists():
            raise FileNotFoundError(f"History CSV not found: {args.history_csv}")

        raw_history = read_fidelity_history(Path(args.history_csv).expanduser())
        history_df = normalize_history(raw_history, args.history_csv)
        crypto_history_df = history_df[history_df["AssetClass"] == "Crypto"].copy()

        print_history_summary(history_df, "Transactions Summary")
        print_history_summary(crypto_history_df, "Crypto Transactions Summary")

        save_preview(preview_dir, "Transactions_preview.csv", history_df)
        save_preview(preview_dir, "CryptoTransactions_preview.csv", crypto_history_df)

    if args.dry_run:
        print("\n✅ Dry-run complete. No Google Sheet updates were made.")
        print(f"Preview directory: {preview_dir}")
        return 0

    sheet_url = resolve_sheet_url(cfg, args.sheet_url)
    if not sheet_url:
        raise RuntimeError("No Google Sheet URL found. Use --sheet-url or config.yaml sheets.url.")

    service_account = resolve_service_account(cfg, args.service_account)
    if not service_account or not Path(service_account).expanduser().exists():
        raise RuntimeError(
            "No service account JSON found. Use --service-account or set google.service_account_json / "
            "GOOGLE_APPLICATION_CREDENTIALS."
        )

    print(f"\n🔑 Authorizing service account: {service_account}")
    print(f"📊 Target sheet: {sheet_url}")
    gc = authorize(service_account)

    if not holdings_df.empty:
        write_df_to_sheet(gc, sheet_url, args.holdings_tab, holdings_df)
        write_df_to_sheet(gc, sheet_url, args.crypto_holdings_tab, crypto_holdings_df)

    if not history_df.empty:
        write_df_to_sheet(gc, sheet_url, args.transactions_tab, history_df)
        write_df_to_sheet(gc, sheet_url, args.crypto_transactions_tab, crypto_history_df)

    print("\n🎯 Fidelity export sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
