#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_positions_source.py

Single source of truth for account positions.

Supported modes:
1) CSV file path, existing behavior:
   POSITIONS_CSV=/path/to/current_positions.csv

2) Google Sheet Holdings tab, preferred long-term behavior:
   POSITIONS_CSV=GOOGLE_SHEET
   or
   POSITIONS_SOURCE=GOOGLE_SHEET

Environment variables:
   WEINSTEIN_POSITIONS_SHEET_ID   required unless config has a detectable spreadsheet id
   WEINSTEIN_POSITIONS_TAB        default: Holdings
   GOOGLE_APPLICATION_CREDENTIALS optional, default: ./creds/gcp_service_account.json

Why Holdings, not Open_Positions?
- Holdings is the raw Fidelity positions upload and includes all account rows.
- Open_Positions is a derived performance tab and may not include every holding.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


GOOGLE_SENTINELS = {"GOOGLE_SHEET", "GSHEET", "SHEETS", "GOOGLE", "HOLDINGS_TAB"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_yaml(path: str | Path = "./config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _walk_values(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_values(v)
    else:
        yield obj


def _extract_sheet_id(value: Any) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    # Full Google Sheets URL.
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    if m:
        return m.group(1)
    # Plain sheet id, usually long.
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", s):
        return s
    return None


def discover_sheet_id(config_path: str | Path = "./config.yaml") -> str:
    env_id = (
        os.getenv("WEINSTEIN_POSITIONS_SHEET_ID")
        or os.getenv("POSITIONS_GOOGLE_SHEET_ID")
        or os.getenv("GOOGLE_SHEET_ID")
        or os.getenv("SPREADSHEET_ID")
    )
    sid = _extract_sheet_id(env_id)
    if sid:
        return sid

    cfg = _load_yaml(config_path)
    for value in _walk_values(cfg):
        sid = _extract_sheet_id(value)
        if sid:
            return sid

    raise RuntimeError(
        "Could not determine Google Sheet ID. Set WEINSTEIN_POSITIONS_SHEET_ID "
        "in ~/.weinstein_env."
    )


def is_google_positions_source(path_or_source: str | None = None) -> bool:
    source = (os.getenv("POSITIONS_SOURCE") or "").upper().strip()
    path = str(path_or_source or os.getenv("POSITIONS_CSV") or "").upper().strip()
    return source in GOOGLE_SENTINELS or path in GOOGLE_SENTINELS or path.startswith("GSHEET://")


def read_google_positions_tab(
    sheet_id: str | None = None,
    tab_name: str | None = None,
    creds_path: str | None = None,
) -> pd.DataFrame:
    try:
        import gspread
    except ImportError as e:
        raise RuntimeError("gspread is required to read Google Sheets positions.") from e

    sheet_id = sheet_id or discover_sheet_id()
    tab_name = tab_name or os.getenv("WEINSTEIN_POSITIONS_TAB") or os.getenv("POSITIONS_GOOGLE_SHEET_TAB") or "Holdings"
    creds_path = (
        creds_path
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or str(_repo_root() / "creds" / "gcp_service_account.json")
    )

    gc = gspread.service_account(filename=creds_path)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(tab_name)
    values = ws.get_all_values()

    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    # Drop completely empty rows.
    if not df.empty:
        df = df.dropna(how="all")
        df = df.loc[~(df.astype(str).apply(lambda r: "".join(r).strip(), axis=1).eq(""))]

    return df


def load_positions_dataframe(path_or_source: str | None = None, config_path: str | Path = "./config.yaml") -> pd.DataFrame:
    """
    Main entry point for account position loading.

    Existing callers can pass a CSV path.
    New callers can pass 'GOOGLE_SHEET' or set POSITIONS_SOURCE=GOOGLE_SHEET.
    """
    if is_google_positions_source(path_or_source):
        df = read_google_positions_tab(
            sheet_id=discover_sheet_id(config_path),
            tab_name=os.getenv("WEINSTEIN_POSITIONS_TAB") or os.getenv("POSITIONS_GOOGLE_SHEET_TAB") or "Holdings",
        )
        print(f"Positions loaded from Google Sheet tab: rows={len(df)}")
        return df

    path = path_or_source or os.getenv("POSITIONS_CSV")
    if not path:
        raise RuntimeError("No positions source provided. Set POSITIONS_CSV or POSITIONS_SOURCE=GOOGLE_SHEET.")

    df = pd.read_csv(path)
    print(f"Positions loaded from CSV: {path}: rows={len(df)}")
    return df


if __name__ == "__main__":
    df = load_positions_dataframe(os.getenv("POSITIONS_CSV") or os.getenv("POSITIONS_SOURCE") or "GOOGLE_SHEET")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))
    cols = [c for c in ["Account Number", "Account Name", "Symbol", "Description", "Quantity", "Current Value"] if c in df.columns]
    if cols:
        print(df[cols].head(20).to_string(index=False))
    else:
        print(df.head(20).to_string(index=False))
