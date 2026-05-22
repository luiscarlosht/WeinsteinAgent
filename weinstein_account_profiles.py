#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Account/profile helpers for WeinsteinAgent.

Purpose:
- Read account_strategy_profiles.yaml
- Read Fidelity positions exports robustly
- Map account numbers to strategy profiles.
"""

from __future__ import annotations

import csv
import os
from typing import Dict

import numpy as np
import pandas as pd
import yaml


def load_profiles(path: str = "account_strategy_profiles.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _detect_header_row(csv_path: str, required_any: list[str]) -> int:
    required = {x.lower() for x in required_any}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            cells = {str(c).strip().lower().replace("\ufeff", "") for c in row if str(c).strip()}
            if cells & required:
                return i
    return 0


def read_fidelity_positions(csv_path: str) -> pd.DataFrame:
    """Read Fidelity positions CSV without column shifting.

    Handles:
    - UTF-8 BOM
    - footer disclaimer rows
    - cash rows
    - trailing commas
    - occasional malformed footer text
    """
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame()

    header_row = _detect_header_row(csv_path, ["Account Number", "Symbol", "Current Value"])
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

    if "Account Number" not in df.columns or "Symbol" not in df.columns:
        return pd.DataFrame()

    # Fidelity real account rows are alphanumeric account IDs.
    acct = df["Account Number"].astype(str).str.strip()
    df = df[acct.str.match(r"^[A-Z0-9]{5,}$", na=False)].copy()

    # Remove footer/disclaimer rows that may still leak through.
    df = df[df["Symbol"].astype(str).str.strip().ne("")]
    return df.reset_index(drop=True)


def money_to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = (
        s.replace("$", "")
        .replace(",", "")
        .replace("+", "")
        .replace("%", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for c in ["Symbol", "Account Number", "Account Name", "Description", "Type"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    out["Account Number"] = out["Account Number"].astype(str).str.strip()
    out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()

    for c in ["Quantity", "Last Price", "Current Value", "Cost Basis Total", "Average Cost Basis"]:
        if c in out.columns:
            out[c + "_Num"] = out[c].apply(money_to_float)

    if "Percent Of Account" in out.columns:
        out["Percent_Of_Account_Num"] = out["Percent Of Account"].apply(money_to_float)

    out["IsCash"] = out["Symbol"].astype(str).str.contains(
        r"FCASH|SPAXX|CASH|\*\*|MONEY MARKET",
        case=False,
        regex=True,
        na=False,
    )
    return out.reset_index(drop=True)


def account_profile_map(profile_cfg: dict) -> Dict[str, dict]:
    m = {}
    for acct in profile_cfg.get("accounts", []) or []:
        num = str(acct.get("account_number", "")).strip()
        if num:
            m[num] = acct
    return m


def attach_profiles(positions: pd.DataFrame, profile_cfg: dict) -> pd.DataFrame:
    if positions.empty:
        return positions

    m = account_profile_map(profile_cfg)
    out = positions.copy()
    out["Profile"] = out["Account Number"].map(lambda x: m.get(str(x).strip(), {}).get("profile", ""))
    out["AccountLabel"] = out["Account Number"].map(lambda x: m.get(str(x).strip(), {}).get("label", ""))
    out["AccountRole"] = out["Account Number"].map(lambda x: m.get(str(x).strip(), {}).get("role", ""))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("positions_csv")
    ap.add_argument("--profiles", default="account_strategy_profiles.yaml")
    args = ap.parse_args()

    cfg = load_profiles(args.profiles)
    pos = attach_profiles(normalize_positions(read_fidelity_positions(args.positions_csv)), cfg)
    print(pos[["Account Number", "AccountLabel", "Profile", "Symbol", "Quantity", "Current Value", "IsCash"]].to_string(index=False))
    print(f"\nRows loaded: {len(pos)}")
