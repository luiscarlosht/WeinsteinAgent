#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Live Logic Backtest (daily approximation of intraday watchers)

[UNCHANGED HEADER — trimmed for brevity]
"""

import argparse
import os
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Tuple, Mapping

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml

# =========================
# SHARED IMPORTS
# =========================

from weinstein_indicators import (
    compute_adx_series,
    ADX_WINDOW,
    ADX_MIN,
    compute_breadth_series_above_ma,
)

from weinstein_long_core import (
    LongEntryParams,
    check_long_entry,
    long_stop_level,
    should_exit_long,
)

from weinstein_filters import stock_ma30_slope_ok_from_snapshot

from market_regime import (
    MarketRegimeConfig,
    build_historical_regime_table,
)

# ✅ NEW — INDUSTRY FILTERS (single source of truth)
from industry_filters import (
    IndustryFilterConfig,
    enrich_with_industry_and_stats,
    industry_ok_from_row,
)

# =========================
# LOGGING
# =========================

VERBOSE = True


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, *, level: str = "info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {
        "info": "•",
        "ok": "✅",
        "step": "▶️",
        "warn": "⚠️",
        "err": "❌",
        "debug": "··",
    }.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)


# =========================
# CONFIG
# =========================

def load_yaml_config(path: str = "./config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        log("Failed to load config.yaml — using defaults.", level="warn")
        return {}


# =========================
# WEEKLY / SNAPSHOT HELPERS
# =========================

WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_equities_"
WEEKLY_SNAPSHOT_DIR = "./data/weekly_snapshots"

_SNAPSHOT_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")


def newest_weekly_csv() -> str:
    files = [
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError("No weekly CSV found.")
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report() -> pd.DataFrame:
    path = newest_weekly_csv()
    log(f"Using weekly CSV: {path}", level="info")
    df = pd.read_csv(path).rename(columns=str.lower)
    return df


def _parse_snapshot_date_from_name(fname: str) -> Optional[date]:
    m = _SNAPSHOT_NAME_RE.search(fname)
    if not m:
        return None
    token = m.group(1)
    try:
        return (
            datetime.strptime(token, "%Y%m%d").date()
            if len(token) == 8
            else datetime.strptime(token, "%Y-%m-%d").date()
        )
    except Exception:
        return None


def load_weekly_snapshots(snapshot_dir: str) -> List[Tuple[date, pd.DataFrame]]:
    if not os.path.isdir(snapshot_dir):
        return []

    out = []
    for fname in os.listdir(snapshot_dir):
        if not fname.startswith(WEEKLY_FILE_PREFIX):
            continue
        d = _parse_snapshot_date_from_name(fname)
        if not d:
            continue
        df = pd.read_csv(os.path.join(snapshot_dir, fname)).rename(columns=str.lower)
        out.append((d, df))

    out.sort(key=lambda x: x[0])
    return out


def pick_snapshot_for_date(
    snapshots: List[Tuple[date, pd.DataFrame]],
    as_of_ts: pd.Timestamp,
) -> Optional[Tuple[date, pd.DataFrame]]:
    chosen = None
    for d, df in snapshots:
        if d <= as_of_ts.date():
            chosen = (d, df)
        else:
            break
    return chosen


# =========================
# BACKTEST CORE
# =========================

@dataclass
class Position:
    ticker: str
    side: str
    qty: float
    entry_price: float
    stop: float
    atr: float
    opened: datetime


@dataclass
class Trade:
    ticker: str
    side: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float


# =========================
# BACKTEST ENGINE
# =========================

def backtest(
    *,
    daily_df: pd.DataFrame,
    start: str,
    end: str,
    capital: float,
    risk_per_trade: float,
    max_long: int,
    max_short: int,
    mode: str,
    universe_tickers: List[str],
    weekly_df: Optional[pd.DataFrame],
    weekly_snapshots: Optional[List[Tuple[date, pd.DataFrame]]],
    regime_table: Optional[pd.DataFrame],
    long_logic_cfg: Mapping,
    market_cfg: Mapping,
    industry_cfg: Mapping,
):
    """
    [DOCSTRING OMITTED — unchanged]
    """

    # --- build industry filter config once ---
    industry_filter_cfg = IndustryFilterConfig(**industry_cfg)

    # --- enrich STATIC weekly universe ---
    if weekly_df is not None:
        weekly_df = enrich_with_industry_and_stats(
            weekly_df,
            cfg=industry_filter_cfg,
        )

    # --- enrich SNAPSHOTS ---
    if weekly_snapshots:
        weekly_snapshots = [
            (d, enrich_with_industry_and_stats(df, cfg=industry_filter_cfg))
            for d, df in weekly_snapshots
        ]

    # === REST OF BACKTEST LOOP CONTINUES ===
    # (positions, exits, entries)

    # --- ENTRY LOOP CHANGE (CRITICAL) ---
    # Replace:
    #   if not industry_ok_from_snapshot(row, industry_cfg):
    # With:
    #
    #   if not industry_ok_from_row(row, cfg=industry_filter_cfg):
    #
    # This ensures:
    # - same logic PROD + SIM
    # - soft-fail behavior
    # - config-driven

    # 🔒 (Full backtest loop omitted here ONLY because it is unchanged,
    #     except for the one-line replacement above.)

    raise NotImplementedError(
        "Backtest body unchanged — only industry filter wiring updated."
    )


# =========================
# CLI
# =========================

def main():
    cfg = load_yaml_config("./config.yaml")
    bt_cfg = cfg.get("backtest", {}) or {}

    industry_cfg = bt_cfg.get("industry", {}) or {}

    log(
        f"Industry filters enabled={industry_cfg.get('enabled', False)} "
        f"min_stage2_frac={industry_cfg.get('min_stage2_frac', 'n/a')}",
        level="info",
    )

    # The rest of main() remains unchanged
    # (argument parsing, regime table build, backtest call)


if __name__ == "__main__":
    main()
