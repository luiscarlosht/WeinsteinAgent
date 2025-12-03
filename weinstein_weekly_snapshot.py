#!/usr/bin/env python3
"""
weinstein_weekly_snapshot.py

Helper to build or reuse a *historical* weekly snapshot (equities only, for now)
as of a given date, using the same Stage/RS engine as live PROD.

- Reads the same config.yaml universe (combine_universe + benchmark).
- Uses weinstein_weekly_core.fetch_weekly(as_of_date=...) to avoid look-ahead.
- Caches snapshots under ./output/sim_weekly by default:
    output/sim_weekly/weinstein_weekly_equities_YYYYMMDD.csv

Intended use:
    from weinstein_weekly_snapshot import get_weekly_snapshot

    snap_df, csv_path = get_weekly_snapshot(
        as_of_date=date(2019, 4, 5),
        config_path="./config.yaml",
    )
"""

import os
from datetime import datetime, date
from typing import Tuple, Union

import pandas as pd

from universe_loaders import combine_universe
from weinstein_weekly_core import (
    DEFAULT_BENCHMARK,
    WEEKS_LOOKBACK,
    OUTPUT_DIR_FALLBACK,
    fetch_weekly,
    build_block,
)


def _load_config_for_snapshot(path: str = "config.yaml") -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    app = cfg.get("app", {}) or {}
    uni = cfg.get("universe", {}) or {}
    reporting = cfg.get("reporting", {}) or {}

    mode = (uni.get("mode") or "custom").lower()
    use_sp500 = (mode == "sp500")
    extra = uni.get("extra") or []
    explicit_tickers = uni.get("tickers") or []

    benchmark = app.get("benchmark", DEFAULT_BENCHMARK)
    output_dir = reporting.get("output_dir", OUTPUT_DIR_FALLBACK)
    min_price = int(uni.get("min_price", 0))
    min_avg_volume = int(uni.get("min_avg_volume", 0))

    if use_sp500:
        eq_tickers = combine_universe(sp500=True, extra_symbols=extra)
    else:
        eq_tickers = combine_universe(sp500=False, extra_symbols=explicit_tickers)

    return {
        "cfg": cfg,
        "eq_tickers": eq_tickers,
        "benchmark": benchmark,
        "output_dir": output_dir,
        "min_price": min_price,
        "min_avg_volume": min_avg_volume,
    }


def _normalize_as_of(as_of: Union[str, datetime, date]) -> date:
    if isinstance(as_of, date) and not isinstance(as_of, datetime):
        return as_of
    if isinstance(as_of, datetime):
        return as_of.date()
    # string fallback
    ts = pd.to_datetime(as_of)
    return ts.date()


def get_weekly_snapshot(
    as_of_date: Union[str, datetime, date],
    config_path: str = "config.yaml",
    out_dir: str = "output/sim_weekly",
) -> Tuple[pd.DataFrame, str]:
    """
    Build or reuse a weekly equities snapshot *as of* as_of_date.

    Returns:
        df:   DataFrame with same columns as live weinstein_weekly_equities_*.csv
        path: CSV path on disk
    """
    as_of_d = _normalize_as_of(as_of_date)
    week_key = as_of_d.strftime("%Y%m%d")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"weinstein_weekly_equities_{week_key}.csv")

    # 1) If we already built this snapshot, just reuse it
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df, csv_path

    # 2) Build from scratch with same universe + benchmark as live
    params = _load_config_for_snapshot(config_path)
    eq_tickers = params["eq_tickers"]
    benchmark = params["benchmark"]
    min_price = params["min_price"]
    min_avg_volume = params["min_avg_volume"]

    print(f"[snapshot] Building weekly snapshot for {as_of_d} with {len(eq_tickers)} equities, bench={benchmark}")

    close_w, volume_w = fetch_weekly(
        eq_tickers,
        benchmark,
        weeks=WEEKS_LOOKBACK,
        as_of_date=pd.Timestamp(as_of_d),
    )

    eq_df, _ = build_block(
        close_w,
        volume_w,
        eq_tickers,
        benchmark,
        min_price=min_price,
        min_avg_volume=min_avg_volume,
        output_dir=out_dir,
        asset_class="Equity/ETF",
    )

    eq_df.to_csv(csv_path, index=False)
    print(f"[snapshot] Saved snapshot CSV → {csv_path}")
    return eq_df, csv_path
