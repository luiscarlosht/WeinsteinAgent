# weinstein_filters.py

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def _pick_first_existing(row: Mapping, candidates: Iterable[str]) -> Optional[float]:
    """
    Look through possible column names and return the first non-NaN value.
    row: dict-like (pd.Series / snapshot row)
    """
    for col in candidates:
        if col in row and not pd.isna(row[col]):
            return float(row[col])
    return None


def stock_ma30_slope_ok_from_snapshot(snapshot_row, long_cfg: Mapping) -> bool:
    """
    Generic per-stock MA30 slope filter using a *weekly snapshot* row.

    long_cfg is expected to have:
      - require_ma30_rising: bool
      - ma30_slope_min: float

    snapshot_row should contain one of:
      - 'ma30_slope_per_wk'
      - 'ma_slope_per_wk'
      - 'ma30_slope'
    """
    require = bool(long_cfg.get("require_ma30_rising", False))
    if not require:
        return True

    slope_min = float(long_cfg.get("ma30_slope_min", 0.0))

    slope = _pick_first_existing(
        snapshot_row,
        ["ma30_slope_per_wk", "ma_slope_per_wk", "ma30_slope"],
    )
    if slope is None:
        # No slope info available → don't block
        return True

    return slope >= slope_min


def stock_ma_proxy_slope_ok_from_series(
    price_series: pd.Series,
    window_days: int,
    long_cfg: Mapping,
) -> bool:
    """
    Per-stock MA slope filter using *daily prices* (e.g., intraday PROD proxy).

    - price_series: daily closes, indexed by date
    - window_days: e.g., 150 as 30-week proxy
    """
    require = bool(long_cfg.get("require_ma30_rising", False))
    if not require:
        return True

    slope_min = float(long_cfg.get("ma30_slope_min", 0.0))

    ma = price_series.rolling(window=window_days, min_periods=window_days // 2).mean()
    if ma.dropna().empty:
        return True  # not enough data to judge; don't block

    # Last two points → approximate slope
    last_ma = ma.iloc[-2:]
    if len(last_ma) < 2:
        return True

    slope = last_ma.iloc[-1] - last_ma.iloc[-2]
    return float(slope) >= slope_min
