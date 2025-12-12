#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
industry_filters.py

Shared industry/sector enrichment + per-industry "health" stats for both:
- PROD watchers (intraday/weekly gating)
- SIM/backtest (using weekly snapshots)

This module is intentionally:
- Cheap to call repeatedly (it relies on industry_utils cache)
- Defensive (if columns missing, it degrades gracefully)
- Snapshot-friendly (computes industry aggregates from per-stock snapshot columns)

Requires:
- industry_utils.py (attach_industry + caching)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from industry_utils import attach_industry


# -----------------------------
# Config / knobs
# -----------------------------

@dataclass
class IndustryFilterConfig:
    enabled: bool = True

    # These align with your config.yaml "backtest.industry.*" knobs
    require_stage2: bool = True
    min_stage2_frac: float = 0.30
    require_rising_ma30: bool = True
    require_rising_rs: bool = True

    # Column naming conventions
    ticker_col: str = "ticker"
    industry_col: str = "industry"
    sector_col: str = "sector"

    # Snapshot per-stock columns (if present) used to build per-industry aggregates
    stage_col: str = "stage"
    ma30_slope_col: str = "ma30_slope_per_wk"   # per-stock
    rs_slope_col: str = "rs_slope_per_wk"       # per-stock

    # Output columns that downstream filters will read
    out_industry_stage_col: str = "industry_stage"
    out_industry_stage2_frac_col: str = "industry_stage2_frac"
    out_industry_ma30_slope_col: str = "industry_ma30_slope_per_wk"
    out_industry_rs_slope_col: str = "industry_rs_slope_per_wk"


def _is_equity_ticker(t: str) -> bool:
    """
    Best-effort to avoid trying to industry-map obvious non-equities.
    Adjust if needed.
    """
    if not t:
        return False
    t = str(t).upper().strip()
    if t.endswith("-USD"):  # crypto in your system
        return False
    if t.startswith("^"):   # indices
        return False
    return True


# -----------------------------
# Core: enrich snapshot/universe with industry + per-industry stats
# -----------------------------

def enrich_with_industry_and_stats(
    df: pd.DataFrame,
    *,
    cfg: Optional[IndustryFilterConfig] = None,
    cache_path: str = "./output/industry_cache.csv",
) -> pd.DataFrame:
    """
    1) Ensures df has industry (and sector) via industry_utils.attach_industry()
    2) Computes per-industry health stats from snapshot columns and attaches them
       back onto each row.

    Safe behavior:
    - If df has no ticker column -> returns df unchanged
    - If no stage/slope columns exist -> stats will be NaN where unavailable
    """
    if cfg is None:
        cfg = IndustryFilterConfig()

    if df is None or df.empty:
        return df

    if cfg.ticker_col not in df.columns:
        return df

    out = df.copy()

    # --- attach industry/sector (equities only) ---
    # For mixed universes: call attach_industry for everything, but it will just map blanks for unknowns.
    # Still, we try to avoid polluting the cache with crypto/indices.
    tickers = (
        out[cfg.ticker_col]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    # If you want strict filtering, only keep equity-like tickers:
    # tickers = [t for t in tickers if _is_equity_ticker(t)]
    # But attach_industry expects the whole DF; we just let it map what it can.

    if cfg.industry_col not in out.columns or cfg.sector_col not in out.columns:
        out = attach_industry(out, ticker_col=cfg.ticker_col, out_col=cfg.industry_col, cache_path=cache_path)
        # attach_industry adds "sector" if missing, so we're good.

    # Normalize industry to avoid groupby weirdness
    out[cfg.industry_col] = out.get(cfg.industry_col, "").fillna("").astype(str)

    # If industry missing everywhere, nothing more to do
    if (out[cfg.industry_col].str.len() == 0).all():
        # Still create the expected columns so downstream logic can reference them
        for c in [
            cfg.out_industry_stage_col,
            cfg.out_industry_stage2_frac_col,
            cfg.out_industry_ma30_slope_col,
            cfg.out_industry_rs_slope_col,
        ]:
            if c not in out.columns:
                out[c] = np.nan
        return out

    # Stage column normalization
    if cfg.stage_col in out.columns:
        stage_s = out[cfg.stage_col].fillna("").astype(str)
    else:
        stage_s = pd.Series([""] * len(out), index=out.index)

    is_stage2 = stage_s.str.contains("Stage 2", case=False, na=False)

    # Numeric coercion for slopes if present
    if cfg.ma30_slope_col in out.columns:
        out[cfg.ma30_slope_col] = pd.to_numeric(out[cfg.ma30_slope_col], errors="coerce")
    if cfg.rs_slope_col in out.columns:
        out[cfg.rs_slope_col] = pd.to_numeric(out[cfg.rs_slope_col], errors="coerce")

    g = out.groupby(cfg.industry_col, dropna=False)

    stats = pd.DataFrame(index=g.size().index)
    stats[cfg.out_industry_stage2_frac_col] = g.apply(lambda x: float(is_stage2.loc[x.index].mean()))

    # Robust aggregation: median across the industry (works well with outliers)
    stats[cfg.out_industry_ma30_slope_col] = (
        g[cfg.ma30_slope_col].median() if cfg.ma30_slope_col in out.columns else np.nan
    )
    stats[cfg.out_industry_rs_slope_col] = (
        g[cfg.rs_slope_col].median() if cfg.rs_slope_col in out.columns else np.nan
    )

    # Simple derived label: Stage 2 if the group fraction is high enough
    stats[cfg.out_industry_stage_col] = np.where(
        stats[cfg.out_industry_stage2_frac_col] >= 0.50,
        "Stage 2 (Uptrend)",
        "Other",
    )

    # Attach back to each row
    out = out.join(stats, on=cfg.industry_col)
    return out


# -----------------------------
# Gate: decide if a row passes industry filter
# -----------------------------

def industry_ok_from_row(
    row: pd.Series,
    *,
    cfg: Optional[IndustryFilterConfig] = None,
) -> bool:
    """
    Returns True if the row passes the industry filter.
    Expects the row already has industry_* columns from enrich_with_industry_and_stats().
    """
    if cfg is None:
        cfg = IndustryFilterConfig()

    if not cfg.enabled:
        return True

    # If no industry, we cannot confirm -> be conservative OR permissive.
    # Conservative: block when enabled and industry is blank.
    industry = str(row.get(cfg.industry_col, "") or "").strip()
    if not industry:
        return False

    # Pull metrics
    ind_stage = str(row.get(cfg.out_industry_stage_col, "") or "")
    ind_frac = row.get(cfg.out_industry_stage2_frac_col, np.nan)
    ind_ma_slope = row.get(cfg.out_industry_ma30_slope_col, np.nan)
    ind_rs_slope = row.get(cfg.out_industry_rs_slope_col, np.nan)

    # 1) require stage2 label (optional)
    if cfg.require_stage2:
        if "Stage 2" not in ind_stage:
            # fallback: if stage label missing but fraction exists, allow if fraction passes
            if not (pd.notna(ind_frac) and float(ind_frac) >= cfg.min_stage2_frac):
                return False

    # 2) require fraction of stage2 members
    if pd.notna(ind_frac):
        if float(ind_frac) < float(cfg.min_stage2_frac):
            return False
    else:
        # Missing data -> conservative
        return False

    # 3) require rising industry MA30 slope
    if cfg.require_rising_ma30:
        if not (pd.notna(ind_ma_slope) and float(ind_ma_slope) >= 0.0):
            return False

    # 4) require rising industry RS slope
    if cfg.require_rising_rs:
        if not (pd.notna(ind_rs_slope) and float(ind_rs_slope) >= 0.0):
            return False

    return True
