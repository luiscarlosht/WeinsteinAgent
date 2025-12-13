#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
industry_filters.py

Shared industry/sector enrichment + per-industry "health" stats for both:
- PROD watchers (intraday / weekly gating)
- SIM / backtest (using weekly snapshots)

Design goals:
- Cheap to call repeatedly (relies on industry_utils cache)
- Defensive: missing data never hard-blocks trades
- Snapshot-friendly: computes industry aggregates from per-stock snapshot columns

Requires:
- industry_utils.py (attach_industry + caching)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from industry_utils import attach_industry


# -----------------------------
# Config / knobs
# -----------------------------

@dataclass
class IndustryFilterConfig:
    enabled: bool = True

    # Aligns with config.yaml → backtest.industry.*
    require_stage2: bool = True
    min_stage2_frac: float = 0.30
    require_rising_ma30: bool = True
    require_rising_rs: bool = True

    # Column naming conventions
    ticker_col: str = "ticker"
    industry_col: str = "industry"
    sector_col: str = "sector"

    # Per-stock snapshot inputs (optional)
    stage_col: str = "stage"
    ma30_slope_col: str = "ma30_slope_per_wk"
    rs_slope_col: str = "rs_slope_per_wk"

    # Output per-industry columns
    out_industry_stage_col: str = "industry_stage"
    out_industry_stage2_frac_col: str = "industry_stage2_frac"
    out_industry_ma30_slope_col: str = "industry_ma30_slope_per_wk"
    out_industry_rs_slope_col: str = "industry_rs_slope_per_wk"


# -----------------------------
# Core: enrich snapshot with industry + stats
# -----------------------------

def enrich_with_industry_and_stats(
    df: pd.DataFrame,
    *,
    cfg: Optional[IndustryFilterConfig] = None,
    cache_path: str = "./output/industry_cache.csv",
) -> pd.DataFrame:
    """
    1) Attach industry / sector (via industry_utils.attach_industry)
    2) Compute per-industry health stats and attach to each row

    Defensive behavior:
    - Missing ticker / industry / slope data never blocks downstream logic
    - Stats default to NaN if unavailable

    Important:
    - This function is designed to be IDPOTENT: you can run it on snapshots that
      already have industry_* columns, and it will overwrite them safely.
    """
    if cfg is None:
        cfg = IndustryFilterConfig()

    if df is None or df.empty or cfg.ticker_col not in df.columns:
        return df

    out = df.copy()

    # ---- Attach industry / sector (cached, cheap) ----
    if cfg.industry_col not in out.columns or cfg.sector_col not in out.columns:
        out = attach_industry(
            out,
            ticker_col=cfg.ticker_col,
            out_col=cfg.industry_col,
            cache_path=cache_path,
        )

    out[cfg.industry_col] = out.get(cfg.industry_col, "").fillna("").astype(str)

    # Ensure expected output columns exist (so downstream code can reference them)
    out_cols = [
        cfg.out_industry_stage_col,
        cfg.out_industry_stage2_frac_col,
        cfg.out_industry_ma30_slope_col,
        cfg.out_industry_rs_slope_col,
    ]
    for c in out_cols:
        if c not in out.columns:
            out[c] = np.nan

    # If no industry info at all → nothing more to compute
    if (out[cfg.industry_col].str.len() == 0).all():
        return out

    # ---- Stage handling ----
    if cfg.stage_col in out.columns:
        stage_s = out[cfg.stage_col].fillna("").astype(str)
    else:
        stage_s = pd.Series("", index=out.index)

    is_stage2 = stage_s.str.contains("Stage 2", case=False, na=False)

    # ---- Numeric coercion ----
    if cfg.ma30_slope_col in out.columns:
        out[cfg.ma30_slope_col] = pd.to_numeric(out[cfg.ma30_slope_col], errors="coerce")
    if cfg.rs_slope_col in out.columns:
        out[cfg.rs_slope_col] = pd.to_numeric(out[cfg.rs_slope_col], errors="coerce")

    g = out.groupby(cfg.industry_col, dropna=False)

    stats = pd.DataFrame(index=g.size().index)

    # Fraction of members in Stage 2
    # (Note: this may emit a pandas FutureWarning depending on pandas version;
    # it is harmless. We can refactor later if you want it silent.)
    stats[cfg.out_industry_stage2_frac_col] = g.apply(
        lambda x: float(is_stage2.loc[x.index].mean())
    )

    # Median slopes (robust to outliers)
    stats[cfg.out_industry_ma30_slope_col] = (
        g[cfg.ma30_slope_col].median()
        if cfg.ma30_slope_col in out.columns
        else np.nan
    )
    stats[cfg.out_industry_rs_slope_col] = (
        g[cfg.rs_slope_col].median()
        if cfg.rs_slope_col in out.columns
        else np.nan
    )

    # Derived industry stage label (threshold uses cfg.min_stage2_frac)
    stats[cfg.out_industry_stage_col] = np.where(
        stats[cfg.out_industry_stage2_frac_col] >= float(cfg.min_stage2_frac),
        "Stage 2 (Uptrend)",
        "Other",
    )

    # -----------------------------
    # ✅ FIX: avoid pandas "columns overlap but no suffix specified"
    # when snapshots already have industry_* columns.
    # We drop the existing output columns, then join the newly computed ones.
    # -----------------------------
    overlap_cols = [c for c in out_cols if c in out.columns]
    if overlap_cols:
        out = out.drop(columns=overlap_cols)

    # Attach back to rows
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

    Defensive philosophy:
    - Missing industry → allow
    - Missing stats → allow
    - Only block when data exists AND violates thresholds
    """
    if cfg is None:
        cfg = IndustryFilterConfig()

    if not cfg.enabled:
        return True

    industry = str(row.get(cfg.industry_col, "") or "").strip()

    # SOFT-FAIL: no industry info → do not block
    if not industry:
        return True

    ind_stage = str(row.get(cfg.out_industry_stage_col, "") or "")
    ind_frac = row.get(cfg.out_industry_stage2_frac_col, np.nan)
    ind_ma_slope = row.get(cfg.out_industry_ma30_slope_col, np.nan)
    ind_rs_slope = row.get(cfg.out_industry_rs_slope_col, np.nan)

    # 1) Require industry Stage 2 (label OR fraction)
    if cfg.require_stage2:
        if "Stage 2" not in ind_stage:
            if pd.notna(ind_frac) and float(ind_frac) < float(cfg.min_stage2_frac):
                return False

    # 2) Require minimum fraction in Stage 2
    if pd.notna(ind_frac):
        if float(ind_frac) < float(cfg.min_stage2_frac):
            return False

    # 3) Require rising industry MA30 slope
    if cfg.require_rising_ma30:
        if pd.notna(ind_ma_slope) and float(ind_ma_slope) < 0.0:
            return False

    # 4) Require rising industry RS slope
    if cfg.require_rising_rs:
        if pd.notna(ind_rs_slope) and float(ind_rs_slope) < 0.0:
            return False

    return True
