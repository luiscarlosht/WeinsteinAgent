#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily PROD vs SIM parity report.

Compares:
- PROD intraday diagnostics: output/intraday_debug.csv
- SIM D replay events
- SIM F effective replay events selected by the latest F meta decision
- Fidelity account positions/profile map

Outputs:
- daily_prod_sim_signal_comparison.csv
- daily_account_recommendations.csv
- daily_meta_f_decisions.csv when available
- sim_F_effective_events.csv enriched for attribution
- daily_prod_sim_summary.html
- optional Google Sheet tabs
- optional email summary

This is a comparison/audit layer only. It does not change Weinstein CORE logic.
"""

from __future__ import annotations

import argparse
import os
import html
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from weinstein_prod_history import read_prod_history_for_date, summarize_prod_history

from weinstein_account_profiles import (
    load_profiles,
    read_fidelity_positions,
    normalize_positions,
    attach_profiles,
)

SIGNALS = {"BUY", "NEAR", "NEAR_BUY", "NEAR-TRIGGER", "SELL", "SELLTRIG", "SELL-TRIGGER", "SELL-WATCH", "SHORT"}


def _norm_signal(x: object) -> str:
    s = str(x or "").strip().upper()
    if s in {"NEAR_BUY", "NEAR-TRIGGER"}:
        return "NEAR"
    if s in {"SELLTRIG", "SELL-TRIGGER", "SELL-WATCH"}:
        return "SELL"
    if s == "SHORT":
        return "SHORT"
    if s in {"BUY", "NEAR", "SELL"}:
        return s
    return s


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _latest_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    date_col = None
    for c in ["date", "Date", "TimestampUTC", "timestamp", "RunDate"]:
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        return df
    dts = pd.to_datetime(df[date_col], errors="coerce")
    if dts.dropna().empty:
        return df
    latest = dts.max().date()
    out = df.loc[dts.dt.date.eq(latest)].copy()
    return out if not out.empty else df


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _safe_num(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "").replace("%", "")
    if not s:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_prod(prod: pd.DataFrame) -> pd.DataFrame:
    if prod.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])
    out = prod.copy()
    if "Ticker" not in out.columns and "ticker" in out.columns:
        out["Ticker"] = out["ticker"]
    if "Signal" not in out.columns and "signal" in out.columns:
        out["Signal"] = out["signal"]
    if "Price" not in out.columns:
        for c in ["PriceNow", "price", "Close", "close"]:
            if c in out.columns:
                out["Price"] = out[c]
                break
    if "Reason" not in out.columns:
        for c in ["Reason", "reason", "Details", "detail"]:
            if c in out.columns:
                out["Reason"] = out[c]
                break
    out["Ticker"] = out.get("Ticker", "").astype(str).str.upper().str.strip()
    out["Signal"] = out.get("Signal", "").apply(_norm_signal)
    out["Source"] = "PROD"
    out = out[out["Signal"].isin({"BUY", "NEAR", "SELL", "SHORT"})]
    return out[["Ticker", "Signal", "Price", "Reason", "Source"]].drop_duplicates()


def normalize_sim(sim: pd.DataFrame, source: str) -> pd.DataFrame:
    if sim.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])
    out = _latest_date_filter(sim.copy())
    if "Ticker" not in out.columns:
        for c in ["ticker", "symbol", "Symbol"]:
            if c in out.columns:
                out["Ticker"] = out[c]
                break
    if "Signal" not in out.columns:
        for c in ["signal", "Signal"]:
            if c in out.columns:
                out["Signal"] = out[c]
                break
    if "Price" not in out.columns:
        for c in ["price", "Price", "PriceNow", "close", "Close"]:
            if c in out.columns:
                out["Price"] = out[c]
                break
    if "Reason" not in out.columns:
        for c in ["reason", "Reason", "detail", "Details"]:
            if c in out.columns:
                out["Reason"] = out[c]
                break
    out["Ticker"] = out.get("Ticker", "").astype(str).str.upper().str.strip()
    out["Signal"] = out.get("Signal", "").apply(_norm_signal)
    out["Source"] = source
    out = out[out["Signal"].isin({"BUY", "NEAR", "SELL", "SHORT"})]
    return out[["Ticker", "Signal", "Price", "Reason", "Source"]].drop_duplicates()


def latest_meta_profile(meta: pd.DataFrame) -> str:
    if meta.empty or "meta_profile" not in meta.columns:
        return ""
    out = meta.copy()
    if "date" in out.columns:
        out["_dt"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("_dt")
    return str(out["meta_profile"].iloc[-1]).strip().upper()


def effective_f_signals(sim_d: pd.DataFrame, sim_e: pd.DataFrame, sim_f_raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Select the effective F shadow stream for the latest META profile."""
    profile = latest_meta_profile(meta)
    if profile == "A":
        out = sim_f_raw[sim_f_raw["Signal"].ne("SHORT")].copy()
    elif profile == "D":
        out = sim_d.copy()
    elif profile == "E":
        out = sim_e.copy()
    else:
        # B and unknown profiles consume the broad both-sides stream.
        out = sim_f_raw.copy()
    out["F_MetaProfile"] = profile or "UNKNOWN"
    return out


def _prepare_raw_sim_for_attribution(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Keep raw SIM columns and add normalized join columns for attribution enrichment."""
    if raw.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])

    out = _latest_date_filter(raw.copy())

    tcol = _first_existing_column(out, ["Ticker", "ticker", "Symbol", "symbol"])
    scol = _first_existing_column(out, ["Signal", "signal"])
    pcol = _first_existing_column(out, ["Price", "price", "PriceNow", "close", "Close"])
    rcol = _first_existing_column(out, ["Reason", "reason", "detail", "Details"])

    out["Ticker"] = out[tcol].astype(str).str.upper().str.strip() if tcol else ""
    out["Signal"] = out[scol].apply(_norm_signal) if scol else ""
    out["Price"] = out[pcol] if pcol else ""
    out["Reason"] = out[rcol] if rcol else ""
    out["Source"] = source
    out = out[out["Signal"].isin({"BUY", "NEAR", "SELL", "SHORT"})].copy()

    # Add raw-source-prefixed columns for columns that would otherwise collide or be lost.
    for col in list(out.columns):
        if col in {"Ticker", "Signal", "Price", "Reason", "Source"}:
            continue
        new_col = f"Raw_{col}"
        if new_col not in out.columns:
            out[new_col] = out[col]

    return out


def _derive_attribution_columns(df: pd.DataFrame, meta: pd.DataFrame, selected_profile: str) -> pd.DataFrame:
    """Add standard columns that the attribution engine can rely on, even when raw data is sparse."""
    out = df.copy()

    # Standard timestamp/profile columns.
    out["AttributionGeneratedUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    out["EffectiveMetaProfile"] = selected_profile or latest_meta_profile(meta) or "UNKNOWN"

    # Date/EventDate.
    date_col = _first_existing_column(out, ["date", "Date", "Raw_date", "Raw_Date", "TimestampUTC", "Raw_TimestampUTC", "timestamp", "Raw_timestamp"])
    if date_col:
        out["EventDate"] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    elif not meta.empty and "date" in meta.columns:
        m = meta.copy()
        m["_dt"] = pd.to_datetime(m["date"], errors="coerce")
        latest_date = m["_dt"].dropna().max()
        out["EventDate"] = latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else ""
    else:
        out["EventDate"] = ""

    # Numeric price and important indicator columns if available.
    out["PriceNum"] = out["Price"].map(_safe_num) if "Price" in out.columns else np.nan

    # Canonical attribution mapping.
    #
    # The replay engines use a few lower-level/raw names. The attribution engine
    # expects stable canonical names. Keep both: preserve the original/raw columns,
    # and populate these canonical columns for dashboards.
    ma_col = _first_existing_column(out, [
        "MA150", "ma150", "SMA150", "sma150", "Raw_MA150", "Raw_ma150", "Raw_SMA150", "Raw_sma150",
        "MA_150", "Raw_MA_150",
    ])
    ma30_col = _first_existing_column(out, [
        "MA30", "ma30", "SMA30", "sma30", "Raw_MA30", "Raw_ma30", "Raw_SMA30", "Raw_sma30",
        "MA_30", "Raw_MA_30",
    ])
    pivot_col = _first_existing_column(out, [
        "Pivot", "pivot", "PivotHigh", "pivot_high", "Raw_Pivot", "Raw_pivot", "Raw_PivotHigh", "Raw_pivot_high",
    ])
    adx_col = _first_existing_column(out, [
        "ADX", "adx", "ADX14", "adx14", "Raw_ADX", "Raw_adx", "Raw_ADX14", "Raw_adx14",
    ])
    atr_col = _first_existing_column(out, [
        "ATR", "atr", "ATR14", "atr14", "Raw_ATR", "Raw_atr", "Raw_ATR14", "Raw_atr14",
    ])
    vol_col = _first_existing_column(out, [
        "VolumeRatio", "volume_ratio", "VolRatio", "vol_ratio", "VolumePace", "volume_pace",
        "VolMult", "vol_mult", "Raw_VolumeRatio", "Raw_volume_ratio", "Raw_VolRatio", "Raw_vol_ratio",
        "Raw_VolumePace", "Raw_volume_pace", "Raw_VolMult", "Raw_vol_mult",
    ])
    rank_col = _first_existing_column(out, [
        "WeeklyRank", "weekly_rank", "Rank", "rank", "Raw_WeeklyRank", "Raw_weekly_rank", "Raw_Rank", "Raw_rank",
    ])
    quality_score_col = _first_existing_column(out, [
        "QualityScore", "quality_score", "Raw_QualityScore", "Raw_quality_score",
    ])
    quality_mult_col = _first_existing_column(out, [
        "QualityMult", "quality_mult", "Raw_QualityMult", "Raw_quality_mult",
    ])
    stage_col = _first_existing_column(out, ["Stage", "stage", "Raw_Stage", "Raw_stage"])
    regime_col = _first_existing_column(out, ["Regime", "regime", "MarketRegime", "market_regime", "Raw_Regime", "Raw_regime"])
    rs_col = _first_existing_column(out, ["RSAboveMA", "rs_above_ma", "Raw_RSAboveMA", "Raw_rs_above_ma"])

    out["MA150"] = out[ma_col].map(_safe_num) if ma_col else np.nan
    out["MA30"] = out[ma30_col].map(_safe_num) if ma30_col else np.nan
    out["Pivot"] = out[pivot_col].map(_safe_num) if pivot_col else np.nan
    out["ADX"] = out[adx_col].map(_safe_num) if adx_col else np.nan
    out["ATR"] = out[atr_col].map(_safe_num) if atr_col else np.nan
    out["VolumeRatio"] = out[vol_col].map(_safe_num) if vol_col else np.nan
    out["WeeklyRank"] = out[rank_col].map(_safe_num) if rank_col else np.nan
    out["QualityScore"] = out[quality_score_col].map(_safe_num) if quality_score_col else np.nan
    out["QualityMult"] = out[quality_mult_col].map(_safe_num) if quality_mult_col else np.nan
    out["Stage"] = out[stage_col] if stage_col else ""
    out["MarketRegime"] = out[regime_col] if regime_col else ""
    out["RSAboveMA"] = out[rs_col] if rs_col else ""

    out["DistanceToMA150Pct"] = np.where(
        out["PriceNum"].notna() & out["MA150"].notna() & (out["MA150"] != 0),
        (out["PriceNum"] - out["MA150"]) / out["MA150"] * 100.0,
        np.nan,
    )
    out["DistanceToPivotPct"] = np.where(
        out["PriceNum"].notna() & out["Pivot"].notna() & (out["Pivot"] != 0),
        (out["PriceNum"] - out["Pivot"]) / out["Pivot"] * 100.0,
        np.nan,
    )

    # Reason term and filter flags.
    out["ReasonTerm"] = out["Reason"].astype(str).str.lower().str.strip() if "Reason" in out.columns else ""
    out["Filter_MA150_Crack"] = out["ReasonTerm"].str.contains("ma150|sma150|below_ma|crack", regex=True, na=False)
    out["Filter_Breakout"] = out["ReasonTerm"].str.contains("breakout|pivot|break", regex=True, na=False)
    out["Filter_ADX_Available"] = out["ADX"].notna()
    out["Filter_Volume_Available"] = out["VolumeRatio"].notna()
    out["Filter_Rank_Available"] = out["WeeklyRank"].notna()
    out["Filter_QualityScore_Available"] = out["QualityScore"].notna()
    out["Filter_QualityMult_Available"] = out["QualityMult"].notna()
    out["Filter_Stage_Available"] = out["Stage"].astype(str).str.len().gt(0)
    out["Filter_Regime_Available"] = out["MarketRegime"].astype(str).str.len().gt(0)

    # PnL placeholders. If raw event files later add PnL/return columns, preserve them into these standards.
    pnl_col = _first_existing_column(out, [
        "PnL", "pnl", "Profit", "profit", "Gain", "gain", "Raw_PnL", "Raw_pnl", "Raw_Profit", "Raw_profit", "Raw_Gain", "Raw_gain",
    ])
    ret_col = _first_existing_column(out, [
        "ReturnPct", "return_pct", "Return", "return", "Raw_ReturnPct", "Raw_return_pct", "Raw_Return", "Raw_return",
    ])
    equity_before_col = _first_existing_column(out, ["EquityBefore", "equity_before", "Raw_EquityBefore", "Raw_equity_before"])
    equity_after_col = _first_existing_column(out, ["EquityAfter", "equity_after", "Raw_EquityAfter", "Raw_equity_after"])

    out["PnL"] = out[pnl_col].map(_safe_num) if pnl_col else np.nan
    out["ReturnPct"] = out[ret_col].map(_safe_num) if ret_col else np.nan
    out["EquityBefore"] = out[equity_before_col].map(_safe_num) if equity_before_col else np.nan
    out["EquityAfter"] = out[equity_after_col].map(_safe_num) if equity_after_col else np.nan

    # Make key attribution columns appear first.
    preferred = [
        "EventDate", "Ticker", "Signal", "Reason", "Source", "F_MetaProfile", "EffectiveMetaProfile",
        "Price", "PriceNum", "MA30", "MA150", "DistanceToMA150Pct", "Pivot", "DistanceToPivotPct",
        "Stage", "WeeklyRank", "ADX", "ATR", "VolumeRatio", "QualityScore", "QualityMult", "RSAboveMA", "MarketRegime",
        "Filter_MA150_Crack", "Filter_Breakout", "Filter_ADX_Available", "Filter_Volume_Available",
        "Filter_Rank_Available", "Filter_QualityScore_Available", "Filter_QualityMult_Available",
        "Filter_Stage_Available", "Filter_Regime_Available",
        "EquityBefore", "EquityAfter", "PnL", "ReturnPct", "ReasonTerm", "AttributionGeneratedUTC",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def enrich_effective_f_events(
    sim_f: pd.DataFrame,
    sim_d_raw: pd.DataFrame,
    sim_e_raw: pd.DataFrame,
    sim_f_raw_raw: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Return attribution-ready SIM F effective events without changing parity logic."""
    if sim_f.empty:
        return sim_f.copy()

    selected = latest_meta_profile(meta) or "UNKNOWN"

    raw_sources = [
        _prepare_raw_sim_for_attribution(sim_d_raw, "SIM_D_RAW_INPUT"),
        _prepare_raw_sim_for_attribution(sim_e_raw, "SIM_E_RAW_INPUT"),
        _prepare_raw_sim_for_attribution(sim_f_raw_raw, "SIM_F_RAW_INPUT"),
    ]
    raw_all = pd.concat([r for r in raw_sources if not r.empty], ignore_index=True) if any(not r.empty for r in raw_sources) else pd.DataFrame()

    base = sim_f.copy()
    base["_join_key"] = (
        base["Ticker"].astype(str).str.upper().str.strip()
        + "|" + base["Signal"].astype(str).str.upper().str.strip()
        + "|" + base["Reason"].astype(str).str.strip()
    )

    if not raw_all.empty:
        raw_all["_join_key"] = (
            raw_all["Ticker"].astype(str).str.upper().str.strip()
            + "|" + raw_all["Signal"].astype(str).str.upper().str.strip()
            + "|" + raw_all["Reason"].astype(str).str.strip()
        )

        # Prefer raw rows from the selected stream when possible.
        source_pref = {
            "D": "SIM_D_RAW_INPUT",
            "E": "SIM_E_RAW_INPUT",
            "A": "SIM_F_RAW_INPUT",
            "B": "SIM_F_RAW_INPUT",
            "UNKNOWN": "SIM_F_RAW_INPUT",
        }.get(selected, "SIM_F_RAW_INPUT")
        raw_all["_source_priority"] = np.where(raw_all["Source"].eq(source_pref), 0, 1)
        raw_all = raw_all.sort_values(["_join_key", "_source_priority"]).drop_duplicates("_join_key", keep="first")

        raw_cols = [c for c in raw_all.columns if c not in {"_source_priority"}]
        enriched = base.merge(raw_all[raw_cols], on="_join_key", how="left", suffixes=("", "_RawJoined"))

        # Fill standard columns from raw joined columns when present.
        for col in ["Ticker", "Signal", "Price", "Reason", "Source"]:
            raw_col = f"{col}_RawJoined"
            if raw_col in enriched.columns:
                enriched[col] = enriched[col].where(enriched[col].astype(str).str.len().gt(0), enriched[raw_col])
    else:
        enriched = base

    enriched = enriched.drop(columns=[c for c in ["_join_key", "Ticker_RawJoined", "Signal_RawJoined", "Price_RawJoined", "Reason_RawJoined", "Source_RawJoined"] if c in enriched.columns])
    return _derive_attribution_columns(enriched, meta, selected)



def _prepare_raw_sim_all_for_attribution(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Keep the full replay stream, not just latest date, for trade outcome attribution."""
    if raw.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])

    out = raw.copy()

    tcol = _first_existing_column(out, ["Ticker", "ticker", "Symbol", "symbol"])
    scol = _first_existing_column(out, ["Signal", "signal"])
    pcol = _first_existing_column(out, ["Price", "price", "PriceNow", "close", "Close"])
    rcol = _first_existing_column(out, ["Reason", "reason", "detail", "Details"])

    out["Ticker"] = out[tcol].astype(str).str.upper().str.strip() if tcol else ""
    out["Signal"] = out[scol].apply(_norm_signal) if scol else ""
    out["Price"] = out[pcol] if pcol else ""
    out["Reason"] = out[rcol] if rcol else ""
    out["Source"] = source
    out = out[out["Signal"].isin({"BUY", "NEAR", "SELL", "SHORT"})].copy()

    for col in list(out.columns):
        if col in {"Ticker", "Signal", "Price", "Reason", "Source"}:
            continue
        new_col = f"Raw_{col}"
        if new_col not in out.columns:
            out[new_col] = out[col]

    return out


def _nearest_future_return_for_group(group: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Return percent move to the first event row on/after EventDate + horizon_days.

    This uses the replay event stream as the available price series. If no future
    row exists for the ticker/horizon, the return is blank.
    """
    g = group.sort_values("EventDateDT").reset_index()
    dates = g["EventDateDT"].to_numpy()
    prices = g["PriceNum"].to_numpy(dtype=float)
    out = np.full(len(g), np.nan)

    for i in range(len(g)):
        if pd.isna(g.loc[i, "EventDateDT"]) or pd.isna(prices[i]) or prices[i] == 0:
            continue
        target = g.loc[i, "EventDateDT"] + pd.Timedelta(days=horizon_days)
        j = int(np.searchsorted(dates, np.datetime64(target), side="left"))
        if j < len(g) and not pd.isna(prices[j]):
            out[i] = (prices[j] - prices[i]) / prices[i] * 100.0

    return pd.Series(out, index=g["index"])


def _future_extremes_for_group(group: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Compute max gain and max drawdown percent through the horizon using available event rows."""
    g = group.sort_values("EventDateDT").reset_index()
    out_gain = np.full(len(g), np.nan)
    out_dd = np.full(len(g), np.nan)

    for i in range(len(g)):
        start_date = g.loc[i, "EventDateDT"]
        start_price = g.loc[i, "PriceNum"]
        if pd.isna(start_date) or pd.isna(start_price) or start_price == 0:
            continue
        end_date = start_date + pd.Timedelta(days=horizon_days)
        window = g[(g["EventDateDT"] > start_date) & (g["EventDateDT"] <= end_date)]
        if window.empty:
            continue
        rets = (window["PriceNum"].astype(float) - float(start_price)) / float(start_price) * 100.0
        out_gain[i] = rets.max()
        out_dd[i] = rets.min()

    return pd.DataFrame({
        f"MaxGain{horizon_days}D": pd.Series(out_gain, index=g["index"]),
        f"MaxDrawdown{horizon_days}D": pd.Series(out_dd, index=g["index"]),
    })


def build_trade_outcome_events(
    sim_d_raw: pd.DataFrame,
    sim_e_raw: pd.DataFrame,
    sim_f_raw_raw: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Build historical trade outcome rows from the selected META F replay stream.

    Daily effective events are latest-date only, so forward outcomes are usually blank there.
    This historical artifact uses the full replay stream and adds:
    - Forward5DReturnPct
    - Forward10DReturnPct
    - Forward20DReturnPct
    - MaxGain20D
    - MaxDrawdown20D
    - TradeOutcome20D
    """
    selected = latest_meta_profile(meta) or "UNKNOWN"

    if selected == "D":
        raw = sim_d_raw
        source = "SIM_D_OUTCOME_STREAM"
    elif selected == "E":
        raw = sim_e_raw
        source = "SIM_E_OUTCOME_STREAM"
    elif selected == "A":
        raw = sim_f_raw_raw
        source = "SIM_F_RAW_OUTCOME_STREAM"
    else:
        raw = sim_f_raw_raw
        source = "SIM_F_RAW_OUTCOME_STREAM"

    prepared = _prepare_raw_sim_all_for_attribution(raw, source)
    if prepared.empty:
        return pd.DataFrame()

    enriched = _derive_attribution_columns(prepared, meta, selected)

    # Keep only rows with usable ticker, date, price.
    enriched["EventDateDT"] = pd.to_datetime(enriched["EventDate"], errors="coerce")
    enriched["PriceNum"] = enriched["PriceNum"].map(_safe_num)
    enriched = enriched[
        enriched["Ticker"].astype(str).str.len().gt(0)
        & enriched["EventDateDT"].notna()
        & enriched["PriceNum"].notna()
    ].copy()

    if enriched.empty:
        return enriched

    for h in [5, 10, 20]:
        enriched[f"Forward{h}DReturnPct"] = np.nan
        for _, group in enriched.groupby("Ticker", sort=False):
            vals = _nearest_future_return_for_group(group, h)
            enriched.loc[vals.index, f"Forward{h}DReturnPct"] = vals

    enriched["MaxGain20D"] = np.nan
    enriched["MaxDrawdown20D"] = np.nan
    for _, group in enriched.groupby("Ticker", sort=False):
        ext = _future_extremes_for_group(group, 20)
        for col in ext.columns:
            enriched.loc[ext.index, col] = ext[col]

    # For SELL/SHORT, a negative forward return is favorable. For BUY/NEAR, positive is favorable.
    def outcome(row):
        ret = row.get("Forward20DReturnPct", np.nan)
        sig = _norm_signal(row.get("Signal"))
        if pd.isna(ret):
            return "PENDING"
        if sig in {"SELL", "SHORT"}:
            return "WIN" if ret < 0 else "LOSS"
        if sig in {"BUY", "NEAR"}:
            return "WIN" if ret > 0 else "LOSS"
        return "UNKNOWN"

    enriched["TradeOutcome20D"] = enriched.apply(outcome, axis=1)
    enriched["OutcomeAttributionMethod"] = "nearest future replay event on/after horizon date"

    # Place outcome fields near the front.
    preferred = [
        "EventDate", "Ticker", "Signal", "Reason", "Source", "EffectiveMetaProfile",
        "Price", "PriceNum", "Forward5DReturnPct", "Forward10DReturnPct", "Forward20DReturnPct",
        "MaxGain20D", "MaxDrawdown20D", "TradeOutcome20D", "OutcomeAttributionMethod",
        "MA30", "MA150", "DistanceToMA150Pct", "Pivot", "DistanceToPivotPct",
        "Stage", "WeeklyRank", "ADX", "ATR", "VolumeRatio", "QualityScore", "QualityMult", "MarketRegime",
    ]
    cols = [c for c in preferred if c in enriched.columns] + [c for c in enriched.columns if c not in preferred and c != "EventDateDT"]
    return enriched[cols]


def compare_signals(prod: pd.DataFrame, sim_d: pd.DataFrame, sim_f: pd.DataFrame, prod_history: pd.DataFrame | None = None, sim_f_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    prod_history = prod_history if prod_history is not None else pd.DataFrame(columns=["Ticker", "Signal"])
    sim_f_raw = sim_f_raw if sim_f_raw is not None else sim_f
    keys = sorted(set(prod["Ticker"]) | set(prod_history["Ticker"]) | set(sim_d["Ticker"]) | set(sim_f["Ticker"]) | set(sim_f_raw["Ticker"]))
    rows = []
    for t in keys:
        p = ",".join(sorted(prod.loc[prod["Ticker"].eq(t), "Signal"].unique()))
        ph = ",".join(sorted(prod_history.loc[prod_history["Ticker"].eq(t), "Signal"].unique())) if not prod_history.empty else ""
        d = ",".join(sorted(sim_d.loc[sim_d["Ticker"].eq(t), "Signal"].unique()))
        f = ",".join(sorted(sim_f.loc[sim_f["Ticker"].eq(t), "Signal"].unique()))
        f_raw = ",".join(sorted(sim_f_raw.loc[sim_f_raw["Ticker"].eq(t), "Signal"].unique()))
        rows.append({
            "Ticker": t,
            "PROD_Latest_Signal": p,
            "PROD_Intraday_Signal": ph,
            "SIM_D_Signal": d,
            "SIM_F_EffectiveSignal": f,
            "SIM_F_RawSignal": f_raw,
            "PROD_Latest_vs_D_Match": bool(p and d and p == d),
            "PROD_Latest_vs_F_Match": bool(p and f and p == f),
            "PROD_Intraday_vs_D_Match": bool(ph and d and ph == d),
            "PROD_Intraday_vs_F_Match": bool(ph and f and ph == f),
            "In_PROD_Latest": bool(p),
            "In_PROD_Intraday": bool(ph),
            "In_SIM_D": bool(d),
            "In_SIM_F": bool(f),
        })
    return pd.DataFrame(rows)


def account_recommendations(sim_d: pd.DataFrame, sim_f: pd.DataFrame, positions: pd.DataFrame, profile_cfg: dict) -> pd.DataFrame:
    """Build account-level recommendations.

    Operational filtering:
    - BUY/NEAR are shown as candidates for the account profile.
    - SELL is shown only when the account owns the ticker.
    - SHORT is shown as a candidate, but can be ignored if the account does not trade shorts.
    """
    accounts = profile_cfg.get("accounts", []) or []
    rows = []
    owned = positions[~positions.get("IsCash", False)].copy() if not positions.empty else pd.DataFrame()

    for acct in accounts:
        acct_num = str(acct.get("account_number", "")).strip()
        profile = str(acct.get("profile", "")).strip().upper()
        label = acct.get("label", "")
        events = sim_f if profile == "F" else sim_d if profile == "D" else pd.DataFrame()

        acct_owned = owned[owned["Account Number"].astype(str).eq(acct_num)].copy() if not owned.empty else pd.DataFrame()
        owned_tickers = set(acct_owned["Symbol"].astype(str).str.upper()) if not acct_owned.empty else set()

        for _, ev in events.iterrows():
            sig = _norm_signal(ev.get("Signal"))
            t = str(ev.get("Ticker", "")).upper().strip()
            if not t or sig not in {"BUY", "NEAR", "SELL", "SHORT"}:
                continue

            is_owned = t in owned_tickers

            # Reduce noise: do not show hundreds of "SELL not owned" rows.
            if sig == "SELL" and not is_owned:
                continue

            owned_row = acct_owned[acct_owned["Symbol"].astype(str).str.upper().eq(t)].head(1) if not acct_owned.empty else pd.DataFrame()

            if sig == "SELL":
                action = "SELL / reduce review"
            elif sig == "BUY" and is_owned:
                action = "BUY / add-to-position candidate"
            elif sig == "BUY":
                action = "BUY candidate"
            elif sig == "NEAR" and is_owned:
                action = "NEAR watch - already owned"
            elif sig == "NEAR":
                action = "NEAR watch"
            elif sig == "SHORT":
                action = "SHORT candidate"
            else:
                action = "Review"

            rows.append({
                "RunUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "AccountNumber": acct_num,
                "AccountLabel": label,
                "Profile": profile,
                "Ticker": t,
                "Signal": sig,
                "RecommendedAction": action,
                "Owned": is_owned,
                "OwnedQty": owned_row["Quantity"].iloc[0] if not owned_row.empty and "Quantity" in owned_row.columns else "",
                "CurrentValue": owned_row["Current Value"].iloc[0] if not owned_row.empty and "Current Value" in owned_row.columns else "",
                "SignalPrice": ev.get("Price", ""),
                "Reason": ev.get("Reason", ""),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "RunUTC", "AccountNumber", "AccountLabel", "Profile", "Ticker", "Signal",
            "RecommendedAction", "Owned", "OwnedQty", "CurrentValue", "SignalPrice", "Reason"
        ])

    priority = {"SELL": 0, "BUY": 1, "NEAR": 2, "SHORT": 3}
    out["_p"] = out["Signal"].map(priority).fillna(9)
    return out.sort_values(["AccountNumber", "_p", "Ticker"]).drop(columns=["_p"])


def read_meta_decisions(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    # Keep compact useful columns if present.
    keep = [c for c in ["date", "meta_profile", "meta_reason", "equity", "cash", "positions", "long_positions", "short_positions"] if c in df.columns]
    return df[keep].copy() if keep else df


def build_html(summary: dict, comparison: pd.DataFrame, recs: pd.DataFrame, meta: pd.DataFrame) -> str:
    def table(df, n=50):
        if df is None or df.empty:
            return "<p><i>No rows.</i></p>"
        return df.head(n).to_html(index=False, escape=True)

    large_d_recs = pd.DataFrame()
    small_meta_recs = pd.DataFrame()

    if recs is not None and not recs.empty and "Profile" in recs.columns:
        profile = recs["Profile"].astype(str).str.upper()
        large_d_recs = recs[profile.eq("D")].copy()
        small_meta_recs = recs[profile.eq("F")].copy()
    elif recs is not None:
        large_d_recs = recs

    parts = [
        "<html><body>",
        "<h2>Daily PROD vs SIM Parity Report</h2>",
        "<h3>Summary</h3>",
        "<ul>",
    ]
    for k, v in summary.items():
        if str(k).startswith("_"):
            continue
        parts.append(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>")
    parts += [
        "</ul>",
        "<h3>Action List — Large Fidelity Account (Profile D)</h3>",
        table(large_d_recs, 100),
        "<h3>Action List — Small Fidelity Account (META F)</h3>",
        table(small_meta_recs, 100),
        "<h3>PROD Intraday Signals Seen Today</h3>",
        table(summary.get("_prod_history_df", pd.DataFrame()), 100),
        "<h3>PROD vs SIM Signal Comparison</h3>",
        table(comparison, 100),
        "<h3>META F Decisions</h3>",
        table(meta.tail(20) if not meta.empty else meta, 20),
        "</body></html>",
    ]
    return "\n".join(parts)


def upload_to_sheets(profile_cfg: dict, comparison: pd.DataFrame, recs: pd.DataFrame, meta: pd.DataFrame):
    gs = profile_cfg.get("google_sheets", {}) or {}
    if not gs.get("enabled", False):
        return

    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(gs["service_account_json"], scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(gs["sheet_url"])
    tabs = gs.get("tabs", {}) or {}

    def write_tab(name, df):
        try:
            ws = sh.worksheet(name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=name, rows=1000, cols=50)
        ws.clear()
        out = df.copy().replace({np.nan: ""})
        values = [list(out.columns)] + out.astype(str).values.tolist()
        if values:
            ws.update(values)

    write_tab(tabs.get("comparison", "Daily_SIM_vs_PROD"), comparison)
    write_tab(tabs.get("account_recs", "Daily_Account_Recommendations"), recs)
    if not meta.empty:
        write_tab(tabs.get("meta_decisions", "Daily_META_F_Decisions"), meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-debug", default="output/intraday_debug.csv")
    ap.add_argument("--prod-history", default="output/prod_intraday_signal_history.csv")
    ap.add_argument("--prod-history-date", default="", help="Central-time date YYYY-MM-DD; defaults to latest date in history")
    ap.add_argument("--sim-d-events", required=True)
    ap.add_argument("--sim-e-events", default="")
    ap.add_argument("--sim-f-events", required=True)
    ap.add_argument("--sim-f-meta", default="")
    ap.add_argument("--positions-csv", default="")
    ap.add_argument("--profiles", default="account_strategy_profiles.yaml")
    ap.add_argument("--out-dir", default="output/daily_parity")
    ap.add_argument("--send-email", action="store_true")
    ap.add_argument("--upload-sheets", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    profile_cfg = load_profiles(args.profiles)
    if args.upload_sheets:
        profile_cfg.setdefault("google_sheets", {})["enabled"] = True

    prod = normalize_prod(_read_csv(args.prod_debug))
    prod_history_raw = read_prod_history_for_date(args.prod_history, args.prod_history_date or None)
    prod_history = summarize_prod_history(prod_history_raw)

    sim_d_raw = _read_csv(args.sim_d_events)
    sim_e_raw = _read_csv(args.sim_e_events)
    sim_f_raw_input = _read_csv(args.sim_f_events)

    sim_d = normalize_sim(sim_d_raw, "SIM_D")
    sim_e = normalize_sim(sim_e_raw, "SIM_E")
    sim_f_raw = normalize_sim(sim_f_raw_input, "SIM_F_RAW")

    meta = read_meta_decisions(args.sim_f_meta)
    sim_f = effective_f_signals(sim_d, sim_e, sim_f_raw, meta)
    sim_f_enriched = enrich_effective_f_events(sim_f, sim_d_raw, sim_e_raw, sim_f_raw_input, meta)
    sim_f_trade_outcomes = build_trade_outcome_events(sim_d_raw, sim_e_raw, sim_f_raw_input, meta)

    if args.positions_csv:
        pos = attach_profiles(normalize_positions(read_fidelity_positions(args.positions_csv)), profile_cfg)
        print(f"Positions loaded from {args.positions_csv}: {len(pos)}")
    else:
        if args.positions_csv:
            print(f"WARNING: positions CSV not found: {args.positions_csv}")
        pos = pd.DataFrame()

    comparison = compare_signals(prod, sim_d, sim_f, prod_history, sim_f_raw)
    recs = account_recommendations(sim_d, sim_f, pos, profile_cfg)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    comp_path = os.path.join(args.out_dir, f"daily_prod_sim_signal_comparison_{stamp}.csv")
    rec_path = os.path.join(args.out_dir, f"daily_account_recommendations_{stamp}.csv")
    meta_path = os.path.join(args.out_dir, f"daily_meta_f_decisions_{stamp}.csv")
    effective_f_path = os.path.join(args.out_dir, "sim_F_effective_events.csv")
    trade_outcomes_path = os.path.join(args.out_dir, "sim_F_trade_outcomes.csv")
    prod_hist_path = os.path.join(args.out_dir, f"daily_prod_intraday_history_{stamp}.csv")
    html_path = os.path.join(args.out_dir, f"daily_prod_sim_summary_{stamp}.html")

    comparison.to_csv(comp_path, index=False)
    recs.to_csv(rec_path, index=False)
    sim_f_enriched.to_csv(effective_f_path, index=False)
    if not sim_f_trade_outcomes.empty:
        sim_f_trade_outcomes.to_csv(trade_outcomes_path, index=False)
    if not meta.empty:
        meta.to_csv(meta_path, index=False)
    if not prod_history.empty:
        prod_history.to_csv(prod_hist_path, index=False)

    summary = {
        "PROD latest snapshot signals": len(prod),
        "PROD intraday signals seen": len(prod_history),
        "SIM D signals": len(sim_d),
        "SIM E signals": len(sim_e),
        "SIM F raw signals": len(sim_f_raw),
        "SIM F effective signals": len(sim_f),
        "SIM F selected profile": latest_meta_profile(meta) or "UNKNOWN",
        "SIM F effective enriched columns": len(sim_f_enriched.columns),
        "SIM F effective enriched rows": len(sim_f_enriched),
        "SIM F attribution mapping": "11.4 canonical mapping",
        "SIM F trade outcome rows": len(sim_f_trade_outcomes),
        "SIM F trade outcome mapping": "11.5 forward replay outcome attribution",
        "Account recommendation rows": len(recs),
        "PROD latest vs D exact ticker/signal matches": int(comparison["PROD_Latest_vs_D_Match"].sum()) if not comparison.empty else 0,
        "PROD latest vs F exact ticker/signal matches": int(comparison["PROD_Latest_vs_F_Match"].sum()) if not comparison.empty else 0,
        "PROD intraday vs D exact ticker/signal matches": int(comparison["PROD_Intraday_vs_D_Match"].sum()) if not comparison.empty else 0,
        "PROD intraday vs F exact ticker/signal matches": int(comparison["PROD_Intraday_vs_F_Match"].sum()) if not comparison.empty else 0,
        "Positions loaded": len(pos),
    }

    summary["_prod_history_df"] = prod_history
    html_body = build_html(summary, comparison, recs, meta)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_body)

    upload_to_sheets(profile_cfg, comparison, recs, meta)

    if args.send_email:
        try:
            from weinstein_mailer import send_email
            text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
            send_email(
                subject="Daily SIM vs PROD Parity Report",
                html_body=html_body,
                text_body=text,
                cfg_path="config.yaml",
                subject_tag="SIM-vs-PROD",
            )
        except Exception as e:
            print(f"WARNING: email failed: {e}")

    print("DONE")
    print(f"Comparison: {comp_path}")
    print(f"Account recommendations: {rec_path}")
    print(f"SIM F effective events: {effective_f_path}")
    if not sim_f_trade_outcomes.empty:
        print(f"SIM F trade outcomes: {trade_outcomes_path}")
    if not meta.empty:
        print(f"META F decisions: {meta_path}")
    if not prod_history.empty:
        print(f"PROD intraday history: {prod_hist_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
