#!/usr/bin/env python3
"""
weinstein_weekly_core.py

Shared core logic for Weinstein weekly classification.

This module centralizes:
- Tunable constants (MA lengths, RS windows, etc.)
- YFinance weekly download helper (with optional as-of date)
- Stage / RS computation
- Block builder that attaches industry/sector and BUY/WATCH/AVOID.

Used by:
- weinstein_report_weekly.py  (weekly email report)
- weinstein_weekly_snapshot.py (historical snapshots for SIM/PROD)
- Any backtest / intraday code that wants consistent Stage-2 universe logic.
"""

import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from industry_utils import attach_industry

# ========= Tunables (crypto-enabled) =========
DEFAULT_BENCHMARK = "SPY"       # equities benchmark
CRYPTO_BENCHMARK  = "BTC-USD"   # crypto benchmark for RS/Stage
WEEKS_LOOKBACK = 180
MA_WEEKS = 30
MA10_WEEKS = 10
SLOPE_WINDOW = 5
NEAR_MA_BAND = 0.05
RS_MA_WEEKS = 30
OUTPUT_DIR_FALLBACK = "./output"
TOP_N_CHARTS = 20


# ========= Utilities =========
def _extract_field(df: pd.DataFrame, field: str, tickers: List[str]) -> pd.DataFrame:
    """Robustly extract a single field (Close/Volume/Adj Close) from a YF MultiIndex frame."""
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned by yfinance.")
    if isinstance(df.columns, pd.MultiIndex):
        avail_top = list(df.columns.get_level_values(0).unique())
        use_field = field if field in avail_top else ("Adj Close" if "Adj Close" in avail_top else None)
        if not use_field:
            raise KeyError(f"Field '{field}' not found; available: {avail_top}")
        out = df[use_field].copy()
        keep = [t for t in tickers if t in out.columns]
        if not keep:
            raise KeyError(f"No requested tickers found in downloaded data. Requested={tickers[:5]}...")
        return out[keep]
    cols = set(df.columns.astype(str))
    if field in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[[field]].copy()
        out.columns = [t0]
        return out
    if "Adj Close" in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[["Adj Close"]].copy()
        out.columns = [t0]
        return out
    raise KeyError(f"Field '{field}' not in downloaded data; got columns: {list(df.columns)}")


def fetch_weekly(
    tickers: List[str],
    benchmark: str,
    weeks: int = WEEKS_LOOKBACK,
    as_of_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download weekly Close + Volume for tickers + benchmark.

    If as_of_date is provided, only data up to that date is fetched (no look-ahead)
    with enough lookback history to support MA/RS calculations.
    """
    uniq = list(dict.fromkeys((tickers or []) + [benchmark]))
    if not uniq:
        raise ValueError("No symbols to download.")

    # How many weeks of history we need for all indicators
    tail_n = max(weeks, MA_WEEKS + RS_MA_WEEKS + SLOPE_WINDOW + 10)

    if as_of_date is None:
        # Current behavior: just grab a long span (10y) ending today.
        data = yf.download(
            uniq,
            interval="1wk",
            period="10y",
            auto_adjust=True,
            ignore_tz=True,
            progress=False,
            group_by="column",
        )
    else:
        # Historical snapshot: bounded window ending at as_of_date (no look-ahead).
        as_of_ts = pd.Timestamp(as_of_date).normalize() + pd.Timedelta(days=1)
        days_back = tail_n * 7 + 28  # a bit extra safety margin
        start = as_of_ts - pd.Timedelta(days=days_back)
        data = yf.download(
            uniq,
            interval="1wk",
            start=start,
            end=as_of_ts,
            auto_adjust=True,
            ignore_tz=True,
            progress=False,
            group_by="column",
        )

    close = _extract_field(data, "Close", uniq)
    volume = _extract_field(data, "Volume", uniq)

    # Final safety: trim to the most recent tail_n rows
    close = close.tail(tail_n)
    volume = volume.tail(tail_n)
    return close, volume


def _weekly_short_term_state(series_price: pd.Series) -> Tuple[str, float, float]:
    s = series_price.dropna()
    if len(s) < max(MA10_WEEKS, MA_WEEKS) + 5:
        return ("Unknown", np.nan, np.nan)
    ma10 = s.rolling(MA10_WEEKS).mean()
    ma30 = s.rolling(MA_WEEKS).mean()
    c = float(s.iloc[-1])
    m10 = float(ma10.iloc[-1])
    m30 = float(ma30.iloc[-1])
    state = "Unknown"
    if pd.notna(m10) and pd.notna(m30):
        if (c > m10) and (m10 > m30):
            state = "ShortTermUptrend"
        elif (c > m30) and not (m10 > m30):
            state = "StageConflict"
        elif (m10 > m30) and not (c > m10):
            state = "StageConflict"
        else:
            state = "Weak"
    return (state, m10, m30)


def compute_stage_for_ticker(closes: pd.Series, bench: pd.Series) -> dict:
    """Compute Weinstein Stage & RS metrics for a single ticker vs benchmark."""
    s = closes.dropna().copy()
    b = bench.reindex_like(s).dropna()
    idx = s.index.intersection(b.index)
    s = s.loc[idx]
    b = b.loc[idx]
    if len(s) < MA_WEEKS + SLOPE_WINDOW + 5 or len(b) < RS_MA_WEEKS + 5:
        return {"error": "insufficient_data"}

    ma = s.rolling(MA_WEEKS).mean()
    ma_slope = ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    ma_slope_last = ma_slope.iloc[-1]
    ma_last = ma.iloc[-1]
    price_last = s.iloc[-1]
    dist_ma_pct = (price_last - ma_last) / ma_last if ma_last and not math.isclose(ma_last, 0.0) else np.nan

    rs = s / b
    rs_ma = rs.rolling(RS_MA_WEEKS).mean()
    rs_slope = rs_ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    rs_last = rs.iloc[-1]
    rs_ma_last = rs_ma.iloc[-1]
    rs_above = bool(rs_last > rs_ma_last)
    rs_slope_last = rs_slope.iloc[-1]

    price_above_ma = bool(price_last > ma_last)
    ma_up = bool(ma_slope_last > 0)
    near_ma = bool(abs(dist_ma_pct) <= NEAR_MA_BAND)
    rs_up = bool(rs_above and rs_slope_last > 0)
    rs_down = bool((not rs_above) and rs_slope_last < 0)

    if price_above_ma and ma_up and rs_up:
        stage = "Stage 2 (Uptrend)"
    elif (not price_above_ma) and (ma_slope_last < 0) and rs_down:
        stage = "Stage 4 (Downtrend)"
    elif near_ma and abs(ma_slope_last) < (abs(ma_last) * 0.0005):
        stage = "Stage 1 (Basing)"
    else:
        stage = "Stage 3 (Topping)"

    notes = []
    if price_above_ma and not ma_up:
        notes.append("Price>MA but MA not rising")
    if (not price_above_ma) and ma_up:
        notes.append("Price<MA but MA rising (watch)")
    if rs_above and rs_slope_last <= 0:
        notes.append("RS above MA but flattening")
    if (not rs_above) and rs_slope_last >= 0:
        notes.append("RS below MA but improving")

    st_state, ma10_last, _ = _weekly_short_term_state(s)

    return {
        "price": float(price_last),
        "ma10": float(ma10_last) if pd.notna(ma10_last) else np.nan,
        "ma30": float(ma_last),
        "dist_ma_pct": float(dist_ma_pct) if pd.notna(dist_ma_pct) else np.nan,
        "ma_slope_per_wk": float(ma_slope_last) if pd.notna(ma_slope_last) else np.nan,
        "rs": float(rs_last),
        "rs_ma30": float(rs_ma_last) if pd.notna(rs_ma_last) else np.nan,
        "rs_above_ma": bool(rs_above),
        "rs_slope_per_wk": float(rs_slope_last) if pd.notna(rs_slope_last) else np.nan,
        "stage": stage,
        "short_term_state_wk": st_state,
        "notes": "; ".join(notes),
    }


def classify_buy_signal(stage: str) -> Tuple[str, str]:
    """Map Stage to BUY/WATCH/AVOID (long bias)."""
    stage = stage or ""
    if stage.startswith("Stage 2"):
        return ("BUY", "BUY")
    if stage.startswith("Stage 1"):
        return ("WATCH", "WATCH")
    if stage == "Filtered":
        return ("AVOID", "AVOID")
    return ("AVOID", "AVOID")


def build_block(
    close_w: pd.DataFrame,
    volume_w: pd.DataFrame,
    tickers: List[str],
    bench_sym: str,
    min_price: int,
    min_avg_volume: int,
    output_dir: str,
    asset_class: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a per-ticker DataFrame with stage/RS/etc. plus industry/sector and buy_signal.

    This is the same logic used in the weekly report; kept here so SIM/PROD can share it.
    """
    if bench_sym not in close_w.columns:
        raise KeyError(f"Benchmark '{bench_sym}' not found in downloaded data.")

    bench_series = close_w[bench_sym].dropna()
    last_close = close_w.ffill().iloc[-1]
    avg_vol_10w = volume_w.rolling(10).mean().ffill().iloc[-1]

    rows = []
    for t in tickers:
        if t not in close_w.columns:
            rows.append({"ticker": t, "stage": "N/A", "notes": "no_data", "asset_class": asset_class})
            continue

        lc = float(last_close.get(t, np.nan)) if pd.notna(last_close.get(t, np.nan)) else np.nan
        av = float(avg_vol_10w.get(t, np.nan)) if pd.notna(avg_vol_10w.get(t, np.nan)) else np.nan

        if (min_price and (pd.isna(lc) or lc < min_price)) or (min_avg_volume and (pd.isna(av) or av < min_avg_volume)):
            s = close_w[t].dropna()
            st_state, ma10_last, _ = _weekly_short_term_state(s)
            rows.append(
                {
                    "ticker": t,
                    "stage": "Filtered",
                    "price": lc,
                    "ma10": float(ma10_last) if pd.notna(ma10_last) else np.nan,
                    "ma30": np.nan,
                    "short_term_state_wk": st_state,
                    "notes": "below min_price/volume",
                    "asset_class": asset_class,
                }
            )
            continue

        res = compute_stage_for_ticker(close_w[t], bench_series)
        res["ticker"] = t
        res["asset_class"] = asset_class
        rows.append(res)

    df = pd.DataFrame(rows)

    # Ensure expected cols
    cols = [
        "ticker",
        "stage",
        "price",
        "ma10",
        "ma30",
        "dist_ma_pct",
        "ma_slope_per_wk",
        "rs",
        "rs_ma30",
        "rs_above_ma",
        "rs_slope_per_wk",
        "short_term_state_wk",
        "notes",
        "asset_class",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # Industry/sector for equities only; crypto will just have blanks
    df = attach_industry(
        df,
        ticker_col="ticker",
        out_col="industry",
        cache_path=f"{output_dir.rstrip('/')}/industry_cache.csv",
    )

    df["buy_signal"] = df["stage"].apply(lambda s: classify_buy_signal(str(s))[0])

    stage_rank = {
        "Stage 2 (Uptrend)": 0,
        "Stage 1 (Basing)": 1,
        "Stage 3 (Topping)": 2,
        "Stage 4 (Downtrend)": 3,
        "Filtered": 8,
        "N/A": 9,
    }
    df["stage_rank"] = df["stage"].map(stage_rank).fillna(9)
    df = df.sort_values(by=["stage_rank", "dist_ma_pct"], ascending=[True, False]).reset_index(drop=True)
    df = df.drop(columns=["stage_rank"])
    return df, bench_series
