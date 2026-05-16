#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_daily_cache.py

Small on-disk cache for yfinance daily OHLCV panels used by Weinstein replay/research.

Why:
- Year/scenario research can call the same daily download repeatedly.
- This cache avoids re-downloading the exact same ticker/start/end panel.

Design:
- Cache key includes tickers, start, end, and a schema version.
- Uses pickle so pandas MultiIndex columns are preserved without pyarrow/fastparquet.
- Safe default: if cache load fails, fall back to download and overwrite cache.
- Set WEINSTEIN_CACHE_REFRESH=1 to force refresh.
- Set WEINSTEIN_DAILY_CACHE_DIR=/path/to/cache to customize.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, List

import pandas as pd

CACHE_SCHEMA_VERSION = "daily_ohlcv_v1"


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    return sorted(set(str(t).strip().upper() for t in tickers if isinstance(t, str) and str(t).strip()))


def _cache_dir() -> Path:
    d = os.environ.get("WEINSTEIN_DAILY_CACHE_DIR", "").strip()
    if not d:
        d = os.path.join(os.getcwd(), "output", "daily_cache")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pad_start(start: str) -> str:
    start_dt = datetime.fromisoformat(str(start))
    return (start_dt - timedelta(days=365)).strftime("%Y-%m-%d")


def daily_cache_key(tickers: Iterable[str], start: str, end: str) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "tickers": _normalize_tickers(tickers),
        "start": str(start),
        "end": str(end),
        "pad_start": _pad_start(str(start)),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def cache_paths(tickers: Iterable[str], start: str, end: str) -> tuple[Path, Path]:
    key = daily_cache_key(tickers, start, end)
    base = _cache_dir() / f"daily_{key}"
    return base.with_suffix(".pkl"), base.with_suffix(".json")


def load_or_download_daily_bars(
    tickers: Iterable[str],
    start: str,
    end: str,
    downloader: Callable[[List[str], str, str], pd.DataFrame],
    log_func: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Load a cached daily OHLCV panel or download and cache it.

    downloader must accept (tickers, start, end) and return the same DataFrame
    shape expected by the existing Weinstein code.
    """
    tickers_norm = _normalize_tickers(tickers)
    pkl_path, meta_path = cache_paths(tickers_norm, start, end)
    refresh = os.environ.get("WEINSTEIN_CACHE_REFRESH", "").strip().lower() in {"1", "true", "yes", "y"}

    def _log(msg: str) -> None:
        if log_func:
            try:
                log_func(msg)
            except TypeError:
                log_func(str(msg))

    if pkl_path.exists() and not refresh:
        try:
            df = pd.read_pickle(pkl_path)
            if df is not None and not df.empty:
                _log(f"Daily bars cache HIT → {pkl_path}")
                return df
            _log(f"Daily bars cache file was empty → {pkl_path}; refreshing.")
        except Exception as e:
            _log(f"Daily bars cache read failed ({e}); refreshing → {pkl_path}")

    _log(f"Daily bars cache MISS → {pkl_path}")
    df = downloader(tickers_norm, start, end)

    try:
        tmp_path = pkl_path.with_suffix(".pkl.tmp")
        df.to_pickle(tmp_path)
        os.replace(tmp_path, pkl_path)
        meta = {
            "schema": CACHE_SCHEMA_VERSION,
            "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start": str(start),
            "end": str(end),
            "pad_start": _pad_start(str(start)),
            "ticker_count": len(tickers_norm),
            "tickers": tickers_norm,
            "rows": int(len(df.index)),
            "columns": int(len(df.columns)),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        _log(f"Daily bars cached → {pkl_path}")
    except Exception as e:
        _log(f"WARNING: failed to write daily bars cache ({e})")

    return df
