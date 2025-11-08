# industry_utils.py
import os, time, csv
from typing import Dict, Iterable
import yfinance as yf

CACHE_PATH_DEFAULT = "./output/industry_cache.csv"

def _read_cache(path: str) -> Dict[str, dict]:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("ticker") or "").strip().upper()
            if t:
                out[t] = {
                    "industry": (row.get("industry") or "").strip(),
                    "sector": (row.get("sector") or "").strip(),
                }
    return out

def _write_cache(path: str, db: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker","industry","sector"])
        w.writeheader()
        for t, rec in sorted(db.items()):
            w.writerow({
                "ticker": t,
                "industry": rec.get("industry",""),
                "sector": rec.get("sector",""),
            })

def _fetch_info_safe(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).get_info()
        # yfinance fields may vary; attempt common keys
        industry = info.get("industry") or info.get("industryKey") or ""
        sector = info.get("sector") or ""
        return {"industry": industry or "", "sector": sector or ""}
    except Exception:
        return {"industry": "", "sector": ""}

def get_industry_map(
    tickers: Iterable[str],
    cache_path: str = CACHE_PATH_DEFAULT,
    sleep_between: float = 0.2,
) -> Dict[str, dict]:
    tickers = [t.upper() for t in tickers if t]
    cache = _read_cache(cache_path)
    dirty = False

    for t in tickers:
        if t not in cache or not (cache[t].get("industry") or cache[t].get("sector")):
            rec = _fetch_info_safe(t)
            cache[t] = {"industry": rec.get("industry","") or "", "sector": rec.get("sector","") or ""}
            dirty = True
            time.sleep(sleep_between)

    if dirty:
        _write_cache(cache_path, cache)

    return cache

def attach_industry(df, ticker_col="ticker", out_col="industry", cache_path=CACHE_PATH_DEFAULT):
    """
    Adds `industry` and (if missing) `sector` columns by mapping `df[ticker_col]`.
    """
    import pandas as pd
    if df is None or df.empty or ticker_col not in df.columns:
        return df
    uniq = df[ticker_col].dropna().astype(str).str.upper().unique().tolist()
    mp = get_industry_map(uniq, cache_path=cache_path)
    df[out_col] = df[ticker_col].astype(str).str.upper().map(lambda t: (mp.get(t) or {}).get("industry",""))
    if "sector" not in df.columns:
        df["sector"] = df[ticker_col].astype(str).str.upper().map(lambda t: (mp.get(t) or {}).get("sector",""))
    return df
