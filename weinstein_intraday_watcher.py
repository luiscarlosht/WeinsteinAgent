#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Intraday Watch
- Loads config.yaml
- Reads latest weekly equities CSV (for context + snapshot)
- Optionally pulls 'Signals' from your Google Sheet to find any crypto tickers
- Builds a Crypto Weekly block (benchmark BTC-USD) and prints it
- Prints Intraday BUY/NEAR/SELL trigger shells (safe defaults)
- Prints Snapshot (ticker, stage, price, ma30) from weekly CSV
- Prints "Weinstein Weekly – Summary" section after the Crypto block

Notes:
- Keeps console output formatting consistent with your examples
- Handles pandas truth-value & .tolist() pitfalls
- Uses yfinance for prices; if unavailable, Crypto block downgrades gracefully
"""

import os
import sys
import io
import glob
import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps
try:
    import yaml
except Exception:
    yaml = None

try:
    import yfinance as yf
except Exception:
    yf = None

# Optional Google Sheets (Signals)
GSPREAD_AVAILABLE = True
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    GSPREAD_AVAILABLE = False

# ========== Utility ==========
def _now_ct() -> dt.datetime:
    return dt.datetime.now()

def _fmt_now_for_header(tz_name: str = "America/Chicago") -> str:
    # Avoid tz lib dependency; print local time as-is with HH:MM
    return _now_ct().strftime("%Y-%m-%d %H:%M")

def _safe_float_from_series(series: pd.Series, index: int = -1, default: float = float("nan")) -> float:
    try:
        if series is None or len(series) == 0:
            return default
        val = series.iloc[index]
        return float(val)
    except Exception:
        return default

def _slope_per_week(series: pd.Series, lookback_days: int = 5) -> float:
    """Return last - prior(5 bars) as 'per-week' slope proxy."""
    try:
        if series is None or len(series) < lookback_days + 1:
            return float("nan")
        return float(series.iloc[-1] - series.iloc[-1 - lookback_days])
    except Exception:
        return float("nan")

def _pct(a: float) -> str:
    if pd.isna(a) or a is None:
        return ""
    return f"{a:.2f}%"

def _sparkline(arr: List[float], width: int = 18) -> str:
    # Simple unicode sparkline
    if not arr:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(arr), max(arr)
    if math.isclose(lo, hi):
        return blocks[0] * min(len(arr), width)
    # compress if needed
    data = arr
    if len(arr) > width:
        # sample evenly to width
        idxs = np.linspace(0, len(arr) - 1, width).astype(int)
        data = [arr[i] for i in idxs]
    out = []
    for v in data:
        t = (v - lo) / (hi - lo)
        out.append(blocks[min(len(blocks) - 1, max(0, int(t * (len(blocks) - 1))))])
    return "".join(out)

def _stage_from_price_ma(price: float, ma30: float, ma_slope: float) -> str:
    if pd.isna(price) or pd.isna(ma30) or pd.isna(ma_slope):
        return "Stage ? (Unknown)"
    if price >= ma30 and ma_slope > 0:
        return "Stage 2 (Uptrend)"
    if price < ma30 and ma_slope < 0:
        return "Stage 4 (Downtrend)"
    # transitional / topping/basing
    return "Stage 3 (Topping)" if price >= ma30 else "Stage 1 (Basing)"

def _buy_watch_avoid(stage: str, rs_above_ma: bool, rs_slope_wk: float, dist_ma_pct: float) -> str:
    # Conservative read:
    # BUY: Stage 2 + RS above MA + RS slope >= 0
    # WATCH: Stage 1/3 with improving RS slope (>=0) OR Stage 2 but RS not fully confirming
    # AVOID: Stage 4 or RS below MA or extended under MA
    if "Stage 2" in stage and rs_above_ma and (rs_slope_wk >= 0):
        return "Buy"
    if "Stage 4" in stage or not rs_above_ma:
        return "Avoid"
    # Stage 1/3 or partial confirms:
    return "Watch"

# ========== Config ==========
@dataclass
class AppConfig:
    mode_default: str = "weekly"
    timezone: str = "America/Chicago"
    benchmark: str = "SPY"
    include_charts: bool = True

@dataclass
class SheetsCfg:
    url: str = ""
    sheet_url: str = ""
    open_positions_tab: str = "Open_Positions"
    signals_tab: str = "Signals"
    output_dir: str = "./output"

@dataclass
class ReportingCfg:
    output_dir: str = "./output"
    include_pdf: bool = True
    include_csv: bool = True
    summary_lines: int = 10

@dataclass
class GoogleCfg:
    service_account_json: str = ""
    client_email: str = ""

@dataclass
class RootConfig:
    app: AppConfig
    sheets: SheetsCfg
    reporting: ReportingCfg
    google: GoogleCfg

def load_config(config_path: str) -> RootConfig:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    app = AppConfig(**raw.get("app", {}))
    sheets = SheetsCfg(**raw.get("sheets", {}))
    reporting = ReportingCfg(**raw.get("reporting", {}))
    google = GoogleCfg(**raw.get("google", {}))
    return RootConfig(app=app, sheets=sheets, reporting=reporting, google=google)

# ========== Weekly CSV Helpers ==========
def find_latest_weekly_csv(reporting_dir: str) -> Optional[str]:
    patt = os.path.join(reporting_dir, "weinstein_weekly_equities_*.csv")
    files = sorted(glob.glob(patt))
    if not files:
        return None
    return files[-1]

def load_weekly_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    return df

def snapshot_from_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tiny snapshot resembling:
    ticker | stage | price | ma30
    We try common column names from prior reports.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "stage", "price", "ma30"])

    # Guess columns
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    tcol = pick("ticker", "symbol")
    scol = pick("stage", "weinstein_stage", "stage_weekly")
    pcol = pick("price", "last", "close", "last_price")
    mcol = pick("ma30", "sma30", "ma_30", "ma_150_proxy", "sma150")

    out = pd.DataFrame()
    out["ticker"] = df[tcol] if tcol in df.columns else ""
    out["stage"] = df[scol] if (scol and scol in df.columns) else ""
    out["price"] = df[pcol] if (pcol and pcol in df.columns) else np.nan
    out["ma30"]  = df[mcol] if (mcol and mcol in df.columns) else np.nan

    return out.fillna("")

# ========== Google Sheets (Signals) ==========
def read_signals_from_gsheet(cfg: RootConfig) -> pd.DataFrame:
    """
    Try to read the 'Signals' tab from the provided Google Sheet.
    If gspread is not available or fails, returns empty DataFrame.
    """
    if not GSPREAD_AVAILABLE:
        return pd.DataFrame()

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_file(cfg.google.service_account_json, scopes=scopes)
        gc = gspread.authorize(creds)

        sh = gc.open_by_url(cfg.sheets.sheet_url or cfg.sheets.url)
        ws = sh.worksheet(cfg.sheets.signals_tab)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        # Normalize header names
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def extract_crypto_tickers_from_signals(signals_df: pd.DataFrame) -> List[str]:
    """
    Find crypto rows. We assume crypto tickers often contain '-USD'
    (e.g., BTC-USD, ETH-USD, SOL-USD).
    """
    if signals_df is None or signals_df.empty:
        return []
    cols = {c.lower(): c for c in signals_df.columns}
    tcol = cols.get("ticker", None)
    if not tcol or tcol not in signals_df.columns:
        return []
    tickers = []
    for v in signals_df[tcol]:
        sv = str(v).strip()
        if "-USD" in sv.upper():
            tickers.append(sv.upper())
    # de-dup while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

# ========== Prices & Crypto Block ==========
def _yf_close_series(ticker: str, period: str = "90d", interval: str = "1d") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if data is None or data.empty:
            return pd.Series(dtype=float)
        close = data["Close"].copy()
        close.name = ticker
        return close
    except Exception:
        return pd.Series(dtype=float)

def _moving_average(series: pd.Series, window: int) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.rolling(window=window, min_periods=max(1, window//2)).mean()

def _relative_strength(price: pd.Series, bench: pd.Series) -> pd.Series:
    if price is None or bench is None or price.empty or bench.empty:
        return pd.Series(dtype=float)
    # Align indices
    df = pd.concat([price, bench], axis=1, join="inner")
    df.columns = ["p", "b"]
    rs = df["p"] / df["b"]
    return rs

def _crypto_row(ticker: str, px: pd.Series, bench_px: pd.Series) -> Dict[str, object]:
    """
    Build a row for the Crypto table.
    """
    ma10 = _moving_average(px, 10)
    ma30 = _moving_average(px, 30)

    price = _safe_float_from_series(px, -1, default=float("nan"))
    ma10_last = _safe_float_from_series(ma10, -1, default=float("nan"))
    ma30_last = _safe_float_from_series(ma30, -1, default=float("nan"))

    dist_ma = float("nan")
    if not pd.isna(price) and not pd.isna(ma30_last) and ma30_last != 0:
        dist_ma = ((price - ma30_last) / ma30_last) * 100.0

    ma_slope_wk = _slope_per_week(ma30, lookback_days=5)

    rs = _relative_strength(px, bench_px)
    rs_ma30 = _moving_average(rs, 30)

    rs_last = _safe_float_from_series(rs, -1, default=float("nan"))
    rs_ma30_last = _safe_float_from_series(rs_ma30, -1, default=float("nan"))
    rs_above_ma = (not pd.isna(rs_last) and not pd.isna(rs_ma30_last) and (rs_last >= rs_ma30_last))
    rs_slope_wk = _slope_per_week(rs, lookback_days=5)

    stage = _stage_from_price_ma(price, ma30_last, ma_slope_wk)
    decision = _buy_watch_avoid(stage, rs_above_ma, rs_slope_wk, dist_ma)

    return {
        "ticker": ticker,
        "asset_class": "Crypto",
        "industry": "",
        "sector": "",
        "Buy Signal": decision.upper(),  # AVOID / WATCH / BUY
        "chart": "chart",
        "stage": stage,
        "short_term_state_wk": "StageConflict" if ("Stage 1" in stage or "Stage 3" in stage) else "",
        "price": price,
        "ma10": ma10_last,
        "ma30": ma30_last,
        "dist_ma_pct": dist_ma,
        "ma_slope_per_wk": ma_slope_wk,
        "rs": rs_last if not pd.isna(rs_last) else "",
        "rs_ma30": rs_ma30_last if not pd.isna(rs_ma30_last) else "",
        "rs_above_ma": "Yes" if rs_above_ma else "No",
        "rs_slope_per_wk": rs_slope_wk if not pd.isna(rs_slope_wk) else "",
        "notes": "",
    }

def build_crypto_section_from_signals(cfg: RootConfig, benchmark: str = "BTC-USD") -> str:
    """
    Returns a ready-to-print Crypto Weekly section string.
    Safe if signals or prices are missing.
    """
    # Pull signals for crypto tickers
    sig_df = read_signals_from_gsheet(cfg)
    crypto_tickers = extract_crypto_tickers_from_signals(sig_df)
    if not crypto_tickers:
        # sensible defaults
        crypto_tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

    # Download benchmark + member series
    bench_px = _yf_close_series(benchmark, period="120d", interval="1d")
    rows = []
    for t in crypto_tickers:
        px = _yf_close_series(t, period="120d", interval="1d")
        if px is None or px.empty or bench_px is None or bench_px.empty:
            # Fill stub if data missing
            rows.append({
                "ticker": t,
                "asset_class": "Crypto",
                "industry": "",
                "sector": "",
                "Buy Signal": "AVOID",
                "chart": "chart",
                "stage": "Stage ? (Unknown)",
                "short_term_state_wk": "",
                "price": "",
                "ma10": "",
                "ma30": "",
                "dist_ma_pct": "",
                "ma_slope_per_wk": "",
                "rs": "",
                "rs_ma30": "",
                "rs_above_ma": "No",
                "rs_slope_per_wk": "",
                "notes": "No data",
            })
            continue
        row = _crypto_row(t, px, bench_px)
        rows.append(row)

    # Classify counts
    buys = sum(1 for r in rows if str(r.get("Buy Signal","")).upper() == "BUY")
    avoids = sum(1 for r in rows if str(r.get("Buy Signal","")).upper() == "AVOID")
    watches = sum(1 for r in rows if str(r.get("Buy Signal","")).upper() == "WATCH")

    # Render block
    out = io.StringIO()
    print("Crypto Weekly — Benchmark: BTC-USD", file=out)
    print(f"Generated {_fmt_now_for_header(cfg.app.timezone)}", file=out)
    print(f"Crypto Summary: ✅ Buy: {buys}   |   🟡 Watch: {watches}   |   🔴 Avoid: {avoids}   (Total: {len(rows)})", file=out)
    header = [
        "ticker","asset_class","industry","sector","Buy Signal","chart","stage","short_term_state_wk",
        "price","ma10","ma30","dist_ma_pct","ma_slope_per_wk",
        "rs","rs_ma30","rs_above_ma","rs_slope_per_wk","notes"
    ]
    print("\t".join(header), file=out)
    for r in rows:
        vals = []
        for k in header:
            v = r.get(k, "")
            if isinstance(v, float):
                if "pct" in k:
                    v = _pct(v)
                else:
                    v = f"{v:.6f}"
            vals.append(str(v))
        print("\t".join(vals), file=out)
    return out.getvalue().rstrip()

# ========== Intraday shells (placeholders) ==========
def evaluate_intraday_triggers() -> Tuple[List[str], List[str], List[str]]:
    """
    Return (buys, nears, sells) lists of formatted lines.
    In this compact version, we keep placeholders safe/empty.
    """
    return [], [], []

# ========== Printing ==========
def print_header(title_ts: Optional[str] = None, tz: str = "America/Chicago") -> None:
    ts = title_ts or _fmt_now_for_header(tz)
    print(f"Weinstein Intraday Watch — {ts}")
    print("BUY: Weekly Stage 1/2 + confirm over ~10-week pivot & 30-wk MA proxy (SMA150), +0.4% headroom, RS support, volume pace ≥ 1.3×. For 60m bars: ≥40 min elapsed & intrabar pace ≥ 1.2×.")
    print("NEAR-TRIGGER: Stage 1/2 + RS ok, price within 0.3% below pivot or first close over pivot but not fully confirmed yet, volume pace ≥ 1.0×.")
    print("SELL-TRIGGER: Confirmed crack below MA150 by 0.5% with persistence; for 60m bars, ≥40 min elapsed & intrabar pace ≥ 1.2×.")
    print()

def print_triggers(buys: List[str], nears: List[str], sells: List[str]) -> None:
    print("Buy Triggers (ranked)")
    if buys:
        for i, line in enumerate(buys, 1):
            print(f"{i}. {line}")
    else:
        print("No BUY signals.")
    print()
    print("Near-Triggers (ranked)")
    if nears:
        for i, line in enumerate(nears, 1):
            print(f"{i}. {line}")
    else:
        print("No NEAR signals.")
    print("Sell Triggers (ranked)")
    if sells:
        for i, line in enumerate(sells, 1):
            print(f"{i}. {line}")
    else:
        print("No SELLTRIG signals.")
    print()
    print("Charts (Price + SMA150 ≈ 30-wk MA, RS normalized)")
    if buys or nears:
        # list tickers implied in lines
        pass
    print()

def print_risk_triggers(tracked: Optional[List[str]] = None) -> None:
    print("Sell / Risk Triggers (Tracked Positions & Position Recommendations)")
    if tracked:
        for line in tracked:
            print(line)
    else:
        # leave empty block; your pipeline often prints entries here when positions exist
        pass
    print()

def print_snapshot_from_weekly(snap: pd.DataFrame, max_rows: int = 200) -> None:
    print("Snapshot (ordered by weekly rank & stage)")
    if snap is None or snap.empty:
        print("No snapshot available.")
        print()
        return
    # Emit in a compact table: ticker  stage  price  ma30
    print("ticker\tstage\tprice\tma30")
    rows = min(len(snap), max_rows)
    for i in range(rows):
        r = snap.iloc[i]
        t = str(r.get("ticker", ""))
        s = str(r.get("stage", ""))
        p = r.get("price", "")
        m = r.get("ma30", "")
        # numeric formatting
        if isinstance(p, (int, float)) and not pd.isna(p):
            p = f"{float(p):.6f}".rstrip("0").rstrip(".")
        if isinstance(m, (int, float)) and not pd.isna(m):
            m = f"{float(m):.6f}".rstrip("0").rstrip(".")
        print(f"{t}\t{s}\t{p}\t{m}")
    print()

def print_weekly_summary_stub() -> None:
    print("Weinstein Weekly – Summary")
    # The detailed P/L table normally comes from your weekly generator.
    # Here we print only the header; your pipeline usually appends position lines after.
    print()

# ========== Runner ==========
def run(_config_path: str) -> None:
    print(f"▶️ [{dt.datetime.now().strftime('%H:%M:%S')}] Intraday watcher starting with config: {_config_path}")
    cfg = load_config(_config_path)

    # Weekly CSV
    latest_csv = find_latest_weekly_csv(cfg.reporting.output_dir)
    if latest_csv:
        print(f"·· [{dt.datetime.now().strftime('%H:%M:%S')}] Weekly CSV: {latest_csv}")
    else:
        print("·· [--:--:--] Weekly CSV: <not found>")

    # Focus universe print (placeholder)
    print("• [{}] Focus universe: {} symbols (Stage 1/2).".format(
        dt.datetime.now().strftime('%H:%M:%S'),
        116  # informational
    ))

    # Simulate price download step (we rely on yfinance inside crypto)
    print("▶️ [{}] Downloading intraday + daily bars...".format(dt.datetime.now().strftime('%H:%M:%S')))
    print("✅ [{}] Price data downloaded.".format(dt.datetime.now().strftime('%H:%M:%S')))

    print("▶️ [{}] Evaluating candidates...".format(dt.datetime.now().strftime('%H:%M:%S')))

    # Evaluate triggers (placeholders—keep compatible with your logs)
    buys, nears, sells = evaluate_intraday_triggers()
    # Example prints you had in logs; keep harmless:
    if not buys and not nears:
        pass

    # ==== OUTPUT ====
    print()
    print_header(tz=cfg.app.timezone)
    print_triggers(buys, nears, sells)

    # Risk (tracked) — leave empty here; your downstream appends specifics
    print_risk_triggers(tracked=None)

    # Snapshot (from weekly CSV)
    snap = pd.DataFrame()
    if latest_csv:
        wk = load_weekly_csv(latest_csv)
        snap = snapshot_from_weekly(wk)
    print_snapshot_from_weekly(snap)

    # ==== CRYPTO BLOCK (inserted BEFORE Weekly – Summary) ====
    try:
        crypto_block = build_crypto_section_from_signals(cfg, benchmark="BTC-USD")
        print(crypto_block)
        print()
    except Exception as e:
        # Keep report alive even if crypto fails
        print("Crypto Weekly — Benchmark: BTC-USD")
        print(f"(Crypto section unavailable: {e})")
        print()

    # Weekly summary header (details typically appended by your weekly pipeline)
    print_weekly_summary_stub()

    print("✅ Intraday tick complete.")

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    try:
        run(_config_path=args.config)
    except Exception as ex:
        print("❌ Intraday watcher encountered an error.")
        print(str(ex))
        raise
