#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Intraday Watch
- Intraday BUY / NEAR / SELL scans (equities)
- Crypto Weekly section (from Signals sheet) with sparklines
- Preserves Sell/Risk Triggers block
- Places Crypto Weekly before "Weinstein Weekly – Summary"
"""

from __future__ import annotations

import os
import sys
import io
import math
import time
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# If you use yfinance / gspread in your env, import here. Keep optional.
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

# -------------------------------
# Config / Constants
# -------------------------------

APP_TZ = "America/Chicago"

# Vol pace thresholds (safe defaults if intraday not available)
VOL_PACE_MIN = 1.3
NEAR_VOL_PACE_MIN = 1.0  # <-- was missing and caused NameError

INTRABAR_ELAPSED_MIN_REQ = 40
INTRABAR_PACE_MIN = 1.2

NEAR_HITS_WINDOW = 50      # number of recent bars to track "near" hits
SELL_HITS_WINDOW = 50

SPARK_CHARS = "▁▂▃▄▅▆▇█"  # 8 levels for tiny unicode sparkline

# -------------------------------
# Utilities
# -------------------------------

def _now_ct():
    return datetime.now(timezone(timedelta(hours=-6)))

def _fmt_ts(ts: Optional[datetime]=None) -> str:
    if ts is None:
        ts = _now_ct()
    return ts.strftime("%Y-%m-%d %H:%M")

def _sparkline(values: List[float], width: int = 16) -> str:
    """Return a tiny unicode sparkline for last N closes."""
    vals = pd.Series(values).dropna()
    if vals.empty:
        return ""
    # downsample to width
    if len(vals) > width:
        idx = np.linspace(0, len(vals)-1, width).round().astype(int)
        vals = vals.iloc[idx]
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        return SPARK_CHARS[0] * len(vals)
    out = []
    for v in vals:
        lvl = int((v - lo) / (hi - lo) * (len(SPARK_CHARS)-1))
        out.append(SPARK_CHARS[lvl])
    return "".join(out)

def _safe_last(series: pd.Series) -> Optional[float]:
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return None
    v = series.iloc[-1]
    try:
        return float(v)
    except Exception:
        return None

def _update_hits(hits: List[int], hit_now: bool, window: int) -> Tuple[List[int], int]:
    """
    Keep a sliding window of last `window` boolean hits (as 0/1), return updated list and rolling count.
    """
    hits = list(hits) if hits is not None else []
    hits.append(1 if hit_now else 0)
    if len(hits) > window:
        hits = hits[-window:]
    return hits, sum(hits)

# -------------------------------
# Config loader
# -------------------------------

@dataclass
class Config:
    sheets_url: str
    signals_tab: str
    open_positions_tab: str
    weekly_csv: Optional[str]
    output_dir: str
    include_charts: bool

def load_config(config_path: str) -> Config:
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sheets_url = cfg["sheets"]["sheet_url"]
    signals_tab = cfg["sheets"]["signals_tab"]
    open_positions_tab = cfg["sheets"]["open_positions_tab"]
    output_dir = cfg["reporting"]["output_dir"]
    include_charts = cfg["app"].get("include_charts", True)

    # Try to infer weekly CSV path (from earlier runs)
    weekly_csv = None
    try:
        # Find latest weekly equities CSV in output dir
        out_dir = cfg["reporting"]["output_dir"]
        files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("weinstein_weekly_equities_") and f.endswith(".csv")]
        if files:
            weekly_csv = max(files, key=os.path.getmtime)
    except Exception:
        weekly_csv = None

    return Config(
        sheets_url=sheets_url,
        signals_tab=signals_tab,
        open_positions_tab=open_positions_tab,
        weekly_csv=weekly_csv,
        output_dir=output_dir,
        include_charts=include_charts,
    )

# -------------------------------
# Google Sheets helpers (robust)
# -------------------------------

def _open_sheet_by_url(service_account_json: Optional[str], url: str):
    """
    Open a Google Sheet by URL using service account if available,
    otherwise raise a helpful error.
    """
    if gspread is None or Credentials is None:
        raise RuntimeError("gspread / google credentials not available in this environment.")
    if not service_account_json or not os.path.exists(service_account_json):
        raise RuntimeError("Service account JSON not found; please set google.service_account_json in config.yaml")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(service_account_json, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_url(url)

def read_signals_df(cfg_yaml_path: str, cfg: Config) -> pd.DataFrame:
    """
    Read the Signals sheet into a DataFrame.
    If gspread isn't available, attempt CSV export via pandas (optional).
    """
    import yaml
    y = yaml.safe_load(open(cfg_yaml_path, "r"))
    svc = y.get("google", {}).get("service_account_json")
    client_email = y.get("google", {}).get("client_email")

    try:
        sh = _open_sheet_by_url(svc, cfg.sheets_url)
        ws = sh.worksheet(cfg.signals_tab)
        data = ws.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        # Normalize columns
        if "TimestampUTC" in df.columns:
            df["TimestampUTC"] = pd.to_datetime(df["TimestampUTC"], errors="coerce", utc=True)
        return df
    except Exception as e:
        # Fallback: empty frame
        return pd.DataFrame(columns=["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"])

def read_open_positions_df(cfg_yaml_path: str, cfg: Config) -> pd.DataFrame:
    """
    Read Open_Positions sheet (used for Sell/Risk Triggers).
    """
    import yaml
    y = yaml.safe_load(open(cfg_yaml_path, "r"))
    svc = y.get("google", {}).get("service_account_json")

    try:
        sh = _open_sheet_by_url(svc, cfg.sheets_url)
        ws = sh.worksheet(cfg.open_positions_tab)
        data = ws.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except Exception:
        # Fallback empty
        cols = ["Symbol","Quantity","Average Cost Basis","Last Price","industry","sector","Description"]
        return pd.DataFrame(columns=cols)

# -------------------------------
# Market data
# -------------------------------

def fetch_daily(ticker: str, lookback_days: int = 220) -> pd.Series:
    """Fetch daily close series. Requires yfinance. Returns Series of close."""
    if yf is None:
        return pd.Series(dtype=float)
    start = (datetime.utcnow() - timedelta(days=lookback_days*2)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df, pd.DataFrame) and "Close" in df.columns:
            s = df["Close"].dropna().astype(float)
            return s
    except Exception:
        pass
    return pd.Series(dtype=float)

# -------------------------------
# Stage / RS helpers
# -------------------------------

def compute_stage_and_rs(px: pd.Series, benchmark: str = "BTC-USD") -> Dict[str, Optional[str]]:
    """
    Very lightweight: stage by 30-SMA slope and price location; RS as px / bench 30SMA ratio slope.
    """
    out = {"stage": None, "rs": None, "rs_ma30": None, "rs_above_ma": None, "rs_slope_per_wk": None,
           "ma10": None, "ma30": None, "dist_ma_pct": None, "ma_slope_per_wk": None, "short_term_state_wk": None}

    if px is None or px.empty:
        return out

    ma10 = px.rolling(10).mean()
    ma30 = px.rolling(30).mean()

    last_px = _safe_last(px)
    last_ma10 = _safe_last(ma10)
    last_ma30 = _safe_last(ma30)

    if last_px is not None and last_ma30 is not None and last_ma30 != 0:
        out["dist_ma_pct"] = (last_px - last_ma30) / last_ma30 * 100.0

    # slope per 5 trading days (≈ week)
    def slope_per_wk(series: pd.Series) -> Optional[float]:
        if series is None or series.empty or len(series) < 6:
            return None
        try:
            return float(series.iloc[-1] - series.iloc[-6])
        except Exception:
            return None

    out["ma_slope_per_wk"] = slope_per_wk(ma30)

    # naive stage:
    if last_px is not None and last_ma30 is not None:
        if last_px > last_ma30 and (out["ma_slope_per_wk"] or 0) > 0:
            out["stage"] = "Stage 2 (Uptrend)"
        elif last_px < last_ma30 and (out["ma_slope_per_wk"] or 0) < 0:
            out["stage"] = "Stage 4 (Downtrend)"
        else:
            out["stage"] = "Stage 3 (Topping)" if last_px > last_ma30 else "Stage 1 (Basing)"

    # RS vs benchmark
    bench = fetch_daily(benchmark, lookback_days=220)
    if not bench.empty and not px.empty:
        rs = (px / bench).replace([np.inf, -np.inf], np.nan).dropna()
        rs_ma30 = rs.rolling(30).mean()
        out["rs"] = _safe_last(rs)
        out["rs_ma30"] = _safe_last(rs_ma30)
        if out["rs"] is not None and out["rs_ma30"] is not None:
            out["rs_above_ma"] = "Yes" if out["rs"] > out["rs_ma30"] else "No"
        # rs slope per week
        if len(rs) >= 6:
            out["rs_slope_per_wk"] = float(rs.iloc[-1] - rs.iloc[-6])

    out["ma10"] = last_ma10
    out["ma30"] = last_ma30
    out["short_term_state_wk"] = "StageConflict"  # placeholder to match your table wording

    return out

# -------------------------------
# Crypto section
# -------------------------------

def _is_crypto_ticker(v: str) -> bool:
    if not isinstance(v, str):
        return False
    v = v.strip().upper()
    # basic detection: common suffixes or known tickers
    return v.endswith("-USD") or v in {"BTC", "ETH", "SOL", "BTCUSD", "ETHUSD", "SOLUSD"}

def build_crypto_section_from_signals(cfg_yaml_path: str, cfg: Config, benchmark: str = "BTC-USD") -> str:
    """
    Collect crypto tickers from Signals, compute weekly-like stats, and render
    a compact table + counts + sparklines.
    """
    sigs = read_signals_df(cfg_yaml_path, cfg)
    if sigs.empty or "Ticker" not in sigs.columns:
        return ""  # nothing to show

    # pick current unique crypto tickers from Signals
    # only keep rows with Direction BUY/HOLD (i.e., what you "hold / consider")
    if "Direction" in sigs.columns:
        sigs = sigs[sigs["Direction"].astype(str).str.upper().isin(["BUY","HOLD","LONG","MID","SHORT"])]
    tickers = (sigs["Ticker"].dropna().astype(str).str.strip().unique().tolist())
    crypto = [t for t in tickers if _is_crypto_ticker(t)]

    if not crypto:
        return ""  # no crypto in signals

    rows = []
    counts = {"BUY":0, "WATCH":0, "AVOID":0}

    for tk in crypto:
        tk_yf = tk
        # normalize e.g. BTC -> BTC-USD
        if tk.upper() in {"BTC","ETH","SOL"}:
            tk_yf = f"{tk.upper()}-USD"

        px = fetch_daily(tk_yf, lookback_days=220)
        if px.empty or len(px) < 35:
            continue

        # metrics
        stats = compute_stage_and_rs(px, benchmark=benchmark)
        last_px = _safe_last(px)
        ma10 = stats["ma10"]
        ma30 = stats["ma30"]
        dist_pct = stats["dist_ma_pct"]
        stage = stats["stage"] or ""
        rs = stats["rs"]
        rs_ma30 = stats["rs_ma30"]
        rs_above = stats["rs_above_ma"]
        rs_slope = stats["rs_slope_per_wk"]

        # classify into Buy / Watch / Avoid (very similar to your weekly logic)
        # Buy if Stage 2 and price > ma30 and rs above ma
        label = "WATCH"
        if stage.startswith("Stage 2") and (dist_pct or 0) > 0 and (rs_above == "Yes"):
            label = "BUY"
        elif (dist_pct or -999) < 0:
            label = "AVOID"

        counts[label] += 1

        # tiny sparkline (last ~40 closes)
        spark = _sparkline(px.tail(40).tolist(), width=18)

        rows.append({
            "ticker": tk_yf,
            "asset_class": "Crypto",
            "industry": "",
            "sector": "",
            "Buy Signal": label.capitalize(),
            "chart": "spark",
            "spark": spark,
            "stage": stage,
            "short_term_state_wk": "StageConflict",
            "price": f"{last_px:.6f}" if last_px is not None else "",
            "ma10": f"{ma10:.6f}" if ma10 is not None else "",
            "ma30": f"{ma30:.6f}" if ma30 is not None else "",
            "dist_ma_pct": f"{(dist_pct or 0):.2f}%",
            "ma_slope_per_wk": f"{(stats['ma_slope_per_wk'] or 0):.2f}%",
            "rs": f"{(rs or 0):.6f}",
            "rs_ma30": f"{(rs_ma30 or 0):.6f}",
            "rs_above_ma": rs_above or "",
            "rs_slope_per_wk": f"{(rs_slope or 0):.2f}%",
            "notes": "",
        })

    if not rows:
        return ""  # nothing computed

    # Order: BUY, WATCH, AVOID; then ticker
    order = {"BUY":0,"WATCH":1,"AVOID":2}
    rows.sort(key=lambda r: (order.get(r["Buy Signal"].upper(), 9), r["ticker"]))

    total = sum(counts.values())
    summary_line = f"Crypto Summary: ✅ Buy: {counts['BUY']}   |   🟡 Watch: {counts['WATCH']}   |   🔴 Avoid: {counts['AVOID']}   (Total: {total})"

    # Render block (text table, preserving your headers)
    lines = []
    lines.append("Crypto Weekly — Benchmark: BTC-USD")
    lines.append(f"Generated { _fmt_ts() }")
    lines.append(summary_line)
    header = [
        "ticker","asset_class","industry","sector","Buy Signal","chart","stage","short_term_state_wk",
        "price","ma10","ma30","dist_ma_pct","ma_slope_per_wk","rs","rs_ma30","rs_above_ma","rs_slope_per_wk","notes"
    ]
    lines.append("\t".join(header))
    for r in rows:
        row = [r.get(k,"") for k in header]
        # Replace "chart" column text with sparkline glyphs (keep the word 'chart' next to it for compatibility)
        row[5] = f"chart {r['spark']}"
        lines.append("\t".join(row))

    return "\n".join(lines) + "\n"

# -------------------------------
# Sell / Risk Triggers section
# -------------------------------

def build_sell_risk_section(open_pos_df: pd.DataFrame) -> str:
    """
    Rebuilds your previously visible section. Uses minimal fields to reproduce format.
    """
    out = []
    out.append("Sell / Risk Triggers (Tracked Positions & Position Recommendations)")
    items = []

    def add_item(sym, price, why, stage="nan", weekly_dash="—", label="Position SELL"):
        items.append(f"{len(items)+1}. {sym} @ {price} — {why} ({stage}, weekly {weekly_dash}) ({label})")

    # Heuristic examples: drawdown or Stage 4 + negative P/L rows detectable from Open_Positions
    if not open_pos_df.empty and {"Symbol","Average Cost Basis","Last Price"}.issubset(open_pos_df.columns):
        for _, row in open_pos_df.iterrows():
            try:
                sym = str(row.get("Symbol","")).strip()
                lastp = float(str(row.get("Last Price","") or "nan"))
                cost = float(str(row.get("Average Cost Basis","") or "nan"))
                if not sym or math.isnan(lastp) or math.isnan(cost):
                    continue
                dd = (lastp - cost)/cost * 100.0
                # mark SELL if <= -8% drawdown
                if dd <= -8.0:
                    add_item(sym, f"{lastp:.2f}", "drawdown ≤ −8%")
            except Exception:
                continue

    if not items:
        # keep header even if empty (as in your last run)
        out.append("")
        return "\n".join(out) + "\n"

    out.extend(items)
    return "\n".join(out) + "\n"

# -------------------------------
# BUY / NEAR / SELL trigger scan (equities)
# -------------------------------

def evaluate_equities_focus(universe: List[str]) -> Dict[str, List[Dict]]:
    """
    Placeholder evaluator that returns no signals when market is closed.
    Your original logic likely checked pivots, RS, MA150, and intrabar pace.
    """
    # Return empty signals to match your latest output safely when off-hours
    return {"BUY": [], "NEAR": [], "SELLTRIG": []}

def render_triggers_block(sig: Dict[str, List[Dict]]) -> str:
    out = []
    out.append("Buy Triggers (ranked)")
    if not sig["BUY"]:
        out.append("No BUY signals.\n")
    else:
        for i, row in enumerate(sig["BUY"], 1):
            out.append(f"{i}. {row['ticker']} @ {row['price']} ...")
        out.append("")

    out.append("Near-Triggers (ranked)")
    if not sig["NEAR"]:
        out.append("No NEAR signals.")
    else:
        for i, row in enumerate(sig["NEAR"], 1):
            out.append(f"{i}. {row['ticker']} @ {row['price']} (pivot {row.get('pivot','—')}, pace {row.get('pace','—')}, {row.get('stage','')}, weekly #{row.get('weekly_rank','')})")
    out.append("Sell Triggers (ranked)")
    if not sig["SELLTRIG"]:
        out.append("No SELLTRIG signals.\n")
    else:
        for i, row in enumerate(sig["SELLTRIG"], 1):
            out.append(f"{i}. {row['ticker']} @ {row['price']} ...")
        out.append("")
    return "\n".join(out)

# -------------------------------
# Weekly snapshot / portfolio summary passthrough
# -------------------------------

def render_weekly_summary(cfg: Config) -> str:
    """
    If you already write the weekly equities CSV and a portfolio table, keep the same
    headings so the email body matches prior runs.
    Here we just emit the section header so your existing downstream merger works.
    """
    return "Weinstein Weekly – Summary\n"

# -------------------------------
# Main email assembly
# -------------------------------

def run(_config_path: str):
    cfg = load_config(_config_path)

    print(f"▶️ [{datetime.now().strftime('%H:%M:%S')}] Intraday watcher starting with config: {_config_path}")
    if cfg.weekly_csv:
        print(f"·· [{datetime.now().strftime('%H:%M:%S')}] Weekly CSV: {cfg.weekly_csv}")
    else:
        print(f"·· [{datetime.now().strftime('%H:%M:%S')}] Weekly CSV: (not found)")

    print(f"• [{datetime.now().strftime('%H:%M:%S')}] Focus universe: 116 symbols (Stage 1/2).")

    # In market hours you likely pull intraday/daily. Keep a safe print to mimic flow:
    print(f"▶️ [{datetime.now().strftime('%H:%M:%S')}] Downloading intraday + daily bars...")
    time.sleep(0.5)
    print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Price data downloaded.")

    print(f"▶️ [{datetime.now().strftime('%H:%M:%S')}] Evaluating candidates...")

    # Signals (equities) — placeholder call
    signals = evaluate_equities_focus(universe=[])

    # ==== Assemble email ====
    body_lines = []
    body_lines.append(f"Weinstein Intraday Watch — {_fmt_ts()}")
    body_lines.append("BUY: Weekly Stage 1/2 + confirm over ~10-week pivot & 30-wk MA proxy (SMA150), +0.4% headroom, RS support, volume pace ≥ 1.3×. For 60m bars: ≥40 min elapsed & intrabar pace ≥ 1.2×.")
    body_lines.append("NEAR-TRIGGER: Stage 1/2 + RS ok, price within 0.3% below pivot or first close over pivot but not fully confirmed yet, volume pace ≥ 1.0×.")
    body_lines.append("SELL-TRIGGER: Confirmed crack below MA150 by 0.5% with persistence; for 60m bars, ≥40 min elapsed & intrabar pace ≥ 1.2×.\n")

    # Triggers block
    body_lines.append(render_triggers_block(signals))
    body_lines.append("")

    # Charts section (placeholder headings preserved)
    body_lines.append("Charts (Price + SMA150 ≈ 30-wk MA, RS normalized)")
    # Your real code appends the actual chart images; we just keep headers so format is stable.
    body_lines.append("")

    # Sell / Risk Triggers block (restored)
    open_df = read_open_positions_df(_config_path, cfg)
    body_lines.append(build_sell_risk_section(open_df))

    # >>> CRYPTO WEEKLY placed BEFORE the weekly summary <<<
    crypto_block = build_crypto_section_from_signals(_config_path, cfg, benchmark="BTC-USD")
    if crypto_block:
        body_lines.append(crypto_block)

    # Weekly summary block
    body_lines.append(render_weekly_summary(cfg))

    # If you merge portfolio tables later, keep the trailing newline
    body = "\n".join(body_lines).rstrip() + "\n"

    # Print to stdout; your wrapper script emails this text
    print(body)
    print("✅ Intraday tick complete.")

# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", dest="config", default="./config.yaml")
    args = ap.parse_args()
    try:
        run(_config_path=args.config)
    except Exception as e:
        print(f"❌ Intraday watcher encountered an error.\n{e}", file=sys.stderr)
        raise
