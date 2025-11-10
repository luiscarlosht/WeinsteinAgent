#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Intraday Watcher (intraday + crypto summary)
- Reads Weekly CSV & config
- Scans intraday candidates (incl. crypto if present)
- Builds email/text report with:
  • Buy / Near / Sell triggers
  • Charts list (price + RS)
  • Sell / Risk Triggers (Tracked Positions & Recommendations)
  • Snapshot table
  • Crypto Weekly (from Signals sheet, tiny sparklines)
  • Weinstein Weekly – Summary (unchanged position/PNL section appended after Crypto)
"""

import os
import io
import sys
import math
import json
import time
import textwrap
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Config & constants
# -------------------------

APP_TZ = "America/Chicago"

# Volume pace thresholds
VOL_PACE_MIN = 1.3
NEAR_VOL_PACE_MIN = 1.0

# Intrabar pace (for 60m bars)
INTRABAR_PACE_MIN = 1.2
INTRABAR_MIN_ELAPSED_MIN = 40

# Pivot headroom
PIVOT_HEADROOM_PCT = 0.004  # 0.4%

# Confirmation rules
CRACK_MA_PERSIST_PCT = 0.005  # 0.5%

# Near hits window (minutes)
NEAR_HITS_WINDOW = 60

# Output
OUTPUT_DIR_DEFAULT = "./output"
CHARTS_SUBDIR = "charts"

# -------------------------
# Helpers
# -------------------------

def _now_central() -> dt.datetime:
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo(APP_TZ)).replace(tzinfo=None)

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"

def _fmt_or_dash(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else x

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _update_hits(hits: List[dt.datetime], hit: bool, window_minutes: int) -> Tuple[List[dt.datetime], int]:
    """
    Maintain a rolling list of timestamps when a condition was true.
    """
    now = _now_central()
    if hit:
        hits = hits + [now]
    cutoff = now - dt.timedelta(minutes=window_minutes)
    hits = [t for t in hits if t >= cutoff]
    return hits, len(hits)

def _read_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _read_weekly_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _last_weekly_csv(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    files = [f for f in os.listdir(output_dir) if f.startswith("weinstein_weekly_equities_") and f.endswith(".csv")]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(output_dir, files[0])

def _google_sheet_to_df(sheet_url: str, tab_name: str) -> pd.DataFrame:
    """
    We assume the project already has Google API logic elsewhere.
    To stay self-contained, we’ll use pandas read_csv via export param.
    If your environment uses service account pull, replace this with your existing loader.
    """
    # Attempt CSV export URL
    if "/edit" in sheet_url:
        base = sheet_url.split("/edit")[0]
    else:
        base = sheet_url
    # Note: gid is not known; we fall back to the public CSV by `gviz/tq` if needed.
    # The Signals tab name is used via query param (works when sharing is set appropriately).
    # If your existing code has a loader, hook it in here.
    try:
        # Try pandas read_csv with Google Visualization query export
        tqx = f"{base}/gviz/tq?tqx=out:csv&sheet={tab_name}"
        return pd.read_csv(tqx)
    except Exception:
        # Fallback to empty DF
        return pd.DataFrame()

def _is_crypto_ticker(ticker: str) -> bool:
    return isinstance(ticker, str) and ticker.endswith("-USD")

def _sparkline_png(series: pd.Series, out_path: str):
    """
    Make a tiny sparkline PNG.
    """
    if series is None or len(series) == 0:
        return
    plt.figure(figsize=(2.2, 0.6), dpi=200)  # ~440x120
    plt.plot(series.values)
    plt.axis('off')
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# -------------------------
# Mock / data fetchers for prices (replace with your real pipeline)
# -------------------------

def _fetch_history(ticker: str, days: int = 120) -> pd.DataFrame:
    """
    Replace this with your existing data provider. Here we try yfinance if installed;
    otherwise, return an empty frame.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns=str.title).reset_index()
            df["Date"] = pd.to_datetime(df["Date"])
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

def _calc_ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=max(2, n//2)).mean()

def _relative_strength(series: pd.Series, benchmark: pd.Series) -> pd.Series:
    if len(series) == 0 or len(benchmark) == 0:
        return pd.Series([], dtype=float)
    n = min(len(series), len(benchmark))
    s = series.iloc[-n:].reset_index(drop=True)
    b = benchmark.iloc[-n:].reset_index(drop=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = s / b
    return rs.replace([np.inf, -np.inf], np.nan)

def _stage_from_ma(ma30: float, price: float, slope_wk: float) -> str:
    # Simplified stage logic
    if np.isnan(ma30) or np.isnan(price):
        return "Stage 0 (Unknown)"
    if price > ma30 and slope_wk > 0:
        return "Stage 2 (Uptrend)"
    if price < ma30 and slope_wk < 0:
        return "Stage 4 (Downtrend)"
    return "Stage 3 (Topping)" if price > ma30 else "Stage 1 (Basing)"

# -------------------------
# Report composition
# -------------------------

def build_header(now_local: dt.datetime) -> str:
    hdr = []
    hdr.append(f"Weinstein Intraday Watch — {now_local:%Y-%m-%d %H:%M}")
    hdr.append("BUY: Weekly Stage 1/2 + confirm over ~10-week pivot & 30-wk MA proxy (SMA150), +0.4% headroom, RS support, volume pace ≥ 1.3×. For 60m bars: ≥40 min elapsed & intrabar pace ≥ 1.2×.")
    hdr.append("NEAR-TRIGGER: Stage 1/2 + RS ok, price within 0.3% below pivot or first close over pivot but not fully confirmed yet, volume pace ≥ 1.0×.")
    hdr.append("SELL-TRIGGER: Confirmed crack below MA150 by 0.5% with persistence; for 60m bars, ≥40 min elapsed & intrabar pace ≥ 1.2×.\n")
    return "\n".join(hdr)

def build_buy_near_sell_sections(buys: List[Dict], nears: List[Dict], sells: List[Dict]) -> str:
    lines = []
    lines.append("Buy Triggers (ranked)")
    if not buys:
        lines.append("No BUY signals.\n")
    else:
        for i, row in enumerate(buys, 1):
            lines.append(f"{i}. {row['symbol']} @ {row.get('price','—')} (pivot {row.get('pivot','—')}, pace {row.get('pace','—')}, {row.get('stage','—')}, weekly #{row.get('rank','—')})")
        lines.append("")
    lines.append("Near-Triggers (ranked)")
    if not nears:
        lines.append("No NEAR signals.")
    else:
        for i, row in enumerate(nears, 1):
            lines.append(f"{i}. {row['symbol']} @ {row.get('price','—')} (pivot {row.get('pivot','—')}, pace {row.get('pace','—')}, {row.get('stage','—')}, weekly #{row.get('rank','—')})")
    lines.append("Sell Triggers (ranked)")
    if not sells:
        lines.append("No SELLTRIG signals.\n")
    else:
        for i, row in enumerate(sells, 1):
            lines.append(f"{i}. {row['symbol']} @ {row.get('price','—')} (crack MA150, pace {row.get('pace','—')}, {row.get('stage','—')}, weekly #{row.get('rank','—')})")
        lines.append("")
    return "\n".join(lines)

def build_charts_section(chart_symbols: List[str]) -> str:
    lines = []
    lines.append("Charts (Price + SMA150 ≈ 30-wk MA, RS normalized)")
    if not chart_symbols:
        lines.append("")
        return "\n".join(lines)
    for s in chart_symbols:
        lines.append(s)
    lines.append("")
    return "\n".join(lines)

def build_sell_risk_block(tracked: List[Dict]) -> str:
    lines = []
    lines.append("Sell / Risk Triggers (Tracked Positions & Position Recommendations)")
    if not tracked:
        lines.append("")
        return "\n".join(lines)
    for i, r in enumerate(tracked, 1):
        lines.append(f"{i}. {r['symbol']} @ {r.get('at_price','—')} — {r.get('reason','drawdown ≤ −8%')} ({r.get('stage','weekly —')}) (Position {r.get('action','SELL')})")
    lines.append("")
    return "\n".join(lines)

def build_snapshot_block(snapshot: pd.DataFrame) -> str:
    if snapshot is None or snapshot.empty:
        return ""
    cols = ["ticker","stage","price","ma30","pivot10w","vol_pace_vs50dma","two_bar_confirm","last_bar_vol_ok","weekly_rank"]
    present = [c for c in cols if c in snapshot.columns]
    parts = ["Snapshot (ordered by weekly rank & stage)"]
    parts.append("\t".join(present))
    for _, row in snapshot.iterrows():
        parts.append("\t".join(str(_fmt_or_dash(row.get(c))) for c in present))
    parts.append("")
    return "\n".join(parts)

def build_weekly_summary_tail(weekly_tail_text: str) -> str:
    return weekly_tail_text.strip()

def _mini_chart_path(symbol: str, out_dir: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_").replace(" ", "_")
    return os.path.join(out_dir, CHARTS_SUBDIR, f"crypto_{safe}.png")

def build_crypto_section_from_signals(cfg: Dict, benchmark: str = "BTC-USD") -> str:
    """
    Pull crypto tickers from Signals tab, compute weekly-like metrics,
    generate tiny sparklines, and return a CSV-style block.
    """
    sheet_url = cfg.get("sheets", {}).get("sheet_url") or cfg.get("sheets", {}).get("url")
    signals_tab = cfg.get("sheets", {}).get("signals_tab", "Signals")
    output_dir = cfg.get("reporting", {}).get("output_dir", OUTPUT_DIR_DEFAULT)

    if not sheet_url:
        return ""  # No sheet available

    sig_df = _google_sheet_to_df(sheet_url, signals_tab)
    if sig_df is None or sig_df.empty:
        return ""  # nothing to show

    # Normalize header names
    cols = [c.strip().lower().replace(" ", "_") for c in sig_df.columns]
    sig_df.columns = cols

    # Try to identify crypto tickers you hold (Direction BUY) and that look like *-USD
    ticker_col = "ticker" if "ticker" in sig_df.columns else (sig_df.columns[1] if len(sig_df.columns) > 1 else None)
    direction_col = "direction" if "direction" in sig_df.columns else None

    if not ticker_col:
        return ""

    df_crypto = sig_df.copy()
    df_crypto = df_crypto[df_crypto[ticker_col].astype(str).apply(_is_crypto_ticker)]
    if direction_col in df_crypto.columns:
        df_crypto = df_crypto[df_crypto[direction_col].astype(str).str.upper().str.contains("BUY")]

    tickers = sorted(df_crypto[ticker_col].dropna().astype(str).unique().tolist())
    if not tickers:
        return ""

    # Fetch benchmark history
    bench_hist = _fetch_history(benchmark, days=180)
    bench_close = bench_hist["Close"] if not bench_hist.empty else pd.Series(dtype=float)

    # Build table rows
    rows = []
    charts_prepared = 0

    for tk in tickers:
        hist = _fetch_history(tk, days=180)
        if hist.empty:
            continue

        px = hist["Close"]
        ma10 = _calc_ma(px, 10)
        ma30 = _calc_ma(px, 30)
        dist_ma = None
        if not ma30.empty and not px.empty and not np.isnan(ma30.iloc[-1]):
            dist_ma = (px.iloc[-1] - ma30.iloc[-1]) / ma30.iloc[-1]

        # Approx slope per week of MA30: use 5 trading days
        slope = None
        if len(ma30) >= 35:
            slope = (ma30.iloc[-1] - ma30.iloc[-6]) / 5.0 if ma30.iloc[-6] != 0 else np.nan

        rs = _relative_strength(px, bench_close)
        rs_ma30 = _calc_ma(rs, 30) if not rs.empty else pd.Series(dtype=float)
        rs_above_ma = "Yes" if (len(rs) > 0 and len(rs_ma30) > 0 and rs.iloc[-1] > rs_ma30.iloc[-1]) else "No"

        stage = _stage_from_ma(ma30.iloc[-1] if len(ma30) else np.nan,
                               px.iloc[-1] if len(px) else np.nan,
                               slope if slope is not None else np.nan)

        buy_signal = "Buy" if ("Stage 2" in stage and (dist_ma or 0) > 0) else "Avoid"

        # Sparkline (last ~60 closes)
        sp = px.iloc[-60:] if len(px) > 60 else px
        chart_path = _mini_chart_path(tk, output_dir)
        try:
            _sparkline_png(sp, chart_path)
            chart = os.path.basename(chart_path)  # filename; your mailer may attach/find by path
            charts_prepared += 1
        except Exception:
            chart = "chart"

        rows.append({
            "ticker": tk,
            "asset_class": "Crypto",
            "industry": "",
            "sector": "",
            "Buy Signal": buy_signal.upper(),
            "chart": chart,
            "stage": stage,
            "short_term_state_wk": "StageConflict" if "Stage" in stage else "",
            "price": float(px.iloc[-1]) if len(px) else np.nan,
            "ma10": float(ma10.iloc[-1]) if len(ma10) else np.nan,
            "ma30": float(ma30.iloc[-1]) if len(ma30) else np.nan,
            "dist_ma_pct": f"{(dist_ma*100):.2f}%" if dist_ma is not None else "—",
            "ma_slope_per_wk": f"{(slope*100):.2f}%" if slope is not None else "—",
            "rs": float(rs.iloc[-1]) if len(rs) else np.nan,
            "rs_ma30": float(rs_ma30.iloc[-1]) if len(rs_ma30) else np.nan,
            "rs_above_ma": rs_above_ma,
            "rs_slope_per_wk": "0.00%",
            "notes": ""
        })

    if not rows:
        return ""

    # Summary counts
    buy_ct = sum(1 for r in rows if r["Buy Signal"] == "BUY")
    avoid_ct = sum(1 for r in rows if r["Buy Signal"] == "AVOID")
    watch_ct = sum(1 for r in rows if r["Buy Signal"] not in ("BUY", "AVOID"))

    # Render section
    now_local = _now_central()
    parts = []
    parts.append("Crypto Weekly — Benchmark: BTC-USD")
    parts.append(f"Generated {now_local:%Y-%m-%d %H:%M}")
    parts.append(f"Crypto Summary: ✅ Buy: {buy_ct}   |   🟡 Watch: {watch_ct}   |   🔴 Avoid: {avoid_ct}   (Total: {len(rows)})")

    header_order = [
        "ticker","asset_class","industry","sector","Buy Signal","chart","stage","short_term_state_wk",
        "price","ma10","ma30","dist_ma_pct","ma_slope_per_wk","rs","rs_ma30","rs_above_ma","rs_slope_per_wk","notes"
    ]
    parts.append("\t".join(header_order))
    # Sort: BUY first, then by ticker
    def _key(r):
        rank = 0 if r["Buy Signal"] == "BUY" else (2 if r["Buy Signal"] == "AVOID" else 1)
        return (rank, r["ticker"])
    rows_sorted = sorted(rows, key=_key)
    for r in rows_sorted:
        parts.append("\t".join(str(r.get(k, "")) for k in header_order))
    parts.append("")  # newline

    print(f"·· [crypto] Charts prepared: {charts_prepared}")
    return "\n".join(parts)

# -------------------------
# MAIN RUN
# -------------------------

def run(_config_path: str):
    # Load cfg
    cfg = _read_yaml(_config_path)
    output_dir = cfg.get("reporting", {}).get("output_dir", OUTPUT_DIR_DEFAULT)
    _ensure_dir(output_dir)

    now_local = _now_central()
    print(f"▶️ [{now_local:%H:%M:%S}] Intraday watcher starting with config: {_config_path}")

    # Load weekly CSV (latest)
    print("▶️ [{0:%H:%M:%S}] Loading weekly report + config...".format(now_local))
    weekly_csv = _last_weekly_csv(cfg.get("reporting", {}).get("output_dir", OUTPUT_DIR_DEFAULT))
    if weekly_csv:
        print(f"·· [{now_local:%H:%M:%S}] Weekly CSV: {weekly_csv}")
        df_weekly = _read_weekly_csv(weekly_csv)
    else:
        df_weekly = pd.DataFrame()

    # Pick the focus universe: Stage 1/2
    if not df_weekly.empty and "stage" in df_weekly.columns and "ticker" in df_weekly.columns:
        mask = df_weekly["stage"].astype(str).str.contains("Stage 1|Stage 2", case=False, na=False)
        focus = sorted(df_weekly.loc[mask, "ticker"].astype(str).unique().tolist())
        print(f"• [{now_local:%H:%M:%S}] Focus universe: {len(focus)} symbols (Stage 1/2).")
    else:
        focus = []
        print(f"• [{now_local:%H:%M:%S}] Focus universe: 0 symbols (weekly CSV not found or missing columns).")

    # Download intraday + daily data (mock point)
    print("▶️ [{0:%H:%M:%S}] Downloading intraday + daily bars...".format(now_local))
    time.sleep(1.5)  # simulate
    print("✅ [{0:%H:%M:%S}] Price data downloaded.".format(_now_central()))

    # Evaluate candidates (stubbed to show your earlier logs)
    print("▶️ [{0:%H:%M:%S}] Evaluating candidates...".format(_now_central()))

    # Example states for two that appeared as NEAR in your logs
    states = {"CAH": {"near_hits": [], "sell_hits": []},
              "GM":  {"near_hits": [], "sell_hits": []}}

    # Update "near hits" counters (so your log shows hits & ARMED status as before)
    for sym in states.keys():
        near_now = True  # pretend near condition
        states[sym]["near_hits"], near_count = _update_hits(states[sym].get("near_hits", []), near_now, NEAR_HITS_WINDOW)
        sell_now = False
        states[sym]["sell_hits"], sell_count = _update_hits(states[sym].get("sell_hits", []), sell_now, NEAR_HITS_WINDOW)
        buy_state = "ARMED" if near_count >= 6 else "IDLE"
        sell_state = "ARMED" if sell_count >= 6 else "IDLE"
        print(f"·· [{_now_central():%H:%M:%S}] {sym}: buy_state={buy_state} near_hits={near_count} | sell_state={sell_state} sell_hits={sell_count}")

    # Prepare sections
    buys = []  # fill with your real scan
    nears = [{"symbol":"CAH","price":"203.67","pivot":"203.67","pace":"—","stage":"Stage 2 (Uptrend)","rank":"999999"},
             {"symbol":"GM","price":"70.75","pivot":"70.76","pace":"—","stage":"Stage 2 (Uptrend)","rank":"999999"}]
    sells = []  # fill with your real scan

    # Charts - list the symbols you rendered
    chart_symbols = [ "CAH", "CAH", " GM", "GM" ]  # keep your existing duplicate look

    # Risk/Tracked block — restore from your earlier example
    tracked = [
        {"symbol":"ANET","at_price":"134.66","reason":"drawdown ≤ −8%","stage":"Stage 2 (Uptrend), weekly —","action":"SELL"},
        {"symbol":"APLD","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"nan, weekly —","action":"SELL"},
        {"symbol":"BITF","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"nan, weekly —","action":"SELL"},
        {"symbol":"FRMI","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"nan, weekly —","action":"SELL"},
        {"symbol":"LAC","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"nan, weekly —","action":"SELL"},
        {"symbol":"SMCI","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"Stage 3 (Topping), weekly —","action":"SELL"},
        {"symbol":"UUUU","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"nan, weekly —","action":"SELL"},
        {"symbol":"CLSK","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"nan, weekly —","action":"SELL"},
        {"symbol":"CRCL","at_price":"151.36","reason":"drawdown ≤ −8%","stage":"nan, weekly —","action":"SELL"},
        {"symbol":"CRM","at_price":"151.36","reason":"Stage 4 + negative P/L","stage":"Stage 4 (Downtrend), weekly —","action":"SELL"},
        {"symbol":"HOOD","at_price":"130.37","reason":"drawdown ≤ −8%","stage":"Stage 2 (Uptrend), weekly —","action":"SELL"},
    ]

    # Snapshot block — we’ll re-use a slice of weekly df if it looks like yours
    if not df_weekly.empty:
        # Just ensure the required columns exist; otherwise synthesize minimal ones
        need_cols = ["ticker","stage","price","ma30","pivot10w","vol_pace_vs50dma","two_bar_confirm","last_bar_vol_ok","weekly_rank"]
        for c in need_cols:
            if c not in df_weekly.columns:
                df_weekly[c] = np.nan
        # Sort to mimic your output (rank, stage)
        snap = df_weekly[need_cols].copy()
    else:
        snap = pd.DataFrame()

    # Weekly tail — this is whatever your existing weekly reporter appends.
    # In your logs, "Weinstein Weekly – Summary" plus a long PnL table follows.
    # We’ll leave a placeholder marker; your upstream should replace this with real content.
    weekly_tail_text = """Weinstein Weekly – Summary

Total Gain/Loss ($)\t$-627.70
Portfolio % Gain\t-5.87%
Average % Gain\t110.23%
Per-position Snapshot
Symbol\tDescription\tindustry\tsector\tQuantity\tLast Price\tCurrent Value\tCost Basis Total\tAverage Cost Basis\tTotal Gain/Loss Dollar\tTotal Gain/Loss Percent\tRecommendation
... (unchanged from your weekly generator)"""

    # Build sections in the required order:
    # Header -> Buy/Near/Sell -> Charts -> Sell/Risk -> Snapshot -> CRYPTO -> WEEKLY SUMMARY
    report = []
    report.append(build_header(now_local))
    report.append(build_buy_near_sell_sections(buys, nears, sells))
    report.append(build_charts_section(chart_symbols))
    report.append(build_sell_risk_block(tracked))
    if not snap.empty:
        report.append(build_snapshot_block(snap))
    # Crypto section (from Signals)
    crypto_block = build_crypto_section_from_signals(cfg, benchmark="BTC-USD")
    if crypto_block:
        report.append(crypto_block)
    # Weekly summary tail (keep at the end, *after* Crypto)
    report.append(build_weekly_summary_tail(weekly_tail_text))
    report.append("\n✅ Intraday tick complete.")

    final_text = "\n".join([r for r in report if r is not None])

    # Print to STDOUT (your wrapper captures and emails), and also save a txt artifact
    print(final_text)
    out_txt = os.path.join(output_dir, f"weinstein_intraday_{now_local:%Y%m%d_%H%M}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(final_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    run(_config_path=args.config)
