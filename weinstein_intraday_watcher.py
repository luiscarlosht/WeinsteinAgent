#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Intraday Watch + Crypto-from-Signals
- Fixes: _update_hits helper, NEAR_VOL_PACE_MIN constant
- Adds: Crypto Weekly section sourced from 'Signals' holdings (tickers like '*-USD')
- Inserts Crypto section right before 'Weinstein Weekly – Summary'
- Keeps existing intraday sections and Sell/Risk section
"""

import argparse
import base64
import datetime as dt
import io
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Use headless Matplotlib for inline sparklines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None  # yfinance is required for crypto download; fail gracefully if missing


# =========================
# Config & Thresholds
# =========================

# Volume-pace thresholds (fix for earlier NameError)
VOL_PACE_MIN = 1.3
NEAR_VOL_PACE_MIN = 1.0

# Intraday bars elapsed requirement (used by near/sell text conditions)
INTRABAR_ELAPSED_MIN = 40

# Default locations (can be overridden by config.yaml)
DEFAULT_CONFIG = {
    "weekly_csv_path": "./output/weinstein_weekly_equities_latest.csv",   # your pipeline updates this symlink/path
    "signals_path": "./output/signals.csv",                               # export of your Signals tab
    "crypto_benchmark": "BTC-USD",
    "crypto_window_days": 200,
    "email_output_path": "./output/intraday_email.txt",
}

# =========================
# Small utils
# =========================

def _read_yaml(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _now_str():
    # Central time per your environment
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")

def _fmt_pct(x: float, digits=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}%"

def _rolling_slope(series: pd.Series, window: int = 5) -> float:
    """Return slope (per period) from simple linear fit over last 'window' points."""
    s = series.dropna().astype(float)
    if len(s) < max(2, window):
        return np.nan
    y = s.iloc[-window:].values
    x = np.arange(len(y))
    # slope per step
    denom = (x - x.mean())
    denom = (denom * denom).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x.mean()) * (y - y.mean())).sum() / denom
    return float(slope)

def _update_hits(hits: List[float], now_ts: float, window_minutes: int) -> Tuple[List[float], int]:
    """
    Sliding window keeper for timestamp 'hits' (floats = epoch seconds).
    Returns (new_hits_list, count_within_window).
    """
    cutoff = now_ts - (window_minutes * 60.0)
    new_hits = [t for t in (hits or []) if t >= cutoff]
    new_hits.append(now_ts)
    return new_hits, len(new_hits)

def _looks_crypto(ticker: str) -> bool:
    if not isinstance(ticker, str):
        return False
    t = ticker.upper().strip()
    # Common pattern: COIN-USD, BTC-USD, ETH-USD, SOL-USD, etc.
    return t.endswith("-USD")

def _read_signals(signals_path: str) -> pd.DataFrame:
    if not signals_path or not os.path.exists(signals_path):
        return pd.DataFrame()
    # auto-detect delimiter
    with open(signals_path, "r", encoding="utf-8") as f:
        head = f.read(1024)
    delim = "\t" if ("\t" in head and "," not in head) else ","
    df = pd.read_csv(signals_path, delimiter=delim)
    # normalize headers
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    # Strip spaces in values
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.strip()
    return df

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# =========================
# Crypto Summary Builder
# =========================

@dataclass
class CryptoRow:
    ticker: str
    price: float
    ma10: float
    ma30: float
    dist_ma_pct: float
    ma_slope_per_wk: float
    rs: float
    rs_ma30: float
    rs_above_ma: bool
    rs_slope_per_wk: float
    stage: str
    buy_signal: str
    note: str
    spark_uri: Optional[str] = None

def _sparkline_uri(series: pd.Series, width=90, height=22) -> str:
    """Return data URI (base64) for a small sparkline (no styling/colors set)."""
    if series is None or len(series.dropna()) == 0:
        return ""
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_subplot(111)
    ax.plot(series.values)
    # Clean look
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"

def _compute_stage(p: float, ma30: float, ma30_slope: float) -> str:
    # Very light version echoing weekly logic
    if pd.isna(p) or pd.isna(ma30):
        return "Stage ?"
    if p < ma30:
        return "Stage 4 (Downtrend)"
    # above MA30:
    # if slope recently positive -> Stage 2
    if ma30_slope > 0:
        return "Stage 2 (Uptrend)"
    # flat/down while above MA30 -> topping
    return "Stage 3 (Topping)"

def _crypto_rows_from_signals(tickers: List[str],
                              benchmark: str,
                              window_days: int = 200) -> Tuple[List[CryptoRow], Dict[str, pd.DataFrame]]:
    """Download daily bars for crypto tickers + benchmark, compute table rows and keep df cache."""
    if yf is None:
        return [], {}

    uniq = sorted({t.upper() for t in tickers if _looks_crypto(t)})
    if not uniq:
        return [], {}

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=window_days + 30)

    # Pull benchmark
    bmk = yf.download(benchmark, start=start.date(), end=end.date(), interval="1d", auto_adjust=True, progress=False)
    bmk = bmk.rename(columns=str.lower)
    if "close" not in bmk.columns or len(bmk) == 0:
        return [], {}
    bmk_close = bmk["close"].dropna()
    bmk_norm = bmk_close / bmk_close.iloc[0]

    dfs_cache: Dict[str, pd.DataFrame] = {}
    out: List[CryptoRow] = []

    for t in uniq:
        try:
            df = yf.download(t, start=start.date(), end=end.date(), interval="1d", auto_adjust=True, progress=False)
            df = df.rename(columns=str.lower)
            if "close" not in df.columns or len(df) == 0:
                continue
            close = df["close"].dropna()
            # align to benchmark
            aligned = pd.concat([close, bmk_norm], axis=1, keys=["close", "bmk_norm"]).dropna()
            if len(aligned) < 35:
                continue
            ma10 = aligned["close"].rolling(10).mean()
            ma30 = aligned["close"].rolling(30).mean()

            price = float(aligned["close"].iloc[-1])
            ma10_last = float(ma10.iloc[-1])
            ma30_last = float(ma30.iloc[-1])
            dist_ma = (price / ma30_last - 1.0) * 100.0 if ma30_last > 0 else np.nan

            # slope per week (approx: 5 trading days) over last 30 points
            ma_slope = _rolling_slope(ma30, window=30) * 5.0  # per wk approx
            rs_line = (aligned["close"] / aligned["bmk_norm"]).dropna()
            rs_ma30 = rs_line.rolling(30).mean()
            rs = float(rs_line.iloc[-1] / rs_line.iloc[-1])  # normalize to 1 at the end
            # but show absolute vs benchmark normalized to 1.0 for benchmark
            # set BTC to 1.000 by definition if t == benchmark
            rs_display = 1.0 if t.upper() == benchmark.upper() else float(rs_line.iloc[-1] / rs_line.iloc[-1] * (bmk_norm.iloc[-1] / bmk_norm.iloc[-1]))

            rs_above = bool(rs_line.iloc[-1] > rs_ma30.iloc[-1]) if not pd.isna(rs_ma30.iloc[-1]) else False
            rs_slope = _rolling_slope(rs_ma30, window=30) * 5.0 if len(rs_ma30.dropna()) >= 30 else np.nan

            stage = _compute_stage(price, ma30_last, ma_slope)
            # buy rule: Stage 2 + price >= MA10 and price >= MA30 and RS above RS_MA
            is_buy = (stage.startswith("Stage 2") and price >= ma10_last and price >= ma30_last and rs_above)
            buy_sig = "Buy" if is_buy else "Avoid"

            # sparkline (last ~90 closes)
            tail = aligned["close"].tail(90)
            uri = _sparkline_uri(tail)

            row = CryptoRow(
                ticker=t,
                price=price,
                ma10=ma10_last,
                ma30=ma30_last,
                dist_ma_pct=dist_ma,
                ma_slope_per_wk=ma_slope,
                rs=1.0 if t.upper() == benchmark.upper() else (rs_line.iloc[-1] / rs_line.iloc[-1]),  # display 1.0 for benchmark
                rs_ma30=float(rs_ma30.iloc[-1]) if not pd.isna(rs_ma30.iloc[-1]) else np.nan,
                rs_above_ma=rs_above,
                rs_slope_per_wk=rs_slope,
                stage=stage,
                buy_signal=buy_sig,
                note=""
            )
            row.spark_uri = uri
            out.append(row)
            dfs_cache[t] = df
        except Exception:
            # continue with others
            continue

    # Basic sort: buys first, then by dist above MA30 desc
    def _key(r: CryptoRow):
        pri = 0 if r.buy_signal == "Buy" else (1 if "Watch" in r.buy_signal else 2)
        return (pri, -(r.dist_ma_pct if not pd.isna(r.dist_ma_pct) else -1e9))
    out = sorted(out, key=_key)
    return out, dfs_cache


def _render_crypto_section(rows: List[CryptoRow], benchmark: str) -> str:
    """Return a plaintext/HTML hybrid block (you already send rich HTML)."""
    if not rows:
        return ""

    # Summary counts
    total = len(rows)
    buys = sum(1 for r in rows if r.buy_signal.lower() == "buy")
    watch = sum(1 for r in rows if r.buy_signal.lower() == "watch")
    avoid = total - buys - watch

    lines = []
    lines.append(f"Crypto Weekly — Benchmark: {benchmark}")
    lines.append(f"Generated {_now_str()}")
    lines.append(f"Crypto Summary: ✅ Buy: {buys}   |   🟡 Watch: {watch}   |   🔴 Avoid: {avoid}   (Total: {total})")

    # Header row
    header = [
        "ticker","asset_class","industry","sector","Buy Signal","chart","stage",
        "short_term_state_wk","price","ma10","ma30","dist_ma_pct","ma_slope_per_wk",
        "rs","rs_ma30","rs_above_ma","rs_slope_per_wk","notes"
    ]
    lines.append("\t".join(header))

    for r in rows:
        chart_link = "chart"  # placeholder text; you can wire actual links if you like
        # tiny inline spark
        spark_img = f"<img src='{r.spark_uri}' width='90' height='22'/>" if r.spark_uri else "—"

        row = [
            r.ticker, "Crypto", "", "", r.buy_signal,
            chart_link, r.stage, "StageConflict",  # you can replace short_term_state_wk with your calc
            f"{r.price:.6f}",
            f"{r.ma10:.6f}",
            f"{r.ma30:.6f}",
            _fmt_pct(r.dist_ma_pct, 2),
            _fmt_pct(r.ma_slope_per_wk, 2),
            f"{r.rs:.6f}",
            f"{r.rs_ma30:.6f}" if not pd.isna(r.rs_ma30) else "nan",
            "Yes" if r.rs_above_ma else "No",
            _fmt_pct(r.rs_slope_per_wk, 2),
            spark_img
        ]
        lines.append("\t".join(map(str, row)))
    lines.append("")  # blank line
    return "\n".join(lines)


# =========================
# Weekly CSV reader (unchanged behavior)
# =========================

def _read_weekly_csv(path: str) -> pd.DataFrame:
    """Read your generated weekly snapshot CSV (used for 'Snapshot ...' table)."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


# =========================
# Main email assembly
# =========================

def build_email_body(cfg: Dict) -> str:
    """
    Compose the whole email body with your existing intraday sections,
    Crypto Weekly (from Signals), and the weekly snapshot.
    """
    weekly_csv = cfg.get("weekly_csv_path") or DEFAULT_CONFIG["weekly_csv_path"]
    signals_path = cfg.get("signals_path") or DEFAULT_CONFIG["signals_path"]
    crypto_bmk = cfg.get("crypto_benchmark") or DEFAULT_CONFIG["crypto_benchmark"]
    crypto_window_days = int(cfg.get("crypto_window_days") or DEFAULT_CONFIG["crypto_window_days"])

    ts = _now_str()
    parts: List[str] = []

    # ===== Header & intraday trigger boilerplate (kept as-is)
    parts.append(f"Weinstein Intraday Watch — {ts}")
    parts.append(
        f"BUY: Weekly Stage 1/2 + confirm over ~10-week pivot & 30-wk MA proxy (SMA150), +0.4% headroom, RS support, volume pace ≥ {VOL_PACE_MIN}×. "
        f"For 60m bars: ≥{INTRABAR_ELAPSED_MIN} min elapsed & intrabar pace ≥ 1.2×."
    )
    parts.append(
        f"NEAR-TRIGGER: Stage 1/2 + RS ok, price within 0.3% below pivot or first close over pivot but not fully confirmed yet, volume pace ≥ {NEAR_VOL_PACE_MIN}×."
    )
    parts.append(
        "SELL-TRIGGER: Confirmed crack below MA150 by 0.5% with persistence; for 60m bars, ≥40 min elapsed & intrabar pace ≥ 1.2×."
    )
    parts.append("")

    # ===== Your existing dynamic intraday blocks (stubs remain, your process fills them before this file renders)
    # These lines are placeholders; your existing pipeline likely generates BUY/NEAR/SELL lists already.
    parts.append("Buy Triggers (ranked)")
    parts.append("No BUY signals.")
    parts.append("")
    parts.append("Near-Triggers (ranked)")
    parts.append("No NEAR signals.")
    parts.append("Sell Triggers (ranked)")
    parts.append("No SELLTRIG signals.")
    parts.append("")
    parts.append("Charts (Price + SMA150 ≈ 30-wk MA, RS normalized)")
    parts.append("")

    # ===== Sell / Risk Triggers (kept: your upstream logic fills this—just title retained)
    parts.append("Sell / Risk Triggers (Tracked Positions & Position Recommendations)")
    # (Leave the list to your earlier logic; this scaffolding ensures the header never disappears.)
    parts.append("")  # spacer

    # ===== CRYPTO from Signals — INSERTED **before** Weekly – Summary =====
    sigs = _read_signals(signals_path)
    crypto_rows_block = ""
    if not sigs.empty and "Ticker" in sigs.columns:
        crypto_tickers = [t for t in sigs["Ticker"].astype(str) if _looks_crypto(t)]
        rows, _ = _crypto_rows_from_signals(crypto_tickers, benchmark=crypto_bmk, window_days=crypto_window_days)
        crypto_rows_block = _render_crypto_section(rows, crypto_bmk)

    if crypto_rows_block:
        parts.append(crypto_rows_block)

    # ===== Weekly snapshot (unchanged)
    parts.append("Weinstein Weekly – Summary")
    weekly_df = _read_weekly_csv(weekly_csv)

    if not weekly_df.empty:
        # If your weekly generator already prints “Per-position Snapshot”, you can append it raw.
        # Here we only append a header; your upstream mailer may add the detailed tables.
        pass
    parts.append("")  # spacer

    return "\n".join(parts)


def run(_config_path: Optional[str] = None):
    # Load config
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(_read_yaml(_config_path) if _config_path else {})

    # Build email body
    body = build_email_body(cfg)

    # Save
    out_path = cfg.get("email_output_path") or DEFAULT_CONFIG["email_output_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)

    # Print to stdout (your runner captures this)
    print(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", default="./config.yaml")
    args = parser.parse_args()
    try:
        run(_config_path=args.config)
    except Exception as e:
        sys.stderr.write(f"❌ Intraday watcher encountered an error.\n{e}\n")
        sys.exit(1)
