#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Crypto Watcher — 24/7 version of the intraday watcher for cryptocurrencies

What this does
- Scans a crypto universe (from config or sensible defaults) on 60m bars (24/7).
- Applies the same Weinstein-style BUY / NEAR / SELL logic you use for equities,
  including 10w pivot, SMA150 ~ 30w proxy, volume pace vs 50-day "day" (24h) average.
- Emits ranked BUY / NEAR / SELL lists with mini charts (price + SMA150 + RS vs BTC).
- Shows a Snapshot table (ordered by weekly rank & stage) for crypto.
- Optionally shows a Per-position Snapshot (worst → best) for *held* crypto from:
    * Google Sheet tab "CryptoHoldings" (Symbol, Quantity, Cost Basis)
    * OR local CSV ./output/crypto_positions.csv in a similar shape
- Saves the HTML in ./output and (unless --dry-run) emails it using weinstein_mailer.send_email

Config
- Uses the same config.yaml keys where relevant:
    app:
      benchmark: "SPY"              # still used for shared helpers, but crypto RS uses BTC-USD
    sheets:
      url: "https://docs.google.com/....."    # Google Sheet (optional)
    google:
      service_account_json: "./secrets/gsa.json"
    crypto:
      tickers: ["BTC-USD","ETH-USD","SOL-USD","ADA-USD","AVAX-USD","DOGE-USD","XRP-USD"]
      positions_csv: "./output/crypto_positions.csv"   # optional local holdings
- If crypto.tickers is absent, a default top list is used.

CLI examples
  python3 weinstein_crypto_watcher.py --config ./config.yaml --dry-run
  python3 weinstein_crypto_watcher.py --config ./config.yaml --only BTC-USD,ETH-USD \
      --log-csv ./output/crypto_intraday_debug.csv --log-json ./output/crypto_intraday_debug.json

Notes
- State is kept at ./state/intraday_crypto_triggers.json
- Interval is 60m by default; you can change INTRADAY_INTERVAL to "30m" if preferred.
"""

import os, io, json, math, base64, yaml, argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weinstein_mailer import send_email

# ---------------- Tunables (mirrored where it makes sense) ----------------
WEEKLY_OUTPUT_DIR = "./output"
# We won't rely on weekly equities CSV for crypto universe; we compute stage proxy from daily.
BENCHMARK_DEFAULT = "SPY"        # kept for shared code; RS for crypto uses BTC-USD below
CRYPTO_RS_BASE    = "BTC-USD"    # RS denominator for crypto charts

INTRADAY_INTERVAL = "60m"        # '60m' or '30m'
LOOKBACK_DAYS = 120              # look back farther since crypto is 24/7
PIVOT_LOOKBACK_WEEKS = 10
VOL_PACE_MIN = 1.20              # crypto tends to have spiky pace; slightly easier than equities
BUY_DIST_ABOVE_MA_MIN = 0.00

CONFIRM_BARS = 2
MIN_BREAKOUT_PCT = 0.004         # ~0.4%
REQUIRE_RISING_BAR_VOL = True
INTRADAY_AVG_VOL_WINDOW = 30
INTRADAY_LASTBAR_AVG_MULT = 1.10

NEAR_BELOW_PIVOT_PCT = 0.004     # a tad wider band for crypto
NEAR_VOL_PACE_MIN = 0.90         # allow slightly below-par pace to flag NEAR

HARD_STOP_PCT = 0.12             # wider hard stop guidance if you track positions
TRAIL_ATR_MULT = 2.5

STATE_FILE = "./state/positions.json"  # unused for crypto positions; kept for parity if you mix
CRYPTO_STATE_FILE = "./state/intraday_crypto_triggers.json"
CHART_DIR = "./output/charts_crypto"
MAX_CHARTS_PER_EMAIL = 12

PRICE_WINDOW_DAYS = 260
SMA_DAYS = 150

VERBOSE = True

# Sheets (optional)
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

TAB_CRYPTO_HOLDINGS = "CryptoHoldings"  # expected columns: Symbol, Quantity, CostBasis (optional)

# ---------- Small helpers ----------
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg, *, level="info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {"info":"•", "ok":"✅", "step":"▶️", "warn":"⚠️", "err":"❌", "debug":"··"}.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)

def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _is_crypto(sym: str) -> bool:
    return (sym or "").upper().endswith("-USD")

# ---------------- Config ----------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}
    crypto = cfg.get("crypto", {}) or {}
    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")
    crypto_tickers = crypto.get("tickers") or [
        # sensible default set (can be overridden in config.yaml)
        "BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD",
        "AVAX-USD","DOGE-USD","LINK-USD","TON-USD","TRX-USD",
        "LTC-USD","BCH-USD","NEAR-USD","APT-USD","ARB-USD"
    ]
    crypto_positions_csv = crypto.get("positions_csv") or "./output/crypto_positions.csv"
    return cfg, benchmark, sheet_url, svc_file, crypto_tickers, crypto_positions_csv

# ---------------- Data ----------------
def get_intraday(tickers: List[str]):
    uniq = list(dict.fromkeys(tickers))
    intraday = yf.download(
        uniq, period=f"{LOOKBACK_DAYS}d", interval=INTRADAY_INTERVAL,
        auto_adjust=True, ignore_tz=True, progress=False
    )
    daily = yf.download(
        uniq, period="36mo", interval="1d",
        auto_adjust=True, ignore_tz=True, progress=False
    )
    return intraday, daily

def last_weekly_pivot_high(ticker, daily_df, weeks=PIVOT_LOOKBACK_WEEKS):
    bars = weeks * 7  # crypto trades 7 days/wk
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)]
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"]
    highs = highs.dropna().tail(bars)
    return float(highs.max()) if len(highs) else np.nan

def get_last_n_intraday_closes(intraday_df, ticker, n=2):
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            s = intraday_df[("Close", ticker)].dropna()
        except KeyError:
            return []
    else:
        s = intraday_df["Close"].dropna()
    return list(map(float, s.tail(n).values))

def get_last_n_intraday_volumes(intraday_df, ticker, n=2):
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            v = intraday_df[("Volume", ticker)].dropna()
        except KeyError:
            return []
    else:
        v = intraday_df["Volume"].dropna()
    return list(map(float, v.tail(n).values))

def get_intraday_avg_volume(intraday_df, ticker, window=INTRADAY_AVG_VOL_WINDOW):
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            v = intraday_df[("Volume", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        v = intraday_df["Volume"].dropna()
    if len(v) < window:
        return np.nan
    return float(v.tail(window).mean())

def intrabar_volume_pace(intraday_df, ticker, avg_window=INTRADAY_AVG_VOL_WINDOW, bar_minutes=60):
    try:
        if isinstance(intraday_df.columns, pd.MultiIndex):
            v = intraday_df[("Volume", ticker)].dropna()
        else:
            v = intraday_df["Volume"].dropna()
    except Exception:
        return np.nan
    if len(v) < max(avg_window, 2):
        return np.nan
    last_bar_vol = float(v.iloc[-1])
    avg_bar_vol = float(v.tail(avg_window).mean())
    # 24/7 bar timing, we can't reliably infer elapsed minutes from utc start (yfinance index is period end).
    # We'll approximate full-bar pace simply as ratio of current bar vs avg (this yields <=1 until bar closes).
    # To avoid under-flagging, we tolerate lower pace for NEAR. For confirm, we still require pace >= 1.0 typically.
    est_full = last_bar_vol  # already the bar's current volume
    return float(_safe_div(est_full, avg_bar_vol))

def volume_pace_today_vs_50dma(ticker, daily_df):
    """Projected 24h volume vs 50-day avg for crypto."""
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            v = daily_df[("Volume", ticker)].copy()
        except KeyError:
            return np.nan
    else:
        v = daily_df["Volume"].copy()
    if v.empty:
        return np.nan

    v50 = v.rolling(50).mean().iloc[-2] if len(v) > 50 else np.nan
    today_vol = v.iloc[-1]
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = max(0.0, (now - day_start).total_seconds())
    fraction = min(1.0, max(0.05, elapsed / (24*3600.0)))

    est_full = today_vol / fraction if fraction > 0 else today_vol
    return float(_safe_div(est_full, v50)) if pd.notna(v50) and v50 > 0 else np.nan

# ---------------- Stage proxy for crypto ----------------
def compute_stage_and_rs(daily_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Compute a minimal 'stage' and rs_above_ma proxy for crypto so we can use the same gating.
    Stage 2 if price > SMA150 and SMA150 slope up (last few values increasing); Stage 1 otherwise.
    rs_above_ma uses RS vs BTC-USD > its SMA30 as a proxy.
    """
    rows = []
    for t in tickers:
        try:
            if isinstance(daily_df.columns, pd.MultiIndex):
                close = daily_df[("Close", t)].dropna()
            else:
                close = daily_df["Close"].dropna()
        except Exception:
            continue
        if len(close) < 160:
            continue
        sma150 = close.rolling(SMA_DAYS).mean()
        # SMA slope: last 5 values increasing?
        slope_up = False
        if len(sma150.dropna()) >= 10:
            s = sma150.dropna().tail(5).values
            slope_up = bool(np.all(np.diff(s) > 0))
        stage = "Stage 2 (Uptrend)" if (close.iloc[-1] > sma150.iloc[-1] if pd.notna(sma150.iloc[-1]) else False) and slope_up else "Stage 1 (Basing)"
        # RS vs BTC-USD
        try:
            base = daily_df[("Close", CRYPTO_RS_BASE)].dropna()
        except Exception:
            base = None
        rs_above_ma = False
        if base is not None and len(base) > 30:
            idx = close.index.intersection(base.index)
            if len(idx) > 40:
                rs = (close.loc[idx] / base.loc[idx])
                rs_ma = rs.rolling(30).mean()
                if pd.notna(rs.iloc[-1]) and pd.notna(rs_ma.iloc[-1]):
                    rs_above_ma = bool(rs.iloc[-1] > rs_ma.iloc[-1])
        rows.append({"ticker": t, "stage": stage, "ma30": float(sma150.iloc[-1]) if pd.notna(sma150.iloc[-1]) else np.nan, "rs_above_ma": rs_above_ma})
    return pd.DataFrame(rows)

# ---------------- Charts ----------------
def make_tiny_chart_png(ticker, benchmark, daily_df):
    os.makedirs(CHART_DIR, exist_ok=True)
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            close_t = daily_df[("Close", ticker)].dropna()
            close_b = daily_df[("Close", benchmark)].dropna()
        except KeyError:
            return None, None
    else:
        return None, None
    close_t = close_t.tail(PRICE_WINDOW_DAYS)
    close_b = close_b.reindex_like(close_t).dropna()
    idx = close_t.index.intersection(close_b.index)
    close_t, close_b = close_t.loc[idx], close_b.loc[idx]
    if len(close_t) < 50 or len(close_b) < 50:
        return None, None
    sma = close_t.rolling(SMA_DAYS).mean()
    rs = (close_t / close_b); rs_norm = rs / rs.iloc[0]

    fig, ax1 = plt.subplots(figsize=(5.0, 2.4), dpi=150)
    ax1.plot(close_t.index, close_t.values, label=f"{ticker}")
    ax1.plot(sma.index, sma.values, label=f"SMA{SMA_DAYS}", linewidth=1.2)
    ax1.set_ylabel("Price")
    ax1.tick_params(axis='x', labelsize=8); ax1.tick_params(axis='y', labelsize=8)
    ax2 = ax1.twinx()
    ax2.plot(rs_norm.index, rs_norm.values, linestyle="--", alpha=0.7, label=f"RS/{benchmark} (norm)")
    ax2.set_ylabel("RS (norm)")
    ax2.tick_params(axis='y', labelsize=8)
    ax1.set_title(f"{ticker} — Price, SMA{SMA_DAYS}, RS/{benchmark}", fontsize=9)
    ax1.grid(alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left", frameon=False)
    chart_path = os.path.join(CHART_DIR, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.tight_layout(pad=0.8); fig.savefig(chart_path, bbox_inches="tight"); plt.close(fig)
    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# ---------------- Sorting helpers ----------------
def stage_order(stage: str) -> int:
    if isinstance(stage, str):
        if stage.startswith("Stage 2"): return 0
        if stage.startswith("Stage 1"): return 1
    return 9

def buy_sort_key(item):
    st = stage_order(item.get("stage", ""))
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    px = item.get("price", np.nan); pivot = item.get("pivot", np.nan); ma = item.get("ma30", np.nan)
    ratio_pivot = (px / pivot) if (pd.notna(px) and pd.notna(pivot) and pivot != 0) else -1e9
    ratio_ma = (px / ma) if (pd.notna(px) and pd.notna(ma) and ma != 0) else -1e9
    return (st, -pace, -ratio_pivot, -ratio_ma)

def near_sort_key(item):
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan); pivot = item.get("pivot", np.nan)
    dist = abs(px - pivot) if (pd.notna(px) and pd.notna(pivot)) else 1e9
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    return (st, dist, -pace)

def sell_sort_key(item):
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan); ma = item.get("ma30", np.nan)
    dist_below = (ma - px) if (pd.notna(px) and pd.notna(ma)) else -1e9
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    return (st, -dist_below, -pace)

# ---------------- Logic helpers ----------------
SELL_NEAR_ABOVE_MA_PCT = 0.006
SELL_BREAK_PCT = 0.006

def _price_below_ma(px, ma): 
    return pd.notna(px) and pd.notna(ma) and px <= ma * (1.0 - SELL_BREAK_PCT)

def _near_sell_zone(px, ma):
    if pd.isna(px) or pd.isna(ma): return False
    return (px >= ma * (1.0 - SELL_BREAK_PCT)) and (px <= ma * (1.0 + SELL_NEAR_ABOVE_MA_PCT))

# ---------------- Holdings helpers (Sheet OR CSV) ----------------
def _coerce_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).replace(",", "").replace("$", "").strip()
    if s.endswith("%"): 
        try: return float(s[:-1])
        except: return np.nan
    try:
        return float(s)
    except:
        return np.nan

def _load_crypto_positions_from_sheet(sheet_url: str, svc_file: str) -> pd.DataFrame | None:
    if not gspread or not Credentials or not sheet_url or not svc_file or not os.path.exists(svc_file):
        return None
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_file(svc_file, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(TAB_CRYPTO_HOLDINGS)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        # normalize columns
        ren = {"symbol":"Symbol","SYM":"Symbol","qty":"Quantity","quantity":"Quantity","cost":"CostBasis","cost basis":"CostBasis"}
        df = df.rename(columns={**{k:v for k,v in ren.items()}, **{c:c for c in df.columns}})
        if "Symbol" not in df.columns: return None
        if "Quantity" not in df.columns: df["Quantity"] = np.nan
        if "CostBasis" not in df.columns: df["CostBasis"] = np.nan
        df["Quantity"] = df["Quantity"].apply(_coerce_float)
        df["CostBasis"] = df["CostBasis"].apply(_coerce_float)
        return df
    except Exception as e:
        log(f"Sheet load failure for crypto holdings: {e}", level="warn")
        return None

def _load_crypto_positions_from_csv(csv_path: str) -> pd.DataFrame | None:
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df is None or df.empty: return None
        ren = {"symbol":"Symbol","SYM":"Symbol","qty":"Quantity","quantity":"Quantity","cost":"CostBasis","cost basis":"CostBasis"}
        df = df.rename(columns=ren)
        if "Symbol" not in df.columns: return None
        if "Quantity" not in df.columns: df["Quantity"] = np.nan
        if "CostBasis" not in df.columns: df["CostBasis"] = np.nan
        df["Quantity"] = df["Quantity"].apply(_coerce_float)
        df["CostBasis"] = df["CostBasis"].apply(_coerce_float)
        return df
    except Exception:
        return None

def _format_money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"

def _format_pct(x):
    return f"{x:.2f}%" if pd.notna(x) else "—"

def _per_position_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No crypto positions found.</p>"
    # sort worst → best
    df = df.sort_values(by="Total Gain/Loss Percent", ascending=True).reset_index(drop=True)
    # color classes based on percent
    css = """
    <style>
      .holdtbl { border-collapse: collapse; width: 100%; }
      .holdtbl th, .holdtbl td { padding: 6px 8px; border-bottom: 1px solid #eee; font-size: 13px; }
      .holdtbl th { text-align:left; background:#fafafa; }
      .pl-pos { color:#0b6b2e; font-weight:600; }
      .pl-neg { color:#a30a0a; font-weight:600; }
      .pl-neu { color:#444; }
      .chip { display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid #eee; background:#fafafa; }
    </style>
    """
    # build rows
    headers = ["Symbol","Quantity","Last Price","Current Value","Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent","Recommendation"]
    for h in headers:
        if h not in df.columns: df[h] = np.nan

    trs = []
    for _, r in df.iterrows():
        pct = r["Total Gain/Loss Percent"]
        cls = "pl-pos" if (pd.notna(pct) and pct > 0) else ("pl-neg" if (pd.notna(pct) and pct < 0) else "pl-neu")
        tds = [
            f"<td>{r.get('Symbol','')}</td>",
            f"<td>{r.get('Quantity','')}</td>",
            f"<td>{_format_money(r.get('Last Price',np.nan))}</td>",
            f"<td>{_format_money(r.get('Current Value',np.nan))}</td>",
            f"<td>{_format_money(r.get('Cost Basis Total',np.nan))}</td>",
            f"<td>{_format_money(r.get('Average Cost Basis',np.nan))}</td>",
            f"<td class='{cls}'>{_format_money(r.get('Total Gain/Loss Dollar',np.nan))}</td>",
            f"<td class='{cls}'>{_format_pct(pct)}</td>",
            f"<td><span class='chip'>{r.get('Recommendation','HOLD')}</span></td>",
        ]
        trs.append("<tr>" + "".join(tds) + "</tr>")
    table = f"""
    {css}
    <table class="holdtbl">
      <thead><tr>{"".join([f"<th>{h}</th>" for h in headers])}</tr></thead>
      <tbody>
        {''.join(trs)}
      </tbody>
    </table>
    """
    return table

# ---------------- Crypto holdings merge/metrics ----------------
def _merge_positions_live(positions: pd.DataFrame, last_price_map: Dict[str, float]) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame()
    out = positions.copy()
    out["Last Price"] = out["Symbol"].apply(lambda s: float(last_price_map.get(str(s).upper(), np.nan)))
    out["Current Value"] = out["Last Price"] * out["Quantity"]
    # If you provided per-unit CostBasis, compute totals; else treat CostBasis as total if Quantity present.
    # We'll try to infer: if CostBasis is small relative to quantity, assume per-unit.
    def _cost_total(row):
        q = row.get("Quantity", np.nan)
        c = row.get("CostBasis", np.nan)
        if pd.notna(q) and pd.notna(c):
            # heuristics
            if c < 1.0 and row.get("Symbol","").upper() not in ("WBTC-USD",):  # many coins had small unit costs years ago; keep simple
                return q * c
            # If positions include per-unit costs like 20000.0 and q is small, still fine as q*c
            return q * c
        return np.nan
    out["Cost Basis Total"] = out.apply(_cost_total, axis=1)
    out["Average Cost Basis"] = out.apply(lambda r: (r["CostBasis"] if pd.notna(r.get("CostBasis", np.nan)) else np.nan), axis=1)
    out["Total Gain/Loss Dollar"] = out["Current Value"] - out["Cost Basis Total"]
    out["Total Gain/Loss Percent"] = (out["Total Gain/Loss Dollar"] / out["Cost Basis Total"] * 100.0)
    # Simple recommendation similar to equities:
    def rec(row):
        pct = row.get("Total Gain/Loss Percent", np.nan)
        if pd.notna(pct) and pct <= -12.0: return "SELL"
        if pd.notna(pct) and pct >= 0: return "HOLD (Strong)"
        return "HOLD"
    out["Recommendation"] = out.apply(rec, axis=1)
    return out

# ---------------- Main scan ----------------
def run(_config_path="./config.yaml", *, only_tickers=None, log_csv=None, log_json=None, dry_run=False):
    log("Crypto watcher starting with config: {0}".format(_config_path), level="step")
    cfg, benchmark, sheet_url, service_account_file, crypto_tickers, crypto_positions_csv = load_config(_config_path)

    # Restrict if --only present
    if only_tickers:
        filt = set([t.strip().upper() for t in only_tickers])
        crypto_tickers = [t for t in crypto_tickers if t.upper() in filt]
    # Ensure RS base included
    needs = sorted(set(crypto_tickers + [CRYPTO_RS_BASE]))

    log(f"Universe: {len(crypto_tickers)} crypto symbols.", level="info")
    log("Downloading intraday + daily bars...", level="step")
    intraday, daily = get_intraday(needs)
    log("Price data downloaded.", level="ok")

    # Build minimal weekly-like dataset (stage + rs_ok)
    weekly_like = compute_stage_and_rs(daily, crypto_tickers)
    if weekly_like is None or weekly_like.empty:
        raise RuntimeError("Unable to compute stage/RS for crypto universe (insufficient data).")

    # Price map (now)
    if isinstance(intraday.columns, pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    def px_now(t):
        if hasattr(last_closes, "index") and (t in last_closes.index):
            return float(last_closes.get(t, np.nan))
        vals = getattr(last_closes, "values", [])
        return float(vals[-1]) if len(vals) else np.nan

    # Trigger state
    trig_path = CRYPTO_STATE_FILE
    os.makedirs(os.path.dirname(trig_path), exist_ok=True)
    if os.path.exists(trig_path):
        with open(trig_path, "r") as f:
            trigger_state = json.load(f)
    else:
        trigger_state = {}

    # Working collections
    buy_signals, near_signals, sell_signals = [], [], []
    sell_triggers, info_rows, debug_rows, chart_imgs = [], [], [], []

    # Iterate universe
    for _, row in weekly_like.iterrows():
        t = row["ticker"]
        px = px_now(t)
        if np.isnan(px):
            continue
        stage = str(row["stage"])
        ma30 = float(row.get("ma30", np.nan))     # SMA150 proxy
        rs_ok = bool(row.get("rs_above_ma", False))
        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace_full = volume_pace_today_vs_50dma(t, daily)
        closes_n = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
        pace_intra = intrabar_volume_pace(intraday, t, bar_minutes=60)

        # BUY confirm
        confirm = False; vol_ok = True; price_ok = False
        if pd.notna(ma30) and pd.notna(pivot) and closes_n:
            def _price_ok(c):
                return (c >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and (c >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN))
            need = max(CONFIRM_BARS, 2)
            price_ok = all(_price_ok(c) for c in closes_n[-need:])
            confirm = price_ok
            if REQUIRE_RISING_BAR_VOL:
                vols2 = get_last_n_intraday_volumes(intraday, t, n=2)
                vavg = get_intraday_avg_volume(intraday, t, window=INTRADAY_AVG_VOL_WINDOW)
                if len(vols2) >= 2 and pd.notna(vavg) and vavg > 0:
                    vol_ok = (vols2[-1] >= INTRADAY_LASTBAR_AVG_MULT * vavg) and (pd.isna(pace_intra) or pace_intra >= 1.0)
                else:
                    vol_ok = False

        # NEAR window
        near_now = False
        if stage in ("Stage 1 (Basing)","Stage 2 (Uptrend)") and rs_ok and pd.notna(pivot) and pd.notna(ma30) and pd.notna(px):
            above_ma = px >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
            if above_ma:
                if (px >= pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)) and (px < pivot * (1.0 + MIN_BREAKOUT_PCT)):
                    near_now = True
                elif (px >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not confirm:
                    near_now = True

        # SELL near/confirm
        sell_near_now = False; sell_confirm = False; sell_vol_ok = True
        sell_price_ok = False
        if pd.notna(ma30) and pd.notna(px):
            sell_near_now = _near_sell_zone(px, ma30)
            closes_n2 = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
            if closes_n2:
                sell_price_ok = all((c <= ma30 * (1.0 - SELL_BREAK_PCT)) for c in closes_n2[-CONFIRM_BARS:])
                sell_confirm = sell_price_ok

        # State machine (simplified for crypto; counts not time-based because 24/7)
        st = trigger_state.get(t, {"state":"IDLE","near_hits":0,"sell_state":"IDLE","sell_hits":0})
        # BUY
        if near_now: st["near_hits"] = int(st.get("near_hits",0)) + 1
        else: st["near_hits"] = 0
        state_now = st.get("state","IDLE")
        if state_now == "IDLE" and near_now: state_now = "NEAR"
        elif state_now in ("IDLE","NEAR") and st["near_hits"] >= 2: state_now = "ARMED"
        elif state_now == "ARMED" and confirm and vol_ok and (pd.isna(pace_full) or pace_full >= VOL_PACE_MIN):
            state_now = "TRIGGERED"
        elif state_now == "TRIGGERED":
            state_now = "COOLDOWN"
        elif not near_now and not confirm:
            state_now = "IDLE"
        st["state"] = state_now

        # SELL
        if sell_near_now: st["sell_hits"] = int(st.get("sell_hits",0)) + 1
        else: st["sell_hits"] = 0
        sell_state = st.get("sell_state","IDLE")
        if sell_state == "IDLE" and sell_near_now: sell_state = "NEAR"
        elif sell_state in ("IDLE","NEAR") and st["sell_hits"] >= 2: sell_state = "ARMED"
        elif sell_state == "ARMED" and sell_confirm:
            sell_state = "TRIGGERED"
        elif sell_state == "TRIGGERED":
            sell_state = "COOLDOWN"
        elif not sell_near_now and not sell_confirm:
            sell_state = "IDLE"
        st["sell_state"] = sell_state

        trigger_state[t] = st

        # Emit
        if st["state"] == "TRIGGERED" and (pd.isna(pace_full) or pace_full >= VOL_PACE_MIN):
            buy_signals.append({
                "ticker": t, "price": px, "pivot": float(pivot) if pd.notna(pivot) else np.nan,
                "pace": None if pd.isna(pace_full) else float(pace_full),
                "stage": stage, "ma30": ma30
            })
            # cool down immediately
            trigger_state[t]["state"] = "COOLDOWN"
        elif st["state"] in ("NEAR","ARMED"):
            if (pd.isna(pace_full) or pace_full >= NEAR_VOL_PACE_MIN):
                near_signals.append({
                    "ticker": t, "price": px, "pivot": float(pivot) if pd.notna(pivot) else np.nan,
                    "pace": None if pd.isna(pace_full) else float(pace_full),
                    "stage": stage, "ma30": ma30, "reason": "near/armed"
                })

        if st["sell_state"] == "TRIGGERED":
            sell_triggers.append({
                "ticker": t, "price": px, "ma30": ma30, "stage": stage,
                "pace": None if pd.isna(pace_full) else float(pace_full)
            })
            trigger_state[t]["sell_state"] = "COOLDOWN"

        info_rows.append({
            "ticker": t, "stage": stage, "price": px, "ma30": ma30, "pivot10w": pivot,
            "vol_pace_vs50dma": None if pd.isna(pace_full) else round(float(pace_full), 2),
            "two_bar_confirm": bool(confirm), "last_bar_vol_ok": bool(vol_ok),
            "buy_state": st["state"], "sell_state": st["sell_state"],
        })
        debug_rows.append({
            "ticker": t, "price": px, "ma30": ma30, "pivot": pivot,
            "pace_full": None if pd.isna(pace_full) else float(pace_full),
            "pace_intrabar": None if pd.isna(pace_intra) else float(pace_intra),
            "confirm": bool(confirm), "near_now": bool(near_now),
            "sell_near_now": bool(sell_near_now), "sell_confirm": bool(sell_confirm),
            "state": st["state"], "sell_state": st["sell_state"],
        })

    log(f"Scan done. Raw counts → BUY:{len(buy_signals)} NEAR:{len(near_signals)} SELLTRIG:{len(sell_triggers)}", level="info")

    # ------- Charts -------
    buy_signals.sort(key=buy_sort_key); near_signals.sort(key=near_sort_key); sell_triggers.sort(key=sell_sort_key)
    charts_added = 0; chart_imgs = []
    for item in buy_signals + near_signals:
        if charts_added >= MAX_CHARTS_PER_EMAIL: break
        t = item["ticker"]
        path, data_uri = make_tiny_chart_png(t, CRYPTO_RS_BASE, daily)
        if data_uri:
            chart_imgs.append((t, data_uri)); charts_added += 1

    # ------- Snapshot table (ordered by stage then ticker) -------
    info_df = pd.DataFrame(info_rows)
    if not info_df.empty:
        info_df["stage_rank"] = info_df["stage"].apply(stage_order)
        info_df = info_df.sort_values(["stage_rank","ticker"]).drop(columns=["stage_rank"])

    # ------- Holdings (Sheet or CSV) -------
    # Build current price map
    price_map = {}
    for t in crypto_tickers:
        try:
            price_map[t] = float(px_now(t))
        except Exception:
            price_map[t] = np.nan

    holdings_df = _load_crypto_positions_from_sheet(sheet_url, service_account_file)
    if holdings_df is None:
        holdings_df = _load_crypto_positions_from_csv(crypto_positions_csv)

    holdings_html = ""
    if holdings_df is not None and not holdings_df.empty:
        merged = _merge_positions_live(holdings_df, price_map)
        holdings_html = "<h4>Per-position Snapshot (worst → best)</h4>" + _per_position_html(merged)

    # ------- Build Email -------
    charts_html = ""
    if chart_imgs:
        charts_html = "<h4>Charts (Price + SMA150 ≈ 30-wk MA, RS normalized vs BTC-USD)</h4>"
        for t, data_uri in chart_imgs:
            charts_html += f"""
            <div style="display:inline-block;margin:6px 8px 10px 0;vertical-align:top;text-align:center;">
              <img src="{data_uri}" alt="{t}" style="border:1px solid #eee;border-radius:6px;max-width:320px;">
              <div style="font-size:12px;color:#555;margin-top:3px;">{t}</div>
            </div>
            """

    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} signals.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            pace_val = it.get("pace", None)
            pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.6g} (↓ SMA150 {ma_str}, pace {pace_str}, {it.get('stage','')})</li>")
            else:
                piv = it.get("pivot", np.nan)
                piv_str = f"{piv:.2f}" if pd.notna(piv) else "—"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.6g} (pivot {piv_str}, pace {pace_str}, {it['stage']})</li>")
        return "<ol>" + "\n".join(lis) + "</ol>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <h3>Weinstein Crypto Watch — {now}</h3>
    <p><i>
      BUY: Stage 1/2 + confirm over ~10-week pivot & SMA150, +{MIN_BREAKOUT_PCT*100:.1f}% headroom, RS vs BTC support,
      and volume pace ≥ {VOL_PACE_MIN}× (24h pacing).<br>
      NEAR-TRIGGER: Stage 1/2 + RS ok, price within {NEAR_BELOW_PIVOT_PCT*100:.1f}% below pivot or first close over pivot but not fully confirmed yet,
      pace ≥ {NEAR_VOL_PACE_MIN}×.<br>
      SELL-TRIGGER: Confirmed crack below SMA150 by {SELL_BREAK_PCT*100:.1f}% with persistence.
    </i></p>
    """

    html += f"""
    <h4>Buy Triggers (ranked)</h4>
    {bullets(buy_signals, "BUY")}
    <h4>Near-Triggers (ranked)</h4>
    {bullets(near_signals, "NEAR")}
    <h4>Sell Triggers (ranked)</h4>
    {bullets(sell_triggers, "SELLTRIG")}
    {charts_html}
    <h4>Snapshot (ordered by stage)</h4>
    {info_df.to_html(index=False) if not info_df.empty else "<p>No snapshot available.</p>"}
    {holdings_html}
    """

    # Plain text
    def _lines(items, kind):
        out = []
        for i, it in enumerate(items, 1):
            pace_val = it.get("pace", None); pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan); ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                out.append(f"{i}. {it['ticker']} @ {it['price']:.6g} (below SMA150 {ma_str}, pace {pace_str}, {it.get('stage','')})")
            else:
                piv = it.get("pivot", np.nan); piv_str = f"{piv:.2f}" if pd.notna(piv) else "—"
                out.append(f"{i}. {it['ticker']} @ {it['price']:.6g} (pivot {piv_str}, pace {pace_str}, {it['stage']})")
        return "\n".join(out) if out else f"No {kind} signals."

    text = f"Weinstein Crypto Watch — {now}\n\nBUY:\n{_lines(buy_signals,'BUY')}\n\nNEAR:\n{_lines(near_signals,'NEAR')}\n\nSELL:\n{_lines(sell_triggers,'SELLTRIG')}\n"

    # Persist state & diagnostics
    try:
        with open(trig_path, "w") as f:
            json.dump(trigger_state, f, indent=2)
    except Exception as e:
        log(f"Cannot save crypto trigger state: {e}", level="warn")

    if log_csv:
        try:
            pd.DataFrame(debug_rows).to_csv(log_csv, index=False)
            log(f"Wrote diagnostics CSV → {log_csv}", level="ok")
        except Exception as e:
            log(f"Failed writing diagnostics CSV: {e}", level="warn")
    if log_json:
        try:
            with open(log_json, "w") as f:
                json.dump({"rows": debug_rows}, f, indent=2, default=str)
            log(f"Wrote diagnostics JSON → {log_json}", level="ok")
        except Exception as e:
            log(f"Failed writing diagnostics JSON: {e}", level="warn")

    # Save HTML & maybe email
    os.makedirs("./output", exist_ok=True)
    html_path = os.path.join("./output", f"crypto_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    subject_counts = f"{len(buy_signals)} BUY / {len(near_signals)} NEAR / {len(sell_triggers)} SELL-TRIG (CRYPTO)"
    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        log("Sending email...", level="step")
        send_email(
            subject=f"Crypto Watch — {subject_counts}",
            html_body=html,
            text_body=text,
            cfg_path=_config_path
        )
        log("Email sent.", level="ok")

# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--quiet", action="store_true", help="reduce console noise")
    ap.add_argument("--only", type=str, default="", help="comma list of tickers to restrict evaluation (e.g. BTC-USD,ETH-USD)")
    ap.add_argument("--log-csv", type=str, default="", help="path to write per-ticker diagnostics CSV")
    ap.add_argument("--log-json", type=str, default="", help="path to write per-ticker diagnostics JSON")
    ap.add_argument("--dry-run", action="store_true", help="don’t send email")
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()] if args.only else None

    log(f"Crypto watcher starting with config: {args.config}", level="step")
    try:
        run(
            _config_path=args.config,
            only_tickers=only,
            log_csv=args.log_csv or None,
            log_json=args.log_json or None,
            dry_run=args.dry_run,
        )
        log("Crypto tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
