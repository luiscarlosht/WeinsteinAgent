#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Intraday Watcher — with trigger test mode + rich diagnostics

Adds:
- Per-position Snapshot (sorted worst → best) with colored $/%, and badge pills
- Keeps weekly summary card; holdings section now bundles both
- Same trigger logic, diagnostics CSV/JSON, HTML save, optional dry run
- "Order Block" proposing entries/stops for BUY/SELL (stocks + crypto)
- NEW: Alert Levels for BUY triggers (low stop + 15% / 20% upside targets)

Email behavior:
- Email is sent ONLY when there is at least one of:
  * Buy Triggers
  * Near-Triggers
  * Sell Triggers
  * SELL / Risk signals (from holdings / positions)
"""

import os, io, json, math, base64, yaml, argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weinstein_mailer import send_email

# ---------------- Tunables ----------------
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
BENCHMARK_DEFAULT = "SPY"
CRYPTO_BENCHMARK  = "BTC-USD"  # for RS in weekly (intraday uses price/MA/pivot only)

INTRADAY_INTERVAL = "60m"     # '60m' or '30m'
LOOKBACK_DAYS = 60
PIVOT_LOOKBACK_WEEKS = 10
VOL_PACE_MIN = 1.30
BUY_DIST_ABOVE_MA_MIN = 0.00

CONFIRM_BARS = 2
MIN_BREAKOUT_PCT = 0.004
REQUIRE_RISING_BAR_VOL = True
INTRADAY_AVG_VOL_WINDOW = 20
INTRADAY_LASTBAR_AVG_MULT = 1.20

NEAR_BELOW_PIVOT_PCT = 0.003
NEAR_VOL_PACE_MIN = 1.00

HARD_STOP_PCT = 0.08
TRAIL_ATR_MULT = 2.0
PIVOT_ENTRY_BUFFER_PCT = 0.002   # +0.20% over pivot for limit-into-strength orders

STATE_FILE = "./state/positions.json"
CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

PRICE_WINDOW_DAYS = 260
SMA_DAYS = 150

OPEN_POSITIONS_CSV_CANDIDATES = [
    "./output/Open_Positions.csv",
    "./output/open_positions.csv",
]

VERBOSE = True

# ---- NEW: Alert-level tunables ----
# For each BUY trigger we will:
# - compute entry & protective stop (Weinstein-style),
# - and upside alerts at +15% and +20% off the entry.
ALERT_TARGET_PCTS = [0.15, 0.20]   # 15% and 20% upside
ALERT_CSV_PATH = "./output/alert_levels_latest.csv"

# ---- Optional Google Sheets pull (Signals) ----
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

TAB_SIGNALS = "Signals"
TAB_MAPPING = "Mapping"

# ---------- Small helpers ----------
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg, *, level="info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {"info":"•", "ok":"✅", "step":"▶️", "warn":"⚠️", "err":"❌", "debug":"··"}.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)

# --- Stateful trigger tunables (BUY) ---
INTRADAY_STATE_FILE = "./state/intraday_triggers.json"
SCAN_INTERVAL_MIN = 10
NEAR_HITS_WINDOW = 6
NEAR_HITS_MIN = 3
COOLDOWN_SCANS = 24

# 60m-specific confirmation easing (BUY)
CONFIRM_BARS_60M = 1
INTRABAR_CONFIRM_MIN_ELAPSED = 40
INTRABAR_VOLPACE_MIN = 1.20

# --- SELL TRIGGERS ---
SELL_NEAR_ABOVE_MA_PCT = 0.005
SELL_BREAK_PCT = 0.005
SELL_NEAR_HITS_WINDOW = 6
SELL_NEAR_HITS_MIN = 3
SELL_COOLDOWN_SCANS = 24
SELL_INTRABAR_CONFIRM_MIN_ELAPSED = 40
SELL_INTRABAR_VOLPACE_MIN = 1.20

# ---------------- Config / IO ----------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}
    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")
    return cfg, benchmark, sheet_url, svc_file

def newest_weekly_csv():
    files = [f for f in os.listdir(WEEKLY_OUTPUT_DIR)
             if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No weekly CSV found in ./output. Run weinstein_report_weekly.py first.")
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])

def load_weekly_report():
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path

def load_positions():
    if not os.path.exists(os.path.dirname(STATE_FILE)):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"positions": {}}

def _load_intraday_state():
    path = INTRADAY_STATE_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _save_intraday_state(st):
    with open(INTRADAY_STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

def _elapsed_in_current_bar_minutes(intraday_df, ticker):
    try:
        if isinstance(intraday_df.columns, pd.MultiIndex):
            ts = intraday_df[("Close", ticker)].dropna().index[-1]
        else:
            ts = intraday_df["Close"].dropna().index[-1]
        last_bar_start = pd.Timestamp(ts).to_pydatetime()
        from datetime import datetime as _dt
        return max(0, int((_dt.utcnow() - last_bar_start).total_seconds() // 60))
    except Exception:
        return 0

def _update_hits(window_arr, hit, window):
    window_arr = (window_arr or [])
    window_arr.append(1 if hit else 0)
    if len(window_arr) > window:
        window_arr = window_arr[-window:]
    return window_arr, sum(window_arr)

# ---------------- Data helpers ----------------
def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _is_crypto(sym: str) -> bool:
    return (sym or "").upper().endswith("-USD")

def get_intraday(tickers):
    uniq = list(dict.fromkeys(tickers))
    intraday = yf.download(
        uniq, period=f"{LOOKBACK_DAYS}d", interval=INTRADAY_INTERVAL,
        auto_adjust=True, ignore_tz=True, progress=False
    )
    daily = yf.download(
        uniq, period="24mo", interval="1d",
        auto_adjust=True, ignore_tz=True, progress=False
    )
    return intraday, daily

def compute_atr(daily_df, t, n=14):
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            sub = daily_df.xs(t, axis=1, level=1)
        except KeyError:
            return np.nan
    else:
        sub = daily_df
    if not set(["High","Low","Close"]).issubset(set(sub.columns)):
        return np.nan
    h, l, c = sub["High"], sub["Low"], sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return float(atr.dropna().iloc[-1]) if len(atr.dropna()) else np.nan

def last_weekly_pivot_high(ticker, daily_df, weeks=PIVOT_LOOKBACK_WEEKS):
    bars = weeks * (7 if _is_crypto(ticker) else 5)
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)]
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"]
    highs = highs.dropna().tail(bars)
    return float(highs.max()) if len(highs) else np.nan

def volume_pace_today_vs_50dma(ticker, daily_df):
    """Projected full-day volume vs 50-day avg.
       For equities: 09:30–16:00 ET pacing (13:30–20:00 UTC).
       For crypto: midnight–midnight UTC pacing (24/7)."""
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

    if _is_crypto(ticker):
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = max(0.0, (now - day_start).total_seconds())
        fraction = min(1.0, max(0.05, elapsed / (24*3600.0)))
    else:
        minutes = now.hour * 60 + now.minute
        start = 13*60 + 30
        end = 20*60 + 0
        if minutes <= start:
            fraction = 0.05
        elif minutes >= end:
            fraction = 1.0
        else:
            fraction = (minutes - start) / (6.5*60)
            fraction = min(1.0, max(0.05, fraction))

    est_full = today_vol / fraction if fraction > 0 else today_vol
    return float(_safe_div(est_full, v50)) if pd.notna(v50) and v50 > 0 else np.nan

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
    elapsed = _elapsed_in_current_bar_minutes(intraday_df, ticker)
    frac = min(1.0, max(0.05, elapsed / float(bar_minutes)))
    est_full = last_bar_vol / frac if frac > 0 else last_bar_vol
    return float(_safe_div(est_full, avg_bar_vol))

# ---------------- Holdings helpers ----------------
def _coerce_numlike(series: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.number)): return float(x)
        s = str(x).replace(",", "").replace("$", "").strip()
        if s.endswith("%"): s = s[:-1]
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.apply(conv)

def _find_open_positions_csv() -> str | None:
    for p in OPEN_POSITIONS_CSV_CANDIDATES:
        if os.path.exists(p): return p
    return None

def _load_open_positions_local() -> pd.DataFrame | None:
    p = _find_open_positions_csv()
    if not p: return None
    try:
        df = pd.read_csv(p)
        if df is None or df.empty: return None
        return df
    except Exception:
        return None

def _normalize_open_positions_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {
        "Ticker":"Symbol","symbol":"Symbol","SYMBOL":"Symbol",
        "Qty":"Quantity","Shares":"Quantity","quantity":"Quantity",
        "Last":"Last Price","Price":"Last Price","LastPrice":"Last Price",
        "Current Value $":"Current Value","Market Value":"Current Value","MarketValue":"Current Value",
        "Cost Basis":"Cost Basis Total","Cost":"Cost Basis Total",
        "Avg Cost":"Average Cost Basis","AvgCost":"Average Cost Basis",
        "Gain $":"Total Gain/Loss Dollar","Gain":"Total Gain/Loss Dollar",
        "Gain %":"Total Gain/Loss Percent","GainPct":"Total Gain/Loss Percent",
        "Name":"Description","Description/Name":"Description",
        "industry":"industry","sector":"sector",
    }
    out = df.rename(columns=ren).copy()
    required = [
        "Symbol","Description","Quantity","Last Price","Current Value",
        "Cost Basis Total","Average Cost Basis",
        "Total Gain/Loss Dollar","Total Gain/Loss Percent"
    ]
    for c in required:
        if c not in out.columns: out[c] = np.nan
    num_cols = ["Quantity","Last Price","Current Value","Cost Basis Total",
                "Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent"]
    for c in num_cols: out[c] = _coerce_numlike(out[c])
    out = out.dropna(how="all")
    return out

def _merge_stage_and_recommend(positions: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    w = weekly_df.rename(columns=str.lower)
    need = ["ticker","stage","rs_above_ma","industry","sector","asset_class"]
    for n in need:
        if n not in w.columns: w[n] = np.nan
    stage_min = w[need].rename(columns={"ticker":"Symbol"})
    out = positions.merge(stage_min, on="Symbol", how="left")

    def recommend(row):
        pct = row.get("Total Gain/Loss Percent", np.nan)
        stage = str(row.get("stage",""))
        if (stage.startswith("Stage 4") and pd.notna(pct) and pct < 0) or (pd.notna(pct) and pct <= -8.0):
            return "SELL"
        return "HOLD (Strong)" if stage.startswith("Stage 2") else "HOLD"
    out["Recommendation"] = out.apply(recommend, axis=1)
    return out

# ---- Summary HTML + Holdings table ----
def _money(x): return f"${x:,.2f}" if (x is not None and pd.notna(x)) else "—"
def _pct(x):   return f"{x:.2f}%" if (x is not None and pd.notna(x)) else "—"

def _compute_portfolio_metrics(pos: pd.DataFrame) -> dict:
    cur = float(pos["Current Value"].fillna(0).sum())
    cost = float(pos["Cost Basis Total"].fillna(0).sum())
    gl_dollar = cur - cost
    port_pct = (gl_dollar / cost * 100.0) if cost else 0.0
    row_pct = pos["Total Gain/Loss Percent"].dropna().astype(float)
    avg_pct = float(row_pct.mean()) if len(row_pct) else 0.0
    return {"gl_dollar": gl_dollar, "port_pct": port_pct, "avg_pct": avg_pct}

def _colored_summary_html(m):
    def cls(v): return "pos" if v > 0 else ("neg" if v < 0 else "neu")
    rows = [
        ("Total Gain/Loss ($)", _money(m["gl_dollar"]), cls(m["gl_dollar"])),
        ("Portfolio % Gain",     _pct(m["port_pct"]),   cls(m["port_pct"])),
        ("Average % Gain",       _pct(m["avg_pct"]),    cls(m["avg_pct"]))
    ]
    tr = "\n".join([f"<tr><td>{k}</td><td class='{c}'><b>{v}</b></td></tr>" for k,v,c in rows])
    css = """
    <style>
      .sumtbl { border-collapse: collapse; width: 100%; max-width: 520px; }
      .sumtbl td { padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 14px; }
      .sumtbl td.pos { color:#0b6b2e; }
      .sumtbl td.neg { color:#a30a0a; }
      .sumtbl td.neu { color:#444; }
      .num-pos { color:#106b21; font-weight:600; }
      .num-neg { color:#8a1111; font-weight:600; }
      .num-neu { color:#444; }
      .rec-badge { display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid transparent;}
      .rec-strong { background:#0a3d1a; color:#eaffea; border-color:#0a3d1a; }
      .rec-hold   { background:#eaffea; color:#106b21; border-color:#b8e7b9; }
      .rec-sell   { background:#ffe8e6; color:#8a1111; border-color:#f3b3ae; }
      .tab-holdings { border-collapse: collapse; width:100%; }
      .tab-holdings th, .tab-holdings td { padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: top; }
      .tab-holdings th { text-align:left; background:#fafafa; position:sticky; top:0; }
      .badge { border-radius: 12px; padding: 2px 8px; font-weight:600; font-size:12px; border:1px solid rgba(0,0,0,0.06) }
    </style>
    """
    return css + f"""
    <div class="blk">
      <h3>Weinstein Weekly – Summary</h3>
      <table class="sumtbl">
        <tbody>{tr}</tbody>
      </table>
    </div>
    """

def _rec_badge(rec: str) -> str:
    r = (rec or "").upper()
    if r.startswith("SELL"):
        return '<span class="rec-sell rec-badge">SELL</span>'
    if "STRONG" in r:
        return '<span class="rec-strong rec-badge">HOLD (Strong)</span>'
    return '<span class="rec-hold rec-badge">HOLD</span>'

def _holdings_snapshot_html(merged: pd.DataFrame) -> str:
    """Build a colored table (worst → best)."""
    df = merged.copy()
    df["Total Gain/Loss Percent"] = pd.to_numeric(df["Total Gain/Loss Percent"], errors="coerce")
    df["Total Gain/Loss Dollar"]  = pd.to_numeric(df["Total Gain/Loss Dollar"], errors="coerce")
    df = df.sort_values(["Total Gain/Loss Percent","Total Gain/Loss Dollar"], ascending=[True, True])

    cols = ["Symbol","Description","industry","sector","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar",
            "Total Gain/Loss Percent","Recommendation"]

    for c in cols:
        if c not in df.columns: df[c] = np.nan

    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = []
    for _, r in df.iterrows():
        gain_d = r["Total Gain/Loss Dollar"]
        gain_p = r["Total Gain/Loss Percent"]
        cls_d = "num-pos" if pd.notna(gain_d) and gain_d > 0 else ("num-neg" if pd.notna(gain_d) and gain_d < 0 else "num-neu")
        cls_p = "num-pos" if pd.notna(gain_p) and gain_p > 0 else ("num-neg" if pd.notna(gain_p) and gain_p < 0 else "num-neu")
        row = [
            str(r.get("Symbol","")),
            str(r.get("Description","")) if pd.notna(r.get("Description")) else "",
            str(r.get("industry","")) if pd.notna(r.get("industry")) else "",
            str(r.get("sector","")) if pd.notna(r.get("sector")) else "",
            f"{float(r.get('Quantity',np.nan)):.3f}" if pd.notna(r.get("Quantity")) else "—",
            _money(r.get("Last Price",np.nan)),
            _money(r.get("Current Value",np.nan)),
            _money(r.get("Cost Basis Total",np.nan)),
            _money(r.get("Average Cost Basis",np.nan)),
            f"<span class='{cls_d}'>{_money(gain_d)}</span>",
            f"<span class='{cls_p}'>{_pct(gain_p)}</span>",
            _rec_badge(str(r.get("Recommendation","")))
        ]
        rows_html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    table = f"""
      <h4>Per-position Snapshot (worst → best)</h4>
      <table class="tab-holdings">
        <thead><tr>{header}</tr></thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    """
    return table

# ---------------- Tiny weekly-like charts (data-URI) ----------------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# ---------------- Ranking helpers ----------------
def stage_order(stage: str) -> int:
    if isinstance(stage, str):
        if stage.startswith("Stage 2"): return 0
        if stage.startswith("Stage 1"): return 1
    return 9

def buy_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    px = item.get("price", np.nan); pivot = item.get("pivot", np.nan); ma = item.get("ma30", np.nan)
    ratio_pivot = (px / pivot) if (pd.notna(px) and pd.notna(pivot) and pivot != 0) else -1e9
    ratio_ma = (px / ma) if (pd.notna(px) and pd.notna(ma) and ma != 0) else -1e9
    return (wr, st, -pace, -ratio_pivot, -ratio_ma)

def near_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan); pivot = item.get("pivot", np.nan)
    dist = abs(px - pivot) if (pd.notna(px) and pd.notna(pivot)) else 1e9
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    return (wr, st, dist, -pace)

def sell_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan); ma = item.get("ma30", np.nan)
    dist_below = (ma - px) if (pd.notna(px) and pd.notna(ma)) else -1e9
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    return (wr, st, -dist_below, -pace)

# ---------------- Logic checks ----------------
def _price_below_ma(px, ma): return pd.notna(px) and pd.notna(ma) and px <= ma * (1.0 - SELL_BREAK_PCT)
def _near_sell_zone(px, ma):
    if pd.isna(px) or pd.isna(ma): return False
    return (px >= ma * (1.0 - SELL_BREAK_PCT)) and (px <= ma * (1.0 + SELL_NEAR_ABOVE_MA_PCT))

# ---------------- ORDER BLOCK HELPERS (NEW) ----------------
def _fmt_num(x):
    if x is None or pd.isna(x): return "—"
    try: return f"{float(x):.2f}"
    except Exception: return "—"

def _propose_entry_stop_for_buy(ticker, px, pivot, ma30, atr):
    """
    Entry: pivot * (1 + PIVOT_ENTRY_BUFFER_PCT) if pivot exists, else current price
    Stop:  max(HARD_STOP_PCT below entry, ATR trail, 3% under MA150 if available)
    """
    entry = None
    if pd.notna(pivot):
        entry = pivot * (1.0 + PIVOT_ENTRY_BUFFER_PCT)
    elif pd.notna(px):
        entry = float(px)

    # candidates for stop
    hard = entry * (1.0 - HARD_STOP_PCT) if entry else np.nan
    atr_trail = (entry - TRAIL_ATR_MULT * atr) if (entry and pd.notna(atr)) else np.nan
    ma_guard = (ma30 * 0.97) if pd.notna(ma30) else np.nan

    cand = [v for v in [hard, atr_trail, ma_guard] if pd.notna(v)]
    stop = min(cand) if cand else np.nan
    return entry, stop

def _propose_protective_stop_for_open(px, ma30, atr, hard_from_positions=None):
    """
    Protective stop for existing holding:
    min(hard stop if provided, ATR trail vs px, 3% under MA150).
    """
    hard = float(hard_from_positions) if (hard_from_positions is not None and pd.notna(hard_from_positions)) else np.nan
    atr_trail = (px - TRAIL_ATR_MULT * atr) if (pd.notna(px) and pd.notna(atr)) else np.nan
    ma_guard = (ma30 * 0.97) if pd.notna(ma30) else np.nan
    cand = [v for v in [hard, atr_trail, ma_guard] if pd.notna(v)]
    return min(cand) if cand else np.nan

def _build_order_block_html(buys, sell_trigs, pos_sells):
    """
    Renders three blocks:
    - Entries (for BUY triggers)
    - Stops for newly bought (same list)
    - Protective exits (for active sell/position alerts)
    """
    if not buys and not sell_trigs and not pos_sells:
        return ""

    def _rows_for_buys(items):
        out = []
        for it in items:
            t = it["ticker"]; px = it.get("price", np.nan)
            pivot = it.get("pivot", np.nan); ma = it.get("ma30", np.nan)
            atr = it.get("atr", np.nan)
            entry, stop = _propose_entry_stop_for_buy(t, px, pivot, ma, atr)
            out.append(f"<tr><td>{t}</td><td>{_fmt_num(px)}</td><td>{_fmt_num(pivot)}</td><td>{_fmt_num(entry)}</td><td>{_fmt_num(stop)}</td></tr>")
        return "\n".join(out)

    def _rows_for_stops(title, items):
        out = []
        for it in items:
            t = it["ticker"]; px = it.get("price", np.nan); ma = it.get("ma30", np.nan)
            atr = it.get("atr", np.nan)
            stop = _propose_protective_stop_for_open(px, ma, atr)
            out.append(f"<tr><td>{t}</td><td>{_fmt_num(px)}</td><td>{_fmt_num(ma)}</td><td>{_fmt_num(stop)}</td></tr>")
        return "\n".join(out)

    css = """
    <style>
      .ordtbl { border-collapse: collapse; width:100%; margin-top:6px; }
      .ordtbl th, .ordtbl td { border-bottom:1px solid #eee; padding:6px 8px; font-size:13px; text-align:left; }
      .ordtbl th { background:#fafafa; }
    </style>
    """
    html = css + "<h4>Order Block (proposed)</h4>"

    if buys:
        html += """
        <div><b>Prospective Entries (on BUY triggers)</b></div>
        <table class="ordtbl">
          <thead><tr><th>Ticker</th><th>Now</th><th>Pivot</th><th>Entry ≥</th><th>Initial Stop ≤</th></tr></thead>
          <tbody>
        """
        html += _rows_for_buys(buys) + "</tbody></table>"

    if sell_trigs or pos_sells:
        html += """
        <div style="margin-top:8px;"><b>Protective Stops (for open positions / sell alerts)</b></div>
        <table class="ordtbl">
          <thead><tr><th>Ticker</th><th>Now</th><th>MA150</th><th>Protective Stop ≤</th></tr></thead>
          <tbody>
        """
        html += _rows_for_stops("Stops", sell_trigs + pos_sells) + "</tbody></table>"

    html += "<div style='font-size:12px;color:#666;margin-top:6px;'>"
    html += f"Rules: Entry ≈ pivot + {PIVOT_ENTRY_BUFFER_PCT*100:.2f}% buffer; Initial stop = min(hard {HARD_STOP_PCT*100:.1f}% below entry, ATR×{TRAIL_ATR_MULT:.1f} trail, 3% under MA150). Protective stop for existing longs uses the same min(). Crypto handled identically.</div>"
    return html

def _build_order_block_text(buys, sell_trigs, pos_sells):
    lines = ["ORDER BLOCK (proposed)"]
    if buys:
        lines.append("Prospective Entries:")
        for it in buys:
            t = it["ticker"]; px = it.get("price", np.nan)
            pivot = it.get("pivot", np.nan); ma = it.get("ma30", np.nan); atr = it.get("atr", np.nan)
            entry, stop = _propose_entry_stop_for_buy(t, px, pivot, ma, atr)
            lines.append(f"- {t}: now={_fmt_num(px)} pivot={_fmt_num(pivot)} entry≥{_fmt_num(entry)} stop≤{_fmt_num(stop)}")
    if sell_trigs or pos_sells:
        lines.append("Protective Stops (open positions / sell alerts):")
        for it in (sell_trigs + pos_sells):
            t = it["ticker"]; px = it.get("price", np.nan); ma = it.get("ma30", np.nan); atr = it.get("atr", np.nan)
            stop = _propose_protective_stop_for_open(px, ma, atr)
            lines.append(f"- {t}: now={_fmt_num(px)} MA150={_fmt_num(ma)} stop≤{_fmt_num(stop)}")
    lines.append(f"Rules: entry≈pivot+{PIVOT_ENTRY_BUFFER_PCT*100:.2f}% | stop=min(hard {HARD_STOP_PCT*100:.1f}%, ATR×{TRAIL_ATR_MULT:.1f}, MA150−3%).")
    return "\n".join(lines)

# ---------------- NEW: Alert Levels helpers ----------------
def _compute_alert_rows_for_buys(buy_signals):
    """
    For each BUY trigger, compute:
      - entry (same as Order Block)
      - Weinstein-style protective stop
      - +15% / +20% upside alerts
    Returns a list of dicts suitable for CSV and email.
    """
    rows = []
    for it in buy_signals:
        t = it["ticker"]
        px = it.get("price", np.nan)
        pivot = it.get("pivot", np.nan)
        ma = it.get("ma30", np.nan)
        atr = it.get("atr", np.nan)
        entry, stop = _propose_entry_stop_for_buy(t, px, pivot, ma, atr)
        if entry is None or pd.isna(entry):
            continue
        targets = []
        for pt in ALERT_TARGET_PCTS:
            targets.append(entry * (1.0 + pt))
        row = {
            "ticker": t,
            "entry": float(entry),
            "protective_stop": float(stop) if pd.notna(stop) else np.nan,
        }
        # label targets as target_15, target_20, etc.
        for pt, val in zip(ALERT_TARGET_PCTS, targets):
            key = f"target_{int(pt*100)}"
            row[key] = float(val)
        rows.append(row)
    return rows

def _build_alert_block_html(alert_rows):
    if not alert_rows:
        return ""
    css = """
    <style>
      .alerttbl { border-collapse: collapse; width:100%; margin-top:10px; }
      .alerttbl th, .alerttbl td { border-bottom:1px solid #eee; padding:6px 8px; font-size:13px; text-align:left; }
      .alerttbl th { background:#fafafa; }
    </style>
    """
    header_cols = ["Ticker","Entry (approx)","Protective Stop (Low Alert)"]
    for pt in ALERT_TARGET_PCTS:
        header_cols.append(f"High Alert +{int(pt*100)}%")
    header_html = "".join(f"<th>{c}</th>" for c in header_cols)
    body_rows = []
    for r in alert_rows:
        cells = [
            r["ticker"],
            _fmt_num(r.get("entry")),
            _fmt_num(r.get("protective_stop")),
        ]
        for pt in ALERT_TARGET_PCTS:
            key = f"target_{int(pt*100)}"
            cells.append(_fmt_num(r.get(key)))
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    html = css + """
    <h4>Alert Levels for New BUY Triggers</h4>
    <p style="font-size:12px;color:#555;">
      Use these as <b>price alerts</b> in Fidelity (or your broker). Protective stop is based on Weinstein-style
      support/MA + ATR; high alerts are simple +15% / +20% upside from the suggested entry.
    </p>
    <table class="alerttbl">
      <thead><tr>{hdr}</tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """.format(hdr=header_html, rows="".join(body_rows))
    return html

def _build_alert_block_text(alert_rows):
    if not alert_rows:
        return ""
    lines = ["ALERT LEVELS for BUY triggers (for broker price alerts):"]
    for r in alert_rows:
        t = r["ticker"]
        entry = _fmt_num(r.get("entry"))
        stop = _fmt_num(r.get("protective_stop"))
        ups = []
        for pt in ALERT_TARGET_PCTS:
            key = f"target_{int(pt*100)}"
            ups.append(f"+{int(pt*100)}%={_fmt_num(r.get(key))}")
        ups_str = ", ".join(ups)
        lines.append(f"- {t}: entry≈{entry}, low alert (protective stop)≤{stop}, high alerts: {ups_str}")
    return "\n".join(lines)

# ---------------- Main logic ----------------
def run(_config_path="./config.yaml", *, only_tickers=None, test_ease=False, log_csv=None, log_json=None, dry_run=False):
    log("Intraday watcher starting with config: {0}".format(_config_path), level="step")
    cfg, benchmark, sheet_url, service_account_file = load_config(_config_path)
    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # Normalize expected columns
    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker","stage","ma30","rs_above_ma","asset_class"]:
        if miss not in w.columns: w[miss] = np.nan
    focus = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])][["ticker","stage","ma30","rs_above_ma","asset_class"]].copy()
    if "rank" in w.columns: focus["weekly_rank"] = w["rank"]
    else: focus["weekly_rank"] = 999999

    if only_tickers:
        filt = set([t.strip().upper() for t in only_tickers])
        focus = focus[focus["ticker"].isin(filt)].copy()

    log(f"Focus universe: {len(focus)} symbols (Stage 1/2).", level="info")

    state = load_positions()
    held = state.get("positions", {}) or {}
    if held:
        log(f"Held symbols detected: {sorted(held.keys())}", level="debug")

    # Benchmarks to ensure RS charts render: equity + crypto
    needs = sorted(set(focus["ticker"].tolist() + [benchmark, CRYPTO_BENCHMARK]))

    log("Downloading intraday + daily bars...", level="step")
    intraday, daily = get_intraday(needs)
    log("Price data downloaded.", level="ok")

    if isinstance(intraday.columns, pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    def px_now(t):
        if hasattr(last_closes, "index") and (t in last_closes.index):
            return float(last_closes.get(t, np.nan))
        vals = getattr(last_closes, "values", [])
        return float(vals[-1]) if len(vals) else np.nan

    trigger_state = _load_intraday_state()

    buy_signals, near_signals, sell_signals = [], [], []
    sell_triggers, sell_from_positions, info_rows, chart_imgs = [], [], [], []

    # Test easing via flag or environment
    ease = test_ease or (os.getenv("INTRADAY_TEST", "0") == "1")
    if ease:
        log("TEST-EASE: lowering thresholds for quick validation.", level="warn")
    _NEAR_HITS_MIN = 1 if ease else NEAR_HITS_MIN
    _SELL_NEAR_HITS_MIN = 1 if ease else SELL_NEAR_HITS_MIN
    _INTRABAR_CONFIRM_MIN_ELAPSED = 0 if ease else INTRABAR_CONFIRM_MIN_ELAPSED
    _INTRABAR_VOLPACE_MIN = 0.0 if ease else INTRABAR_VOLPACE_MIN
    _SELL_INTRABAR_CONFIRM_MIN_ELAPSED = 0 if ease else SELL_INTRABAR_CONFIRM_MIN_ELAPSED
    _SELL_INTRABAR_VOLPACE_MIN = 0.0 if ease else SELL_INTRABAR_VOLPACE_MIN

    log("Evaluating candidates...", level="step")
    debug_rows = []

    for _, row in focus.iterrows():
        t = row["ticker"]
        if t in (benchmark, CRYPTO_BENCHMARK):
            continue
        px = px_now(t)
        if np.isnan(px):
            continue

        stage = str(row["stage"]); ma30 = float(row.get("ma30", np.nan))
        rs_above = bool(row.get("rs_above_ma", False))
        weekly_rank = float(row.get("weekly_rank", np.nan))
        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma(t, daily)
        atr = compute_atr(daily, t, n=14)

        closes_n = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
        ma_ok = pd.notna(ma30); pivot_ok = pd.notna(pivot); rs_ok = rs_above

        elapsed = _elapsed_in_current_bar_minutes(intraday, t) if INTRADAY_INTERVAL == "60m" else None
        pace_intra = intrabar_volume_pace(intraday, t, bar_minutes=60) if INTRADAY_INTERVAL == "60m" else None

        d = {"why": [], "cond": {}, "metrics": {}}
        d["metrics"].update({
            "price": px, "ma30": ma30, "pivot": pivot, "atr": atr,
            "pace_full_vs50dma": None if pd.isna(pace) else float(pace),
            "pace_intrabar": None if pd.isna(pace_intra) else float(pace_intra),
            "elapsed_min": elapsed,
            "held": (t in held),
        })
        d["cond"]["weekly_stage_ok"] = stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)")
        d["cond"]["rs_ok"] = bool(rs_ok)
        d["cond"]["ma_ok"] = bool(ma_ok)
        d["cond"]["pivot_ok"] = bool(pivot_ok)

        # --- BUY confirm ---
        confirm = False; vol_ok = True; price_ok = False
        if ma_ok and pivot_ok and closes_n:
            def _price_ok(c):
                return (c >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and (c >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN))
            if INTRADAY_INTERVAL == "60m":
                last_c = closes_n[-1]; price_ok = _price_ok(last_c)
                vol_ok = (pd.isna(pace_intra) or pace_intra >= _INTRABAR_VOLPACE_MIN)
                confirm = price_ok and (elapsed is not None and elapsed >= _INTRABAR_CONFIRM_MIN_ELAPSED) and vol_ok
            else:
                need = max(CONFIRM_BARS, 2)
                price_ok = all(_price_ok(c) for c in closes_n[-need:])
                confirm = price_ok
                if REQUIRE_RISING_BAR_VOL:
                    vols2 = get_last_n_intraday_volumes(intraday, t, n=2)
                    vavg = get_intraday_avg_volume(intraday, t, window=INTRADAY_AVG_VOL_WINDOW)
                    if len(vols2) >= 2 and pd.notna(vavg) and vavg > 0:
                        vol_ok = (vols2[-1] >= INTRADAY_LASTBAR_AVG_MULT * vavg)
                    else:
                        vol_ok = False

        d["cond"].update({
            "buy_price_ok": bool(price_ok),
            "buy_vol_ok": bool(vol_ok),
            "buy_confirm": bool(confirm),
            "pace_full_gate": (pd.isna(pace) or pace >= VOL_PACE_MIN),
            "near_pace_gate": (pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN),
        })
        if not d["cond"]["weekly_stage_ok"]: d["why"].append("Not Stage 1/2")
        if not d["cond"]["rs_ok"]: d["why"].append("RS below MA")
        if not d["cond"]["ma_ok"]: d["why"].append("No MA30")
        if not d["cond"]["pivot_ok"]: d["why"].append("No 10w pivot")

        # --- BUY near flag ---
        near_now = False
        if d["cond"]["weekly_stage_ok"] and rs_ok and pivot_ok and ma_ok and pd.notna(px):
            above_ma = px >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
            if above_ma:
                if (px >= pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)) and (px < pivot * (1.0 + MIN_BREAKOUT_PCT)):
                    near_now = True
                elif (px >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not confirm:
                    near_now = True
        d["cond"]["near_now"] = bool(near_now)

        # --- SELL near/confirm ---
        sell_near_now = False; sell_confirm = False; sell_vol_ok = True
        sell_price_ok = False
        if ma_ok and pd.notna(px):
            sell_near_now = _near_sell_zone(px, ma30)
            if INTRADAY_INTERVAL == "60m":
                sell_price_ok = _price_below_ma(px, ma30)
                sell_vol_ok = (pd.isna(pace_intra) or (pace_intra >= _SELL_INTRABAR_VOLPACE_MIN))
                sell_confirm = bool(sell_price_ok and (elapsed is not None and elapsed >= _SELL_INTRABAR_CONFIRM_MIN_ELAPSED) and sell_vol_ok)
            else:
                closes_n2 = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
                if closes_n2:
                    sell_price_ok = all((c <= ma30 * (1.0 - SELL_BREAK_PCT)) for c in closes_n2[-CONFIRM_BARS:])
                    sell_confirm = sell_price_ok
        d["cond"].update({
            "sell_near_now": bool(sell_near_now),
            "sell_price_ok": bool(sell_price_ok),
            "sell_vol_ok": bool(sell_vol_ok),
            "sell_confirm": bool(sell_confirm),
        })

        # --- promotion state ---
        ts_key = t
        st = trigger_state.get(ts_key, {
            "state":"IDLE", "near_hits":[], "cooldown":0,
            "sell_state":"IDLE", "sell_hits":[], "sell_cooldown":0
        })
        # BUY hits
        st["near_hits"], near_count = _update_hits(st.get("near_hits", []), near_now, NEAR_HITS_WINDOW)
        if st.get("cooldown", 0) > 0: st["cooldown"] = int(st["cooldown"]) - 1
        state_now = st.get("state", "IDLE")
        if state_now == "IDLE" and near_now: state_now = "NEAR"
        elif state_now in ("IDLE","NEAR") and near_count >= _NEAR_HITS_MIN: state_now = "ARMED"
        elif state_now == "ARMED" and confirm and vol_ok and (pd.isna(pace) or pace >= VOL_PACE_MIN):
            state_now = "TRIGGERED"; st["cooldown"] = COOLDOWN_SCANS
        elif st["cooldown"] > 0 and not near_now: state_now = "COOLDOWN"
        elif st["cooldown"] == 0 and not near_now and not confirm: state_now = "IDLE"
        st["state"] = state_now

        # SELL hits
        st["sell_hits"], sell_hit_count = _update_hits(st.get("sell_hits", []), sell_near_now, SELL_NEAR_HITS_WINDOW)
        if st.get("sell_cooldown", 0) > 0: st["sell_cooldown"] = int(st["sell_cooldown"]) - 1
        sell_state = st.get("sell_state", "IDLE")
        if sell_state == "IDLE" and sell_near_now: sell_state = "NEAR"
        elif sell_state in ("IDLE","NEAR") and sell_hit_count >= _SELL_NEAR_HITS_MIN: sell_state = "ARMED"
        elif sell_state == "ARMED" and sell_confirm and sell_vol_ok:
            sell_state = "TRIGGERED"; st["sell_cooldown"] = SELL_COOLDOWN_SCANS
        elif st["sell_cooldown"] > 0 and not sell_near_now: sell_state = "COOLDOWN"
        elif st["sell_cooldown"] == 0 and not sell_near_now and not sell_confirm: sell_state = "IDLE"
        st["sell_state"] = sell_state

        trigger_state[ts_key] = st

        # --- SELL risk (tracked positions.json) ---
        pos = held.get(t)
        if pos:
            entry = float(pos.get("entry", np.nan))
            hard_stop = float(pos.get("stop", np.nan)) if pd.notna(pos.get("stop", np.nan)) \
                        else (entry * (1 - HARD_STOP_PCT) if pd.notna(entry) else np.nan)
            atr_pos = atr
            trail = px - TRAIL_ATR_MULT * atr_pos if pd.notna(atr_pos) else None
            breach_hard = (pd.notna(hard_stop) and px <= hard_stop)
            breach_ma = (pd.notna(ma30) and px <= ma30 * 0.97)
            breach_trail = (trail is not None and px <= trail)
            if breach_hard or breach_ma or breach_trail:
                why = []
                if breach_hard:  why.append(f"≤ hard stop ({hard_stop:.2f})")
                if breach_ma:    why.append("≤ 30-wk MA proxy (−3%)")
                if breach_trail: why.append(f"≤ ATR trail ({TRAIL_ATR_MULT}×)")
                sell_signals.append({
                    "ticker": t, "price": px, "reasons": ", ".join(why),
                    "stage": stage, "weekly_rank": weekly_rank, "source": "risk",
                    "atr": atr_pos, "ma30": ma30
                })

        # --- EMIT by state ---
        if st["state"] == "TRIGGERED" and (pd.isna(pace) or pace >= VOL_PACE_MIN):
            buy_signals.append({
                "ticker": t, "price": px, "pivot": pivot, "pace": None if pd.isna(pace) else float(pace),
                "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank, "atr": atr
            })
            trigger_state[t]["state"] = "COOLDOWN"
        elif st["state"] in ("NEAR","ARMED"):
            if (pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN):
                near_signals.append({
                    "ticker": t, "price": px, "pivot": pivot, "pace": None if pd.isna(pace) else float(pace),
                    "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank, "reason": "near/armed", "atr": atr
                })

        if st["sell_state"] == "TRIGGERED":
            sell_triggers.append({
                "ticker": t, "price": px, "ma30": ma30, "stage": stage,
                "weekly_rank": weekly_rank, "pace": None if pd.isna(pace) else float(pace), "atr": atr
            })
            trigger_state[t]["sell_state"] = "COOLDOWN"

        info_rows.append({
            "ticker": t, "stage": stage, "price": px, "ma30": ma30, "pivot10w": pivot,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace), 2),
            "two_bar_confirm": confirm, "last_bar_vol_ok": vol_ok if 'vol_ok' in locals() else None,
            "weekly_rank": weekly_rank,
            "buy_state": st["state"], "sell_state": st["sell_state"],
        })
        debug_rows.append({"ticker": t, **d["metrics"],
                           **{f"cond_{k}": v for k,v in d["cond"].items()},
                           "state": st["state"], "sell_state": st["sell_state"]})

    log(f"Scan done. Raw counts → BUY:{len(buy_signals)} NEAR:{len(near_signals)} SELLTRIG:{len(sell_triggers)}", level="info")

    # ---------- SELL recommendations from holdings ----------
    holdings_block_html = ""
    holdings_raw = _load_open_positions_local()
    if holdings_raw is not None and not holdings_raw.empty:
        pos_norm = _normalize_open_positions_columns(holdings_raw)
        merged = _merge_stage_and_recommend(pos_norm, weekly_df)

        # build "Position SELL" list for email bullets (strategy rules)
        sell_from_positions_map = {}
        for _, r in merged.iterrows():
            rec = str(r.get("Recommendation", "")).upper()
            if not rec.startswith("SELL"): continue
            sym = str(r.get("Symbol", "")).strip()
            if not sym: continue
            reasons = []
            pct = r.get("Total Gain/Loss Percent", np.nan)
            stg = str(r.get("stage", ""))
            if pd.notna(pct) and pct <= -8.0: reasons.append("drawdown ≤ −8%")
            if stg.startswith("Stage 4") and (pd.notna(pct) and pct < 0): reasons.append("Stage 4 + negative P/L")
            if not reasons: reasons.append("strategy rule")
            entry = sell_from_positions_map.get(sym)
            if entry is None:
                sell_from_positions_map[sym] = {
                    "ticker": sym, "price": np.nan,   # live not required
                    "reasons": set(reasons), "stage": stg, "weekly_rank": np.nan, "source": "positions",
                    "ma30": r.get("ma30", np.nan), "atr": np.nan
                }
            else:
                entry["reasons"].update(reasons)

        sell_from_positions = []
        for sym, entry in sell_from_positions_map.items():
            entry["reasons"] = "; ".join(sorted(entry["reasons"]))
            sell_from_positions.append(entry)

        # Summary + colored snapshot (worst → best)
        metrics = _compute_portfolio_metrics(pos_norm)
        holdings_block_html = _colored_summary_html(metrics) + _holdings_snapshot_html(merged)

    # -------- Ranking & charts --------
    buy_signals.sort(key=buy_sort_key); near_signals.sort(key=near_sort_key); sell_triggers.sort(key=sell_sort_key)

    charts_added = 0; chart_imgs = []
    for item in buy_signals:
        if charts_added >= MAX_CHARTS_PER_EMAIL: break
        t = item["ticker"]
        bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
        path, data_uri = make_tiny_chart_png(t, bmk, daily)
        if data_uri:
            chart_imgs.append((t, data_uri)); charts_added += 1
    if charts_added < MAX_CHARTS_PER_EMAIL:
        for item in near_signals:
            if charts_added >= MAX_CHARTS_PER_EMAIL: break
            t = item["ticker"]
            bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
            path, data_uri = make_tiny_chart_png(t, bmk, daily)
            if data_uri:
                chart_imgs.append((t, data_uri)); charts_added += 1

    log(f"Charts prepared: {len(chart_imgs)}", level="debug")

    # -------- Alert levels for BUY triggers --------
    alert_rows = _compute_alert_rows_for_buys(buy_signals)
    if alert_rows:
        try:
            os.makedirs(os.path.dirname(ALERT_CSV_PATH), exist_ok=True)
            pd.DataFrame(alert_rows).to_csv(ALERT_CSV_PATH, index=False)
            log(f"Wrote alert levels CSV → {ALERT_CSV_PATH}", level="ok")
        except Exception as e:
            log(f"Failed writing alert levels CSV: {e}", level="warn")

    # -------- Build Email --------
    info_df = pd.DataFrame(info_rows)
    if not info_df.empty:
        info_df["stage_rank"] = info_df["stage"].apply(stage_order)
        info_df["weekly_rank"] = pd.to_numeric(info_df["weekly_rank"], errors="coerce").fillna(999999).astype(int)
        info_df = info_df.sort_values(["weekly_rank","stage_rank","ticker"]).drop(columns=["stage_rank"])

    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} signals.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            src = it.get("source", "")
            src_label = " (Position SELL)" if src == "positions" else ""
            if kind == "SELL":
                price_str = f"{it['price']:.2f}" if pd.notna(it.get("price", np.nan)) else "—"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {price_str} — {it.get('reasons','')} (" \
                           f"{it.get('stage','')}, weekly {wr_str}){src_label}</li>")
            elif kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} (↓ MA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})</li>")
            else:
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} (pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']}, weekly {wr_str})</li>")
        return "<ol>" + "\n".join(lis) + "</ol>"

    charts_html = ""
    if chart_imgs:
        charts_html = "<h4>Charts (Price + SMA150 ≈ 30-wk MA, RS normalized)</h4>"
        for t, data_uri in chart_imgs:
            charts_html += f"""
            <div style="display:inline-block;margin:6px 8px 10px 0;vertical-align:top;text-align:center;">
              <img src="{data_uri}" alt="{t}" style="border:1px solid #eee;border-radius:6px;max-width:320px;">
              <div style="font-size:12px;color:#555;margin-top:3px;">{t}</div>
            </div>
            """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <h3>Weinstein Intraday Watch — {now}</h3>
    <p><i>
      BUY: Weekly Stage 1/2 + confirm over ~10-week pivot & 30-wk MA proxy (SMA150),
      +{MIN_BREAKOUT_PCT*100:.1f}% headroom, RS support, volume pace ≥ {VOL_PACE_MIN}×.
      For 60m bars: ≥{INTRABAR_CONFIRM_MIN_ELAPSED} min elapsed & intrabar pace ≥ {INTRABAR_VOLPACE_MIN}×.<br>
      NEAR-TRIGGER: Stage 1/2 + RS ok, price within {NEAR_BELOW_PIVOT_PCT*100:.1f}% below pivot or first close over pivot but not fully confirmed yet,
      volume pace ≥ {NEAR_VOL_PACE_MIN}×.<br>
      SELL-TRIGGER: Confirmed crack below MA150 by {SELL_BREAK_PCT*100:.1f}% with persistence; for 60m bars, ≥{SELL_INTRABAR_CONFIRM_MIN_ELAPSED} min elapsed & intrabar pace ≥ {SELL_INTRABAR_VOLPACE_MIN}×.
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
    <h4>Sell / Risk Triggers (Tracked Positions & Position Recommendations)</h4>
    {bullets(sell_signals + sell_from_positions, "SELL")}
    """

    # ------ ORDER BLOCK (HTML + TEXT) ------
    order_block_html = _build_order_block_html(buy_signals, sell_triggers, sell_from_positions)
    if order_block_html:
        html += order_block_html

    # ------ ALERT BLOCK (HTML) ------
    alert_block_html = _build_alert_block_html(alert_rows)
    if alert_block_html:
        html += alert_block_html

    html += f"""
    <h4>Snapshot (ordered by weekly rank & stage)</h4>
    {pd.DataFrame(info_rows).to_html(index=False)}
    """

    if holdings_block_html:
        html += "<hr/>" + holdings_block_html

    # Plain text body
    def _lines(items, kind):
        out = []
        for i, it in enumerate(items, 1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                out.append(f"{i}. {it['ticker']} @ {it['price']:.2f} (below MA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})")
            elif kind == "SELL":
                price_str = f"{it['price']:.2f}" if pd.notna(it.get("price", np.nan)) else "—"
                out.append(f"{i}. {it['ticker']} @ {price_str} — {it.get('reasons','')} ({it.get('stage','')}, weekly {wr_str})")
            else:
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                out.append(f"{i}. {it['ticker']} @ {it['price']:.2f} (pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']}, weekly {wr_str})")
        return "\n".join(out) if out else f"No {kind} signals."

    text = (
        f"Weinstein Intraday Watch — {now}\n\n"
        f"BUY (ranked):\n{_lines(buy_signals,'BUY')}\n\n"
        f"NEAR-TRIGGER (ranked):\n{_lines(near_signals,'NEAR')}\n\n"
        f"SELL TRIGGERS (ranked):\n{_lines(sell_triggers,'SELLTRIG')}\n\n"
        f"SELL / RISK:\n{_lines(sell_signals + sell_from_positions,'SELL')}\n\n"
    )

    order_block_text = _build_order_block_text(buy_signals, sell_triggers, sell_from_positions)
    if order_block_text:
        text += "\n" + order_block_text + "\n"

    alert_block_text = _build_alert_block_text(alert_rows)
    if alert_block_text:
        text += "\n" + alert_block_text + "\n"

    # Persist state & diagnostics
    _save_intraday_state(trigger_state)

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

    # Save HTML alongside sending
    os.makedirs("./output", exist_ok=True)
    html_path = os.path.join("./output", f"intraday_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    # -------- NEW: Email only when we have signals --------
    has_signals = bool(
        buy_signals
        or near_signals
        or sell_triggers
        or sell_signals
        or sell_from_positions
    )

    if not has_signals:
        log("No BUY/NEAR/SELL triggers present — skipping email send.", level="info")
        if dry_run:
            log("DRY-RUN set — no email would be sent anyway.", level="debug")
        return

    subject_counts = f"{len(buy_signals)} BUY / {len(near_signals)} NEAR / {len(sell_triggers)} SELL-TRIG / {len(sell_signals)} SELL"
    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        log("Sending email...", level="step")
        send_email(
            subject=f"Intraday Watch — {subject_counts}",
            html_body=html,
            text_body=text,
            cfg_path=_config_path
        )
        log("Email sent.", level="ok")

# ---------------- Charting (kept at bottom so helpers exist) ----------------
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
    ax2.plot(rs_norm.index, rs_norm.values, linestyle="--", alpha=0.7, label="RS (norm)")
    ax2.set_ylabel("RS (norm)")
    ax2.tick_params(axis='y', labelsize=8)
    ax1.set_title(f"{ticker} — Price, SMA150, RS/{benchmark}", fontsize=9)
    ax1.grid(alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left", frameon=False)
    chart_path = os.path.join(CHART_DIR, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.tight_layout(pad=0.8); fig.savefig(chart_path, bbox_inches="tight"); plt.close(fig)
    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--quiet", action="store_true", help="reduce console noise")
    ap.add_argument("--only", type=str, default="", help="comma list of tickers to restrict evaluation (e.g. MU,DDOG)")
    ap.add_argument("--test-ease", action="store_true", help="enable trigger easing for testing (or set INTRADAY_TEST=1)")
    ap.add_argument("--log-csv", type=str, default="", help="path to write per-ticker diagnostics CSV")
    ap.add_argument("--log-json", type=str, default="", help="path to write per-ticker diagnostics JSON")
    ap.add_argument("--dry-run", action="store_true", help="don’t send email")
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()] if args.only else None

    log(f"Intraday watcher starting with config: {args.config}", level="step")
    try:
        run(
            _config_path=args.config,
            only_tickers=only,
            test_ease=args.test_ease,
            log_csv=args.log_csv or None,
            log_json=args.log_json or None,
            dry_run=args.dry_run,
        )
        log("Intraday tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
