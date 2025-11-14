#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Short Intraday Watcher — Stage 4 short setups

- Universe: Stage 4 (Downtrend) names from weekly CSV
- Logic: Find near-short setups just above breakdown zones, and live short triggers
- Output: Email with ranked lists + charts + suggested entries/stops/targets

Short-side is symmetric to long-side logic, but:
- We look for price breaking DOWN through ~10-week pivot lows and/or SMA150
- Volume pace is used the same way to confirm expansion on breakdowns
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
CRYPTO_BENCHMARK  = "BTC-USD"  # for RS in weekly-style charts

INTRADAY_INTERVAL = "60m"     # '60m' or '30m'
LOOKBACK_DAYS = 60
PIVOT_LOOKBACK_WEEKS = 10
VOL_PACE_MIN_TRIG = 1.30      # min pace vs 50d avg for short triggers
VOL_PACE_MIN_NEAR = 1.00      # min pace for near-short

CONFIRM_BARS = 2
MIN_BREAKDOWN_PCT = 0.004     # 0.4% below pivot/MA for confirmed short breakdown
REQUIRE_RISING_BAR_VOL = True
INTRADAY_AVG_VOL_WINDOW = 20
INTRADAY_LASTBAR_AVG_MULT = 1.20

NEAR_ABOVE_PIVOT_PCT = 0.003  # 0.3% above pivot low
NEAR_ABOVE_MA_PCT    = 0.003  # 0.3% above MA150

SHORT_INTRADAY_STATE_FILE = "./state/short_intraday_triggers.json"
SCAN_INTERVAL_MIN = 10
NEAR_HITS_WINDOW = 6
NEAR_HITS_MIN = 3
COOLDOWN_SCANS = 24

# 60m-specific confirmation easing (SHORT)
SHORT_CONFIRM_BARS_60M = 1
INTRABAR_CONFIRM_MIN_ELAPSED = 40
INTRABAR_VOLPACE_MIN = 1.20

PRICE_WINDOW_DAYS = 260
SMA_DAYS = 150
CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

VERBOSE = True

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

def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _is_crypto(sym: str) -> bool:
    return (sym or "").upper().endswith("-USD")

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

def _load_short_state():
    path = SHORT_INTRADAY_STATE_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _save_short_state(st):
    with open(SHORT_INTRADAY_STATE_FILE, "w") as f:
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

def last_weekly_pivot_low(ticker, daily_df, weeks=PIVOT_LOOKBACK_WEEKS):
    bars = weeks * (7 if _is_crypto(ticker) else 5)
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            lows = daily_df[("Low", ticker)]
        except KeyError:
            return np.nan
    else:
        lows = daily_df["Low"]
    lows = lows.dropna().tail(bars)
    return float(lows.min()) if len(lows) else np.nan

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

# ---------------- Ranking helpers ----------------
def stage_order(stage: str) -> int:
    if isinstance(stage, str):
        if stage.startswith("Stage 4"): return 0
        if stage.startswith("Stage 3"): return 1
    return 9

def short_sort_key(item):
    # rank by: weekly_rank, stage_order, distance below pivot/MA, volume pace
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan)
    ma = item.get("ma30", np.nan)
    pivot = item.get("pivot_low_10w", np.nan)
    dist = 0.0
    if pd.notna(px) and pd.notna(pivot):
        dist += (pivot - px)  # bigger breakdown = more negative, but we want more distance → sort by -dist
    if pd.notna(px) and pd.notna(ma):
        dist += (ma - px)
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    return (wr, st, -dist, -pace)

def near_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan)
    pivot = item.get("pivot_low_10w", np.nan)
    dist = abs(px - pivot) if (pd.notna(px) and pd.notna(pivot)) else 1e9
    pace = item.get("pace", np.nan); pace = pace if pd.notna(pace) else -1e9
    return (wr, st, dist, -pace)

# ---------------- Logic checks ----------------
def _price_below_break_zone(px, pivot_low, ma):
    """
    Confirmed short breakdown:
    - price below pivot low by MIN_BREAKDOWN_PCT
    - and/or below MA150 by same buffer
    We'll say: require either pivot breakdown OR MA breakdown.
    """
    if pd.isna(px): return False
    cond_pivot = (pd.notna(pivot_low) and px <= pivot_low * (1.0 - MIN_BREAKDOWN_PCT))
    cond_ma    = (pd.notna(ma) and px <= ma * (1.0 - MIN_BREAKDOWN_PCT))
    return cond_pivot or cond_ma

def _near_short_zone(px, pivot_low, ma):
    """
    NEAR-short:
    - Price just ABOVE pivot low or MA150, within NEAR_ABOVE_*% band
    - Idea: stock hanging above floor, at risk of breaking.
    """
    if pd.isna(px):
        return False
    cond_pivot = False
    cond_ma = False
    if pd.notna(pivot_low):
        cond_pivot = (px >= pivot_low * (1.0 - NEAR_ABOVE_PIVOT_PCT)) and (px <= pivot_low * (1.0 + NEAR_ABOVE_PIVOT_PCT))
    if pd.notna(ma):
        cond_ma = (px >= ma * (1.0 - NEAR_ABOVE_MA_PCT)) and (px <= ma * (1.0 + NEAR_ABOVE_MA_PCT))
    return cond_pivot or cond_ma

# ---------------- ORDER BLOCK HELPERS (short) ----------------
def _fmt_num(x):
    if x is None or pd.isna(x): return "—"
    try: return f"{float(x):.2f}"
    except Exception: return "—"

def _propose_short_entry_stop_targets(ticker, px, pivot_low, ma30, atr):
    """
    For short entries:
    - Entry: just under pivot_low or MA150 (if we are already under them in trigger land).
      In practice for the email we show current price as 'entry≈px' once TRIG.
    - Stop: above recent consolidation / MA150 / ATR.
      We'll use a symmetric idea to the long side but inverted:
        * stop = max(px + HARD_STOP_PCT*px, ma30 * 1.03, px + ATR*TRAIL_ATR_MULT)
      For simplicity, let's assume:
        - HARD_STOP_PCT (same 8% notion) but used upwards.
    - Targets: 1R = entry − (stop − entry), 2R = entry − 2*(stop − entry)
    """
    HARD_STOP_PCT = 0.08
    TRAIL_ATR_MULT = 2.0

    entry = px if pd.notna(px) else None
    if entry is None:
        return None, None, None, None

    hard = entry * (1.0 + HARD_STOP_PCT)
    atr_stop = (entry + TRAIL_ATR_MULT * atr) if (pd.notna(atr)) else np.nan
    ma_guard = (ma30 * 1.03) if pd.notna(ma30) else np.nan

    cand = [v for v in [hard, atr_stop, ma_guard] if pd.notna(v)]
    stop = max(cand) if cand else entry * 1.08

    risk = stop - entry
    tgt1 = entry - risk
    tgt2 = entry - 2.0 * risk

    return entry, stop, tgt1, tgt2

def _build_order_block_html(short_trigs, near_shorts):
    """
    Order block for short-side:
      - For TRIG shorts: entry/stop/targets
      - For NEAR shorts: show 'prospective' values
    """
    if not short_trigs and not near_shorts:
        return ""

    css = """
    <style>
      .ordtbl { border-collapse: collapse; width:100%; margin-top:6px; }
      .ordtbl th, .ordtbl td { border-bottom:1px solid #eee; padding:6px 8px; font-size:13px; text-align:left; }
      .ordtbl th { background:#fafafa; }
    </style>
    """
    html = css + "<h4>Short Order Block (proposed)</h4>"

    def _rows(items, label):
        out = []
        for it in items:
            t = it["ticker"]; px = it.get("price", np.nan)
            pivot_low = it.get("pivot_low_10w", np.nan); ma = it.get("ma30", np.nan); atr = it.get("atr", np.nan)
            entry, stop, tgt1, tgt2 = _propose_short_entry_stop_targets(t, px, pivot_low, ma, atr)
            out.append(
                f"<tr><td>{t}</td><td>{label}</td><td>{_fmt_num(px)}</td><td>{_fmt_num(pivot_low)}</td>"
                f"<td>{_fmt_num(entry)}</td><td>{_fmt_num(stop)}</td><td>{_fmt_num(tgt1)}</td><td>{_fmt_num(tgt2)}</td></tr>"
            )
        return "\n".join(out)

    html += """
    <table class="ordtbl">
      <thead>
        <tr>
          <th>Ticker</th><th>Type</th><th>Now</th><th>Pivot Low (10w)</th>
          <th>Entry ≈</th><th>Stop ≥</th><th>Target1 ↓</th><th>Target2 ↓</th>
        </tr>
      </thead>
      <tbody>
    """
    if short_trigs:
        html += _rows(short_trigs, "TRIG")
    if near_shorts:
        html += _rows(near_shorts, "NEAR")
    html += "</tbody></table>"
    html += "<div style='font-size:12px;color:#666;margin-top:6px;'>"
    html += "Short-side rules: entry≈current price once TRIG, stop≥max(hard+8%, MA150+3%, ATR×2), targets at 1R/2R below entry."
    html += "</div>"
    return html

def _build_order_block_text(short_trigs, near_shorts):
    lines = ["SHORT ORDER BLOCK (proposed)"]
    def _lines(items, label):
        for it in items:
            t = it["ticker"]; px = it.get("price", np.nan)
            pivot_low = it.get("pivot_low_10w", np.nan); ma = it.get("ma30", np.nan); atr = it.get("atr", np.nan)
            entry, stop, tgt1, tgt2 = _propose_short_entry_stop_targets(t, px, pivot_low, ma, atr)
            lines.append(
                f"- {label} {t}: now={_fmt_num(px)} pivotLow={_fmt_num(pivot_low)} "
                f"entry≈{_fmt_num(entry)} stop≥{_fmt_num(stop)} targets↓ [{_fmt_num(tgt1)}, {_fmt_num(tgt2)}]"
            )

    if short_trigs:
        _lines(short_trigs, "TRIG")
    if near_shorts:
        _lines(near_shorts, "NEAR")

    lines.append("Short rules: entry≈current price (TRIG), stop≥max(hard+8%, MA150+3%, ATR×2), targets at 1R/2R below entry.")
    return "\n".join(lines)

# ---------------- Charting ----------------
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
    chart_path = os.path.join(CHART_DIR, f"{ticker}_SHORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.tight_layout(pad=0.8); fig.savefig(chart_path, bbox_inches="tight"); plt.close(fig)
    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# ---------------- Main logic ----------------
def run(_config_path="./config.yaml", *, only_tickers=None, test_ease=False, dry_run=False):
    log("Short watcher starting with config: {0}".format(_config_path), level="step")
    cfg, benchmark, sheet_url, service_account_file = load_config(_config_path)
    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # Normalize expected columns
    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker","stage","ma30","asset_class"]:
        if miss not in w.columns: w[miss] = np.nan

    # Stage 4 universe
    shorts = w[w["stage"].isin(["Stage 4 (Downtrend)"])][["ticker","stage","ma30","asset_class"]].copy()
    if "rank" in w.columns: shorts["weekly_rank"] = w["rank"]
    else: shorts["weekly_rank"] = 999999

    if only_tickers:
        filt = set([t.strip().upper() for t in only_tickers])
        shorts = shorts[shorts["ticker"].isin(filt)].copy()

    log(f"Short universe: {len(shorts)} symbols (Stage 4).", level="info")

    if shorts.empty:
        log("No Stage 4 names found in weekly report. Exiting.", level="warn")
        return

    # Benchmarks to ensure RS charts render: equity + crypto
    needs = sorted(set(shorts["ticker"].tolist() + [benchmark, CRYPTO_BENCHMARK]))

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

    short_state = _load_short_state()

    short_trigs, near_shorts = [], []
    info_rows, debug_rows, chart_imgs = [], [], []

    # Test easing via flag or environment
    ease = test_ease or (os.getenv("SHORT_TEST", "0") == "1")
    if ease:
        log("TEST-EASE: lowering thresholds for quick validation.", level="warn")
    _NEAR_HITS_MIN = 1 if ease else NEAR_HITS_MIN
    _INTRABAR_CONFIRM_MIN_ELAPSED = 0 if ease else INTRABAR_CONFIRM_MIN_ELAPSED
    _INTRABAR_VOLPACE_MIN = 0.0 if ease else INTRABAR_VOLPACE_MIN

    log("Evaluating short candidates...", level="step")

    for _, row in shorts.iterrows():
        t = row["ticker"]
        px = px_now(t)
        if np.isnan(px):
            continue

        stage = str(row["stage"]); ma30 = float(row.get("ma30", np.nan))
        weekly_rank = float(row.get("weekly_rank", np.nan))
        pivot_low = last_weekly_pivot_low(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma(t, daily)
        atr = compute_atr(daily, t, n=14)

        closes_n = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
        ma_ok = pd.notna(ma30); pivot_ok = pd.notna(pivot_low)

        elapsed = _elapsed_in_current_bar_minutes(intraday, t) if INTRADAY_INTERVAL == "60m" else None
        pace_intra = intrabar_volume_pace(intraday, t, bar_minutes=60) if INTRADAY_INTERVAL == "60m" else None

        d = {"why": [], "cond": {}, "metrics": {}}
        d["metrics"].update({
            "price": px, "ma30": ma30, "pivot_low_10w": pivot_low, "atr": atr,
            "pace_full_vs50dma": None if pd.isna(pace) else float(pace),
            "pace_intrabar": None if pd.isna(pace_intra) else float(pace_intra),
            "elapsed_min": elapsed,
        })
        d["cond"]["weekly_stage_ok"] = stage.startswith("Stage 4")
        d["cond"]["ma_ok"] = bool(ma_ok)
        d["cond"]["pivot_ok"] = bool(pivot_ok)

        # --- SHORT confirm ---
        short_confirm = False; vol_ok = True; price_ok = False
        if ma_ok or pivot_ok:
            if INTRADAY_INTERVAL == "60m":
                price_ok = _price_below_break_zone(px, pivot_low, ma30)
                vol_ok = (pd.isna(pace_intra) or pace_intra >= _INTRABAR_VOLPACE_MIN)
                short_confirm = price_ok and (elapsed is not None and elapsed >= _INTRABAR_CONFIRM_MIN_ELAPSED) and vol_ok
            else:
                closes_n2 = closes_n
                if closes_n2:
                    price_ok = all(_price_below_break_zone(c, pivot_low, ma30) for c in closes_n2[-CONFIRM_BARS:])
                    short_confirm = price_ok
                if REQUIRE_RISING_BAR_VOL:
                    vols2 = get_last_n_intraday_volumes(intraday, t, n=2)
                    vavg = get_intraday_avg_volume(intraday, t, window=INTRADAY_AVG_VOL_WINDOW)
                    if len(vols2) >= 2 and pd.notna(vavg) and vavg > 0:
                        vol_ok = (vols2[-1] >= INTRADAY_LASTBAR_AVG_MULT * vavg)
                    else:
                        vol_ok = False

        d["cond"].update({
            "short_price_ok": bool(price_ok),
            "short_vol_ok": bool(vol_ok),
            "short_confirm": bool(short_confirm),
            "pace_full_gate_trig": (pd.isna(pace) or pace >= VOL_PACE_MIN_TRIG),
            "pace_full_gate_near": (pd.isna(pace) or pace >= VOL_PACE_MIN_NEAR),
        })

        if not d["cond"]["weekly_stage_ok"]: d["why"].append("Not Stage 4")
        if not d["cond"]["ma_ok"]: d["why"].append("No MA30")
        if not d["cond"]["pivot_ok"]: d["why"].append("No 10w pivot low")

        # --- NEAR-short flag ---
        near_now = False
        if d["cond"]["weekly_stage_ok"] and (ma_ok or pivot_ok) and pd.notna(px):
            if _near_short_zone(px, pivot_low, ma30):
                near_now = True
        d["cond"]["near_now"] = bool(near_now)

        # --- promotion state ---
        ts_key = t
        st = short_state.get(ts_key, {
            "short_state":"IDLE", "near_hits":[], "cooldown":0
        })

        # NEAR hits
        st["near_hits"], near_count = _update_hits(st.get("near_hits", []), near_now, NEAR_HITS_WINDOW)
        if st.get("cooldown", 0) > 0: st["cooldown"] = int(st["cooldown"]) - 1
        short_state_now = st.get("short_state", "IDLE")
        if short_state_now == "IDLE" and near_now: short_state_now = "NEAR"
        elif short_state_now in ("IDLE","NEAR") and near_count >= _NEAR_HITS_MIN: short_state_now = "ARMED"
        elif short_state_now == "ARMED" and short_confirm and vol_ok and (pd.isna(pace) or pace >= VOL_PACE_MIN_TRIG):
            short_state_now = "TRIGGERED"; st["cooldown"] = COOLDOWN_SCANS
        elif st["cooldown"] > 0 and not near_now: short_state_now = "COOLDOWN"
        elif st["cooldown"] == 0 and not near_now and not short_confirm: short_state_now = "IDLE"
        st["short_state"] = short_state_now

        short_state[ts_key] = st

        # Emit by state
        if st["short_state"] == "TRIGGERED" and (pd.isna(pace) or pace >= VOL_PACE_MIN_TRIG):
            short_trigs.append({
                "ticker": t, "price": px, "pivot_low_10w": pivot_low,
                "pace": None if pd.isna(pace) else float(pace),
                "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank, "atr": atr
            })
            short_state[t]["short_state"] = "COOLDOWN"
        elif st["short_state"] in ("NEAR","ARMED"):
            if (pd.isna(pace) or pace >= VOL_PACE_MIN_NEAR):
                near_shorts.append({
                    "ticker": t, "price": px, "pivot_low_10w": pivot_low,
                    "pace": None if pd.isna(pace) else float(pace),
                    "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank,
                    "atr": atr
                })

        info_rows.append({
            "ticker": t, "stage": stage, "price": px, "ma30": ma30,
            "pivot_low_10w": pivot_low,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace), 2),
            "short_confirm": short_confirm, "short_vol_ok": vol_ok,
            "weekly_rank": weekly_rank,
            "short_state": st["short_state"],
        })
        debug_rows.append({"ticker": t, **d["metrics"],
                           **{f"cond_{k}": v for k,v in d["cond"].items()},
                           "short_state": st["short_state"]})

    log(f"Scan done. Shorts → NEAR:{len(near_shorts)} TRIG:{len(short_trigs)}", level="info")

    # -------- Ranking & charts --------
    short_trigs.sort(key=short_sort_key)
    near_shorts.sort(key=near_sort_key)

    charts_added = 0; chart_imgs = []
    for item in short_trigs:
        if charts_added >= MAX_CHARTS_PER_EMAIL: break
        t = item["ticker"]
        bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
        path, data_uri = make_tiny_chart_png(t, bmk, daily)
        if data_uri:
            chart_imgs.append((t, data_uri)); charts_added += 1
    if charts_added < MAX_CHARTS_PER_EMAIL:
        for item in near_shorts:
            if charts_added >= MAX_CHARTS_PER_EMAIL: break
            t = item["ticker"]
            bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
            path, data_uri = make_tiny_chart_png(t, bmk, daily)
            if data_uri:
                chart_imgs.append((t, data_uri)); charts_added += 1

    log(f"Charts prepared: {len(chart_imgs)}", level="debug")

    # -------- Build Email --------
    info_df = pd.DataFrame(info_rows)
    if not info_df.empty:
        info_df["stage_rank"] = info_df["stage"].apply(stage_order)
        info_df["weekly_rank"] = pd.to_numeric(info_df["weekly_rank"], errors="coerce").fillna(999999).astype(int)
        info_df = info_df.sort_values(["weekly_rank","stage_rank","ticker"]).drop(columns=["stage_rank"])

    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} shorts.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "TRIG":
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                entry, stop, tgt1, tgt2 = _propose_short_entry_stop_targets(
                    it["ticker"], it.get("price", np.nan),
                    it.get("pivot_low_10w", np.nan),
                    it.get("ma30", np.nan),
                    it.get("atr", np.nan)
                )
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(TRIG short, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(tgt1)}, {_fmt_num(tgt2)}], "
                    f"{it.get('stage','')}, weekly {wr_str}, pace {pace_str})</li>"
                )
            else:
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                entry, stop, tgt1, tgt2 = _propose_short_entry_stop_targets(
                    it["ticker"], it.get("price", np.nan),
                    it.get("pivot_low_10w", np.nan),
                    it.get("ma30", np.nan),
                    it.get("atr", np.nan)
                )
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(NEAR short, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(tgt1)}, {_fmt_num(tgt2)}], "
                    f"{it.get('stage','')}, weekly {wr_str})</li>"
                )
        return "<ol>" + "\n".join(lis) + "</ol>"

    charts_html = ""
    if chart_imgs:
        charts_html = "<h4>Charts (Price + SMA150, RS / benchmark)</h4>"
        for t, data_uri in chart_imgs:
            charts_html += f"""
            <div style="display:inline-block;margin:6px 8px 10px 0;vertical-align:top;text-align:center;">
              <img src="{data_uri}" alt="{t}" style="border:1px solid #eee;border-radius:6px;max-width:320px;">
              <div style="font-size:12px;color:#555;margin-top:3px;">{t}</div>
            </div>
            """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <h3>Weinstein Short Intraday Watch — {now}</h3>
    <p><i>
      SHORT TRIGGER: Weekly Stage 4 (Downtrend) + breakdown under ~10-week pivot low and/or 30-wk MA proxy (SMA150),
      by ≈{MIN_BREAKDOWN_PCT*100:.1f}% with volume pace ≥ {VOL_PACE_MIN_TRIG:.1f}× and intrabar checks (≥{INTRABAR_CONFIRM_MIN_ELAPSED} min, pace ≥ {INTRABAR_VOLPACE_MIN:.1f}×).<br>
      NEAR-SHORT: Stage 4 + price hanging just above the pivot/MA breakdown zone,
      volume pace ≥ {VOL_PACE_MIN_NEAR:.1f}×.
    </i></p>
    """

    html += f"""
    <h4>Short Triggers (ranked)</h4>
    {bullets(short_trigs, "TRIG")}
    <h4>Near Short Setups (ranked)</h4>
    {bullets(near_shorts, "NEAR")}
    {charts_html}
    """

    # ------ ORDER BLOCK (HTML + TEXT) ------
    order_block_html = _build_order_block_html(short_trigs, near_shorts)
    if order_block_html:
        html += order_block_html

    html += f"""
    <h4>Snapshot</h4>
    {info_df.to_html(index=False) if not info_df.empty else "<p>No snapshot data.</p>"}
    """

    # Plain text body
    def _lines(items, kind):
        out = []
        for i, it in enumerate(items, 1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            entry, stop, tgt1, tgt2 = _propose_short_entry_stop_targets(
                it["ticker"], it.get("price", np.nan),
                it.get("pivot_low_10w", np.nan),
                it.get("ma30", np.nan),
                it.get("atr", np.nan)
            )
            if kind == "TRIG":
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} "
                    f"(TRIG short, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(tgt1)}, {_fmt_num(tgt2)}], "
                    f"{it.get('stage','')}, weekly {wr_str}, pace {pace_str})"
                )
            else:
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} "
                    f"(NEAR short, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(tgt1)}, {_fmt_num(tgt2)}], "
                    f"{it.get('stage','')}, weekly {wr_str})"
                )
        return "\n".join(out) if out else f"No {kind} shorts."

    text = (
        f"Weinstein Short Intraday Watch — {now}\n\n"
        f"SHORT TRIGGERS (ranked):\n{_lines(short_trigs,'TRIG')}\n\n"
        f"NEAR SHORT SETUPS (ranked):\n{_lines(near_shorts,'NEAR')}\n\n"
    )

    order_block_text = _build_order_block_text(short_trigs, near_shorts)
    if order_block_text:
        text += "\n" + order_block_text + "\n"

    # Persist state
    _save_short_state(short_state)

    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        log("Sending email...", level="step")
        subject_counts = f"{len(short_trigs)} TRIG / {len(near_shorts)} NEAR"
        send_email(
            subject=f"Short Intraday Watch — {subject_counts}",
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
    ap.add_argument("--only", type=str, default="", help="comma list of tickers to restrict evaluation (e.g. CRM,FDS)")
    ap.add_argument("--test-ease", action="store_true", help="enable trigger easing for testing (or set SHORT_TEST=1)")
    ap.add_argument("--dry-run", action="store_true", help="don’t send email")
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()] if args.only else None

    log(f"Short watcher starting with config: {args.config}", level="step")
    try:
        run(
            _config_path=args.config,
            only_tickers=only,
            test_ease=args.test_ease,
            dry_run=args.dry_run,
        )
        log("Short tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
