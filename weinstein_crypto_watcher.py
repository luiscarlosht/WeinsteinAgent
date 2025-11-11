#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Crypto Watcher — mirrors Intraday Watcher behavior for crypto

Features:
- BUY / NEAR / SELL trigger engine (60m by default) using pivot~10w + 30-wk proxy (SMA150 daily)
- Colored per-position snapshot (worst → best) for CryptoHoldings tab
- Ranked bullets, tiny charts (price + SMA150 + RS vs BTC), HTML save + email
- Works even if CryptoHoldings tab is missing
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
CRYPTO_BENCHMARK  = "BTC-USD"   # RS reference
BENCHMARK_DEFAULT = "BTC-USD"

INTRADAY_INTERVAL = "60m"       # 24/7 assets — 60m works well
LOOKBACK_DAYS = 90
PIVOT_LOOKBACK_WEEKS = 10
VOL_PACE_MIN = 1.10             # crypto volume normalization is noisier
BUY_DIST_ABOVE_MA_MIN = 0.00

CONFIRM_BARS = 2
MIN_BREAKOUT_PCT = 0.004
REQUIRE_RISING_BAR_VOL = False  # intrabar gate used instead
INTRADAY_AVG_VOL_WINDOW = 20
INTRADAY_LASTBAR_AVG_MULT = 1.10

NEAR_BELOW_PIVOT_PCT = 0.003
NEAR_VOL_PACE_MIN = 0.90

HARD_STOP_PCT = 0.15            # crypto wider default
TRAIL_ATR_MULT = 2.5

STATE_FILE = "./state/crypto_positions.json"        # optional user-edited
INTRADAY_STATE_FILE = "./state/crypto_triggers.json"
CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

PRICE_WINDOW_DAYS = 260
SMA_DAYS = 150

# Default universe if not passed with --only and not derivable from weekly CSV
DEFAULT_UNIVERSE = [
    "BTC-USD","ETH-USD","SOL-USD","ADA-USD","AVAX-USD","DOGE-USD",
    "LINK-USD","MATIC-USD","ATOM-USD","LTC-USD","XRP-USD","TON-USD",
]

VERBOSE = True

# ---- Optional Google Sheets pull (Signals & Holdings) ----
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

TAB_SIGNALS      = "Signals"           # optional, for future: crypto signals
TAB_CRYPTO_HOLD  = "CryptoHoldings"    # expected holdings tab name
TAB_MAPPING      = "Mapping"           # optional (not needed for -USD pairs)

# ---------- Small helpers ----------
def _ts(): return datetime.now().strftime("%H:%M:%S")
def log(msg, *, level="info"):
    if not VERBOSE and level == "debug": return
    prefix = {"info":"•", "ok":"✅", "step":"▶️", "warn":"⚠️", "err":"❌", "debug":"··"}.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)

# 60m-specific confirmation easing (BUY/SELL)
INTRABAR_CONFIRM_MIN_ELAPSED = 40
INTRABAR_VOLPACE_MIN = 1.10

SELL_NEAR_ABOVE_MA_PCT = 0.005
SELL_BREAK_PCT = 0.005
SELL_INTRABAR_CONFIRM_MIN_ELAPSED = 40
SELL_INTRABAR_VOLPACE_MIN = 1.10

# ---------------- Config / IO ----------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")
    return cfg, sheet_url, svc_file

def newest_weekly_csv():
    files = [f for f in os.listdir(WEEKLY_OUTPUT_DIR)
             if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No weekly CSV found in ./output. Run weekly first.")
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])

def load_weekly_report():
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path

def _load_state():
    os.makedirs(os.path.dirname(INTRADAY_STATE_FILE), exist_ok=True)
    if os.path.exists(INTRADAY_STATE_FILE):
        with open(INTRADAY_STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_state(st):
    with open(INTRADAY_STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

# ---------------- Data helpers ----------------
def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)): return np.nan
        return a / b
    except Exception:
        return np.nan

def _is_crypto(sym: str) -> bool: return (sym or "").upper().endswith("-USD")

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
        try: sub = daily_df.xs(t, axis=1, level=1)
        except KeyError: return np.nan
    else:
        sub = daily_df
    if not set(["High","Low","Close"]).issubset(set(sub.columns)): return np.nan
    h, l, c = sub["High"], sub["Low"], sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return float(atr.dropna().iloc[-1]) if len(atr.dropna()) else np.nan

def last_weekly_pivot_high(ticker, daily_df, weeks=PIVOT_LOOKBACK_WEEKS):
    bars = weeks * 7  # crypto 24/7
    if isinstance(daily_df.columns, pd.MultiIndex):
        try: highs = daily_df[("High", ticker)]
        except KeyError: return np.nan
    else:
        highs = daily_df["High"]
    highs = highs.dropna().tail(bars)
    return float(highs.max()) if len(highs) else np.nan

def volume_pace_today_vs_50dma(ticker, daily_df):
    if isinstance(daily_df.columns, pd.MultiIndex):
        try: v = daily_df[("Volume", ticker)].copy()
        except KeyError: return np.nan
    else:
        v = daily_df["Volume"].copy()
    if v.empty: return np.nan
    v50 = v.rolling(50).mean().iloc[-2] if len(v) > 50 else np.nan
    today_vol = v.iloc[-1]
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = max(0.0, (now - day_start).total_seconds())
    fraction = min(1.0, max(0.05, elapsed / (24*3600.0)))
    est_full = today_vol / fraction if fraction > 0 else today_vol
    return float(_safe_div(est_full, v50)) if pd.notna(v50) and v50 > 0 else np.nan

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

def get_last_n_intraday_closes(intraday_df, ticker, n=2):
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try: s = intraday_df[("Close", ticker)].dropna()
        except KeyError: return []
    else:
        s = intraday_df["Close"].dropna()
    return list(map(float, s.tail(n).values))

def intrabar_volume_pace(intraday_df, ticker, avg_window=INTRADAY_AVG_VOL_WINDOW, bar_minutes=60):
    try:
        v = intraday_df[("Volume", ticker)].dropna() if isinstance(intraday_df.columns, pd.MultiIndex) else intraday_df["Volume"].dropna()
    except Exception:
        return np.nan
    if len(v) < max(avg_window, 2): return np.nan
    last_bar_vol = float(v.iloc[-1])
    avg_bar_vol = float(v.tail(avg_window).mean())
    elapsed = _elapsed_in_current_bar_minutes(intraday_df, ticker)
    frac = min(1.0, max(0.05, elapsed / float(bar_minutes)))
    est_full = last_bar_vol / frac if frac > 0 else last_bar_vol
    return float(_safe_div(est_full, avg_bar_vol))

# ---------------- Holdings helpers (CryptoHoldings tab) ----------------
def _coerce_num(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int,float,np.number)): return float(s)
    x = str(s).replace("$","").replace(",","").strip()
    if x.endswith("%"): x = x[:-1]
    try: return float(x)
    except: return np.nan

def read_sheet_tab(sheet_url, svc_file, tab_name):
    if not gspread: return None
    try:
        creds = Credentials.from_service_account_file(svc_file, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ])
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(tab_name)
        vals = ws.get_all_values()
        if not vals: return pd.DataFrame()
        header, rows = vals[0], vals[1:]
        df = pd.DataFrame(rows, columns=[h.strip() for h in header])
        return df
    except Exception:
        return None

def load_crypto_holdings(sheet_url, svc_file):
    df = read_sheet_tab(sheet_url, svc_file, TAB_CRYPTO_HOLD)
    if df is None or df.empty:
        log(f"Sheet load failure for crypto holdings: {TAB_CRYPTO_HOLD}", level="warn")
        return pd.DataFrame()
    # Expected columns: Symbol, Quantity, Average Cost Basis (or Cost Basis Total)
    out = pd.DataFrame()
    out["Symbol"] = df.get("Symbol", "")
    out["Quantity"] = df.get("Quantity", "").map(_coerce_num)
    if "Average Cost Basis" in df.columns:
        out["Average Cost Basis"] = df["Average Cost Basis"].map(_coerce_num)
        out["Cost Basis Total"] = out["Quantity"] * out["Average Cost Basis"]
    elif "Cost Basis Total" in df.columns:
        out["Cost Basis Total"] = df["Cost Basis Total"].map(_coerce_num)
        out["Average Cost Basis"] = out["Cost Basis Total"] / out["Quantity"].replace(0,np.nan)
    else:
        out["Average Cost Basis"] = np.nan
        out["Cost Basis Total"] = np.nan
    return out.fillna("")

# ---- Colored snapshot like equities ----
def _money(x): return f"${x:,.2f}" if (x is not None and pd.notna(x)) else "—"
def _pct(x):   return f"{x:.2f}%" if (x is not None and pd.notna(x)) else "—"

def build_crypto_snapshot(hold_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich with live prices & gains, return standard columns."""
    if hold_df is None or hold_df.empty:
        return pd.DataFrame(columns=[
            "Symbol","Description","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent","Recommendation"
        ])
    df = hold_df.copy()
    syms = [s for s in df["Symbol"].astype(str) if s]
    prices = {}
    if syms:
        try:
            hist = yf.download(sorted(set(syms)), period="5d", interval="1d", auto_adjust=True, progress=False)
            if isinstance(hist.columns, pd.MultiIndex) and ("Close" in hist.columns.get_level_values(0)):
                close = hist["Close"]
                for s in syms:
                    try:
                        prices[s] = float(close[s].dropna().iloc[-1])
                    except Exception:
                        pass
            elif "Close" in hist.columns:
                prices[syms[0]] = float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass

    df["Last Price"] = df["Symbol"].map(prices).astype(float)
    df["Current Value"] = (df["Quantity"].map(_coerce_num) * df["Last Price"]).astype(float)
    cb = df["Cost Basis Total"].map(_coerce_num)
    if cb.isna().all():
        cb = df["Quantity"].map(_coerce_num) * df["Average Cost Basis"].map(_coerce_num)
    df["Cost Basis Total"] = cb.astype(float)
    df["Average Cost Basis"] = df["Average Cost Basis"].map(_coerce_num)
    df["Total Gain/Loss Dollar"] = (df["Current Value"] - df["Cost Basis Total"]).astype(float)
    df["Total Gain/Loss Percent"] = np.where(
        df["Average Cost Basis"].fillna(0) > 0,
        (df["Last Price"] / df["Average Cost Basis"] - 1) * 100.0,
        np.nan
    )
    df["Description"] = ""
    # Simple rec rule for crypto (can tune later)
    def rec(row):
        pct = row.get("Total Gain/Loss Percent", np.nan)
        return "SELL" if (pd.notna(pct) and pct <= -15.0) else "HOLD"
    df["Recommendation"] = df.apply(rec, axis=1)

    cols = ["Symbol","Description","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent","Recommendation"]
    return df[cols]

def _snapshot_html(merged: pd.DataFrame) -> str:
    df = merged.copy()
    df["Total Gain/Loss Percent"] = pd.to_numeric(df["Total Gain/Loss Percent"], errors="coerce")
    df["Total Gain/Loss Dollar"]  = pd.to_numeric(df["Total Gain/Loss Dollar"], errors="coerce")
    df = df.sort_values(["Total Gain/Loss Percent","Total Gain/Loss Dollar"], ascending=[True, True])

    cols = ["Symbol","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis",
            "Total Gain/Loss Dollar","Total Gain/Loss Percent","Recommendation"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan

    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = []
    for _, r in df.iterrows():
        gain_d = r["Total Gain/Loss Dollar"]; gain_p = r["Total Gain/Loss Percent"]
        cls_d = "num-pos" if pd.notna(gain_d) and gain_d > 0 else ("num-neg" if pd.notna(gain_d) and gain_d < 0 else "num-neu")
        cls_p = "num-pos" if pd.notna(gain_p) and gain_p > 0 else ("num-neg" if pd.notna(gain_p) and gain_p < 0 else "num-neu")
        row = [
            str(r.get("Symbol","")),
            f"{float(r.get('Quantity',np.nan)):.6f}" if pd.notna(r.get("Quantity")) else "—",
            _money(r.get("Last Price",np.nan)),
            _money(r.get("Current Value",np.nan)),
            _money(r.get("Cost Basis Total",np.nan)),
            _money(r.get("Average Cost Basis",np.nan)),
            f"<span class='{cls_d}'>{_money(gain_d)}</span>",
            f"<span class='{cls_p}'>{_pct(gain_p)}</span>",
            str(r.get("Recommendation","")),
        ]
        rows_html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    css = """
      <style>
        .tab-h { border-collapse: collapse; width:100%; }
        .tab-h th, .tab-h td { padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 13px; }
        .tab-h th { text-align:left; background:#fafafa; position:sticky; top:0; }
        .num-pos { color:#106b21; font-weight:600; }
        .num-neg { color:#8a1111; font-weight:600; }
        .num-neu { color:#444; }
      </style>
    """
    return css + f"""
      <h4>Crypto — Per-position Snapshot (worst → best)</h4>
      <table class="tab-h">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    """

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

# ---------------- Logic helpers ----------------
def stage_order(stage: str) -> int:
    # Weekly CSV for crypto may tag stages similarly; default to neutral
    if isinstance(stage, str):
        if stage.startswith("Stage 2"): return 0
        if stage.startswith("Stage 1"): return 1
    return 9

def _price_below_ma(px, ma): return pd.notna(px) and pd.notna(ma) and px <= ma * (1.0 - SELL_BREAK_PCT)
def _near_sell_zone(px, ma):
    if pd.isna(px) or pd.isna(ma): return False
    return (px >= ma * (1.0 - SELL_BREAK_PCT)) and (px <= ma * (1.0 + SELL_NEAR_ABOVE_MA_PCT))

# ---------------- Main run ----------------
def run(_config_path="./config.yaml", *, only_tickers=None, test_ease=False, dry_run=False):
    log("Crypto watcher starting with config: {0}".format(_config_path), level="step")
    cfg, sheet_url, service_account_file = load_config(_config_path)
    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker","stage","ma30","asset_class","rank"]:
        if miss not in w.columns: w[miss] = np.nan
    # Crypto focus: prefer rows tagged as crypto; fallback to tickers ending with -USD
    focus = w[(w["asset_class"].astype(str).str.contains("crypto", case=False, na=False)) | (w["ticker"].astype(str).str.endswith("-USD"))][["ticker","stage","ma30","rank"]].copy()
    if focus.empty:
        focus = pd.DataFrame({"ticker": DEFAULT_UNIVERSE, "stage": "", "ma30": np.nan, "rank": 999999})

    if only_tickers:
        filt = set([t.strip().upper() for t in only_tickers])
        focus = focus[focus["ticker"].isin(filt)].copy()

    # Ensure benchmark present
    needs = sorted(set(focus["ticker"].tolist() + [CRYPTO_BENCHMARK]))

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

    trigger_state = _load_state()

    buy_signals, near_signals, sell_triggers, info_rows = [], [], [], []

    # Easing
    ease = test_ease or (os.getenv("INTRADAY_TEST", "0") == "1")
    _ELAPSED = 0 if ease else INTRABAR_CONFIRM_MIN_ELAPSED
    _PACE_MIN = 0.0 if ease else INTRABAR_VOLPACE_MIN
    _SELL_ELAPSED = 0 if ease else SELL_INTRABAR_CONFIRM_MIN_ELAPSED
    _SELL_PACE_MIN = 0.0 if ease else SELL_INTRABAR_VOLPACE_MIN

    log("Evaluating candidates...", level="step")
    for _, row in focus.iterrows():
        t = row["ticker"]
        if t == CRYPTO_BENCHMARK:  # skip benchmark
            continue
        px = px_now(t)
        if np.isnan(px): continue

        stage = str(row.get("stage",""))
        ma30 = float(row.get("ma30", np.nan))
        weekly_rank = float(row.get("rank", np.nan))
        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma(t, daily)
        closes_n = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
        elapsed = _elapsed_in_current_bar_minutes(intraday, t)
        pace_intra = intrabar_volume_pace(intraday, t, bar_minutes=60)

        # BUY confirm
        confirm = False; price_ok = False; vol_ok = True
        if pd.notna(ma30) and pd.notna(pivot) and closes_n:
            def _price_ok(c):
                return (c >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and (c >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN))
            last_c = closes_n[-1]
            price_ok = _price_ok(last_c)
            vol_ok = (pd.isna(pace_intra) or pace_intra >= _PACE_MIN)
            confirm = price_ok and (elapsed is not None and elapsed >= _ELAPSED) and vol_ok

        # NEAR
        near_now = False
        if pd.notna(ma30) and pd.notna(pivot) and pd.notna(px):
            above_ma = px >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
            if above_ma:
                if (px >= pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)) and (px < pivot * (1.0 + MIN_BREAKOUT_PCT)):
                    near_now = True
                elif (px >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not confirm:
                    near_now = True

        # SELL confirm
        sell_confirm = False; sell_price_ok = False; sell_vol_ok = True
        if pd.notna(ma30) and pd.notna(px):
            sell_price_ok = _price_below_ma(px, ma30)
            sell_vol_ok = (pd.isna(pace_intra) or (pace_intra >= _SELL_PACE_MIN))
            sell_confirm = bool(sell_price_ok and (elapsed is not None and elapsed >= _SELL_ELAPSED) and sell_vol_ok)

        # Promote (simple — crypto watcher doesn’t keep NEAR hits window; intraday scans are frequent)
        if confirm and (pd.isna(pace) or pace >= VOL_PACE_MIN):
            buy_signals.append({
                "ticker": t, "price": px, "pivot": pivot,
                "pace": None if pd.isna(pace) else float(pace),
                "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank
            })
        elif near_now and (pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN):
            near_signals.append({
                "ticker": t, "price": px, "pivot": pivot,
                "pace": None if pd.isna(pace) else float(pace),
                "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank
            })

        if sell_confirm:
            sell_triggers.append({
                "ticker": t, "price": px, "ma30": ma30, "stage": stage,
                "weekly_rank": weekly_rank, "pace": None if pd.isna(pace) else float(pace)
            })

        info_rows.append({
            "ticker": t, "stage": stage, "price": px, "ma30": ma30, "pivot10w": pivot,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace), 2),
            "two_bar_confirm": confirm, "weekly_rank": weekly_rank,
        })

    log(f"Scan done. Raw counts → BUY:{len(buy_signals)} NEAR:{len(near_signals)} SELLTRIG:{len(sell_triggers)}", level="info")

    # -------- Charts --------
    def stage_order_key(s): return 0 if str(s).startswith("Stage 2") else (1 if str(s).startswith("Stage 1") else 9)
    buy_signals.sort(key=lambda it: (int(it.get("weekly_rank") or 999999), stage_order_key(it.get("stage",""))))
    near_signals.sort(key=lambda it: (int(it.get("weekly_rank") or 999999), stage_order_key(it.get("stage",""))))
    sell_triggers.sort(key=lambda it: (int(it.get("weekly_rank") or 999999), stage_order_key(it.get("stage",""))))

    charts_added = 0; chart_imgs = []
    for bucket in (buy_signals, near_signals):
        for it in bucket:
            if charts_added >= MAX_CHARTS_PER_EMAIL: break
            t = it["ticker"]
            path, data_uri = make_tiny_chart_png(t, CRYPTO_BENCHMARK, daily)
            if data_uri:
                chart_imgs.append((t, data_uri)); charts_added += 1
        if charts_added >= MAX_CHARTS_PER_EMAIL: break
    log(f"Charts prepared: {len(chart_imgs)}", level="debug")

    # -------- Holdings snapshot (CryptoHoldings tab) --------
    snapshot_html = ""
    hold_df = load_crypto_holdings(sheet_url, service_account_file) if (sheet_url and service_account_file) else pd.DataFrame()
    if not hold_df.empty:
        snap = build_crypto_snapshot(hold_df)
        snapshot_html = _snapshot_html(snap)

    # -------- Build Email --------
    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} signals.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan); ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None); pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} (↓ MA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})</li>")
            else:
                pace_val = it.get("pace", None); pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} (pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']}, weekly {wr_str})</li>")
        return "<ol>" + "\n".join(lis) + "</ol>"

    charts_html = ""
    if chart_imgs:
        charts_html = "<h4>Charts (Price + SMA150, RS vs BTC)</h4>"
        for t, data_uri in chart_imgs:
            charts_html += f"""
            <div style="display:inline-block;margin:6px 8px 10px 0;vertical-align:top;text-align:center;">
              <img src="{data_uri}" alt="{t}" style="border:1px solid #eee;border-radius:6px;max-width:320px;">
              <div style="font-size:12px;color:#555;margin-top:3px;">{t}</div>
            </div>
            """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <h3>Weinstein Crypto Watch — {now}</h3>
    <p><i>
      BUY: Break over ~10-week pivot & SMA150 (30-wk proxy) by {MIN_BREAKOUT_PCT*100:.1f}%,
      intrabar pace ≥ {INTRABAR_VOLPACE_MIN:.2f}× and ≥{INTRABAR_CONFIRM_MIN_ELAPSED} min elapsed (60m bars).<br>
      NEAR: within {NEAR_BELOW_PIVOT_PCT*100:.1f}% below pivot or first close over pivot, not fully confirmed yet.<br>
      SELL: Crack below SMA150 by {SELL_BREAK_PCT*100:.1f}% with ≥{SELL_INTRABAR_CONFIRM_MIN_ELAPSED} min elapsed & intrabar pace ≥ {SELL_INTRABAR_VOLPACE_MIN:.2f}×.
    </i></p>

    <h4>Buy Triggers (ranked)</h4>
    {bullets(buy_signals, "BUY")}
    <h4>Near-Triggers (ranked)</h4>
    {bullets(near_signals, "NEAR")}
    <h4>Sell Triggers (ranked)</h4>
    {bullets(sell_triggers, "SELLTRIG")}
    {charts_html}
    """

    if snapshot_html:
        html += "<hr/>" + snapshot_html

    text = f"Weinstein Crypto Watch — {now}\n\n"
    text += "BUY (ranked):\n" + ("\n".join([f"{i+1}. {it['ticker']} @ {it['price']:.2f} (pivot {it['pivot']:.2f})"
                                           for i,it in enumerate(buy_signals)]) or "None") + "\n\n"
    text += "NEAR (ranked):\n" + ("\n".join([f"{i+1}. {it['ticker']} @ {it['price']:.2f} (pivot {it['pivot']:.2f})"
                                            for i,it in enumerate(near_signals)]) or "None") + "\n\n"
    text += "SELL TRIGGERS (ranked):\n" + ("\n".join([f"{i+1}. {it['ticker']} @ {it['price']:.2f}"
                                                    for i,it in enumerate(sell_triggers)]) or "None") + "\n"

    # Save and send
    os.makedirs("./output", exist_ok=True)
    html_path = os.path.join("./output", f"crypto_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    subject_counts = f"{len(buy_signals)} BUY / {len(near_signals)} NEAR / {len(sell_triggers)} SELL-TRIG"
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

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--only", type=str, default="", help="comma list of crypto tickers (e.g. BTC-USD,ETH-USD)")
    ap.add_argument("--test-ease", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()] if args.only else None

    try:
        run(
            _config_path=args.config,
            only_tickers=only,
            test_ease=args.test_ease,
            dry_run=args.dry_run,
        )
        log("Crypto tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
