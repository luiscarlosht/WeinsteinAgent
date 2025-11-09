# === weinstein_intraday_watcher.py ===
import os, io, json, math, time, base64, yaml, argparse, sys
from datetime import datetime, timezone, timedelta

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

# ---------------- CLI ----------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config.yaml")
    p.add_argument("--quiet", action="store_true", help="reduce console noise")
    return p.parse_args()

# ---------------- Config / IO ----------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    return cfg, benchmark

def newest_weekly_csv():
    files = [f for f in os.listdir(WEEKLY_OUTPUT_DIR) if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")]
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

def save_positions(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

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
        return max(0, int((datetime.utcnow() - last_bar_start).total_seconds() // 60))
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
    bars = weeks * 5
    if _is_crypto(ticker):
        bars = weeks * 7  # crypto trades 7 days a week
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
        # Equity session: 13:30–20:00 UTC (6.5h)
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

def get_intraday_avg_volume(intraday_df, ticker, window=20):
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

# ---- Summary HTML helpers (unchanged) ----
def _money(x): return f"${x:,.2f}" if (x is not None and pd.notna(x)) else ""
def _pct(x):   return f"{x:.2f}%" if (x is not None and pd.notna(x)) else ""

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
        ("Average % Gain",       _pct(m["avg_pct"]),    cls(m["avg_pct"])),
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
      .tab-holdings th { text-align:left; background:#fafafa; }
    </style>
    """
    return css + f"""
    <div class="blk">
      <h3>Weinstein Weekly – Summary</h3>
      <table class="sumtbl">
        <tbody>
          {tr}
        </tbody>
      </table>
    </div>
    """

def _format_holdings_table(df: pd.DataFrame) -> str:
    for c in ["industry","sector"]:
        if c not in df.columns: df[c] = np.nan
    cols = [
        "Symbol","Description","industry","sector","Quantity","Last Price","Current Value",
        "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent","Recommendation"
    ]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    num = df[cols].copy()
    d = df[cols].copy()
    def money(x): return _money(x)
    def pctv(x):  return (f"{float(x):.2f}%" if pd.notna(x) else "")
    d["Last Price"] = d["Last Price"].apply(money)
    d["Current Value"] = d["Current Value"].apply(money)
    d["Cost Basis Total"] = d["Cost Basis Total"].apply(money)
    d["Average Cost Basis"] = d["Average Cost Basis"].apply(money)
    d["Total Gain/Loss Dollar"] = d["Total Gain/Loss Dollar"].apply(money)
    d["Total Gain/Loss Percent"] = d["Total Gain/Loss Percent"].apply(pctv)
    def rec_badge(s):
        s = str(s or "")
        if s.upper().startswith("SELL"): return "<span class='rec-badge rec-sell'>SELL</span>"
        if s.upper().startswith("HOLD (STRONG"): return "<span class='rec-badge rec-strong'>HOLD (Strong)</span>"
        if s.upper().startswith("HOLD"): return "<span class='rec-badge rec-hold'>HOLD</span>"
        return s
    d["Recommendation"] = d["Recommendation"].apply(rec_badge)
    th = "".join([f"<th>{c}</th>" for c in cols])
    rows = []
    for i in range(len(d)):
        r = d.iloc[i]; rn = num.iloc[i]
        def sign_cls(val):
            if pd.isna(val): return "num-neu"
            return "num-pos" if val > 0 else ("num-neg" if val < 0 else "num-neu")
        gl_d_cls = sign_cls(rn["Total Gain/Loss Dollar"])
        gl_p_cls = sign_cls(rn["Total Gain/Loss Percent"])
        tds = []
        for c in cols:
            if c == "Total Gain/Loss Dollar":
                tds.append(f"<td class='{gl_d_cls}'>{r[c]}</td>")
            elif c == "Total Gain/Loss Percent":
                tds.append(f"<td class='{gl_p_cls}'>{r[c]}</td>")
            else:
                val = r[c] if pd.notna(r[c]) else ""
                tds.append(f"<td>{val}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")
    body = "\n".join(rows)
    return f"""
    <div class="blk">
      <h3>Per-position Snapshot</h3>
      <table class="tab-holdings">
        <thead><tr>{th}</tr></thead>
        <tbody>
          {body}
        </tbody>
      </table>
    </div>
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
    ax1.legend(lines1
