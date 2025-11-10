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

# ================= Tunables =================
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
BENCHMARK_DEFAULT = "SPY"
CRYPTO_BENCHMARK  = "BTC-USD"  # for RS in weekly (intraday uses price/MA/pivot only)

INTRADAY_INTERVAL = "60m"     # '60m' or '30m'
LOOKBACK_DAYS = 60
PIVOT_LOOKBACK_WEEKS = 10

VOL_PACE_MIN = 1.30                 # for confirmed BUY
BUY_DIST_ABOVE_MA_MIN = 0.00

CONFIRM_BARS = 2
MIN_BREAKOUT_PCT = 0.004
REQUIRE_RISING_BAR_VOL = True
INTRADAY_AVG_VOL_WINDOW = 20
INTRADAY_LASTBAR_AVG_MULT = 1.20

NEAR_BELOW_PIVOT_PCT = 0.003
NEAR_VOL_PACE_MIN = 1.00            # <-- keep defined and available (fixes NameError)

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

# ---------- Weekly logic tunables for crypto block ----------
WEEKS_LOOKBACK = 180
MA_WEEKS = 30
MA10_WEEKS = 10
RS_MA_WEEKS = 30
SLOPE_WINDOW = 5
NEAR_MA_BAND = 0.05
TOP_N_CHARTS = 20  # tiny weekly charts inside crypto table

# ---------- Optional Google Sheets (to discover crypto from Signals) ----------
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

TAB_SIGNALS = "Signals"
TAB_MAPPING = "Mapping"

# ---------------- CLI ----------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config.yaml")
    p.add_argument("--quiet", action="store_true", help="reduce console noise")
    return p.parse_args()

# ---------------- Logging ----------------
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg, *, level="info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {"info":"•", "ok":"✅", "step":"▶️", "warn":"⚠️", "err":"❌", "debug":"··"}.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)

# ---------------- Config / IO ----------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    # sheets bits for crypto discovery
    sheets = (cfg.get("sheets") or {})
    google = (cfg.get("google") or {})
    return cfg, benchmark, sheets.get("url") or sheets.get("sheet_url"), google.get("service_account_json")

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
    path = "./state/intraday_triggers.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _save_intraday_state(st):
    with open("./state/intraday_triggers.json", "w") as f:
        json.dump(st, f, indent=2)

# ---------------- Small helpers ----------------
def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _is_crypto(sym: str) -> bool:
    return (sym or "").upper().endswith("-USD")

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

# ---------------- Tiny price+RS chart (daily SMA150 ≈ 30wk proxy) ----------------
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

# ---------------- Weekly helpers for Crypto block ----------------
def _extract_field(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned by yfinance.")
    if isinstance(df.columns, pd.MultiIndex):
        avail_top = list(df.columns.get_level_values(0).unique())
        use_field = field if field in avail_top else ("Adj Close" if "Adj Close" in avail_top else None)
        if not use_field:
            raise KeyError(f"Field '{field}' not found; available: {avail_top}")
        out = df[use_field].copy()
        keep = [t for t in tickers if t in out.columns]
        if not keep:
            raise KeyError(f"No requested tickers found in downloaded data. Requested={tickers[:5]}...")
        return out[keep]
    cols = set(df.columns.astype(str))
    if field in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[[field]].copy(); out.columns = [t0]; return out
    if "Adj Close" in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[["Adj Close"]].copy(); out.columns = [t0]; return out
    raise KeyError(f"Field '{field}' not in downloaded data; got columns: {list(df.columns)}")

def fetch_weekly_close_volume(tickers, weeks=WEEKS_LOOKBACK):
    uniq = list(dict.fromkeys(tickers))
    data = yf.download(
        uniq, interval="1wk", period="10y",
        auto_adjust=True, ignore_tz=True, progress=False, group_by="column"
    )
    close = _extract_field(data, "Close", uniq)
    volume = _extract_field(data, "Volume", uniq)
    tail_n = max(weeks, MA_WEEKS + RS_MA_WEEKS + SLOPE_WINDOW + 10)
    close = close.tail(tail_n); volume = volume.tail(tail_n)
    return close, volume

def _weekly_short_state(series_price: pd.Series):
    s = series_price.dropna()
    if len(s) < max(MA10_WEEKS, MA_WEEKS) + 5:
        return ("Unknown", np.nan, np.nan)
    ma10 = s.rolling(MA10_WEEKS).mean()
    ma30 = s.rolling(MA_WEEKS).mean()
    c = float(s.iloc[-1]); m10 = float(ma10.iloc[-1]); m30 = float(ma30.iloc[-1])
    state = "Unknown"
    if pd.notna(m10) and pd.notna(m30):
        if (c > m10) and (m10 > m30): state = "ShortTermUptrend"
        elif (c > m30) and not (m10 > m30): state = "StageConflict"
        elif (m10 > m30) and not (c > m10): state = "StageConflict"
        else: state = "Weak"
    return (state, m10, m30)

def compute_stage_weekly(closes: pd.Series, bench: pd.Series):
    s = closes.dropna().copy(); b = bench.reindex_like(s).dropna()
    idx = s.index.intersection(b.index); s = s.loc[idx]; b = b.loc[idx]
    if len(s) < MA_WEEKS + SLOPE_WINDOW + 5 or len(b) < RS_MA_WEEKS + 5:
        return {"stage":"N/A","price":float(s.iloc[-1]) if len(s) else np.nan}
    ma = s.rolling(MA_WEEKS).mean()
    ma_slope = ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    ma_slope_last = ma_slope.iloc[-1]; ma_last = ma.iloc[-1]; price_last = s.iloc[-1]
    dist_ma_pct = (price_last - ma_last) / ma_last if ma_last and not math.isclose(ma_last, 0.0) else np.nan
    rs = s / b
    rs_ma = rs.rolling(RS_MA_WEEKS).mean()
    rs_slope = rs_ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    rs_last = rs.iloc[-1]; rs_ma_last = rs_ma.iloc[-1]
    rs_above = bool(rs_last > rs_ma_last); rs_slope_last = rs_slope.iloc[-1]
    price_above_ma = bool(price_last > ma_last); ma_up = bool(ma_slope_last > 0)
    near_ma = bool(abs(dist_ma_pct) <= NEAR_MA_BAND)
    rs_up = bool(rs_above and rs_slope_last > 0)
    rs_down = bool((not rs_above) and rs_slope_last < 0)
    if price_above_ma and ma_up and rs_up:
        stage = "Stage 2 (Uptrend)"
    elif (not price_above_ma) and (ma_slope_last < 0) and rs_down:
        stage = "Stage 4 (Downtrend)"
    elif near_ma and abs(ma_slope_last) < (abs(ma_last) * 0.0005):
        stage = "Stage 1 (Basing)"
    else:
        stage = "Stage 3 (Topping)"
    st_state, ma10_last, _ = _weekly_short_state(s)
    return {
        "stage": stage,
        "price": float(price_last),
        "ma10": float(ma10_last) if pd.notna(ma10_last) else np.nan,
        "ma30": float(ma_last),
        "dist_ma_pct": float(dist_ma_pct) if pd.notna(dist_ma_pct) else np.nan,
        "ma_slope_per_wk": float(ma_slope_last) if pd.notna(ma_slope_last) else np.nan,
        "rs": float(rs_last),
        "rs_ma30": float(rs_ma_last) if pd.notna(rs_ma_last) else np.nan,
        "rs_above_ma": bool(rs_above),
        "rs_slope_per_wk": float(rs_slope_last) if pd.notna(rs_slope_last) else np.nan,
        "short_term_state_wk": st_state,
        "notes": "",
    }

def _auth_sheets(service_account_file: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(service_account_file, scopes=scopes)
    return gspread.authorize(creds)

def _read_tab(gc, sheet_url: str, title: str) -> pd.DataFrame:
    sh = gc.open_by_url(sheet_url)
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return pd.DataFrame()
    vals = ws.get_all_values()
    if not vals: return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    for c in df.columns:
        df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def _is_crypto_symbol(sym: str) -> bool:
    s = (sym or "").strip().upper()
    return s.endswith("-USD") and len(s) >= 6 and all(ch.isalnum() or ch in "-." for ch in s)

def _signals_crypto_universe(sheet_url: str, service_account_file: str) -> list[str]:
    """Harvest crypto tickers from existing 'Signals' tab only (no rewiring)."""
    if not (gspread and Credentials and sheet_url and service_account_file and os.path.exists(service_account_file)):
        return []
    try:
        gc = _auth_sheets(service_account_file)
        sig = _read_tab(gc, sheet_url, TAB_SIGNALS)
        if sig.empty: return []
        mapping = {}
        try:
            m = _read_tab(gc, sheet_url, TAB_MAPPING)
            if not m.empty and "Ticker" in m.columns:
                for _, r in m.iterrows():
                    t = str(r.get("Ticker","")).strip().upper()
                    tyf = str(r.get("TickerYF","")).strip().upper()
                    if t: mapping[t] = tyf or ""
        except Exception:
            pass
        tcol = next((c for c in sig.columns if c.lower() in ("ticker","symbol")), "Ticker")
        raw = sig[tcol].astype(str).str.upper().str.strip()
        out = []
        for t in raw:
            yf_sym = mapping.get(t, t)
            if _is_crypto_symbol(yf_sym):
                out.append(yf_sym)
        return list(dict.fromkeys(out))
    except Exception:
        return []

def _tiny_weekly_chart_data_uri(series_price: pd.Series, bench_series: pd.Series):
    s = series_price.dropna(); b = bench_series.reindex_like(s).dropna()
    idx = s.index.intersection(b.index)
    if len(idx) < (MA_WEEKS + 6): return ""
    s = s.loc[idx]; b = b.loc[idx]
    ma30 = s.rolling(MA_WEEKS).mean()
    ma10 = s.rolling(MA10_WEEKS).mean()
    rs = (s / b).rolling(RS_MA_WEEKS).mean()
    fig, ax1 = plt.subplots(figsize=(3.0, 1.2), dpi=120)
    ax1.plot(s.index, s.values, linewidth=1.2)
    ax1.plot(ma10.index, ma10.values, linewidth=1.0)
    ax1.plot(ma30.index, ma30.values, linewidth=1.0)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    ax2 = ax1.twinx()
    ax2.plot(rs.index, rs.values, linewidth=0.8)
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in (*ax1.spines.values(), *ax2.spines.values()):
        spine.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def build_crypto_weekly_block_from_signals(sheet_url: str, service_account_file: str):
    """
    Returns (crypto_df, bench_series, html_block)
    - crypto_df has columns like weekly report (ticker, stage, ma10/ma30, rs, etc., buy_signal)
    - html_block is the rendered table with tiny sparklines
    """
    tickers = _signals_crypto_universe(sheet_url, service_account_file)
    tickers = [t for t in tickers if t != CRYPTO_BENCHMARK]
    if not tickers:
        return pd.DataFrame(), None, ""

    all_syms = list(dict.fromkeys(tickers + [CRYPTO_BENCHMARK]))
    close_w, volume_w = fetch_weekly_close_volume(all_syms, weeks=WEEKS_LOOKBACK)
    if CRYPTO_BENCHMARK not in close_w.columns:
        return pd.DataFrame(), None, ""

    bench = close_w[CRYPTO_BENCHMARK].dropna()
    rows = []
    for t in tickers:
        if t not in close_w.columns:
            continue
        res = compute_stage_weekly(close_w[t], bench)
        res["ticker"] = t
        res["asset_class"] = "Crypto"
        rows.append(res)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), bench, ""

    # buy_signal
    def _classify(stage: str):
        if isinstance(stage, str) and stage.startswith("Stage 2"): return "BUY"
        if isinstance(stage, str) and stage.startswith("Stage 1"): return "WATCH"
        return "AVOID"
    df["Buy Signal"] = df["stage"].apply(_classify)

    # add tiny weekly sparkline charts
    df["chart"] = ""
    for i, r in df.iterrows():
        try:
            df.at[i, "chart"] = _tiny_weekly_chart_data_uri(close_w[r["ticker"]], bench)
        except Exception:
            df.at[i, "chart"] = ""

    # order
    order = ["ticker","asset_class","Buy Signal","chart","stage","short_term_state_wk",
             "price","ma10","ma30","dist_ma_pct","ma_slope_per_wk",
             "rs","rs_ma30","rs_above_ma","rs_slope_per_wk","notes"]
    for c in order:
        if c not in df.columns: df[c] = ""
    df = df[order]

    # summary + HTML table
    cb = int((df["Buy Signal"] == "BUY").sum())
    cw = int((df["Buy Signal"] == "WATCH").sum())
    ca = int((df["Buy Signal"] == "AVOID").sum())
    ct = int(len(df))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_line = f"<strong>Crypto Summary:</strong> ✅ Buy: {cb} &nbsp; | &nbsp; 🟡 Watch: {cw} &nbsp; | &nbsp; 🔴 Avoid: {ca} &nbsp; (Total: {ct})"
    # pretty format
    styled = df.copy()
    def _pct2(x):
        try:
            return f"{float(x)*100:.2f}%" if pd.notna(x) else ""
        except Exception:
            return ""
    for c in ["dist_ma_pct","ma_slope_per_wk","rs_slope_per_wk"]:
        styled[c] = styled[c].apply(_pct2)
    styled["rs_above_ma"] = styled["rs_above_ma"].map({True:"Yes", False:"No"})
    # badges
    def _badge(s):
        k = str(s or "").upper()
        if k == "BUY": return '<span style="display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;background:#eaffea;color:#0f5e1d;border:1px solid #b8e7b9">Buy</span>'
        if k == "WATCH": return '<span style="display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;background:#effaf0;color:#1e7a1e;border:1px solid #cdebd0">Watch</span>'
        return '<span style="display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;background:#ffe8e6;color:#8a1111;border:1px solid #f3b3ae">Avoid</span>'
    styled["Buy Signal"] = styled["Buy Signal"].apply(_badge)

    table_html = styled.to_html(index=False, border=0, justify="center", escape=False)
    css = """
    <style>
      .crypto-block h3 { margin:0 0 4px 0; }
      .crypto-block .sub { color:#666; margin-bottom:12px; }
      .crypto-block table { border-collapse: collapse; width: 100%; }
      .crypto-block th, .crypto-block td { padding: 6px 8px; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: top; }
      .crypto-block th { text-align:left; background:#fafafa; }
    </style>
    """
    html = f"""{css}
    <div class="crypto-block">
      <h3>Crypto Weekly — Benchmark: {CRYPTO_BENCHMARK}</h3>
      <div class="sub">Generated {now}</div>
      <div class="summary" style="background:#f6f8fa;border:1px solid #eaecef;padding:8px 10px;border-radius:8px;margin:8px 0 12px 0;">{summary_line}</div>
      {table_html}
    </div>
    """
    return df, bench, html

# ---------------- Build Email blocks: summary + holdings snapshot ----------------
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

# ---------------- Main logic ----------------
def run(_config_path="./config.yaml"):
    log("Intraday watcher starting with config: ./config.yaml", level="step")
    log("Loading weekly report + config...", level="step")
    cfg, benchmark, sheet_url, service_account_file = load_config(_config_path)
    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # Normalize expected columns (equities weekly csv)
    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker","stage","ma30","rs_above_ma","asset_class"]:
        if miss not in w.columns: w[miss] = np.nan
    focus_equities = w[w["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])][["ticker","stage","ma30","rs_above_ma","asset_class"]].copy()
    focus_equities["weekly_rank"] = (w["rank"] if "rank" in w.columns else 999999)

    # ---------- Build Crypto Weekly block (from Signals) & merge into focus ----------
    crypto_df_block, crypto_bench_series, crypto_html_block = build_crypto_weekly_block_from_signals(sheet_url, service_account_file)
    if not crypto_df_block.empty:
        # extend focus with Stage 1/2 cryptos
        cf = crypto_df_block.rename(columns=str.lower).copy()
        cf = cf[cf["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])]
        cf["weekly_rank"] = 999999  # no external rank; place after ranked equities
        add_cols = ["ticker","stage","ma30","rs_above_ma","asset_class","weekly_rank"]
        for c in add_cols:
            if c not in cf.columns: cf[c] = np.nan
        focus = pd.concat([focus_equities, cf[add_cols]], ignore_index=True)
    else:
        focus = focus_equities.copy()

    log(f"Focus universe: {len(focus)} symbols (Stage 1/2).", level="info")

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

    state = load_positions(); held = state.get("positions", {})
    trigger_state = _load_intraday_state()

    buy_signals, near_signals, sell_signals = [], [], []
    sell_triggers, sell_from_positions, info_rows, chart_imgs = [], [], [], []

    log("Evaluating candidates...", level="step")
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

        closes_n = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
        ma_ok = pd.notna(ma30); pivot_ok = pd.notna(pivot); rs_ok = rs_above

        elapsed = _elapsed_in_current_bar_minutes(intraday, t) if INTRADAY_INTERVAL == "60m" else None
        pace_intra = intrabar_volume_pace(intraday, t, bar_minutes=60) if INTRADAY_INTERVAL == "60m" else None

        # --- BUY confirm ---
        confirm = False; vol_ok = True
        if ma_ok and pivot_ok and closes_n:
            def _price_ok(c):
                return (c >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and (c >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN))
            if INTRADAY_INTERVAL == "60m":
                last_c = closes_n[-1]; price_ok = _price_ok(last_c)
                vol_ok = (pd.isna(pace_intra) or pace_intra >= INTRABAR_VOLPACE_MIN)
                confirm = price_ok and (elapsed is not None and elapsed >= INTRABAR_CONFIRM_MIN_ELAPSED) and vol_ok
            else:
                need = max(CONFIRM_BARS, 2)
                confirm = all(_price_ok(c) for c in closes_n[-need:])
                if REQUIRE_RISING_BAR_VOL:
                    vols2 = get_last_n_intraday_volumes(intraday, t, n=2)
                    vavg = get_intraday_avg_volume(intraday, t, window=INTRADAY_AVG_VOL_WINDOW)
                    if len(vols2) >= 2 and pd.notna(vavg) and vavg > 0:
                        vol_ok = (vols2[-1] >= INTRADAY_LASTBAR_AVG_MULT * vavg)
                    else:
                        vol_ok = False

        # --- BUY near flag ---
        near_now = False
        if stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)") and rs_ok and pivot_ok and ma_ok and pd.notna(px):
            above_ma = px >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
            if above_ma:
                if (px >= pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)) and (px < pivot * (1.0 + MIN_BREAKOUT_PCT)):
                    near_now = True
                elif (px >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not confirm:
                    near_now = True

        # --- SELL near/confirm ---
        sell_near_now = False; sell_confirm = False; sell_vol_ok = True
        if ma_ok and pd.notna(px):
            sell_near_now = (px >= ma30 * (1.0 - SELL_BREAK_PCT)) and (px <= ma30 * (1.0 + SELL_NEAR_ABOVE_MA_PCT))
            if INTRADAY_INTERVAL == "60m":
                sell_price_ok = (px <= ma30 * (1.0 - SELL_BREAK_PCT))
                sell_vol_ok = (pace_intra is None) or (pace_intra >= SELL_INTRABAR_VOLPACE_MIN)
                sell_confirm = bool(sell_price_ok and (elapsed is not None and elapsed >= SELL_INTRABAR_CONFIRM_MIN_ELAPSED) and sell_vol_ok)
            else:
                closes_n2 = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
                if closes_n2:
                    sell_confirm = all((c <= ma30 * (1.0 - SELL_BREAK_PCT)) for c in closes_n2[-CONFIRM_BARS:])

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
        elif state_now in ("IDLE","NEAR") and near_count >= NEAR_HITS_MIN: state_now = "ARMED"
        elif state_now == "ARMED" and confirm and vol_ok:
            state_now = "TRIGGERED"; st["cooldown"] = COOLDOWN_SCANS
        elif state_now == "TRIGGERED": pass
        elif st["cooldown"] > 0 and not near_now: state_now = "COOLDOWN"
        elif st["cooldown"] == 0 and not near_now and not confirm: state_now = "IDLE"
        st["state"] = state_now

        # SELL hits
        st["sell_hits"], sell_hit_count = _update_hits(st.get("sell_hits", []), sell_near_now, SELL_NEAR_HITS_WINDOW)
        if st.get("sell_cooldown", 0) > 0: st["sell_cooldown"] = int(st["sell_cooldown"]) - 1
        sell_state = st.get("sell_state", "IDLE")
        if sell_state == "IDLE" and sell_near_now: sell_state = "NEAR"
        elif sell_state in ("IDLE","NEAR") and sell_hit_count >= SELL_NEAR_HITS_MIN: sell_state = "ARMED"
        elif sell_state == "ARMED" and sell_confirm and sell_vol_ok:
            sell_state = "TRIGGERED"; st["sell_cooldown"] = SELL_COOLDOWN_SCANS
        elif sell_state == "TRIGGERED": pass
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
            atr = compute_atr(daily, t, n=14)
            trail = px - TRAIL_ATR_MULT * atr if pd.notna(atr) else None
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
                    "stage": stage, "weekly_rank": weekly_rank, "source": "risk"
                })

        # --- EMIT by state ---
        if st["state"] == "TRIGGERED" and (
            stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)")
            and rs_ok and confirm and vol_ok
            and (pd.isna(pace) or pace >= VOL_PACE_MIN)
        ):
            buy_signals.append({
                "ticker": t, "price": px, "pivot": pivot, "pace": None if pd.isna(pace) else float(pace),
                "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank,
            })
            trigger_state[t]["state"] = "COOLDOWN"
        elif st["state"] in ("NEAR","ARMED"):
            if (pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN):
                near_signals.append({
                    "ticker": t, "price": px, "pivot": pivot, "pace": None if pd.isna(pace) else float(pace),
                    "stage": stage, "ma30": ma30, "weekly_rank": weekly_rank, "reason": "near/armed"
                })

        if st["sell_state"] == "TRIGGERED":
            sell_triggers.append({
                "ticker": t, "price": px, "ma30": ma30, "stage": stage,
                "weekly_rank": weekly_rank, "pace": None if pd.isna(pace) else float(pace)
            })
            trigger_state[t]["sell_state"] = "COOLDOWN"

        info_rows.append({
            "ticker": t, "stage": stage, "price": px, "ma30": ma30, "pivot10w": pivot,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace), 2),
            "two_bar_confirm": confirm, "last_bar_vol_ok": vol_ok if 'vol_ok' in locals() else None,
            "weekly_rank": weekly_rank
        })
        log(f"{t}: buy_state={st['state']} near_hits={sum(st.get('near_hits', []))} | "
            f"sell_state={st['sell_state']} sell_hits={sum(st.get('sell_hits', []))}", level="debug")

    log(f"Scan done. Raw counts → BUY:{len(buy_signals)} NEAR:{len(near_signals)} SELLTRIG:{len(sell_triggers)}", level="info")

    # ---------- SELL recommendations from holdings ----------
    holdings_block_html = ""
    holdings_raw = _load_open_positions_local()
    if holdings_raw is not None and not holdings_raw.empty:
        pos_norm = _normalize_open_positions_columns(holdings_raw)
        # Merge stage signals from BOTH equities weekly csv + crypto block (so Recommendation colors make sense for crypto too)
        merged_weekly_for_holdings = weekly_df.copy()
        if not crypto_df_block.empty:
            merged_weekly_for_holdings = pd.concat([merged_weekly_for_holdings, crypto_df_block], ignore_index=True)
        merged = _merge_stage_and_recommend(pos_norm, merged_weekly_for_holdings)

        sell_from_positions_map = {}
        for _, r in merged.iterrows():
            rec = str(r.get("Recommendation", "")).upper()
            if not rec.startswith("SELL"): continue
            sym = str(r.get("Symbol", "")).strip()
            if not sym: continue
            live_px = px_now(sym)
            use_px = live_px if pd.notna(live_px) else float(r.get("Last Price", np.nan))
            reasons = []
            pct = r.get("Total Gain/Loss Percent", np.nan)
            stg = str(r.get("stage", ""))
            if pd.notna(pct) and pct <= -8.0: reasons.append("drawdown ≤ −8%")
            if stg.startswith("Stage 4") and (pd.notna(pct) and pct < 0): reasons.append("Stage 4 + negative P/L")
            if not reasons: reasons.append("strategy rule")
            entry = sell_from_positions_map.get(sym)
            if entry is None:
                sell_from_positions_map[sym] = {
                    "ticker": sym, "price": use_px if pd.notna(use_px) else np.nan,
                    "reasons": set(reasons), "stage": stg, "weekly_rank": np.nan, "source": "positions",
                }
            else:
                entry["reasons"].update(reasons)
                if pd.isna(entry["price"]) and pd.notna(use_px):
                    entry["price"] = use_px

        sell_from_positions = []
        for sym, entry in sell_from_positions_map.items():
            entry["reasons"] = "; ".join(sorted(entry["reasons"]))
            sell_from_positions.append(entry)

        metrics = _compute_portfolio_metrics(pos_norm)
        holdings_block_html = _colored_summary_html(metrics) + _format_holdings_table(merged)

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
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {price_str} — {it.get('reasons','')} "
                           f"({it.get('stage','')}, weekly {wr_str}){src_label}</li>")
            elif kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                           f"(↓ MA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})</li>")
            else:
                pace_val = it.get("pace", None)
                pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                           f"(pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']}, weekly {wr_str})</li>")
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
    <h4>Buy Triggers (ranked)</h4>
    {bullets(buy_signals, "BUY")}
    <h4>Near-Triggers (ranked)</h4>
    {bullets(near_signals, "NEAR")}
    <h4>Sell Triggers (ranked)</h4>
    {bullets(sell_triggers, "SELLTRIG")}
    {charts_html}
    """

    # Append Crypto Weekly block (from Signals) just like your sample
    if crypto_html_block:
        html += "<hr/>" + crypto_html_block

    # Holdings summary + colored recommendations (HOLD/HOLD Strong/SELL)
    if holdings_block_html:
        html += "<hr/>" + holdings_block_html

    # Plain text
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

    text = f"Weinstein Intraday Watch — {now}\n\nBUY (ranked):\n{_lines(buy_signals,'BUY')}\n\nNEAR-TRIGGER (ranked):\n{_lines(near_signals,'NEAR')}\n\nSELL TRIGGERS (ranked):\n{_lines(sell_triggers,'SELLTRIG')}\n\nSELL / RISK:\n{_lines(sell_signals + sell_from_positions,'SELL')}\n"

    _save_intraday_state(trigger_state)
    subject_counts = f"{len(buy_signals)} BUY / {len(near_signals)} NEAR / {len(sell_triggers)} SELL-TRIG / {len(sell_signals)+len(sell_from_positions)} SELL"
    log("Sending email...", level="step")
    send_email(
        subject=f"Intraday Watch — {subject_counts}",
        html_body=html,
        text_body=text,
        cfg_path=_config_path
    )
    log("Email sent.", level="ok")

# ---------------- Intraday trigger state knobs (kept at bottom for clarity) ----------------
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

# ---------------- Main ----------------
if __name__ == "__main__":
    args = _parse_args()
    VERBOSE = not args.quiet
    try:
        run(_config_path=args.config)
        log("Intraday tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
