#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Short Watcher — Stage 4 short setups

- Uses weekly Weinstein scan (equities) to build a Stage 4 (Downtrend) universe
- Intraday checks around 10-week pivot LOW and 30-week MA proxy (SMA150)
- Emits:
    * TRIG shorts: breakdown below pivot/MA with volume + intrabar confirmations
    * NEAR shorts: price hanging just above breakdown zone, with basic volume pacing
    * READY-TO-CLOSE shorts: Stage 4 names that have reclaimed MA150 by ~0.5%+,
      suggesting the short thesis is weakening (time to consider covering).
- Email contains:
    * Ranked list of short triggers + near shorts + ready-to-close shorts
    * Order block: entry≈now, protective stop, 15% & 20% downside targets
    * Tiny charts for top names
- NEW:
    * READY-TO-CLOSE shorts are filtered to tickers that appear in the Google Sheet
      "Signals" tab with Direction = SELL/SHORT (so you only see shorts you actually track).
    * --log-csv / --log-json diagnostics (per-symbol metrics + conditions + state)

Email behavior:
- Email is sent ONLY when there is at least one of:
  * Short Triggers (TRIG)
  * Near Short Setups (NEAR)
  * Ready-to-close Shorts (READY)
"""

import os, io, json, math, base64, argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weinstein_mailer import send_email

# Optional: Google Sheets integration for READY filter
try:
    import gspread
except ImportError:
    gspread = None

# ---------------- Tunables ----------------
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_equities_"
BENCHMARK_DEFAULT = "SPY"

INTRADAY_INTERVAL = "60m"     # '60m' or '30m'
LOOKBACK_DAYS     = 60
PIVOT_LOOKBACK_WEEKS = 10
PRICE_WINDOW_DAYS = 260
SMA_DAYS          = 150

# Short trigger thresholds
SHORT_BREAK_PCT          = 0.004   # 0.4% below pivot/MA for confirm
NEAR_ABOVE_PIVOT_PCT     = 0.02    # considers "just above" pivot if within +2%
VOL_PACE_MIN             = 1.30    # full-day vol pace gate for TRIG shorts
NEAR_VOL_PACE_MIN        = 1.00    # for NEAR shorts
INTRADAY_AVG_VOL_WINDOW  = 20
INTRADAY_LASTBAR_MULT    = 1.20    # (used only for non-60m modes)

# 60m intrabar confirmations
INTRABAR_CONFIRM_MIN_ELAPSED  = 40    # minutes into bar
INTRABAR_VOLPACE_MIN          = 1.20  # intrabar pace vs avg for confirm

# READY-TO-CLOSE threshold
READY_ABOVE_MA_PCT       = 0.005  # 0.5% above MA150 => consider short "ready to close"

# Stateful short triggers
SHORT_STATE_FILE       = "./state/short_triggers.json"
SCAN_INTERVAL_MIN      = 10
SHORT_NEAR_HITS_WINDOW = 6
SHORT_NEAR_HITS_MIN    = 3
SHORT_COOLDOWN_SCANS   = 24

# Short risk/profit mapping
SHORT_HARD_STOP_PCT    = 0.20   # 20% above entry
SHORT_TRAIL_ATR_MULT   = 2.0
SHORT_MA_GUARD_PCT     = 0.03   # 3% over MA150
SHORT_TARGET1_PCT      = 0.15   # 15% downside
SHORT_TARGET2_PCT      = 0.20   # 20% downside

CHART_DIR           = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

VERBOSE = True

# ---------------- Small helpers ----------------
def _ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, *, level="info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {
        "info": "•",
        "ok": "✅",
        "step": "▶️",
        "warn": "⚠️",
        "err": "❌",
        "debug": "··",
    }.get(level, "•")
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
    import yaml
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
    files = [
        f
        for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found in {WEEKLY_OUTPUT_DIR}. "
            f"Run weinstein_report_weekly.py first."
        )
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report():
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path


def load_ready_short_tickers_from_signals(cfg, sheet_url, service_account_file):
    """
    Load tickers from the Google Sheet Signals tab that should be treated as
    "held"/tracked shorts for READY-TO-CLOSE.

    We treat any row with:
      - non-empty Ticker
      - Direction in {SELL, SHORT}  (case-insensitive)
    as a tracked short. This matches your Signals schema:

      TimestampUTC | Ticker | Source | Direction | Price | Timeframe

    If anything fails (no gspread, bad creds, etc.), we return None and
    skip filtering (READY list will behave as before, on full Stage 4 universe).
    """
    sheets_cfg = (cfg.get("sheets") or {})
    tab_name = sheets_cfg.get("signals_tab", "Signals")

    if not sheet_url or not service_account_file:
        log("READY filter: missing sheet_url or service_account_json; disabled.", level="debug")
        return None

    if gspread is None:
        log("READY filter: gspread not installed; disabled.", level="warn")
        return None

    try:
        gc = gspread.service_account(filename=service_account_file)
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(tab_name)
        rows = ws.get_all_records()
    except Exception as e:
        log(f"READY filter: could not load Signals tab '{tab_name}': {e}", level="warn")
        return None

    def _cell_to_str(v) -> str:
        """Safely coerce any cell value (int/float/None/str) to a string."""
        if v is None:
            return ""
        try:
            return str(v)
        except Exception:
            return ""

    tickers = set()
    for r in rows:
        raw_t = r.get("Ticker", None)
        if raw_t is None:
            raw_t = r.get("ticker", None)
        t = _cell_to_str(raw_t).strip().upper()
        if not t:
            continue

        raw_dir = r.get("Direction", None)
        if raw_dir is None:
            raw_dir = r.get("direction", None)
        direction = _cell_to_str(raw_dir).strip().upper()

        # Only care about shorts / sells
        if direction not in ("SELL", "SHORT"):
            continue

        tickers.add(t)

    log(
        f"READY filter: loaded {len(tickers)} short tickers from Signals tab '{tab_name}'.",
        level="info",
    )
    return tickers or None

# ---------------- State helpers ----------------
def _load_short_state():
    path = SHORT_STATE_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_short_state(st):
    with open(SHORT_STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


def _update_hits(window_arr, hit, window):
    window_arr = (window_arr or [])
    window_arr.append(1 if hit else 0)
    if len(window_arr) > window:
        window_arr = window_arr[-window:]
    return window_arr, sum(window_arr)


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

# ---------------- Data helpers ----------------
def get_intraday(tickers):
    uniq = list(dict.fromkeys(tickers))
    intraday = yf.download(
        uniq,
        period=f"{LOOKBACK_DAYS}d",
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )
    daily = yf.download(
        uniq,
        period="24mo",
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
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
    if not {"High", "Low", "Close"}.issubset(set(sub.columns)):
        return np.nan
    h, l, c = sub["High"], sub["Low"], sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1,
    ).max(axis=1)
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
    """Projected full-day volume vs 50-day avg."""
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
        fraction = min(1.0, max(0.05, elapsed / (24 * 3600.0)))
    else:
        minutes = now.hour * 60 + now.minute
        start = 13 * 60 + 30
        end = 20 * 60 + 0
        if minutes <= start:
            fraction = 0.05
        elif minutes >= end:
            fraction = 1.0
        else:
            fraction = (minutes - start) / (6.5 * 60)
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


def intrabar_volume_pace(
    intraday_df, ticker, avg_window=INTRADAY_AVG_VOL_WINDOW, bar_minutes=60
):
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

# ---------------- Short logic helpers ----------------
def _short_price_break(px, ma, pivot_low):
    """True if price has broken below short zone (~pivot low/MA150)."""
    conds = []
    if pd.notna(pivot_low):
        conds.append(px <= pivot_low * (1.0 - SHORT_BREAK_PCT))
    if pd.notna(ma):
        conds.append(px <= ma * (1.0 - SHORT_BREAK_PCT))
    return any(conds) if conds else False


def _short_near_zone(px, ma, pivot_low):
    """Near-breakdown zone: under MA150 but not yet breaking pivot/MA too hard."""
    if pd.isna(px) or (pd.isna(ma) and pd.isna(pivot_low)):
        return False
    # must be below MA150 (downtrend active)
    below_ma = (pd.notna(ma) and px < ma)
    if not below_ma:
        return False
    # treat "near" as above pivot low but not crazy far
    if pd.notna(pivot_low):
        if px <= pivot_low:  # already at/below pivot; let full trigger handle
            return False
        if px <= pivot_low * (1.0 + NEAR_ABOVE_PIVOT_PCT):
            return True
    # fallback: a mild cushion below MA150 but not full 0.4% break
    if pd.notna(ma):
        if (px <= ma) and (px >= ma * (1.0 - SHORT_BREAK_PCT)):
            return True
    return False


def _short_ready_to_close(px, ma):
    """
    READY-TO-CLOSE short:
      - price has reclaimed MA150 by READY_ABOVE_MA_PCT (e.g. 0.5%+ above MA150),
        suggesting downtrend thesis is weakening.
    """
    if pd.isna(px) or pd.isna(ma):
        return False
    return px >= ma * (1.0 + READY_ABOVE_MA_PCT)


def stage_order(stage: str) -> int:
    if isinstance(stage, str):
        if stage.startswith("Stage 4"):
            return 0
        if stage.startswith("Stage 3"):
            return 1
    return 9


def short_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(
        item.get("weekly_rank", np.nan)
    ) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan)
    ma = item.get("ma30", np.nan)
    dist_below = (ma - px) if (pd.notna(px) and pd.notna(ma)) else -1e9
    pace = item.get("pace", np.nan)
    pace = pace if pd.notna(pace) else -1e9
    return (wr, st, dist_below, -pace)

# ---------------- Order block (stops + targets) ----------------
def _fmt_num(x):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def _short_entry_stop_targets(px, ma30, pivot_low, atr):
    """
    For shorts:
      entry ≈ px (current price)
      stop  = max(
                 entry * (1 + SHORT_HARD_STOP_PCT),
                 entry + SHORT_TRAIL_ATR_MULT * ATR,
                 ma30 * (1 + SHORT_MA_GUARD_PCT)
              )
      targets = [ -15%, -20% from entry ]
    """
    if pd.isna(px):
        return np.nan, np.nan, np.nan, np.nan
    entry = float(px)

    hard = entry * (1.0 + SHORT_HARD_STOP_PCT)
    atr_stop = (entry + SHORT_TRAIL_ATR_MULT * atr) if pd.notna(atr) else np.nan
    ma_guard = (ma30 * (1.0 + SHORT_MA_GUARD_PCT)) if pd.notna(ma30) else np.nan

    cand = [c for c in [hard, atr_stop, ma_guard] if pd.notna(c)]
    stop = max(cand) if cand else hard

    t1 = entry * (1.0 - SHORT_TARGET1_PCT)
    t2 = entry * (1.0 - SHORT_TARGET2_PCT)
    return entry, stop, t1, t2


def _build_order_block_html(short_trigs, near_shorts, cover_shorts):
    items = short_trigs + near_shorts + cover_shorts
    if not items:
        return ""

    css = """
    <style>
      .ordtbl { border-collapse: collapse; width:100%; margin-top:6px; }
      .ordtbl th, .ordtbl td {
        border-bottom:1px solid #eee; padding:6px 8px;
        font-size:13px; text-align:left;
      }
      .ordtbl th { background:#fafafa; }
    </style>
    """

    rows = []
    for it in items:
        t   = it["ticker"]
        px  = it.get("price", np.nan)
        ma  = it.get("ma30", np.nan)
        piv = it.get("pivot_low", np.nan)
        atr = it.get("atr", np.nan)
        entry, stop, t1, t2 = _short_entry_stop_targets(px, ma, piv, atr)
        rows.append(
            f"<tr>"
            f"<td>{t}</td>"
            f"<td>{_fmt_num(px)}</td>"
            f"<td>{_fmt_num(piv)}</td>"
            f"<td>{_fmt_num(ma)}</td>"
            f"<td>{_fmt_num(entry)}</td>"
            f"<td>{_fmt_num(stop)}</td>"
            f"<td>{_fmt_num(t1)}</td>"
            f"<td>{_fmt_num(t2)}</td>"
            f"</tr>"
        )

    html = (
        css
        + """
    <h4>Order Block (short-side, proposed)</h4>
    <table class="ordtbl">
      <thead>
        <tr>
          <th>Ticker</th><th>Now</th><th>Pivot Low</th><th>MA150</th>
          <th>Entry ≈</th><th>Stop ≥</th><th>Target1 ↓ (15%)</th><th>Target2 ↓ (20%)</th>
        </tr>
      </thead>
      <tbody>
    """
        + "\n".join(rows)
        + "</tbody></table>"
    )

    html += (
        "<div style='font-size:12px;color:#666;margin-top:6px;'>"
        f"Rules: entry≈current price; stop = max(entry+{SHORT_HARD_STOP_PCT*100:.0f}%, "
        f"ATR×{SHORT_TRAIL_ATR_MULT:.1f} above, "
        f"MA150+{SHORT_MA_GUARD_PCT*100:.0f}%). Targets at −{SHORT_TARGET1_PCT*100:.0f}% "
        f"and −{SHORT_TARGET2_PCT*100:.0f}% from entry as initial profit milestones "
        "(based on Weinstein risk discipline)."
        "</div>"
    )
    return html


def _build_order_block_text(short_trigs, near_shorts, cover_shorts):
    lines = ["ORDER BLOCK (short-side, proposed)"]
    for it in (short_trigs + near_shorts + cover_shorts):
        t   = it["ticker"]
        px  = it.get("price", np.nan)
        ma  = it.get("ma30", np.nan)
        piv = it.get("pivot_low", np.nan)
        atr = it.get("atr", np.nan)
        entry, stop, t1, t2 = _short_entry_stop_targets(px, ma, piv, atr)
        lines.append(
            f"- {t}: now={_fmt_num(px)} pivot_low={_fmt_num(piv)} "
            f"MA150={_fmt_num(ma)} entry≈{_fmt_num(entry)} stop≥{_fmt_num(stop)} "
            f"targets↓ [{_fmt_num(t1)}, {_fmt_num(t2)}]"
        )
    if lines:
        lines.append(
            f"Rules: entry≈price; stop=max(entry+{SHORT_HARD_STOP_PCT*100:.0f}%, "
            f"ATR×{SHORT_TRAIL_ATR_MULT:.1f} above, MA150+{SHORT_MA_GUARD_PCT*100:.0f}%); "
            f"targets at −{SHORT_TARGET1_PCT*100:.0f}% and −{SHORT_TARGET2_PCT*100:.0f}% from entry."
        )
    return "\n".join(lines)

# ---------------- Charting ----------------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(
        "ascii"
    )


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
    rs = (close_t / close_b)
    rs_norm = rs / rs.iloc[0]

    fig, ax1 = plt.subplots(figsize=(5.0, 2.4), dpi=150)
    ax1.plot(close_t.index, close_t.values, label=f"{ticker}")
    ax1.plot(sma.index, sma.values, label=f"SMA{SMA_DAYS}", linewidth=1.2)
    ax1.set_ylabel("Price")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(
        rs_norm.index, rs_norm.values, linestyle="--", alpha=0.7, label="RS (norm)"
    )
    ax2.set_ylabel("RS (norm)")
    ax2.tick_params(axis="y", labelsize=8)

    ax1.set_title(f"{ticker} — Price, SMA150, RS/{benchmark}", fontsize=9)
    ax1.grid(alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=7,
        loc="upper left",
        frameon=False,
    )

    chart_path = os.path.join(
        CHART_DIR, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)
    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# ---------------- Main logic ----------------
def run(
    _config_path="./config.yaml",
    *,
    only_tickers=None,
    test_ease=False,
    log_csv=None,
    log_json=None,
    dry_run=False,
):

    log(f"Short watcher starting with config: {_config_path}", level="step")
    cfg, benchmark, sheet_url, service_account_file = load_config(_config_path)

    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # Load tickers from Signals tab for READY-TO-CLOSE filtering
    held_ready_tickers = load_ready_short_tickers_from_signals(
        cfg, sheet_url, service_account_file
    )

    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker", "stage", "ma30", "rs_above_ma"]:
        if miss not in w.columns:
            w[miss] = np.nan

    # Stage 4 downtrend universe
    short_universe = w[w["stage"].isin(["Stage 4 (Downtrend)"])][
        ["ticker", "stage", "ma30", "rs_above_ma"]
    ].copy()
    if "rank" in w.columns:
        short_universe["weekly_rank"] = w["rank"]
    else:
        short_universe["weekly_rank"] = 999999

    if only_tickers:
        filt = set([t.strip().upper() for t in only_tickers])
        short_universe = short_universe[short_universe["ticker"].isin(filt)].copy()

    log(f"Short universe: {len(short_universe)} symbols (Stage 4).", level="info")

    needs = sorted(set(short_universe["ticker"].tolist() + [benchmark]))

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

    near_shorts, trig_shorts, cover_shorts = [], [], []
    info_rows, chart_imgs, debug_rows = [], [], []

    # Test-ease thresholds
    if test_ease or (os.getenv("INTRADAY_TEST", "0") == "1"):
        log("TEST-EASE: lowering thresholds for quick validation.", level="warn")
        _SHORT_NEAR_HITS_MIN = 1
        _INTRABAR_CONFIRM_MIN_ELAPSED = 0
        _INTRABAR_VOLPACE_MIN = 0.0
    else:
        _SHORT_NEAR_HITS_MIN = SHORT_NEAR_HITS_MIN
        _INTRABAR_CONFIRM_MIN_ELAPSED = INTRABAR_CONFIRM_MIN_ELAPSED
        _INTRABAR_VOLPACE_MIN = INTRABAR_VOLPACE_MIN

    log("Evaluating short candidates...", level="step")

    for _, row in short_universe.iterrows():
        t = row["ticker"]
        px = px_now(t)
        if np.isnan(px):
            continue

        stage = str(row["stage"])
        ma30 = float(row.get("ma30", np.nan))
        rs_above = bool(row.get("rs_above_ma", False))
        rs_ok = (
            not rs_above
        )  # for shorts, we prefer RS not above its MA
        weekly_rank = float(row.get("weekly_rank", np.nan))

        pivot_low = last_weekly_pivot_low(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace_full = volume_pace_today_vs_50dma(t, daily)
        atr = compute_atr(daily, t, n=14)

        closes_n = get_last_n_intraday_closes(intraday, t, n=2)

        elapsed = (
            _elapsed_in_current_bar_minutes(intraday, t)
            if INTRADAY_INTERVAL == "60m"
            else None
        )
        pace_intra = (
            intrabar_volume_pace(intraday, t, bar_minutes=60)
            if INTRADAY_INTERVAL == "60m"
            else None
        )

        metrics = {
            "price": px,
            "ma30": ma30,
            "pivot_low": pivot_low,
            "atr": atr,
            "pace_full_vs50dma": None if pd.isna(pace_full) else float(pace_full),
            "pace_intrabar": None if pd.isna(pace_intra) else float(pace_intra),
            "elapsed_min": elapsed,
        }

        cond = {}
        cond["weekly_stage_ok"] = stage.startswith("Stage 4")
        cond["rs_ok"] = rs_ok
        cond["ma_ok"] = pd.notna(ma30)
        cond["pivot_ok"] = pd.notna(pivot_low)

        # Short near / trigger / ready-close
        short_near_now = False
        short_price_ok = False
        short_vol_ok = True
        short_confirm = False
        ready_close_now = False

        if cond["ma_ok"] and cond["pivot_ok"]:
            short_near_now = _short_near_zone(px, ma30, pivot_low)

            if INTRADAY_INTERVAL == "60m":
                short_price_ok = _short_price_break(px, ma30, pivot_low)
                short_vol_ok = (
                    pd.isna(pace_intra) or pace_intra >= _INTRABAR_VOLPACE_MIN
                )
                short_confirm = bool(
                    short_price_ok
                    and (
                        elapsed is not None
                        and elapsed >= _INTRABAR_CONFIRM_MIN_ELAPSED
                    )
                    and short_vol_ok
                )
            else:
                closes_n2 = get_last_n_intraday_closes(intraday, t, n=2)
                if closes_n2:
                    short_price_ok = all(
                        _short_price_break(c, ma30, pivot_low) for c in closes_n2
                    )
                    short_confirm = short_price_ok
                    if len(closes_n2) >= 2:
                        vols2 = get_last_n_intraday_volumes(intraday, t, n=2)
                        vavg = get_intraday_avg_volume(
                            intraday, t, window=INTRADAY_AVG_VOL_WINDOW
                        )
                        if len(vols2) >= 2 and pd.notna(vavg) and vavg > 0:
                            short_vol_ok = (
                                vols2[-1] >= INTRADAY_LASTBAR_MULT * vavg
                            )
                        else:
                            short_vol_ok = False

        # READY-to-close: reclaimed MA150 by ~0.5%+
        if cond["ma_ok"]:
            ready_close_now = _short_ready_to_close(px, ma30)

        cond["short_near_now"] = bool(short_near_now)
        cond["short_price_ok"] = bool(short_price_ok)
        cond["short_vol_ok"] = bool(short_vol_ok)
        cond["short_confirm"] = bool(short_confirm)
        cond["pace_full_gate"] = pd.isna(pace_full) or pace_full >= VOL_PACE_MIN
        cond["near_pace_gate"] = pd.isna(pace_full) or pace_full >= NEAR_VOL_PACE_MIN
        cond["ready_close_now"] = bool(ready_close_now)

        # Stateful promotion (short_state)
        st = short_state.get(
            t,
            {
                "short_state": "IDLE",
                "short_hits": [],
                "short_cooldown": 0,
            },
        )

        # Short NEAR hits
        st["short_hits"], short_hit_count = _update_hits(
            st.get("short_hits", []),
            short_near_now,
            SHORT_NEAR_HITS_WINDOW,
        )
        if st.get("short_cooldown", 0) > 0:
            st["short_cooldown"] = int(st["short_cooldown"]) - 1

        sstate = st.get("short_state", "IDLE")
        if sstate == "IDLE" and short_near_now:
            sstate = "NEAR"
        elif sstate in ("IDLE", "NEAR") and short_hit_count >= _SHORT_NEAR_HITS_MIN:
            sstate = "ARMED"
        elif (
            sstate == "ARMED"
            and short_confirm
            and short_vol_ok
            and cond["pace_full_gate"]
        ):
            sstate = "TRIGGERED"
            st["short_cooldown"] = SHORT_COOLDOWN_SCANS
        elif st["short_cooldown"] > 0 and not short_near_now:
            sstate = "COOLDOWN"
        elif st["short_cooldown"] == 0 and not short_near_now and not short_confirm:
            sstate = "IDLE"

        st["short_state"] = sstate
        short_state[t] = st

        # Emit short lists
        if st["short_state"] == "TRIGGERED" and cond["pace_full_gate"]:
            trig_shorts.append(
                {
                    "ticker": t,
                    "price": px,
                    "ma30": ma30,
                    "pivot_low": pivot_low,
                    "stage": stage,
                    "weekly_rank": weekly_rank,
                    "pace": None if pd.isna(pace_full) else float(pace_full),
                    "atr": atr,
                }
            )
            short_state[t]["short_state"] = "COOLDOWN"
        elif st["short_state"] in ("NEAR", "ARMED"):
            if cond["near_pace_gate"]:
                near_shorts.append(
                    {
                        "ticker": t,
                        "price": px,
                        "ma30": ma30,
                        "pivot_low": pivot_low,
                        "stage": stage,
                        "weekly_rank": weekly_rank,
                        "pace": None if pd.isna(pace_full) else float(pace_full),
                        "atr": atr,
                    }
                )

        # READY-to-close list (pre-filter)
        if ready_close_now:
            cover_shorts.append(
                {
                    "ticker": t,
                    "price": px,
                    "ma30": ma30,
                    "pivot_low": pivot_low,
                    "stage": stage,
                    "weekly_rank": weekly_rank,
                    "pace": None if pd.isna(pace_full) else float(pace_full),
                    "atr": atr,
                }
            )

        info_rows.append(
            {
                "ticker": t,
                "stage": stage,
                "price": px,
                "ma30": ma30,
                "pivot_low_10w": pivot_low,
                "vol_pace_vs50dma": None
                if pd.isna(pace_full)
                else round(float(pace_full), 2),
                "weekly_rank": weekly_rank,
                "short_state": st["short_state"],
            }
        )

        row_debug = {
            "ticker": t,
            **metrics,
            **{f"cond_{k}": v for k, v in cond.items()},
            "short_state": st["short_state"],
            "short_hits": st.get("short_hits", []),
            "short_cooldown": st.get("short_cooldown", 0),
        }
        debug_rows.append(row_debug)

    log(
        f"Scan done. Shorts → NEAR:{len(near_shorts)} TRIG:{len(trig_shorts)} READY:{len(cover_shorts)}",
        level="info",
    )

    # Restrict READY-TO-CLOSE list to tickers present in Signals tab (if available)
    if held_ready_tickers:
        before = len(cover_shorts)
        cover_shorts = [it for it in cover_shorts if it["ticker"] in held_ready_tickers]
        log(
            f"READY filter: {before} candidates → {len(cover_shorts)} after restricting to Signals tab shorts.",
            level="info",
        )

    # ---- TEST MODE SUMMARY (SHORTS) ----
    if os.getenv("INTRADAY_TEST", "0") == "1":
        print("=== SHORT TEST MODE SUMMARY ===")
        print(f"Short TRIG signals: {len(trig_shorts)}")
        print(f"Short NEAR signals: {len(near_shorts)}")
        print(f"Ready-to-close shorts: {len(cover_shorts)}")
        if trig_shorts or near_shorts or cover_shorts:
            print("Email gate (short): WOULD SEND (TRIG/NEAR/READY present).")
        else:
            print("Email gate (short): SKIPPED (no TRIG/NEAR/READY shorts).")
        print("================================")

    # Ranking & charts
    near_shorts.sort(key=short_sort_key)
    trig_shorts.sort(key=short_sort_key)
    cover_shorts.sort(key=short_sort_key)

    charts_added = 0
    for item in trig_shorts:
        if charts_added >= MAX_CHARTS_PER_EMAIL:
            break
        t = item["ticker"]
        path, data_uri = make_tiny_chart_png(t, BENCHMARK_DEFAULT, daily)
        if data_uri:
            chart_imgs.append((t, data_uri))
            charts_added += 1
    if charts_added < MAX_CHARTS_PER_EMAIL:
        for item in near_shorts:
            if charts_added >= MAX_CHARTS_PER_EMAIL:
                break
            t = item["ticker"]
            path, data_uri = make_tiny_chart_png(t, BENCHMARK_DEFAULT, daily)
            if data_uri:
                chart_imgs.append((t, data_uri))
                charts_added += 1
    if charts_added < MAX_CHARTS_PER_EMAIL:
        for item in cover_shorts:
            if charts_added >= MAX_CHARTS_PER_EMAIL:
                break
            t = item["ticker"]
            path, data_uri = make_tiny_chart_png(t, BENCHMARK_DEFAULT, daily)
            if data_uri:
                chart_imgs.append((t, data_uri))
                charts_added += 1

    log(f"Charts prepared: {len(chart_imgs)}", level="debug")

    # Build email
    def bullets(items, kind):
        if not items:
            if kind == "TRIG":
                return "<p>No TRIG shorts.</p>"
            if kind == "NEAR":
                return "<p>No NEAR shorts.</p>"
            if kind == "READY":
                return "<p>No READY-TO-CLOSE shorts.</p>"
            return "<p>None.</p>"

        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            px = it.get("price", np.nan)
            piv = it.get("pivot_low", np.nan)
            ma = it.get("ma30", np.nan)
            pace_val = it.get("pace", None)
            pace_str = (
                "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
            )
            atr = it.get("atr", np.nan)
            entry, stop, t1, t2 = _short_entry_stop_targets(px, ma, piv, atr)
            if kind == "TRIG":
                label = "TRIG short"
            elif kind == "NEAR":
                label = "NEAR short"
            else:
                label = "READY-CLOSE short"
            lis.append(
                f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {px:.2f} "
                f"({label}, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                f"targets↓ [{_fmt_num(t1)}, {_fmt_num(t2)}], "
                f"{it.get('stage','')}, weekly {wr_str}, pace {pace_str})</li>"
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
      SHORT-TRIGGER: Weekly Stage 4 (Downtrend) + confirmed breakdown under ~10-week pivot low and/or 30-wk MA proxy (SMA150),
      by ≈{SHORT_BREAK_PCT*100:.1f}% with volume pace ≥ {VOL_PACE_MIN:.1f}×. For 60m bars: ≥{INTRABAR_CONFIRM_MIN_ELAPSED} min elapsed & intrabar pace ≥ {INTRABAR_VOLPACE_MIN:.1f}×.<br>
      NEAR-SHORT: Stage 4 + RS not above its MA, price hanging just above the pivot/MA breakdown zone (within +{NEAR_ABOVE_PIVOT_PCT*100:.1f}% over pivot or hugging MA150),
      volume pace ≥ {NEAR_VOL_PACE_MIN:.1f}×.<br>
      READY-TO-CLOSE (SHORT): Stage 4 names where price has reclaimed MA150 by ≈{READY_ABOVE_MA_PCT*100:.1f}% or more,
      suggesting the short thesis is weakening and it's time to consider covering (restricted to shorts listed in your Signals tab).
    </i></p>
    """

    html += f"""
    <h4>Short Triggers (ranked)</h4>
    {bullets(trig_shorts, "TRIG")}
    <h4>Near Short Setups (ranked)</h4>
    {bullets(near_shorts, "NEAR")}
    <h4>Ready-to-Close Shorts (ranked)</h4>
    {bullets(cover_shorts, "READY")}
    {charts_html}
    """

    order_block_html = _build_order_block_html(trig_shorts, near_shorts, cover_shorts)
    if order_block_html:
        html += order_block_html

    # Snapshot table
    if info_rows:
        info_df = pd.DataFrame(info_rows)
        info_df["stage_rank"] = info_df["stage"].apply(stage_order)
        info_df["weekly_rank"] = (
            pd.to_numeric(info_df["weekly_rank"], errors="coerce")
            .fillna(999999)
            .astype(int)
        )
        info_df = info_df.sort_values(
            ["weekly_rank", "stage_rank", "ticker"]
        ).drop(columns=["stage_rank"])
        html += "<h4>Snapshot</h4>" + info_df.to_html(index=False)

    # Plain text
    def _lines(items, kind):
        if not items:
            if kind == "TRIG":
                return "No TRIG shorts."
            if kind == "NEAR":
                return "No NEAR shorts."
            if kind == "READY":
                return "No READY-TO-CLOSE shorts."
            return "None."
        out = []
        for i, it in enumerate(items, 1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            px = it.get("price", np.nan)
            piv = it.get("pivot_low", np.nan)
            ma = it.get("ma30", np.nan)
            pace_val = it.get("pace", None)
            pace_str = (
                "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
            )
            atr = it.get("atr", np.nan)
            entry, stop, t1, t2 = _short_entry_stop_targets(px, ma, piv, atr)
            if kind == "TRIG":
                label = "TRIG short"
            elif kind == "NEAR":
                label = "NEAR short"
            else:
                label = "READY-CLOSE short"
            out.append(
                f"{i}. {it['ticker']} @ {px:.2f} "
                f"({label}, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                f"targets↓ [{_fmt_num(t1)}, {_fmt_num(t2)}], "
                f"{it.get('stage','')}, weekly {wr_str}, pace {pace_str})"
            )
        return "\n".join(out)

    text = (
        f"Weinstein Short Intraday Watch — {now}\n\n"
        f"Short TRIGGERS (ranked):\n{_lines(trig_shorts, 'TRIG')}\n\n"
        f"NEAR short setups (ranked):\n{_lines(near_shorts, 'NEAR')}\n\n"
        f"READY-TO-CLOSE shorts (ranked):\n{_lines(cover_shorts, 'READY')}\n\n"
    )

    order_block_text = _build_order_block_text(trig_shorts, near_shorts, cover_shorts)
    if order_block_text:
        text += "\n" + order_block_text + "\n"

    # Persist state & diagnostics
    _save_short_state(short_state)

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

    # Save HTML
    os.makedirs("./output", exist_ok=True)
    html_path = os.path.join(
        "./output", f"short_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    # -------- Email only when we actually have something --------
    has_shorts = bool(trig_shorts or near_shorts or cover_shorts)

    if not has_shorts:
        log("No short TRIG/NEAR/READY setups present — skipping email send.", level="info")
        if dry_run:
            log("DRY-RUN set — no email would be sent anyway.", level="debug")
        return

    subject_counts = f"{len(trig_shorts)} TRIG / {len(near_shorts)} NEAR / {len(cover_shorts)} READY"
    if dry_run:
        log("DRY-RUN set — would send email (short TRIG/NEAR/READY present).", level="warn")
    else:
        log("Sending email...", level="step")
        send_email(
            subject=f"Short Intraday Watch — {subject_counts}",
            html_body=html,
            text_body=text,
            cfg_path=_config_path,
        )
        log("Email sent.", level="ok")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--quiet", action="store_true", help="reduce console noise")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="comma list of tickers to restrict evaluation (e.g. CRM,FDS)",
    )
    ap.add_argument(
        "--test-ease",
        action="store_true",
        help="enable trigger easing for testing (or set INTRADAY_TEST=1)",
    )
    ap.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="path to write per-ticker diagnostics CSV",
    )
    ap.add_argument(
        "--log-json",
        type=str,
        default="",
        help="path to write per-ticker diagnostics JSON",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="don’t send email"
    )
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = (
        [s.strip().upper() for s in args.only.split(",") if s.strip()]
        if args.only
        else None
    )

    try:
        run(
            _config_path=args.config,
            only_tickers=only,
            test_ease=args.test_ease,
            log_csv=args.log_csv or None,
            log_json=args.log_json or None,
            dry_run=args.dry_run,
        )
        log("Short tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
