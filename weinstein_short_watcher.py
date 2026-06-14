#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Short Watcher — Stage 4 short setups (with Chapter 8 + VIX market regime filter)

- Uses weekly Weinstein scan (equities) to build a Stage 4 (Downtrend) universe
- Intraday checks around 10-week pivot LOW and 30-week MA proxy (SMA150)
- Emits:
    * TRIG shorts: breakdown below pivot/MA with volume + intrabar confirmations
    * NEAR shorts: price hanging just above breakdown zone, with basic volume pacing
    * READY-TO-CLOSE shorts: Stage 4 names that have reclaimed MA150 by ~0.5+%,
      suggesting the short thesis is weakening (time to consider covering).

- Email contains:
    * Ranked list of short triggers + near shorts + ready-to-close shorts
    * Order block: entry≈now, protective stop, 15% & 20% downside targets
    * Tiny charts for top names

- READY-TO-CLOSE shorts are filtered to tickers that appear in the Google Sheet
  "Signals" tab with Direction = SELL / SHORT (case-insensitive),
  so you only see shorts you explicitly mark there.

- NEW:
    * --log-csv / --log-json diagnostics (per-symbol metrics + conditions + state)
    * Chapter 8 + VIX market regime integration via market_regime.py:
        - If market_regime.short_ok is False, the short scan is skipped.
    * Weinstein Option 4 short regime via SPY weekly stage:
        - New shorts are only emitted when SPY itself is in Stage 4
          on the weekly report. READY-to-close still works always.
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

# Shared short-side core (used by watcher + backtest)
from weinstein_short_core import (
    INTRADAY_INTERVAL,
    PIVOT_LOOKBACK_WEEKS,
    SHORT_BREAK_PCT as CORE_SHORT_BREAK_PCT,
    NEAR_ABOVE_PIVOT_PCT as CORE_NEAR_ABOVE_PIVOT_PCT,
    VOL_PACE_MIN as CORE_VOL_PACE_MIN,
    NEAR_VOL_PACE_MIN as CORE_NEAR_VOL_PACE_MIN,
    INTRADAY_AVG_VOL_WINDOW,
    INTRADAY_LASTBAR_MULT,
    INTRABAR_CONFIRM_MIN_ELAPSED as CORE_INTRABAR_CONFIRM_MIN_ELAPSED,
    INTRABAR_VOLPACE_MIN as CORE_INTRABAR_VOLPACE_MIN,
    READY_ABOVE_MA_PCT as CORE_READY_ABOVE_MA_PCT,
    SHORT_NEAR_HITS_WINDOW as CORE_SHORT_NEAR_HITS_WINDOW,
    SHORT_NEAR_HITS_MIN as CORE_SHORT_NEAR_HITS_MIN,
    SHORT_COOLDOWN_SCANS as CORE_SHORT_COOLDOWN_SCANS,
    SHORT_HARD_STOP_PCT as CORE_SHORT_HARD_STOP_PCT,
    SHORT_TRAIL_ATR_MULT as CORE_SHORT_TRAIL_ATR_MULT,
    SHORT_MA_GUARD_PCT as CORE_SHORT_MA_GUARD_PCT,
    SHORT_TARGET1_PCT as CORE_SHORT_TARGET1_PCT,
    SHORT_TARGET2_PCT as CORE_SHORT_TARGET2_PCT,
    _short_price_break,
    _short_near_zone,
    _short_ready_to_close,
    _short_entry_stop_targets,
    eval_short_bar,
    ShortRegimeContext,
    build_short_regime_from_spy_stage,
)

# ---- Effective short-side tunables (config overrides Option B) ----
# These are the values actually used by THIS watcher. They default to the core
# constants but can be overridden from config.intraday.short.
SHORT_BREAK_PCT         = CORE_SHORT_BREAK_PCT
NEAR_ABOVE_PIVOT_PCT    = CORE_NEAR_ABOVE_PIVOT_PCT
VOL_PACE_MIN            = CORE_VOL_PACE_MIN
NEAR_VOL_PACE_MIN       = CORE_NEAR_VOL_PACE_MIN
INTRABAR_CONFIRM_MIN_ELAPSED = CORE_INTRABAR_CONFIRM_MIN_ELAPSED
INTRABAR_VOLPACE_MIN    = CORE_INTRABAR_VOLPACE_MIN
READY_ABOVE_MA_PCT      = CORE_READY_ABOVE_MA_PCT
SHORT_NEAR_HITS_WINDOW  = CORE_SHORT_NEAR_HITS_WINDOW
SHORT_NEAR_HITS_MIN     = CORE_SHORT_NEAR_HITS_MIN
SHORT_COOLDOWN_SCANS    = CORE_SHORT_COOLDOWN_SCANS
SHORT_HARD_STOP_PCT     = CORE_SHORT_HARD_STOP_PCT
SHORT_TRAIL_ATR_MULT    = CORE_SHORT_TRAIL_ATR_MULT
SHORT_MA_GUARD_PCT      = CORE_SHORT_MA_GUARD_PCT
SHORT_TARGET1_PCT       = CORE_SHORT_TARGET1_PCT
SHORT_TARGET2_PCT       = CORE_SHORT_TARGET2_PCT

# Optional: Google Sheets integration for READY filter
try:
    import gspread
except ImportError:
    gspread = None

# ---------------- Tunables ----------------
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_equities_"
BENCHMARK_DEFAULT = "SPY"

LOOKBACK_DAYS     = 60
PRICE_WINDOW_DAYS = 260
SMA_DAYS          = 150

# Stateful short triggers (file path + cadence for cron)
SHORT_STATE_FILE       = "./state/short_triggers.json"
SCAN_INTERVAL_MIN      = 10

CHART_DIR            = "./output/charts"
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


def _cell_to_str(v) -> str:
    """
    Safe cell → string helper for Signals tab.
    Avoids 'int' object has no attribute 'strip' by always casting to str.
    """
    if v is None:
        return ""
    try:
        return str(v)
    except Exception:
        return ""

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
        if f.startswith(WEEKLY_FILE_PREFIX)
        and f.endswith(".csv")
        and "YYYY" not in f
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found in {WEEKLY_OUTPUT_DIR}. "
            f"Run weinstein_report_weekly.py first."
        )

    # Pick newest generated weekly file by modified time, not filename.
    files.sort(
        key=lambda f: os.path.getmtime(os.path.join(WEEKLY_OUTPUT_DIR, f)),
        reverse=True,
    )
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
      - Direction in {SELL, SHORT} (case-insensitive)

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

    tickers = set()
    for r in rows:
        raw_t  = r.get("Ticker") or r.get("ticker")
        raw_dir = r.get("Direction") or r.get("direction")

        t = _cell_to_str(raw_t).strip().upper()
        if not t:
            continue

        direction = _cell_to_str(raw_dir).strip().upper()
        # Only rows explicitly marked as SELL/SHORT count as short positions
        if direction not in ("SELL", "SHORT"):
            continue

        tickers.add(t)

    log(f"READY filter: loaded {len(tickers)} short tickers from Signals tab '{tab_name}'.", level="info")
    return tickers


def write_empty_short_csv(path: str) -> None:
    """
    When shorts are disabled by regime, write a CSV with headers and zero rows,
    so short_signal_engine.py can read it without EmptyDataError and simply
    treat it as "no rows to summarize".
    """
    cols = [
        "ticker",
        "price",
        "ma30",
        "pivot_low",
        "atr",
        "pace_full_vs50dma",
        "pace_intrabar",
        "elapsed_min",
        "cond_short_near_now",
        "cond_short_confirm",
        "short_state",
        "short_hits",
        "short_cooldown",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(columns=cols)
    df.to_csv(path, index=False)

# ---------------- State helpers ----------------
def _load_short_state():
    """
    Robust loader for short_triggers.json.

    - Ensures ./state exists
    - On success: returns parsed JSON (dict)
    - On JSON corruption: backs up the bad file with a timestamp suffix,
      logs a warning, and returns {} (clean reset)
    - On any other error: logs and returns {}
    """
    path = SHORT_STATE_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # Corrupted JSON — back it up and reset
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{path}.corrupt_{ts}"
            os.replace(path, backup_path)
            log(
                f"Short state file corrupted (JSONDecodeError: {e}). "
                f"Backed up to {backup_path} and resetting short state to empty.",
                level="warn",
            )
        except Exception as e2:
            log(
                f"Short state file corrupted and backup failed ({e2}). "
                f"Resetting short state in-memory only.",
                level="err",
            )
        return {}
    except Exception as e:
        log(
            f"Failed to load short state file {path}: {e}. "
            f"Resetting short state to empty.",
            level="warn",
        )
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

# ---------------- Sorting helpers ----------------
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

# ---------------- Market Regime (Chapter 8 + VIX) helper ----------------
def _compute_market_regime_flags(weekly_df, benchmark):
    """
    Wrapper around market_regime.inspect().

    Returns:
      (label, long_ok, short_ok)

    - label: e.g. "BULL (Ch8+VIX)"
    - long_ok, short_ok: already include both Chapter 8 regime AND VIX gates,
      exactly as used by the backtest (--use-regime-long / --use-regime-short).
    """
    try:
        import market_regime as mr
    except ImportError:
        label = "UNKNOWN (no market_regime.py)"
        return label, True, True  # fail-open so system still runs

    # Prefer the tiny inspect helper introduced in market_regime.py
    inspect_fn = None
    for name in ("inspect", "inspect_for_intraday", "inspect_market_regime"):
        fn = getattr(mr, name, None)
        if callable(fn):
            inspect_fn = fn
            break

    if inspect_fn is None:
        # Fall back: treat as neutral, allow shorts (so watcher still runs)
        return "NEUTRAL (no inspect helper)", True, True

    try:
        label, long_ok, short_ok = inspect_fn()
        # label is already "BULL"/"BEAR"/"NEUTRAL"/"UNKNOWN"
        return f"{label} (Ch8+VIX)", bool(long_ok), bool(short_ok)
    except Exception as e:
        log(f"Market regime inspect() failed: {e}; treating as NEUTRAL.", level="warn")
        return "NEUTRAL (inspect error)", True, True

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

    # ---- Override short tunables from config.intraday.short (Option B) ----
    short_cfg = ((cfg.get("intraday") or {}).get("short") or {})

    global SHORT_BREAK_PCT, NEAR_ABOVE_PIVOT_PCT
    global VOL_PACE_MIN, NEAR_VOL_PACE_MIN
    global INTRABAR_CONFIRM_MIN_ELAPSED, INTRABAR_VOLPACE_MIN
    global READY_ABOVE_MA_PCT
    global SHORT_NEAR_HITS_WINDOW, SHORT_NEAR_HITS_MIN, SHORT_COOLDOWN_SCANS
    global SHORT_HARD_STOP_PCT, SHORT_TRAIL_ATR_MULT, SHORT_MA_GUARD_PCT
    global SHORT_TARGET1_PCT, SHORT_TARGET2_PCT

    def _g(name, default):
        val = short_cfg.get(name, default)
        return val if val is not None else default

    SHORT_BREAK_PCT              = float(_g("break_pct", CORE_SHORT_BREAK_PCT))
    NEAR_ABOVE_PIVOT_PCT         = float(_g("near_above_pivot_pct", CORE_NEAR_ABOVE_PIVOT_PCT))
    VOL_PACE_MIN                 = float(_g("vol_pace_min", CORE_VOL_PACE_MIN))
    NEAR_VOL_PACE_MIN            = float(_g("near_vol_pace_min", CORE_NEAR_VOL_PACE_MIN))
    INTRABAR_CONFIRM_MIN_ELAPSED = int(_g("intrabar_confirm_min_elapsed", CORE_INTRABAR_CONFIRM_MIN_ELAPSED))
    INTRABAR_VOLPACE_MIN         = float(_g("intrabar_volpace_min", CORE_INTRABAR_VOLPACE_MIN))
    READY_ABOVE_MA_PCT           = float(_g("ready_above_ma_pct", CORE_READY_ABOVE_MA_PCT))
    SHORT_NEAR_HITS_WINDOW       = int(_g("near_hits_window", CORE_SHORT_NEAR_HITS_WINDOW))
    SHORT_NEAR_HITS_MIN          = int(_g("near_hits_min", CORE_SHORT_NEAR_HITS_MIN))
    SHORT_COOLDOWN_SCANS         = int(_g("cooldown_scans", CORE_SHORT_COOLDOWN_SCANS))
    SHORT_HARD_STOP_PCT          = float(_g("stop_hard_pct", CORE_SHORT_HARD_STOP_PCT))
    SHORT_TRAIL_ATR_MULT         = float(_g("trail_atr_mult", CORE_SHORT_TRAIL_ATR_MULT))
    SHORT_MA_GUARD_PCT           = float(_g("ma_guard_pct", CORE_SHORT_MA_GUARD_PCT))
    SHORT_TARGET1_PCT            = float(_g("target1_pct", CORE_SHORT_TARGET1_PCT))
    SHORT_TARGET2_PCT            = float(_g("target2_pct", CORE_SHORT_TARGET2_PCT))

    log(
        f"Short config: break={SHORT_BREAK_PCT:.4f}, "
        f"near_zone=+{NEAR_ABOVE_PIVOT_PCT*100:.1f}%, "
        f"vol_pace_min={VOL_PACE_MIN:.2f}x, near_vol_pace_min={NEAR_VOL_PACE_MIN:.2f}x, "
        f"intrabar_confirm≥{INTRABAR_CONFIRM_MIN_ELAPSED}m @ pace≥{INTRABAR_VOLPACE_MIN:.2f}x, "
        f"hits_window={SHORT_NEAR_HITS_WINDOW}, hits_min={SHORT_NEAR_HITS_MIN}, cooldown={SHORT_COOLDOWN_SCANS}, "
        f"ready_above_ma={READY_ABOVE_MA_PCT*100:.2f}%, "
        f"stop_hard={SHORT_HARD_STOP_PCT*100:.0f}%, trail_ATR={SHORT_TRAIL_ATR_MULT:.1f}×, "
        f"targets↓ [{SHORT_TARGET1_PCT*100:.0f}%, {SHORT_TARGET2_PCT*100:.0f}%]",
        level="info",
    )

    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # ---- Chapter 8 + VIX market regime filter (short side) ----
    regime_label, long_ok, short_ok = _compute_market_regime_flags(weekly_df, benchmark)
    log(
        f"Market regime (Ch8+VIX): {regime_label} | long_ok={long_ok} short_ok={short_ok}",
        level="info",
    )

    # 11.7c audit: print detailed market-regime gates so we can distinguish
    # Chapter 8 regime gates from VIX / fast-crash overlays.
    try:
        import market_regime as _mr_audit
        _cfg = _mr_audit.MarketRegimeConfig(verbose=False)
        _snap = _mr_audit.detect_market_regime(_cfg)
        _long_regime, _short_regime_gate = _mr_audit._compute_long_short_flags(_snap.regime)
        _fast_crash = bool(getattr(_snap, "fast_crash", False))
        if _fast_crash:
            _long_regime_effective = False
        else:
            _long_regime_effective = _long_regime
        _vix_last, _long_vix, _short_vix = _mr_audit._compute_vix_gates(_cfg)
        _combined_long = bool(_long_regime_effective and _long_vix)
        _combined_short = bool(_short_regime_gate and _short_vix)
        _vix_txt = "nan" if pd.isna(_vix_last) else f"{_vix_last:.2f}"
        log(
            "Regime gate audit: "
            f"regime={getattr(_snap.regime, 'value', _snap.regime)} "
            f"long_regime={_long_regime} "
            f"long_regime_effective={_long_regime_effective} "
            f"short_regime={_short_regime_gate} "
            f"fast_crash={_fast_crash} "
            f"vix={_vix_txt} "
            f"long_vix={_long_vix} "
            f"short_vix={_short_vix} "
            f"combined_long={_combined_long} "
            f"combined_short={_combined_short}",
            level="info",
        )
    except Exception as e:
        log(f"Regime gate audit unavailable: {e}", level="warn")

    # If regime says "no shorts", exit early (writing a header-only CSV if requested)
    if not short_ok:
        log(
            "Chapter 8 + VIX regime filter: short side is DISABLED in current regime — "
            "skipping short scan.",
            level="warn",
        )
        if log_csv:
            try:
                write_empty_short_csv(log_csv)
                log(f"Wrote empty diagnostics CSV (shorts disabled) → {log_csv}", level="ok")
            except Exception as e:
                log(f"Failed writing empty diagnostics CSV: {e}", level="warn")
        log("Short tick complete (shorts disabled by regime).", level="ok")
        return

    # Load tickers from Signals tab for READY-TO-CLOSE filtering
    held_ready_tickers = load_ready_short_tickers_from_signals(
        cfg, sheet_url, service_account_file
    )

    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker", "stage", "ma30", "rs_above_ma"]:
        if miss not in w.columns:
            w[miss] = np.nan

    # ---- Weinstein Option 4 SPY-stage regime (shorts only if SPY in Stage 4) ----
    spy_stage = None
    try:
        spy_rows = w[w["ticker"].astype(str).str.upper() == benchmark.upper()]
        if not spy_rows.empty:
            spy_stage = spy_rows.iloc[0].get("stage", None)
    except Exception as e:
        log(f"SPY stage detection failed for short regime: {e}", level="warn")

    short_regime: ShortRegimeContext = build_short_regime_from_spy_stage(
        spy_stage,
        as_of=datetime.now().strftime("%Y-%m-%d"),
    )
    log(f"Weinstein SPY short regime: {short_regime.note}", level="info")

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
        # Delegated to SHORT CORE so PROD watcher and SIM share the same stateful trigger rules.
        st = short_state.get(
            t,
            {
                "short_state": "IDLE",
                "short_hits": [],
                "short_cooldown": 0,
            },
        )

        closes_tail = get_last_n_intraday_closes(intraday, t, n=2)

        st, flags = eval_short_bar(
            price=px,
            ma=ma30,
            pivot_low=pivot_low,
            pace_full=pace_full,
            pace_intra=pace_intra,
            elapsed_min=elapsed,
            closes_tail=closes_tail,
            state=st,
            intraday_interval=INTRADAY_INTERVAL,
            test_ease=bool(test_ease or (os.getenv("INTRADAY_TEST", "0") == "1")),
            break_pct=SHORT_BREAK_PCT,
            near_above_pivot_pct=NEAR_ABOVE_PIVOT_PCT,
            vol_pace_min=VOL_PACE_MIN,
            near_vol_pace_min=NEAR_VOL_PACE_MIN,
            intrabar_confirm_min_elapsed=INTRABAR_CONFIRM_MIN_ELAPSED,
            intrabar_volpace_min=INTRABAR_VOLPACE_MIN,
            near_hits_window=SHORT_NEAR_HITS_WINDOW,
            near_hits_min=SHORT_NEAR_HITS_MIN,
            cooldown_scans=SHORT_COOLDOWN_SCANS,
            ready_above_ma_pct=READY_ABOVE_MA_PCT,
        )

        short_state[t] = st

        short_near_now = bool(flags.get("short_near_now", False))
        short_price_ok = bool(flags.get("short_price_ok", False))
        short_vol_ok = bool(flags.get("short_vol_ok", False))
        short_confirm = bool(flags.get("short_confirm", False))
        ready_close_now = bool(flags.get("ready_close_now", False))
        short_trigger_now = bool(flags.get("short_trigger_now", False))

        cond["short_near_now"] = short_near_now
        cond["short_price_ok"] = short_price_ok
        cond["short_vol_ok"] = short_vol_ok
        cond["short_confirm"] = short_confirm
        cond["pace_full_gate"] = pd.isna(pace_full) or pace_full >= VOL_PACE_MIN
        cond["near_pace_gate"] = pd.isna(pace_full) or pace_full >= NEAR_VOL_PACE_MIN
        cond["ready_close_now"] = ready_close_now
        cond["short_trigger_now"] = short_trigger_now
        cond["spy_regime_allow_shorts"] = bool(short_regime.allow_shorts)

        # Weinstein Option 4 gate for new shorts (SPY must be Stage 4)
        allow_new_shorts = short_regime.allow_shorts

        # Emit short lists
        if (
            allow_new_shorts
            and short_trigger_now
            and cond["pace_full_gate"]
        ):
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
        elif allow_new_shorts and st["short_state"] in ("NEAR", "ARMED"):
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

        # READY-to-close list (pre-filter) — NOTE: not gated by SPY regime
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
      suggesting the short thesis is weakening and it's time to consider covering (restricted to shorts listed in your Signals tab).<br>
      SPY short regime (Option 4): new shorts are only emitted while SPY is in Stage 4 on the weekly report.
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
