#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Short Intraday Watcher

- Focus: Weekly Stage 4 (Downtrend) universe from latest weekly scan
- SHORT TRIGGER: breakdown under ~10-week pivot LOW and/or 30-wk MA proxy (SMA150)
                 by ≈0.4% with volume pace + intrabar checks
- NEAR SHORT: price hanging just above the breakdown zone with volume pacing
- Produces:
    * Short Triggers (ranked)
    * Near Short Setups (ranked) with proposed:
        - entry (≈ current price)
        - stop (≈ 3% above MA150 / breakdown zone)
        - targets (≈ 15% and 20% below entry)
    * Snapshot table
    * Small charts (Price + SMA150 + RS vs benchmark)
    * Optional debug CSV
"""

import os, io, json, math, base64, argparse, yaml
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
CRYPTO_BENCHMARK  = "BTC-USD"  # not really used for shorts, but kept for symmetry

INTRADAY_INTERVAL = "60m"
LOOKBACK_DAYS = 60
PIVOT_LOOKBACK_WEEKS = 10

SMA_DAYS = 150
PRICE_WINDOW_DAYS = 260

# Volume pacing
SHORT_VOL_PACE_MIN = 1.30      # full-day pace vs 50dma for TRIGGER
SHORT_NEAR_VOL_PACE_MIN = 1.00 # for NEAR shorts

# Price confirmation thresholds
SHORT_BREAK_PCT = 0.004        # ~0.4% under breakdown level
SHORT_NEAR_BAND_PCT = 0.01     # ±1% band above breakdown for "NEAR"

# Intrabar checks (only relevant for 60m)
SHORT_INTRABAR_CONFIRM_MIN_ELAPSED = 40
SHORT_INTRABAR_VOLPACE_MIN = 1.20

# Short trade planning:
SHORT_TARGET1_PCT = 0.15       # 15% profit target below entry
SHORT_TARGET2_PCT = 0.20       # 20% profit target below entry
SHORT_STOP_MA_BUFFER_PCT = 0.03  # ~3% above MA150 / breakdown zone

# Email / charting
CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

VERBOSE = True


# ---------- Small helpers ----------
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, *, level: str = "info") -> None:
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
def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}
    benchmark = app.get("benchmark", BENCHMARK_DEFAULT)
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")
    return cfg, benchmark, sheet_url, svc_file


def newest_weekly_csv() -> str:
    files = [
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            "No weekly CSV found in ./output. Run weinstein_report_weekly.py first."
        )
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR, files[0])


def load_weekly_report():
    path = newest_weekly_csv()
    df = pd.read_csv(path)
    return df, path


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
    if not {"High", "Low", "Close"}.issubset(sub.columns):
        return np.nan
    h, l, c = sub["High"], sub["Low"], sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    return float(atr.dropna().iloc[-1]) if len(atr.dropna()) else np.nan


def last_weekly_pivot_low(ticker, daily_df, weeks=PIVOT_LOOKBACK_WEEKS):
    """Approx ~10-week pivot LOW from daily bars."""
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
    """
    Projected full-day volume vs 50-day avg.
    Equities: 09:30–16:00 ET (13:30–20:00 UTC equivalent).
    Crypto: 24h UTC.
    """
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
        end = 20 * 60
        if minutes <= start:
            fraction = 0.05
        elif minutes >= end:
            fraction = 1.0
        else:
            fraction = (minutes - start) / (6.5 * 60)
            fraction = min(1.0, max(0.05, fraction))
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
        try:
            s = intraday_df[("Close", ticker)].dropna()
        except KeyError:
            return []
    else:
        s = intraday_df["Close"].dropna()
    return list(map(float, s.tail(n).values))


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


def intrabar_volume_pace(intraday_df, ticker, avg_window=20, bar_minutes=60):
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
    # Short side: Stage 4 is the main target
    if isinstance(stage, str):
        if stage.startswith("Stage 4"):
            return 0
    return 9


def short_sort_key(item):
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan)
    ma = item.get("ma30", np.nan)
    pivot_low = item.get("pivot_low", np.nan)
    # deeper below breakdown → earlier
    breakdown_level = max(ma, pivot_low) if (pd.notna(ma) and pd.notna(pivot_low)) else pivot_low
    dist_below = (breakdown_level - px) if (pd.notna(px) and pd.notna(breakdown_level)) else -1e9
    pace = item.get("pace", np.nan)
    pace = pace if pd.notna(pace) else -1e9
    return (wr, st, -dist_below, -pace)


# ---------------- ORDER BLOCK HELPERS (Short side) ----------------
def _fmt_num(x):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def _propose_short_levels(entry, ma30, pivot_low):
    """
    Short trade planning:
    - Entry: ≈ current price
    - Stop:  max(MA150, pivot_low) * (1 + SHORT_STOP_MA_BUFFER_PCT)
    - Targets: entry - 15%, entry - 20%
    """
    if pd.isna(entry):
        return np.nan, np.nan, np.nan, np.nan
    base = None
    if pd.notna(ma30) and pd.notna(pivot_low):
        base = max(ma30, pivot_low)
    elif pd.notna(ma30):
        base = ma30
    elif pd.notna(pivot_low):
        base = pivot_low
    stop = (base * (1.0 + SHORT_STOP_MA_BUFFER_PCT)) if base is not None else entry * (1.0 + SHORT_STOP_MA_BUFFER_PCT)
    t1 = entry * (1.0 - SHORT_TARGET1_PCT)
    t2 = entry * (1.0 - SHORT_TARGET2_PCT)
    return entry, stop, t1, t2


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
    rs = (close_t / close_b)
    rs_norm = rs / rs.iloc[0]

    fig, ax1 = plt.subplots(figsize=(5.0, 2.4), dpi=150)
    ax1.plot(close_t.index, close_t.values, label=f"{ticker}")
    ax1.plot(sma.index, sma.values, label=f"SMA{SMA_DAYS}", linewidth=1.2)
    ax1.set_ylabel("Price")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(rs_norm.index, rs_norm.values, linestyle="--", alpha=0.7, label="RS (norm)")
    ax2.set_ylabel("RS (norm)")
    ax2.tick_params(axis="y", labelsize=8)

    ax1.set_title(f"{ticker} — Price, SMA{SMA_DAYS}, RS/{benchmark}", fontsize=9)
    ax1.grid(alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left", frameon=False)

    chart_path = os.path.join(CHART_DIR, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.tight_layout(pad=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"


# ---------------- Main logic ----------------
def run(_config_path="./config.yaml", *, only_tickers=None, test_ease=False, log_csv=None, dry_run=False):
    log(f"Short watcher starting with config: {_config_path}", level="step")
    cfg, benchmark, sheet_url, service_account_file = load_config(_config_path)
    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker", "stage", "ma30", "asset_class"]:
        if miss not in w.columns:
            w[miss] = np.nan

    # Short universe: Stage 4 only
    short_universe = w[w["stage"].astype(str).str.startswith("Stage 4")][
        ["ticker", "stage", "ma30", "asset_class"]
    ].copy()

    if "rank" in w.columns:
        short_universe["weekly_rank"] = w["rank"]
    else:
        short_universe["weekly_rank"] = 999999

    if only_tickers:
        filt = set([t.strip().upper() for t in only_tickers])
        short_universe = short_universe[short_universe["ticker"].isin(filt)].copy()

    log(f"Short universe: {len(short_universe)} symbols (Stage 4).", level="info")

    needs = sorted(set(short_universe["ticker"].tolist() + [benchmark, CRYPTO_BENCHMARK]))
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

    short_triggers = []
    near_shorts = []
    info_rows = []
    debug_rows = []

    # Test easing
    ease = test_ease or (os.getenv("SHORT_TEST", "0") == "1")
    if ease:
        log("TEST-EASE: lowering thresholds for quick validation.", level="warn")
        _SHORT_BREAK_PCT = 0.0
        _SHORT_VOL_PACE_MIN = 0.0
        _SHORT_INTRABAR_CONFIRM_MIN_ELAPSED = 0
        _SHORT_INTRABAR_VOLPACE_MIN = 0.0
        _SHORT_NEAR_VOL_PACE_MIN = 0.0
    else:
        _SHORT_BREAK_PCT = SHORT_BREAK_PCT
        _SHORT_VOL_PACE_MIN = SHORT_VOL_PACE_MIN
        _SHORT_INTRABAR_CONFIRM_MIN_ELAPSED = SHORT_INTRABAR_CONFIRM_MIN_ELAPSED
        _SHORT_INTRABAR_VOLPACE_MIN = SHORT_INTRABAR_VOLPACE_MIN
        _SHORT_NEAR_VOL_PACE_MIN = SHORT_NEAR_VOL_PACE_MIN

    log("Evaluating short candidates...", level="step")

    for _, row in short_universe.iterrows():
        t = row["ticker"]
        if t in (benchmark, CRYPTO_BENCHMARK):
            continue

        px = px_now(t)
        if np.isnan(px):
            continue

        stage = str(row["stage"])
        ma30 = float(row.get("ma30", np.nan))
        weekly_rank = float(row.get("weekly_rank", np.nan))

        pivot_low = last_weekly_pivot_low(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace_full = volume_pace_today_vs_50dma(t, daily)
        atr = compute_atr(daily, t, n=14)

        elapsed = _elapsed_in_current_bar_minutes(intraday, t) if INTRADAY_INTERVAL == "60m" else None
        pace_intra = intrabar_volume_pace(intraday, t, bar_minutes=60) if INTRADAY_INTERVAL == "60m" else None

        # Conditions
        cond_stage_ok = stage.startswith("Stage 4")
        cond_ma_ok = pd.notna(ma30)
        cond_pivot_ok = pd.notna(pivot_low)
        breakdown_base = max(ma30, pivot_low) if (pd.notna(ma30) and pd.notna(pivot_low)) else (
            ma30 if pd.notna(ma30) else pivot_low
        )

        cond_short_price_ok = False
        cond_short_vol_ok = True
        cond_short_confirm = False
        cond_short_near_now = False

        # Price logic
        if cond_stage_ok and pd.notna(px) and pd.notna(breakdown_base):
            # Breakdown (TRIGGER)
            break_level = breakdown_base * (1.0 - _SHORT_BREAK_PCT)
            if INTRADAY_INTERVAL == "60m":
                cond_short_price_ok = px <= break_level
                cond_short_vol_ok = (
                    pd.isna(pace_intra) or pace_intra >= _SHORT_INTRABAR_VOLPACE_MIN
                )
                cond_short_confirm = bool(
                    cond_short_price_ok
                    and cond_short_vol_ok
                    and (elapsed is not None and elapsed >= _SHORT_INTRABAR_CONFIRM_MIN_ELAPSED)
                    and (pd.isna(pace_full) or pace_full >= _SHORT_VOL_PACE_MIN)
                )
            else:
                closes_n = get_last_n_intraday_closes(intraday, t, n=2)
                if closes_n:
                    cond_short_price_ok = all(c <= break_level for c in closes_n[-2:])
                    cond_short_confirm = cond_short_price_ok and (
                        pd.isna(pace_full) or pace_full >= _SHORT_VOL_PACE_MIN
                    )

            # NEAR region: just above breakdown zone
            near_floor = breakdown_base * (1.0 - SHORT_NEAR_BAND_PCT)
            near_ceiling = breakdown_base * (1.0 + SHORT_NEAR_BAND_PCT)
            if (px >= near_floor) and (px > break_level) and (px <= near_ceiling):
                if pd.isna(pace_full) or pace_full >= _SHORT_NEAR_VOL_PACE_MIN:
                    cond_short_near_now = True

        # short_state for snapshot
        if cond_short_confirm:
            short_state = "TRIG"
        elif cond_short_near_now:
            short_state = "NEAR"
        else:
            short_state = "IDLE"

        # Collect rows for snapshot & debug
        info_rows.append(
            {
                "ticker": t,
                "stage": stage,
                "price": px,
                "ma30": ma30,
                "pivot_low_10w": pivot_low,
                "vol_pace_vs50dma": None if pd.isna(pace_full) else float(pace_full),
                "short_state": short_state,
            }
        )

        debug_rows.append(
            {
                "ticker": t,
                "price": px,
                "ma30": ma30,
                "pivot_low_10w": pivot_low,
                "atr": atr,
                "pace_full_vs50dma": None if pd.isna(pace_full) else float(pace_full),
                "pace_intrabar": None if pd.isna(pace_intra) else float(pace_intra),
                "elapsed_min": elapsed,
                "cond_stage_ok": cond_stage_ok,
                "cond_ma_ok": cond_ma_ok,
                "cond_pivot_ok": cond_pivot_ok,
                "cond_short_near_now": cond_short_near_now,
                "cond_short_price_ok": cond_short_price_ok,
                "cond_short_vol_ok": cond_short_vol_ok,
                "cond_short_confirm": cond_short_confirm,
                "short_state": short_state,
            }
        )

        # Emit lists
        if cond_short_confirm:
            entry, stop, tgt1, tgt2 = _propose_short_levels(px, ma30, pivot_low)
            short_triggers.append(
                {
                    "ticker": t,
                    "price": px,
                    "entry": entry,
                    "stop": stop,
                    "target1": tgt1,
                    "target2": tgt2,
                    "ma30": ma30,
                    "pivot_low": pivot_low,
                    "stage": stage,
                    "weekly_rank": weekly_rank,
                    "pace": None if pd.isna(pace_full) else float(pace_full),
                }
            )
        elif cond_short_near_now:
            entry, stop, tgt1, tgt2 = _propose_short_levels(px, ma30, pivot_low)
            near_shorts.append(
                {
                    "ticker": t,
                    "price": px,
                    "entry": entry,
                    "stop": stop,
                    "target1": tgt1,
                    "target2": tgt2,
                    "ma30": ma30,
                    "pivot_low": pivot_low,
                    "stage": stage,
                    "weekly_rank": weekly_rank,
                    "pace": None if pd.isna(pace_full) else float(pace_full),
                }
            )

    log(f"Scan done. Shorts → NEAR:{len(near_shorts)} TRIG:{len(short_triggers)}", level="info")

    # -------- Ranking & charts --------
    short_triggers.sort(key=short_sort_key)
    near_shorts.sort(key=short_sort_key)

    charts_added = 0
    chart_imgs = []

    # Prioritize TRIG shorts, then NEAR
    for item in short_triggers:
        if charts_added >= MAX_CHARTS_PER_EMAIL:
            break
        t = item["ticker"]
        bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
        path, data_uri = make_tiny_chart_png(t, bmk, daily)
        if data_uri:
            chart_imgs.append((t, data_uri))
            charts_added += 1

    if charts_added < MAX_CHARTS_PER_EMAIL:
        for item in near_shorts:
            if charts_added >= MAX_CHARTS_PER_EMAIL:
                break
            t = item["ticker"]
            bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
            path, data_uri = make_tiny_chart_png(t, bmk, daily)
            if data_uri:
                chart_imgs.append((t, data_uri))
                charts_added += 1

    log(f"Charts prepared: {len(chart_imgs)}", level="debug")

    # -------- Build Email --------
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} shorts.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            pace_val = it.get("pace", None)
            pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
            entry = it.get("entry", np.nan)
            stop = it.get("stop", np.nan)
            t1 = it.get("target1", np.nan)
            t2 = it.get("target2", np.nan)

            if kind == "TRIG":
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(SHORT trigger, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(t1)}, {_fmt_num(t2)}], "
                    f"{it.get('stage','')}, weekly {wr_str}, pace {pace_str})</li>"
                )
            else:
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(NEAR short, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(t1)}, {_fmt_num(t2)}], "
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

    # Snapshot table
    info_df = pd.DataFrame(info_rows)
    snapshot_html = ""
    if not info_df.empty:
        info_df["stage_rank"] = info_df["stage"].apply(stage_order)
        if "weekly_rank" in w.columns:
            info_df = info_df.merge(
                w[["ticker", "rank"]].rename(columns={"rank": "weekly_rank"}),
                on="ticker",
                how="left",
            )
        else:
            info_df["weekly_rank"] = 999999
        info_df["weekly_rank"] = pd.to_numeric(info_df["weekly_rank"], errors="coerce").fillna(999999).astype(int)
        info_df = info_df.sort_values(["weekly_rank", "stage_rank", "ticker"]).drop(columns=["stage_rank"])
        snapshot_html = "<h4>Snapshot</h4>\n" + info_df.to_html(index=False, float_format=lambda x: f"{x:.6f}")

    html = f"""
    <h3>Weinstein Short Intraday Watch — {now_str}</h3>
    <p><i>
      SHORT TRIGGER: Weekly Stage 4 (Downtrend) + breakdown under ~10-week pivot low and/or 30-wk MA proxy (SMA150),
      by ≈{SHORT_BREAK_PCT*100:.1f}% with volume pace ≥ {SHORT_VOL_PACE_MIN:.1f}× and intrabar checks
      (≥{SHORT_INTRABAR_CONFIRM_MIN_ELAPSED} min, intrabar pace ≥ {SHORT_INTRABAR_VOLPACE_MIN:.1f}×).<br>
      NEAR-SHORT: Stage 4 + price hanging just above the pivot/MA breakdown zone, volume pace ≥ {SHORT_NEAR_VOL_PACE_MIN:.1f}×.
    </i></p>
    """

    html += f"""
    <h4>Short Triggers (ranked)</h4>
    {bullets(short_triggers, "TRIG")}
    <h4>Near Short Setups (ranked)</h4>
    {bullets(near_shorts, "NEAR")}
    {charts_html}
    {snapshot_html}
    """

    # Plain text body
    def _lines(items, kind):
        if not items:
            return f"No {kind} shorts."
        out = []
        for i, it in enumerate(items, 1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            entry = it.get("entry", np.nan)
            stop = it.get("stop", np.nan)
            t1 = it.get("target1", np.nan)
            t2 = it.get("target2", np.nan)
            if kind == "TRIG":
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} "
                    f"(SHORT trigger, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(t1)}, {_fmt_num(t2)}], "
                    f"{it.get('stage','')}, weekly {wr_str})"
                )
            else:
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} "
                    f"(NEAR short, entry≈{_fmt_num(entry)}, stop≥{_fmt_num(stop)}, "
                    f"targets↓ [{_fmt_num(t1)}, {_fmt_num(t2)}], "
                    f"{it.get('stage','')}, weekly {wr_str})"
                )
        return "\n".join(out)

    text = (
        f"Weinstein Short Intraday Watch — {now_str}\n\n"
        f"SHORT TRIGGERS (ranked):\n{_lines(short_triggers, 'TRIG')}\n\n"
        f"NEAR SHORT SETUPS (ranked):\n{_lines(near_shorts, 'NEAR')}\n"
    )

    # -------- Persist debug CSV (if requested) --------
    if log_csv:
        try:
            os.makedirs(os.path.dirname(log_csv), exist_ok=True)
            pd.DataFrame(debug_rows).to_csv(log_csv, index=False)
            log(f"Wrote short diagnostics CSV → {log_csv}", level="ok")
        except Exception as e:
            log(f"Failed writing short diagnostics CSV: {e}", level="warn")

    # -------- Save HTML --------
    os.makedirs("./output", exist_ok=True)
    html_path = os.path.join(
        "./output",
        f"short_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    subject_counts = f"{len(short_triggers)} TRIG / {len(near_shorts)} NEAR"
    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        log("Sending email...", level="step")
        send_email(
            subject=f"Short Intraday Watch — {subject_counts}",
            html_body=html,
            text_body=text,
            cfg_path=_config_path,
        )
        log("Email sent.", level="ok")


# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--quiet", action="store_true", help="reduce console noise")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="comma list of tickers to restrict evaluation (e.g. CRM,ADBE)",
    )
    ap.add_argument(
        "--test-ease",
        action="store_true",
        help="enable trigger easing for testing (or set SHORT_TEST=1)",
    )
    ap.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="path to write per-ticker short diagnostics CSV",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="don’t send email (still writes HTML/CSV)",
    )
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = (
        [s.strip().upper() for s in args.only.split(",") if s.strip()]
        if args.only
        else None
    )

    log(f"Short watcher starting with config: {args.config}", level="step")
    try:
        run(
            _config_path=args.config,
            only_tickers=only,
            test_ease=args.test_ease,
            log_csv=args.log_csv or None,
            dry_run=args.dry_run,
        )
        log("Short tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
