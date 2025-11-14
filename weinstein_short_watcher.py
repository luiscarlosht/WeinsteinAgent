#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Short Intraday Watcher

Short-side companion to weinstein_intraday_watcher.py.

Logic (Weinstein-style):
- Universe: Weekly Stage 4 (Downtrend) names from latest weekly CSV.
- SHORT TRIGGER:
    * Stage 4 weekly
    * Breakdown under ~10-week pivot LOW and/or 30-wk MA proxy (SMA150),
      by ≈0.4% (SHORT_BREAK_PCT) with:
        - Full-day volume pace vs 50dma ≥ 1.3× (if available)
        - For 60m bars: ≥40 min elapsed in current bar & intrabar pace ≥ 1.2×
- NEAR SHORT:
    * Stage 4 weekly
    * Price hanging just above pivot/MA breakdown zone (within small buffer),
      with volume pace ≥ 1.0× (if available).

Order Block (for each candidate):
- Entry: current price (for simplicity; you can refine to exact breakdown levels)
- Stop: ≈ MA150 + 3% (if MA150 available; otherwise entry + 20%)
- Targets: -15% and -20% from entry price (T1, T2)

Email:
- Title: 'Weinstein Short Intraday Watch — {timestamp}'
- Sections:
    Short Triggers (ranked)
    Near Short Setups (ranked)
    Charts (Price + SMA150, RS / SPY)
    Snapshot table
"""

import os
import io
import json
import math
import base64
import argparse
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
CRYPTO_BENCHMARK  = "BTC-USD"  # kept for chart RS normalization, though shorts are equities

INTRADAY_INTERVAL = "60m"     # '60m' or '30m'
LOOKBACK_DAYS = 60

PIVOT_LOOKBACK_WEEKS = 10

# Volume pace thresholds
VOL_PACE_MIN = 1.30          # for confirmed TRIGGER shorts
NEAR_VOL_PACE_MIN = 1.00     # for NEAR shorts

# Breakdown geometry
SHORT_BREAK_PCT = 0.004      # ≈0.4% under pivot low / MA150 to count as breakdown

# 60m intrabar confirmation
INTRABAR_CONFIRM_MIN_ELAPSED = 40        # minutes
INTRABAR_VOLPACE_MIN = 1.20             # intrabar pace vs avg

# Stop / target logic (short)
SHORT_STOP_MA_BUFFER_PCT = 0.03         # stop ≈ MA150 * 1.03
SHORT_FALLBACK_STOP_PCT = 0.20          # if no MA150, 20% above entry
SHORT_TARGET1_PCT = 0.15                # -15% target
SHORT_TARGET2_PCT = 0.20                # -20% target

# Charting
PRICE_WINDOW_DAYS = 260
SMA_DAYS = 150
CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

VERBOSE = True

# ---------------- Small helpers ----------------
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str, *, level: str = "info"):
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
        f for f in os.listdir(WEEKLY_OUTPUT_DIR)
        if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No weekly CSV found in {WEEKLY_OUTPUT_DIR}. "
            "Run weinstein_report_weekly.py first."
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
    if not set(["High", "Low", "Close"]).issubset(set(sub.columns)):
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
    """
    Approximate 10-week pivot LOW using daily data.
    For equities: ~5 trading days/week; for crypto: 7.
    """
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

    For equities: 09:30–16:00 ET pacing (13:30–20:00 UTC).
    For crypto: midnight–midnight UTC pacing (24/7).
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
        fraction = min(1.0, max(0.05, elapsed / (24.0 * 3600.0)))
    else:
        minutes = now.hour * 60 + now.minute
        start = 13 * 60 + 30
        end = 20 * 60
        if minutes <= start:
            fraction = 0.05
        elif minutes >= end:
            fraction = 1.0
        else:
            fraction = (minutes - start) / (6.5 * 60.0)
            fraction = min(1.0, max(0.05, fraction))

    est_full = today_vol / fraction if fraction > 0 else today_vol
    return float(_safe_div(est_full, v50)) if pd.notna(v50) and v50 > 0 else np.nan

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

    # approximate elapsed fraction in current bar
    try:
        if isinstance(intraday_df.columns, pd.MultiIndex):
            ts = intraday_df[("Close", ticker)].dropna().index[-1]
        else:
            ts = intraday_df["Close"].dropna().index[-1]
        last_bar_start = pd.Timestamp(ts).to_pydatetime()
        from datetime import datetime as _dt
        elapsed = max(0, int((_dt.utcnow() - last_bar_start).total_seconds() // 60))
    except Exception:
        elapsed = bar_minutes  # assume full bar if unknown

    frac = min(1.0, max(0.05, elapsed / float(bar_minutes)))
    est_full = last_bar_vol / frac if frac > 0 else last_bar_vol
    return float(_safe_div(est_full, avg_bar_vol))

def stage_order(stage: str) -> int:
    if isinstance(stage, str):
        if stage.startswith("Stage 4"):  # we only care about Stage 4, but keep ranking idea
            return 0
    return 9

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

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
    rs = close_t / close_b
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
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=7,
        loc="upper left",
        frameon=False,
    )

    chart_path = os.path.join(
        CHART_DIR,
        f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# ---------------- Order block helpers (short) ----------------
def _fmt_num(x):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"

def _short_entry_stop_targets(px, ma30):
    """
    For short:
      - Entry: current price
      - Stop: MA150 * (1 + SHORT_STOP_MA_BUFFER_PCT) if MA150 exists,
              else entry * (1 + SHORT_FALLBACK_STOP_PCT)
      - Targets: entry * (1 - 15%), entry * (1 - 20%)
    """
    if pd.isna(px):
        return np.nan, np.nan, np.nan, np.nan
    entry = float(px)
    if pd.notna(ma30):
        stop = float(ma30) * (1.0 + SHORT_STOP_MA_BUFFER_PCT)
    else:
        stop = entry * (1.0 + SHORT_FALLBACK_STOP_PCT)
    t1 = entry * (1.0 - SHORT_TARGET1_PCT)
    t2 = entry * (1.0 - SHORT_TARGET2_PCT)
    return entry, stop, t1, t2

# ---------------- Main logic ----------------
def run(_config_path="./config.yaml", *, only_tickers=None, test_ease=False, dry_run=False):
    log(f"Short watcher starting with config: {_config_path}", level="step")
    cfg, benchmark, sheet_url, svc_file = load_config(_config_path)
    weekly_df, weekly_csv_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # Normalize columns
    w = weekly_df.rename(columns=str.lower)
    for miss in ["ticker", "stage", "ma30", "rs_above_ma", "asset_class"]:
        if miss not in w.columns:
            w[miss] = np.nan

    # Universe: Stage 4 (Downtrend)
    focus = w[w["stage"].astype(str).str.startswith("Stage 4")][
        ["ticker", "stage", "ma30", "rs_above_ma", "asset_class"]
    ].copy()

    if "rank" in w.columns:
        focus["weekly_rank"] = w["rank"]
    else:
        focus["weekly_rank"] = 999999

    if only_tickers:
        filt = set([t.strip().upper() for t in only_tickers])
        focus = focus[focus["ticker"].isin(filt)].copy()

    log(f"Short universe: {len(focus)} symbols (Stage 4).", level="info")

    # Benchmarks for RS charts
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

    # Easing for testing
    ease = test_ease or (os.getenv("SHORT_TEST", "0") == "1")
    if ease:
        log("TEST-EASE: lowering thresholds for quick validation.", level="warn")

    _VOL_PACE_MIN = 0.0 if ease else VOL_PACE_MIN
    _NEAR_VOL_PACE_MIN = 0.0 if ease else NEAR_VOL_PACE_MIN
    _INTRABAR_CONFIRM_MIN_ELAPSED = 0 if ease else INTRABAR_CONFIRM_MIN_ELAPSED
    _INTRABAR_VOLPACE_MIN = 0.0 if ease else INTRABAR_VOLPACE_MIN

    # Collections
    near_shorts = []
    trig_shorts = []
    info_rows = []
    chart_imgs = []

    log("Evaluating short candidates...", level="step")

    for _, row in focus.iterrows():
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

        # Intrabar metrics (60m)
        elapsed = None
        pace_intra = None
        if INTRADAY_INTERVAL == "60m":
            pace_intra = intrabar_volume_pace(intraday, t, avg_window=20, bar_minutes=60)
            try:
                if isinstance(intraday.columns, pd.MultiIndex):
                    ts = intraday[("Close", t)].dropna().index[-1]
                else:
                    ts = intraday["Close"].dropna().index[-1]
                last_bar_start = pd.Timestamp(ts).to_pydatetime()
                from datetime import datetime as _dt
                elapsed = max(0, int((_dt.utcnow() - last_bar_start).total_seconds() // 60))
            except Exception:
                elapsed = None

        # Gating
        full_pace_ok = (pd.isna(pace_full) or pace_full >= _VOL_PACE_MIN)
        near_pace_ok = (pd.isna(pace_full) or pace_full >= _NEAR_VOL_PACE_MIN)
        intrabar_ok = True
        if INTRADAY_INTERVAL == "60m":
            intrabar_ok = (
                (elapsed is None or elapsed >= _INTRABAR_CONFIRM_MIN_ELAPSED)
                and (pd.isna(pace_intra) or pace_intra >= _INTRABAR_VOLPACE_MIN)
            )

        # Breakdown / near geometry
        pivot_ok = pd.notna(pivot_low)
        ma_ok = pd.notna(ma30)

        breakdown_ref = None
        if pivot_ok and ma_ok:
            breakdown_ref = max(pivot_low, ma30)
        elif pivot_ok:
            breakdown_ref = pivot_low
        elif ma_ok:
            breakdown_ref = ma30

        cond_short_trigger = False
        cond_short_near = False

        if breakdown_ref is not None and pd.notna(px):
            breakdown_level = breakdown_ref * (1.0 - SHORT_BREAK_PCT)

            # TRIGGER: price ≤ breakdown_level with volume + intrabar checks
            if (px <= breakdown_level) and full_pace_ok and intrabar_ok:
                cond_short_trigger = True
            else:
                # NEAR: hanging just above breakdown zone
                upper_near = breakdown_ref * (1.0 + SHORT_BREAK_PCT)
                if (px > breakdown_level) and (px <= upper_near) and near_pace_ok:
                    cond_short_near = True

        # Build info row (for snapshot table)
        info_rows.append(
            {
                "ticker": t,
                "stage": stage,
                "price": px,
                "ma30": ma30,
                "pivot_low_10w": pivot_low,
                "vol_pace_vs50dma": None if pd.isna(pace_full) else round(float(pace_full), 2),
                "short_state": (
                    "TRIG" if cond_short_trigger else ("NEAR" if cond_short_near else "IDLE")
                ),
            }
        )

        # Classify & store
        entry, stop, t1, t2 = _short_entry_stop_targets(px, ma30)

        base_item = {
            "ticker": t,
            "price": px,
            "entry": entry,
            "stop": stop,
            "t1": t1,
            "t2": t2,
            "stage": stage,
            "weekly_rank": weekly_rank,
            "ma30": ma30,
            "pivot_low": pivot_low,
            "pace_full": pace_full,
            "pace_intra": pace_intra,
            "atr": atr,
        }

        if cond_short_trigger:
            trig_shorts.append(base_item)
        elif cond_short_near:
            near_shorts.append(base_item)

    log(f"Scan done. Shorts → NEAR:{len(near_shorts)} TRIG:{len(trig_shorts)}", level="info")

    # -------- Ranking & charts --------
    def short_sort_key(item):
        wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
        st = stage_order(item.get("stage", ""))
        px = item.get("price", np.nan)
        ma = item.get("ma30", np.nan)
        dist = (px - ma) if (pd.notna(px) and pd.notna(ma)) else 0.0
        pace = item.get("pace_full", np.nan)
        pace = pace if pd.notna(pace) else -1e9
        return (wr, st, dist, -pace)

    near_shorts.sort(key=short_sort_key)
    trig_shorts.sort(key=short_sort_key)

    charts_added = 0
    for item in trig_shorts:
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
    def bullets(items, kind: str) -> str:
        if not items:
            return f"<p>No {kind} shorts.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            entry = it.get("entry", np.nan)
            stop = it.get("stop", np.nan)
            t1 = it.get("t1", np.nan)
            t2 = it.get("t2", np.nan)
            if kind == "TRIG":
                tag = "SHORT TRIG"
            else:
                tag = "NEAR short"
            lis.append(
                f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                f"({tag}, entry≈{entry:.2f}, stop≥{stop:.2f}, "
                f"targets↓ [{t1:.2f}, {t2:.2f}], {it['stage']}, weekly {wr_str})</li>"
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
      SHORT TRIGGER: Weekly Stage 4 (Downtrend) + breakdown under ~10-week pivot low and/or
      30-wk MA proxy (SMA{SMA_DAYS}), by ≈{SHORT_BREAK_PCT*100:.1f}% with volume pace ≥ {VOL_PACE_MIN:.1f}×
      and intrabar checks (≥{INTRABAR_CONFIRM_MIN_ELAPSED} min, pace ≥ {INTRABAR_VOLPACE_MIN:.1f}×).<br>
      NEAR-SHORT: Stage 4 + price hanging just above the pivot/MA breakdown zone,
      volume pace ≥ {NEAR_VOL_PACE_MIN:.1f}×.<br>
      Order block: Entry ≈ current price; Stop ≈ MA150 + {SHORT_STOP_MA_BUFFER_PCT*100:.1f}%;
      targets at −{SHORT_TARGET1_PCT*100:.0f}% and −{SHORT_TARGET2_PCT*100:.0f}% from entry.
    </i></p>
    """

    html += f"""
    <h4>Short Triggers (ranked)</h4>
    {bullets(trig_shorts, "TRIG")}
    <h4>Near Short Setups (ranked)</h4>
    {bullets(near_shorts, "NEAR")}
    {charts_html}
    """

    # Snapshot table
    info_df = pd.DataFrame(info_rows)
    if not info_df.empty:
        html += "<h4>Snapshot</h4>\n"
        html += info_df.to_html(index=False)

    # Plain text
    def _lines(items, kind: str):
        out = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            entry = it.get("entry", np.nan)
            stop = it.get("stop", np.nan)
            t1 = it.get("t1", np.nan)
            t2 = it.get("t2", np.nan)
            tag = "SHORT TRIG" if kind == "TRIG" else "NEAR short"
            out.append(
                f"{i}. {it['ticker']} @ {it['price']:.2f} "
                f"({tag}, entry≈{entry:.2f}, stop≥{stop:.2f}, targets↓ [{t1:.2f}, {t2:.2f}], "
                f"{it['stage']}, weekly {wr_str})"
            )
        return "\n".join(out) if out else f"No {kind} shorts."

    text = (
        f"Weinstein Short Intraday Watch — {now}\n\n"
        f"Short TRIG (ranked):\n{_lines(trig_shorts, 'TRIG')}\n\n"
        f"Near Shorts (ranked):\n{_lines(near_shorts, 'NEAR')}\n"
    )

    # Save HTML
    os.makedirs("./output", exist_ok=True)
    html_path = os.path.join(
        "./output",
        f"short_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    )
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    subject = f"Short Intraday Watch — {len(near_shorts)} NEAR / {len(trig_shorts)} TRIG"

    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        log("Sending email...", level="step")
        send_email(
            subject=subject,
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
        help="comma list of tickers to restrict evaluation (e.g. FDS,CRM)",
    )
    ap.add_argument(
        "--test-ease",
        action="store_true",
        help="enable trigger easing for testing (or set SHORT_TEST=1)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="don’t send email (but still generate HTML)",
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
            dry_run=args.dry_run,
        )
        log("Short tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
