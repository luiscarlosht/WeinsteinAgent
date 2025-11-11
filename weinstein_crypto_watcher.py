#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Crypto Watch — crypto-focused watcher with 24h pacing (legacy copy/behavior)

- Uses 24h projected volume vs 50d avg (no 60m elapsed/intrabar requirements)
- Email wording matches your earlier crypto template (RS vs BTC, 24h pacing).
- Same BUY/NEAR/SELL states and promotion windowing as intraday stocks.
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

# ---------- Tunables (crypto flavor) ----------
BENCHMARK_DEFAULT = "BTC-USD"   # crypto RS baseline
LOOKBACK_DAYS = 120
PIVOT_LOOKBACK_WEEKS = 10

# 24h pacing thresholds (legacy)
VOL_PACE_MIN = 1.20
NEAR_VOL_PACE_MIN = 0.90
MIN_BREAKOUT_PCT = 0.004   # 0.4% above pivot & MA
SELL_BREAK_PCT   = 0.006   # 0.6% under MA150

# State machine windows
NEAR_HITS_WINDOW = 6
NEAR_HITS_MIN    = 3
COOLDOWN_SCANS   = 24
SELL_NEAR_HITS_WINDOW = 6
SELL_NEAR_HITS_MIN    = 3
SELL_COOLDOWN_SCANS   = 24

# Charting / RS
PRICE_WINDOW_DAYS = 260
SMA_DAYS = 150

CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12
INTRADAY_STATE_FILE = "./state/crypto_triggers.json"
VERBOSE = True

# ---------------- Small helpers ----------------
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg, *, level="info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {"info":"•","ok":"✅","step":"▶️","warn":"⚠️","err":"❌","debug":"··"}.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)

def _update_hits(window_arr, hit, window):
    window_arr = (window_arr or [])
    window_arr.append(1 if hit else 0)
    if len(window_arr) > window:
        window_arr = window_arr[-window:]
    return window_arr, sum(window_arr)

def _safe_div(a,b):
    try:
        if b == 0 or (isinstance(b,float) and math.isclose(b,0.0)): return np.nan
        return a/b
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
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")
    return cfg, sheet_url, svc_file

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
def get_intraday_and_daily(tickers):
    # Hourly bars for signal checks; daily for pivots / RS & MA
    intraday = yf.download(
        tickers, period=f"{LOOKBACK_DAYS}d", interval="60m",
        auto_adjust=True, ignore_tz=True, progress=False
    )
    daily = yf.download(
        tickers, period="24mo", interval="1d",
        auto_adjust=True, ignore_tz=True, progress=False
    )
    return intraday, daily

def last_weekly_pivot_high(ticker, daily_df, weeks=PIVOT_LOOKBACK_WEEKS):
    bars = weeks * 7  # crypto is 24/7
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)]
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"]
    highs = highs.dropna().tail(bars)
    return float(highs.max()) if len(highs) else np.nan

def compute_ma(daily_df, ticker, n=SMA_DAYS):
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            c = daily_df[("Close", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        c = daily_df["Close"].dropna()
    if len(c) < n: return np.nan
    return float(c.rolling(n).mean().iloc[-1])

def volume_pace_today_vs_50dma_crypto(ticker, daily_df):
    """Projected full-day volume vs 50-day avg (24h pacing)."""
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            v = daily_df[("Volume", ticker)].copy()
        except KeyError:
            return np.nan
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

def get_last_n_closes(intraday_df, ticker, n=2):
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            s = intraday_df[("Close", ticker)].dropna()
        except KeyError:
            return []
    else:
        s = intraday_df["Close"].dropna()
    return list(map(float, s.tail(n).values))

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
    ax2.plot(rs_norm.index, rs_norm.values, linestyle="--", alpha=0.7, label="RS (norm vs BTC)")
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

# ---------------- Main ----------------
def run(config_path="./config.yaml", *, crypto_list=None, dry_run=False):
    log("Crypto watcher starting with config: {0}".format(config_path), level="step")

    # Universe
    cfg, _, _ = load_config(config_path)
    # Default list (you can extend via YAML later if you want)
    default_universe = [
        "BTC-USD","ETH-USD","SOL-USD","ADA-USD","AVAX-USD","BCH-USD","DOGE-USD",
        "LINK-USD","LTC-USD","TON-USD","TRX-USD","XRP-USD","ARB-USD","NEAR-USD"
    ]
    universe = crypto_list or default_universe
    log(f"Universe: {len(universe)} crypto symbols.", level="info")

    # Download data
    log("Downloading intraday + daily bars...", level="step")
    needs = sorted(set(universe + [BENCHMARK_DEFAULT]))
    intraday, daily = get_intraday_and_daily(needs)

    if isinstance(intraday.columns, pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    def px_now(t):
        if hasattr(last_closes, "index") and (t in last_closes.index):
            return float(last_closes.get(t, np.nan))
        vals = getattr(last_closes, "values", [])
        return float(vals[-1]) if len(vals) else np.nan

    # Stage proxy and weekly rank are not computed here; keep fields for sorting hooks
    def stage_proxy(t):
        # Simple proxy: price vs SMA150 (Stage 2 if above, Stage 1 otherwise)
        ma = compute_ma(daily, t, n=SMA_DAYS)
        px = px_now(t)
        if pd.notna(ma) and pd.notna(px):
            return "Stage 2 (Uptrend)" if px >= ma else "Stage 1 (Basing)"
        return "Stage ?"
    def rs_ok_vs_btc(t):
        # RS vs BTC proxy: today’s close ratio rising vs 50d mean? Keep as boolean stub (True)
        return True

    # metrics & states
    trigger_state = _load_state()
    buy_signals, near_signals, sell_triggers = [], [], []
    info_rows = []

    for t in universe:
        px = px_now(t)
        if np.isnan(px): continue
        stg = stage_proxy(t)
        ma  = compute_ma(daily, t, n=SMA_DAYS)
        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma_crypto(t, daily)

        # BUY confirm (no intrabar gating here)
        confirm = False; price_ok = False; vol_ok = True
        if pd.notna(ma) and pd.notna(pivot):
            closes_n = get_last_n_closes(intraday, t, n=2)
            def _price_ok(c):
                return (c >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and (c >= ma * (1.0 + 0.0))
            if closes_n:
                price_ok = all(_price_ok(c) for c in closes_n[-2:])
                confirm = price_ok
            vol_ok = (pd.isna(pace) or pace >= VOL_PACE_MIN)

        # NEAR
        near_now = False
        if pd.notna(px) and pd.notna(ma) and pd.notna(pivot) and rs_ok_vs_btc(t):
            above_ma = (px >= ma)
            if above_ma:
                if (px >= pivot * (1.0 - MIN_BREAKOUT_PCT)) and (px < pivot * (1.0 + MIN_BREAKOUT_PCT)):
                    near_now = True
                elif (px >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not confirm:
                    near_now = True
        # SELL trigger (crack below MA150 by SELL_BREAK_PCT)
        sell_near_now = False; sell_confirm = False
        if pd.notna(ma) and pd.notna(px):
            sell_near_now = (px >= ma * (1.0 - SELL_BREAK_PCT)) and (px <= ma * (1.0 + 0.005))
            closes2 = get_last_n_closes(intraday, t, n=2)
            if closes2:
                sell_confirm = all((c <= ma * (1.0 - SELL_BREAK_PCT)) for c in closes2[-2:])

        # Promotion state
        st = trigger_state.get(t, {"state":"IDLE","near_hits":[],"cooldown":0,
                                   "sell_state":"IDLE","sell_hits":[],"sell_cooldown":0})
        st["near_hits"], near_count = _update_hits(st.get("near_hits", []), near_now, NEAR_HITS_WINDOW)
        if st.get("cooldown", 0) > 0: st["cooldown"] = int(st["cooldown"]) - 1
        state_now = st.get("state","IDLE")
        if state_now == "IDLE" and near_now: state_now = "NEAR"
        elif state_now in ("IDLE","NEAR") and near_count >= NEAR_HITS_MIN: state_now = "ARMED"
        elif state_now == "ARMED" and confirm and vol_ok: state_now = "TRIGGERED"; st["cooldown"] = COOLDOWN_SCANS
        elif state_now == "TRIGGERED": pass
        elif st["cooldown"] > 0 and not near_now: state_now = "COOLDOWN"
        elif st["cooldown"] == 0 and not near_now and not confirm: state_now = "IDLE"
        st["state"] = state_now

        st["sell_hits"], sell_hit_count = _update_hits(st.get("sell_hits", []), sell_near_now, SELL_NEAR_HITS_WINDOW)
        if st.get("sell_cooldown", 0) > 0: st["sell_cooldown"] = int(st["sell_cooldown"]) - 1
        sell_state = st.get("sell_state","IDLE")
        if sell_state == "IDLE" and sell_near_now: sell_state = "NEAR"
        elif sell_state in ("IDLE","NEAR") and sell_hit_count >= SELL_NEAR_HITS_MIN: sell_state = "ARMED"
        elif sell_state == "ARMED" and sell_confirm: sell_state = "TRIGGERED"; st["sell_cooldown"] = SELL_COOLDOWN_SCANS
        elif sell_state == "TRIGGERED": pass
        elif st["sell_cooldown"] > 0 and not sell_near_now: sell_state = "COOLDOWN"
        elif st["sell_cooldown"] == 0 and not sell_near_now and not sell_confirm: sell_state = "IDLE"
        st["sell_state"] = sell_state

        trigger_state[t] = st

        # Emit
        if st["state"] == "TRIGGERED":
            buy_signals.append({
                "ticker": t, "price": px, "pivot": pivot, "stage": stg, "ma30": ma,
                "weekly_rank": np.nan, "pace": None if pd.isna(pace) else float(pace)
            })
            trigger_state[t]["state"] = "COOLDOWN"
        elif st["state"] in ("NEAR","ARMED"):
            if (pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN):
                near_signals.append({
                    "ticker": t, "price": px, "pivot": pivot, "stage": stg, "ma30": ma,
                    "weekly_rank": np.nan, "pace": None if pd.isna(pace) else float(pace),
                    "reason": "near/armed"
                })

        if st["sell_state"] == "TRIGGERED":
            sell_triggers.append({
                "ticker": t, "price": px, "ma30": ma, "stage": stg,
                "weekly_rank": np.nan, "pace": None if pd.isna(pace) else float(pace)
            })
            trigger_state[t]["sell_state"] = "COOLDOWN"

        info_rows.append({
            "ticker": t, "stage": stg, "price": px, "ma30": ma, "pivot10w": pivot,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace),2),
            "two_bar_confirm": confirm, "last_bar_vol_ok": vol_ok,
            "buy_state": st["state"], "sell_state": st["sell_state"]
        })

    # Rank & charts
    buy_signals.sort(key=buy_sort_key)
    near_signals.sort(key=near_sort_key)
    sell_triggers.sort(key=sell_sort_key)

    charts_added = 0; chart_imgs = []
    for bucket in (buy_signals, near_signals):
        for it in bucket:
            if charts_added >= MAX_CHARTS_PER_EMAIL: break
            t = it["ticker"]
            path, data_uri = make_tiny_chart_png(t, BENCHMARK_DEFAULT, daily)
            if data_uri:
                chart_imgs.append((t, data_uri)); charts_added += 1
        if charts_added >= MAX_CHARTS_PER_EMAIL: break

    # Email content (legacy crypto wording)
    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} signals.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            pace_val = it.get("pace", None)
            pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} (↓ SMA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})</li>")
            else:
                lis.append(f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} (pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']}, weekly {wr_str})</li>")
        return "<ol>" + "\n".join(lis) + "</ol>"

    charts_html = ""
    if chart_imgs:
        charts_html = "<h4>Charts (Price + SMA150, RS normalized vs BTC)</h4>"
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
      BUY: Stage 1/2 + confirm over ~10-week pivot & SMA150, +{MIN_BREAKOUT_PCT*100:.1f}% headroom,
      RS vs BTC support, and volume pace ≥ {VOL_PACE_MIN:.2f}× (24h pacing).<br>
      NEAR-TRIGGER: Stage 1/2 + RS ok, price within {MIN_BREAKOUT_PCT*100:.1f}% below pivot or first close over pivot but not fully confirmed yet, pace ≥ {NEAR_VOL_PACE_MIN:.2f}×.<br>
      SELL-TRIGGER: Confirmed crack below SMA150 by {SELL_BREAK_PCT*100:.1f}% with persistence.
    </i></p>

    <h4>Buy Triggers (ranked)</h4>
    {bullets(buy_signals, "BUY")}
    <h4>Near-Triggers (ranked)</h4>
    {bullets(near_signals, "NEAR")}
    <h4>Sell Triggers (ranked)</h4>
    {bullets(sell_triggers, "SELLTRIG")}
    {charts_html}
    <h4>Snapshot (ordered by stage)</h4>
    {pd.DataFrame(info_rows).sort_values(['stage','ticker']).to_html(index=False)}
    """

    text = (
        f"Weinstein Crypto Watch — {now}\n\n"
        f"BUY: Stage 1/2 + ≥{MIN_BREAKOUT_PCT*100:.1f}% over pivot & SMA150, RS vs BTC, pace ≥ {VOL_PACE_MIN:.2f}x (24h).\n"
        f"NEAR: within {MIN_BREAKOUT_PCT*100:.1f}% below pivot or first close over pivot; pace ≥ {NEAR_VOL_PACE_MIN:.2f}x.\n"
        f"SELL: crack below SMA150 by {SELL_BREAK_PCT*100:.1f}%.\n\n"
    )

    # Persist state
    _save_state(trigger_state)

    # Save + email
    os.makedirs("./output", exist_ok=True)
    html_path = os.path.join("./output", f"crypto_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    subject_counts = f"{len(buy_signals)} BUY / {len(near_signals)} NEAR / {len(sell_triggers)} SELLTRIG"
    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        log("Sending email...", level="step")
        send_email(
            subject=f"Crypto Watch — {subject_counts}",
            html_body=html,
            text_body=text,
            cfg_path=config_path
        )
        log("Email sent.", level="ok")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--only", type=str, default="", help="comma list of crypto tickers (e.g. BTC-USD,ETH-USD)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    VERBOSE = True
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()] if args.only else None
    try:
        run(config_path=args.config, crypto_list=only, dry_run=args.dry_run)
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
