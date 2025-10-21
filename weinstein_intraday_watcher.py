# === weinstein_intraday_watcher.py ===
import os, io, json, math, time, base64, yaml
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from weinstein_mailer import send_email

# -------- Tunables --------
WEEKLY_OUTPUT_DIR = "./output"            # where weekly CSVs are saved
WEEKLY_FILE_PREFIX = "weinstein_weekly_"  # we pick the newest
BENCHMARK_DEFAULT = "SPY"

INTRADAY_INTERVAL = "60m"   # '60m' or '30m'
LOOKBACK_DAYS = 60          # intraday history window
PIVOT_LOOKBACK_WEEKS = 10   # breakout pivot high window (weekly proxy)
VOL_PACE_MIN = 1.30         # today's volume pace vs 50-DMA
BUY_DIST_ABOVE_MA_MIN = 0.00  # >= 0% above 30-wk MA proxy
CONFIRM_BARS = 2            # require last N bars >= pivot & >= MA proxy

HARD_STOP_PCT = 0.08        # 8% default hard stop for tracked positions
TRAIL_ATR_MULT = 2.0        # ATR(14d) trailing stop multiplier

STATE_FILE = "./state/positions.json"     # track entries/stops (optional)
CHART_DIR = "./output/charts"             # where tiny PNGs go
MAX_CHARTS_PER_EMAIL = 12                 # to keep emails light

# Chart look
PRICE_WINDOW_DAYS = 260   # ~1y of trading days
SMA_DAYS = 150            # ~30-week MA proxy


# -------- Config / IO --------
def load_config(path="config.yaml"):
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

# -------- Data helpers --------
def _safe_div(a, b):
    try:
        if b == 0 or (isinstance(b, float) and math.isclose(b, 0.0)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

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
    bars = weeks * 5  # proxy for N weeks
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
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            v = daily_df[("Volume", ticker)].copy()
        except KeyError:
            return np.nan
    else:
        v = daily_df["Volume"].copy()
    if v.empty:
        return np.nan
    # Use second-to-last as complete prior day average; avoid partial last bar
    v50 = v.rolling(50).mean().iloc[-2] if len(v) > 50 else np.nan
    today_vol = v.iloc[-1]
    # crude intraday fraction estimate (UTC time heuristic):
    now = datetime.utcnow()
    # US cash session approx 13:30–20:00 UTC (9:30–16:00 ET) -> 6.5h
    minutes = now.hour * 60 + now.minute
    start = 13*60 + 30
    fraction = (minutes - start) / (6.5*60)
    fraction = min(1.0, max(0.1, fraction))
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

# -------- Charting --------
def make_tiny_chart_png(ticker, benchmark, daily_df):
    """
    Returns (chart_path, base64_img) for inline embedding.
    Price (last ~260 trading days) + 150d SMA; RS line normalized.
    """
    os.makedirs(CHART_DIR, exist_ok=True)
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            close_t = daily_df[("Close", ticker)].dropna()
            close_b = daily_df[("Close", benchmark)].dropna()
        except KeyError:
            return None, None
    else:
        # single symbol case unlikely here
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
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(rs_norm.index, rs_norm.values, linestyle="--", alpha=0.7, label="RS (norm)")
    ax2.set_ylabel("RS (norm)")
    ax2.tick_params(axis='y', labelsize=8)

    ax1.set_title(f"{ticker} — Price, SMA{SMA_DAYS}, RS/{benchmark}", fontsize=9)
    ax1.grid(alpha=0.2)

    # Compose small legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lgd = ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left", frameon=False)

    chart_path = os.path.join(CHART_DIR, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.tight_layout(pad=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# -------- Logic --------
def run():
    cfg, benchmark = load_config()
    weekly_df, weekly_csv_path = load_weekly_report()

    # Focus universe: Stage 1 + Stage 2 from the weekly file
    focus = weekly_df[weekly_df["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])].copy()
    tickers = sorted(set(focus["ticker"].tolist() + [benchmark]))

    intraday, daily = get_intraday(tickers)

    # current prices from intraday
    if isinstance(intraday.columns, pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    state = load_positions()
    held = state.get("positions", {})

    buy_signals = []
    sell_signals = []
    info_rows = []
    chart_imgs = []  # list[(ticker, data_uri)]

    for _, row in focus.iterrows():
        t = row["ticker"]
        if t == benchmark:
            continue

        # last price
        px = float(last_closes.get(t, np.nan)) if t in last_closes.index else np.nan
        if np.isnan(px):
            continue

        stage = row["stage"]
        ma30 = float(row.get("ma30", np.nan))
        rs_above = bool(row.get("rs_above_ma", False))
        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma(t, daily)

        # Two-bar confirmation: last N intraday closes >= pivot AND >= MA proxy
        closes_n = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
        ma_ok = (pd.notna(ma30))
        if ma_ok and closes_n and pd.notna(pivot):
            confirm = all((c >= pivot) and (c >= ma30*(1 + BUY_DIST_ABOVE_MA_MIN)) for c in closes_n[-CONFIRM_BARS:])
        else:
            confirm = False

        rs_ok = rs_above  # use weekly CSV RS flag (weekly truth)

        # BUY: weekly Stage1/2, two-bar confirm above pivot & MA, RS ok, volume pace supportive (if available)
        if (
            stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)")
            and confirm
            and rs_ok
            and (pd.isna(pace) or pace >= VOL_PACE_MIN)
        ):
            buy_signals.append({
                "ticker": t,
                "price": px,
                "pivot": pivot,
                "pace": None if pd.isna(pace) else float(pace),
                "stage": stage
            })
            # Chart (limit to MAX_CHARTS_PER_EMAIL)
            if len(chart_imgs) < MAX_CHARTS_PER_EMAIL:
                _, data_uri = make_tiny_chart_png(t, benchmark, daily)
                if data_uri:
                    chart_imgs.append((t, data_uri))

        # SELL logic for held positions
        pos = held.get(t)
        if pos:
            entry = float(pos.get("entry", np.nan))
            hard_stop = float(pos.get("stop", np.nan)) if pd.notna(pos.get("stop", np.nan)) else entry * (1 - HARD_STOP_PCT)
            atr = compute_atr(daily, t, n=14)
            trail = px - TRAIL_ATR_MULT * atr if pd.notna(atr) else None

            breach_hard = (pd.notna(hard_stop) and px <= hard_stop)
            breach_ma = (pd.notna(ma30) and px <= ma30 * 0.97)  # 3% below MA30 proxy guard
            breach_trail = (trail is not None and px <= trail)

            if breach_hard or breach_ma or breach_trail:
                why = []
                if breach_hard:  why.append(f"≤ hard stop ({hard_stop:.2f})")
                if breach_ma:    why.append("≤ 30-wk MA proxy (−3%)")
                if breach_trail: why.append(f"≤ ATR trail ({TRAIL_ATR_MULT}×)")
                sell_signals.append({
                    "ticker": t,
                    "price": px,
                    "reasons": ", ".join(why),
                    "stage": stage
                })

        info_rows.append({
            "ticker": t,
            "stage": stage,
            "price": px,
            "ma30": ma30,
            "pivot10w": pivot,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace), 2),
            "two_bar_confirm": confirm
        })

    # --- Build Email ---
    info_df = pd.DataFrame(info_rows).sort_values(["stage","ticker"])

    def bullets(items, kind):
        if not items: return f"<p>No {kind} signals.</p>"
        lis = []
        for it in items:
            if kind == "BUY":
                pace_str = ("—" if it["pace"] is None else f"{it['pace']:.2f}x")
                lis.append(
                    f"<li><b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']})</li>"
                )
            else:
                lis.append(
                    f"<li><b>{it['ticker']}</b> @ {it['price']:.2f} — {it['reasons']} ({it['stage']})</li>"
                )
        return "<ul>" + "\n".join(lis) + "</ul>"

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
    <p><i>Rules: weekly Stage1/2 + two-bar confirm over ~10-week pivot & 30-wk MA proxy (SMA150), RS support, and volume pace ≥ {VOL_PACE_MIN}× 50-DMA.</i></p>
    <h4>Buy Triggers</h4>
    {bullets(buy_signals, "BUY")}
    {charts_html}
    <h4>Sell / Risk Triggers (Tracked Positions)</h4>
    {bullets(sell_signals, "SELL")}
    <h4>Snapshot</h4>
    {info_df.to_html(index=False)}
    """

    text = f"Weinstein Intraday Watch — {now}\n\nBUY:\n"
    for b in buy_signals:
        pace_str = ("—" if b["pace"] is None else f"{b['pace']:.2f}x")
        text += f"- {b['ticker']} @ {b['price']:.2f} (pivot {b['pivot']:.2f}, pace {pace_str}, {b['stage']})\n"
    text += "\nSELL:\n"
    for s in sell_signals:
        text += f"- {s['ticker']} @ {s['price']:.2f} — {s['reasons']} ({s['stage']})\n"

    send_email(
        subject=f"Intraday Watch — {len(buy_signals)} BUY / {len(sell_signals)} SELL",
        html_body=html,
        text_body=text,
        cfg_path="config.yaml"
    )

if __name__ == "__main__":
    run()
