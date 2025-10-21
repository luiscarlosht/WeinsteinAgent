# === weinstein_intraday_watcher.py ===
import os, io, json, math, time, base64, yaml
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from weinstein_mailer import send_email

# ---------------- Tunables ----------------
WEEKLY_OUTPUT_DIR = "./output"            # where weekly CSVs are saved
WEEKLY_FILE_PREFIX = "weinstein_weekly_"  # we pick the newest
BENCHMARK_DEFAULT = "SPY"

INTRADAY_INTERVAL = "60m"     # '60m' or '30m'
LOOKBACK_DAYS = 60            # intraday history window
PIVOT_LOOKBACK_WEEKS = 10     # breakout pivot high window (weekly proxy)
VOL_PACE_MIN = 1.30           # today's est. full-day vol vs 50-DMA for BUY
BUY_DIST_ABOVE_MA_MIN = 0.00  # >= 0% above 30-wk MA proxy (SMA150)
CONFIRM_BARS = 2              # require last N bars >= pivot & >= MA proxy

# Breakout quality guards
MIN_BREAKOUT_PCT = 0.005      # +0.5% above pivot for BUY confirm
REQUIRE_RISING_BAR_VOL = True
INTRADAY_AVG_VOL_WINDOW = 20
INTRADAY_LASTBAR_AVG_MULT = 1.20

# Near-trigger sensitivity (early heads-up)
NEAR_BELOW_PIVOT_PCT = 0.003  # within 0.3% below pivot counts as near
NEAR_VOL_PACE_MIN = 1.00      # not weak vs 50-DMA

# Risk for SELL logic (optional tracked positions)
HARD_STOP_PCT = 0.08          # 8% default hard stop for tracked positions
TRAIL_ATR_MULT = 2.0          # ATR(14d) trailing stop multiplier

STATE_FILE = "./state/positions.json"     # track entries/stops (optional)
CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

# Chart look
PRICE_WINDOW_DAYS = 260   # ~1y of trading days
SMA_DAYS = 150            # ~30-week MA proxy

# ---------------- Config / IO ----------------
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
    # normalize column names (some pandas versions may lowercase)
    cols = {c.lower(): c for c in df.columns}
    # Ensure 'ticker' and 'stage' exist
    def getcol(name):
        return cols.get(name, name) if name in cols else name
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

# ---------------- Data helpers ----------------
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
    v50 = v.rolling(50).mean().iloc[-2] if len(v) > 50 else np.nan
    today_vol = v.iloc[-1]
    # crude intraday fraction estimate (UTC time heuristic):
    now = datetime.utcnow()
    minutes = now.hour * 60 + now.minute
    start = 13*60 + 30  # 13:30 UTC (09:30 ET)
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
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

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
    fig.tight_layout(pad=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# ---------------- Ranking helpers ----------------
def stage_order(stage: str) -> int:
    # Stage 2 first (0), then Stage 1 (1), others later
    if isinstance(stage, str):
        if stage.startswith("Stage 2"):
            return 0
        if stage.startswith("Stage 1"):
            return 1
    return 9

def buy_sort_key(item):
    """
    Sort BUY by:
    weekly_rank asc, stage_order asc, volume pace desc, px/pivot desc, px/ma desc
    """
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    pace = item.get("pace", np.nan)
    pace = pace if pd.notna(pace) else -1e9
    px = item.get("price", np.nan)
    pivot = item.get("pivot", np.nan)
    ma = item.get("ma30", np.nan)
    ratio_pivot = (px / pivot) if (pd.notna(px) and pd.notna(pivot) and pivot != 0) else -1e9
    ratio_ma = (px / ma) if (pd.notna(px) and pd.notna(ma) and ma != 0) else -1e9
    return (wr, st, -pace, -ratio_pivot, -ratio_ma)

def near_sort_key(item):
    """
    Sort NEAR by:
    weekly_rank asc, stage_order asc, |px - pivot| asc (closer better), volume pace desc
    """
    wr = int(item.get("weekly_rank", 999999)) if pd.notna(item.get("weekly_rank", np.nan)) else 999999
    st = stage_order(item.get("stage", ""))
    px = item.get("price", np.nan)
    pivot = item.get("pivot", np.nan)
    dist = abs(px - pivot) if (pd.notna(px) and pd.notna(pivot)) else 1e9
    pace = item.get("pace", np.nan)
    pace = pace if pd.notna(pace) else -1e9
    return (wr, st, dist, -pace)

# ---------------- Logic ----------------
def run():
    cfg, benchmark = load_config()
    weekly_df, weekly_csv_path = load_weekly_report()

    # Pull & normalize the columns we need from weekly
    # Expect columns: ticker, stage, ma30, rs_above_ma, rank (from weekly report)
    # Normalize in case casing differs
    wcols = {c.lower(): c for c in weekly_df.columns}
    col_ticker = wcols.get("ticker", "ticker")
    col_stage  = wcols.get("stage", "stage")
    col_ma30   = wcols.get("ma30", "ma30")
    col_rs_abv = wcols.get("rs_above_ma", "rs_above_ma")
    col_rank   = wcols.get("rank", "rank") if "rank" in wcols else None

    focus = weekly_df[
        weekly_df[col_stage].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])
    ][[col_ticker, col_stage, col_ma30, col_rs_abv] + ([col_rank] if col_rank else [])].copy()

    focus.rename(columns={
        col_ticker: "ticker",
        col_stage: "stage",
        col_ma30: "ma30",
        col_rs_abv: "rs_above_ma",
        (col_rank if col_rank else "rank"): "weekly_rank"
    }, inplace=True)

    # Safety: if no weekly_rank, fill with large numbers to de-prioritize uniformly
    if "weekly_rank" not in focus.columns:
        focus["weekly_rank"] = 999999

    tickers = sorted(set(focus["ticker"].tolist() + [benchmark]))

    intraday, daily = get_intraday(tickers)

    # Current prices from intraday
    if isinstance(intraday.columns, pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    state = load_positions()
    held = state.get("positions", {})

    buy_signals = []
    near_signals = []
    sell_signals = []
    info_rows = []
    chart_imgs = []

    for _, row in focus.iterrows():
        t = row["ticker"]
        if t == benchmark:
            continue

        # Last price
        px = float(last_closes.get(t, np.nan)) if (hasattr(last_closes, "index") and t in last_closes.index) else (float(last_closes.values[-1]) if len(last_closes.values) else np.nan)
        if np.isnan(px):
            continue

        stage = str(row["stage"])
        ma30 = float(row.get("ma30", np.nan))
        rs_above = bool(row.get("rs_above_ma", False))
        weekly_rank = float(row.get("weekly_rank", np.nan))
        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma(t, daily)

        # Two-bar confirm above pivot(+headroom) AND above MA proxy
        closes_n = get_last_n_intraday_closes(intraday, t, n=max(CONFIRM_BARS, 2))
        ma_ok = (pd.notna(ma30))
        pivot_ok = pd.notna(pivot)
        if ma_ok and pivot_ok and closes_n:
            confirm = all(
                (c >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and
                (c >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN))
                for c in closes_n[-CONFIRM_BARS:]
            )
        else:
            confirm = False

        # Volume confirmation on last bar
        vol_ok = True
        if REQUIRE_RISING_BAR_VOL:
            vols2 = get_last_n_intraday_volumes(intraday, t, n=2)
            vavg = get_intraday_avg_volume(intraday, t, window=INTRADAY_AVG_VOL_WINDOW)
            if len(vols2) >= 2 and pd.notna(vavg) and vavg > 0:
                vol_ok = (vols2[-1] > vols2[-2]) and (vols2[-1] >= INTRADAY_LASTBAR_AVG_MULT * vavg)
            else:
                vol_ok = False

        rs_ok = rs_above

        # BUY signal
        if (
            stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)")
            and confirm
            and rs_ok
            and (pd.isna(pace) or pace >= VOL_PACE_MIN)
            and vol_ok
        ):
            item = {
                "ticker": t,
                "price": px,
                "pivot": pivot,
                "pace": None if pd.isna(pace) else float(pace),
                "stage": stage,
                "ma30": ma30,
                "weekly_rank": weekly_rank,
            }
            buy_signals.append(item)

            # charts for top candidates only (we'll sort first, then attach)
        else:
            # ---- NEAR-TRIGGER logic (early heads-up) ----
            if stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)") and rs_ok and pivot_ok and ma_ok and pd.notna(px):
                near = False
                above_ma = px >= ma30 * (1.0 + BUY_DIST_ABOVE_MA_MIN)
                if above_ma:
                    # within 0.3% below pivot and not yet a full confirm
                    if (px >= pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)) and (px < pivot * (1.0 + MIN_BREAKOUT_PCT)):
                        near = True
                    # or first close above pivot*(1+headroom) but lacking 2-bar confirm
                    elif (px >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not confirm:
                        near = True

                vol_near_ok = (pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN)

                if near and vol_near_ok:
                    near_signals.append({
                        "ticker": t,
                        "price": px,
                        "pivot": pivot,
                        "pace": None if pd.isna(pace) else float(pace),
                        "stage": stage,
                        "ma30": ma30,
                        "weekly_rank": weekly_rank,
                        "reason": "near pivot/confirm"
                    })

        # SELL logic for tracked positions
        pos = held.get(t)
        if pos:
            entry = float(pos.get("entry", np.nan))
            hard_stop = float(pos.get("stop", np.nan)) if pd.notna(pos.get("stop", np.nan)) else (entry * (1 - HARD_STOP_PCT) if pd.notna(entry) else np.nan)
            atr = compute_atr(daily, t, n=14)
            trail = px - TRAIL_ATR_MULT * atr if pd.notna(atr) else None

            breach_hard = (pd.notna(hard_stop) and px <= hard_stop)
            breach_ma = (pd.notna(ma30) and px <= ma30 * 0.97)  # 3% below MA30
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
                    "stage": stage,
                    "weekly_rank": weekly_rank
                })

        info_rows.append({
            "ticker": t,
            "stage": stage,
            "price": px,
            "ma30": ma30,
            "pivot10w": pivot,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace), 2),
            "two_bar_confirm": confirm,
            "last_bar_vol_ok": vol_ok if 'vol_ok' in locals() else None,
            "weekly_rank": weekly_rank
        })

    # -------- Ranking & charts for top-n --------
    buy_signals.sort(key=buy_sort_key)
    near_signals.sort(key=near_sort_key)

    # add inline charts for top-ranked buys first, then near (until limit)
    charts_added = 0
    chart_imgs = []
    for item in buy_signals:
        if charts_added >= MAX_CHARTS_PER_EMAIL:
            break
        t = item["ticker"]
        path, data_uri = make_tiny_chart_png(t, benchmark, daily)
        if data_uri:
            chart_imgs.append((t, data_uri))
            charts_added += 1
    if charts_added < MAX_CHARTS_PER_EMAIL:
        for item in near_signals:
            if charts_added >= MAX_CHARTS_PER_EMAIL:
                break
            t = item["ticker"]
            path, data_uri = make_tiny_chart_png(t, benchmark, daily)
            if data_uri:
                chart_imgs.append((t, data_uri))
                charts_added += 1

    # -------- Build Email --------
    info_df = pd.DataFrame(info_rows)
    # Order snapshot roughly by recommendation: weekly_rank, stage_order, ticker
    if not info_df.empty:
        info_df["stage_rank"] = info_df["stage"].apply(stage_order)
        info_df["weekly_rank"] = pd.to_numeric(info_df["weekly_rank"], errors="coerce").fillna(999999).astype(int)
        info_df = info_df.sort_values(["weekly_rank", "stage_rank", "ticker"]).drop(columns=["stage_rank"])

    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} signals.</p>"
        lis = []
        for i, it in enumerate(items, start=1):
            pace_val = it.get("pace", None)
            pace_str = "—" if (pace_val is None or pd.isna(pace_val)) else f"{pace_val:.2f}x"
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "SELL":
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} — {it['reasons']} "
                    f"({it['stage']}, weekly {wr_str})</li>"
                )
            else:
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"(pivot {it['pivot']:.2f}, pace {pace_str}, {it['stage']}, weekly {wr_str})</li>"
                )
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
      BUY: Weekly Stage 1/2 + two-bar confirm over ~10-week pivot & 30-wk MA proxy (SMA150),
      +{MIN_BREAKOUT_PCT*100:.1f}% headroom, RS support, volume pace ≥ {VOL_PACE_MIN}×,
      and last bar vol rising & ≥ {INTRADAY_LASTBAR_AVG_MULT:.1f}× intraday {INTRADAY_AVG_VOL_WINDOW}-bar avg.<br>
      NEAR-TRIGGER: Stage 1/2 + RS ok, price within {NEAR_BELOW_PIVOT_PCT*100:.1f}% below pivot or first close over pivot but not fully confirmed yet,
      volume pace ≥ {NEAR_VOL_PACE_MIN}×.
    </i></p>
    <h4>Buy Triggers (ranked)</h4>
    {bullets(buy_signals, "BUY")}
    <h4>Near-Triggers (ranked)</h4>
    {bullets(near_signals, "NEAR")}
    {charts_html}
    <h4>Sell / Risk Triggers (Tracked Positions)</h4>
    {bullets(sell_signals, "SELL")}
    <h4>Snapshot (ordered by weekly rank & stage)</h4>
    {info_df.to_html(index=False)}
    """

    # Plain-text
    text = f"Weinstein Intraday Watch — {now}\n\nBUY (ranked):\n"
    for i, b in enumerate(buy_signals, start=1):
        pace_str = "—" if (b.get("pace") is None or pd.isna(b.get("pace"))) else f"{b['pace']:.2f}x"
        wr = b.get("weekly_rank", None)
        wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
        text += f"{i}. {b['ticker']} @ {b['price']:.2f} (pivot {b['pivot']:.2f}, pace {pace_str}, {b['stage']}, weekly {wr_str})\n"

    text += "\nNEAR-TRIGGER (ranked):\n"
    for i, n in enumerate(near_signals, start=1):
        pace_str = "—" if (n.get("pace") is None or pd.isna(n.get("pace"))) else f"{n['pace']:.2f}x"
        wr = n.get("weekly_rank", None)
        wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
        text += f"{i}. {n['ticker']} @ {n['price']:.2f} (pivot {n['pivot']:.2f}, pace {pace_str}, {n['stage']}, weekly {wr_str})\n"

    text += "\nSELL:\n"
    for i, s in enumerate(sell_signals, start=1):
        wr = s.get("weekly_rank", None)
        wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
        text += f"{i}. {s['ticker']} @ {s['price']:.2f} — {s['reasons']} ({s['stage']}, weekly {wr_str})\n"

    send_email(
        subject=f"Intraday Watch — {len(buy_signals)} BUY / {len(near_signals)} NEAR / {len(sell_signals)} SELL",
        html_body=html,
        text_body=text,
        cfg_path="config.yaml"
    )

if __name__ == "__main__":
    run()
