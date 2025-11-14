#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Short Intraday Watcher — short-sell focused engine

- Uses weekly Stage 4 (Downtrend) candidates.
- Finds intraday short setups (NEAR + TRIGGERED) using MA150 and 10-week pivot lows.
- Proposes entries, protective buy-to-cover stops, and take-profit tiers (15% / 20%).
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

# ------------ Tunables ------------
WEEKLY_OUTPUT_DIR = "./output"
WEEKLY_FILE_PREFIX = "weinstein_weekly_"
BENCHMARK_DEFAULT = "SPY"
CRYPTO_BENCHMARK  = "BTC-USD"

INTRADAY_INTERVAL = "60m"
LOOKBACK_DAYS = 60
PIVOT_LOOKBACK_WEEKS = 10

INTRADAY_AVG_VOL_WINDOW = 20
INTRABAR_CONFIRM_MIN_ELAPSED = 40
INTRABAR_VOLPACE_MIN = 1.20

VOL_PACE_MIN = 1.30
NEAR_VOL_PACE_MIN = 1.00

SHORT_BREAK_PCT = 0.004      # 0.4% under pivot/MA
SHORT_NEAR_ABOVE_PCT = 0.003 # within 0.3% above pivot/MA

# State for intraday shorts
SHORT_STATE_FILE = "./state/intraday_shorts.json"
SCAN_INTERVAL_MIN = 10
NEAR_HITS_WINDOW = 6
NEAR_HITS_MIN = 3
COOLDOWN_SCANS = 24

PRICE_WINDOW_DAYS = 260
SMA_DAYS = 150

CHART_DIR = "./output/charts"
MAX_CHARTS_PER_EMAIL = 12

# Short alert targets (profit side)
SHORT_TARGET_PCTS = [0.15, 0.20]   # 15% and 20% in your favor

VERBOSE = True

# ------------ Small helpers ------------
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg, *, level="info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {"info":"•", "ok":"✅", "step":"▶️", "warn":"⚠️", "err":"❌", "debug":"··"}.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)

def _safe_div(a,b):
    try:
        if b == 0 or (isinstance(b,float) and math.isclose(b,0.0)):
            return np.nan
        return a/b
    except Exception:
        return np.nan

def _is_crypto(sym: str) -> bool:
    return (sym or "").upper().endswith("-USD")

def _load_short_state():
    p = SHORT_STATE_FILE
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        with open(p,"r") as f:
            return json.load(f)
    return {}

def _save_short_state(st):
    with open(SHORT_STATE_FILE,"w") as f:
        json.dump(st,f,indent=2)

def _update_hits(window_arr, hit, window):
    window_arr = (window_arr or [])
    window_arr.append(1 if hit else 0)
    if len(window_arr) > window:
        window_arr = window_arr[-window:]
    return window_arr, sum(window_arr)

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

def _elapsed_in_current_bar_minutes(intraday_df, ticker):
    try:
        if isinstance(intraday_df.columns, pd.MultiIndex):
            ts = intraday_df[("Close",ticker)].dropna().index[-1]
        else:
            ts = intraday_df["Close"].dropna().index[-1]
        last_bar_start = pd.Timestamp(ts).to_pydatetime()
        from datetime import datetime as _dt
        return max(0,int((_dt.utcnow()-last_bar_start).total_seconds()//60))
    except Exception:
        return 0

def intrabar_volume_pace(intraday_df, ticker, avg_window=INTRADAY_AVG_VOL_WINDOW, bar_minutes=60):
    try:
        if isinstance(intraday_df.columns, pd.MultiIndex):
            v = intraday_df[("Volume",ticker)].dropna()
        else:
            v = intraday_df["Volume"].dropna()
    except Exception:
        return np.nan
    if len(v) < max(avg_window,2):
        return np.nan
    last_bar_vol = float(v.iloc[-1])
    avg_bar_vol = float(v.tail(avg_window).mean())
    elapsed = _elapsed_in_current_bar_minutes(intraday_df,ticker)
    frac = min(1.0,max(0.05,elapsed/float(bar_minutes)))
    est_full = last_bar_vol/frac if frac>0 else last_bar_vol
    return float(_safe_div(est_full,avg_bar_vol))

def compute_atr(daily_df,t,n=14):
    if isinstance(daily_df.columns,pd.MultiIndex):
        try:
            sub = daily_df.xs(t,axis=1,level=1)
        except KeyError:
            return np.nan
    else:
        sub = daily_df
    if not set(["High","Low","Close"]).issubset(set(sub.columns)):
        return np.nan
    h,l,c = sub["High"],sub["Low"],sub["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l),(h-prev_c).abs(),(l-prev_c).abs()],axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return float(atr.dropna().iloc[-1]) if len(atr.dropna()) else np.nan

def last_weekly_pivot_low(ticker,daily_df,weeks=PIVOT_LOOKBACK_WEEKS):
    bars = weeks * (7 if _is_crypto(ticker) else 5)
    if isinstance(daily_df.columns,pd.MultiIndex):
        try:
            lows = daily_df[("Low",ticker)]
        except KeyError:
            return np.nan
    else:
        lows = daily_df["Low"]
    lows = lows.dropna().tail(bars)
    return float(lows.min()) if len(lows) else np.nan

def volume_pace_today_vs_50dma(ticker,daily_df):
    if isinstance(daily_df.columns,pd.MultiIndex):
        try:
            v = daily_df[("Volume",ticker)].copy()
        except KeyError:
            return np.nan
    else:
        v = daily_df["Volume"].copy()
    if v.empty: return np.nan
    v50 = v.rolling(50).mean().iloc[-2] if len(v)>50 else np.nan
    today_vol = v.iloc[-1]
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    if _is_crypto(ticker):
        day_start = now.replace(hour=0,minute=0,second=0,microsecond=0)
        elapsed = max(0.0,(now-day_start).total_seconds())
        fraction = min(1.0,max(0.05,elapsed/(24*3600.0)))
    else:
        minutes = now.hour*60+now.minute
        start = 13*60+30
        end   = 20*60+0
        if minutes <= start:
            fraction = 0.05
        elif minutes >= end:
            fraction = 1.0
        else:
            fraction = (minutes-start)/(6.5*60)
            fraction = min(1.0,max(0.05,fraction))
    est_full = today_vol/fraction if fraction>0 else today_vol
    return float(_safe_div(est_full,v50)) if pd.notna(v50) and v50>0 else np.nan

def get_last_n_intraday_closes(intraday_df,ticker,n=2):
    if isinstance(intraday_df.columns,pd.MultiIndex):
        try:
            s = intraday_df[("Close",ticker)].dropna()
        except KeyError:
            return []
    else:
        s = intraday_df["Close"].dropna()
    return list(map(float,s.tail(n).values))

# ------------ Short logic helpers ------------
def _short_price_confirm(px, pivot_low, ma30):
    if pd.isna(px) or (pd.isna(pivot_low) and pd.isna(ma30)):
        return False
    ok_pivot = (pd.notna(pivot_low) and px <= pivot_low * (1.0 - SHORT_BREAK_PCT))
    ok_ma    = (pd.notna(ma30)     and px <= ma30      * (1.0 - SHORT_BREAK_PCT))
    # require break of at least one, prefer both
    return ok_pivot or ok_ma

def _short_near_zone(px, pivot_low, ma30):
    if pd.isna(px): return False
    zones = []
    if pd.notna(pivot_low):
        zones.append(pivot_low * (1.0 + SHORT_NEAR_ABOVE_PCT))
    if pd.notna(ma30):
        zones.append(ma30 * (1.0 + SHORT_NEAR_ABOVE_PCT))
    if not zones: return False
    upper = min(zones)  # small cushion above supports
    return px >= upper * (1.0 - 2*SHORT_NEAR_ABOVE_PCT) and px >= upper   # hanging just above breakdown zone

def _short_entry_stop_targets(px_now, pivot_low, ma30, atr):
    """
    For a fresh short:
      - Entry ≈ price now or slightly below last pivot/MA
      - Protective stop ≈ a bit above MA150 (3%) or ATR trail
      - Profit targets: 15% & 20% in your favor
    """
    if pd.isna(px_now):
        return np.nan, np.nan, [], []
    entry = float(px_now)
    # protective stop (you get hurt if price goes up)
    ma_guard = (ma30 * 1.03) if pd.notna(ma30) else np.nan
    atr_trail = (entry + 2.0*atr) if pd.notna(atr) else np.nan
    cand_stop = [v for v in [ma_guard, atr_trail] if pd.notna(v)]
    stop = max(cand_stop) if cand_stop else entry * 1.08

    # targets (profitable, lower prices)
    targets = []
    for pt in SHORT_TARGET_PCTS:
        targets.append(entry * (1.0 - pt))

    return entry, stop, targets, cand_stop

# ------------ Config / weekly ------------
def load_config(path):
    with open(path,"r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app",{}) or {}
    benchmark = app.get("benchmark",BENCHMARK_DEFAULT)
    return cfg, benchmark

def newest_weekly_csv():
    files = [f for f in os.listdir(WEEKLY_OUTPUT_DIR)
             if f.startswith(WEEKLY_FILE_PREFIX) and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No weekly CSV found in ./output.")
    files.sort(reverse=True)
    return os.path.join(WEEKLY_OUTPUT_DIR,files[0])

def load_weekly_report():
    p = newest_weekly_csv()
    df = pd.read_csv(p)
    return df,p

# ------------ Charting for email ------------
def make_tiny_chart_png(ticker, benchmark, daily_df):
    os.makedirs(CHART_DIR, exist_ok=True)
    if isinstance(daily_df.columns,pd.MultiIndex):
        try:
            close_t = daily_df[("Close",ticker)].dropna()
            close_b = daily_df[("Close",benchmark)].dropna()
        except KeyError:
            return None,None
    else:
        return None,None
    close_t = close_t.tail(PRICE_WINDOW_DAYS)
    close_b = close_b.reindex_like(close_t).dropna()
    idx = close_t.index.intersection(close_b.index)
    close_t,close_b = close_t.loc[idx],close_b.loc[idx]
    if len(close_t)<50 or len(close_b)<50:
        return None,None
    sma = close_t.rolling(SMA_DAYS).mean()
    rs = (close_t/close_b); rs_norm = rs/rs.iloc[0]
    fig,ax1 = plt.subplots(figsize=(5.0,2.4),dpi=150)
    ax1.plot(close_t.index,close_t.values,label=f"{ticker}")
    ax1.plot(sma.index,sma.values,label=f"SMA{SMA_DAYS}",linewidth=1.2)
    ax1.set_ylabel("Price")
    ax1.tick_params(axis='x',labelsize=8); ax1.tick_params(axis='y',labelsize=8)
    ax2 = ax1.twinx()
    ax2.plot(rs_norm.index,rs_norm.values,linestyle="--",alpha=0.7,label="RS (norm)")
    ax2.set_ylabel("RS (norm)")
    ax2.tick_params(axis='y',labelsize=8)
    ax1.set_title(f"{ticker} — Price, SMA150, RS/{benchmark}",fontsize=9)
    ax1.grid(alpha=0.2)
    lines1,labels1 = ax1.get_legend_handles_labels()
    lines2,labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2,labels1+labels2,fontsize=7,loc="upper left",frameon=False)
    chart_path = os.path.join(CHART_DIR,f"{ticker}_short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.tight_layout(pad=0.8); fig.savefig(chart_path,bbox_inches="tight"); plt.close(fig)
    with open(chart_path,"rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"

# ------------ Main short run ------------
def run_short(_config_path="./config.yaml", *, only_tickers=None, test_ease=False, dry_run=False):
    log(f"Short watcher starting with config: {_config_path}", level="step")
    cfg, benchmark = load_config(_config_path)
    weekly_df, weekly_path = load_weekly_report()
    log(f"Weekly CSV: {weekly_path}", level="debug")

    w = weekly_df.rename(columns=str.lower)
    for col in ["ticker","stage","ma30","rs_above_ma","asset_class"]:
        if col not in w.columns:
            w[col] = np.nan

    # Universe: Stage 4 (Downtrend) only for now
    focus = w[w["stage"].astype(str).str.startswith("Stage 4")][["ticker","stage","ma30","rs_above_ma","asset_class"]].copy()
    if "rank" in w.columns:
        focus["weekly_rank"] = w["rank"]
    else:
        focus["weekly_rank"] = 999999

    if only_tickers:
        filt = set(t.strip().upper() for t in only_tickers)
        focus = focus[focus["ticker"].isin(filt)]

    log(f"Short universe: {len(focus)} symbols (Stage 4).", level="info")

    needs = sorted(set(focus["ticker"].tolist() + [benchmark, CRYPTO_BENCHMARK]))
    log("Downloading intraday + daily bars...", level="step")
    intraday, daily = get_intraday(needs)
    log("Price data downloaded.", level="ok")

    if isinstance(intraday.columns,pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    def px_now(t):
        if hasattr(last_closes,"index") and (t in last_closes.index):
            return float(last_closes.get(t,np.nan))
        vals = getattr(last_closes,"values",[])
        return float(vals[-1]) if len(vals) else np.nan

    state = _load_short_state()

    near_shorts, short_triggers = [], []
    info_rows, debug_rows = [], []
    charts = []

    ease = test_ease or (os.getenv("INTRADAY_TEST","0")=="1")
    if ease:
        log("TEST-EASE: lowering thresholds for shorts.", level="warn")
    _NEAR_HITS_MIN = 1 if ease else NEAR_HITS_MIN
    _INTRABAR_CONFIRM_MIN_ELAPSED = 0 if ease else INTRABAR_CONFIRM_MIN_ELAPSED
    _INTRABAR_VOLPACE_MIN = 0.0 if ease else INTRABAR_VOLPACE_MIN

    log("Evaluating short candidates...", level="step")

    for _, row in focus.iterrows():
        t = row["ticker"]
        if t in (benchmark, CRYPTO_BENCHMARK):
            continue
        px = px_now(t)
        if np.isnan(px):
            continue

        stage = str(row["stage"])
        ma30 = float(row.get("ma30",np.nan))
        weekly_rank = float(row.get("weekly_rank",np.nan))
        pivot_low = last_weekly_pivot_low(t,daily,weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma(t,daily)
        atr = compute_atr(daily,t,n=14)

        elapsed = _elapsed_in_current_bar_minutes(intraday,t) if INTRADAY_INTERVAL=="60m" else None
        pace_intra = intrabar_volume_pace(intraday,t,bar_minutes=60) if INTRADAY_INTERVAL=="60m" else None

        closes_n = get_last_n_intraday_closes(intraday,t,n=2)

        d = {"cond":{}, "metrics":{}}
        d["metrics"].update({
            "price": px, "ma30": ma30, "pivot_low": pivot_low, "atr": atr,
            "pace_full_vs50dma": None if pd.isna(pace) else float(pace),
            "pace_intrabar": None if pd.isna(pace_intra) else float(pace_intra),
            "elapsed_min": elapsed,
        })

        # conditions
        d["cond"]["ma_ok"] = pd.notna(ma30)
        d["cond"]["pivot_ok"] = pd.notna(pivot_low)

        # short near + confirm
        near_now = False
        confirm = False
        vol_ok = True

        # near if hanging above breakdown zone
        if pd.notna(px) and (pd.notna(pivot_low) or pd.notna(ma30)):
            near_now = _short_near_zone(px,pivot_low,ma30)

        if d["cond"]["ma_ok"] or d["cond"]["pivot_ok"]:
            if INTRADAY_INTERVAL=="60m":
                price_ok = _short_price_confirm(px,pivot_low,ma30)
                vol_ok = (pd.isna(pace_intra) or pace_intra>=_INTRABAR_VOLPACE_MIN)
                confirm = price_ok and (elapsed is not None and elapsed>=_INTRABAR_CONFIRM_MIN_ELAPSED) and vol_ok
            else:
                if closes_n:
                    price_ok = all(_short_price_confirm(c,pivot_low,ma30) for c in closes_n)
                    confirm = price_ok

        d["cond"]["short_near_now"] = bool(near_now)
        d["cond"]["short_confirm"] = bool(confirm)
        d["cond"]["short_vol_ok"] = bool(vol_ok)
        d["cond"]["pace_full_gate"] = (pd.isna(pace) or pace>=VOL_PACE_MIN)
        d["cond"]["near_pace_gate"] = (pd.isna(pace) or pace>=NEAR_VOL_PACE_MIN)

        key = t
        st = state.get(key,{
            "short_state":"IDLE",
            "short_hits":[],
            "short_cooldown":0
        })

        # NEAR hit tracking
        st["short_hits"], near_count = _update_hits(st.get("short_hits",[]), near_now, NEAR_HITS_WINDOW)
        if st.get("short_cooldown",0)>0:
            st["short_cooldown"] = int(st["short_cooldown"]) - 1

        short_state = st.get("short_state","IDLE")
        if short_state=="IDLE" and near_now:
            short_state="NEAR"
        elif short_state in ("IDLE","NEAR") and near_count>=_NEAR_HITS_MIN:
            short_state="ARMED"
        elif short_state=="ARMED" and confirm and vol_ok and (pd.isna(pace) or pace>=VOL_PACE_MIN):
            short_state="TRIGGERED"; st["short_cooldown"]=COOLDOWN_SCANS
        elif st["short_cooldown"]>0 and not near_now:
            short_state="COOLDOWN"
        elif st["short_cooldown"]==0 and not near_now and not confirm:
            short_state="IDLE"
        st["short_state"] = short_state

        state[key]=st

        # emit
        if st["short_state"]=="TRIGGERED" and (pd.isna(pace) or pace>=VOL_PACE_MIN):
            entry, stop, targets, _ = _short_entry_stop_targets(px,pivot_low,ma30,atr)
            short_triggers.append({
                "ticker":t,
                "price": px,
                "entry": entry,
                "stop": stop,
                "targets": targets,
                "ma30": ma30,
                "pivot_low": pivot_low,
                "stage": stage,
                "weekly_rank": weekly_rank,
                "atr": atr,
                "pace": None if pd.isna(pace) else float(pace),
            })
            state[key]["short_state"]="COOLDOWN"
        elif st["short_state"] in ("NEAR","ARMED"):
            if (pd.isna(pace) or pace>=NEAR_VOL_PACE_MIN):
                entry, stop, targets, _ = _short_entry_stop_targets(px,pivot_low,ma30,atr)
                near_shorts.append({
                    "ticker":t,
                    "price": px,
                    "entry": entry,
                    "stop": stop,
                    "targets": targets,
                    "ma30": ma30,
                    "pivot_low": pivot_low,
                    "stage": stage,
                    "weekly_rank": weekly_rank,
                    "atr": atr,
                    "pace": None if pd.isna(pace) else float(pace),
                    "reason":"near/armed",
                })

        info_rows.append({
            "ticker":t,
            "stage":stage,
            "price":px,
            "ma30":ma30,
            "pivot_low_10w":pivot_low,
            "vol_pace_vs50dma": None if pd.isna(pace) else round(float(pace),2),
            "short_state":st["short_state"],
        })
        debug_rows.append({
            "ticker":t,
            **d["metrics"],
            **{f"cond_{k}":v for k,v in d["cond"].items()},
            "short_state":st["short_state"],
        })

    log(f"Scan done. Shorts → NEAR:{len(near_shorts)} TRIG:{len(short_triggers)}", level="info")

    # rank by weekly + distance from MA
    def _short_sort_key(it):
        wr = it.get("weekly_rank",999999)
        ma = it.get("ma30",np.nan)
        px = it.get("price",np.nan)
        dist = (ma-px) if (pd.notna(ma) and pd.notna(px)) else 0.0
        return (wr, -dist)

    near_shorts.sort(key=_short_sort_key)
    short_triggers.sort(key=_short_sort_key)

    # charts
    charts = []
    charts_added = 0
    for item in short_triggers:
        if charts_added>=MAX_CHARTS_PER_EMAIL: break
        t = item["ticker"]
        bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
        path,data_uri = make_tiny_chart_png(t,bmk,daily)
        if data_uri:
            charts.append((t,data_uri)); charts_added += 1
    if charts_added<MAX_CHARTS_PER_EMAIL:
        for item in near_shorts:
            if charts_added>=MAX_CHARTS_PER_EMAIL: break
            t = item["ticker"]
            bmk = CRYPTO_BENCHMARK if _is_crypto(t) else BENCHMARK_DEFAULT
            path,data_uri = make_tiny_chart_png(t,bmk,daily)
            if data_uri:
                charts.append((t,data_uri)); charts_added += 1

    log(f"Charts prepared: {len(charts)}", level="debug")

    # email sections
    def bullets(items,kind):
        if not items:
            return f"<p>No {kind} shorts.</p>"
        lis = []
        for i,it in enumerate(items,1):
            wr = it.get("weekly_rank",None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            px = it.get("price",np.nan)
            entry = it.get("entry",np.nan)
            stop = it.get("stop",np.nan)
            targets = it.get("targets",[]) or []
            tgt_str = ", ".join(f"{targets[j]:.2f}" for j in range(len(targets)))
            if kind=="TRIG":
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {px:.2f} — "
                    f"entry≈{entry:.2f}, stop≥{stop:.2f}, targets↓ [{tgt_str}] "
                    f"({it.get('stage','')}, weekly {wr_str})</li>"
                )
            else:
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {px:.2f} (NEAR short, entry≈{entry:.2f}, stop≥{stop:.2f}, "
                    f"targets↓ [{tgt_str}], {it.get('stage','')}, weekly {wr_str})</li>"
                )
        return "<ol>" + "\n".join(lis) + "</ol>"

    charts_html = ""
    if charts:
        charts_html = "<h4>Charts (Price + SMA150, RS / benchmark)</h4>"
        for t,data_uri in charts:
            charts_html += f"""
            <div style="display:inline-block;margin:6px 8px 10px 0;vertical-align:top;text-align:center;">
              <img src="{data_uri}" alt="{t}" style="border:1px solid #eee;border-radius:6px;max-width:320px;">
              <div style="font-size:12px;color:#555;margin-top:3px;">{t}</div>
            </div>
            """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <h3>Weinstein Short Intraday Watch — {now}</h3>
    <p style="font-size:13px;">
      SHORT TRIGGER: Weekly Stage 4 (Downtrend) + breakdown under ~10-week pivot low and/or 30-wk MA proxy (SMA150),
      by ≈{SHORT_BREAK_PCT*100:.1f}% with volume pace ≥ {VOL_PACE_MIN}× and intrabar checks (≥{INTRABAR_CONFIRM_MIN_ELAPSED} min, pace ≥ {INTRABAR_VOLPACE_MIN}×).<br>
      NEAR-SHORT: Stage 4 + price hanging just above the pivot/MA breakdown zone, volume pace ≥ {NEAR_VOL_PACE_MIN}×.
    </p>
    <h4>Short Triggers (ranked)</h4>
    {bullets(short_triggers,"TRIG")}
    <h4>Near Short Setups (ranked)</h4>
    {bullets(near_shorts,"NEAR")}
    {charts_html}
    <h4>Snapshot</h4>
    {pd.DataFrame(info_rows).to_html(index=False)}
    """

    # text body
    def _lines(items,kind):
        out=[]
        for i,it in enumerate(items,1):
            wr = it.get("weekly_rank",None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            px=it.get("price",np.nan)
            entry=it.get("entry",np.nan)
            stop=it.get("stop",np.nan)
            targets=it.get("targets",[]) or []
            tgt_str=", ".join(f"{x:.2f}" for x in targets)
            if kind=="TRIG":
                out.append(f"{i}. {it['ticker']} @ {px:.2f} — entry≈{entry:.2f}, stop≥{stop:.2f}, targets↓ [{tgt_str}] ({it.get('stage','')}, weekly {wr_str})")
            else:
                out.append(f"{i}. {it['ticker']} @ {px:.2f} (NEAR short, entry≈{entry:.2f}, stop≥{stop:.2f}, targets↓ [{tgt_str}], {it.get('stage','')}, weekly {wr_str})")
        return "\n".join(out) if out else f"No {kind} shorts."

    text = (
        f"Weinstein Short Intraday Watch — {now}\n\n"
        f"Short Triggers:\n{_lines(short_triggers,'TRIG')}\n\n"
        f"Near Short Setups:\n{_lines(near_shorts,'NEAR')}\n"
    )

    # persist short state
    _save_short_state(state)

    # save HTML
    os.makedirs("./output",exist_ok=True)
    html_path = os.path.join("./output",f"short_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    try:
        with open(html_path,"w",encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    subject_counts = f"{len(short_triggers)} SHORT / {len(near_shorts)} NEAR"
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

# ------------ Main ------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--only", type=str, default="", help="comma-separated list of tickers")
    ap.add_argument("--test-ease", action="store_true", help="easier thresholds for testing")
    ap.add_argument("--dry-run", action="store_true", help="don't send email")
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()] if args.only else None

    log(f"Short watcher starting with config: {args.config}", level="step")
    try:
        run_short(
            _config_path=args.config,
            only_tickers=only,
            test_ease=args.test_ease,
            dry_run=args.dry_run,
        )
        log("Short tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
