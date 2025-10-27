#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_intraday_watcher.py

Intraday watcher that:
- Reads holdings + Source from Google Sheets (or ./output/Open_Positions.csv fallback)
- Pulls intraday prices from Yahoo Finance
- Computes portfolio P&L summary and per-position snapshot
- Generates BUY/HOLD/SELL for manual watchlists (Signals tab or CLI --sources/--symbols)
- Sends an HTML email via weinstein_mailer.send_email

CLI:
  python3 weinstein_intraday_watcher.py --config ./config.yaml \
      [--sources "Sarkee Capital,SuperiorStar"] \
      [--symbols "NVDA,SOXX"] \
      [--no-email] [--dry-run]

Requires:
  pip install yfinance pyyaml pandas numpy matplotlib
  (Optional for Google Sheets, with service account):
  pip install gspread google-auth

Service account:
  - Point GOOGLE_APPLICATION_CREDENTIALS to your JSON key, or
  - Put path in config under google.service_account_json

Author: you & ChatGPT
"""

from __future__ import annotations
import os, sys, io, math, json, time, base64, argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except Exception:
    HAS_GSPREAD = False

import yaml

# Local mailer (already in your project)
from weinstein_mailer import send_email

# ---------------- Tunables ----------------
INTRADAY_INTERVAL = "60m"     # '60m' or '30m'
LOOKBACK_DAYS = 60            # intraday history window for bars/volume checks
PIVOT_LOOKBACK_WEEKS = 10     # breakout pivot high window (weekly proxy)
SMA_DAYS = 150                # ~30-week MA proxy

# Breakout confirmation (for BUY)
MIN_BREAKOUT_PCT = 0.005      # +0.5% above pivot
CONFIRM_BARS = 2
VOL_PACE_MIN = 1.30           # est full-day vol / 50dma
REQUIRE_RISING_BAR_VOL = True
INTRADAY_AVG_VOL_WINDOW = 20
INTRADAY_LASTBAR_AVG_MULT = 1.20
NEAR_BELOW_PIVOT_PCT = 0.003  # near trigger buffer

# SELL risk rules for held positions
HARD_STOP_PCT = 0.12          # default: -12% loss ⇒ SELL
UNDER_MA_CUTOFF = 0.97        # price < 0.97 * SMA150 ⇒ SELL

# Output
OUTPUT_DIR = "./output"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
MAX_CHARTS = 10

# --------- Utilities ---------
def load_config(path: str) -> Tuple[dict, str]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    app = cfg.get("app", {}) or {}
    benchmark = app.get("benchmark", "SPY")
    return cfg, benchmark

def pick_col(df: pd.DataFrame, *candidates) -> Optional[str]:
    cols = list(df.columns)
    lc = {c.lower(): c for c in cols}
    for name in candidates:
        if name in cols:
            return name
        if name.lower() in lc:
            return lc[name.lower()]
    return None

def _safe_div(a, b):
    try:
        if b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan

def newest_weekly_csv() -> Optional[str]:
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("weinstein_weekly_") and f.endswith(".csv")]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(OUTPUT_DIR, files[0])

def load_weekly_stage_map() -> Dict[str, str]:
    """
    Load ticker->stage (and rs_above_ma) from latest weekly CSV if available.
    """
    path = newest_weekly_csv()
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    c_t = pick_col(df, "ticker", "symbol", "Ticker", "Symbol")
    c_s = pick_col(df, "stage", "Stage", "weinst_stage", "stage_name")
    if not (c_t and c_s):
        return {}
    stage_map = {}
    for _, r in df[[c_t, c_s]].dropna().iterrows():
        stage_map[str(r[c_t]).upper()] = str(r[c_s])
    return stage_map

# --------- Google Sheets (robust) ---------
def _open_sheet(sheet_url: str, creds_path: Optional[str]):
    if not HAS_GSPREAD:
        raise RuntimeError("gspread/google-auth not installed.")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    if creds_path and os.path.exists(creds_path):
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    else:
        # Try env var GOOGLE_APPLICATION_CREDENTIALS
        env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if env_path and os.path.exists(env_path):
            creds = Credentials.from_service_account_file(env_path, scopes=scopes)
        else:
            raise RuntimeError("Service account JSON not found. Set google.service_account_json in config or GOOGLE_APPLICATION_CREDENTIALS env.")
    gc = gspread.authorize(creds)
    return gc.open_by_url(sheet_url)

def read_sheet_tab(sheet_url: str, tab_name: str, creds_path: Optional[str]) -> pd.DataFrame:
    sh = _open_sheet(sheet_url, creds_path)
    ws = sh.worksheet(tab_name)
    rows = ws.get_all_records()
    df = pd.DataFrame(rows)
    return df

# --------- Data helpers ---------
def get_intraday_and_daily(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def last_weekly_pivot_high(ticker: str, daily_df: pd.DataFrame, weeks=PIVOT_LOOKBACK_WEEKS) -> float:
    bars = weeks * 5
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            highs = daily_df[("High", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        highs = daily_df["High"].dropna()
    highs = highs.tail(bars)
    return float(highs.max()) if len(highs) else np.nan

def compute_sma(daily_df: pd.DataFrame, ticker: str, n=SMA_DAYS) -> float:
    if isinstance(daily_df.columns, pd.MultiIndex):
        try:
            close = daily_df[("Close", ticker)].dropna()
        except KeyError:
            return np.nan
    else:
        close = daily_df["Close"].dropna()
    if len(close) < n:
        return np.nan
    return float(close.rolling(n).mean().iloc[-1])

def get_last_n(intraday_df: pd.DataFrame, ticker: str, kind="Close", n=2) -> List[float]:
    if isinstance(intraday_df.columns, pd.MultiIndex):
        try:
            s = intraday_df[(kind, ticker)].dropna()
        except KeyError:
            return []
    else:
        s = intraday_df[kind].dropna()
    return list(map(float, s.tail(n).values))

def intraday_last_price(intraday_df: pd.DataFrame, ticker: str) -> float:
    vals = get_last_n(intraday_df, ticker, "Close", 1)
    return float(vals[-1]) if vals else np.nan

def volume_pace_today_vs_50dma(ticker: str, daily_df: pd.DataFrame) -> float:
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
    # crude intraday fraction estimate (UTC → market open ~13:30)
    now = datetime.now(timezone.utc)
    minutes = now.hour * 60 + now.minute
    start = 13*60 + 30
    fraction = (minutes - start) / (6.5*60)
    fraction = min(1.0, max(0.1, fraction))
    est_full = today_vol / fraction if fraction > 0 else today_vol
    return float(_safe_div(est_full, v50)) if pd.notna(v50) and v50 > 0 else np.nan

# --------- Recommendation logic ---------
def weinstein_reco(stage: str, price: float, sma150: float, pct_gain: float) -> str:
    """
    Simple SELL/HOLD using Weinstein + risk:
      - Stage 3/4: SELL unless (price >= sma150 and pct_gain > 10%) ⇒ HOLD
      - Stage 1/2: HOLD unless (price < 0.97*sma150 or pct_gain <= -12%) ⇒ SELL
      - Unknown stage: fallback to price vs sma150 and loss rule
    """
    stg = (stage or "").strip()
    above_ma = (pd.notna(price) and pd.notna(sma150) and price >= sma150)
    if stg.startswith("Stage 3") or stg.startswith("Stage 4"):
        if above_ma and pct_gain > 10.0:
            return "HOLD"
        return "SELL"
    if stg.startswith("Stage 1") or stg.startswith("Stage 2"):
        if pd.notna(sma150) and price < UNDER_MA_CUTOFF * sma150:
            return "SELL"
        if pct_gain <= -HARD_STOP_PCT*100:
            return "SELL"
        return "HOLD"
    # Unknown stage
    if pd.notna(sma150) and price < UNDER_MA_CUTOFF * sma150:
        return "SELL"
    if pct_gain <= -HARD_STOP_PCT*100:
        return "SELL"
    return "HOLD"

def buy_signal_intraday(ticker: str, stage: str, rs_ok: bool,
                        price: float, sma150: float, pivot: float,
                        intraday_df: pd.DataFrame, daily_df: pd.DataFrame) -> Tuple[bool, bool, float]:
    """
    Return (is_buy, is_near, pace) using the intraday two-bar/pivot/MA/volume rules.
    """
    if not (stage.startswith("Stage 1") or stage.startswith("Stage 2")):
        return (False, False, np.nan)
    if not (pd.notna(price) and pd.notna(sma150) and pd.notna(pivot)):
        return (False, False, np.nan)

    last_closes = get_last_n(intraday_df, ticker, "Close", max(CONFIRM_BARS, 2))
    if not last_closes:
        return (False, False, np.nan)
    pace = volume_pace_today_vs_50dma(ticker, daily_df)
    # volume rising on last bar and >= intraday avg * multiplier
    vol_ok = True
    if REQUIRE_RISING_BAR_VOL:
        vols2 = get_last_n(intraday_df, ticker, "Volume", 2)
        vavg = np.nan
        if isinstance(intraday_df.columns, pd.MultiIndex):
            try:
                vseries = intraday_df[("Volume", ticker)].dropna()
                if len(vseries) >= INTRADAY_AVG_VOL_WINDOW:
                    vavg = float(vseries.tail(INTRADAY_AVG_VOL_WINDOW).mean())
            except KeyError:
                pass
        else:
            vseries = intraday_df["Volume"].dropna()
            if len(vseries) >= INTRADAY_AVG_VOL_WINDOW:
                vavg = float(vseries.tail(INTRADAY_AVG_VOL_WINDOW).mean())
        if len(vols2) >= 2 and pd.notna(vavg) and vavg > 0:
            vol_ok = (vols2[-1] > vols2[-2]) and (vols2[-1] >= INTRADAY_LASTBAR_AVG_MULT * vavg)
        else:
            vol_ok = False

    confirm = all(
        (c >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and
        (c >= sma150)
        for c in last_closes[-CONFIRM_BARS:]
    )

    if confirm and rs_ok and (pd.isna(pace) or pace >= VOL_PACE_MIN) and vol_ok:
        return (True, False, pace)

    # NEAR: above MA and within buffer below pivot or first close above pivot but not confirmed
    near = False
    if price >= sma150:
        if (price >= pivot * (1.0 - NEAR_BELOW_PIVOT_PCT)) and (price < pivot * (1.0 + MIN_BREAKOUT_PCT)):
            near = True
        elif (price >= pivot * (1.0 + MIN_BREAKOUT_PCT)) and not confirm:
            near = True
    if near and (pd.isna(pace) or pace >= 1.0) and rs_ok:
        return (False, True, pace)
    return (False, False, pace)

# --------- Charts ---------
def tiny_chart(data_close: pd.Series, sma: pd.Series, rs_norm: pd.Series, title: str) -> str:
    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(4.6, 2.3), dpi=150)
    ax1.plot(data_close.index, data_close.values, label="Price")
    ax1.plot(sma.index, sma.values, label=f"SMA{SMA_DAYS}", linewidth=1.2)
    ax1.grid(alpha=0.2)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)
    ax1.set_title(title, fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(rs_norm.index, rs_norm.values, linestyle="--", alpha=0.7, label="RS (norm)")
    ax2.tick_params(axis="y", labelsize=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left", frameon=False)

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(CHART_DIR, f"chart_{int(time.time()*1000)}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def make_inline_chart(ticker: str, benchmark: str, daily_df: pd.DataFrame) -> Optional[str]:
    if not isinstance(daily_df.columns, pd.MultiIndex):
        return None
    try:
        close_t = daily_df[("Close", ticker)].dropna().tail(260)
        close_b = daily_df[("Close", benchmark)].dropna()
    except KeyError:
        return None
    # align
    idx = close_t.index.intersection(close_b.index)
    if len(idx) < 50:
        return None
    close_t = close_t.loc[idx]
    close_b = close_b.loc[idx]
    sma = close_t.rolling(SMA_DAYS).mean()
    rs = close_t / close_b
    rs_norm = rs / rs.iloc[0]
    return tiny_chart(close_t, sma, rs_norm, f"{ticker} — Price, SMA{SMA_DAYS}, RS/{benchmark}")

# --------- Loading holdings & signals ---------
def load_holdings(cfg: dict) -> pd.DataFrame:
    # Try Google Sheets first
    sheet_url = (cfg.get("sheets") or {}).get("sheet_url")
    open_tab = (cfg.get("sheets") or {}).get("open_positions_tab", "Open_Positions")
    creds_path = (cfg.get("google") or {}).get("service_account_json")
    if sheet_url:
        try:
            df = read_sheet_tab(sheet_url, open_tab, creds_path)
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️  Sheets read failed ({open_tab}): {e}. Falling back to CSV.")

    # Fallback to CSV created by weekly run
    csv_guess = os.path.join(OUTPUT_DIR, "Open_Positions.csv")
    if os.path.exists(csv_guess):
        return pd.read_csv(csv_guess)
    print("⚠️  Could not load holdings from Sheets or CSV fallback.")
    return pd.DataFrame()

def load_signals(cfg: dict, tab_name: Optional[str] = None) -> pd.DataFrame:
    sheet_url = (cfg.get("sheets") or {}).get("sheet_url")
    signals_tab = tab_name or (cfg.get("sheets") or {}).get("signals_tab", "Signals")
    creds_path = (cfg.get("google") or {}).get("service_account_json")
    if sheet_url:
        try:
            df = read_sheet_tab(sheet_url, signals_tab, creds_path)
            return df
        except Exception as e:
            print(f"ℹ️  No signals tab '{signals_tab}' available: {e}")
    return pd.DataFrame()

# --------- Formatting helpers ---------
def money(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"${x:,.2f}"

def pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.2f}%"

# --------- Main run ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--sources", default="", help="Comma-separated Sources to include in manual intraday ideas (e.g., 'Sarkee Capital,SuperiorStar')")
    ap.add_argument("--symbols", default="", help="Comma-separated tickers to include in manual intraday ideas (e.g., 'NVDA,SOXX')")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Compute and print but do not email")
    args = ap.parse_args()

    cfg, benchmark = load_config(args.config)

    # Load holdings
    holdings_raw = load_holdings(cfg).copy()
    if holdings_raw.empty:
        print("❗ No holdings found. Exiting.")
        sys.exit(0)

    # Normalize columns we need
    c_sym = pick_col(holdings_raw, "Symbol", "Ticker", "ticker", "SYMBOL")
    c_qty = pick_col(holdings_raw, "Quantity", "Qty", "qty", "Shares")
    c_last = pick_col(holdings_raw, "Last Price", "Last", "Price", "price")
    c_val  = pick_col(holdings_raw, "Current Value", "Value", "Mkt Value", "market_value")
    c_cb   = pick_col(holdings_raw, "Cost Basis Total", "Cost Basis", "cost_basis_total", "cost")
    c_acb  = pick_col(holdings_raw, "Average Cost Basis", "Avg Cost", "average_cost_basis")
    c_src  = pick_col(holdings_raw, "Source", "source")

    # If any are missing we’ll compute later
    for need in [c_sym, c_qty]:
        if not need:
            raise RuntimeError("Holdings must include Symbol and Quantity columns (names can vary).")

    # Upper-case tickers (exclude option rows that start with '-' per your example)
    holdings_raw[c_sym] = holdings_raw[c_sym].astype(str)
    # Keep option rows too; they’ll flow through but price via yfinance may be NaN
    tickers = sorted(set([t for t in holdings_raw[c_sym].str.replace("^[-]+", "", regex=True).str.upper().tolist()] + [benchmark]))

    # Bring in weekly stage map for recommendations
    stage_map = load_weekly_stage_map()

    # Price data
    intraday, daily = get_intraday_and_daily(tickers)

    # Build per-position snapshot
    rows = []
    for _, r in holdings_raw.iterrows():
        sym = str(r[c_sym]).replace("-", "") if str(r[c_sym]).startswith("-") else str(r[c_sym])
        sym = sym.upper()

        qty = float(r[c_qty]) if pd.notna(r[c_qty]) else np.nan
        src = str(r[c_src]) if c_src and pd.notna(r[c_src]) else ""

        # Live last price
        last_px = intraday_last_price(intraday, sym)
        if pd.isna(last_px) and pd.notna(r.get(c_last, np.nan)):
            last_px = float(r[c_last])

        # Values and costs
        cur_val = float(qty) * float(last_px) if (pd.notna(qty) and pd.notna(last_px)) else np.nan
        cost_total = float(r[c_cb]) if c_cb and pd.notna(r.get(c_cb, np.nan)) else np.nan
        avg_cost  = float(r[c_acb]) if c_acb and pd.notna(r.get(c_acb, np.nan)) else (float(cost_total)/float(qty) if (pd.notna(cost_total) and pd.notna(qty) and qty!=0) else np.nan)

        gain_dol = cur_val - cost_total if (pd.notna(cur_val) and pd.notna(cost_total)) else np.nan
        gain_pct = _safe_div(gain_dol, cost_total) * 100.0 if (pd.notna(gain_dol) and pd.notna(cost_total) and cost_total != 0) else np.nan

        sma150 = compute_sma(daily, sym, SMA_DAYS)
        stage = stage_map.get(sym, "Unknown")
        reco = weinstein_reco(stage, last_px, sma150, gain_pct if pd.notna(gain_pct) else np.nan)

        rows.append({
            "Symbol": sym,
            "Description": str(r.get("Description", "")),
            "Quantity": qty,
            "Last Price": last_px,
            "Current Value": cur_val,
            "Cost Basis Total": cost_total,
            "Average Cost Basis": avg_cost,
            "Total Gain/Loss Dollar": gain_dol,
            "Total Gain/Loss Percent": gain_pct,
            "Stage": stage,
            "SMA150": sma150,
            "Source": src,
            "Recommendation": reco,
        })

    snap = pd.DataFrame(rows)

    # Portfolio summary
    total_cost = snap["Cost Basis Total"].sum(skipna=True)
    total_val  = snap["Current Value"].sum(skipna=True)
    port_gain  = total_val - total_cost if (pd.notna(total_val) and pd.notna(total_cost)) else np.nan
    port_pct   = _safe_div(port_gain, total_cost) * 100.0 if (pd.notna(total_cost) and total_cost != 0) else np.nan
    avg_pct    = snap["Total Gain/Loss Percent"].dropna().mean() if not snap.empty else np.nan

    # ------ Manual intraday ideas (Signals tab or CLI) ------
    manual_sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    manual_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    sig_df = load_signals(cfg, tab_name=(cfg.get("sheets") or {}).get("signals_tab", "Signals"))
    # Try alternate manual tab if present
    if sig_df.empty:
        manual_df = load_signals(cfg, tab_name="Intraday_Manual")
        if not manual_df.empty:
            sig_df = manual_df

    ideas: List[Dict] = []
    idea_tickers: List[str] = []

    if not sig_df.empty:
        c_t = pick_col(sig_df, "Ticker", "Symbol", "ticker", "symbol")
        c_s = pick_col(sig_df, "Source", "source")
        if c_t:
            df = sig_df.copy()
            df[c_t] = df[c_t].astype(str).str.upper()
            if manual_sources and c_s:
                df = df[df[c_s].astype(str).isin(manual_sources)]
            if manual_symbols:
                df = df[df[c_t].isin(manual_symbols)]
            for _, rr in df.iterrows():
                t = str(rr[c_t]).upper()
                s = str(rr[c_s]) if c_s and pd.notna(rr.get(c_s, "")) else ""
                ideas.append({"ticker": t, "source": s})
                idea_tickers.append(t)

    # Add CLI-only symbols if not already present
    for t in manual_symbols:
        if t not in [x["ticker"] for x in ideas]:
            ideas.append({"ticker": t, "source": "CLI"})
            idea_tickers.append(t)

    if idea_tickers:
        # Ensure we have data for ideas
        missing = [t for t in idea_tickers if t not in tickers]
        if missing:
            add_intraday, add_daily = get_intraday_and_daily(missing + [benchmark])
            # merge in
            if isinstance(intraday.columns, pd.MultiIndex) and isinstance(add_intraday.columns, pd.MultiIndex):
                intraday = pd.concat([intraday, add_intraday], axis=1)
            if isinstance(daily.columns, pd.MultiIndex) and isinstance(add_daily.columns, pd.MultiIndex):
                daily = pd.concat([daily, add_daily], axis=1)

    # Evaluate intraday BUY/HOLD/SELL for ideas
    idea_rows = []
    charts_inline = []
    for it in ideas:
        t = it["ticker"]
        price = intraday_last_price(intraday, t)
        sma150 = compute_sma(daily, t, SMA_DAYS)
        stage = stage_map.get(t, "Unknown")
        pivot = last_weekly_pivot_high(t, daily, PIVOT_LOOKBACK_WEEKS)
        # Relative strength 'OK' heuristic: use weekly map if we had rs_above_ma;
        # as an approximation, require price >= SMA150 for RS OK
        rs_ok = (pd.notna(sma150) and pd.notna(price) and price >= sma150)
        is_buy, is_near, pace = buy_signal_intraday(
            ticker=t, stage=stage, rs_ok=rs_ok,
            price=price, sma150=sma150, pivot=pivot,
            intraday_df=intraday, daily_df=daily
        )
        reco = "BUY" if is_buy else ("HOLD" if (pd.notna(price) and pd.notna(sma150) and price >= UNDER_MA_CUTOFF*sma150) else "SELL")
        idea_rows.append({
            "Ticker": t,
            "Source": it["source"],
            "Price": price,
            "SMA150": sma150,
            "Pivot(≈10w)": pivot,
            "Vol Pace vs 50d": None if pd.isna(pace) else round(float(pace), 2),
            "Stage": stage,
            "Signal": "BUY" if is_buy else ("NEAR" if is_near else "—"),
            "Recommendation": reco
        })
        if len(charts_inline) < MAX_CHARTS:
            ch = make_inline_chart(t, benchmark, daily)
            if ch:
                charts_inline.append((t, ch))

    ideas_df = pd.DataFrame(idea_rows) if idea_rows else pd.DataFrame()

    # --------- Render HTML ---------
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_html = f"""
    <h2>Weinstein Intraday — {now}</h2>
    <h3>Weinstein Weekly - Summary</h3>
    <table border="1" cellspacing="0" cellpadding="6">
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Total Gain/Loss ($)</td><td>{money(port_gain)}</td></tr>
      <tr><td>Portfolio % Gain</td><td>{pct(port_pct)}</td></tr>
      <tr><td>Average % Gain</td><td>{pct(avg_pct)}</td></tr>
    </table>
    """

    # Per-position snapshot table (match your column names/order)
    ordered_cols = [
        "Symbol","Description","Quantity","Last Price","Current Value",
        "Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar",
        "Total Gain/Loss Percent","Recommendation"
    ]
    display_cols = [c for c in ordered_cols if c in snap.columns]
    styled = snap.copy()
    # Format currency and percent
    for c in ["Last Price","Current Value","Cost Basis Total","Average Cost Basis","Total Gain/Loss Dollar"]:
        if c in styled.columns:
            styled[c] = styled[c].apply(money)
    if "Total Gain/Loss Percent" in styled.columns:
        styled["Total Gain/Loss Percent"] = styled["Total Gain/Loss Percent"].apply(pct)

    perpos_html = "<h3>Per-position Snapshot</h3>" + styled[display_cols].to_html(index=False, escape=False)

    # Manual intraday ideas
    ideas_html = ""
    if not ideas_df.empty:
        # Format
        fmt = ideas_df.copy()
        if "Price" in fmt.columns: fmt["Price"] = fmt["Price"].apply(money)
        if "SMA150" in fmt.columns: fmt["SMA150"] = fmt["SMA150"].apply(lambda x: money(x) if pd.notna(x) else "—")
        if "Vol Pace vs 50d" in fmt.columns: fmt["Vol Pace vs 50d"] = fmt["Vol Pace vs 50d"].apply(lambda x: f"{x:.2f}×" if pd.notna(x) else "—")
        ideas_html = "<h3>Intraday Ideas (Manual Sources / CLI)</h3>" + fmt.to_html(index=False, escape=False)

    charts_html = ""
    if charts_inline:
        charts_html = "<h4>Charts</h4>"
        for t, data_uri in charts_inline:
            charts_html += f"""
            <div style="display:inline-block;margin:6px 8px 10px 0;vertical-align:top;text-align:center;">
              <img src="{data_uri}" alt="{t}" style="border:1px solid #eee;border-radius:6px;max-width:320px;">
              <div style="font-size:12px;color:#555;margin-top:3px;">{t}</div>
            </div>
            """

    html = summary_html + perpos_html + ideas_html + charts_html

    # Plain text fallback
    text = f"Weinstein Intraday — {now}\n\n"
    text += f"Total Gain/Loss: {money(port_gain)}\n"
    text += f"Portfolio % Gain: {pct(port_pct)}\n"
    text += f"Average % Gain: {pct(avg_pct)}\n\n"
    if not snap.empty:
        text += "Positions:\n"
        for _, r in snap.iterrows():
            text += f"- {r['Symbol']}: {r.get('Recommendation','')}, P&L {pct(r.get('Total Gain/Loss Percent', np.nan))}\n"
    if not ideas_df.empty:
        text += "\nIntraday Ideas:\n"
        for _, r in ideas_df.iterrows():
            text += f"- {r['Ticker']}: {r.get('Signal','—')} → {r.get('Recommendation','')}\n"

    # Save HTML copy
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_html = os.path.join(OUTPUT_DIR, "intraday_email.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"✅ Intraday HTML written: {out_html}")

    if args.dry_run:
        print("ℹ️  dry-run: email suppressed.")
        return

    if not args.no_email and ((cfg.get("notifications") or {}).get("email") or {}).get("enabled", True):
        subj = "Weinstein Intraday — Portfolio & Ideas"
        send_email(
            subject=subj,
            html_body=html,
            text_body=text,
            cfg_path=args.config
        )
        print("📧 Email sent.")
    else:
        print("ℹ️  Email disabled by flags or config.")

if __name__ == "__main__":
    main()
