# === weinstein_report_weekly.py ===
import os
import io
import sys
import math
import yaml
import time
import base64
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from weinstein_mailer import send_email

# --------- Tunables (sensible defaults, override in config.yaml) ----------
DEFAULT_BENCHMARK = "SPY"
WEEKS_LOOKBACK = 180  # ~3.5 years of weekly bars
MA_WEEKS = 30
SLOPE_WINDOW = 5  # weeks used to measure MA slope
NEAR_MA_BAND = 0.05  # within ±5% of MA == "near/flat"
RS_MA_WEEKS = 30
OUTPUT_DIR_FALLBACK = "./output"

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {})
    uni = cfg.get("universe", {}) if isinstance(cfg.get("universe"), dict) else {}
    # Support universe as dict or as list
    if isinstance(cfg.get("universe"), dict):
        tickers = uni.get("tickers", [])
    else:
        tickers = cfg.get("universe", [])
    benchmark = app.get("benchmark", DEFAULT_BENCHMARK)
    tz = app.get("timezone", "America/Chicago")
    output_dir = cfg.get("reporting", {}).get("output_dir", OUTPUT_DIR_FALLBACK)
    include_pdf = cfg.get("reporting", {}).get("include_pdf", False)  # HTML+CSV are always saved
    return cfg, tickers, benchmark, tz, output_dir, include_pdf

def fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK):
    # yfinance accepts interval="1wk", period="5y" etc.
    # We’ll use period="10y" then trim last N bars to ensure enough history for MAs
    uniq = list(dict.fromkeys(tickers + [benchmark]))  # de-dup, preserve order
    df = yf.download(uniq, interval="1wk", period="10y", auto_adjust=True, ignore_tz=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
    else:
        close = df[["Close"]].copy()
        close.columns = [uniq[0]]
    # keep last N weeks to speed up ops but ensure >= MA windows
    close = close.tail(max(weeks, MA_WEEKS + RS_MA_WEEKS + SLOPE_WINDOW + 10))
    return close

def compute_stage_for_ticker(closes: pd.Series, bench: pd.Series):
    """
    Returns dict with indicators + stage label.
    """
    s = closes.dropna().copy()
    b = bench.reindex_like(s).dropna()
    # Align again
    idx = s.index.intersection(b.index)
    s = s.loc[idx]
    b = b.loc[idx]

    if len(s) < MA_WEEKS + SLOPE_WINDOW + 5 or len(b) < RS_MA_WEEKS + 5:
        return {"error": "insufficient_data"}

    ma = s.rolling(MA_WEEKS).mean()
    # MA slope over last SLOPE_WINDOW weeks
    ma_slope = ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    ma_slope_last = ma_slope.iloc[-1]
    ma_last = ma.iloc[-1]
    price_last = s.iloc[-1]
    dist_ma_pct = (price_last - ma_last) / ma_last if ma_last and not math.isclose(ma_last, 0.0) else np.nan

    # Relative Strength line vs. benchmark
    rs = s / b
    rs_ma = rs.rolling(RS_MA_WEEKS).mean()
    rs_slope = rs_ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    rs_last = rs.iloc[-1]
    rs_ma_last = rs_ma.iloc[-1]
    rs_above = bool(rs_last > rs_ma_last)
    rs_slope_last = rs_slope.iloc[-1]

    # Flags
    price_above_ma = bool(price_last > ma_last)
    ma_up = bool(ma_slope_last > 0)
    ma_down = bool(ma_slope_last < 0)
    near_ma = bool(abs(dist_ma_pct) <= NEAR_MA_BAND)
    rs_up = bool(rs_above and rs_slope_last > 0)
    rs_down = bool((not rs_above) and rs_slope_last < 0)

    # Stage rules (heuristic, classic Weinstein spirit)
    # Stage 2: strong uptrend
    if price_above_ma and ma_up and rs_up:
        stage = "Stage 2 (Uptrend)"
    # Stage 4: strong downtrend
    elif (not price_above_ma) and ma_down and rs_down:
        stage = "Stage 4 (Downtrend)"
    # Stage 1: basing
    elif near_ma and abs(ma_slope_last) < (ma_last * 0.0005):  # flat-ish MA
        stage = "Stage 1 (Basing)"
    # Stage 3: topping/rolling over
    else:
        stage = "Stage 3 (Topping)"

    notes = []
    if price_above_ma and not ma_up:
        notes.append("Price>MA but MA not rising")
    if (not price_above_ma) and ma_up:
        notes.append("Price<MA but MA rising (watch)")
    if rs_above and rs_slope_last <= 0:
        notes.append("RS above MA but flattening")
    if (not rs_above) and rs_slope_last >= 0:
        notes.append("RS below MA but improving")

    return {
        "price": float(price_last),
        "ma30": float(ma_last),
        "dist_ma_pct": float(dist_ma_pct) if pd.notna(dist_ma_pct) else np.nan,
        "ma_slope_per_wk": float(ma_slope_last) if pd.notna(ma_slope_last) else np.nan,
        "rs": float(rs_last),
        "rs_ma30": float(rs_ma_last) if pd.notna(rs_ma_last) else np.nan,
        "rs_above_ma": bool(rs_above),
        "rs_slope_per_wk": float(rs_slope_last) if pd.notna(rs_slope_last) else np.nan,
        "stage": stage,
        "notes": "; ".join(notes)
    }

def build_report_df(close_w: pd.DataFrame, tickers, benchmark):
    bench = close_w[benchmark].dropna()
    rows = []
    for t in tickers:
        if t not in close_w.columns:
            rows.append({"ticker": t, "stage": "N/A", "notes": "no_data"})
            continue
        res = compute_stage_for_ticker(close_w[t], bench)
        res["ticker"] = t
        rows.append(res)
    df = pd.DataFrame(rows)
    # Clean ordering
    cols = ["ticker","stage","price","ma30","dist_ma_pct","ma_slope_per_wk",
            "rs","rs_ma30","rs_above_ma","rs_slope_per_wk","notes"]
    df = df.reindex(columns=cols)
    # Sort: Stage 2 first, then by distance above MA
    stage_rank = {"Stage 2 (Uptrend)":0,"Stage 1 (Basing)":1,"Stage 3 (Topping)":2,"Stage 4 (Downtrend)":3,"N/A":9}
    df["stage_rank"] = df["stage"].map(stage_rank).fillna(9)
    df = df.sort_values(by=["stage_rank","dist_ma_pct"], ascending=[True,False]).reset_index(drop=True)
    return df.drop(columns=["stage_rank"])

def df_to_html(df: pd.DataFrame, title: str):
    styled = df.copy()
    # Pretty percentages
    for c in ["dist_ma_pct","ma_slope_per_wk","rs_slope_per_wk"]:
        if c in styled.columns:
            styled[c] = styled[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
    styled["rs_above_ma"] = styled["rs_above_ma"].map({True:"Yes", False:"No"})
    # Basic HTML
    table_html = styled.to_html(index=False, border=0, justify="center", escape=False)
    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height:1.4; padding:20px; }
      h2 { margin: 0 0 8px 0; }
      .sub { color:#666; margin-bottom:16px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 14px; }
      th { text-align:left; background:#fafafa; }
      .stage { font-weight:600; }
    </style>
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""{css}
    <h2>{title}</h2>
    <div class="sub">Generated {now}</div>
    {table_html}
    """
    return html

def main():
    cfg, tickers, benchmark, tz, output_dir, include_pdf = load_config()
    if not tickers:
        print("No tickers configured in config.yaml under 'universe.tickers'.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading weekly data…")
    close_w = fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK)

    print("Computing Weinstein stages…")
    report_df = build_report_df(close_w, tickers, benchmark)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.csv")
    html_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.html")

    report_df.to_csv(csv_path, index=False)
    html = df_to_html(report_df, title=f"Weinstein Weekly — Benchmark: {benchmark}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Email it
    subject = f"Weinstein Weekly Report — {datetime.now().strftime('%b %d, %Y')}"
    body_text = f"Weinstein Weekly Report generated. Files:\n- {csv_path}\n- {html_path}\n\nTop lines:\n{report_df.head(10).to_string(index=False)}"
    send_email(subject=subject, html_body=html, text_body=body_text, cfg_path="config.yaml")

    print(f"Saved:\n - {csv_path}\n - {html_path}")
    print("Done.")

if __name__ == "__main__":
    main()
