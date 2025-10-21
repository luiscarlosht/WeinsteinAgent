# === weinstein_report_weekly.py ===
import os
import sys
import math
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from weinstein_mailer import send_email
from universe_loaders import combine_universe

# --------- Tunables (sensible defaults, override in config.yaml) ----------
DEFAULT_BENCHMARK = "SPY"
WEEKS_LOOKBACK = 180   # ~3.5 years of weekly bars
MA_WEEKS = 30
SLOPE_WINDOW = 5       # weeks used to measure MA slope
NEAR_MA_BAND = 0.05    # within ±5% of MA == "near/flat"
RS_MA_WEEKS = 30
OUTPUT_DIR_FALLBACK = "./output"

# --------- Utilities ---------
def _extract_field(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """
    From a yfinance download DataFrame, extract a single field (e.g., 'Close'/'Volume')
    and return a 2D DataFrame with columns = tickers found.
    Handles both MultiIndex (many tickers) and single-index (one ticker) cases.
    """
    if isinstance(df.columns, pd.MultiIndex):
        out = df[field].copy()
        # Ensure we keep only the columns we asked for (and in that order if present)
        keep = [t for t in tickers if t in out.columns]
        return out[keep]
    else:
        # Single ticker; df has columns like ['Open','High','Low','Close','Adj Close','Volume']
        if field not in df.columns:
            raise KeyError(f"Field '{field}' not found in downloaded data.")
        # Best guess at the ticker name (first in list)
        t0 = tickers[0] if tickers else "TICKER"
        out = df[[field]].copy()
        out.columns = [t0]
        return out

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    app = cfg.get("app", {}) or {}
    uni = cfg.get("universe", {}) or {}
    reporting = cfg.get("reporting", {}) or {}

    # Universe mode: sp500 | custom
    mode = (uni.get("mode") or "custom").lower()
    use_sp500 = (mode == "sp500")
    extra = uni.get("extra") or []                 # optional extras when mode=sp500
    explicit_tickers = uni.get("tickers") or []    # for mode=custom (or legacy list)

    # Build final ticker list
    if use_sp500:
        tickers = combine_universe(sp500=True, extra_symbols=extra)
    else:
        tickers = combine_universe(sp500=False, extra_symbols=explicit_tickers)

    benchmark = app.get("benchmark", DEFAULT_BENCHMARK)
    tz = app.get("timezone", "America/Chicago")

    output_dir = reporting.get("output_dir", OUTPUT_DIR_FALLBACK)
    include_pdf = reporting.get("include_pdf", False)  # (HTML+CSV always saved)

    # Optional hygiene filters
    min_price = int(uni.get("min_price", 0))
    min_avg_volume = int(uni.get("min_avg_volume", 0))

    return cfg, tickers, benchmark, tz, output_dir, include_pdf, min_price, min_avg_volume

def fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK):
    """
    Download weekly OHLCV for all tickers + benchmark. Returns (close_w, volume_w).
    """
    uniq = list(dict.fromkeys((tickers or []) + [benchmark]))  # de-dup, preserve order
    if not uniq:
        raise ValueError("No symbols to download.")

    # interval='1wk' + period='10y' gives ample history; use auto_adjust so Close is split/div adjusted
    data = yf.download(
        uniq,
        interval="1wk",
        period="10y",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )

    # Extract Close / Volume as 2D frames
    close = _extract_field(data, "Close", uniq)
    volume = _extract_field(data, "Volume", uniq)

    # keep last N weeks to speed up ops but ensure >= MA windows
    tail_n = max(weeks, MA_WEEKS + RS_MA_WEEKS + SLOPE_WINDOW + 10)
    close = close.tail(tail_n)
    volume = volume.tail(tail_n)

    return close, volume

def compute_stage_for_ticker(closes: pd.Series, bench: pd.Series):
    """
    Compute Weinstein metrics for one ticker vs. benchmark series (weekly closes).
    Returns dict with indicators + stage label.
    """
    s = closes.dropna().copy()
    b = bench.reindex_like(s).dropna()

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

    # Relative Strength line vs. benchmark (price ratio)
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

    # Stage rules (heuristic in Weinstein spirit)
    if price_above_ma and ma_up and rs_up:
        stage = "Stage 2 (Uptrend)"
    elif (not price_above_ma) and ma_down and rs_down:
        stage = "Stage 4 (Downtrend)"
    elif near_ma and abs(ma_slope_last) < (abs(ma_last) * 0.0005):  # flat-ish MA
        stage = "Stage 1 (Basing)"
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
        "notes": "; ".join(notes),
    }

def build_report_df(close_w: pd.DataFrame,
                    volume_w: pd.DataFrame,
                    tickers: list[str],
                    benchmark: str,
                    min_price: int = 0,
                    min_avg_volume: int = 0):
    """
    Build final report DataFrame. Applies optional hygiene filters using last close and 10w avg volume.
    """
    if benchmark not in close_w.columns:
        raise KeyError(f"Benchmark '{benchmark}' not found in downloaded data.")
    bench = close_w[benchmark].dropna()

    # Hygiene metrics (10-week avg volume, last close)
    last_close = close_w.ffill().iloc[-1]
    avg_vol_10w = volume_w.rolling(10).mean().ffill().iloc[-1]

    # Start rows
    rows = []
    for t in tickers:
        if t not in close_w.columns:
            rows.append({"ticker": t, "stage": "N/A", "notes": "no_data"})
            continue

        # Enforce hygiene filters (if provided)
        lc = float(last_close.get(t, np.nan)) if pd.notna(last_close.get(t, np.nan)) else np.nan
        av = float(avg_vol_10w.get(t, np.nan)) if pd.notna(avg_vol_10w.get(t, np.nan)) else np.nan
        if (min_price and (pd.isna(lc) or lc < min_price)) or (min_avg_volume and (pd.isna(av) or av < min_avg_volume)):
            rows.append({"ticker": t, "stage": "Filtered", "price": lc, "notes": "below min_price/volume"})
            continue

        res = compute_stage_for_ticker(close_w[t], bench)
        res["ticker"] = t
        rows.append(res)

    df = pd.DataFrame(rows)

    # Clean ordering
    cols = [
        "ticker", "stage", "price", "ma30", "dist_ma_pct", "ma_slope_per_wk",
        "rs", "rs_ma30", "rs_above_ma", "rs_slope_per_wk", "notes"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    # Sort: Stage 2 first, then by distance above MA
    stage_rank = {
        "Stage 2 (Uptrend)": 0,
        "Stage 1 (Basing)": 1,
        "Stage 3 (Topping)": 2,
        "Stage 4 (Downtrend)": 3,
        "Filtered": 8,
        "N/A": 9,
    }
    df["stage_rank"] = df["stage"].map(stage_rank).fillna(9)
    df = df.sort_values(by=["stage_rank", "dist_ma_pct"], ascending=[True, False]).reset_index(drop=True)
    return df.drop(columns=["stage_rank"])

def df_to_html(df: pd.DataFrame, title: str):
    styled = df.copy()

    # Pretty percentages
    for c in ["dist_ma_pct", "ma_slope_per_wk", "rs_slope_per_wk"]:
        if c in styled.columns:
            styled[c] = styled[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")

    if "rs_above_ma" in styled.columns:
        styled["rs_above_ma"] = styled["rs_above_ma"].map({True: "Yes", False: "No"})

    # Basic HTML
    table_html = styled.to_html(index=False, border=0, justify="center", escape=False)
    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height:1.45; padding:20px; }
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
    cfg, tickers, benchmark, tz, output_dir, include_pdf, min_price, min_avg_volume = load_config()

    if not tickers:
        print("No tickers resolved from config.yaml (check universe.mode/custom/extra).")
        sys.exit(1)

    # Ensure output dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Universe size: {len(tickers)} tickers (benchmark: {benchmark})")
    print("Downloading weekly data…")
    close_w, volume_w = fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK)

    print("Computing Weinstein stages…")
    report_df = build_report_df(
        close_w=close_w,
        volume_w=volume_w,
        tickers=tickers,
        benchmark=benchmark,
        min_price=min_price,
        min_avg_volume=min_avg_volume,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.csv")
    html_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.html")

    report_df.to_csv(csv_path, index=False)
    html = df_to_html(report_df, title=f"Weinstein Weekly — Benchmark: {benchmark}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Email it
    subject = f"Weinstein Weekly Report — {datetime.now().strftime('%b %d, %Y')}"
    top_lines = report_df.head(12).to_string(index=False)
    body_text = (
        f"Weinstein Weekly Report generated.\n\nFiles:\n"
        f"- {csv_path}\n- {html_path}\n\nTop lines:\n{top_lines}\n"
    )
    send_email(subject=subject, html_body=html, text_body=body_text, cfg_path="config.yaml")

    print(f"Saved:\n - {csv_path}\n - {html_path}")
    print("Done.")

if __name__ == "__main__":
    main()
