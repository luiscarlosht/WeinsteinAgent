#!/usr/bin/env python3
# === weinstein_report_weekly.py (Yahoo Finance source, robust batching + flexible column extraction) ===

import os
import sys
import math
import io
import base64
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from weinstein_mailer import send_email
from universe_loaders import combine_universe

# Optional Sheets logging (does NOT affect report content)
try:
    from gsheet_helpers import log_signal
except Exception:
    log_signal = None

# Tiny charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------- Tunables ----------
DEFAULT_BENCHMARK = "SPY"
WEEKS_LOOKBACK = 180
MA_WEEKS = 30
MA10_WEEKS = 10
SLOPE_WINDOW = 5
NEAR_MA_BAND = 0.05
RS_MA_WEEKS = 30
OUTPUT_DIR_FALLBACK = "./output"
TOP_N_CHARTS = 20
BATCH_SIZE = 90  # yfinance is happier with <100 tickers per call

# --------- Utilities ----------
def _extract_field_flexible(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """
    Robust extraction for yfinance outputs:
    - MultiIndex (('AAPL','Close') layout) -> level=1 has 'Close'
    - MultiIndex (('Close','AAPL') layout) -> level=0 has 'Close'
    - Single-index (single ticker) -> columns include 'Close'
    Falls back to 'Adj Close' if 'Close' is missing.
    """
    if df is None or df.empty:
        raise KeyError("Downloaded data is empty.")

    def _reindex_cols(out: pd.DataFrame) -> pd.DataFrame:
        keep = [t for t in tickers if t in out.columns]
        return out[keep] if keep else out

    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0).unique()
        lv1 = df.columns.get_level_values(1).unique()
        if field in lv0:
            out = df.loc[:, field].copy()
            return _reindex_cols(out)
        if field in lv1:
            out = df.xs(field, axis=1, level=1).copy()
            return _reindex_cols(out)
        if field == "Close":
            if "Adj Close" in lv0:
                out = df.loc[:, "Adj Close"].copy()
                return _reindex_cols(out)
            if "Adj Close" in lv1:
                out = df.xs("Adj Close", axis=1, level=1).copy()
                return _reindex_cols(out)
        raise KeyError(f"Field '{field}' not in downloaded data (MultiIndex).")
    else:
        if field in df.columns:
            t0 = tickers[0] if tickers else "TICKER"
            out = df[[field]].copy()
            out.columns = [t0]
            return out
        if field == "Close" and "Adj Close" in df.columns:
            t0 = tickers[0] if tickers else "TICKER"
            out = df[["Adj Close"]].copy()
            out.columns = [t0]
            return out
        raise KeyError(f"Field '{field}' not in downloaded data (single-index).")

def _extract_field(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    return _extract_field_flexible(df, field, tickers)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    app = cfg.get("app", {}) or {}
    uni = cfg.get("universe", {}) or {}
    reporting = cfg.get("reporting", {}) or {}

    mode = (uni.get("mode") or "custom").lower()
    use_sp500 = (mode == "sp500")
    extra = uni.get("extra") or []
    explicit_tickers = uni.get("tickers") or []

    if use_sp500:
        tickers = combine_universe(sp500=True, extra_symbols=extra)
    else:
        tickers = combine_universe(sp500=False, extra_symbols=explicit_tickers)

    benchmark = app.get("benchmark", DEFAULT_BENCHMARK)
    tz = app.get("timezone", "America/Chicago")

    output_dir = reporting.get("output_dir", OUTPUT_DIR_FALLBACK)
    include_pdf = reporting.get("include_pdf", False)

    min_price = int(uni.get("min_price", 0))
    min_avg_volume = int(uni.get("min_avg_volume", 0))

    return cfg, tickers, benchmark, tz, output_dir, include_pdf, min_price, min_avg_volume

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK):
    """
    Download in batches to avoid yfinance empty/partial returns.
    Returns wide DataFrames: close, volume (columns=tickers incl. benchmark).
    """
    all_syms = list(dict.fromkeys((tickers or []) + [benchmark]))
    if not all_syms:
        raise ValueError("No symbols to download.")

    close_frames = []
    volume_frames = []

    for batch in _chunks(all_syms, BATCH_SIZE):
        if benchmark not in batch:
            batch = batch + [benchmark]
        data = yf.download(
            batch,
            interval="1wk",
            period="10y",
            auto_adjust=True,
            ignore_tz=True,
            progress=False,
        )
        if data is None or data.empty:
            continue

        try:
            close_b = _extract_field(data, "Close", batch)
        except KeyError:
            close_b = _extract_field(data, "Adj Close", batch)

        volume_b = _extract_field(data, "Volume", batch)

        close_frames.append(close_b)
        volume_frames.append(volume_b)

    if not close_frames or not volume_frames:
        raise KeyError("No data fetched from Yahoo Finance (all batches empty).")

    close = pd.concat(close_frames, axis=1)
    volume = pd.concat(volume_frames, axis=1)

    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    ordered = [t for t in all_syms if t in close.columns]
    close = close[ordered]
    volume = volume[ordered]

    tail_n = max(weeks, MA_WEEKS + RS_MA_WEEKS + SLOPE_WINDOW + 10)
    close = close.tail(tail_n)
    volume = volume.tail(tail_n)

    if close.empty or volume.empty:
        raise KeyError("Downloaded data became empty after trimming.")

    return close, volume

def _weekly_short_term_state(series_price: pd.Series) -> tuple[str, float, float]:
    s = series_price.dropna()
    if len(s) < max(MA10_WEEKS, MA_WEEKS) + 5:
        return ("Unknown", np.nan, np.nan)
    ma10 = s.rolling(MA10_WEEKS).mean()
    ma30 = s.rolling(MA_WEEKS).mean()
    c = float(s.iloc[-1])
    m10 = float(ma10.iloc[-1])
    m30 = float(ma30.iloc[-1])
    if (c > m10) and (m10 > m30):
        state = "ShortTermUptrend"
    elif (c > m30) and not (m10 > m30):
        state = "StageConflict"
    elif (m10 > m30) and not (c > m10):
        state = "StageConflict"
    else:
        state = "Weak"
    return (state, m10, m30)

def compute_stage_for_ticker(closes: pd.Series, bench: pd.Series):
    s = closes.dropna().copy()
    b = bench.reindex_like(s).dropna()

    idx = s.index.intersection(b.index)
    s = s.loc[idx]
    b = b.loc[idx]

    if len(s) < MA_WEEKS + SLOPE_WINDOW + 5 or len(b) < RS_MA_WEEKS + 5:
        return {"error": "insufficient_data"}

    ma = s.rolling(MA_WEEKS).mean()

    ma_slope = ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    ma_slope_last = ma_slope.iloc[-1]
    ma_last = ma.iloc[-1]
    price_last = s.iloc[-1]
    dist_ma_pct = (price_last - ma_last) / ma_last if ma_last and not math.isclose(ma_last, 0.0) else np.nan

    rs = s / b
    rs_ma = rs.rolling(RS_MA_WEEKS).mean()
    rs_slope = rs_ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    rs_last = rs.iloc[-1]
    rs_ma_last = rs_ma.iloc[-1]
    rs_above = bool(rs_last > rs_ma_last)
    rs_slope_last = rs_slope.iloc[-1]

    price_above_ma = bool(price_last > ma_last)
    ma_up = bool(ma_slope_last > 0)
    near_ma = bool(abs(dist_ma_pct) <= NEAR_MA_BAND)
    rs_up = bool(rs_above and rs_slope_last > 0)
    rs_down = bool((not rs_above) and rs_slope_last < 0)

    if price_above_ma and ma_up and rs_up:
        stage = "Stage 2 (Uptrend)"
    elif (not price_above_ma) and (ma_slope_last < 0) and rs_down:
        stage = "Stage 4 (Downtrend)"
    elif near_ma and abs(ma_slope_last) < (abs(ma_last) * 0.0005):
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

    st_state, ma10_last, _ = _weekly_short_term_state(s)

    return {
        "price": float(price_last),
        "ma10": float(ma10_last) if pd.notna(ma10_last) else np.nan,
        "ma30": float(ma_last),
        "dist_ma_pct": float(dist_ma_pct) if pd.notna(dist_ma_pct) else np.nan,
        "ma_slope_per_wk": float(ma_slope_last) if pd.notna(ma_slope_last) else np.nan,
        "rs": float(rs_last),
        "rs_ma30": float(rs_ma_last) if pd.notna(rs_ma_last) else np.nan,
        "rs_above_ma": bool(rs_above),
        "rs_slope_per_wk": float(rs_slope_last) if pd.notna(rs_slope_last) else np.nan,
        "stage": stage,
        "short_term_state_wk": st_state,
        "notes": "; ".join(notes),
    }

def classify_buy_signal(stage: str) -> tuple[str, str]:
    stage = stage or ""
    if stage.startswith("Stage 2"):
        return ("BUY", '<span class="badge buy">Buy</span>')
    if stage.startswith("Stage 1"):
        return ("WATCH", '<span class="badge watch">Watch</span>')
    if stage == "Filtered":
        return ("AVOID", '<span class="badge avoid">Avoid</span>')
    return ("AVOID", '<span class="badge avoid">Avoid</span>')

def build_report_df(close_w: pd.DataFrame,
                    volume_w: pd.DataFrame,
                    tickers: list[str],
                    benchmark: str,
                    min_price: int = 0,
                    min_avg_volume: int = 0):
    if benchmark not in close_w.columns:
        raise KeyError(f"Benchmark '{benchmark}' not found in downloaded data.")
    bench = close_w[benchmark].dropna()

    last_close = close_w.ffill().iloc[-1]
    avg_vol_10w = volume_w.rolling(10).mean().ffill().iloc[-1]

    rows = []
    for t in tickers:
        if t not in close_w.columns:
            rows.append({"ticker": t, "stage": "N/A", "notes": "no_data"})
            continue

        lc = float(last_close.get(t, np.nan)) if pd.notna(last_close.get(t, np.nan)) else np.nan
        av = float(avg_vol_10w.get(t, np.nan)) if pd.notna(avg_vol_10w.get(t, np.nan)) else np.nan
        if (min_price and (pd.isna(lc) or lc < min_price)) or (min_avg_volume and (pd.isna(av) or av < min_avg_volume)):
            s = close_w[t].dropna()
            st_state, ma10_last, _ = _weekly_short_term_state(s)
            rows.append({
                "ticker": t, "stage": "Filtered", "price": lc,
                "ma10": float(ma10_last) if pd.notna(ma10_last) else np.nan,
                "ma30": np.nan, "short_term_state_wk": st_state,
                "notes": "below min_price/volume"
            })
            continue

        res = compute_stage_for_ticker(close_w[t], bench)
        res["ticker"] = t
        rows.append(res)

    df = pd.DataFrame(rows)

    cols = [
        "ticker", "stage", "price", "ma10", "ma30", "dist_ma_pct", "ma_slope_per_wk",
        "rs", "rs_ma30", "rs_above_ma", "rs_slope_per_wk", "short_term_state_wk", "notes"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    df["buy_signal"] = df["stage"].apply(lambda s: classify_buy_signal(str(s))[0])
    df["buy_signal_html"] = df["stage"].apply(lambda s: classify_buy_signal(str(s))[1])

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

# ---------- Tiny inline charts ----------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def make_tiny_chart_html(series_price: pd.Series, benchmark: pd.Series) -> str:
    s = series_price.dropna()
    b = benchmark.reindex_like(s).dropna()
    idx = s.index.intersection(b.index)
    if len(idx) < MA_WEEKS + 5:
        return ""
    s = s.loc[idx]
    b = b.loc[idx]

    ma30 = s.rolling(MA_WEEKS).mean()
    ma10 = s.rolling(MA10_WEEKS).mean()
    rs = (s / b).rolling(RS_MA_WEEKS).mean()

    fig, ax1 = plt.subplots(figsize=(3.0, 1.4))
    ax1.plot(s.index, s.values, linewidth=1.2)
    ax1.plot(ma10.index, ma10.values, linewidth=1.0)
    ax1.plot(ma30.index, ma30.values, linewidth=1.0)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    ax2 = ax1.twinx()
    ax2.plot(rs.index, rs.values, linewidth=0.8)
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in (*ax1.spines.values(), *ax2.spines.values()):
        spine.set_visible(False)
    img_src = _fig_to_base64(fig)
    return f'<img src="{img_src}" alt="chart" style="display:block;width:100%;max-width:240px;height:auto;border:0" />'

def attach_tiny_charts(df: pd.DataFrame, close_w: pd.DataFrame, benchmark: str, top_n: int = TOP_N_CHARTS) -> pd.DataFrame:
    out = df.copy()
    out["chart"] = ""
    bench_series = close_w[benchmark].dropna()
    for i, row in out.head(top_n).iterrows():
        t = row["ticker"]
        if t in close_w.columns:
            try:
                out.at[i, "chart"] = make_tiny_chart_html(close_w[t], bench_series)
            except Exception:
                out.at[i, "chart"] = ""
    return out

# ---------- HTML ----------
def df_to_html(df: pd.DataFrame, title: str, summary_line: str):
    styled = df.copy()

    for c in ["dist_ma_pct", "ma_slope_per_wk", "rs_slope_per_wk"]:
        if c in styled.columns:
            styled[c] = styled[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")

    if "rs_above_ma" in styled.columns:
        styled["rs_above_ma"] = styled["rs_above_ma"].map({True: "Yes", False: "No"})

    if "buy_signal_html" in styled.columns:
        styled["Buy Signal"] = styled["buy_signal_html"]
    else:
        styled["Buy Signal"] = styled.get("buy_signal", "")

    columns_order = [
        "ticker", "Buy Signal", "chart",
        "stage", "short_term_state_wk",
        "price", "ma10", "ma30", "dist_ma_pct",
        "ma_slope_per_wk", "rs", "rs_ma30", "rs_above_ma", "rs_slope_per_wk",
        "notes"
    ]
    for c in columns_order:
        if c not in styled.columns:
            styled[c] = ""
    styled = styled[columns_order]

    table_html = styled.to_html(index=False, border=0, justify="center", escape=False)
    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height:1.45; padding:20px; color:#111; }
      h2 { margin: 0 0 4px 0; }
      .sub { color:#666; margin-bottom:16px; }
      .summary { background:#f6f8fa; border:1px solid #eaecef; padding:10px 12px; border-radius:8px; margin:10px 0 16px 0; }
      table { border-collapse: collapse; width: 100%; }
      th, td { padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 14px; vertical-align: top; }
      th { text-align:left; background:#fafafa; }
      .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid transparent; }
      .badge.buy { background:#eaffea; color:#106b21; border-color:#b8e7b9; }
      .badge.watch { background:#fff6d6; color:#6b5310; border-color:#f0e2a6; }
      .badge.avoid { background:#ffe8e6; color:#8a1111; border-color:#f3b3ae; }
      img { image-rendering:-webkit-optimize-contrast; }
    </style>
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""{css}
    <h2>{title}</h2>
    <div class="sub">Generated {now}</div>
    <div class="summary">{summary_line}</div>
    {table_html}
    """
    return html

# ---------- Main ----------
def main():
    cfg, tickers, benchmark, tz, output_dir, include_pdf, min_price, min_avg_volume = load_config()

    if not tickers:
        print("No tickers resolved from config.yaml (check universe.mode/custom/extra).")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Universe size: {len(tickers)} tickers (benchmark: {benchmark})")
    print("Downloading weekly data (Yahoo Finance)…")
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

    report_with_charts = attach_tiny_charts(report_df, close_w, benchmark, top_n=TOP_N_CHARTS)

    buy_count = int((report_df["buy_signal"] == "BUY").sum())
    watch_count = int((report_df["buy_signal"] == "WATCH").sum())
    avoid_count = int((report_df["buy_signal"] == "AVOID").sum())
    total = int(len(report_df))
    summary_line = f"<strong>Summary:</strong> ✅ Buy: {buy_count} &nbsp; | &nbsp; 🟡 Watch: {watch_count} &nbsp; | &nbsp; 🔴 Avoid: {avoid_count} &nbsp; (Total: {total})"

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.csv")
    html_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.html")

    # Save CSVs (timestamped + pipeline-friendly)
    report_df.to_csv(csv_path, index=False)
    classic_csv_path = os.path.join(output_dir, "scan_sp500.csv")
    try:
        report_df.to_csv(classic_csv_path, index=False)
    except Exception as e:
        print(f"⚠️  Could not write {classic_csv_path}: {e}")

    # Build + write HTML (timestamped + pipeline-friendly combined)
    html = df_to_html(
        report_with_charts,
        title=f"Weinstein Weekly — Benchmark: {benchmark}",
        summary_line=summary_line
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    combined_html = os.path.join(output_dir, "combined_weekly_email.html")
    try:
        with open(combined_html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ Combined weekly report written: {combined_html}")
    except Exception as e:
        print(f"⚠️  Could not write {combined_html}: {e}")

    # Email
    subject = f"Weinstein Weekly Report — {datetime.now().strftime('%b %d, %Y')}"
    top_lines = report_df[["ticker", "stage", "buy_signal"]].head(12).to_string(index=False)
    body_text = (
        f"Summary: BUY={buy_count}, WATCH={watch_count}, AVOID={avoid_count} (Total={total})\n\n"
        f"Files:\n- {csv_path}\n- {html_path}\n- {classic_csv_path}\n- {combined_html}\n\nTop lines:\n{top_lines}\n"
    )
    send_email(subject=subject, html_body=html, text_body=body_text, cfg_path="config.yaml")

    # Optional: log to Sheets
    if log_signal is not None:
        print("Logging weekly signals to Google Sheets…")
        for _, r in report_df.iterrows():
            try:
                event = str(r.get("buy_signal", "AVOID"))
                stage = str(r.get("stage", ""))
                st_state = str(r.get("short_term_state_wk", ""))
                price = None if pd.isna(r.get("price", np.nan)) else float(r["price"])
                log_signal(
                    event=event,
                    ticker=str(r["ticker"]),
                    price=price,
                    pivot=None,
                    stage=stage,
                    short_term_state=st_state,
                    vol_pace=None,
                    notes="",
                    source="weekly"
                )
            except Exception as e:
                print(f"  Sheets log failed for {r.get('ticker')}: {e}")
    else:
        print("gsheet_helpers.log_signal not available; skipping Sheets logging.")

    print(f"Saved:\n - {csv_path}\n - {html_path}")
    print("Done.")

if __name__ == "__main__":
    main()
