# === weinstein_report_weekly.py ===
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

# 🔹 Industry helper (same module used for intraday)
from industry_utils import attach_industry

# Optional Sheets signal logger
try:
    from gsheet_helpers import log_signal
except Exception:
    log_signal = None

# ---- Matplotlib for tiny inline charts ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Optional Google Sheets pull (Signals + Mapping) ----
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

# ========= Tunables (crypto-enabled) =========
DEFAULT_BENCHMARK = "SPY"       # equities benchmark
CRYPTO_BENCHMARK  = "BTC-USD"   # crypto benchmark for RS/Stage
WEEKS_LOOKBACK = 180
MA_WEEKS = 30
MA10_WEEKS = 10
SLOPE_WINDOW = 5
NEAR_MA_BAND = 0.05
RS_MA_WEEKS = 30
OUTPUT_DIR_FALLBACK = "./output"
TOP_N_CHARTS = 20

# Spreadsheet tabs (only used if Sheets is configured)
TAB_SIGNALS = "Signals"
TAB_MAPPING = "Mapping"

# ========= Utilities =========
def _extract_field(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned by yfinance.")
    if isinstance(df.columns, pd.MultiIndex):
        avail_top = list(df.columns.get_level_values(0).unique())
        use_field = field if field in avail_top else ("Adj Close" if "Adj Close" in avail_top else None)
        if not use_field:
            raise KeyError(f"Field '{field}' not found; available: {avail_top}")
        out = df[use_field].copy()
        keep = [t for t in tickers if t in out.columns]
        if not keep:
            raise KeyError(f"No requested tickers found in downloaded data. Requested={tickers[:5]}...")
        return out[keep]
    cols = set(df.columns.astype(str))
    if field in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[[field]].copy(); out.columns = [t0]; return out
    if "Adj Close" in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[["Adj Close"]].copy(); out.columns = [t0]; return out
    raise KeyError(f"Field '{field}' not in downloaded data; got columns: {list(df.columns)}")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    app = cfg.get("app", {}) or {}
    uni = cfg.get("universe", {}) or {}
    reporting = cfg.get("reporting", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}

    mode = (uni.get("mode") or "custom").lower()
    use_sp500 = (mode == "sp500")
    extra = uni.get("extra") or []
    explicit_tickers = uni.get("tickers") or []
    benchmark = app.get("benchmark", DEFAULT_BENCHMARK)

    output_dir = reporting.get("output_dir", OUTPUT_DIR_FALLBACK)
    include_pdf = reporting.get("include_pdf", False)
    tz = app.get("timezone", "America/Chicago")
    min_price = int(uni.get("min_price", 0))
    min_avg_volume = int(uni.get("min_avg_volume", 0))

    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file  = google.get("service_account_json")

    if use_sp500:
        eq_tickers = combine_universe(sp500=True, extra_symbols=extra)
    else:
        eq_tickers = combine_universe(sp500=False, extra_symbols=explicit_tickers)

    return {
        "cfg": cfg,
        "eq_tickers": eq_tickers,
        "benchmark": benchmark,
        "tz": tz,
        "output_dir": output_dir,
        "include_pdf": include_pdf,
        "min_price": min_price,
        "min_avg_volume": min_avg_volume,
        "sheet_url": sheet_url,
        "service_account_file": svc_file
    }

def _is_crypto_symbol(sym: str) -> bool:
    s = (sym or "").strip().upper()
    # Prefer explicit YF-style like BTC-USD / ETH-USD / SOL-USD…
    return s.endswith("-USD") and len(s) >= 6 and all(ch.isalnum() or ch in "-." for ch in s)

def _auth_sheets(service_account_file: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(service_account_file, scopes=scopes)
    return gspread.authorize(creds)

def _read_tab(gc, sheet_url: str, title: str) -> pd.DataFrame:
    sh = gc.open_by_url(sheet_url)
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return pd.DataFrame()
    vals = ws.get_all_values()
    if not vals: return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    for c in df.columns:
        df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def _signals_crypto_universe(sheet_url: str, service_account_file: str) -> list[str]:
    """Harvest crypto tickers from the existing 'Signals' tab only (no rewiring)."""
    if not (gspread and Credentials and sheet_url and service_account_file and os.path.exists(service_account_file)):
        return []
    try:
        gc = _auth_sheets(service_account_file)
        sig = _read_tab(gc, sheet_url, TAB_SIGNALS)
        if sig.empty: return []
        # Prefer Mapping.TickerYF if available and looks like crypto; else use raw Ticker
        mapping = {}
        try:
            m = _read_tab(gc, sheet_url, TAB_MAPPING)
            if not m.empty and "Ticker" in m.columns:
                for _, r in m.iterrows():
                    t = str(r.get("Ticker","")).strip().upper()
                    tyf = str(r.get("TickerYF","")).strip().upper()
                    if t: mapping[t] = tyf or ""
        except Exception:
            pass

        tcol = next((c for c in sig.columns if c.lower() in ("ticker","symbol")), "Ticker")
        raw = sig[tcol].astype(str).str.upper().str.strip()
        out = []
        for t in raw:
            yf_sym = mapping.get(t, t)
            if _is_crypto_symbol(yf_sym):
                out.append(yf_sym)
        # de-dup preserve order
        uniq = list(dict.fromkeys(out))
        return uniq
    except Exception:
        return []

def fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK):
    uniq = list(dict.fromkeys((tickers or []) + [benchmark]))
    if not uniq:
        raise ValueError("No symbols to download.")
    data = yf.download(
        uniq, interval="1wk", period="10y",
        auto_adjust=True, ignore_tz=True, progress=False, group_by="column"
    )
    close = _extract_field(data, "Close", uniq)
    volume = _extract_field(data, "Volume", uniq)
    tail_n = max(weeks, MA_WEEKS + RS_MA_WEEKS + SLOPE_WINDOW + 10)
    close = close.tail(tail_n); volume = volume.tail(tail_n)
    return close, volume

def _weekly_short_term_state(series_price: pd.Series) -> tuple[str, float, float]:
    s = series_price.dropna()
    if len(s) < max(MA10_WEEKS, MA_WEEKS) + 5:
        return ("Unknown", np.nan, np.nan)
    ma10 = s.rolling(MA10_WEEKS).mean()
    ma30 = s.rolling(MA_WEEKS).mean()
    c = float(s.iloc[-1]); m10 = float(ma10.iloc[-1]); m30 = float(ma30.iloc[-1])
    state = "Unknown"
    if pd.notna(m10) and pd.notna(m30):
        if (c > m10) and (m10 > m30): state = "ShortTermUptrend"
        elif (c > m30) and not (m10 > m30): state = "StageConflict"
        elif (m10 > m30) and not (c > m10): state = "StageConflict"
        else: state = "Weak"
    return (state, m10, m30)

def compute_stage_for_ticker(closes: pd.Series, bench: pd.Series):
    s = closes.dropna().copy(); b = bench.reindex_like(s).dropna()
    idx = s.index.intersection(b.index); s = s.loc[idx]; b = b.loc[idx]
    if len(s) < MA_WEEKS + SLOPE_WINDOW + 5 or len(b) < RS_MA_WEEKS + 5:
        return {"error": "insufficient_data"}
    ma = s.rolling(MA_WEEKS).mean()
    ma_slope = ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    ma_slope_last = ma_slope.iloc[-1]; ma_last = ma.iloc[-1]; price_last = s.iloc[-1]
    dist_ma_pct = (price_last - ma_last) / ma_last if ma_last and not math.isclose(ma_last, 0.0) else np.nan
    rs = s / b
    rs_ma = rs.rolling(RS_MA_WEEKS).mean()
    rs_slope = rs_ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    rs_last = rs.iloc[-1]; rs_ma_last = rs_ma.iloc[-1]
    rs_above = bool(rs_last > rs_ma_last); rs_slope_last = rs_slope.iloc[-1]
    price_above_ma = bool(price_last > ma_last); ma_up = bool(ma_slope_last > 0)
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
    if price_above_ma and not ma_up: notes.append("Price>MA but MA not rising")
    if (not price_above_ma) and ma_up: notes.append("Price<MA but MA rising (watch)")
    if rs_above and rs_slope_last <= 0: notes.append("RS above MA but flattening")
    if (not rs_above) and rs_slope_last >= 0: notes.append("RS below MA but improving")
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
    if stage.startswith("Stage 2"): return ("BUY","BUY")
    if stage.startswith("Stage 1"): return ("WATCH","WATCH")
    if stage == "Filtered": return ("AVOID","AVOID")
    return ("AVOID","AVOID")

# ---------- Charts ----------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def make_tiny_chart_html(series_price: pd.Series, benchmark: pd.Series) -> str:
    s = series_price.dropna(); b = benchmark.reindex_like(s).dropna()
    idx = s.index.intersection(b.index)
    if len(idx) < MA_WEEKS + 5: return ""
    s = s.loc[idx]; b = b.loc[idx]
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

def attach_tiny_charts(df: pd.DataFrame, close_w: pd.DataFrame, bench_series: pd.Series, top_n: int = TOP_N_CHARTS) -> pd.DataFrame:
    out = df.copy()
    out["chart"] = ""
    for i, row in out.head(top_n).iterrows():
        t = row["ticker"]
        if t in close_w.columns:
            try:
                out.at[i, "chart"] = make_tiny_chart_html(close_w[t], bench_series)
            except Exception:
                out.at[i, "chart"] = ""
    return out

# ---------- HTML helpers ----------
def _rec_badge_html(text: str) -> str:
    t = (text or "").strip().upper()
    if t == "BUY":   cls, label = "rec rec-strong", "Buy"
    elif t == "WATCH": cls, label = "rec rec-hold", "Watch"
    elif t == "AVOID": cls, label = "rec rec-sell", "Avoid"
    else: cls, label = "rec rec-neu", (text or "—")
    return f'<span class="{cls}">{label}</span>'

def df_to_html(df: pd.DataFrame, title: str, summary_line: str):
    styled = df.copy()
    for c in ["dist_ma_pct","ma_slope_per_wk","rs_slope_per_wk"]:
        if c in styled.columns:
            styled[c] = styled[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
    if "rs_above_ma" in styled.columns:
        styled["rs_above_ma"] = styled["rs_above_ma"].map({True: "Yes", False: "No"})
    styled["Buy Signal"] = styled.get("buy_signal","").apply(_rec_badge_html)
    columns_order = [
        "ticker","asset_class","industry","sector","Buy Signal","chart",
        "stage","short_term_state_wk",
        "price","ma10","ma30","dist_ma_pct",
        "ma_slope_per_wk","rs","rs_ma30","rs_above_ma","rs_slope_per_wk","notes"
    ]
    for c in columns_order:
        if c not in styled.columns: styled[c] = ""
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
      .rec { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid transparent; letter-spacing:0.2px; }
      .rec-strong { background:#eaffea; color:#0f5e1d; border-color:#b8e7b9; }
      .rec-hold   { background:#effaf0; color:#1e7a1e; border-color:#cdebd0; }
      .rec-sell   { background:#ffe8e6; color:#8a1111; border-color:#f3b3ae; }
      .rec-neu    { background:#eef1f6; color:#4b5563; border-color:#d7dde8; }
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

# ---------- Build report ----------
def build_block(close_w: pd.DataFrame, volume_w: pd.DataFrame, tickers: list[str], bench_sym: str,
                min_price: int, min_avg_volume: int, output_dir: str, asset_class: str) -> tuple[pd.DataFrame, pd.Series]:
    if bench_sym not in close_w.columns:
        raise KeyError(f"Benchmark '{bench_sym}' not found in downloaded data.")
    bench_series = close_w[bench_sym].dropna()
    last_close = close_w.ffill().iloc[-1]
    avg_vol_10w = volume_w.rolling(10).mean().ffill().iloc[-1]

    rows = []
    for t in tickers:
        if t not in close_w.columns:
            rows.append({"ticker": t, "stage": "N/A", "notes": "no_data", "asset_class": asset_class})
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
                "notes": "below min_price/volume", "asset_class": asset_class
            })
            continue
        res = compute_stage_for_ticker(close_w[t], bench_series)
        res["ticker"] = t
        res["asset_class"] = asset_class
        rows.append(res)

    df = pd.DataFrame(rows)

    # Ensure expected cols
    cols = ["ticker","stage","price","ma10","ma30","dist_ma_pct","ma_slope_per_wk",
            "rs","rs_ma30","rs_above_ma","rs_slope_per_wk","short_term_state_wk","notes","asset_class"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    # Industry/sector for equities only; crypto will just have blanks
    df = attach_industry(
        df,
        ticker_col="ticker",
        out_col="industry",
        cache_path=os.path.join(output_dir, "industry_cache.csv")
    )
    df["buy_signal"] = df["stage"].apply(lambda s: classify_buy_signal(str(s))[0])

    stage_rank = {
        "Stage 2 (Uptrend)": 0,
        "Stage 1 (Basing)": 1,
        "Stage 3 (Topping)": 2,
        "Stage 4 (Downtrend)": 3,
        "Filtered": 8,
        "N/A": 9,
    }
    df["stage_rank"] = df["stage"].map(stage_rank).fillna(9)
    df = df.sort_values(by=["stage_rank","dist_ma_pct"], ascending=[True, False]).reset_index(drop=True)
    df = df.drop(columns=["stage_rank"])
    return df, bench_series

# ---------- Main ----------
def main():
    params = load_config()
    cfg = params["cfg"]
    eq_tickers = params["eq_tickers"]
    benchmark = params["benchmark"]
    tz = params["tz"]
    output_dir = params["output_dir"]
    include_pdf = params["include_pdf"]
    min_price = params["min_price"]
    min_avg_volume = params["min_avg_volume"]
    sheet_url = params["sheet_url"]
    service_account_file = params["service_account_file"]

    os.makedirs(output_dir, exist_ok=True)

    # Crypto universe discovered non-disruptively from existing "Signals" tab
    crypto_tickers = _signals_crypto_universe(sheet_url, service_account_file)

    # Build combined download universe: equities + crypto + both benchmarks
    all_syms = list(dict.fromkeys(eq_tickers + crypto_tickers + [benchmark, CRYPTO_BENCHMARK]))
    print(f"Universe: equities={len(eq_tickers)} crypto={len(crypto_tickers)} (bench={benchmark}, crypto_bench={CRYPTO_BENCHMARK})")
    print("Downloading weekly data (Yahoo Finance)…")
    close_w, volume_w = fetch_weekly(all_syms, benchmark, weeks=WEEKS_LOOKBACK)

    # Split blocks & build
    print("Computing Weinstein stages…")
    eq_df, eq_bench_series = build_block(
        close_w, volume_w, eq_tickers, benchmark,
        min_price=min_price, min_avg_volume=min_avg_volume,
        output_dir=output_dir, asset_class="Equity/ETF"
    )
    crypto_df = pd.DataFrame()
    crypto_bench_series = None
    if crypto_tickers:
        crypto_df, crypto_bench_series = build_block(
            close_w, volume_w, crypto_tickers, CRYPTO_BENCHMARK,
            # Crypto has no volume filter in equities units; keep min filters at 0 for safety
            min_price=0, min_avg_volume=0,
            output_dir=output_dir, asset_class="Crypto"
        )

    # Add tiny charts (separate benchmarks)
    eq_with_charts = attach_tiny_charts(eq_df, close_w, eq_bench_series, top_n=TOP_N_CHARTS)
    if crypto_tickers:
        crypto_with_charts = attach_tiny_charts(crypto_df, close_w, crypto_bench_series, top_n=TOP_N_CHARTS)
    else:
        crypto_with_charts = pd.DataFrame(columns=eq_with_charts.columns)

    # Combined CSV for downstream (intraday watcher, email attachments)
    combined = pd.concat([eq_df, crypto_df], ignore_index=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.csv")
    html_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.html")
    combined.to_csv(csv_path, index=False)

    # HTML email: Equity section + Crypto section (if any)
    buy_count = int((eq_df["buy_signal"] == "BUY").sum())
    watch_count = int((eq_df["buy_signal"] == "WATCH").sum())
    avoid_count = int((eq_df["buy_signal"] == "AVOID").sum())
    total = int(len(eq_df))
    summary_line_eq = f"<strong>Equities Summary:</strong> ✅ Buy: {buy_count} &nbsp; | &nbsp; 🟡 Watch: {watch_count} &nbsp; | &nbsp; 🔴 Avoid: {avoid_count} &nbsp; (Total: {total})"

    html_core = df_to_html(
        eq_with_charts,
        title=f"Weinstein Weekly — Equities (Benchmark: {benchmark})",
        summary_line=summary_line_eq
    )

    html_crypto = ""
    if not crypto_with_charts.empty:
        cb = int((crypto_df["buy_signal"] == "BUY").sum())
        cw = int((crypto_df["buy_signal"] == "WATCH").sum())
        ca = int((crypto_df["buy_signal"] == "AVOID").sum())
        ct = int(len(crypto_df))
        summary_line_cr = f"<strong>Crypto Summary:</strong> ✅ Buy: {cb} &nbsp; | &nbsp; 🟡 Watch: {cw} &nbsp; | &nbsp; 🔴 Avoid: {ca} &nbsp; (Total: {ct})"
        html_crypto = df_to_html(
            crypto_with_charts,
            title=f"Crypto Weekly — Benchmark: {CRYPTO_BENCHMARK}",
            summary_line=summary_line_cr
        )

    html = html_core + ("<hr/>" + html_crypto if html_crypto else "")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Email it
    subject = f"Weinstein Weekly Report — {datetime.now().strftime('%b %d, %Y')}"
    top_lines = combined[["ticker","asset_class","stage","buy_signal"]].head(12).to_string(index=False)
    body_text = (
        f"Files:\n- {csv_path}\n- {html_path}\n\nTop lines:\n{top_lines}\n"
    )
    send_email(subject=subject, html_body=html, text_body=body_text, cfg_path="config.yaml")

    # Optional: log signals to Google Sheets
    if log_signal is not None:
        print("Logging weekly signals to Google Sheets…")
        for _, r in combined.iterrows():
            try:
                event = str(r.get("buy_signal","AVOID"))
                stage = str(r.get("stage",""))
                st_state = str(r.get("short_term_state_wk",""))
                price = None if pd.isna(r.get("price", np.nan)) else float(r["price"])
                log_signal(
                    event=event,
                    ticker=str(r["ticker"]),
                    price=price,
                    pivot=None,
                    stage=stage,
                    short_term_state=st_state,
                    vol_pace=None,
                    notes=str(r.get("asset_class","")),
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
