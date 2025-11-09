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
from universe_loaders import combine_universe  # still supported as fallback

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


# --------- Tunables ----------
DEFAULT_BENCHMARK = "SPY"
WEEKS_LOOKBACK = 180      # rows to keep when computing indicators
MA_WEEKS = 30
MA10_WEEKS = 10
SLOPE_WINDOW = 5
NEAR_MA_BAND = 0.05
RS_MA_WEEKS = 30
OUTPUT_DIR_FALLBACK = "./output"
TOP_N_CHARTS = 20


# --------- Utilities ----------
def _extract_field(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """
    Robustly extract a single OHLCV field from yfinance's return (multi- or single-index).
    Falls back to 'Adj Close' if 'Close' not present.
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned by yfinance.")

    # MultiIndex (typical when requesting multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        avail_top = list(df.columns.get_level_values(0).unique())
        use_field = field if field in avail_top else ("Adj Close" if "Adj Close" in avail_top else None)
        if not use_field:
            raise KeyError(f"Field '{field}' not found; available top-level columns: {avail_top}")
        out = df[use_field].copy()
        keep = [t for t in tickers if t in out.columns]
        if not keep:
            raise KeyError(f"No requested tickers found in downloaded data. Requested={tickers[:5]}...")
        return out[keep]

    # Single ticker -> single-level columns
    cols = set(df.columns.astype(str))
    if field in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[[field]].copy()
        out.columns = [t0]
        return out
    if "Adj Close" in cols:
        t0 = tickers[0] if tickers else "TICKER"
        out = df[["Adj Close"]].copy()
        out.columns = [t0]
        return out

    raise KeyError(f"Field '{field}' not in downloaded data; got columns: {list(df.columns)}")


def _read_sheet(cfg: dict, tab_name: str) -> pd.DataFrame | None:
    """
    Read a Google Sheet tab into a DataFrame using service account creds from config.
    Returns None if config not present / read fails.
    """
    try:
        import gspread  # lazy import
        from oauth2client.service_account import ServiceAccountCredentials
        sheets_cfg = (cfg.get("sheets", {}) or {})
        sheet_url = sheets_cfg.get("sheet_url") or sheets_cfg.get("url")
        keyfile = (cfg.get("google", {}) or {}).get("service_account_json")
        if not sheet_url or not keyfile:
            return None
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(keyfile, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(tab_name)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        if df is not None and not df.empty:
            return df
        return None
    except Exception:
        return None


def _read_local_csv(path: str) -> pd.DataFrame | None:
    try:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                return df
    except Exception:
        pass
    return None


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    app = cfg.get("app", {}) or {}
    uni = cfg.get("universe", {}) or {}
    crypto_cfg = cfg.get("crypto", {}) or {}
    reporting = cfg.get("reporting", {}) or {}

    # ---- equity tickers (from sheet OR config) ----
    sheet_equities_tab = (cfg.get("sheets", {}) or {}).get("equities_tab", "Equities")
    equities_df = None
    if (cfg.get("sheets", {}) or {}).get("sheet_url") and (cfg.get("google", {}) or {}).get("service_account_json"):
        equities_df = _read_sheet(cfg, sheet_equities_tab)

    if equities_df is None:
        # fallback: universe.mode/custom + explicit tickers/extra (existing behavior)
        mode = (uni.get("mode") or "custom").lower()
        use_sp500 = (mode == "sp500")
        extra = uni.get("extra") or []
        explicit_tickers = uni.get("tickers") or []
        if use_sp500:
            equities = combine_universe(sp500=True, extra_symbols=extra)
        else:
            equities = combine_universe(sp500=False, extra_symbols=explicit_tickers)
    else:
        # From sheet: expect a column "ticker" and optional "enabled" (bool or yes/no/1/0)
        cols = {c.lower(): c for c in equities_df.columns}
        tcol = cols.get("ticker")
        ecol = cols.get("enabled")
        if not tcol:
            raise RuntimeError(f"Equities sheet '{sheet_equities_tab}' must include a 'ticker' column.")
        tmp = equities_df.copy()
        if ecol:
            def _is_on(x):
                s = str(x).strip().lower()
                return s in ("1", "true", "yes", "y", "on")
            tmp = tmp[tmp[ecol].apply(_is_on)]
        equities = [str(x).strip().upper() for x in tmp[tcol].dropna().tolist() if str(x).strip()]

    # ---- crypto tickers (from sheet OR config) ----
    crypto_enabled = bool(crypto_cfg.get("enabled", False))
    crypto_tickers = []
    if crypto_enabled:
        # prefer sheet tab, then local CSV, then config list
        sheet_crypto_tab = (cfg.get("sheets", {}) or {}).get("crypto_tab", "Crypto")
        crypto_df = _read_sheet(cfg, sheet_crypto_tab)
        if crypto_df is None:
            crypto_csv_path = crypto_cfg.get("csv")  # optional local CSV
            crypto_df = _read_local_csv(crypto_csv_path)

        if crypto_df is not None:
            # Expect column "symbol" (e.g., BTC-USD, ETH-USD) and optional "enabled"
            cols = {c.lower(): c for c in crypto_df.columns}
            sym = cols.get("symbol") or cols.get("ticker")
            ecol = cols.get("enabled")
            if not sym:
                raise RuntimeError(f"Crypto sheet/CSV must include 'symbol' (or 'ticker') column.")
            tmp = crypto_df.copy()
            if ecol:
                def _is_on(x):
                    s = str(x).strip().lower()
                    return s in ("1", "true", "yes", "y", "on")
                tmp = tmp[tmp[ecol].apply(_is_on)]
            crypto_tickers = [str(x).strip().upper() for x in tmp[sym].dropna().tolist() if str(x).strip()]
        else:
            # config list: crypto.symbols: ["BTC-USD", "ETH-USD", ...]
            crypto_tickers = [str(x).strip().upper() for x in (crypto_cfg.get("symbols") or []) if str(x).strip()]

    benchmark = app.get("benchmark", DEFAULT_BENCHMARK)
    tz = app.get("timezone", "America/Chicago")

    output_dir = reporting.get("output_dir", OUTPUT_DIR_FALLBACK)
    include_pdf = reporting.get("include_pdf", False)

    min_price = int(uni.get("min_price", 0))
    min_avg_volume = int(uni.get("min_avg_volume", 0))

    return cfg, equities, crypto_tickers, benchmark, tz, output_dir, include_pdf, min_price, min_avg_volume, crypto_enabled


def fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK):
    uniq = list(dict.fromkeys((tickers or []) + ([benchmark] if benchmark else [])))  # de-dup preserve order
    if not uniq:
        raise ValueError("No symbols to download.")

    data = yf.download(
        uniq,
        interval="1wk",
        period="10y",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
        group_by="column",
    )

    close = _extract_field(data, "Close", uniq)
    volume = _extract_field(data, "Volume", uniq)

    tail_n = max(weeks, MA_WEEKS + RS_MA_WEEKS + SLOPE_WINDOW + 10)
    close = close.tail(tail_n)
    volume = volume.tail(tail_n)

    return close, volume


def _weekly_short_term_state(series_price: pd.Series) -> tuple[str, float, float]:
    """
    Weekly short-term vs long-term using 10-wk vs 30-wk MAs.
    Returns (state, ma10_last, ma30_last)
    """
    s = series_price.dropna()
    if len(s) < max(MA10_WEEKS, MA_WEEKS) + 5:
        return ("Unknown", np.nan, np.nan)

    ma10 = s.rolling(MA10_WEEKS).mean()
    ma30 = s.rolling(MA_WEEKS).mean()
    c = float(s.iloc[-1])
    m10 = float(ma10.iloc[-1])
    m30 = float(ma30.iloc[-1])

    state = "Unknown"
    if pd.notna(m10) and pd.notna(m30):
        if (c > m10) and (m10 > m30):
            state = "ShortTermUptrend"
        elif (c > m30) and not (m10 > m30):
            state = "StageConflict"
        elif (m10 > m30) and not (c > m10):
            state = "StageConflict"
        else:
            state = "Weak"
    return (state, m10, m30)


def compute_stage_for_ticker(closes: pd.Series, bench: pd.Series | None):
    s = closes.dropna().copy()
    if bench is not None:
        b = bench.reindex_like(s).dropna()
        idx = s.index.intersection(b.index)
        s = s.loc[idx]
        b = b.loc[idx]
        if len(s) < MA_WEEKS + SLOPE_WINDOW + 5 or len(b) < RS_MA_WEEKS + 5:
            return {"error": "insufficient_data"}
    else:
        # Crypto without a separate benchmark: use self-RS (flat = 1.0) so logic still runs
        b = pd.Series(np.ones(len(s)), index=s.index)
        if len(s) < MA_WEEKS + SLOPE_WINDOW + 5:
            return {"error": "insufficient_data"}

    ma = s.rolling(MA_WEEKS).mean()

    # MA slope over last SLOPE_WINDOW weeks
    ma_slope = ma.diff(SLOPE_WINDOW) / float(SLOPE_WINDOW)
    ma_slope_last = ma_slope.iloc[-1]
    ma_last = ma.iloc[-1]
    price_last = s.iloc[-1]
    dist_ma_pct = (price_last - ma_last) / ma_last if ma_last and not math.isclose(ma_last, 0.0) else np.nan

    # Relative Strength line vs benchmark (or flat line if bench is None)
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
    near_ma = bool(abs(dist_ma_pct) <= NEAR_MA_BAND)
    rs_up = bool(rs_above and rs_slope_last > 0)
    rs_down = bool((not rs_above) and rs_slope_last < 0)

    # Stage rules (heuristic)
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

    st_state, ma10_last, _ma30_chk = _weekly_short_term_state(s)

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
        return ("BUY", "BUY")
    if stage.startswith("Stage 1"):
        return ("WATCH", "WATCH")
    if stage == "Filtered":
        return ("AVOID", "AVOID")
    return ("AVOID", "AVOID")


def _build_report_df_core(close_w: pd.DataFrame,
                          volume_w: pd.DataFrame,
                          tickers: list[str],
                          benchmark: str | None,
                          min_price: int = 0,
                          min_avg_volume: int = 0,
                          output_dir: str = OUTPUT_DIR_FALLBACK,
                          asset_class: str = "Equity") -> pd.DataFrame:
    """
    Core builder used for both Equities and Crypto.
    For Crypto, pass benchmark=None to compute RS against a flat series.
    """
    bench_series = None
    if benchmark and benchmark in close_w.columns:
        bench_series = close_w[benchmark].dropna()

    last_close = close_w.ffill().iloc[-1]
    avg_vol_10w = volume_w.rolling(10).mean().ffill().iloc[-1]

    rows = []
    for t in tickers:
        if t not in close_w.columns:
            rows.append({"ticker": t, "stage": "N/A", "notes": "no_data", "asset_class": asset_class})
            continue

        lc = float(last_close.get(t, np.nan)) if pd.notna(last_close.get(t, np.nan)) else np.nan
        av = float(avg_vol_10w.get(t, np.nan)) if pd.notna(avg_vol_10w.get(t, np.nan)) else np.nan

        # For equities only, enforce min_price/volume (crypto often lacks meaningful volume in YF weekly)
        if asset_class == "Equity" and ((min_price and (pd.isna(lc) or lc < min_price)) or (min_avg_volume and (pd.isna(av) or av < min_avg_volume))):
            s = close_w[t].dropna()
            st_state, ma10_last, _ = _weekly_short_term_state(s)
            rows.append({
                "ticker": t, "stage": "Filtered", "price": lc,
                "ma10": float(ma10_last) if pd.notna(ma10_last) else np.nan,
                "ma30": np.nan, "short_term_state_wk": st_state,
                "notes": "below min_price/volume",
                "asset_class": asset_class
            })
            continue

        res = compute_stage_for_ticker(close_w[t], bench_series)
        res["ticker"] = t
        res["asset_class"] = asset_class
        rows.append(res)

    df = pd.DataFrame(rows)

    # Ensure expected columns exist
    cols = [
        "ticker", "stage", "price", "ma10", "ma30", "dist_ma_pct", "ma_slope_per_wk",
        "rs", "rs_ma30", "rs_above_ma", "rs_slope_per_wk", "short_term_state_wk", "notes", "asset_class"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    # Attach industry/sector for equities only (keeps existing cache/use)
    if asset_class == "Equity":
        df = attach_industry(
            df,
            ticker_col="ticker",
            out_col="industry",
            cache_path=os.path.join(output_dir, "industry_cache.csv")
        )
    else:
        # create empty columns for homogeneous layout
        if "industry" not in df.columns: df["industry"] = ""
        if "sector" not in df.columns: df["sector"] = ""

    # Buy signal (plain text; HTML badge is applied in df_to_html)
    df["buy_signal"] = df["stage"].apply(lambda s: classify_buy_signal(str(s))[0])

    # Sort per asset class
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


def build_report_df(close_w: pd.DataFrame,
                    volume_w: pd.DataFrame,
                    equities: list[str],
                    crypto: list[str],
                    benchmark: str,
                    min_price: int,
                    min_avg_volume: int,
                    output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (equity_df, crypto_df)
    """
    eq_df = _build_report_df_core(
        close_w=close_w, volume_w=volume_w, tickers=equities,
        benchmark=benchmark, min_price=min_price, min_avg_volume=min_avg_volume,
        output_dir=output_dir, asset_class="Equity"
    )

    cr_df = pd.DataFrame(columns=eq_df.columns)  # default empty with same schema
    if crypto:
        cr_df = _build_report_df_core(
            close_w=close_w, volume_w=volume_w, tickers=crypto,
            benchmark=None,  # RS vs flat
            min_price=0, min_avg_volume=0,  # no equity filters
            output_dir=output_dir, asset_class="Crypto"
        )

    return eq_df, cr_df


# ---------- Tiny inline charts ----------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def make_tiny_chart_html(series_price: pd.Series, benchmark: pd.Series | None) -> str:
    s = series_price.dropna()
    if benchmark is not None:
        b = benchmark.reindex_like(s).dropna()
        idx = s.index.intersection(b.index)
        if len(idx) < MA_WEEKS + 5:
            return ""
        s = s.loc[idx]
        b = b.loc[idx]
    else:
        # Crypto chart without separate benchmark: show price + MAs (no RS)
        if len(s) < MA_WEEKS + 5:
            return ""
        b = None

    ma30 = s.rolling(MA_WEEKS).mean()
    ma10 = s.rolling(MA10_WEEKS).mean()

    fig, ax1 = plt.subplots(figsize=(3.0, 1.4))
    ax1.plot(s.index, s.values, linewidth=1.2)
    ax1.plot(ma10.index, ma10.values, linewidth=1.0)
    ax1.plot(ma30.index, ma30.values, linewidth=1.0)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)

    if b is not None:
        rs = (s / b).rolling(RS_MA_WEEKS).mean()
        ax2 = ax1.twinx()
        ax2.plot(rs.index, rs.values, linewidth=0.8)
        ax2.set_xticks([]); ax2.set_yticks([])
        for spine in (*ax1.spines.values(), *ax2.spines.values()):
            spine.set_visible(False)
    else:
        for spine in (*ax1.spines.values(),):
            spine.set_visible(False)

    img_src = _fig_to_base64(fig)
    return f'<img src="{img_src}" alt="chart" style="display:block;width:100%;max-width:240px;height:auto;border:0" />'


def attach_tiny_charts(df: pd.DataFrame,
                       close_w: pd.DataFrame,
                       benchmark: str | None,
                       top_n: int = TOP_N_CHARTS) -> pd.DataFrame:
    out = df.copy()
    out["chart"] = ""
    bench_series = None
    if benchmark and benchmark in close_w.columns:
        bench_series = close_w[benchmark].dropna()

    for i, row in out.head(top_n).iterrows():
        t = row["ticker"]
        if t in close_w.columns:
            try:
                img = make_tiny_chart_html(close_w[t], bench_series if row.get("asset_class") == "Equity" else None)
                out.at[i, "chart"] = img
            except Exception:
                out.at[i, "chart"] = ""
    return out


# ---------- HTML helpers (colors/badges) ----------
def _rec_badge_html(text: str) -> str:
    t = (text or "").strip().upper()
    if t == "BUY":
        cls = "rec rec-strong"; label = "Buy"
    elif t == "WATCH":
        cls = "rec rec-hold"; label = "Watch"
    elif t == "AVOID":
        cls = "rec rec-sell"; label = "Avoid"
    else:
        cls = "rec rec-neu"; label = text or "—"
    return f'<span class="{cls}">{label}</span>'


def _hold_badge_html(text: str) -> str:
    t = (text or "").strip().upper()
    if t == "HOLD (STRONG)":
        cls = "rec rec-strong"; label = "HOLD (Strong)"
    elif t == "HOLD":
        cls = "rec rec-hold"; label = "HOLD"
    elif t == "SELL":
        cls = "rec rec-sell"; label = "SELL"
    else:
        cls = "rec rec-neu"; label = text or "—"
    return f'<span class="{cls}">{label}</span>'


def _pct_cell_html_percent_units(pct_number):
    if pct_number is None or pd.isna(pct_number):
        klass = "pct neu"
    else:
        if pct_number > 0: klass = "pct pos"
        elif pct_number < 0: klass = "pct neg"
        else: klass = "pct neu"
    txt = f"{pct_number:.2f}%" if pct_number is not None and pd.notna(pct_number) else ""
    return f'<span class="{klass}">{txt}</span>'


def _money(amount_float):
    return f"${amount_float:,.2f}" if (amount_float is not None and pd.notna(amount_float)) else ""


def _money_cell_html(amount_float):
    if amount_float is None or pd.isna(amount_float):
        klass = "money neu"
    else:
        if amount_float > 0: klass = "money pos"
        elif amount_float < 0: klass = "money neg"
        else: klass = "money neu"
    return f'<span class="{klass}">{_money(amount_float)}</span>'


def _pct_str_fraction(x_float_as_fraction):
    return f"{x_float_as_fraction*100:.2f}%" if (x_float_as_fraction is not None and pd.notna(x_float_as_fraction)) else ""


def _summary_row_html(metric: str, value_str: str, numeric_value: float | None) -> str:
    if numeric_value is None or pd.isna(numeric_value):
        klass = "val neu"
    else:
        if numeric_value > 0: klass = "val pos"
        elif numeric_value < 0: klass = "val neg"
        else: klass = "val neu"
    return f"<tr><td class='metric'>{metric}</td><td class='{klass}'>{value_str}</td></tr>"


def df_to_html(df: pd.DataFrame, title: str, summary_line: str, is_crypto: bool = False):
    styled = df.copy()

    # Pretty percentages
    for c in ["dist_ma_pct", "ma_slope_per_wk", "rs_slope_per_wk"]:
        if c in styled.columns:
            styled[c] = styled[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")

    if "rs_above_ma" in styled.columns:
        styled["rs_above_ma"] = styled["rs_above_ma"].map({True: "Yes", False: "No"})

    # Badges for buy_signal
    styled["Buy Signal"] = styled.get("buy_signal", "").apply(_rec_badge_html)

    # Include industry/sector for equities; omit for crypto
    if is_crypto:
        columns_order = [
            "ticker", "Buy Signal", "chart",
            "stage", "short_term_state_wk",
            "price", "ma10", "ma30", "dist_ma_pct",
            "ma_slope_per_wk", "rs", "rs_ma30", "rs_above_ma", "rs_slope_per_wk",
            "notes"
        ]
    else:
        columns_order = [
            "ticker", "industry", "sector", "Buy Signal", "chart",
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

      .rec {
        display:inline-block; padding:2px 8px; border-radius:999px;
        font-size:12px; font-weight:700; border:1px solid transparent; letter-spacing:0.2px;
      }
      .rec-strong { background:#eaffea; color:#0f5e1d; border-color:#b8e7b9; }
      .rec-hold   { background:#effaf0; color:#1e7a1e; border-color:#cdebd0; }
      .rec-sell   { background:#ffe8e6; color:#8a1111; border-color:#f3b3ae; }
      .rec-neu    { background:#eef1f6; color:#4b5563; border-color:#d7dde8; }

      .pct, .money { font-weight: 600; }
      .pct.pos, .money.pos { color: #106b21; }
      .pct.neg, .money.neg { color: #8a1111; }
      .pct.neu, .money.neu { color: #555; }

      .summary-table { width: 100%; margin: 4px 0 10px 0; }
      .summary-table th { background:#f8f9fb; color:#333; }
      .metric { width: 50%; }
      .val { font-weight: 700; }
      .val.pos { color: #106b21; }
      .val.neg { color: #8a1111; }
      .val.neu { color: #555; }

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


# --- Holdings + snapshot helpers (colored, with industry/sector) ---------------
def _try_read_open_positions_local(output_dir: str) -> pd.DataFrame | None:
    for fname in ["Open_Positions.csv", "open_positions.csv"]:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    return df
            except Exception:
                pass
    return None


def _read_open_positions_gsheet(cfg: dict, tab_name: str = "Open_Positions") -> pd.DataFrame:
    import gspread  # lazy import
    from oauth2client.service_account import ServiceAccountCredentials

    sheet_url = (cfg.get("sheets", {}) or {}).get("sheet_url") or (cfg.get("sheets", {}) or {}).get("url")
    keyfile = (cfg.get("google", {}) or {}).get("service_account_json")
    if not sheet_url or not keyfile:
        raise RuntimeError("Missing sheets.sheet_url/url or google.service_account_json in config.yaml")

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(keyfile, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(sheet_url)

    if (cfg.get("sheets", {}) or {}).get("open_positions_tab"):
        tab_name = cfg["sheets"]["open_positions_tab"]
    ws = sh.worksheet(tab_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)


def _coerce_numlike(series: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).replace(",", "").replace("$", "").strip()
        if s.endswith("%"):
            s = s[:-1]
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.apply(conv)


def _normalize_open_positions_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {
        "Ticker": "Symbol", "symbol": "Symbol", "SYMBOL": "Symbol",
        "Qty": "Quantity", "Shares": "Quantity", "quantity": "Quantity",
        "Last": "Last Price", "Price": "Last Price", "LastPrice": "Last Price",
        "Current Value $": "Current Value", "Market Value": "Current Value", "MarketValue": "Current Value",
        "Cost Basis": "Cost Basis Total", "Cost": "Cost Basis Total",
        "Avg Cost": "Average Cost Basis", "AvgCost": "Average Cost Basis",
        "Gain $": "Total Gain/Loss Dollar", "Gain": "Total Gain/Loss Dollar",
        "Gain %": "Total Gain/Loss Percent", "GainPct": "Total Gain/Loss Percent",
        "Name": "Description", "Description/Name": "Description",
    }
    out = df.rename(columns=ren).copy()

    required = [
        "Symbol","Description","Quantity","Last Price","Current Value",
        "Cost Basis Total","Average Cost Basis",
        "Total Gain/Loss Dollar","Total Gain/Loss Percent"
    ]
    for c in required:
        if c not in out.columns:
            out[c] = np.nan

    num_cols = ["Quantity","Last Price","Current Value","Cost Basis Total",
                "Average Cost Basis","Total Gain/Loss Dollar","Total Gain/Loss Percent"]
    for c in num_cols:
        out[c] = _coerce_numlike(out[c])

    cols = ["Symbol","Description","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis",
            "Total Gain/Loss Dollar","Total Gain/Loss Percent"]
    out = out[cols]
    out = out.dropna(how="all")
    return out


def _compute_portfolio_metrics(positions: pd.DataFrame) -> dict:
    cur = float(positions["Current Value"].fillna(0).sum())
    cost = float(positions["Cost Basis Total"].fillna(0).sum())
    gl_dollar = cur - cost
    port_pct = (gl_dollar / cost) if cost else 0.0

    row_pct = positions["Total Gain/Loss Percent"].dropna().astype(float)
    avg_pct = float(row_pct.mean())/100.0 if len(row_pct) else 0.0

    return {
        "total_gl_dollar": gl_dollar,
        "portfolio_pct_gain": port_pct,
        "average_pct_gain": avg_pct,
        "total_current_value": cur,
        "total_cost_basis": cost,
    }


def _merge_stage_and_recommend(positions: pd.DataFrame, stage_df: pd.DataFrame) -> pd.DataFrame:
    stage_min = stage_df[["ticker","stage","rs_above_ma","industry","sector"]].rename(columns={"ticker":"Symbol"})
    out = positions.merge(stage_min, on="Symbol", how="left")

    def recommend(row):
        pct = row.get("Total Gain/Loss Percent", np.nan)  # PERCENT units
        stage = str(row.get("stage", ""))
        rs_above = bool(row.get("rs_above_ma", False))

        if stage.startswith("Stage 2") and (rs_above or (pd.notna(pct) and pct >= 20)):
            return "HOLD (Strong)"
        if stage.startswith("Stage 4") and pd.notna(pct) and pct < 0:
            return "SELL"
        if pd.notna(pct) and pct <= -8:
            return "SELL"
        return "HOLD"

    out["Recommendation"] = out.apply(recommend, axis=1)
    return out


def holdings_sections_html(positions_merged: pd.DataFrame, metrics: dict) -> str:
    total_gl = metrics["total_gl_dollar"]
    port_pct = metrics["portfolio_pct_gain"]
    avg_pct  = metrics["average_pct_gain"]

    def _pct_str_fraction(x): return f"{x*100:.2f}%"

    def _summary_row_html(metric: str, value_str: str, numeric_value: float | None) -> str:
        if numeric_value is None or pd.isna(numeric_value):
            klass = "val neu"
        else:
            if numeric_value > 0: klass = "val pos"
            elif numeric_value < 0: klass = "val neg"
            else: klass = "val neu"
        return f"<tr><td class='metric'>{metric}</td><td class='{klass}'>{value_str}</td></tr>"

    summary_rows = [
        _summary_row_html("Total Gain/Loss ($)", _money(total_gl), total_gl),
        _summary_row_html("Portfolio % Gain",     _pct_str_fraction(port_pct),  port_pct),
        _summary_row_html("Average % Gain",       _pct_str_fraction(avg_pct),   avg_pct),
    ]
    summary_html = f"""
    <table class="summary-table">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {''.join(summary_rows)}
      </tbody>
    </table>
    """

    snap = positions_merged.copy()
    raw_pct = snap["Total Gain/Loss Percent"].copy()
    raw_gl_dollar = snap["Total Gain/Loss Dollar"].copy()

    snap["Last Price"] = snap["Last Price"].apply(_money)
    snap["Current Value"] = snap["Current Value"].apply(_money)
    snap["Cost Basis Total"] = snap["Cost Basis Total"].apply(_money)
    snap["Average Cost Basis"] = snap["Average Cost Basis"].apply(_money)

    def _pct_cell_html_percent_units(pct_number):
        if pct_number is None or pd.isna(pct_number):
            klass = "pct neu"
        else:
            if pct_number > 0: klass = "pct pos"
            elif pct_number < 0: klass = "pct neg"
            else: klass = "pct neu"
        txt = f"{pct_number:.2f}%" if pct_number is not None and pd.notna(pct_number) else ""
        return f'<span class="{klass}">{txt}</span>'

    def _money_cell_html(amount_float):
        if amount_float is None or pd.isna(amount_float):
            klass = "money neu"
        else:
            if amount_float > 0: klass = "money pos"
            elif amount_float < 0: klass = "money neg"
            else: klass = "money neu"
        return f'<span class="{klass}">{_money(amount_float)}</span>'

    snap["TGLP_colored"] = raw_pct.apply(_pct_cell_html_percent_units)
    snap["TGLD_colored"] = raw_gl_dollar.apply(_money_cell_html)
    snap["RecommendationBadge"] = snap["Recommendation"].apply(_hold_badge_html)

    cols = ["Symbol","Description","industry","sector","Quantity","Last Price","Current Value",
            "Cost Basis Total","Average Cost Basis",
            "TGLD_colored","TGLP_colored","RecommendationBadge"]
    for c in cols:
        if c not in snap.columns:
            snap[c] = ""
    snap = snap[cols].rename(columns={
        "TGLD_colored": "Total Gain/Loss Dollar",
        "TGLP_colored": "Total Gain/Loss Percent",
        "RecommendationBadge": "Recommendation"
    })

    snapshot_html = snap.to_html(index=False, border=0, escape=False)

    css = """
    <style>
      .blk { margin-top:20px; }
      .blk h3 { margin: 0 0 6px 0; }
      table { border-collapse: collapse; width: 100%; }
      th, td { padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 14px; vertical-align: top; }
      th { text-align:left; background:#fafafa; }
      .summary-table { width: 100%; margin: 4px 0 10px 0; }
      .summary-table th { background:#f8f9fb; color:#333; }
      .metric { width: 50%; }
      .val { font-weight: 700; }
      .val.pos { color: #106b21; }
      .val.neg { color: #8a1111; }
      .val.neu { color: #555; }
      .rec { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; border: 1px solid transparent; letter-spacing: 0.2px; }
      .rec-strong { background: #eaffea; color: #0f5e1d; border-color: #b8e7b9; }
      .rec-hold   { background: #effaf0; color: #1e7a1e; border-color: #cdebd0; }
      .rec-sell   { background: #ffe8e6; color: #8a1111; border-color: #f3b3ae; }
      .rec-neu    { background: #eef1f6; color: #4b5563; border-color: #d7dde8; }
      .pct, .money { font-weight: 700; }
      .pct.pos, .money.pos { color: #106b21; }
      .pct.neg, .money.neg { color: #8a1111; }
      .pct.neu, .money.neu { color: #555; }
    </style>
    """
    return f"""{css}
    <div class="blk">
      <h3>Weinstein Weekly – Summary</h3>
      {summary_html}
    </div>
    <div class="blk">
      <h3>Per-position Snapshot</h3>
      {snapshot_html}
    </div>
    """


# ---------- Main ----------
def main():
    (cfg, equities, crypto, benchmark, tz, output_dir, include_pdf,
     min_price, min_avg_volume, crypto_enabled) = load_config()

    if not equities and not crypto:
        print("No tickers resolved from config/sheets.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Universe for download = equities + crypto + (benchmark if present)
    universe = list(dict.fromkeys((equities or []) + (crypto or []) + ([benchmark] if benchmark else [])))
    print(f"Universe size: {len(universe)} symbols (benchmark: {benchmark})")
    print("Downloading weekly data (Yahoo Finance)…")
    close_w, volume_w = fetch_weekly(universe, benchmark, weeks=WEEKS_LOOKBACK)

    print("Computing Weinstein stages…")
    eq_df, cr_df = build_report_df(
        close_w=close_w,
        volume_w=volume_w,
        equities=equities,
        crypto=crypto,
        benchmark=benchmark,
        min_price=min_price,
        min_avg_volume=min_avg_volume,
        output_dir=output_dir
    )

    # Add tiny charts
    eq_with_charts = attach_tiny_charts(eq_df, close_w, benchmark, top_n=TOP_N_CHARTS)
    cr_with_charts = attach_tiny_charts(cr_df, close_w, None, top_n=TOP_N_CHARTS) if crypto_enabled and not cr_df.empty else cr_df

    # Summary counts
    def _counts(df):
        if df is None or df.empty:
            return (0,0,0,0)
        buy = int((df["buy_signal"] == "BUY").sum())
        watch = int((df["buy_signal"] == "WATCH").sum())
        avoid = int((df["buy_signal"] == "AVOID").sum())
        total = int(len(df))
        return (buy, watch, avoid, total)

    buy_e, watch_e, avoid_e, total_e = _counts(eq_df)
    buy_c, watch_c, avoid_c, total_c = _counts(cr_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path_eq = os.path.join(output_dir, f"weinstein_weekly_equities_{ts}.csv")
    csv_path_cr = os.path.join(output_dir, f"weinstein_weekly_crypto_{ts}.csv")
    html_path = os.path.join(output_dir, f"weinstein_weekly_{ts}.html")

    # Save CSVs
    eq_df.to_csv(csv_path_eq, index=False)
    if crypto_enabled and not cr_df.empty:
        cr_df.to_csv(csv_path_cr, index=False)

    # Build HTML (two sections)
    summary_line_eq = f"<strong>Equities:</strong> ✅ Buy: {buy_e} &nbsp; | &nbsp; 🟡 Watch: {watch_e} &nbsp; | &nbsp; 🔴 Avoid: {avoid_e} &nbsp; (Total: {total_e})"
    html_eq = df_to_html(eq_with_charts, title=f"Weinstein Weekly — Equities (Benchmark: {benchmark})", summary_line=summary_line_eq, is_crypto=False)

    if crypto_enabled:
        summary_line_cr = f"<strong>Crypto:</strong> ✅ Buy: {buy_c} &nbsp; | &nbsp; 🟡 Watch: {watch_c} &nbsp; | &nbsp; 🔴 Avoid: {avoid_c} &nbsp; (Total: {total_c})"
        html_cr = df_to_html(cr_with_charts, title="Weinstein Weekly — Crypto", summary_line=summary_line_cr, is_crypto=True)
    else:
        html_cr = "<!-- crypto disabled -->"

    # --- Holdings block (colored summary + colored cells + badges) ---
    holdings_df = _try_read_open_positions_local(output_dir)
    if holdings_df is None:
        try:
            holdings_df = _read_open_positions_gsheet(cfg, tab_name=cfg.get("sheets", {}).get("open_positions_tab","Open_Positions"))
        except Exception as e:
            print(f"⚠️  Could not load Open_Positions (no CSV and sheet read failed): {e}")
            holdings_df = None

    extra_html = ""
    if holdings_df is not None and not holdings_df.empty:
        # only need minimal stage/rs/industry/sector for held symbols (equities only merge makes sense)
        stage_df_for_merge = eq_df[["ticker","stage","rs_above_ma","industry","sector"]].copy()
        pos_norm = _normalize_open_positions_columns(holdings_df)
        pos_merged = _merge_stage_and_recommend(pos_norm, stage_df_for_merge)
        metrics = _compute_portfolio_metrics(pos_norm)
        extra_html = holdings_sections_html(pos_merged, metrics)
    else:
        extra_html = "<!-- holdings not available -->"

    html = html_eq + "<hr/>" + html_cr + extra_html

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Email it
    subject = f"Weinstein Weekly Report — {datetime.now().strftime('%b %d, %Y')}"
    top_lines_eq = eq_df[["ticker", "stage", "buy_signal"]].head(12).to_string(index=False)
    if crypto_enabled and not cr_df.empty:
        top_lines_cr = cr_df[["ticker", "stage", "buy_signal"]].head(12).to_string(index=False)
    else:
        top_lines_cr = "(no crypto configured)"

    body_text = (
        f"Equities Summary: BUY={buy_e}, WATCH={watch_e}, AVOID={avoid_e} (Total={total_e})\n"
        f"Crypto Summary:   BUY={buy_c}, WATCH={watch_c}, AVOID={avoid_c} (Total={total_c})\n\n"
        f"Files:\n- {csv_path_eq}\n"
        + (f"- {csv_path_cr}\n" if crypto_enabled and not cr_df.empty else "")
        + f"- {html_path}\n\nTop (Equities):\n{top_lines_eq}\n\nTop (Crypto):\n{top_lines_cr}\n"
    )
    send_email(subject=subject, html_body=html, text_body=body_text, cfg_path="config.yaml")

    # Optional: log signals to Google Sheets
    if log_signal is not None:
        print("Logging weekly signals to Google Sheets…")
        # equities
        for _, r in eq_df.iterrows():
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
                    source="weekly_equities"
                )
            except Exception as e:
                print(f"  Sheets log failed (equity) for {r.get('ticker')}: {e}")
        # crypto
        for _, r in cr_df.iterrows():
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
                    source="weekly_crypto"
                )
            except Exception as e:
                print(f"  Sheets log failed (crypto) for {r.get('ticker')}: {e}")
    else:
        print("gsheet_helpers.log_signal not available; skipping Sheets logging.")

    print(f"Saved:\n - {csv_path_eq}\n" + (f" - {csv_path_cr}\n" if crypto_enabled and not cr_df.empty else "") + f" - {html_path}")
    print("Done.")


if __name__ == "__main__":
    main()
