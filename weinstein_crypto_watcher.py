#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weinstein Crypto Watch — crypto-specific intraday scan (no intrabar/elapsed gating)

- Stage 1/2 + pivot + SMA150 confirmation
- 24h volume pace vs 50d avg (projected for the current UTC day)
- NEAR/ARMED/COOLDOWN state machine (like equities) but without 60m elapsed/pace checks
- Ownership-aware snapshot/actions from CryptoHoldings Google Sheet tab
- Tiny RS charts (RS vs BTC-USD), same mailer as equities
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
from weinstein_long_core import LongEntryParams, evaluate_long_signal

# ---------------- Tunables (crypto) ----------------
OUTPUT_DIR = "./output"
CHART_DIR = "./output/charts"
STATE_FILE = "./state/crypto_triggers.json"

BENCHMARK_DEFAULT = "BTC-USD"   # RS baseline for crypto charts/text
INTRADAY_INTERVAL = "60m"       # 60m is fine; we do NOT require intrabar/elapsed confirmation
LOOKBACK_DAYS = 60
PIVOT_LOOKBACK_WEEKS = 10

# BUY / NEAR thresholds
MIN_BREAKOUT_PCT = 0.004            # +0.4% above ~10-week pivot & SMA150
BUY_DIST_ABOVE_MA_MIN = 0.00        # allow = at/just above MA
VOL_PACE_MIN = 1.20                 # 24h pace vs 50dma must be >= 1.2×
NEAR_BELOW_PIVOT_PCT = 0.004        # within 0.4% below pivot qualifies as NEAR
NEAR_VOL_PACE_MIN = 0.90            # NEAR gate for pace

# SELL trigger threshold
SELL_BREAK_PCT = 0.006              # break below SMA150 by 0.6% (confirmed)
SMA_DAYS = 150                      # daily proxy for 30-wk MA on charts

# State machine
SCAN_INTERVAL_MIN = 10
NEAR_HITS_WINDOW = 6
NEAR_HITS_MIN = 3
COOLDOWN_SCANS = 24

MAX_CHARTS_PER_EMAIL = 12
VERBOSE = True

# Optional Sheets (for CryptoHoldings tab)
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread, Credentials = None, None

TAB_SIGNALS  = "Signals"          # used only to find the Sheet / Mapping
TAB_MAPPING  = "Mapping"
TAB_CRYPTO_HOLD = "CryptoHoldings"


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, *, level="info"):
    if not VERBOSE and level == "debug":
        return
    prefix = {
        "info": "•",
        "ok": "✅",
        "step": "▶️",
        "warn": "⚠️",
        "err": "❌",
        "debug": "··",
    }.get(level, "•")
    print(f"{prefix} [{_ts()}] {msg}", flush=True)


# ---------------- Config helpers ----------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {}) or {}
    sheets = cfg.get("sheets", {}) or {}
    google = cfg.get("google", {}) or {}
    sheet_url = sheets.get("url") or sheets.get("sheet_url")
    svc_file = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or google.get("service_account_json")
    )

    # optional crypto universe from YAML
    crypto = cfg.get("crypto", {}) or {}
    universe = crypto.get("universe") or [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "BCH-USD",
        "DOGE-USD", "LINK-USD", "LTC-USD", "MATIC-USD", "NEAR-USD",
        "TON-USD", "TRX-USD", "XRP-USD"
        # ARB-USD intentionally removed from default universe
    ]
    return cfg, sheet_url, svc_file, universe


def newest_weekly_csv():
    files = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("weinstein_weekly_") and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            "No weekly CSV found in ./output. Run your weekly report first."
        )

    # Pick the newest generated weekly file by modified time, not filename.
    # This avoids selecting placeholder/stale files such as:
    #   weinstein_weekly_equities_YYYY.csv
    files.sort(
        key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        reverse=True,
    )
    return os.path.join(OUTPUT_DIR, files[0])


def load_weekly():
    p = newest_weekly_csv()
    df = pd.read_csv(p)
    return df, p


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
        uniq,
        period=f"{LOOKBACK_DAYS}d",
        interval=INTRADAY_INTERVAL,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
    )
    daily = yf.download(
        uniq,
        period="24mo",
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
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


def volume_pace_today_vs_50dma_crypto(ticker, daily_df):
    """24h projected pace vs 50-day average (UTC midnight-to-midnight)."""
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
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = max(0.0, (now - day_start).total_seconds())
    fraction = min(1.0, max(0.05, elapsed / (24 * 3600.0)))
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


# ---------------- State helpers ----------------
def _load_state():
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


def _update_hits(arr, hit, window):
    arr = (arr or [])
    arr.append(1 if hit else 0)
    if len(arr) > window:
        arr = arr[-window:]
    return arr, sum(arr)


def stage_order(stage: str) -> int:
    if isinstance(stage, str):
        if stage.startswith("Stage 2"):
            return 0
        if stage.startswith("Stage 1"):
            return 1
    return 9


# ---------------- Charts ----------------
def make_tiny_chart_png(ticker, benchmark, daily_df):
    os.makedirs(CHART_DIR, exist_ok=True)
    if not isinstance(daily_df.columns, pd.MultiIndex):
        return None, None
    try:
        close_t = daily_df[("Close", ticker)].dropna().tail(260)
        close_b = daily_df[("Close", benchmark)].dropna()
    except KeyError:
        return None, None

    idx = close_t.index.intersection(close_b.index)
    if len(idx) < 50:
        return None, None

    close_t, close_b = close_t.loc[idx], close_b.loc[idx]
    sma = close_t.rolling(SMA_DAYS).mean()
    rs = (close_t / close_b)
    rs_norm = rs / rs.iloc[0]

    fig, ax1 = plt.subplots(figsize=(5.0, 2.4), dpi=150)
    ax1.plot(close_t.index, close_t.values, label=f"{ticker}")
    ax1.plot(sma.index, sma.values, label=f"SMA{SMA_DAYS}", linewidth=1.2)
    ax1.set_ylabel("Price")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(
        rs_norm.index,
        rs_norm.values,
        linestyle="--",
        alpha=0.7,
        label="RS (norm vs BTC)",
    )
    ax2.set_ylabel("RS (norm)")
    ax2.tick_params(axis="y", labelsize=8)

    ax1.set_title(f"{ticker} — Price, SMA{SMA_DAYS}, RS/{benchmark}", fontsize=9)
    ax1.grid(alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=7,
        loc="upper left",
        frameon=False,
    )

    chart_path = os.path.join(
        CHART_DIR, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return chart_path, f"data:image/png;base64,{b64}"


# ---------------- Google Sheets (optional CryptoHoldings) ----------------
def auth_gspread(service_account_file: str):
    if not gspread or not Credentials:
        return None
    creds = Credentials.from_service_account_file(
        service_account_file,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def open_ws(gc, sheet_url: str, tab: str):
    sh = gc.open_by_url(sheet_url)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=2000, cols=26)


def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    return pd.DataFrame(rows, columns=[h.strip() for h in header])


def norm_crypto_symbol(value) -> str:
    """Normalize Fidelity / sheet crypto symbols to yfinance format."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    mapping = {
        "BTC": "BTC-USD",
        "BTC/USD": "BTC-USD",
        "BITCOIN": "BTC-USD",
        "ETH": "ETH-USD",
        "ETH/USD": "ETH-USD",
        "ETHEREUM": "ETH-USD",
        "SOL": "SOL-USD",
        "SOL/USD": "SOL-USD",
        "SOLANA": "SOL-USD",
        "LTC": "LTC-USD",
        "LTC/USD": "LTC-USD",
        "LITECOIN": "LTC-USD",
        "USD***": "USD***",
        "USD": "USD",
    }
    if s in mapping:
        return mapping[s]
    if s.endswith("/USD"):
        return s.replace("/USD", "-USD")
    return s


def to_float(value, default=0.0):
    """Parse strings like '$1,234.56', '(12.3)', blanks, and numerics."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "--"}:
        return default
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return default


def first_present(row, names, default=""):
    for name in names:
        if name in row and str(row.get(name, "")).strip() != "":
            return row.get(name)
    return default


def build_crypto_holdings_map(dfh: pd.DataFrame) -> dict:
    """
    Build a ticker -> holding dict from the CryptoHoldings tab.

    Supports both the normalized importer columns:
      NormalizedSymbol, QuantityNum, CurrentValueNum, CostBasisTotalNum

    and raw Fidelity export columns:
      Symbol, Quantity, Current Value, Cost Basis Total
    """
    holdings = {}
    if dfh is None or dfh.empty:
        return holdings

    for _, row in dfh.iterrows():
        raw_symbol = first_present(row, ["NormalizedSymbol", "Ticker", "Symbol", "ticker", "symbol"])
        ticker = norm_crypto_symbol(raw_symbol)
        if not ticker or ticker in {"USD", "USD***", "CASH"}:
            continue

        qty = to_float(first_present(row, ["QuantityNum", "Quantity", "Qty", "quantity", "qty"]), 0.0)
        value = to_float(first_present(row, ["CurrentValueNum", "Current Value", "CurrentValue", "Market Value", "Value"]), 0.0)
        cost_basis = to_float(first_present(row, ["CostBasisTotalNum", "Cost Basis Total", "CostBasis", "Cost Basis"]), 0.0)
        avg_cost = to_float(first_present(row, ["AverageCostBasisNum", "Average Cost Basis", "AvgCost", "Average Cost"]), 0.0)
        last_price = to_float(first_present(row, ["LastPriceNum", "Last Price", "Price"]), 0.0)
        account_name = str(first_present(row, ["Account Name", "AccountName", "Account", "WalletOrExchange"], "")).strip()
        account_number = str(first_present(row, ["Account Number", "AccountNumber"], "")).strip()

        if ticker not in holdings:
            holdings[ticker] = {
                "ticker": ticker,
                "owned_qty": 0.0,
                "current_value": 0.0,
                "cost_basis": 0.0,
                "avg_cost": avg_cost,
                "last_price": last_price,
                "accounts": set(),
                "account_numbers": set(),
            }

        h = holdings[ticker]
        h["owned_qty"] += qty
        h["current_value"] += value
        h["cost_basis"] += cost_basis
        if avg_cost:
            h["avg_cost"] = avg_cost
        if last_price:
            h["last_price"] = last_price
        if account_name:
            h["accounts"].add(account_name)
        if account_number:
            h["account_numbers"].add(account_number)

    for h in holdings.values():
        h["owned"] = (abs(h.get("owned_qty", 0.0)) > 0) or (abs(h.get("current_value", 0.0)) > 0)
        h["accounts"] = ", ".join(sorted(h["accounts"]))
        h["account_numbers"] = ", ".join(sorted(h["account_numbers"]))
    return holdings


def ownership_action(signal_kind: str, owned: bool) -> str:
    """Convert signal + ownership into an actionable report phrase."""
    if signal_kind == "SELLTRIG":
        return "SELL risk / review exit" if owned else "SELL signal but not owned"
    if signal_kind == "BUY":
        return "BUY / add-to-position candidate" if owned else "BUY candidate"
    if signal_kind == "NEAR":
        return "WATCH owned position" if owned else "WATCH candidate"
    if owned:
        return "HOLD / monitor"
    return ""


# ---------------- Core scan ----------------
def run(config_path="./config.yaml", *, only=None, dry_run=False, force_email=False):
    log(f"Crypto watcher starting with config: {config_path}", level="step")
    cfg, sheet_url, service_account_file, universe = load_config(config_path)

    weekly_df, weekly_csv_path = load_weekly()
    log(f"Weekly CSV: {weekly_csv_path}", level="debug")

    # Normalize weekly
    w = weekly_df.rename(columns=str.lower)
    for c in ["ticker", "stage", "ma30", "rs_above_ma", "asset_class", "rank"]:
        if c not in w.columns:
            w[c] = np.nan

    # Crypto focus (Stage 1/2)
    focus = w[w["asset_class"].fillna("").str.lower().str.contains("crypto")].copy()
    focus = focus[
        focus["stage"].isin(["Stage 1 (Basing)", "Stage 2 (Uptrend)"])
    ]
    if focus.empty:
        # Fallback: use configured universe with blank stage, fetch ma via daily
        focus = pd.DataFrame(
            {
                "ticker": universe,
                "stage": "Stage 1 (Basing)",
                "ma30": np.nan,
                "rs_above_ma": True,
                "rank": 999999,
            }
        )
    else:
        focus = focus[["ticker", "stage", "ma30", "rs_above_ma", "rank"]].copy()
        focus["rank"] = (
            pd.to_numeric(focus["rank"], errors="coerce")
            .fillna(999999)
            .astype(int)
        )

    # Explicitly drop ARB-USD from the crypto scan universe
    focus = focus[focus["ticker"] != "ARB-USD"].copy()

    if only:
        filt = set([t.strip().upper() for t in only])
        focus = focus[focus["ticker"].isin(filt)].copy()

    # Optional CryptoHoldings snapshot and ownership map.
    # This makes signals actionable: BUY candidate vs add-to-position,
    # SELL risk only when the asset is actually owned, etc.
    #
    # IMPORTANT: load this before downloading price data so owned assets are
    # always included in the scan snapshot, even if they are not in the weekly
    # crypto focus list or if the asset is the benchmark itself (BTC-USD).
    crypto_holdings_df = pd.DataFrame()
    crypto_holdings_by_symbol = {}
    holdings_html = ""
    try:
        if sheet_url and service_account_file and gspread:
            gc = auth_gspread(service_account_file)
            if gc:
                ws = open_ws(gc, sheet_url, TAB_CRYPTO_HOLD)
                crypto_holdings_df = read_tab(ws)
                if not crypto_holdings_df.empty:
                    crypto_holdings_by_symbol = build_crypto_holdings_map(crypto_holdings_df)
                    keep_cols = [c for c in crypto_holdings_df.columns if c.strip()]
                    holdings_html = (
                        "<h4>Crypto Holdings (from Google Sheet)</h4>"
                        + crypto_holdings_df[keep_cols].to_html(index=False)
                    )
                    log(
                        f"Loaded CryptoHoldings: rows={len(crypto_holdings_df)} owned_symbols={len(crypto_holdings_by_symbol)}",
                        level="ok",
                    )
    except Exception as e:
        log(
            f"Sheet load failure for crypto holdings: CryptoHoldings ({e})",
            level="warn",
        )

    # Always include owned crypto assets in the scan/snapshot.
    owned_symbols = sorted(
        t for t, h in crypto_holdings_by_symbol.items()
        if h.get("owned") and t not in {"USD", "USD***", "CASH"}
    )
    if owned_symbols:
        existing = set(focus["ticker"].astype(str).str.upper()) if not focus.empty else set()
        missing_owned = [t for t in owned_symbols if t not in existing]
        if missing_owned:
            focus = pd.concat(
                [
                    focus,
                    pd.DataFrame(
                        {
                            "ticker": missing_owned,
                            "stage": "Stage 1 (Basing)",
                            "ma30": np.nan,
                            "rs_above_ma": True,
                            "rank": 999999,
                        }
                    ),
                ],
                ignore_index=True,
            )
            log(
                f"Added owned crypto to snapshot universe: {', '.join(missing_owned)}",
                level="debug",
            )

    # Symbols needed (focus + owned symbols + benchmark)
    symbols = sorted(set(focus["ticker"].tolist() + owned_symbols + [BENCHMARK_DEFAULT]))

    log("Downloading intraday + daily bars...", level="step")
    intraday, daily = get_intraday(symbols)
    log("Price data downloaded.", level="ok")

    # Determine which tickers actually have Close data in intraday
    if isinstance(intraday.columns, pd.MultiIndex):
        available_tickers = set(intraday["Close"].columns)
    else:
        # Single symbol case: use the symbol list directly
        available_tickers = set(symbols)

    # Filter focus universe to tickers with data
    focus = focus[focus["ticker"].isin(available_tickers)].copy()

    # Price accessor — return NaN if ticker missing
    if isinstance(intraday.columns, pd.MultiIndex):
        last_closes = intraday["Close"].ffill().iloc[-1]
    else:
        last_closes = intraday["Close"].ffill().tail(1)

    def px_now(t):
        if hasattr(last_closes, "index") and (t in last_closes.index):
            return float(last_closes.get(t, np.nan))
        # If we don't have this ticker in intraday Close, treat as missing price
        return np.nan

    # State
    trig = _load_state()

    # Crypto now consumes the shared Weinstein LONG CORE for BUY/NEAR
    # signal classification.  The crypto watcher keeps its existing
    # 24/7 data handling, state machine, SELL logic, reports, and email
    # behavior; only the long-side BUY/NEAR eligibility calculation is
    # delegated to the shared core.
    crypto_long_params = LongEntryParams(
        min_break_pct=MIN_BREAKOUT_PCT,
        dist_above_ma_min=BUY_DIST_ABOVE_MA_MIN,
        vol_min=VOL_PACE_MIN,
        adx_min=0.0,
    )

    buy_signals, near_signals, sell_triggers = [], [], []
    info_rows = []

    log("Evaluating candidates...", level="step")

    for _, row in focus.iterrows():
        t = row["ticker"]
        # Normally the benchmark is only used for RS comparison, but include it
        # in the snapshot when it is an owned crypto holding (for example BTC-USD).
        if t == BENCHMARK_DEFAULT and norm_crypto_symbol(t) not in crypto_holdings_by_symbol:
            continue

        price = px_now(t)
        if np.isnan(price):
            # No real price → skip this symbol
            continue

        holding = crypto_holdings_by_symbol.get(norm_crypto_symbol(t), {})
        owned = bool(holding.get("owned", False))
        owned_qty = holding.get("owned_qty", 0.0)
        current_value = holding.get("current_value", 0.0)
        cost_basis = holding.get("cost_basis", 0.0)
        avg_cost = holding.get("avg_cost", 0.0)
        holding_accounts = holding.get("accounts", "")

        stage = str(row.get("stage", ""))
        ma30 = row.get("ma30", np.nan)

        # compute quick SMA150 from daily if missing
        if pd.isna(ma30):
            try:
                if isinstance(daily.columns, pd.MultiIndex):
                    close = daily[("Close", t)].dropna()
                else:
                    close = daily["Close"].dropna()
                if len(close) >= SMA_DAYS:
                    ma30 = float(close.rolling(SMA_DAYS).mean().dropna().iloc[-1])
            except Exception:
                ma30 = np.nan

        rs_ok = bool(row.get("rs_above_ma", True))
        weekly_rank = (
            int(row.get("rank", 999999))
            if pd.notna(row.get("rank", np.nan))
            else 999999
        )

        pivot = last_weekly_pivot_high(t, daily, weeks=PIVOT_LOOKBACK_WEEKS)
        pace = volume_pace_today_vs_50dma_crypto(t, daily)

        # BUY / NEAR classification now comes from the shared LONG CORE.
        #
        # IMPORTANT: the legacy crypto watcher confirmed BUY using the most
        # recent intraday close, while NEAR used the current intraday price.
        # To preserve that behavior as closely as possible, we call CORE twice:
        #   - core_buy_result: last intraday close, used for confirm
        #   - core_near_result: current price, used for NEAR/ARMED state
        #
        # The legacy crypto watcher treated missing volume pace as allowed.
        # The long CORE requires a non-NaN volume input, so when pace is missing
        # we pass the required threshold to preserve the old permissive behavior.
        confirm = False
        vol_ok = True
        price_ok = False
        last_c = np.nan
        core_vol_mult = float(pace) if pd.notna(pace) else VOL_PACE_MIN

        core_buy_signal = "NONE"
        core_buy_reason = "not_evaluated"
        core_near_signal = "NONE"
        core_near_reason = "not_evaluated"

        closes_n = get_last_n_intraday_closes(intraday, t, n=2)
        if closes_n:
            last_c = closes_n[-1]

        if (
            stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)")
            and pd.notna(ma30)
            and pd.notna(pivot)
            and pd.notna(last_c)
        ):
            core_buy_result = evaluate_long_signal(
                price=float(last_c),
                ma_val=float(ma30),
                pivot=float(pivot),
                rs_above_ma=rs_ok,
                vol_mult=core_vol_mult,
                adx_val=np.nan,
                params=crypto_long_params,
                near_below_pivot_pct=NEAR_BELOW_PIVOT_PCT,
                near_vol_min=NEAR_VOL_PACE_MIN,
            )
            core_buy_signal = core_buy_result.signal
            core_buy_reason = core_buy_result.reason
            confirm = core_buy_result.signal == "BUY"
            price_ok = core_buy_result.signal == "BUY"

        # volume gate (24h vs 50dma)
        pace_full_gate = (pd.isna(pace) or pace >= VOL_PACE_MIN)
        near_pace_gate = (pd.isna(pace) or pace >= NEAR_VOL_PACE_MIN)

        # NEAR zone from shared LONG CORE, evaluated on current intraday price.
        near_now = False
        if (
            stage in ("Stage 1 (Basing)", "Stage 2 (Uptrend)")
            and pd.notna(ma30)
            and pd.notna(pivot)
            and pd.notna(price)
        ):
            core_near_result = evaluate_long_signal(
                price=float(price),
                ma_val=float(ma30),
                pivot=float(pivot),
                rs_above_ma=rs_ok,
                vol_mult=core_vol_mult,
                adx_val=np.nan,
                params=crypto_long_params,
                near_below_pivot_pct=NEAR_BELOW_PIVOT_PCT,
                near_vol_min=NEAR_VOL_PACE_MIN,
            )
            core_near_signal = core_near_result.signal
            core_near_reason = core_near_result.reason

            # Preserve legacy behavior: if current price has broken out but
            # the last-close BUY confirmation has not yet triggered, keep it
            # in NEAR/ARMED territory.
            near_now = (
                core_near_result.signal == "NEAR"
                or (core_near_result.signal == "BUY" and not confirm)
            )

        # SELL near/confirm: crack below SMA150 by SELL_BREAK_PCT (no elapsed checks)
        sell_near_now = False
        sell_confirm = False
        if pd.notna(ma30) and pd.notna(price):
            sell_near_now = (
                price >= ma30 * (1.0 - SELL_BREAK_PCT)
                and price <= ma30 * (1.0 + SELL_BREAK_PCT)
            )
            closes2 = get_last_n_intraday_closes(intraday, t, n=2)
            if closes2:
                sell_confirm = all(
                    (c <= ma30 * (1.0 - SELL_BREAK_PCT)) for c in closes2[-1:]
                )

        # promotion state
        st = trig.get(
            t,
            {
                "state": "IDLE",
                "near_hits": [],
                "cooldown": 0,
                "sell_state": "IDLE",
                "sell_hits": [],
                "sell_cooldown": 0,
            },
        )

        st["near_hits"], near_count = _update_hits(
            st.get("near_hits", []), near_now, NEAR_HITS_WINDOW
        )
        if st.get("cooldown", 0) > 0:
            st["cooldown"] = int(st["cooldown"]) - 1

        state_now = st.get("state", "IDLE")
        if state_now == "IDLE" and near_now:
            state_now = "NEAR"
        elif state_now in ("IDLE", "NEAR") and near_count >= NEAR_HITS_MIN:
            state_now = "ARMED"
        elif state_now == "ARMED" and confirm and vol_ok and pace_full_gate:
            state_now = "TRIGGERED"
            st["cooldown"] = COOLDOWN_SCANS
        elif state_now == "TRIGGERED":
            # let a triggered state exist for one scan; we flip to COOLDOWN after emit
            pass
        elif st["cooldown"] > 0 and not near_now:
            state_now = "COOLDOWN"
        elif st["cooldown"] == 0 and not near_now and not confirm:
            state_now = "IDLE"
        st["state"] = state_now

        # SELL state
        st["sell_hits"], sell_hit_count = _update_hits(
            st.get("sell_hits", []), sell_near_now, NEAR_HITS_WINDOW
        )
        if st.get("sell_cooldown", 0) > 0:
            st["sell_cooldown"] = int(st["sell_cooldown"]) - 1

        sell_state = st.get("sell_state", "IDLE")
        if sell_state == "IDLE" and sell_near_now:
            sell_state = "NEAR"
        elif sell_state in ("IDLE", "NEAR") and sell_hit_count >= NEAR_HITS_MIN:
            sell_state = "ARMED"
        elif sell_state == "ARMED" and sell_confirm:
            sell_state = "TRIGGERED"
            st["sell_cooldown"] = COOLDOWN_SCANS
        elif sell_state == "TRIGGERED":
            pass
        elif st["sell_cooldown"] > 0 and not sell_near_now:
            sell_state = "COOLDOWN"
        elif st["sell_cooldown"] == 0 and not sell_near_now and not sell_confirm:
            sell_state = "IDLE"
        st["sell_state"] = sell_state

        trig[t] = st

        # Emit by state
        if st["state"] == "TRIGGERED" and pace_full_gate:
            buy_signals.append(
                {
                    "ticker": t,
                    "price": price,
                    "pivot": pivot,
                    "pace": None if pd.isna(pace) else float(pace),
                    "stage": stage,
                    "ma30": ma30,
                    "weekly_rank": weekly_rank,
                    "owned": owned,
                    "owned_qty": owned_qty,
                    "current_value": current_value,
                    "cost_basis": cost_basis,
                    "action": ownership_action("BUY", owned),
                }
            )
            trig[t]["state"] = "COOLDOWN"
        elif st["state"] in ("NEAR", "ARMED"):
            if near_pace_gate:
                near_signals.append(
                    {
                        "ticker": t,
                        "price": price,
                        "pivot": pivot,
                        "pace": None if pd.isna(pace) else float(pace),
                        "stage": stage,
                        "ma30": ma30,
                        "weekly_rank": weekly_rank,
                        "reason": "near/armed",
                        "owned": owned,
                        "owned_qty": owned_qty,
                        "current_value": current_value,
                        "cost_basis": cost_basis,
                        "action": ownership_action("NEAR", owned),
                    }
                )

        if st["sell_state"] == "TRIGGERED":
            sell_triggers.append(
                {
                    "ticker": t,
                    "price": price,
                    "ma30": ma30,
                    "stage": stage,
                    "weekly_rank": weekly_rank,
                    "pace": None if pd.isna(pace) else float(pace),
                    "owned": owned,
                    "owned_qty": owned_qty,
                    "current_value": current_value,
                    "cost_basis": cost_basis,
                    "action": ownership_action("SELLTRIG", owned),
                }
            )
            trig[t]["sell_state"] = "COOLDOWN"

        info_rows.append(
            {
                "ticker": t,
                "stage": stage,
                "price": price,
                "ma30": ma30,
                "pivot10w": pivot,
                "vol_pace_vs50dma": None
                if pd.isna(pace)
                else round(float(pace), 2),
                "two_bar_confirm": confirm,
                "last_bar_vol_ok": True,  # kept for continuity; not used to gate
                "core_buy_signal": core_buy_signal,
                "core_buy_reason": core_buy_reason,
                "core_near_signal": core_near_signal,
                "core_near_reason": core_near_reason,
                "owned": owned,
                "owned_qty": owned_qty,
                "current_value": current_value,
                "cost_basis": cost_basis,
                "avg_cost": avg_cost,
                "holding_accounts": holding_accounts,
                "ownership_action": (
                    ownership_action("SELLTRIG", owned)
                    if st["sell_state"] in ("TRIGGERED", "ARMED", "NEAR") and owned
                    else ownership_action("BUY", owned)
                    if confirm
                    else ownership_action("NEAR", owned)
                    if near_now
                    else ownership_action("", owned)
                ),
                "buy_state": st["state"],
                "sell_state": st["sell_state"],
            }
        )

    log(
        f"Scan done. Raw counts → BUY:{len(buy_signals)} NEAR:{len(near_signals)} SELLTRIG:{len(sell_triggers)}",
        level="info",
    )

    # Charts (prefer BUY then NEAR)
    def _stage_rank(s):
        return 0 if str(s).startswith("Stage 2") else (
            1 if str(s).startswith("Stage 1") else 9
        )

    buy_signals.sort(
        key=lambda it: (
            int(it.get("weekly_rank", 999999)),
            _stage_rank(it.get("stage", "")),
            -float(it.get("pace") or -1),
        )
    )
    near_signals.sort(
        key=lambda it: (
            int(it.get("weekly_rank", 999999)),
            _stage_rank(it.get("stage", "")),
            abs((it.get("price") or 0) - (it.get("pivot") or 0)),
        )
    )
    sell_triggers.sort(
        key=lambda it: (
            int(it.get("weekly_rank", 999999)),
            _stage_rank(it.get("stage", "")),
            -float(it.get("pace") or -1),
        )
    )

    charts = []
    charts_added = 0
    for bucket in (buy_signals, near_signals):
        for it in bucket:
            if charts_added >= MAX_CHARTS_PER_EMAIL:
                break
            t = it["ticker"]
            p, data_uri = make_tiny_chart_png(t, BENCHMARK_DEFAULT, daily)
            if data_uri:
                charts.append((t, data_uri))
                charts_added += 1

    log(f"Charts prepared: {len(charts)}", level="debug")

    # CryptoHoldings were loaded before evaluation so ownership can be attached
    # directly to snapshot rows and signal/action text.

    # HTML/text
    def bullets(items, kind):
        if not items:
            return f"<p>No {kind} signals.</p>"
        lis = []
        for i, it in enumerate(items, 1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None)
                pace_str = (
                    "—"
                    if (pace_val is None or pd.isna(pace_val))
                    else f"{pace_val:.2f}x"
                )
                action = it.get("action") or ownership_action("SELLTRIG", bool(it.get("owned", False)))
                owned_txt = "owned" if it.get("owned") else "not owned"
                qty_txt = f", qty {float(it.get('owned_qty') or 0):.6g}" if it.get("owned") else ""
                val_txt = f", value ${float(it.get('current_value') or 0):,.2f}" if it.get("owned") else ""
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"— <b>{action}</b> ({owned_txt}{qty_txt}{val_txt}; "
                    f"↓ SMA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})</li>"
                )
            else:
                pivot_val = it.get("pivot", np.nan)
                pivot_str = f"{pivot_val:.2f}" if pd.notna(pivot_val) else "—"
                pace_val = it.get("pace", None)
                pace_str = (
                    "—"
                    if (pace_val is None or pd.isna(pace_val))
                    else f"{pace_val:.2f}x"
                )
                action = it.get("action") or ownership_action(kind, bool(it.get("owned", False)))
                owned_txt = "owned" if it.get("owned") else "not owned"
                qty_txt = f", qty {float(it.get('owned_qty') or 0):.6g}" if it.get("owned") else ""
                val_txt = f", value ${float(it.get('current_value') or 0):,.2f}" if it.get("owned") else ""
                lis.append(
                    f"<li><b>{i}.</b> <b>{it['ticker']}</b> @ {it['price']:.2f} "
                    f"— <b>{action}</b> ({owned_txt}{qty_txt}{val_txt}; "
                    f"pivot {pivot_str}, pace {pace_str}, {it['stage']}, weekly {wr_str})</li>"
                )
        return "<ol>" + "\n".join(lis) + "</ol>"

    charts_html = ""
    if charts:
        charts_html = "<h4>Charts (Price + SMA150, RS vs BTC)</h4>"
        for t, data_uri in charts:
            charts_html += f"""
            <div style="display:inline-block;margin:6px 8px 10px 0;vertical-align:top;text-align:center;">
              <img src="{data_uri}" alt="{t}" style="border:1px solid #eee;border-radius:6px;max-width:320px;">
              <div style="font-size:12px;color:#555;margin-top:3px;">{t}</div>
            </div>
            """

    snapshot_df = pd.DataFrame(info_rows)
    if not snapshot_df.empty:
        cols_order = [
            "ticker",
            "stage",
            "price",
            "ma30",
            "pivot10w",
            "vol_pace_vs50dma",
            "two_bar_confirm",
            "last_bar_vol_ok",
            "core_buy_signal",
            "core_buy_reason",
            "core_near_signal",
            "core_near_reason",
            "owned",
            "owned_qty",
            "current_value",
            "cost_basis",
            "avg_cost",
            "holding_accounts",
            "ownership_action",
            "buy_state",
            "sell_state",
        ]
        for c in cols_order:
            if c not in snapshot_df.columns:
                snapshot_df[c] = ""

        snapshot_df["__stage_rank"] = snapshot_df["stage"].apply(stage_order)
        snapshot_df = snapshot_df.sort_values(
            ["__stage_rank", "ticker"]
        ).drop(columns="__stage_rank")
        snapshot_html = (
            "<h4>Snapshot (ordered by stage)</h4>"
            + snapshot_df[cols_order].to_html(index=False)
        )
    else:
        snapshot_html = ""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    header_html = f"""
    <h3>Weinstein Crypto Watch — {now}</h3>
    <p><i>
      BUY: Stage 1/2 + confirm over ~10-week pivot & SMA150, +{MIN_BREAKOUT_PCT*100:.1f}% headroom,
      RS vs BTC support, and volume pace ≥ {VOL_PACE_MIN:.1f}× (24h pacing).<br>
      NEAR-TRIGGER: Stage 1/2 + RS ok, price within {NEAR_BELOW_PIVOT_PCT*100:.1f}% below pivot or first close over pivot but not fully confirmed yet, pace ≥ {NEAR_VOL_PACE_MIN:.1f}×.<br>
      SELL-TRIGGER: Confirmed crack below SMA150 by {SELL_BREAK_PCT*100:.1f}% with persistence.
    </i></p>
    """

    html = (
        header_html
        + "<h4>Buy Triggers (ranked)</h4>"
        + bullets(buy_signals, "BUY")
        + "<h4>Near-Triggers (ranked)</h4>"
        + bullets(near_signals, "NEAR")
        + "<h4>Sell Triggers (ranked)</h4>"
        + bullets(sell_triggers, "SELLTRIG")
        + charts_html
        + snapshot_html
        + (("<hr/>" + holdings_html) if holdings_html else "")
    )

    # Text version (compact)
    def _lines(items, kind):
        if not items:
            return f"No {kind} signals."
        out = []
        for i, it in enumerate(items, 1):
            wr = it.get("weekly_rank", None)
            wr_str = f"#{int(wr)}" if (wr is not None and pd.notna(wr)) else "—"
            if kind == "SELLTRIG":
                ma = it.get("ma30", np.nan)
                ma_str = f"{ma:.2f}" if pd.notna(ma) else "—"
                pace_val = it.get("pace", None)
                pace_str = (
                    "—"
                    if (pace_val is None or pd.isna(pace_val))
                    else f"{pace_val:.2f}x"
                )
                action = it.get("action") or ownership_action("SELLTRIG", bool(it.get("owned", False)))
                owned_txt = "owned" if it.get("owned") else "not owned"
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} — {action} "
                    f"({owned_txt}; below SMA150 {ma_str}, pace {pace_str}, {it.get('stage','')}, weekly {wr_str})"
                )
            else:
                pivot_val = it.get("pivot", np.nan)
                pivot_str = f"{pivot_val:.2f}" if pd.notna(pivot_val) else "—"
                pace_val = it.get("pace", None)
                pace_str = (
                    "—"
                    if (pace_val is None or pd.isna(pace_val))
                    else f"{pace_val:.2f}x"
                )
                action = it.get("action") or ownership_action(kind, bool(it.get("owned", False)))
                owned_txt = "owned" if it.get("owned") else "not owned"
                out.append(
                    f"{i}. {it['ticker']} @ {it['price']:.2f} — {action} "
                    f"({owned_txt}; pivot {pivot_str}, pace {pace_str}, {it['stage']}, weekly {wr_str})"
                )
        return "\n".join(out)

    text = (
        f"Weinstein Crypto Watch — {now}\n\n"
        f"BUY (ranked):\n{_lines(buy_signals, 'BUY')}\n\n"
        f"NEAR-TRIGGER (ranked):\n{_lines(near_signals, 'NEAR')}\n\n"
        f"SELL TRIGGERS (ranked):\n{_lines(sell_triggers, 'SELLTRIG')}\n"
    )

    # persist state & write html
    _save_state(trig)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html_path = os.path.join(
        OUTPUT_DIR, f"crypto_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"Saved HTML → {html_path}", level="ok")
    except Exception as e:
        log(f"Cannot save HTML: {e}", level="warn")

    # email
    if dry_run:
        log("DRY-RUN set — skipping email send.", level="warn")
    else:
        # Default behavior: only send email when there is at least one BUY or SELL trigger.
        # With --force-email, send the full status report even when there are no triggers.
        triggers_present = (len(buy_signals) > 0) or (len(sell_triggers) > 0)
        if not triggers_present and not force_email:
            log("No BUY/SELL triggers present — skipping email send.", level="info")
        else:
            counts = (
                f"{len(buy_signals)} BUY / {len(near_signals)} NEAR / "
                f"{len(sell_triggers)} SELL-TRIG"
            )
            if force_email and not triggers_present:
                counts = f"STATUS / {len(near_signals)} NEAR / no BUY or SELL triggers"
            log("Sending email...", level="step")
            send_email(
                subject=f"Crypto Watch — {counts}",
                html_body=html,
                text_body=text,
                cfg_path=config_path,
            )
            log("Email sent.", level="ok")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="comma list of tickers to restrict evaluation (e.g. BTC-USD,ETH-USD)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force-email",
        action="store_true",
        help="Send the crypto report even when there are no BUY/SELL triggers.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    VERBOSE = not args.quiet
    only = (
        [s.strip().upper() for s in args.only.split(",") if s.strip()]
        if args.only
        else None
    )

    try:
        run(config_path=args.config, only=only, dry_run=args.dry_run, force_email=args.force_email)
        log("Crypto tick complete.", level="ok")
    except Exception as e:
        log(f"Error: {e}", level="err")
        raise
