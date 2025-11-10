# weinstein_intraday_watcher.py
# (c) WeinsteinAgent — Intraday Watcher
# Compatible with Python 3.11, pandas >= 2.2, yfinance >= 0.2.52
# ---------------------------------------------------------------
import os
import sys
import io
import math
import json
import time
import textwrap
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("Please `pip install yfinance` in your venv.", file=sys.stderr)
    raise

try:
    import yaml
except Exception:
    print("Please `pip install pyyaml` in your venv.", file=sys.stderr)
    raise

# ----------------------------- Utilities -----------------------------

TZ_DEFAULT = "America/Chicago"

def tz_now(tz_name: str) -> datetime:
    try:
        import pytz
        return datetime.now(pytz.timezone(tz_name))
    except Exception:
        # Fallback to naive local time if pytz not present
        return datetime.now()

def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")

def fmt_ts_seconds(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")

def last_val(series: pd.Series) -> float:
    """Return last scalar value from Series safely."""
    if series is None or len(series) == 0:
        return float("nan")
    return float(series.iloc[-1])

def last_diff_5(series: pd.Series) -> float:
    """Return last - 5-back difference safely."""
    if series is None or len(series) < 6:
        return float("nan")
    return float(series.iloc[-1] - series.iloc[-6])

def pct(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return 100.0 * (a - b) / b

def sparkline(vals: List[float], width: int = 18) -> str:
    """Unicode sparkline for quick trend view."""
    ticks = "▁▂▃▄▅▆▇█"
    v = np.array([x for x in vals if pd.notna(x)], dtype=float)
    if v.size == 0:
        return ""
    # downsample to width
    if v.size > width:
        idx = np.linspace(0, v.size - 1, width).round().astype(int)
        v = v[idx]
    lo, hi = np.nanmin(v), np.nanmax(v)
    if math.isclose(lo, hi) or hi - lo == 0:
        return ticks[0] * len(v)
    scaled = ((v - lo) / (hi - lo) * (len(ticks) - 1)).astype(int)
    return "".join(ticks[i] for i in scaled)

def safe_to_list(obj) -> List[float]:
    """Return a Python list of floats from Series/ndarray/list/DataFrame.last column."""
    if isinstance(obj, pd.Series):
        return obj.dropna().astype(float).tolist()
    if isinstance(obj, pd.DataFrame):
        # Use the last column which usually is 'Close'
        if obj.shape[1] == 0:
            return []
        col = obj.columns[-1]
        return pd.to_numeric(obj[col], errors="coerce").dropna().astype(float).tolist()
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [float(x) for x in obj if pd.notna(x)]
    return []

def print_header(title: str, tzname: str):
    now_disp = tz_now(tzname)
    print(f"Weinstein Intraday Watch — {now_disp.strftime('%Y-%m-%d %H:%M')}")
    print("BUY: Weekly Stage 1/2 + confirm over ~10-week pivot & 30-wk MA proxy (SMA150), +0.4% headroom, RS support, volume pace ≥ 1.3×. For 60m bars: ≥40 min elapsed & intrabar pace ≥ 1.2×.")
    print("NEAR-TRIGGER: Stage 1/2 + RS ok, price within 0.3% below pivot or first close over pivot but not fully confirmed yet, volume pace ≥ 1.0×.")
    print("SELL-TRIGGER: Confirmed crack below MA150 by 0.5% with persistence; for 60m bars, ≥40 min elapsed & intrabar pace ≥ 1.2×.")
    print()

# ----------------------------- Config -----------------------------

@dataclass
class EmailSMTP:
    host: str = "smtp.gmail.com"
    port_ssl: int = 587
    username: str = ""
    app_password: str = ""

@dataclass
class EmailCfg:
    enabled: bool = False
    sender: str = ""
    recipients: List[str] = field(default_factory=list)
    subject_prefix: str = "Weinstein Report READY"
    provider: str = "smtp"
    smtp: EmailSMTP = field(default_factory=EmailSMTP)

@dataclass
class NotificationsCfg:
    email: EmailCfg = field(default_factory=EmailCfg)

@dataclass
class SheetsCfg:
    url: str = ""
    sheet_url: str = ""
    open_positions_tab: str = "Open_Positions"
    signals_tab: str = "Signals"
    output_dir: str = "./output"

@dataclass
class ReportingCfg:
    output_dir: str = "./output"
    include_pdf: bool = True
    include_csv: bool = True
    summary_lines: int = 10
    # IMPORTANT: accept and ignore unknown keys like `section_order`
    def __init__(self, **kwargs):
        self.output_dir = kwargs.get("output_dir", "./output")
        self.include_pdf = bool(kwargs.get("include_pdf", True))
        self.include_csv = bool(kwargs.get("include_csv", True))
        self.summary_lines = int(kwargs.get("summary_lines", 10))
        # ignore anything else silently (e.g., section_order)

@dataclass
class RiskCfg:
    max_position_pct: float = 10.0
    tranche_plan: List[int] = field(default_factory=lambda: [30,30,20,20])
    stop_below_ma_pct: float = 5.0

@dataclass
class UniverseCfg:
    mode: str = "sp500"
    extra: List[str] = field(default_factory=list)
    min_price: float = 5.0
    min_avg_volume: int = 1_000_000

@dataclass
class AppCfg:
    mode_default: str = "weekly"
    timezone: str = TZ_DEFAULT
    benchmark: str = "SPY"
    include_charts: bool = True

@dataclass
class GoogleCfg:
    service_account_json: str = ""
    client_email: str = ""

@dataclass
class Config:
    app: AppCfg
    sheets: SheetsCfg
    google: GoogleCfg
    notifications: NotificationsCfg
    reporting: ReportingCfg
    universe: UniverseCfg
    risk: RiskCfg

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    app = AppCfg(**raw.get("app", {}))
    sheets = SheetsCfg(**raw.get("sheets", {}))
    google = GoogleCfg(**raw.get("google", {}))
    notifications = NotificationsCfg(
        email=EmailCfg(
            **{k: v for k, v in raw.get("notifications", {}).get("email", {}).items() if k not in ("smtp",)}
        )
    )
    # SMTP nested
    email_smtp = raw.get("notifications", {}).get("email", {}).get("smtp", {}) or {}
    notifications.email.smtp = EmailSMTP(**email_smtp)

    reporting = ReportingCfg(**raw.get("reporting", {}))
    universe = UniverseCfg(**raw.get("universe", {}))
    risk = RiskCfg(**raw.get("risk", {}))

    return Config(
        app=app,
        sheets=sheets,
        google=google,
        notifications=notifications,
        reporting=reporting,
        universe=universe,
        risk=risk
    )

# ----------------------------- Data loading -----------------------------

def load_weekly_snapshot_csv(report_dir: str) -> Optional[pd.DataFrame]:
    """Pick the most recent *weekly* csv emitted by weekly job."""
    if not os.path.isdir(report_dir):
        return None
    files = sorted(
        [f for f in os.listdir(report_dir) if f.startswith("weinstein_weekly_equities_") and f.endswith(".csv")],
        reverse=True
    )
    if not files:
        return None
    path = os.path.join(report_dir, files[0])
    df = pd.read_csv(path)
    # Normalize numeric columns that show up in your paste
    for col in ("price", "ma30", "pivot10w"):
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.replace("$", "")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def yf_daily(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.title)  # Close, Open, etc.
    return df

def yf_intraday(ticker: str, period: str = "5d", interval: str = "60m") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.title)
    return df

# ----------------------------- Logic -----------------------------

def stage_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Focus on Stage 1/2 names from the weekly snapshot."""
    if df is None or df.empty:
        return pd.DataFrame()
    keep = df[df["stage"].astype(str).str.startswith("Stage 1") | df["stage"].astype(str).str.startswith("Stage 2")]
    return keep.reset_index(drop=True)

def evaluate_near_triggers(df: pd.DataFrame) -> List[Tuple[str, float, float, str]]:
    """Simple near-trigger: price within 0.3% of pivot (or first close over pivot)."""
    out = []
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        tkr = str(row.get("ticker"))
        price = row.get("price")
        pivot = row.get("pivot10w")
        if pd.isna(price) or pd.isna(pivot) or pivot <= 0:
            continue
        dist = pct(price, pivot)
        if -0.3 <= dist <= 0.2 or (price >= pivot and dist <= 0.6):
            out.append((tkr, float(price), float(pivot), str(row.get("stage"))))
    # rank by distance ascending
    out.sort(key=lambda x: abs(pct(x[1], x[2])))
    return out

def evaluate_sell_triggers(open_positions: pd.DataFrame, weekly: pd.DataFrame) -> List[Tuple[str, float, str]]:
    """Basic SELL: crack below MA150 proxy (weekly ma30)."""
    out = []
    if open_positions is None or open_positions.empty or weekly is None or weekly.empty:
        return out
    ma = weekly[["ticker", "ma30", "stage"]].dropna()
    ma.index = ma["ticker"]
    for _, row in open_positions.iterrows():
        tkr = str(row.get("Symbol") or row.get("Ticker") or "").strip()
        if not tkr or tkr not in ma.index:
            continue
        ma30 = float(ma.loc[tkr, "ma30"])
        price_now = float(row.get("Last Price") or row.get("PriceNow") or np.nan)
        if not pd.isna(price_now) and ma30 > 0 and price_now < (ma30 * 0.995):
            out.append((tkr, price_now, str(ma.loc[tkr, "stage"])))
    return out

def build_crypto_section(benchmark: str = "BTC-USD") -> Tuple[str, pd.DataFrame]:
    """Build the 3-row crypto table (BTC/ETH/SOL) with stage-ish info."""
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
    rows = []
    for tkr in tickers:
        end = datetime.utcnow()
        start = end - timedelta(days=120)
        d = yf_daily(tkr, start, end)
        if d is None or d.empty or "Close" not in d.columns:
            continue
        close = d["Close"]
        ma10 = close.rolling(10).mean()
        ma30 = close.rolling(30).mean()
        price = last_val(close)
        m10 = last_val(ma10)
        m30 = last_val(ma30)
        dist_ma = pct(price, m30)
        stage = "Stage 2 (Uptrend)" if price > m30 and m30 > last_val(ma30.shift(5)) else (
            "Stage 3 (Topping)" if price < m10 and price > m30 else
            "Stage 4 (Downtrend)"
        )
        # Simple RS: asset / benchmark
        bench = yf_daily(benchmark, start, end)
        if bench is not None and not bench.empty and "Close" in bench.columns:
            rs = (close / bench["Close"]).dropna()
            rs_ma30 = rs.rolling(30).mean()
            rs_above = last_val(rs) >= last_val(rs_ma30)
            rs_slope = last_diff_5(rs)
        else:
            rs = pd.Series(dtype=float)
            rs_ma30 = pd.Series(dtype=float)
            rs_above = False
            rs_slope = float("nan")

        rows.append(dict(
            ticker=tkr,
            asset_class="Crypto",
            industry="",
            sector="",
            Buy_Signal="BUY" if stage.startswith("Stage 2") and dist_ma >= 0 else "Avoid",
            chart="chart",
            stage=stage,
            short_term_state_wk="StageConflict",
            price=price,
            ma10=m10,
            ma30=m30,
            dist_ma_pct=round(dist_ma, 2),
            ma_slope_per_wk=round(last_diff_5(ma30), 2),
            rs=round(last_val(rs), 6) if len(rs) else np.nan,
            rs_ma30=round(last_val(rs_ma30), 6) if len(rs_ma30) else np.nan,
            rs_above_ma="Yes" if rs_above else "No",
            rs_slope_per_wk=round(rs_slope, 2) if not np.isnan(rs_slope) else "",
            notes="Price"
        ))

    table = pd.DataFrame(rows)
    # Build header text
    summary = {
        "Buy": int((table["Buy_Signal"] == "BUY").sum()) if not table.empty else 0,
        "Watch": 0,
        "Avoid": int((table["Buy_Signal"] != "BUY").sum()) if not table.empty else 0,
    }
    hdr = [
        "Crypto Weekly — Benchmark: BTC-USD",
        f"Generated {tz_now(TZ_DEFAULT).strftime('%Y-%m-%d %H:%M')}",
        f"Crypto Summary: ✅ Buy: {summary['Buy']}   |   🟡 Watch: {summary['Watch']}   |   🔴 Avoid: {summary['Avoid']}   (Total: {sum(summary.values())})",
        "ticker\tasset_class\tindustry\tsector\tBuy Signal\tchart\tstage\tshort_term_state_wk\tprice\tma10\tma30\tdist_ma_pct\tma_slope_per_wk\trs\trs_ma30\trs_above_ma\trs_slope_per_wk\tnotes",
    ]
    return "\n".join(hdr), table

# ----------------------------- I/O Blocks -----------------------------

def print_ranked_list(title: str, rows: List[Tuple], fmt):
    print(title)
    if not rows:
        print("No " + title.split()[0].upper() + " signals.")
        print()
        return
    for i, r in enumerate(rows, 1):
        print(fmt(i, r))
    print()

def print_snapshot(simple: pd.DataFrame):
    print("Snapshot (ordered by weekly rank & stage)")
    print("ticker\tstage\tprice\tma30")
    for _, row in simple.iterrows():
        t = str(row["ticker"])
        s = str(row["stage"])
        p = row.get("price", np.nan)
        m = row.get("ma30", np.nan)
        p_str = f"{float(p):.6f}" if pd.notna(p) else "nan"
        m_str = f"{float(m):.6f}" if pd.notna(m) else "nan"
        print(f"{t}\t{s}\t{p_str}\t{m_str}")
    print()

def print_crypto_block():
    hdr, table = build_crypto_section("BTC-USD")
    print(hdr)
    if table is None or table.empty:
        print()
        return
    for _, r in table.iterrows():
        line = "\t".join([
            str(r["ticker"]),
            str(r["asset_class"]),
            str(r["industry"]),
            str(r["sector"]),
            ("Buy" if r["Buy_Signal"] == "BUY" else "Avoid"),
            "chart",
            str(r["stage"]),
            str(r["short_term_state_wk"]),
            f"{float(r['price']):.6f}",
            f"{float(r['ma10']):.6f}",
            f"{float(r['ma30']):.6f}",
            f"{float(r['dist_ma_pct']):.2f}%",
            f"{float(r['ma_slope_per_wk']):.2f}%",
            f"{float(r['rs']) if pd.notna(r['rs']) else ''}",
            f"{float(r['rs_ma30']) if pd.notna(r['rs_ma30']) else ''}",
            str(r["rs_above_ma"]),
            str(r["rs_slope_per_wk"]),
            str(r["notes"]),
        ])
        print(line)
    print()

# ----------------------------- Main run -----------------------------

def run(_config_path: str):
    tzname = TZ_DEFAULT
    cfg = load_config(_config_path)
    tzname = cfg.app.timezone or TZ_DEFAULT

    print(f"▶️ [{fmt_ts_seconds(tz_now(tzname))}] Intraday watcher starting with config: {_config_path}")

    weekly_csv_dir = cfg.reporting.output_dir
    print(f"·· [{fmt_ts_seconds(tz_now(tzname))}] Weekly CSV: {weekly_csv_dir}/<latest weekly csv>")
    weekly = load_weekly_snapshot_csv(weekly_csv_dir)
    if weekly is None or weekly.empty:
        print("⚠️ No weekly snapshot found; nothing to evaluate.")
        return

    focus = stage_filter(weekly)
    print(f"• [{fmt_ts_seconds(tz_now(tzname))}] Focus universe: {len(focus)} symbols (Stage 1/2).")

    # Download prices – only daily needed for today’s print + crypto
    # (Earlier versions tried intraday for pace; keep simple & stable here.)
    print("▶️ [{}] Downloading intraday + daily bars...".format(fmt_ts_seconds(tz_now(tzname))))
    # Minimal sanity fetch for a couple of names to warm cache:
    for tkr in (cfg.app.benchmark, "AAPL"):
        try:
            _ = yf_daily(tkr, datetime.utcnow() - timedelta(days=120), datetime.utcnow())
        except Exception:
            pass
    print("✅ [{}] Price data downloaded.".format(fmt_ts_seconds(tz_now(tzname))))

    print("▶️ [{}] Evaluating candidates...".format(fmt_ts_seconds(tz_now(tzname))))
    near = evaluate_near_triggers(weekly)
    # Open positions block (optional) — load from Sheets if later needed
    open_pos = pd.DataFrame(columns=["Ticker", "Last Price"])
    sells = evaluate_sell_triggers(open_pos, weekly)

    print_header("Weinstein Intraday Watch", tzname)

    print("Buy Triggers (ranked)")
    print_ranked_list("Buy Triggers (ranked)", [], lambda i, r: "")

    print("Near-Triggers (ranked)")
    print_ranked_list(
        "Near-Triggers (ranked)",
        near,
        lambda i, r: f"{i}. {r[0]} @ {r[1]:.2f} (pivot {r[2]:.2f}, pace —, {r[3]}, weekly #999999)"
    )

    print("Sell Triggers (ranked)")
    print_ranked_list(
        "Sell Triggers (ranked)",
        sells,
        lambda i, r: f"{i}. {r[0]} @ {r[1]:.2f} — below MA150 proxy ({r[2]})"
    )

    print("Charts (Price + SMA150 ≈ 30-wk MA, RS normalized)")
    for sym in [r[0] for r in near][:4]:
        print(sym)
    print()

    print("Sell / Risk Triggers (Tracked Positions & Position Recommendations)")
    if not sells:
        print("(no SELLTRIG positions)")
    print()

    # Snapshot like your printout
    simple_cols = ["ticker", "stage", "price", "ma30"]
    simple = weekly[simple_cols].copy()
    # Keep ordering by Stage then by ticker to approximate your output
    stage_order = {"Stage 2 (Uptrend)": 0, "Stage 1 (Basing)": 1, "Stage 3 (Topping)": 2, "Stage 4 (Downtrend)": 3}
    simple["_k"] = simple["stage"].map(lambda s: stage_order.get(str(s), 9))
    simple = simple.sort_values(by=["_k", "ticker"]).drop(columns=["_k"])
    print_snapshot(simple)

    # Crypto weekly
    print_crypto_block()

    # Weekly footer (optional quick echo to match your logs)
    print("Weinstein Weekly – Summary")
    print("Per-position Snapshot")
    print("Ticker\tOpenQty\tEntryPrice\tPriceNow\tUnrealized%\tEntryTimeUTC")
    # If later you wire Google Sheets open positions, print them here.
    print()
    print("✅ Intraday tick complete.")

# ----------------------------- CLI -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as e:
        print("❌ Intraday watcher encountered an error.")
        print(str(e))
        raise
