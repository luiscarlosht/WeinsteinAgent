#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weinstein Intraday Watcher
- Prints intraday trigger scan
- Prints snapshot (from latest weekly CSV)
- Prints Crypto Weekly (from Signals sheet & yfinance), placed BEFORE the "Weinstein Weekly – Summary"
- Ends with the weekly portfolio summary snapshot (from Open_Positions sheet if present)

Notes:
- Requires: pyyaml, pandas, numpy, yfinance, gspread, oauth2client (for Google Sheets)
- Uses service account JSON from config to read the Google Sheet (Signals, Open_Positions)
"""

from __future__ import annotations

import os
import sys
import glob
import math
import time
import json
import textwrap
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# soft deps
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import yaml
except Exception as e:
    print("Missing dependency pyyaml. pip install pyyaml", file=sys.stderr)
    raise

# Google Sheets (optional: only needed if you actually read tabs)
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    gspread = None
    ServiceAccountCredentials = None


# ----------------------------
# Small utils
# ----------------------------

def _now_tz(tzname: str) -> dt.datetime:
    """Return timezone-aware 'now' in tzname."""
    # Simple local offset approach; relies on system tz database if available.
    # For Linux with tzdata, this is fine. If not, fallback to naive localtime.
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tzname)
        return dt.datetime.now(tz)
    except Exception:
        return dt.datetime.now()


def _fmt_pct(x: float, places: int = 2) -> str:
    try:
        return f"{x:.{places}f}%"
    except Exception:
        return ""


def _fmt_usd(x: float, places: int = 2) -> str:
    try:
        return f"${x:,.{places}f}"
    except Exception:
        return ""


def _sparkline(arr: List[float], width: int = 18) -> str:
    """Return a tiny unicode sparkline from values."""
    if not arr:
        return ""
    # downsample to width
    if len(arr) > width:
        idx = np.linspace(0, len(arr) - 1, width).astype(int)
        arr = [arr[i] for i in idx]
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(arr), max(arr)
    if hi - lo < 1e-12:
        return chars[0] * len(arr)
    out = []
    for v in arr:
        t = (v - lo) / (hi - lo)
        bucket = int(round(t * (len(chars) - 1)))
        bucket = max(0, min(bucket, len(chars) - 1))
        out.append(chars[bucket])
    return "".join(out)


def _last(series: pd.Series) -> Optional[float]:
    """Safe last value from a Series as float."""
    if series is None or series.empty:
        return None
    # Ensure single value
    return float(series.iloc[-1])


def _slope_last_5(series: pd.Series) -> Optional[float]:
    """Slope (last - value 5 ago)."""
    if series is None or len(series) < 6:
        return None
    return float(series.iloc[-1] - series.iloc[-6])


def _safe_bool(x: Any) -> bool:
    """Avoid pandas ambiguous truth-value traps."""
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        return not x.empty
    return bool(x)


# ----------------------------
# Config
# ----------------------------

@dataclass
class AppCfg:
    mode_default: str = "weekly"
    timezone: str = "America/Chicago"
    benchmark: str = "SPY"
    include_charts: bool = True


@dataclass
class SheetsCfg:
    url: str = ""
    sheet_url: str = ""
    open_positions_tab: str = "Open_Positions"
    signals_tab: str = "Signals"
    output_dir: str = "./output"


@dataclass
class GoogleCfg:
    service_account_json: str = ""
    client_email: str = ""


@dataclass
class SMTPAuthCfg:
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
    smtp: SMTPAuthCfg = field(default_factory=SMTPAuthCfg)


@dataclass
class NotificationsCfg:
    email: EmailCfg = field(default_factory=EmailCfg)


@dataclass
class ReportingCfg:
    output_dir: str = "./output"
    include_pdf: bool = True
    include_csv: bool = True
    summary_lines: int = 10
    # Accept optional section_order to avoid TypeError in newer configs
    section_order: Optional[List[str]] = None  # ignored here; included for compatibility


@dataclass
class UniverseCfg:
    mode: str = "sp500"
    extra: List[str] = field(default_factory=list)
    min_price: float = 5.0
    min_avg_volume: int = 1_000_000


@dataclass
class RiskCfg:
    max_position_pct: float = 10.0
    tranche_plan: List[int] = field(default_factory=lambda: [30, 30, 20, 20])
    stop_below_ma_pct: float = 5.0


@dataclass
class Cfg:
    app: AppCfg = field(default_factory=AppCfg)
    sheets: SheetsCfg = field(default_factory=SheetsCfg)
    google: GoogleCfg = field(default_factory=GoogleCfg)
    notifications: NotificationsCfg = field(default_factory=NotificationsCfg)
    reporting: ReportingCfg = field(default_factory=ReportingCfg)
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)


def load_config(path: str) -> Cfg:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    app = AppCfg(**raw.get("app", {}))
    sheets = SheetsCfg(**raw.get("sheets", {}))
    google = GoogleCfg(**raw.get("google", {}))

    notif_raw = raw.get("notifications", {})
    email_raw = notif_raw.get("email", {})
    smtp_raw = (email_raw or {}).get("smtp", {})
    email = EmailCfg(
        enabled=email_raw.get("enabled", False),
        sender=email_raw.get("sender", ""),
        recipients=email_raw.get("recipients", []) or [],
        subject_prefix=email_raw.get("subject_prefix", "Weinstein Report READY"),
        provider=email_raw.get("provider", "smtp"),
        smtp=SMTPAuthCfg(
            host=smtp_raw.get("host", "smtp.gmail.com"),
            port_ssl=int(smtp_raw.get("port_ssl", 587)),
            username=smtp_raw.get("username", ""),
            app_password=smtp_raw.get("app_password", ""),
        ),
    )
    notifications = NotificationsCfg(email=email)

    reporting = ReportingCfg(**raw.get("reporting", {}))
    universe = UniverseCfg(**raw.get("universe", {}))
    risk = RiskCfg(**raw.get("risk", {}))

    return Cfg(
        app=app,
        sheets=sheets,
        google=google,
        notifications=notifications,
        reporting=reporting,
        universe=universe,
        risk=risk,
    )


# ----------------------------
# Google Sheets helpers
# ----------------------------

def _open_gsheet(cfg: Cfg) -> Optional[gspread.Spreadsheet]:
    if gspread is None or ServiceAccountCredentials is None:
        return None
    if not cfg.google.service_account_json or not os.path.exists(cfg.google.service_account_json):
        return None
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(cfg.google.service_account_json, scope)
        client = gspread.authorize(creds)
        return client.open_by_url(cfg.sheets.url or cfg.sheets.sheet_url)
    except Exception as e:
        print(f"⚠️  Could not open Google Sheet: {e}", file=sys.stderr)
        return None


def _read_sheet_as_df(sh: gspread.Spreadsheet, tab_name: str) -> pd.DataFrame:
    ws = sh.worksheet(tab_name)
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    # try numeric casting on common columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col].str.replace(",", "").str.replace("$", ""), errors="ignore")
        except Exception:
            pass
    return df


# ----------------------------
# Weekly CSV discovery
# ----------------------------

def _find_latest_weekly_csv(reporting_dir: str) -> Optional[str]:
    # pattern like: weinstein_weekly_equities_YYYYMMDD_HHMM.csv
    pattern = os.path.join(reporting_dir, "weinstein_weekly_equities_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


# ----------------------------
# Snapshot and Sell/Risk logic
# ----------------------------

def build_snapshot_from_weekly(latest_weekly_csv: Optional[str]) -> Tuple[str, pd.DataFrame]:
    if not latest_weekly_csv or not os.path.exists(latest_weekly_csv):
        return "Snapshot (ordered by weekly rank & stage)\n(no weekly CSV found)\n", pd.DataFrame()

    try:
        df = pd.read_csv(latest_weekly_csv)
    except Exception as e:
        return f"Snapshot (ordered by weekly rank & stage)\n(weekly CSV read error: {e})\n", pd.DataFrame()

    # Try to normalize column names we need
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("ticker", "symbol"):
            rename_map[c] = "ticker"
        elif lc in ("stage",):
            rename_map[c] = "stage"
        elif lc in ("price", "last", "last_price", "close"):
            rename_map[c] = "price"
        elif lc in ("ma30", "sma30"):
            rename_map[c] = "ma30"
        elif lc in ("weekly_rank", "rank", "rank_weekly"):
            rename_map[c] = "weekly_rank"

    df = df.rename(columns=rename_map)

    # If no rank, fabricate one
    if "weekly_rank" not in df.columns:
        df["weekly_rank"] = 999999

    # Keep the most important columns
    cols = [c for c in ["ticker", "stage", "price", "ma30", "pivot10w", "vol_pace_vs50dma", "two_bar_confirm", "last_bar_vol_ok", "weekly_rank"] if c in df.columns]
    if not cols:
        # fallback to whatever is there
        cols = list(df.columns)[:4]

    # Order by weekly_rank then stage
    df_small = df[cols].copy()
    df_small = df_small.sort_values(by=["weekly_rank"] if "weekly_rank" in df_small.columns else cols[:1])

    # Pretty text block (compact)
    lines = ["Snapshot (ordered by weekly rank & stage)", "ticker\tstage\tprice\tma30"]
    for _, r in df_small.iterrows():
        ticker = str(r.get("ticker", ""))
        stage = str(r.get("stage", ""))
        price = r.get("price", "")
        ma30 = r.get("ma30", "")
        # format floats
        price_s = f"{float(price):.6f}" if isinstance(price, (int, float, np.floating)) or str(price).replace('.', '', 1).isdigit() else str(price)
        ma30_s = f"{float(ma30):.6f}" if isinstance(ma30, (int, float, np.floating)) or str(ma30).replace('.', '', 1).isdigit() else str(ma30)
        lines.append(f"{ticker}\t{stage}\t{price_s}\t{ma30_s}")
    return "\n".join(lines) + "\n", df_small


def build_sell_risk_from_positions(sh: Optional[gspread.Spreadsheet],
                                   open_positions_tab: str,
                                   weekly_df: pd.DataFrame) -> str:
    """
    Simple SELL rules:
    - If Total Gain/Loss Percent <= -8%  => Position SELL
    - If weekly stage is 'Stage 4 (Downtrend)' AND Total Gain/Loss Dollar < 0 => Position SELL
    """
    lines = ["Sell / Risk Triggers (Tracked Positions & Position Recommendations)"]
    if sh is None:
        lines.append("(positions: sheet access unavailable)\n")
        return "\n".join(lines)

    try:
        dfp = _read_sheet_as_df(sh, open_positions_tab)
    except Exception as e:
        lines.append(f"(positions read error: {e})\n")
        return "\n".join(lines)

    if dfp.empty:
        lines.append("(no open positions)\n")
        return "\n".join(lines)

    # Normalize headers we use
    def _col(df, candidates: List[str]) -> Optional[str]:
        lcmap = {c.lower().strip(): c for c in df.columns}
        for name in candidates:
            if name.lower() in lcmap:
                return lcmap[name.lower()]
        return None

    sym_col = _col(dfp, ["Symbol", "Ticker", "symbol", "ticker"])
    last_price_col = _col(dfp, ["Last Price", "last", "Price"])
    avg_cost_col = _col(dfp, ["Average Cost Basis", "Avg Cost", "Average Cost"])
    gl_pct_col = _col(dfp, ["Total Gain/Loss Percent", "P/L %", "GL %"])
    gl_dol_col = _col(dfp, ["Total Gain/Loss Dollar", "P/L", "GL $"])

    # Build weekly stage map
    stage_map = {}
    if not weekly_df.empty and "ticker" in weekly_df.columns and "stage" in weekly_df.columns:
        for _, r in weekly_df.iterrows():
            stage_map[str(r["ticker"])] = str(r["stage"])

    recs = []
    idx = 1

    for _, r in dfp.iterrows():
        sym = str(r.get(sym_col, "")).strip() if sym_col else ""
        if not sym:
            continue

        # compute loss percent if not present
        gl_pct = None
        if gl_pct_col and pd.notna(r.get(gl_pct_col)):
            try:
                gl_pct = float(str(r[gl_pct_col]).replace("%", "").strip())
            except Exception:
                gl_pct = None
        if gl_pct is None and last_price_col and avg_cost_col:
            try:
                last_p = float(str(r[last_price_col]).replace("$", "").replace(",", ""))
                avg_c = float(str(r[avg_cost_col]).replace("$", "").replace(",", ""))
                if avg_c > 0:
                    gl_pct = (last_p - avg_c) / avg_c * 100.0
            except Exception:
                gl_pct = None

        gl_dol = None
        if gl_dol_col and pd.notna(r.get(gl_dol_col)):
            try:
                gl_dol = float(str(r[gl_dol_col]).replace("$", "").replace(",", ""))
            except Exception:
                gl_dol = None

        stage = stage_map.get(sym, "—")
        reason = None

        if gl_pct is not None and gl_pct <= -8.0:
            reason = "drawdown ≤ −8%"
        elif "Stage 4" in stage and gl_dol is not None and gl_dol < 0:
            reason = "Stage 4 + negative P/L"

        if reason:
            # Try grabbing a last price to echo
            px = None
            if last_price_col and pd.notna(r.get(last_price_col)):
                try:
                    px = float(str(r[last_price_col]).replace("$", "").replace(",", ""))
                except Exception:
                    px = None
            px_s = f"{px:.2f}" if px is not None else "—"
            recs.append(f"{idx}. {sym} @ {px_s} — {reason} ({stage}, weekly —) (Position SELL)")
            idx += 1

    if not recs:
        lines.append("(no SELLTRIG positions)\n")
        return "\n".join(lines)

    lines.extend(recs)
    return "\n".join(lines)


# ----------------------------
# Crypto Section
# ----------------------------

def _classify_stage(price: float, ma30: float, ma30_slope: float) -> Tuple[str, str]:
    """
    Rough stage classification:
    - Stage 2: price > ma30 and slope > 0
    - Stage 4: price < ma30 and slope < 0
    - Stage 3: price > ma30 and slope <= 0
    - Stage 1: price < ma30 and slope >= 0
    Returns (stage_label, short_term_state_wk)
    """
    if price is None or ma30 is None or ma30_slope is None:
        return ("Unknown", "Unknown")

    if price > ma30 and ma30_slope > 0:
        return ("Stage 2 (Uptrend)", "StageConflict")
    if price < ma30 and ma30_slope < 0:
        return ("Stage 4 (Downtrend)", "StageConflict")
    if price > ma30 and ma30_slope <= 0:
        return ("Stage 3 (Topping)", "StageConflict")
    if price < ma30 and ma30_slope >= 0:
        return ("Stage 1 (Basing)", "StageConflict")
    return ("Unknown", "StageConflict")


def _decide_buy_flag(price: float, ma10: float, ma30: float, rs_last: float, rs_ma30: float, stage: str) -> str:
    """
    Very simple decision:
    - BUY if stage is Stage 2 AND price > ma10 AND rs_last >= rs_ma30
    - WATCH if stage in (Stage 1, Stage 3) and rs_last >= rs_ma30
    - otherwise AVOID
    """
    if any(v is None for v in (price, ma10, ma30, rs_last, rs_ma30)):
        return "Avoid"

    if "Stage 2" in stage and price > ma10 and rs_last >= rs_ma30:
        return "Buy"
    if ("Stage 1" in stage or "Stage 3" in stage) and rs_last >= rs_ma30:
        return "Watch"
    return "Avoid"


def _rs_series(asset_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    # Align
    df = pd.concat([asset_close, bench_close], axis=1).dropna()
    if df.shape[1] < 2 or df.empty:
        return pd.Series(dtype=float)
    rs = df.iloc[:, 0] / df.iloc[:, 1]
    rs.name = "rs"
    return rs


def build_crypto_section_from_signals(config_path: str, cfg: Cfg, benchmark: str = "BTC-USD") -> str:
    """
    - Read Signals sheet; pick crypto tickers (ending with -USD). If none found, use [BTC-USD, ETH-USD, SOL-USD]
    - Download ~180 days of daily data via yfinance
    - Compute ma10, ma30, rs vs benchmark
    - Decide Buy/Watch/Avoid
    - Build a table like the one you showed
    """
    tz = cfg.app.timezone or "America/Chicago"
    now = _now_tz(tz)
    # Discover crypto tickers from Signals
    crypto_tickers: List[str] = []
    sh = _open_gsheet(cfg)
    if sh is not None:
        try:
            sig = _read_sheet_as_df(sh, cfg.sheets.signals_tab)
            if not sig.empty:
                if "Ticker" in sig.columns:
                    for t in sig["Ticker"].astype(str):
                        t = t.strip().upper()
                        if t.endswith("-USD"):
                            crypto_tickers.append(t)
        except Exception as e:
            print(f"⚠️  Signals read failed: {e}", file=sys.stderr)

    if not crypto_tickers:
        crypto_tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

    # Unique and keep order
    seen = set()
    ordered_cryptos = []
    for t in crypto_tickers:
        if t not in seen:
            seen.add(t)
            ordered_cryptos.append(t)

    # need yfinance
    if yf is None:
        return "Crypto Weekly — (yfinance unavailable) \n"

    # Fetch prices
    period_days = 220
    start = (now - dt.timedelta(days=period_days)).date().isoformat()
    end = (now + dt.timedelta(days=1)).date().isoformat()

    # Benchmark series
    try:
        bench = yf.download(benchmark, start=start, end=end, interval="1d", progress=False)
        bench_close = bench["Close"].dropna() if not bench.empty and "Close" in bench.columns else pd.Series(dtype=float)
    except Exception:
        bench_close = pd.Series(dtype=float)

    rows = []
    summary_counts = {"Buy": 0, "Watch": 0, "Avoid": 0}

    for tkr in ordered_cryptos:
        try:
            df = yf.download(tkr, start=start, end=end, interval="1d", progress=False)
        except Exception:
            df = pd.DataFrame()

        if df.empty or "Close" not in df.columns:
            # Blank row
            rows.append(
                dict(
                    ticker=tkr,
                    asset_class="Crypto",
                    industry="",
                    sector="",
                    BuySignal="AVOID",
                    chart="chart",
                    stage="Unknown",
                    short_term_state_wk="",
                    price=np.nan,
                    ma10=np.nan,
                    ma30=np.nan,
                    dist_ma_pct=np.nan,
                    ma_slope_per_wk=np.nan,
                    rs=np.nan,
                    rs_ma30=np.nan,
                    rs_above_ma="No",
                    rs_slope_per_wk=np.nan,
                    notes="NoData",
                )
            )
            summary_counts["Avoid"] += 1
            continue

        close = df["Close"].dropna()
        ma10 = close.rolling(10).mean()
        ma30 = close.rolling(30).mean()

        price = _last(close)
        ma10_last = _last(ma10)
        ma30_last = _last(ma30)

        # Distance to MA30
        dist_ma_pct = None
        if ma30_last and ma30_last != 0 and price is not None:
            dist_ma_pct = (price - ma30_last) / ma30_last * 100.0

        # slope per "wk" ~ last - value 5 days ago
        ma30_slope = _slope_last_5(ma30)

        # RS
        rs = _rs_series(close, bench_close) if not bench_close.empty else pd.Series(dtype=float)
        rs_ma30 = rs.rolling(30).mean() if not rs.empty else pd.Series(dtype=float)
        rs_last = _last(rs) if not rs.empty else None
        rs_ma30_last = _last(rs_ma30) if not rs_ma30.empty else None
        rs_slope = _slope_last_5(rs) if not rs.empty else None

        stage, short_conf = _classify_stage(price, ma30_last if ma30_last is not None else np.nan,
                                            ma30_slope if ma30_slope is not None else np.nan)

        decision = _decide_buy_flag(price, ma10_last, ma30_last, rs_last if rs_last is not None else -np.inf,
                                    rs_ma30_last if rs_ma30_last is not None else np.inf, stage)

        # Count
        summary_counts[decision.capitalize() if decision else "Avoid"] = summary_counts.get(decision.capitalize(), 0) + 1

        # notes
        notes = ""
        if decision.lower() == "avoid" and price is not None and ma30_last is not None and price < ma30_last:
            notes = "Price"
        rows.append(
            dict(
                ticker=tkr,
                asset_class="Crypto",
                industry="",
                sector="",
                BuySignal=decision.upper(),
                chart="chart",
                stage=stage,
                short_term_state_wk=short_conf,
                price=price,
                ma10=ma10_last,
                ma30=ma30_last,
                dist_ma_pct=dist_ma_pct,
                ma_slope_per_wk=ma30_slope,
                rs=rs_last if rs_last is not None else np.nan,
                rs_ma30=rs_ma30_last if rs_ma30_last is not None else np.nan,
                rs_above_ma="Yes" if (rs_last is not None and rs_ma30_last is not None and rs_last >= rs_ma30_last) else "No",
                rs_slope_per_wk=rs_slope if rs_slope is not None else np.nan,
                notes=notes,
            )
        )

    # Compose text block
    total = sum(summary_counts.values())
    line1 = f"Crypto Weekly — Benchmark: {benchmark}"
    line2 = f"Generated {now.strftime('%Y-%m-%d %H:%M')}"
    line3 = f"Crypto Summary: ✅ Buy: {summary_counts.get('Buy',0)}   |   🟡 Watch: {summary_counts.get('Watch',0)}   |   🔴 Avoid: {summary_counts.get('Avoid',0)}   (Total: {total})"

    header = "\t".join([
        "ticker", "asset_class", "industry", "sector", "Buy Signal", "chart", "stage",
        "short_term_state_wk", "price", "ma10", "ma30", "dist_ma_pct",
        "ma_slope_per_wk", "rs", "rs_ma30", "rs_above_ma", "rs_slope_per_wk", "notes"
    ])

    body_lines = []
    for r in rows:
        body_lines.append("\t".join([
            str(r["ticker"]),
            str(r["asset_class"]),
            str(r["industry"]),
            str(r["sector"]),
            str(r["BuySignal"]).title(),  # Show like Buy/Watch/Avoid
            str(r["chart"]),
            str(r["stage"]),
            str(r["short_term_state_wk"]),
            f"{r['price']:.6f}" if isinstance(r['price'], (int, float, np.floating)) and not math.isnan(r['price']) else str(r['price']),
            f"{r['ma10']:.6f}" if isinstance(r['ma10'], (int, float, np.floating)) and not math.isnan(r['ma10']) else str(r['ma10']),
            f"{r['ma30']:.6f}" if isinstance(r['ma30'], (int, float, np.floating)) and not math.isnan(r['ma30']) else str(r['ma30']),
            _fmt_pct(r['dist_ma_pct'], 2) if isinstance(r['dist_ma_pct'], (int, float, np.floating)) and not math.isnan(r['dist_ma_pct']) else str(r['dist_ma_pct']),
            f"{r['ma_slope_per_wk']:.2f}%" if isinstance(r['ma_slope_per_wk'], (int, float, np.floating)) and not math.isnan(r['ma_slope_per_wk']) else str(r['ma_slope_per_wk']),
            f"{r['rs']:.6f}" if isinstance(r['rs'], (int, float, np.floating)) and not math.isnan(r['rs']) else str(r['rs']),
            f"{r['rs_ma30']:.6f}" if isinstance(r['rs_ma30'], (int, float, np.floating)) and not math.isnan(r['rs_ma30']) else str(r['rs_ma30']),
            str(r['rs_above_ma']),
            f"{r['rs_slope_per_wk']:.2f}%" if isinstance(r['rs_slope_per_wk'], (int, float, np.floating)) and not math.isnan(r['rs_slope_per_wk']) else str(r['rs_slope_per_wk']),
            str(r['notes']),
        ]))

    block = "\n".join([line1, line2, line3, header] + body_lines)
    return block + "\n"


# ----------------------------
# Intraday triggers (skeleton)
# ----------------------------

def evaluate_intraday_candidates(cfg: Cfg) -> Tuple[List[str], List[str], List[str]]:
    """
    Placeholder logic:
    - Using the weekly CSV 'Stage 1/2' set as focus list (if that column exists)
    - No real intraday checks here (to avoid heavy data dependency). Prints 'No signals' when none.
    """
    latest_weekly = _find_latest_weekly_csv(cfg.reporting.output_dir)
    focus = []
    if latest_weekly and os.path.exists(latest_weekly):
        try:
            df = pd.read_csv(latest_weekly)
            # Try to select Stage 1/2
            if "stage" in df.columns:
                focus = df.loc[df["stage"].astype(str).str.contains("Stage 1|Stage 2", case=False, regex=True), "ticker"].astype(str).tolist()
            elif "Ticker" in df.columns:
                focus = df["Ticker"].astype(str).tolist()
        except Exception:
            pass

    # Here you would fetch 60m bars & compute:
    #  - over 10-week pivot
    #  - price vs SMA150 (as proxy 30-wk)
    #  - RS support
    #  - volume pace >= thresholds, etc.
    # For now: return empty results to match your “No BUY/NEAR/SELL” when nothing qualifies.
    buys, nears, sells = [], [], []
    return buys, nears, sells


# ----------------------------
# Report assembly
# ----------------------------

def assemble_report(cfg: Cfg) -> str:
    tz = cfg.app.timezone or "America/Chicago"
    now = _now_tz(tz)
    header = [
        f"Weinstein Intraday Watch — {now.strftime('%Y-%m-%d %H:%M')}",
        "BUY: Weekly Stage 1/2 + confirm over ~10-week pivot & 30-wk MA proxy (SMA150), +0.4% headroom, RS support, volume pace ≥ 1.3×. For 60m bars: ≥40 min elapsed & intrabar pace ≥ 1.2×.",
        "NEAR-TRIGGER: Stage 1/2 + RS ok, price within 0.3% below pivot or first close over pivot but not fully confirmed yet, volume pace ≥ 1.0×.",
        "SELL-TRIGGER: Confirmed crack below MA150 by 0.5% with persistence; for 60m bars, ≥40 min elapsed & intrabar pace ≥ 1.2×.",
        "",
    ]

    buys, nears, sells = evaluate_intraday_candidates(cfg)

    # BUY
    lines = ["Buy Triggers (ranked)"]
    if not buys:
        lines.append("No BUY signals.")
    else:
        for i, s in enumerate(buys, 1):
            lines.append(f"{i}. {s}")
    lines.append("")

    # NEAR
    lines.append("Near-Triggers (ranked)")
    if not nears:
        lines.append("No NEAR signals.")
    else:
        for i, s in enumerate(nears, 1):
            lines.append(f"{i}. {s}")
    lines.append("")

    # SELL
    lines.append("Sell Triggers (ranked)")
    if not sells:
        lines.append("No SELLTRIG signals.")
    else:
        for i, s in enumerate(sells, 1):
            lines.append(f"{i}. {s}")
    lines.append("")

    # Charts placeholder
    lines.append("Charts (Price + SMA150 ≈ 30-wk MA, RS normalized)")
    if buys or nears:
        for t in (buys + nears)[:4]:
            lines.append(t)
    lines.append("")  # blank

    # Sell / Risk Triggers (from positions)
    sh = _open_gsheet(cfg)
    # snapshot uses weekly csv
    latest_weekly = _find_latest_weekly_csv(cfg.reporting.output_dir)
    snapshot_text, snapshot_df = build_snapshot_from_weekly(latest_weekly)
    sell_block = build_sell_risk_from_positions(sh, cfg.sheets.open_positions_tab, snapshot_df)

    lines.append(sell_block)
    lines.append("")

    # Snapshot table
    lines.append(snapshot_text.rstrip())
    lines.append("")

    # --------- CRYPTO WEEKLY goes here (BEFORE weekly summary) ----------
    crypto_block = build_crypto_section_from_signals("", cfg, benchmark="BTC-USD")
    lines.append(crypto_block.rstrip())
    lines.append("")

    # Weekly summary block (very light – pull a few lines if Open_Positions exists)
    lines.append("Weinstein Weekly – Summary")
    # If Open_Positions has totals, we can echo a few
    if sh is not None:
        try:
            dfp = _read_sheet_as_df(sh, cfg.sheets.open_positions_tab)
            # Try to display a couple of “headline” totals if present
            # Otherwise just show a header and a cue line
            if not dfp.empty:
                lines.append("Per-position Snapshot")
                # print a compact 2-4 columns if available
                keep_cols = []
                for c in ["Symbol", "Description", "industry", "sector", "Quantity", "Last Price",
                          "Current Value", "Cost Basis Total", "Average Cost Basis",
                          "Total Gain/Loss Dollar", "Total Gain/Loss Percent", "Recommendation"]:
                    if c in dfp.columns:
                        keep_cols.append(c)
                if not keep_cols:
                    keep_cols = dfp.columns[:6].tolist()
                # show a limited set
                sub = dfp[keep_cols].copy()
                # Make a compact print
                # mimic your email style with tab-sep
                lines.append("\t".join(list(sub.columns)))
                for _, r in sub.iterrows():
                    parts = []
                    for c in sub.columns:
                        v = r[c]
                        parts.append(str(v))
                    lines.append("\t".join(parts))
            else:
                lines.append("(no positions found)")
        except Exception as e:
            lines.append(f"(weekly summary error: {e})")
    else:
        lines.append("(weekly summary: sheet access unavailable)")

    lines.append("")
    lines.append("✅ Intraday tick complete.")

    return "\n".join(header + lines)


# ----------------------------
# CLI
# ----------------------------

def run(_config_path: str):
    print(f"▶️ [{time.strftime('%H:%M:%S')}] Intraday watcher starting with config: {_config_path}")
    cfg = load_config(_config_path)

    # Try to note weekly csv
    latest_weekly = _find_latest_weekly_csv(cfg.reporting.output_dir)
    if latest_weekly:
        print(f"·· [{time.strftime('%H:%M:%S')}] Weekly CSV: {latest_weekly}")

    # Minimal universe log (if weekly available)
    if latest_weekly and os.path.exists(latest_weekly):
        try:
            dfw = pd.read_csv(latest_weekly)
            # stage 1/2 rows
            if "stage" in dfw.columns:
                focus_n = (dfw["stage"].astype(str).str.contains("Stage 1|Stage 2", case=False, regex=True)).sum()
                print(f"• [{time.strftime('%H:%M:%S')}] Focus universe: {int(focus_n)} symbols (Stage 1/2).")
        except Exception:
            pass

    # We won't do heavy intraday downloads here; just assemble & print
    report = assemble_report(cfg)
    print(report)


if __name__ == "__main__":
    try:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=str, default="./config.yaml")
        args = ap.parse_args()
        run(_config_path=args.config)
    except Exception as e:
        print("❌ Intraday watcher encountered an error.")
        print(str(e))
        raise
