# buffett_options_engine.py

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

import gspread
from google.oauth2.service_account import Credentials

try:
    from weinstein_mailer import send_email as weinstein_send_email
except ImportError:
    weinstein_send_email = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "config.yaml"
BUFFETT_TAB_NAME = "Buffett_Put_Signals"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Google Sheets helpers
# ---------------------------------------------------------------------------

def get_gspread_client(cfg: Dict[str, Any]) -> gspread.Client:
    google_cfg = cfg.get("google", {})
    sa_path = google_cfg.get("service_account_json")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(sa_path, scopes=scopes)
    return gspread.authorize(creds)


def open_sheet(cfg: Dict[str, Any]) -> gspread.Spreadsheet:
    sheets_cfg = cfg.get("sheets", {})
    url = sheets_cfg.get("sheet_url") or sheets_cfg.get("url")
    client = get_gspread_client(cfg)
    return client.open_by_url(url)


def extract_tickers_from_tab(
    sh: gspread.Spreadsheet,
    tab_name: str,
    ticker_header_candidates: List[str] = None,
) -> List[str]:
    """
    Generic helper to pull a Ticker/Symbol column from a worksheet.

    It looks for any header name in `ticker_header_candidates` and falls back to
    the first column if nothing matches.
    """
    if ticker_header_candidates is None:
        ticker_header_candidates = ["Ticker", "Symbol", "ticker", "symbol"]

    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        return []

    rows = ws.get_all_values()
    if not rows:
        return []

    header = rows[0]
    data_rows = rows[1:]

    # Find ticker column index
    idx = None
    for i, col_name in enumerate(header):
        if col_name.strip() in ticker_header_candidates:
            idx = i
            break
    if idx is None:
        idx = 0  # fallback to first column

    tickers = []
    for row in data_rows:
        if idx < len(row):
            val = row[idx].strip()
            if val:
                tickers.append(val)
    return tickers


def get_buffett_universe(cfg: Dict[str, Any]) -> List[str]:
    """
    Build the Buffett CSP universe based on:
      - Tickers from Signals tab
      - Tickers from Open_Positions tab
      - Optional SP500 tickers + extra list
    """
    sh = open_sheet(cfg)
    sheets_cfg = cfg.get("sheets", {})
    signals_tab = sheets_cfg.get("signals_tab", "Signals")
    open_pos_tab = sheets_cfg.get("open_positions_tab", "Open_Positions")

    print(f"🔍 Loading tickers from tab '{signals_tab}'...")
    sig_tickers = extract_tickers_from_tab(sh, signals_tab)

    print(f"🔍 Loading tickers from tab '{open_pos_tab}'...")
    pos_tickers = extract_tickers_from_tab(sh, open_pos_tab)

    universe_cfg = cfg.get("universe", {})
    # New knob (optional) – if missing, default is "hybrid"
    buffett_mode = universe_cfg.get("buffett_mode", "hybrid")  # signals_only | sp500 | hybrid
    extra = universe_cfg.get("extra", []) or []

    tickers: List[str] = []

    if buffett_mode in ("signals_only", "hybrid"):
        tickers.extend(sig_tickers)
        tickers.extend(pos_tickers)

    if buffett_mode in ("sp500", "hybrid"):
        sp500 = load_sp500_tickers()
        tickers.extend(sp500)
        tickers.extend(extra)

    # Clean & normalize
    cleaned = []
    for t in tickers:
        t = t.strip()
        if not t:
            continue
        # Skip known non-tickers from Fidelity exports etc.
        if any(bad in t.upper() for bad in ["FCASH", "SPAXX", "PENDING", "**", "USD"]):
            continue
        # Skip obvious option-style strings (start with '-' or contain spaces)
        if t.startswith("-") or " " in t:
            continue
        # Normalize BRK-B style to Yahoo BRK-B (already ok) or BRK.B
        if t.upper() in ("BRKB", "BRK-B"):
            t = "BRK-B"
        # Simple filter: letters, numbers, dot, hyphen allowed
        cleaned.append(t)

    unique = sorted(set(cleaned))
    print(f"✅ Using {len(unique)} tickers in Buffett CSP universe.")
    return unique


def load_sp500_tickers() -> List[str]:
    """
    Load S&P 500 tickers from Wikipedia via pandas.read_html.

    If it fails for any reason, returns an empty list so the engine can still run.
    """
    try:
        import pandas as _pd  # local alias to avoid confusion
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = _pd.read_html(url)
        if not tables:
            return []
        df = tables[0]
        if "Symbol" not in df.columns:
            return []
        syms = df["Symbol"].astype(str).str.strip().tolist()
        # Replace "." with "-" for Yahoo tickers (e.g., BRK.B -> BRK-B)
        syms = [s.replace(".", "-") for s in syms]
        return syms
    except Exception as e:
        print(f"⚠️ Could not load S&P 500 tickers: {e}")
        return []


# ---------------------------------------------------------------------------
# Options scanning
# ---------------------------------------------------------------------------

def get_buffett_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve Buffett-specific parameters from config with sensible defaults.
    """
    app_cfg = cfg.get("app", {})
    ordering = app_cfg.get("ordering", {})

    buffett_cfg = cfg.get("buffett", {}) or {}

    params = {
        "min_dte": buffett_cfg.get("min_dte", 7),
        "max_dte": buffett_cfg.get("max_dte", 45),
        "risk_buffer": buffett_cfg.get("buffer_pct", 0.10),  # 10% OTM
        "min_annual_yield": buffett_cfg.get("min_annual_yield", 0.05),  # 5%
        "min_bid": buffett_cfg.get("min_bid", 0.10),
        "min_open_interest": buffett_cfg.get("min_open_interest", 50),
        "min_volume": buffett_cfg.get("min_volume", 1),
        "max_candidates_per_ticker": buffett_cfg.get("max_candidates_per_ticker", 50),
        "account_size": ordering.get("account_size", 5000),
        "risk_per_trade_pct": ordering.get("risk_per_trade_pct", 0.01),
    }
    return params


def fetch_current_price(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="1d")
    if hist.empty:
        raise ValueError(f"No price history for {ticker}")
    # Use iloc[-1] to avoid FutureWarning
    return float(hist["Close"].iloc[-1])


def get_option_candidates_for_ticker(
    ticker: str,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    For a single ticker:
      - fetch option chain
      - keep puts with strikes below (1 - buffer) * price
      - keep expiries between min_dte and max_dte
      - compute annualized yield and position sizing
      - filter by liquidity & yield
    """
    stock = yf.Ticker(ticker)
    try:
        expiry_dates = stock.options
    except Exception:
        return pd.DataFrame()

    today = datetime.utcnow().date()
    min_dte = params["min_dte"]
    max_dte = params["max_dte"]
    risk_buffer = params["risk_buffer"]
    min_annual_yield = params["min_annual_yield"]
    min_bid = params["min_bid"]
    min_oi = params["min_open_interest"]
    min_vol = params["min_volume"]
    account_size = params["account_size"]
    risk_per_trade_pct = params["risk_per_trade_pct"]

    try:
        current_price = fetch_current_price(ticker)
    except Exception:
        return pd.DataFrame()

    buffer_price = round(current_price * (1 - risk_buffer), 2)

    rows = []
    for expiry in expiry_dates:
        try:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        except Exception:
            continue

        dte = (exp_date - today).days
        if dte < min_dte or dte > max_dte:
            continue

        try:
            puts = stock.option_chain(expiry).puts.copy()
        except Exception:
            continue

        if puts.empty:
            continue

        # Basic numeric cleaning
        for col in ["strike", "bid", "ask", "openInterest", "volume", "impliedVolatility"]:
            if col in puts.columns:
                puts[col] = pd.to_numeric(puts[col], errors="coerce")

        # liquidity + OTM buffer
        mask = (
            (puts["strike"] <= buffer_price)
            & (puts["bid"] >= min_bid)
            & (puts["openInterest"].fillna(0) >= min_oi)
            & (puts["volume"].fillna(0) >= min_vol)
        )
        candidates = puts.loc[mask].copy()
        if candidates.empty:
            continue

        # Compute metrics
        dte_float = float(dte)
        candidates["ticker"] = ticker
        candidates["underlying_price"] = current_price
        candidates["dte"] = dte
        candidates["target_strike"] = buffer_price

        # Annualized yield based on bid
        candidates["yield_pct"] = candidates["bid"] / current_price / dte_float * 365.0

        # Buffer from current price
        candidates["buffer_pct"] = (1.0 - candidates["strike"] / current_price) * 100.0

        # Position sizing: max contracts given risk_per_trade
        max_capital = account_size * risk_per_trade_pct
        # A bit conservative: use strike * 100 as notional per contract
        candidates["notional_per_contract"] = candidates["strike"] * 100.0
        candidates["max_contracts"] = np.floor(max_capital / candidates["notional_per_contract"]).astype(int)
        candidates["max_contracts"] = candidates["max_contracts"].clip(lower=0)

        # Premium per contract / total
        candidates["premium_per_contract"] = candidates["bid"] * 100.0
        candidates["premium_total_for_max"] = candidates["premium_per_contract"] * candidates["max_contracts"]

        # Grade quality (A/B/C) – simple heuristic
        def grade_row(row):
            y = row["yield_pct"]
            b = row["buffer_pct"]
            if y >= 0.20 and b >= 15:
                return "A"
            if y >= 0.12 and b >= 12:
                return "B"
            return "C"

        candidates["grade"] = candidates.apply(grade_row, axis=1)

        # Filter by yield after computing
        candidates = candidates[candidates["yield_pct"] >= min_annual_yield]
        if candidates.empty:
            continue

        # Keep at most N per ticker, highest yield first
        candidates.sort_values("yield_pct", ascending=False, inplace=True)
        candidates = candidates.head(params["max_candidates_per_ticker"])

        candidates["expiry"] = expiry
        rows.append(candidates)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def scan_universe(
    tickers: List[str],
    params: Dict[str, Any],
) -> pd.DataFrame:
    all_results = []
    for tkr in tickers:
        print(f"Scanning {tkr}...")
        df = get_option_candidates_for_ticker(tkr, params)
        if not df.empty:
            all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined.sort_values(["yield_pct"], ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ---------------------------------------------------------------------------
# Sheet + CSV + Email
# ---------------------------------------------------------------------------

def to_string_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all cells to basic Python scalars / strings so they can be
    serialized safely into JSON for the Google Sheets API.
    """
    def conv(x):
        if isinstance(x, (pd.Timestamp, datetime)):
            return x.isoformat()
        if isinstance(x, float):
            if np.isfinite(x):
                return round(x, 6)
            return ""
        if pd.isna(x):
            return ""
        return x
    return df.applymap(conv)


def upload_df_to_sheet(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    tab_name: str = BUFFETT_TAB_NAME,
) -> None:
    sh = open_sheet(cfg)
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows="1000", cols="26")

    df_str = to_string_df(df)
    header = list(df_str.columns)
    values = df_str.values.tolist()

    # Clear old content
    ws.clear()

    # New gspread signature prefers values first, then range_name, but named args are safest.
    ws.update(range_name="A1", values=[header] + values)
    print(f"✅ Updated sheet tab '{tab_name}' with {len(df)} rows.")


def save_to_csv(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> str:
    reporting_cfg = cfg.get("reporting", {})
    out_dir = reporting_cfg.get("output_dir") or cfg.get("output", {}).get("dir") or "./output"
    os.makedirs(out_dir, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = os.path.join(out_dir, f"Buffett_Put_Signals_{now_str}.csv")
    df.to_csv(fpath, index=False)
    print(f"✅ Saved: {fpath}")
    return fpath


def build_email_body(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    csv_path: str,
) -> Tuple[str, str]:
    sheets_cfg = cfg.get("sheets", {})
    sheet_url = sheets_cfg.get("sheet_url") or sheets_cfg.get("url")

    universe_size = len(get_buffett_universe(cfg))
    candidates = len(df)

    # Build a small top-15 table
    top = df.head(15).copy()
    cols = [
        "ticker",
        "strike",
        "expiry",
        "underlying_price",
        "dte",
        "target_strike",
        "yield_pct",
        "buffer_pct",
        "grade",
        "max_contracts",
        "premium_per_contract",
        "premium_total_for_max",
    ]
    existing_cols = [c for c in cols if c in top.columns]
    top = top[existing_cols]

    def fmt_pct(x):
        try:
            return f"{float(x)*100:0.2f}%"
        except Exception:
            return str(x)

    if "yield_pct" in top.columns:
        top["yield_pct"] = top["yield_pct"].apply(fmt_pct)
    if "buffer_pct" in top.columns:
        top["buffer_pct"] = top["buffer_pct"].apply(
            lambda x: f"{float(x):0.2f}%" if x != "" else ""
        )

    # Text body (plain)
    lines = []
    lines.append("Buffett CSP Scan")
    lines.append("")
    lines.append(f"Universe size: {universe_size}")
    lines.append(f"Candidates found: {candidates}")
    lines.append(f"Google Sheet: {sheet_url}")
    lines.append(f"CSV path on server: {csv_path}")
    lines.append("")
    lines.append("Top 15 candidates by annualized yield:")
    lines.append("")
    lines.append("\t".join(existing_cols))
    for _, row in top.iterrows():
        vals = [str(row[c]) for c in existing_cols]
        lines.append("\t".join(vals))
    lines.append("")
    lines.append(
        "Notes: Yield is annualized using bid / underlying / DTE * 365. "
        "All strikes are at least the configured buffer below the current price, "
        "with expirations between the configured DTE range."
    )
    text_body = "\n".join(lines)

    # HTML body (simple)
    html_lines = []
    html_lines.append("<h2>Buffett CSP Scan</h2>")
    html_lines.append(f"<p><b>Universe size:</b> {universe_size}<br>")
    html_lines.append(f"<b>Candidates found:</b> {candidates}<br>")
    html_lines.append(f"<b>Google Sheet:</b> <a href='{sheet_url}'>{sheet_url}</a><br>")
    html_lines.append(f"<b>CSV path on server:</b> {csv_path}</p>")
    html_lines.append("<h3>Top 15 candidates by annualized yield</h3>")
    html_lines.append("<table border='1' cellspacing='0' cellpadding='4'>")
    html_lines.append("<tr>" + "".join(f"<th>{c}</th>" for c in existing_cols) + "</tr>")
    for _, row in top.iterrows():
        html_lines.append(
            "<tr>" + "".join(f"<td>{row[c]}</td>" for c in existing_cols) + "</tr>"
        )
    html_lines.append("</table>")
    html_lines.append(
        "<p><i>Notes:</i> Yield is annualized using bid / underlying / DTE * 365. "
        "All strikes are at least the configured buffer below the current price, "
        "with expirations between the configured DTE range.</p>"
    )
    html_body = "\n".join(html_lines)

    return text_body, html_body


def send_buffett_email(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    csv_path: str,
) -> None:
    if weinstein_send_email is None:
        print("⚠️ weinstein_mailer.send_email not available; skipping email.")
        return

    text_body, html_body = build_email_body(df, cfg, csv_path)

    email_cfg = cfg.get("notifications", {}).get("email", {})
    if not email_cfg.get("enabled", True):
        print("✉️  Email notifications disabled in config.notifications.email.enabled")
        return

    subj_prefix = email_cfg.get("subject_prefix", "Weinstein Report READY")
    subject = f"{subj_prefix} – Buffett CSP Scan"

    weinstein_send_email(subject=subject, html_body=html_body, text_body=text_body)
    print("✅ Buffett CSP summary email sent.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("🚀 Running Buffett Options Engine…")
    cfg = load_config()

    tickers = get_buffett_universe(cfg)
    params = get_buffett_params(cfg)

    df = scan_universe(tickers, params)
    if df.empty:
        print("⚠️ No suitable options found.")
        return

    csv_path = save_to_csv(df, cfg)
    upload_df_to_sheet(df, cfg, BUFFETT_TAB_NAME)
    send_buffett_email(df, cfg, csv_path)


if __name__ == "__main__":
    main()
