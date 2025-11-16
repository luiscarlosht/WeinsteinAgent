# buffett_options_engine.py

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

import gspread
from google.oauth2 import service_account

from weinstein_mailer import send_email

# ------------------------------
# Config / constants
# ------------------------------

CONFIG_PATH = "config.yaml"

# Conservative Buffett-style parameters
RISK_BUFFER = 0.10   # 10% below current price
MIN_DTE = 7          # skip ultra-short expiries
MAX_DTE = 45         # up to 45 DTE
MIN_YIELD = 0.008    # 0.8% annualized minimum yield

BUFFETT_TAB_NAME = "Buffett_Put_Signals"


# ------------------------------
# Helpers: config + Google Sheets
# ------------------------------

def load_config(cfg_path: str = CONFIG_PATH) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_gspread_client(cfg: dict):
    google_cfg = cfg["google"]
    sa_path = google_cfg["service_account_json"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = service_account.Credentials.from_service_account_file(sa_path, scopes=scopes)
    return gspread.authorize(creds)


def open_trading_hub_sheet(client, cfg: dict):
    sheets_cfg = cfg["sheets"]
    sheet_url = sheets_cfg.get("sheet_url") or sheets_cfg["url"]
    return client.open_by_url(sheet_url)


def worksheet_to_df(ws) -> pd.DataFrame:
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ------------------------------
# Universe building from Sheets
# ------------------------------

def is_equity_ticker(raw: str) -> bool:
    """
    Heuristic filter to keep plain equities/ETFs and drop:
    - crypto (e.g., BTC-USD)
    - options (start with '-' or long symbol+date)
    - cash placeholders (FCASH, SPAXX, PENDING ACTIVITY, etc.)
    """
    if not raw:
        return False

    t = str(raw).strip().upper()

    # Explicit junk / cash / placeholders
    blacklist_exact = {
        "FCASH", "SPAXX", "SPAXX**",
        "PENDING ACTIVITY", "PENDING", "FCASH**",
    }
    if t in blacklist_exact:
        return False

    # Crypto style tickers
    if t.endswith("-USD"):
        return False

    # Obvious options from brokers (leading -)
    if t.startswith("-"):
        return False

    # CUSIPs / all digits
    if t.replace(".", "").isdigit():
        return False

    # Long + digits: likely option codes
    if len(t) > 8 and any(ch.isdigit() for ch in t):
        return False

    return True


def load_tickers_from_sheets(cfg: dict) -> list[str]:
    client = get_gspread_client(cfg)
    sh = open_trading_hub_sheet(client, cfg)
    sheets_cfg = cfg["sheets"]

    signals_tab = sheets_cfg.get("signals_tab", "Signals")
    open_pos_tab = sheets_cfg.get("open_positions_tab", "Open_Positions")

    print(f"🔍 Loading tickers from tab '{signals_tab}'...")
    try:
        ws_signals = sh.worksheet(signals_tab)
        df_signals = worksheet_to_df(ws_signals)
    except Exception as e:
        print(f"⚠️ Could not read Signals tab '{signals_tab}': {e}")
        df_signals = pd.DataFrame()

    print(f"🔍 Loading tickers from tab '{open_pos_tab}'...")
    try:
        ws_open = sh.worksheet(open_pos_tab)
        df_open = worksheet_to_df(ws_open)
    except Exception as e:
        print(f"⚠️ Could not read Open_Positions tab '{open_pos_tab}': {e}")
        df_open = pd.DataFrame()

    tickers: set[str] = set()

    # From Signals
    if not df_signals.empty:
        ticker_col_candidates = [c for c in df_signals.columns if c.lower() in ("ticker", "symbol")]
        if ticker_col_candidates:
            tcol = ticker_col_candidates[0]
            for t in df_signals[tcol].dropna().unique():
                if is_equity_ticker(t):
                    tickers.add(str(t).strip().upper())

    # From Open_Positions
    if not df_open.empty:
        sym_col_candidates = [c for c in df_open.columns if c.lower() in ("symbol", "ticker")]
        if sym_col_candidates:
            scol = sym_col_candidates[0]
            for t in df_open[scol].dropna().unique():
                if is_equity_ticker(t):
                    tickers.add(str(t).strip().upper())

    tickers_list = sorted(tickers)
    print(f"✅ Using {len(tickers_list)} tickers in Buffett CSP universe.")
    return tickers_list


# ------------------------------
# Options scanning logic
# ------------------------------

def get_option_chain_for_ticker(
    ticker: str,
    risk_buffer: float = RISK_BUFFER,
    min_dte: int = MIN_DTE,
    max_dte: int = MAX_DTE,
) -> pd.DataFrame:
    stock = yf.Ticker(ticker)

    try:
        expiry_dates = stock.options
        if not expiry_dates:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    today = datetime.utcnow().date()
    valid_dates: list[tuple[str, int]] = []

    for d in expiry_dates:
        try:
            d_date = datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (d_date - today).days
        if min_dte <= dte <= max_dte:
            valid_dates.append((d, dte))

    if not valid_dates:
        return pd.DataFrame()

    # Current underlying price
    try:
        hist = stock.history(period="1d")
        if hist.empty:
            return pd.DataFrame()
        current_price = float(hist["Close"].iloc[-1])
    except Exception:
        return pd.DataFrame()

    buffer_price = round(current_price * (1 - risk_buffer), 2)
    results = []

    for expiry_str, dte in valid_dates:
        try:
            chain = stock.option_chain(expiry_str)
            puts = chain.puts
        except Exception:
            continue

        if puts is None or puts.empty:
            continue

        candidates = puts[puts["strike"] <= buffer_price].copy()
        if candidates.empty:
            continue

        candidates["ticker"] = ticker
        candidates["underlying_price"] = current_price
        candidates["dte"] = dte
        candidates["target_strike"] = buffer_price
        candidates["expiry"] = expiry_str

        # Annualized yield: bid / underlying / DTE * 365
        candidates["yield_pct"] = (
            candidates["bid"].astype(float) /
            current_price /
            dte * 365.0
        )

        results.append(candidates)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    return df


def scan_all(tickers: list[str]) -> pd.DataFrame:
    all_results = []

    for tkr in tickers:
        print(f"Scanning {tkr}...")
        try:
            df = get_option_chain_for_ticker(tkr)
        except Exception as e:
            print(f"  ⚠️ Error scanning {tkr}: {e}")
            continue

        if df is None or df.empty:
            continue

        df = df[df["yield_pct"] >= MIN_YIELD]
        if df.empty:
            continue

        all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined.sort_values(["yield_pct"], ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ------------------------------
# Output helpers
# ------------------------------

def get_output_dir(cfg: dict) -> str:
    reporting_cfg = cfg.get("reporting", {})
    sheets_cfg = cfg.get("sheets", {})
    return (
        reporting_cfg.get("output_dir")
        or sheets_cfg.get("output_dir")
        or "./output"
    )


def save_to_csv(df: pd.DataFrame, cfg: dict) -> str:
    out_dir = get_output_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = os.path.join(out_dir, f"Buffett_Put_Signals_{now}.csv")
    df.to_csv(fpath, index=False)
    print(f"✅ Saved: {fpath}")
    return fpath


def normalize_df_for_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all values to JSON/Sheets-safe Python types."""
    from pandas import Timestamp

    def conv(x):
        if isinstance(x, Timestamp):
            return x.isoformat()
        if isinstance(x, datetime):
            return x.isoformat()
        if pd.isna(x):
            return ""
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        return x

    return df.applymap(conv)


def upload_df_to_sheet(df: pd.DataFrame, cfg: dict, tab_name: str = BUFFETT_TAB_NAME):
    client = get_gspread_client(cfg)
    sh = open_trading_hub_sheet(client, cfg)

    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        rows = max(len(df) + 10, 100)
        cols = max(len(df.columns) + 5, 20)
        ws = sh.add_worksheet(title=tab_name, rows=rows, cols=cols)

    ws.clear()

    if df.empty:
        ws.update(values=[["No qualifying CSPs found."]], range_name="A1")
        print(f"✅ Updated sheet tab '{tab_name}' (no rows).")
        return

    df_norm = normalize_df_for_sheet(df)
    header = df_norm.columns.tolist()
    values = df_norm.values.tolist()

    # New gspread signature: values first, then range_name (or named args)
    ws.update(values=[header] + values, range_name="A1")
    print(f"✅ Updated sheet tab '{tab_name}' with {len(df)} rows.")


def send_buffett_email(
    df: pd.DataFrame,
    cfg: dict,
    csv_path: str,
    universe_size: int,
):
    sheets_cfg = cfg["sheets"]
    sheet_url = sheets_cfg.get("sheet_url") or sheets_cfg["url"]

    total = len(df)
    top_n = min(15, total)

    if top_n > 0:
        top = df.head(top_n).copy()
        display_cols = [
            "ticker", "strike", "expiry",
            "underlying_price", "dte",
            "target_strike", "yield_pct",
        ]
        display_cols = [c for c in display_cols if c in top.columns]

        # Format yield as %
        if "yield_pct" in top.columns:
            top["yield_pct"] = (top["yield_pct"] * 100.0).round(2)

        html_rows = []
        for _, row in top.iterrows():
            cells = []
            for col in display_cols:
                cells.append(f"<td>{row.get(col, '')}</td>")
            html_rows.append("<tr>" + "".join(cells) + "</tr>")

        html_table_header = "".join(f"<th>{c}</th>" for c in display_cols)
        html_table = (
            "<table border='1' cellspacing='0' cellpadding='4'>"
            f"<tr>{html_table_header}</tr>"
            + "".join(html_rows)
            + "</table>"
        )
    else:
        html_table = "<p>No qualifying CSP candidates today.</p>"

    subject = "Buffett Cash-Secured Put Scan"

    html_body = f"""
    <h2>Buffett CSP Scan</h2>
    <p>
      Universe size: <b>{universe_size}</b><br/>
      Candidates found: <b>{total}</b><br/>
      Google Sheet: <a href="{sheet_url}">{sheet_url}</a><br/>
      CSV path on server: <code>{csv_path}</code>
    </p>
    <p>Top {top_n} candidates by annualized yield:</p>
    {html_table}
    <p style="font-size: 0.9em; color: #555;">
      Notes: Yield is annualized using bid / underlying / DTE * 365. 
      All strikes are at least {int(RISK_BUFFER * 100)}% below the current price, 
      with expirations between {MIN_DTE} and {MAX_DTE} days out.
    </p>
    """

    text_body = (
        f"Buffett CSP Scan\n"
        f"Universe size: {universe_size}\n"
        f"Candidates found: {total}\n"
        f"Google Sheet: {sheet_url}\n"
        f"CSV (server): {csv_path}\n"
    )

    try:
        send_email(subject=subject, html_body=html_body, text_body=text_body, cfg_path=CONFIG_PATH)
        print("✅ Buffett CSP summary email sent.")
    except Exception as e:
        print(f"⚠️ Failed to send Buffett email: {e}")


# ------------------------------
# Main
# ------------------------------

def main(cfg_path: str = CONFIG_PATH):
    print("🚀 Running Buffett Options Engine…")
    cfg = load_config(cfg_path)

    tickers = load_tickers_from_sheets(cfg)
    if not tickers:
        print("⚠️ No tickers found in Signals / Open_Positions.")
        return

    df = scan_all(tickers)

    if df.empty:
        print("⚠️ No suitable options found (after yield / DTE filters).")
        csv_path = save_to_csv(df, cfg)
        upload_df_to_sheet(df, cfg, BUFFETT_TAB_NAME)
        send_buffett_email(df, cfg, csv_path, universe_size=len(tickers))
        return

    print(f"✅ {len(df)} options found.")
    csv_path = save_to_csv(df, cfg)
    upload_df_to_sheet(df, cfg, BUFFETT_TAB_NAME)
    send_buffett_email(df, cfg, csv_path, universe_size=len(tickers))


if __name__ == "__main__":
    main()
