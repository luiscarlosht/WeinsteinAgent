# buffett_options_engine.py
#
# Buffett-style Cash Secured Put (CSP) scanner for a "safe / moderate" universe.
#
# Features:
# - Automatically loads tickers from Google Sheets (Signals + Open_Positions)
#   using config.yaml (see GOOGLE SHEETS CONFIG section below).
# - Applies Option B logic:
#     * Buffett-core quality tickers
#     * plus moderate-quality names you own / track
#     * excludes crypto, ETFs, penny/microcaps & very speculative names
# - Scans puts 7–45 DTE, 10% below current price, and filters by minimum yield.
#
# Output:
# - Writes ./output/Buffett_Put_Signals_YYYYMMDD_HHMMSS.csv

import os
from datetime import datetime
from typing import List, Set

import numpy as np
import pandas as pd
import yfinance as yf

# Optional imports for Google Sheets
try:
    import yaml
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_SHEETS = True
except ImportError:
    HAS_SHEETS = False


# ============================
#   CORE PARAMETERS
# ============================

# Risk: how far below current price we sell the put
RISK_BUFFER = 0.10  # 10% below current price

# Minimum annualized yield (premium / current_price, annualized by DTE)
MIN_YIELD = 0.008  # 0.8% annualized minimum

# Max days to expiration
MAX_DTE = 45
MIN_DTE = 7

# Default (fallback) Option B universe
# If Google Sheets loading fails, we use this list.
BUFFETT_UNIVERSE_DEFAULT: List[str] = [
    # Buffett-core style names
    "AAPL", "HCA", "APH", "PAYX", "OTIS",
    # Moderately safe, big cap / quality
    "AMD", "ANET", "NET", "META", "TSM", "CRM", "MS",
    # Weinstein longs (moderate risk)
    "ALB", "EME", "F", "GM",
    # Additional quality pick
    "CARR",
]

# Names that we NEVER want to run CSP on.
BANNED_TICKERS: Set[str] = {
    # Crypto / no options or extreme volatility
    "ETH-USD", "SOL-USD",
    # ETFs, not ideal for CSP in this engine
    "QQQM", "VOO", "VUG", "SOXX",
    # Microcaps / penny / speculative
    "BITF", "FRMI", "IONQ", "CLSK", "CORZ", "CRCL", "HOOD", "UUUU",
    # Extremely volatile or commodity-sensitive
    "LAC", "SMCI",
}


# ============================
#   GOOGLE SHEETS CONFIG
# ============================

# We assume a config.yaml in the repo root, with something like:
#
# google_sheets:
#   spreadsheet_id: "your-spreadsheet-id"
#   service_account_file: "service_account.json"
#   signals_tab: "Signals"
#   open_positions_tab: "Open_Positions"
#
# If any of this is missing or gspread/yaml are not installed,
# we fall back to BUFFETT_UNIVERSE_DEFAULT.


def _load_config(path: str = "config.yaml") -> dict:
    if not HAS_SHEETS:
        return {}
    if not os.path.exists(path):
        print(f"⚠️ config.yaml not found at {path}, using fallback universe.")
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Failed to read {path}: {e}")
        return {}


def _get_sheets_client(cfg: dict):
    if not HAS_SHEETS:
        return None

    gs_cfg = (cfg or {}).get("google_sheets") or {}
    creds_file = gs_cfg.get("service_account_file")
    if not creds_file:
        print("⚠️ google_sheets.service_account_file missing in config.yaml")
        return None
    if not os.path.exists(creds_file):
        print(f"⚠️ Service account file not found: {creds_file}")
        return None

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    try:
        creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"⚠️ Failed to authorize Google Sheets client: {e}")
        return None


def _load_tickers_from_tab(client, spreadsheet_id: str, tab_name: str) -> Set[str]:
    """Load a set of tickers from a given worksheet/tab."""
    try:
        sh = client.open_by_key(spreadsheet_id)
        ws = sh.worksheet(tab_name)
        rows = ws.get_all_records()
        if not rows:
            return set()
        df = pd.DataFrame(rows)

        # Try to find the symbol column.
        symbol_col_candidates = ["Symbol", "Ticker", "symbol", "ticker"]
        symbol_col = None
        for col in symbol_col_candidates:
            if col in df.columns:
                symbol_col = col
                break
        if symbol_col is None:
            print(f"⚠️ No Symbol/Ticker column in tab '{tab_name}'")
            return set()

        tickers = (
            df[symbol_col]
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
        )

        # Filter out obviously invalid or synthetic symbols (options, CUSIPs etc.)
        # Option-style and weird synthetic symbols tend to have digits + letters + hyphens etc.
        # We'll allow standard stock tickers: letters + dots + dashes.
        tickers = tickers[tickers.str.match(r"^[A-Za-z.\-]+$")]

        # If there's a "Side" / "Direction" column, keep only long if present.
        side_col = None
        for cand in ["Side", "side", "Direction", "direction", "Position", "position"]:
            if cand in df.columns:
                side_col = cand
                break
        if side_col is not None:
            side_series = df[side_col].astype(str).str.lower()
            long_mask = side_series.isin(["long", "buy", "bought"])
            tickers = tickers[long_mask]

        # Final clean + dedupe
        out = {t.upper() for t in tickers}
        return out

    except Exception as e:
        print(f"⚠️ Failed to load tickers from tab '{tab_name}': {e}")
        return set()


def load_universe_from_google_sheets() -> List[str]:
    """Load Buffett CSP universe dynamically from Google Sheets."""
    if not HAS_SHEETS:
        print("ℹ️ gspread/yaml not installed; using fallback universe.")
        return BUFFETT_UNIVERSE_DEFAULT

    cfg = _load_config()
    gs_cfg = cfg.get("google_sheets") or {}
    spreadsheet_id = gs_cfg.get("spreadsheet_id")
    if not spreadsheet_id:
        print("⚠️ google_sheets.spreadsheet_id missing in config.yaml, using fallback.")
        return BUFFETT_UNIVERSE_DEFAULT

    client = _get_sheets_client(cfg)
    if client is None:
        print("⚠️ Could not create Sheets client, using fallback universe.")
        return BUFFETT_UNIVERSE_DEFAULT

    signals_tab = gs_cfg.get("signals_tab", "Signals")
    open_pos_tab = gs_cfg.get("open_positions_tab", "Open_Positions")

    tickers: Set[str] = set()

    # Load from Signals
    print(f"🔍 Loading tickers from tab '{signals_tab}'...")
    tickers |= _load_tickers_from_tab(client, spreadsheet_id, signals_tab)

    # Load from Open_Positions (if exists)
    if open_pos_tab:
        print(f"🔍 Loading tickers from tab '{open_pos_tab}'...")
        tickers |= _load_tickers_from_tab(client, spreadsheet_id, open_pos_tab)

    # Always include our default Option B universe as a baseline
    tickers |= set(BUFFETT_UNIVERSE_DEFAULT)

    # Remove banned names
    tickers -= BANNED_TICKERS

    if not tickers:
        print("⚠️ No tickers found from Sheets; falling back to default universe.")
        return BUFFETT_UNIVERSE_DEFAULT

    final_list = sorted(tickers)
    print(f"✅ Using {len(final_list)} tickers in Buffett CSP universe.")
    return final_list


# ============================
#   OPTION CHAIN SCANNING
# ============================

def get_option_chain(ticker: str, max_dte: int = MAX_DTE) -> pd.DataFrame:
    """Fetch put options for a ticker and compute yield for candidates."""
    stock = yf.Ticker(ticker)
    try:
        expiry_dates = stock.options
    except Exception:
        return pd.DataFrame()

    today = datetime.utcnow().date()
    valid_dates = []
    for d in expiry_dates:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (dt - today).days
        if MIN_DTE <= dte <= max_dte:
            valid_dates.append(d)

    if not valid_dates:
        return pd.DataFrame()

    results = []

    # Get last close price; use iloc to avoid FutureWarning
    try:
        hist = stock.history(period="1d")
        if hist.empty:
            return pd.DataFrame()
        current_price = float(hist["Close"].iloc[-1])
    except Exception:
        return pd.DataFrame()

    buffer_price = round(current_price * (1 - RISK_BUFFER), 2)

    for expiry in valid_dates:
        try:
            opt = stock.option_chain(expiry).puts
        except Exception:
            continue

        # Only consider strikes at or below our buffer price
        candidates = opt[opt["strike"] <= buffer_price].copy()
        if candidates.empty:
            continue

        dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (dt - today).days

        candidates["ticker"] = ticker
        candidates["underlying_price"] = current_price
        candidates["dte"] = dte
        candidates["target_strike"] = buffer_price
        candidates["expiry"] = expiry

        # Use bid to be conservative; avoid division by zero
        candidates["yield_pct"] = (
            candidates["bid"] / current_price / candidates["dte"] * 365
        ).replace([np.inf, -np.inf], np.nan)

        results.append(candidates)

    if results:
        df = pd.concat(results, ignore_index=True)
        # Filter out NaN yields
        df = df.dropna(subset=["yield_pct"])
        return df

    return pd.DataFrame()


def scan_all() -> pd.DataFrame:
    universe = load_universe_from_google_sheets()
    all_results = []

    for tkr in universe:
        print(f"Scanning {tkr}...")
        df = get_option_chain(tkr)
        if df.empty:
            continue

        # Apply minimum yield filter
        df = df[df["yield_pct"] >= MIN_YIELD]
        if df.empty:
            continue

        all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined.sort_values(["yield_pct"], ascending=False, inplace=True)
    return combined


# ============================
#   OUTPUT
# ============================

def save_to_csv(df: pd.DataFrame):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    fpath = f"./output/Buffett_Put_Signals_{now}.csv"
    df.to_csv(fpath, index=False)
    print(f"✅ Saved: {fpath}")


# ============================
#   MAIN
# ============================

if __name__ == "__main__":
    print("🚀 Running Buffett Options Engine…")
    df = scan_all()
    if df.empty:
        print("⚠️ No suitable options found.")
    else:
        print(f"✅ {len(df)} options found.")
        save_to_csv(df)
