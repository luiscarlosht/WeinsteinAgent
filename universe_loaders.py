# universe_loaders.py
import io
import re
import time
import urllib.request
import pandas as pd

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Some tickers need mapping for Yahoo-style symbols (e.g., BRK.B -> BRK-B)
YF_REPLACEMENTS = {
    r"\.": "-",   # any dot -> dash (covers BRK.B, BF.B etc)
}

def _to_yahoo_symbol(t: str) -> str:
    s = t.strip().upper()
    for pat, rep in YF_REPLACEMENTS.items():
        s = re.sub(pat, rep, s)
    return s

def load_sp500_from_wikipedia(max_retries: int = 3, sleep_secs: float = 1.0) -> list[str]:
    """
    Scrapes the S&P 500 constituents table from Wikipedia and returns
    Yahoo-compatible tickers.
    """
    last_err = None
    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(WIKI_SP500_URL, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            dfs = pd.read_html(io.StringIO(html))
            # Find the table with 'Symbol' column
            table = None
            for df in dfs:
                cols = [c.lower() for c in df.columns.astype(str)]
                if any("symbol" in c for c in cols):
                    table = df
                    break
            if table is None:
                raise RuntimeError("Could not find S&P 500 table with a 'Symbol' column.")
            syms = table.iloc[:, 0].astype(str).tolist()
            syms = [_to_yahoo_symbol(t) for t in syms if isinstance(t, str) and t.strip()]
            # Dedup while preserving order
            seen = set()
            uniq = []
            for s in syms:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            return uniq
        except Exception as e:
            last_err = e
            time.sleep(sleep_secs)
    raise RuntimeError(f"Failed to load S&P 500 from Wikipedia: {last_err}")

def combine_universe(sp500: bool, extra_symbols: list[str] | None) -> list[str]:
    base = []
    if sp500:
        base = load_sp500_from_wikipedia()

    # Normalize & map extras
    extras = []
    for t in (extra_symbols or []):
        if not t:
            continue
        extras.append(_to_yahoo_symbol(str(t)))

    # Merge, de-dup
    seen = set()
    merged = []
    for s in base + extras:
        if s not in seen:
            seen.add(s)
            merged.append(s)

    # Optional: basic hygiene filters (drop clearly broken names)
    merged = [m for m in merged if re.match(r"^[A-Z0-9\-]+$", m)]
    return merged
