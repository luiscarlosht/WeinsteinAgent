#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real vs Sim Comparison — Fidelity positions (Option A) vs Weinstein backtest

Goal (v1):
    - Compare your simulated equity (starting capital + realized P/L from backtest)
      against your real Fidelity portfolio value from a positions CSV export.

What this script DOES:
    - Reads a backtest trade log CSV from weinstein_live_logic_backtest.py
      (assumes there is a numeric 'pnl' column and a close/exit date column).
    - Sums realized P/L up to a given "as-of" date.
    - Computes sim_equity = starting_capital + realized_pnl_to_date.
    - Reads a Fidelity positions CSV (daily export, "Option A").
    - Detects Symbol column and Market/Current Value column heuristically.
    - Computes real_equity = sum of current value of all positions.
    - Prints a clear comparison and a small breakdown of your real positions.

What this script does NOT do (yet):
    - Rebuild exact open positions from sim and mark them to market.
    - Match every single trade 1:1 (ticker+entry+exit) against your real trade history.
    - Compute real P/L (needs two snapshots or actual trade history).

Usage example:

    # Example: compare up to 2025-10-31 using a backtest trades CSV and a Fidelity CSV
    python3 real_vs_sim_compare.py \
        --sim-trades ./output/live_logic_bt_trades_20251130_183154.csv \
        --real-positions ./Portfolio_Positions_2025-10-31.csv \
        --capital 10000 \
        --start 2025-01-01 \
        --as-of 2025-10-31

Assumptions:
    - Backtest CSV has at least:
        * 'pnl'       (realized P/L per trade)
        * an exit/close date column; script will try:
              'exit_date', 'close_date', 'date', 'exit', 'closed_at'
          The date must be parseable by pandas.to_datetime.
    - Fidelity CSV has:
        * One "Symbol" column (Symbol, SYMBOL, symbol, etc.).
        * One "value" column, typically named like:
              "Current Value", "Market Value", "Total Value", etc.
"""

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# --------------- Small helpers ---------------

def _log(msg: str):
    print(msg, flush=True)


def _detect_date_column(df: pd.DataFrame) -> str:
    """
    Try to detect a date column in the backtest trades CSV.
    Preference order is tuned for weinstein_live_logic_backtest.py.
    """
    candidates_ordered = [
        "exit_date",
        "close_date",
        "date",
        "exit",
        "closed_at",
        "timestamp",
    ]

    cols_lower = {c.lower(): c for c in df.columns}

    # Step 1: direct name match by preference
    for name in candidates_ordered:
        if name in cols_lower:
            return cols_lower[name]

    # Step 2: any column whose name contains "date"
    for c in df.columns:
        if "date" in c.lower():
            return c

    raise ValueError(
        f"Could not detect a date/exit column in sim trades CSV. "
        f"Available columns: {list(df.columns)}"
    )


def _detect_pnl_column(df: pd.DataFrame) -> str:
    """
    Detect the PnL column in the backtest trades CSV.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for name in ["pnl", "p&l", "profit", "pl"]:
        if name in cols_lower:
            return cols_lower[name]

    # As a fallback, look for any column with "pnl" or "profit" in the name.
    for c in df.columns:
        cl = c.lower()
        if "pnl" in cl or "profit" in cl or "p&l" in cl:
            return c

    raise ValueError(
        f"Could not detect a PnL column in sim trades CSV. "
        f"Available columns: {list(df.columns)}"
    )


def _detect_symbol_column(df: pd.DataFrame) -> str:
    """
    Detect the symbol column in the Fidelity CSV.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    # Common names
    for name in ["symbol", "ticker", "security", "security symbol"]:
        if name in cols_lower:
            return cols_lower[name]

    # Fallback: any column that looks like "symbol"
    for c in df.columns:
        if "symbol" in c.lower() or "ticker" in c.lower():
            return c

    raise ValueError(
        f"Could not detect a Symbol column in Fidelity CSV. "
        f"Available columns: {list(df.columns)}"
    )


def _detect_value_column(df: pd.DataFrame) -> str:
    """
    Detect the "current value" / "market value" column in Fidelity CSV.

    We'll favor columns that include 'value' or 'market' but not 'cost'.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    preferred_keywords = [
        "current value",
        "market value",
        "total value",
        "value"
    ]

    # Step 1: exact-ish matches
    for key in preferred_keywords:
        if key in cols_lower:
            return cols_lower[key]

    # Step 2: heuristics: any col with 'value' OR 'market' but NOT 'cost'
    candidates = []
    for c in df.columns:
        cl = c.lower()
        if ("value" in cl or "market" in cl) and ("cost" not in cl):
            candidates.append(c)

    if candidates:
        # If multiple, pick the first
        return candidates[0]

    raise ValueError(
        f"Could not detect a market/current value column in Fidelity CSV. "
        f"Available columns: {list(df.columns)}"
    )


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


# --------------- Core comparison logic ---------------

def compute_sim_equity(sim_trades_path: str, capital: float, start: str, as_of: str):
    """
    Read the backtest trades CSV and compute:

        realized_pnl = sum of PnL for trades whose exit/close date is in [start, as_of]
        sim_equity   = capital + realized_pnl

    Returns (sim_equity, realized_pnl, n_trades_used).
    """

    _log(f"• Loading sim trades from: {sim_trades_path}")
    df = pd.read_csv(sim_trades_path)

    if df.empty:
        _log("⚠️ Sim trades CSV is empty. Assuming no trades (PnL=0).")
        return capital, 0.0, 0

    date_col = _detect_date_column(df)
    pnl_col = _detect_pnl_column(df)

    _log(f"  → Using date column: {date_col}")
    _log(f"  → Using PnL  column: {pnl_col}")

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    start_dt = _parse_date(start)
    as_of_dt = _parse_date(as_of)

    mask = (df[date_col] >= start_dt) & (df[date_col] <= as_of_dt)
    used = df.loc[mask].copy()

    if used.empty:
        _log("⚠️ No trades in the sim within the requested period. PnL=0.")
        return capital, 0.0, 0

    used[pnl_col] = pd.to_numeric(used[pnl_col], errors="coerce").fillna(0.0)

    realized_pnl = float(used[pnl_col].sum())
    sim_equity = capital + realized_pnl

    _log(f"  → Trades in period: {len(used)}")
    _log(f"  → Realized PnL in period: ${realized_pnl:,.2f}")
    _log(f"  → Simulated equity (capital + realized PnL): ${sim_equity:,.2f}")

    return sim_equity, realized_pnl, len(used)


def compute_real_equity(fidelity_positions_path: str):
    """
    Read Fidelity daily export and compute:

        real_equity = sum of current/market value of all positions

    Also returns a small positions DataFrame for breakdown.
    """
    _log(f"• Loading Fidelity positions from: {fidelity_positions_path}")
    df = pd.read_csv(fidelity_positions_path)

    if df.empty:
        raise ValueError("Fidelity positions CSV is empty — no rows found.")

    sym_col = _detect_symbol_column(df)
    val_col = _detect_value_column(df)

    _log(f"  → Using Symbol column: {sym_col}")
    _log(f"  → Using Value  column: {val_col}")

    # Clean up value column: remove $ and commas, coerce to float
    vals = (
        df[val_col]
        .astype(str)
        .str.replace("[,$]", "", regex=True)
        .replace({"": np.nan})
    )
    vals = pd.to_numeric(vals, errors="coerce").fillna(0.0)

    df["_clean_value"] = vals
    real_equity = float(vals.sum())

    _log(f"  → Number of positions (rows): {len(df)}")
    _log(f"  → Real equity (sum of {val_col}): ${real_equity:,.2f}")

    positions = df[[sym_col, "_clean_value"]].rename(
        columns={sym_col: "Symbol", "_clean_value": "Value"}
    )

    return real_equity, positions


def compare_real_vs_sim(sim_trades_path: str,
                        fidelity_positions_path: str,
                        capital: float,
                        start: str,
                        as_of: str):
    """
    High-level orchestrator:
        - compute sim equity
        - compute real equity
        - print comparison + small breakdown
    """

    _log("======================================")
    _log(" REAL vs SIM — Equity Comparison v1 ")
    _log("======================================")
    _log(f"Period: {start} → {as_of}")
    _log(f"Starting capital (sim): ${capital:,.2f}")
    _log("")

    sim_equity, sim_pnl, n_trades = compute_sim_equity(
        sim_trades_path=sim_trades_path,
        capital=capital,
        start=start,
        as_of=as_of,
    )

    _log("")
    real_equity, positions = compute_real_equity(fidelity_positions_path)

    _log("")
    _log("---------- SUMMARY ----------")
    _log(f"Simulated equity (closed trades only): ${sim_equity:,.2f}")
    _log(f"Real Fidelity equity (snapshot):       ${real_equity:,.2f}")

    diff = real_equity - sim_equity
    pct = (diff / sim_equity * 100.0) if sim_equity != 0 else float("nan")

    _log("")
    _log(f"Difference (Real - Sim): ${diff:,.2f} ({pct:+.2f}%)")
    _log(f"Trades counted in sim PnL: {n_trades}")
    _log("")
    _log("NOTE:")
    _log("  • Sim equity = capital + SUM(closed trade PnL) in [start, as_of].")
    _log("  • Open positions in the sim are NOT marked-to-market here (v1).")
    _log("  • Real equity is snapshot-based, including all open positions.")
    _log("  → Some gap is expected due to:")
    _log("       - different timing of entries/exits,")
    _log("       - open trades,")
    _log("       - execution price slippage,")
    _log("       - any manual trades not driven by the system.\n")

    # Show top 10 real positions by value
    _log("Top 10 real positions by Value (Fidelity snapshot):")
    top = positions.sort_values("Value", ascending=False).head(10)
    for _, row in top.iterrows():
        _log(f"  {row['Symbol']:6s}  ${row['Value']:>12,.2f}")

    _log("\nDone.\n")


# --------------- CLI ---------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Weinstein backtest simulated equity vs Fidelity positions snapshot."
    )
    parser.add_argument(
        "--sim-trades",
        required=True,
        help="Path to backtest trades CSV (e.g. ./output/live_logic_bt_trades_20251130_183154.csv)",
    )
    parser.add_argument(
        "--real-positions",
        required=True,
        help="Path to Fidelity positions CSV export (Option A).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        required=True,
        help="Starting capital used in the sim (e.g. 10000).",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Sim start date (YYYY-MM-DD), e.g. 2025-01-01.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        required=True,
        help="Comparison 'as-of' date (YYYY-MM-DD), should match or be close to the Fidelity snapshot date.",
    )

    args = parser.parse_args(argv)

    try:
        compare_real_vs_sim(
            sim_trades_path=args.sim_trades,
            fidelity_positions_path=args.real_positions,
            capital=args.capital,
            start=args.start,
            as_of=args.as_of,
        )
    except Exception as e:
        _log(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
