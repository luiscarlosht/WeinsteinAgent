#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_real_vs_sim_monthly.py

Compare REAL (broker) Weinstein trades vs SIM (weinstein_live_logic_backtest)
on a monthly basis.

Inputs:
  - Simulated monthly PnL CSV from weinstein_live_logic_backtest.py
  - Real trades CSV (export from broker, already filtered to Weinstein trades)

Outputs:
  - ./output/real_vs_sim_monthly.csv  (Month, real vs sim PnL & equity)
  - Printed summary table to stdout

Usage example:

  python3 weinstein_real_vs_sim_monthly.py \
    --sim-monthly ./output/live_logic_bt_monthly_20251130_191557.csv \
    --real-trades ./data/weinstein_real_trades_2023.csv \
    --initial-capital 10000 \
    --real-date-col Date \
    --real-pnl-col Realized_PnL
"""

import argparse
import os
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare REAL vs SIM Weinstein monthly performance."
    )
    p.add_argument(
        "--sim-monthly",
        required=True,
        help="Path to sim monthly CSV from weinstein_live_logic_backtest.py",
    )
    p.add_argument(
        "--real-trades",
        required=True,
        help="Path to REAL trades CSV (already filtered to Weinstein trades).",
    )
    p.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Starting equity used for both real and sim curves (default: 10000).",
    )
    p.add_argument(
        "--real-date-col",
        default="Date",
        help="Column name in REAL CSV with trade date (default: Date).",
    )
    p.add_argument(
        "--real-pnl-col",
        default="Realized_PnL",
        help="Column name in REAL CSV with realized PnL in dollars (default: Realized_PnL).",
    )
    p.add_argument(
        "--output",
        default="./output/real_vs_sim_monthly.csv",
        help="Output CSV path (default: ./output/real_vs_sim_monthly.csv)",
    )
    return p.parse_args()


def _find_column_by_substring(df: pd.DataFrame, substr: str) -> str:
    """
    Helper: find first column whose name contains `substr` (case-insensitive).
    Raises if not found.
    """
    target = substr.lower()
    for col in df.columns:
        if target in col.lower():
            return col
    raise KeyError(f"No column containing '{substr}' found in columns: {list(df.columns)}")


def _clean_money_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series with money-like strings (e.g. '$-1,159.82') or floats
    into float dollars.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    # assume string-like
    return (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", "0")
        .astype(float)
    )


def load_sim_monthly(sim_path: str) -> pd.DataFrame:
    """
    Load sim monthly CSV and return DataFrame with:
       index = MonthKey (timestamp at month start)
       column 'Sim_PnL'
    """
    sim_df = pd.read_csv(sim_path)

    # Guess month & PnL columns by name
    try:
        month_col = _find_column_by_substring(sim_df, "month")
    except KeyError:
        # fallback to first column
        month_col = sim_df.columns[0]

    try:
        pnl_col = _find_column_by_substring(sim_df, "pnl")
    except KeyError:
        raise KeyError(
            "Could not locate a PnL column in sim CSV. "
            "Make sure a column name contains 'PnL' (case-insensitive)."
        )

    sim_df[month_col] = pd.to_datetime(sim_df[month_col])
    sim_df["MonthKey"] = sim_df[month_col].dt.to_period("M").dt.to_timestamp()

    sim_df[pnl_col] = _clean_money_series(sim_df[pnl_col])

    sim_monthly = (
        sim_df.groupby("MonthKey")[pnl_col]
        .sum()
        .to_frame(name="Sim_PnL")
        .sort_index()
    )
    return sim_monthly


def load_real_monthly(
    real_path: str,
    date_col: str,
    pnl_col: str,
) -> pd.DataFrame:
    """
    Load REAL trades CSV and aggregate to monthly PnL.

    Assumes:
      - date_col: trade date
      - pnl_col: realized PnL in dollars for each trade or fill

    Returns DataFrame with:
      index = MonthKey (timestamp at month start)
      column 'Real_PnL'
    """
    real_df = pd.read_csv(real_path)

    if date_col not in real_df.columns:
        raise KeyError(
            f"REAL date column '{date_col}' not found in columns: {list(real_df.columns)}"
        )
    if pnl_col not in real_df.columns:
        raise KeyError(
            f"REAL PnL column '{pnl_col}' not found in columns: {list(real_df.columns)}"
        )

    real_df[date_col] = pd.to_datetime(real_df[date_col], errors="coerce")
    real_df = real_df.dropna(subset=[date_col]).copy()

    real_df["MonthKey"] = real_df[date_col].dt.to_period("M").dt.to_timestamp()
    real_df[pnl_col] = _clean_money_series(real_df[pnl_col])

    real_monthly = (
        real_df.groupby("MonthKey")[pnl_col]
        .sum()
        .to_frame(name="Real_PnL")
        .sort_index()
    )
    return real_monthly


def build_equity_curve(pnl_series: pd.Series, initial_capital: float) -> pd.Series:
    """
    Given a monthly PnL series (indexed by MonthKey), build an equity curve.
    """
    equity = []
    eq = initial_capital
    for _idx, pnl in pnl_series.items():
        eq += pnl
        equity.append(eq)
    return pd.Series(equity, index=pnl_series.index, name="Equity")


def compare_real_vs_sim(
    sim_monthly: pd.DataFrame,
    real_monthly: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Combine sim + real monthly PnL, build equity curves and return a single DataFrame.
    """
    combined = sim_monthly.join(real_monthly, how="outer").fillna(0.0).sort_index()

    combined["Sim_Equity"] = build_equity_curve(combined["Sim_PnL"], initial_capital)
    combined["Real_Equity"] = build_equity_curve(combined["Real_PnL"], initial_capital)

    combined["Sim_ReturnPct"] = (combined["Sim_PnL"] / initial_capital) * 100.0
    combined["Real_ReturnPct"] = (combined["Real_PnL"] / initial_capital) * 100.0
    combined["Diff_PnL"] = combined["Real_PnL"] - combined["Sim_PnL"]
    combined["Diff_ReturnPct"] = combined["Real_ReturnPct"] - combined["Sim_ReturnPct"]

    # Make Month column explicit for readability/output
    combined = combined.reset_index().rename(columns={"MonthKey": "Month"})
    return combined


def print_summary(df: pd.DataFrame, initial_capital: float) -> None:
    """
    Print a concise textual summary of real vs sim performance.
    """
    if df.empty:
        print("No overlapping months to compare.")
        return

    df_local = df.copy()
    df_local["Month"] = df_local["Month"].dt.strftime("%Y-%m")

    cols = [
        "Month",
        "Real_PnL",
        "Sim_PnL",
        "Diff_PnL",
        "Real_ReturnPct",
        "Sim_ReturnPct",
        "Diff_ReturnPct",
    ]

    print("\n=== Monthly REAL vs SIM (PnL in $; Returns % of initial capital) ===")
    print(df_local[cols].to_string(index=False, formatters={
        "Real_PnL": lambda x: f"{x:9.2f}",
        "Sim_PnL": lambda x: f"{x:9.2f}",
        "Diff_PnL": lambda x: f"{x:9.2f}",
        "Real_ReturnPct": lambda x: f"{x:7.2f}",
        "Sim_ReturnPct": lambda x: f"{x:7.2f}",
        "Diff_ReturnPct": lambda x: f"{x:7.2f}",
    }))

    total_real = df["Real_PnL"].sum()
    total_sim = df["Sim_PnL"].sum()

    print("\n=== Totals over all months ===")
    print(f"Initial capital : ${initial_capital:,.2f}")
    print(f"REAL total PnL  : ${total_real:,.2f} ({total_real/initial_capital*100:,.2f}%)")
    print(f"SIM  total PnL  : ${total_sim:,.2f} ({total_sim/initial_capital*100:,.2f}%)")
    print(f"REAL - SIM diff : ${(total_real-total_sim):,.2f} "
          f"({(total_real-total_sim)/initial_capital*100:,.2f}%)")


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    sim_monthly = load_sim_monthly(args.sim_monthly)
    real_monthly = load_real_monthly(
        args.real_trades,
        date_col=args.real_date_col,
        pnl_col=args.real_pnl_col,
    )

    combined = compare_real_vs_sim(sim_monthly, real_monthly, args.initial_capital)
    combined.to_csv(args.output, index=False)

    print(f"\n✅ Wrote REAL vs SIM monthly comparison → {args.output}")
    print_summary(combined, args.initial_capital)


if __name__ == "__main__":
    main()
