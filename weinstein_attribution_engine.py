#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_attribution_engine.py

Attribution Intelligence for WeinsteinAgent.

Reads the latest output/daily_parity/<run> folder and creates:
  output/attribution/<run>/
    attribution_dashboard.html
    attribution_summary.csv
    attribution_signal_counts.csv
    attribution_short_candidates.csv
    attribution_meta_recent.csv
    attribution_profile_contribution.csv
    attribution_action_recommendations.csv
    attribution_event_breakdown.csv
    attribution_inputs.json

Usage:
  python3 weinstein_attribution_engine.py
  python3 weinstein_attribution_engine.py --run-dir output/daily_parity/20260605_113227
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PARITY_ROOT = Path("output/daily_parity")
DEFAULT_OUTPUT_ROOT = Path("output/attribution")


def latest_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Parity root not found: {root}")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No parity run folders found under: {root}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def latest_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_csv(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def clean_signal_value(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().upper()
    if s in {"", "NAN", "NONE", "NULL"}:
        return ""
    return s


def contains_token(v, token: str) -> bool:
    s = clean_signal_value(v)
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return token.upper() in parts


def safe_num(v) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().replace("$", "").replace(",", "").replace("%", "")
    if not s:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def pct_change(first: float, last: float) -> float:
    if pd.isna(first) or pd.isna(last) or first == 0:
        return np.nan
    return (last - first) / first * 100.0


def fmt_money(v) -> str:
    return "—" if pd.isna(v) else f"${v:,.2f}"


def fmt_pct(v) -> str:
    return "—" if pd.isna(v) else f"{v:,.2f}%"


def html_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "<p><i>No rows.</i></p>"
    return df.head(max_rows).to_html(index=False, escape=True, border=0, classes="data-table")


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    lookup = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def build_signal_attribution(comparison: pd.DataFrame):
    if comparison.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    ticker = find_col(comparison, "Ticker", "Symbol")
    prod_latest = find_col(comparison, "PROD_Latest_Signal", "PROD_Latest")
    prod_intraday = find_col(comparison, "PROD_Intraday_Signal", "PROD_Intraday")
    sim_d = find_col(comparison, "SIM_D_Signal", "SIM_D")
    sim_f_eff = find_col(comparison, "SIM_F_EffectiveSignal", "SIM_F_Effective_Signal", "SIM_F_Effective")
    sim_f_raw = find_col(comparison, "SIM_F_RawSignal", "SIM_F_Raw_Signal", "SIM_F_Raw")

    signal_cols = [c for c in [prod_latest, prod_intraday, sim_d, sim_f_eff, sim_f_raw] if c]
    rows = []
    for col in signal_cols:
        ser = comparison[col].map(clean_signal_value)
        rows.append({
            "SignalColumn": col,
            "RowsWithSignal": int((ser != "").sum()),
            "BUY": int(ser.map(lambda x: contains_token(x, "BUY")).sum()),
            "SELL": int(ser.map(lambda x: contains_token(x, "SELL")).sum()),
            "SHORT": int(ser.map(lambda x: contains_token(x, "SHORT")).sum()),
            "OtherNonBlank": int(((ser != "") & ~ser.str.contains("BUY|SELL|SHORT", regex=True, na=False)).sum()),
        })
    counts = pd.DataFrame(rows)

    short_mask = pd.Series(False, index=comparison.index)
    for col in [sim_d, sim_f_eff, sim_f_raw]:
        if col:
            short_mask = short_mask | comparison[col].map(lambda x: contains_token(x, "SHORT"))
    short_candidates = comparison.loc[short_mask].copy()
    if ticker and not short_candidates.empty:
        cols = [ticker] + [c for c in [prod_latest, prod_intraday, sim_d, sim_f_eff, sim_f_raw] if c]
        short_candidates = short_candidates[cols]

    summary = {
        "comparison_rows": int(len(comparison)),
        "prod_latest_signals": int(comparison[prod_latest].map(clean_signal_value).ne("").sum()) if prod_latest else 0,
        "prod_intraday_signals": int(comparison[prod_intraday].map(clean_signal_value).ne("").sum()) if prod_intraday else 0,
        "sim_d_signals": int(comparison[sim_d].map(clean_signal_value).ne("").sum()) if sim_d else 0,
        "sim_f_effective_signals": int(comparison[sim_f_eff].map(clean_signal_value).ne("").sum()) if sim_f_eff else 0,
        "sim_f_raw_signals": int(comparison[sim_f_raw].map(clean_signal_value).ne("").sum()) if sim_f_raw else 0,
        "short_candidate_rows": int(len(short_candidates)),
    }
    return counts, short_candidates, summary


def build_meta_attribution(meta: pd.DataFrame):
    if meta.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    df = meta.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    df["equity_num"] = df["equity"].map(safe_num) if "equity" in df.columns else np.nan
    df["cash_num"] = df["cash"].map(safe_num) if "cash" in df.columns else np.nan
    df["equity_delta"] = df["equity_num"].diff()

    group_cols = [c for c in ["meta_profile", "meta_reason"] if c in df.columns]
    if group_cols:
        profile = (
            df.groupby(group_cols, dropna=False)
              .agg(
                  days=("equity_num", "size"),
                  avg_equity=("equity_num", "mean"),
                  total_equity_delta=("equity_delta", "sum"),
                  avg_equity_delta=("equity_delta", "mean"),
                  avg_cash=("cash_num", "mean"),
              )
              .reset_index()
              .sort_values(["total_equity_delta", "days"], ascending=[False, False])
        )
    else:
        profile = pd.DataFrame()

    eq = df["equity_num"].dropna()
    first_equity = eq.iloc[0] if len(eq) else np.nan
    last_equity = eq.iloc[-1] if len(eq) else np.nan

    summary = {
        "meta_rows": int(len(df)),
        "first_equity": float(first_equity) if pd.notna(first_equity) else np.nan,
        "last_equity": float(last_equity) if pd.notna(last_equity) else np.nan,
        "equity_change": float(last_equity - first_equity) if pd.notna(first_equity) and pd.notna(last_equity) else np.nan,
        "equity_change_pct": pct_change(first_equity, last_equity),
        "latest_meta_profile": str(df["meta_profile"].iloc[-1]) if "meta_profile" in df.columns and len(df) else "",
        "latest_meta_reason": str(df["meta_reason"].iloc[-1]) if "meta_reason" in df.columns and len(df) else "",
    }
    return df.tail(25), profile, summary


def build_action_summary(actions: pd.DataFrame):
    if actions.empty:
        return pd.DataFrame(), {"action_rows": 0}
    if "RecommendedAction" in actions.columns:
        out = actions.groupby("RecommendedAction", dropna=False).size().reset_index(name="Rows")
    elif "Signal" in actions.columns:
        out = actions.groupby("Signal", dropna=False).size().reset_index(name="Rows")
    else:
        out = pd.DataFrame({"Metric": ["Rows"], "Value": [len(actions)]})
    return out, {"action_rows": int(len(actions))}


def find_numeric_signal_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ["pnl", "profit", "return", "gain", "delta"]):
            vals = df[col].map(safe_num)
            if vals.notna().sum() > 0:
                candidates.append(col)
    return candidates


def build_event_attribution(events: pd.DataFrame):
    if events.empty:
        return pd.DataFrame(), {"event_rows": 0, "event_pnl_column": ""}

    pnl_cols = find_numeric_signal_columns(events)
    summary = {"event_rows": int(len(events)), "event_pnl_column": pnl_cols[0] if pnl_cols else ""}
    if not pnl_cols:
        return pd.DataFrame(), summary

    pnl_col = pnl_cols[0]
    df = events.copy()
    df["_pnl_num"] = df[pnl_col].map(safe_num)

    group_candidates = [
        "Signal", "signal", "Action", "action", "Stage", "stage",
        "Sector", "sector", "Industry", "industry", "meta_profile", "regime", "Regime",
    ]
    frames = []
    for g in group_candidates:
        if g in df.columns:
            tmp = (
                df.groupby(g, dropna=False)
                  .agg(rows=("_pnl_num", "size"), total_pnl=("_pnl_num", "sum"), avg_pnl=("_pnl_num", "mean"))
                  .reset_index()
                  .rename(columns={g: "Bucket"})
            )
            tmp.insert(0, "AttributionType", g)
            frames.append(tmp)
    if not frames:
        return pd.DataFrame(), summary
    return pd.concat(frames, ignore_index=True).sort_values("total_pnl", ascending=False), summary


def metric_card(label, value, klass=""):
    return f"""
    <div class="card">
      <div class="label">{html.escape(str(label))}</div>
      <div class="metric {klass}">{html.escape(str(value))}</div>
    </div>
    """


def build_html(out_path: Path, run_dir: Path, files: dict, summary: dict, signal_counts, shorts, meta_recent, profile, actions, event_attr):
    equity_change = safe_num(summary.get("equity_change"))
    equity_change_pct = safe_num(summary.get("equity_change_pct"))

    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
      .subtle { color: #6b7280; }
      .grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }
      .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; background: #fff; }
      .metric { font-size: 24px; font-weight: bold; margin-top: 4px; }
      .label { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
      .good { color: #146c2e; }
      .bad { color: #b42318; }
      .warn { color: #9a6700; }
      table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 10px 0; }
      table.data-table th { background: #f3f4f6; text-align: left; padding: 6px; border: 1px solid #e5e7eb; }
      table.data-table td { padding: 6px; border: 1px solid #e5e7eb; }
      code { background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }
    </style>
    """

    body = f"""<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Weinstein Attribution Intelligence</title>{css}</head>
<body>
<h1>Weinstein Attribution Intelligence</h1>
<div class="subtle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
<div class="subtle">Source parity run: <code>{html.escape(str(run_dir))}</code></div>

<h2>Executive Summary</h2>
<div class="grid">
  {metric_card("PROD Latest Signals", summary.get("prod_latest_signals", 0))}
  {metric_card("SIM D Signals", summary.get("sim_d_signals", 0))}
  {metric_card("SIM F Effective", summary.get("sim_f_effective_signals", 0))}
  {metric_card("SHORT Flags", summary.get("short_candidate_rows", 0), "warn" if safe_num(summary.get("short_candidate_rows", 0)) else "")}
  {metric_card("Action Rows", summary.get("action_rows", 0))}
  {metric_card("Latest META Profile", summary.get("latest_meta_profile", "—"))}
  {metric_card("Equity Change", fmt_money(equity_change), "good" if equity_change > 0 else "bad")}
  {metric_card("Equity Change %", fmt_pct(equity_change_pct), "good" if equity_change_pct > 0 else "bad")}
</div>

<h2>Interpretation</h2>
<ul>
  <li><b>Signal attribution:</b> counts SIM/PROD BUY/SELL/SHORT flags from the parity comparison.</li>
  <li><b>Short flags:</b> treated as research/validation only until short trading is explicitly enabled.</li>
  <li><b>META attribution:</b> groups equity changes by selected META profile and reason.</li>
  <li><b>Event attribution:</b> uses event-level PnL/return columns when those columns exist in SIM event files.</li>
</ul>

<h2>Input Files</h2>
<ul>
  <li>Comparison: <code>{html.escape(str(files.get("comparison") or "NONE"))}</code></li>
  <li>Actions: <code>{html.escape(str(files.get("actions") or "NONE"))}</code></li>
  <li>META decisions: <code>{html.escape(str(files.get("meta") or "NONE"))}</code></li>
  <li>SIM F events: <code>{html.escape(str(files.get("events") or "NONE"))}</code></li>
</ul>

<h2>Signal Counts</h2>{html_table(signal_counts)}
<h2>SHORT Candidate Flags</h2>{html_table(shorts)}
<h2>Action Recommendation Summary</h2>{html_table(actions)}
<h2>META Profile Contribution</h2>{html_table(profile)}
<h2>Recent META Decisions</h2>{html_table(meta_recent)}
<h2>Event Attribution</h2>{html_table(event_attr)}
</body></html>"""
    out_path.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="", help="Specific output/daily_parity/<run> folder. Defaults to latest.")
    ap.add_argument("--parity-root", default=str(DEFAULT_PARITY_ROOT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_dir(Path(args.parity_root))
    out_dir = Path(args.output_dir) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "comparison": latest_file(run_dir, "daily_prod_sim_signal_comparison_*.csv"),
        "actions": latest_file(run_dir, "daily_account_recommendations_*.csv"),
        "meta": latest_file(run_dir, "daily_meta_f_decisions_*.csv"),
        "events": latest_file(run_dir, "sim_F_effective_events.csv"),
    }

    comparison = read_csv(files["comparison"])
    actions_df = read_csv(files["actions"])
    meta = read_csv(files["meta"])
    events = read_csv(files["events"])

    signal_counts, shorts, s1 = build_signal_attribution(comparison)
    meta_recent, profile, s2 = build_meta_attribution(meta)
    action_summary, s3 = build_action_summary(actions_df)
    event_attr, s4 = build_event_attribution(events)

    summary = {}
    summary.update(s1)
    summary.update(s2)
    summary.update(s3)
    summary.update(s4)

    pd.DataFrame([summary]).to_csv(out_dir / "attribution_summary.csv", index=False)
    signal_counts.to_csv(out_dir / "attribution_signal_counts.csv", index=False)
    shorts.to_csv(out_dir / "attribution_short_candidates.csv", index=False)
    meta_recent.to_csv(out_dir / "attribution_meta_recent.csv", index=False)
    profile.to_csv(out_dir / "attribution_profile_contribution.csv", index=False)
    action_summary.to_csv(out_dir / "attribution_action_recommendations.csv", index=False)
    event_attr.to_csv(out_dir / "attribution_event_breakdown.csv", index=False)
    (out_dir / "attribution_inputs.json").write_text(json.dumps({k: str(v) if v else None for k, v in files.items()}, indent=2))

    html_path = out_dir / "attribution_dashboard.html"
    build_html(html_path, run_dir, files, summary, signal_counts, shorts, meta_recent, profile, action_summary, event_attr)

    print(f"DONE attribution: {html_path}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
