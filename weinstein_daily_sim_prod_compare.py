#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily PROD vs SIM parity report.

Compares:
- PROD intraday diagnostics: output/intraday_debug.csv
- SIM D replay events
- SIM F effective replay events selected by the latest F meta decision
- Fidelity account positions/profile map

Outputs:
- daily_prod_sim_signal_comparison.csv
- daily_account_recommendations.csv
- daily_meta_f_decisions.csv when available
- daily_prod_sim_summary.html
- optional Google Sheet tabs
- optional email summary

This is a comparison/audit layer only. It does not change Weinstein CORE logic.
"""

from __future__ import annotations

import argparse
import os
import html
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from weinstein_prod_history import read_prod_history_for_date, summarize_prod_history

from weinstein_account_profiles import (
    load_profiles,
    read_fidelity_positions,
    normalize_positions,
    attach_profiles,
)

SIGNALS = {"BUY", "NEAR", "NEAR_BUY", "NEAR-TRIGGER", "SELL", "SELLTRIG", "SELL-TRIGGER", "SELL-WATCH", "SHORT"}


def _norm_signal(x: object) -> str:
    s = str(x or "").strip().upper()
    if s in {"NEAR_BUY", "NEAR-TRIGGER"}:
        return "NEAR"
    if s in {"SELLTRIG", "SELL-TRIGGER", "SELL-WATCH"}:
        return "SELL"
    if s == "SHORT":
        return "SHORT"
    if s in {"BUY", "NEAR", "SELL"}:
        return s
    return s


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _latest_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    date_col = None
    for c in ["date", "Date", "TimestampUTC", "timestamp", "RunDate"]:
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        return df
    dts = pd.to_datetime(df[date_col], errors="coerce")
    if dts.dropna().empty:
        return df
    latest = dts.max().date()
    out = df.loc[dts.dt.date.eq(latest)].copy()
    return out if not out.empty else df


def normalize_prod(prod: pd.DataFrame) -> pd.DataFrame:
    if prod.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])
    out = prod.copy()
    if "Ticker" not in out.columns and "ticker" in out.columns:
        out["Ticker"] = out["ticker"]
    if "Signal" not in out.columns and "signal" in out.columns:
        out["Signal"] = out["signal"]
    if "Price" not in out.columns:
        for c in ["PriceNow", "price", "Close", "close"]:
            if c in out.columns:
                out["Price"] = out[c]
                break
    if "Reason" not in out.columns:
        for c in ["Reason", "reason", "Details", "detail"]:
            if c in out.columns:
                out["Reason"] = out[c]
                break
    out["Ticker"] = out.get("Ticker", "").astype(str).str.upper().str.strip()
    out["Signal"] = out.get("Signal", "").apply(_norm_signal)
    out["Source"] = "PROD"
    out = out[out["Signal"].isin({"BUY", "NEAR", "SELL", "SHORT"})]
    return out[["Ticker", "Signal", "Price", "Reason", "Source"]].drop_duplicates()


def normalize_sim(sim: pd.DataFrame, source: str) -> pd.DataFrame:
    if sim.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])
    out = _latest_date_filter(sim.copy())
    if "Ticker" not in out.columns:
        for c in ["ticker", "symbol", "Symbol"]:
            if c in out.columns:
                out["Ticker"] = out[c]
                break
    if "Signal" not in out.columns:
        for c in ["signal", "Signal"]:
            if c in out.columns:
                out["Signal"] = out[c]
                break
    if "Price" not in out.columns:
        for c in ["price", "Price", "PriceNow", "close", "Close"]:
            if c in out.columns:
                out["Price"] = out[c]
                break
    if "Reason" not in out.columns:
        for c in ["reason", "Reason", "detail", "Details"]:
            if c in out.columns:
                out["Reason"] = out[c]
                break
    out["Ticker"] = out.get("Ticker", "").astype(str).str.upper().str.strip()
    out["Signal"] = out.get("Signal", "").apply(_norm_signal)
    out["Source"] = source
    out = out[out["Signal"].isin({"BUY", "NEAR", "SELL", "SHORT"})]
    return out[["Ticker", "Signal", "Price", "Reason", "Source"]].drop_duplicates()


def latest_meta_profile(meta: pd.DataFrame) -> str:
    if meta.empty or "meta_profile" not in meta.columns:
        return ""
    out = meta.copy()
    if "date" in out.columns:
        out["_dt"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("_dt")
    return str(out["meta_profile"].iloc[-1]).strip().upper()


def effective_f_signals(sim_d: pd.DataFrame, sim_e: pd.DataFrame, sim_f_raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Select the effective F shadow stream for the latest META profile."""
    profile = latest_meta_profile(meta)
    if profile == "A":
        out = sim_f_raw[sim_f_raw["Signal"].ne("SHORT")].copy()
    elif profile == "D":
        out = sim_d.copy()
    elif profile == "E":
        out = sim_e.copy()
    else:
        # B and unknown profiles consume the broad both-sides stream.
        out = sim_f_raw.copy()
    out["F_MetaProfile"] = profile or "UNKNOWN"
    return out


def compare_signals(prod: pd.DataFrame, sim_d: pd.DataFrame, sim_f: pd.DataFrame, prod_history: pd.DataFrame | None = None, sim_f_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    prod_history = prod_history if prod_history is not None else pd.DataFrame(columns=["Ticker", "Signal"])
    sim_f_raw = sim_f_raw if sim_f_raw is not None else sim_f
    keys = sorted(set(prod["Ticker"]) | set(prod_history["Ticker"]) | set(sim_d["Ticker"]) | set(sim_f["Ticker"]) | set(sim_f_raw["Ticker"]))
    rows = []
    for t in keys:
        p = ",".join(sorted(prod.loc[prod["Ticker"].eq(t), "Signal"].unique()))
        ph = ",".join(sorted(prod_history.loc[prod_history["Ticker"].eq(t), "Signal"].unique())) if not prod_history.empty else ""
        d = ",".join(sorted(sim_d.loc[sim_d["Ticker"].eq(t), "Signal"].unique()))
        f = ",".join(sorted(sim_f.loc[sim_f["Ticker"].eq(t), "Signal"].unique()))
        f_raw = ",".join(sorted(sim_f_raw.loc[sim_f_raw["Ticker"].eq(t), "Signal"].unique()))
        rows.append({
            "Ticker": t,
            "PROD_Latest_Signal": p,
            "PROD_Intraday_Signal": ph,
            "SIM_D_Signal": d,
            "SIM_F_EffectiveSignal": f,
            "SIM_F_RawSignal": f_raw,
            "PROD_Latest_vs_D_Match": bool(p and d and p == d),
            "PROD_Latest_vs_F_Match": bool(p and f and p == f),
            "PROD_Intraday_vs_D_Match": bool(ph and d and ph == d),
            "PROD_Intraday_vs_F_Match": bool(ph and f and ph == f),
            "In_PROD_Latest": bool(p),
            "In_PROD_Intraday": bool(ph),
            "In_SIM_D": bool(d),
            "In_SIM_F": bool(f),
        })
    return pd.DataFrame(rows)


def account_recommendations(sim_d: pd.DataFrame, sim_f: pd.DataFrame, positions: pd.DataFrame, profile_cfg: dict) -> pd.DataFrame:
    """Build account-level recommendations.

    Operational filtering:
    - BUY/NEAR are shown as candidates for the account profile.
    - SELL is shown only when the account owns the ticker.
    - SHORT is shown as a candidate, but can be ignored if the account does not trade shorts.
    """
    accounts = profile_cfg.get("accounts", []) or []
    rows = []
    owned = positions[~positions.get("IsCash", False)].copy() if not positions.empty else pd.DataFrame()

    for acct in accounts:
        acct_num = str(acct.get("account_number", "")).strip()
        profile = str(acct.get("profile", "")).strip().upper()
        label = acct.get("label", "")
        events = sim_f if profile == "F" else sim_d if profile == "D" else pd.DataFrame()

        acct_owned = owned[owned["Account Number"].astype(str).eq(acct_num)].copy() if not owned.empty else pd.DataFrame()
        owned_tickers = set(acct_owned["Symbol"].astype(str).str.upper()) if not acct_owned.empty else set()

        for _, ev in events.iterrows():
            sig = _norm_signal(ev.get("Signal"))
            t = str(ev.get("Ticker", "")).upper().strip()
            if not t or sig not in {"BUY", "NEAR", "SELL", "SHORT"}:
                continue

            is_owned = t in owned_tickers

            # Reduce noise: do not show hundreds of "SELL not owned" rows.
            if sig == "SELL" and not is_owned:
                continue

            owned_row = acct_owned[acct_owned["Symbol"].astype(str).str.upper().eq(t)].head(1) if not acct_owned.empty else pd.DataFrame()

            if sig == "SELL":
                action = "SELL / reduce review"
            elif sig == "BUY" and is_owned:
                action = "BUY / add-to-position candidate"
            elif sig == "BUY":
                action = "BUY candidate"
            elif sig == "NEAR" and is_owned:
                action = "NEAR watch - already owned"
            elif sig == "NEAR":
                action = "NEAR watch"
            elif sig == "SHORT":
                action = "SHORT candidate"
            else:
                action = "Review"

            rows.append({
                "RunUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "AccountNumber": acct_num,
                "AccountLabel": label,
                "Profile": profile,
                "Ticker": t,
                "Signal": sig,
                "RecommendedAction": action,
                "Owned": is_owned,
                "OwnedQty": owned_row["Quantity"].iloc[0] if not owned_row.empty and "Quantity" in owned_row.columns else "",
                "CurrentValue": owned_row["Current Value"].iloc[0] if not owned_row.empty and "Current Value" in owned_row.columns else "",
                "SignalPrice": ev.get("Price", ""),
                "Reason": ev.get("Reason", ""),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "RunUTC", "AccountNumber", "AccountLabel", "Profile", "Ticker", "Signal",
            "RecommendedAction", "Owned", "OwnedQty", "CurrentValue", "SignalPrice", "Reason"
        ])

    priority = {"SELL": 0, "BUY": 1, "NEAR": 2, "SHORT": 3}
    out["_p"] = out["Signal"].map(priority).fillna(9)
    return out.sort_values(["AccountNumber", "_p", "Ticker"]).drop(columns=["_p"])

def read_meta_decisions(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    # Keep compact useful columns if present.
    keep = [c for c in ["date", "meta_profile", "meta_reason", "equity", "cash", "positions", "long_positions", "short_positions"] if c in df.columns]
    return df[keep].copy() if keep else df


def build_html(summary: dict, comparison: pd.DataFrame, recs: pd.DataFrame, meta: pd.DataFrame) -> str:
    def table(df, n=50):
        if df is None or df.empty:
            return "<p><i>No rows.</i></p>"
        return df.head(n).to_html(index=False, escape=True)

    parts = [
        "<html><body>",
        "<h2>Daily PROD vs SIM Parity Report</h2>",
        "<h3>Summary</h3>",
        "<ul>",
    ]
    for k, v in summary.items():
        if str(k).startswith("_"):
            continue
        parts.append(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>")
    parts += [
        "</ul>",
        "<h3>Action List — Large Fidelity Account (Profile D)</h3>",
        table(
            recs[recs["Profile"].astype(str).str.upper().eq("D")]
            if not recs.empty and "Profile" in recs.columns
            else recs,
            100,
        ),
        "<h3>Action List — Small Fidelity Account (META F)</h3>",
        table(
            recs[recs["Profile"].astype(str).str.upper().eq("F")]
            if not recs.empty and "Profile" in recs.columns
            else pd.DataFrame(),
            100,
        ),
        "<h3>PROD Intraday Signals Seen Today</h3>",
        table(summary.get("_prod_history_df", pd.DataFrame()), 100),
        "<h3>PROD vs SIM Signal Comparison</h3>",
        table(comparison, 100),
        "<h3>META F Decisions</h3>",
        table(meta.tail(20) if not meta.empty else meta, 20),
        "</body></html>",
    ]
    return "\n".join(parts)


def upload_to_sheets(profile_cfg: dict, comparison: pd.DataFrame, recs: pd.DataFrame, meta: pd.DataFrame):
    gs = profile_cfg.get("google_sheets", {}) or {}
    if not gs.get("enabled", False):
        return

    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(gs["service_account_json"], scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(gs["sheet_url"])
    tabs = gs.get("tabs", {}) or {}

    def write_tab(name, df):
        try:
            ws = sh.worksheet(name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=name, rows=1000, cols=50)
        ws.clear()
        out = df.copy().replace({np.nan: ""})
        values = [list(out.columns)] + out.astype(str).values.tolist()
        if values:
            ws.update(values)

    write_tab(tabs.get("comparison", "Daily_SIM_vs_PROD"), comparison)
    write_tab(tabs.get("account_recs", "Daily_Account_Recommendations"), recs)
    if not meta.empty:
        write_tab(tabs.get("meta_decisions", "Daily_META_F_Decisions"), meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-debug", default="output/intraday_debug.csv")
    ap.add_argument("--prod-history", default="output/prod_intraday_signal_history.csv")
    ap.add_argument("--prod-history-date", default="", help="Central-time date YYYY-MM-DD; defaults to latest date in history")
    ap.add_argument("--sim-d-events", required=True)
    ap.add_argument("--sim-e-events", default="")
    ap.add_argument("--sim-f-events", required=True)
    ap.add_argument("--sim-f-meta", default="")
    ap.add_argument("--positions-csv", default="")
    ap.add_argument("--profiles", default="account_strategy_profiles.yaml")
    ap.add_argument("--out-dir", default="output/daily_parity")
    ap.add_argument("--send-email", action="store_true")
    ap.add_argument("--upload-sheets", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    profile_cfg = load_profiles(args.profiles)
    if args.upload_sheets:
        profile_cfg.setdefault("google_sheets", {})["enabled"] = True

    prod = normalize_prod(_read_csv(args.prod_debug))
    prod_history_raw = read_prod_history_for_date(args.prod_history, args.prod_history_date or None)
    prod_history = summarize_prod_history(prod_history_raw)
    sim_d = normalize_sim(_read_csv(args.sim_d_events), "SIM_D")
    sim_e = normalize_sim(_read_csv(args.sim_e_events), "SIM_E")
    sim_f_raw = normalize_sim(_read_csv(args.sim_f_events), "SIM_F_RAW")
    meta = read_meta_decisions(args.sim_f_meta)
    sim_f = effective_f_signals(sim_d, sim_e, sim_f_raw, meta)

    if args.positions_csv:
        pos = attach_profiles(normalize_positions(read_fidelity_positions(args.positions_csv)), profile_cfg)
        print(f"Positions loaded from {args.positions_csv}: {len(pos)}")
    else:
        if args.positions_csv:
            print(f"WARNING: positions CSV not found: {args.positions_csv}")
        pos = pd.DataFrame()

    comparison = compare_signals(prod, sim_d, sim_f, prod_history, sim_f_raw)
    recs = account_recommendations(sim_d, sim_f, pos, profile_cfg)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    comp_path = os.path.join(args.out_dir, f"daily_prod_sim_signal_comparison_{stamp}.csv")
    rec_path = os.path.join(args.out_dir, f"daily_account_recommendations_{stamp}.csv")
    meta_path = os.path.join(args.out_dir, f"daily_meta_f_decisions_{stamp}.csv")
    effective_f_path = os.path.join(args.out_dir, "sim_F_effective_events.csv")
    prod_hist_path = os.path.join(args.out_dir, f"daily_prod_intraday_history_{stamp}.csv")
    html_path = os.path.join(args.out_dir, f"daily_prod_sim_summary_{stamp}.html")

    comparison.to_csv(comp_path, index=False)
    recs.to_csv(rec_path, index=False)
    sim_f.to_csv(effective_f_path, index=False)
    if not meta.empty:
        meta.to_csv(meta_path, index=False)
    if not prod_history.empty:
        prod_history.to_csv(prod_hist_path, index=False)

    summary = {
        "PROD latest snapshot signals": len(prod),
        "PROD intraday signals seen": len(prod_history),
        "SIM D signals": len(sim_d),
        "SIM E signals": len(sim_e),
        "SIM F raw signals": len(sim_f_raw),
        "SIM F effective signals": len(sim_f),
        "SIM F selected profile": latest_meta_profile(meta) or "UNKNOWN",
        "Account recommendation rows": len(recs),
        "PROD latest vs D exact ticker/signal matches": int(comparison["PROD_Latest_vs_D_Match"].sum()) if not comparison.empty else 0,
        "PROD latest vs F exact ticker/signal matches": int(comparison["PROD_Latest_vs_F_Match"].sum()) if not comparison.empty else 0,
        "PROD intraday vs D exact ticker/signal matches": int(comparison["PROD_Intraday_vs_D_Match"].sum()) if not comparison.empty else 0,
        "PROD intraday vs F exact ticker/signal matches": int(comparison["PROD_Intraday_vs_F_Match"].sum()) if not comparison.empty else 0,
        "Positions loaded": len(pos),
    }

    summary["_prod_history_df"] = prod_history
    html_body = build_html(summary, comparison, recs, meta)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_body)

    upload_to_sheets(profile_cfg, comparison, recs, meta)

    if args.send_email:
        try:
            from weinstein_mailer import send_email
            text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
            send_email(
                subject="Daily SIM vs PROD Parity Report",
                html_body=html_body,
                text_body=text,
                cfg_path="config.yaml",
                subject_tag="SIM-vs-PROD",
            )
        except Exception as e:
            print(f"WARNING: email failed: {e}")

    print("DONE")
    print(f"Comparison: {comp_path}")
    print(f"Account recommendations: {rec_path}")
    print(f"SIM F effective events: {effective_f_path}")
    if not meta.empty:
        print(f"META F decisions: {meta_path}")
    if not prod_history.empty:
        print(f"PROD intraday history: {prod_hist_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
