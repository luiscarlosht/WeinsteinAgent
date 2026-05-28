#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_prod_account_router.py

Phase 2 PROD account-routing report for WeinsteinAgent.

Purpose:
- Keep Weinstein CORE unchanged.
- Read current PROD diagnostics from output/intraday_debug.csv.
- Read latest daily D/F parity outputs when available.
- Route recommendations by account profile:
    X48354910 -> D baseline
    Z30958579 -> F META adaptive
- Filter SELL recommendations to owned tickers only.
- Include latest META F mode/reason from daily parity.
- Send a PROD-account-routing email.

This is intentionally an operational/reporting layer, not a signal-core change.
"""

from __future__ import annotations

import argparse
import glob
import html
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from weinstein_prod_history import read_prod_history_for_date, summarize_prod_history

from weinstein_account_profiles import (
    load_profiles,
    read_fidelity_positions,
    normalize_positions,
    attach_profiles,
)


SIGNALS = {"BUY", "NEAR", "NEAR_BUY", "NEAR-TRIGGER", "SELL", "SELLTRIG", "SELL-TRIGGER", "SELL-WATCH", "SHORT"}


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, engine="python", on_bad_lines="skip")


def _norm_signal(x: object) -> str:
    s = str(x or "").strip().upper()
    if s in {"NEAR_BUY", "NEAR-TRIGGER"}:
        return "NEAR"
    if s in {"SELLTRIG", "SELL-TRIGGER", "SELL-WATCH"}:
        return "SELL"
    if s in {"BUY", "NEAR", "SELL", "SHORT"}:
        return s
    return s


def _latest_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ["date", "Date", "TimestampUTC", "timestamp", "RunDate", "RunUTC"]:
        if c in df.columns:
            dts = pd.to_datetime(df[c], errors="coerce")
            if not dts.dropna().empty:
                latest = dts.max().date()
                out = df.loc[dts.dt.date.eq(latest)].copy()
                return out if not out.empty else df
    return df


def normalize_signal_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Signal", "Price", "Reason", "Source"])
    out = _latest_date_filter(df.copy())

    if "Ticker" not in out.columns:
        for c in ["ticker", "symbol", "Symbol"]:
            if c in out.columns:
                out["Ticker"] = out[c]
                break

    if "Signal" not in out.columns:
        for c in ["signal", "Signal", "Action", "action"]:
            if c in out.columns:
                out["Signal"] = out[c]
                break

    if "Price" not in out.columns:
        for c in ["PriceNow", "price", "Price", "Close", "close", "close_price"]:
            if c in out.columns:
                out["Price"] = out[c]
                break

    if "Reason" not in out.columns:
        for c in ["Reason", "reason", "Details", "detail", "why"]:
            if c in out.columns:
                out["Reason"] = out[c]
                break

    out["Ticker"] = out.get("Ticker", "").astype(str).str.upper().str.strip()
    out["Signal"] = out.get("Signal", "").apply(_norm_signal)
    out["Source"] = source
    out = out[out["Signal"].isin({"BUY", "NEAR", "SELL", "SHORT"})]

    for c in ["Price", "Reason"]:
        if c not in out.columns:
            out[c] = ""

    return out[["Ticker", "Signal", "Price", "Reason", "Source"]].drop_duplicates().reset_index(drop=True)


def latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        return ""
    return max(files, key=os.path.getmtime)


def latest_parity_dir(out_dir: str = "output/daily_parity") -> str:
    dirs = [p for p in glob.glob(os.path.join(out_dir, "*")) if os.path.isdir(p)]
    if not dirs:
        return ""
    # Prefer dirs containing at least one useful daily parity output.
    dirs = sorted(dirs, key=os.path.getmtime, reverse=True)
    for d in dirs:
        if glob.glob(os.path.join(d, "sim_D_replay_events.csv")) or glob.glob(os.path.join(d, "daily_account_recommendations_*.csv")):
            return d
    return dirs[0]


def read_latest_parity(parity_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not parity_dir:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    sim_d = _read_csv(os.path.join(parity_dir, "sim_D_replay_events.csv"))
    sim_f = _read_csv(os.path.join(parity_dir, "sim_F_base_events.csv"))

    meta_path = latest_file(os.path.join(parity_dir, "daily_meta_f_decisions_*.csv")) or os.path.join(parity_dir, "sim_F_meta_equity.csv")
    meta = _read_csv(meta_path)

    info = {
        "parity_dir": parity_dir,
        "sim_d_rows": len(sim_d),
        "sim_f_rows": len(sim_f),
        "meta_rows": len(meta),
        "meta_path": meta_path if meta_path and Path(meta_path).exists() else "",
    }
    return sim_d, sim_f, meta, info


def latest_meta_state(meta: pd.DataFrame) -> dict:
    if meta.empty:
        return {"meta_profile": "", "meta_reason": "", "date": "", "equity": ""}
    df = meta.copy()
    if "date" in df.columns:
        df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("_dt")
    row = df.tail(1).iloc[0].to_dict()
    return {
        "date": row.get("date", ""),
        "meta_profile": row.get("meta_profile", ""),
        "meta_reason": row.get("meta_reason", ""),
        "equity": row.get("equity", ""),
        "cash": row.get("cash", ""),
        "positions": row.get("positions", ""),
        "long_positions": row.get("long_positions", ""),
        "short_positions": row.get("short_positions", ""),
    }


def load_positions(path: str, profiles_path: str) -> pd.DataFrame:
    cfg = load_profiles(profiles_path)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return attach_profiles(normalize_positions(read_fidelity_positions(path)), cfg)


def account_routed_recommendations(
    profile_cfg: dict,
    positions: pd.DataFrame,
    prod: pd.DataFrame,
    sim_d: pd.DataFrame,
    sim_f: pd.DataFrame,
    d_source: str = "auto",
    f_source: str = "sim",
) -> pd.DataFrame:
    """Build account-specific recommendations.

    Source behavior:
    - D account uses PROD if available when d_source=auto, otherwise latest SIM D.
    - F account uses latest SIM F for now. This preserves CORE safety while F is still an overlay/shadow engine.
    """
    accounts = profile_cfg.get("accounts", []) or []
    owned = positions[~positions.get("IsCash", False)].copy() if not positions.empty else pd.DataFrame()

    rows = []
    for acct in accounts:
        acct_num = str(acct.get("account_number", "")).strip()
        label = str(acct.get("label", "")).strip()
        profile = str(acct.get("profile", "")).strip().upper()
        role = str(acct.get("role", "")).strip()

        if profile == "D":
            if d_source == "prod":
                source_df = prod
                source_label = "PROD_D"
            elif d_source == "sim":
                source_df = sim_d
                source_label = "SIM_D"
            else:
                source_df = prod if not prod.empty else sim_d
                source_label = "PROD_D" if not prod.empty else "SIM_D_FALLBACK"
        elif profile == "F":
            source_df = sim_f
            source_label = "SIM_F_META"
        else:
            continue

        acct_owned = owned[owned["Account Number"].astype(str).eq(acct_num)].copy() if not owned.empty else pd.DataFrame()
        owned_tickers = set(acct_owned["Symbol"].astype(str).str.upper()) if not acct_owned.empty else set()

        for _, ev in source_df.iterrows():
            ticker = str(ev.get("Ticker", "")).upper().strip()
            sig = _norm_signal(ev.get("Signal"))
            if not ticker or sig not in {"BUY", "NEAR", "SELL", "SHORT"}:
                continue

            is_owned = ticker in owned_tickers
            if sig == "SELL" and not is_owned:
                continue

            owned_row = acct_owned[acct_owned["Symbol"].astype(str).str.upper().eq(ticker)].head(1) if not acct_owned.empty else pd.DataFrame()

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
                "Role": role,
                "Profile": profile,
                "Source": source_label,
                "Ticker": ticker,
                "Signal": sig,
                "RecommendedAction": action,
                "Owned": bool(is_owned),
                "OwnedQty": owned_row["Quantity"].iloc[0] if not owned_row.empty and "Quantity" in owned_row.columns else "",
                "CurrentValue": owned_row["Current Value"].iloc[0] if not owned_row.empty and "Current Value" in owned_row.columns else "",
                "SignalPrice": ev.get("Price", ""),
                "Reason": ev.get("Reason", ""),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "RunUTC", "AccountNumber", "AccountLabel", "Role", "Profile", "Source", "Ticker", "Signal",
            "RecommendedAction", "Owned", "OwnedQty", "CurrentValue", "SignalPrice", "Reason"
        ])
    priority = {"SELL": 0, "BUY": 1, "NEAR": 2, "SHORT": 3}
    out["_p"] = out["Signal"].map(priority).fillna(9)
    return out.sort_values(["AccountNumber", "_p", "Ticker"]).drop(columns=["_p"]).reset_index(drop=True)


def differences(d_df: pd.DataFrame, f_df: pd.DataFrame) -> pd.DataFrame:
    d = d_df.groupby("Ticker")["Signal"].apply(lambda s: ",".join(sorted(set(s)))).to_dict() if not d_df.empty else {}
    f = f_df.groupby("Ticker")["Signal"].apply(lambda s: ",".join(sorted(set(s)))).to_dict() if not f_df.empty else {}
    keys = sorted(set(d) | set(f))
    rows = []
    for k in keys:
        if d.get(k, "") != f.get(k, ""):
            rows.append({"Ticker": k, "D_Signal": d.get(k, ""), "F_Signal": f.get(k, "")})
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame, n: int = 100) -> str:
    if df is None or df.empty:
        return "<p><i>No rows.</i></p>"
    return df.head(n).to_html(index=False, escape=True)


def build_html(summary: dict, recs: pd.DataFrame, diffs: pd.DataFrame, meta: pd.DataFrame, positions: pd.DataFrame) -> str:
    meta_state = summary.get("meta_state", {}) or {}
    parts = [
        "<html><body>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;color:#222} table{border-collapse:collapse;width:100%;margin:8px 0} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px} th{background:#f6f6f6} .note{color:#666;font-size:12px}</style>",
        "<h2>PROD Account Routing — D Baseline + META F</h2>",
        "<h3>Current Routing</h3>",
        "<ul>",
        "<li><b>Large Fidelity:</b> D baseline</li>",
        "<li><b>Small Fidelity:</b> META F validation</li>",
        "</ul>",
        "<h3>META F Current State</h3>",
        "<ul>",
        f"<li><b>Date:</b> {html.escape(str(meta_state.get('date','')))}</li>",
        f"<li><b>Selected profile:</b> {html.escape(str(meta_state.get('meta_profile','')))}</li>",
        f"<li><b>Reason:</b> {html.escape(str(meta_state.get('meta_reason','')))}</li>",
        f"<li><b>Equity:</b> {html.escape(str(meta_state.get('equity','')))}</li>",
        "</ul>",
        "<h3>Summary</h3>",
        "<ul>",
    ]
    for k, v in summary.items():
        if k == "meta_state" or str(k).startswith("_"):
            continue
        parts.append(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>")
    parts += [
        "</ul>",
        "<h3>PROD Intraday Signals Seen Today</h3>",
        _table(summary.get("_prod_history_df", pd.DataFrame()), 100),
        "<h3>Account-Routed Recommendations</h3>",
        _table(recs, 100),
        "<h3>D vs F Differences</h3>",
        _table(diffs, 100),
        "<h3>Recent META F Decisions</h3>",
        _table(meta.tail(20) if not meta.empty else meta, 20),
        "<h3>Loaded Positions</h3>",
        _table(positions[[c for c in ["Account Number", "AccountLabel", "Profile", "Symbol", "Quantity", "Current Value", "IsCash"] if c in positions.columns]].head(50) if not positions.empty else positions, 50),
        "<p class='note'>This report is an operational routing layer. It does not place trades and does not alter Weinstein CORE logic.</p>",
        "</body></html>",
    ]
    return "\n".join(parts)


def write_outputs(out_dir: str, recs: pd.DataFrame, diffs: pd.DataFrame, html_body: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rec_path = os.path.join(out_dir, f"prod_account_recommendations_{stamp}.csv")
    diff_path = os.path.join(out_dir, f"prod_d_vs_f_differences_{stamp}.csv")
    html_path = os.path.join(out_dir, f"prod_account_routing_summary_{stamp}.html")
    recs.to_csv(rec_path, index=False)
    diffs.to_csv(diff_path, index=False)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_body)
    return {"recommendations": rec_path, "differences": diff_path, "html": html_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-debug", default="output/intraday_debug.csv")
    ap.add_argument("--prod-history", default="output/prod_intraday_signal_history.csv")
    ap.add_argument("--prod-history-date", default="", help="Central-time date YYYY-MM-DD; defaults to latest date in history")
    ap.add_argument("--positions-csv", default="")
    ap.add_argument("--profiles", default="account_strategy_profiles.yaml")
    ap.add_argument("--parity-dir", default="")
    ap.add_argument("--out-dir", default="output/prod_account_routing")
    ap.add_argument("--d-source", choices=["auto", "prod", "sim"], default="auto")
    ap.add_argument("--send-email", action="store_true")
    args = ap.parse_args()

    profile_cfg = load_profiles(args.profiles)

    parity_dir = args.parity_dir or latest_parity_dir()
    raw_sim_d, raw_sim_f, meta, parity_info = read_latest_parity(parity_dir)

    prod_latest = normalize_signal_df(_read_csv(args.prod_debug), "PROD_LATEST")
    prod_history_raw = read_prod_history_for_date(args.prod_history, args.prod_history_date or None)
    prod_history = summarize_prod_history(prod_history_raw)
    # For routing, prefer signals actually seen at any point intraday; fall back to latest snapshot.
    prod = prod_history if not prod_history.empty else prod_latest
    sim_d = normalize_signal_df(raw_sim_d, "SIM_D")
    sim_f = normalize_signal_df(raw_sim_f, "SIM_F")

    positions = load_positions(args.positions_csv, args.profiles) if args.positions_csv else pd.DataFrame()

    recs = account_routed_recommendations(
        profile_cfg=profile_cfg,
        positions=positions,
        prod=prod,
        sim_d=sim_d,
        sim_f=sim_f,
        d_source=args.d_source,
    )
    diffs = differences(sim_d, sim_f)
    meta_state = latest_meta_state(meta)

    summary = {
        "PROD latest snapshot signals": len(prod_latest),
        "PROD intraday signals seen": len(prod_history),
        "PROD routing signals used": len(prod),
        "SIM D signals available": len(sim_d),
        "SIM F signals available": len(sim_f),
        "Positions loaded": len(positions),
        "Account recommendation rows": len(recs),
        "D vs F difference rows": len(diffs),
        "D source mode": args.d_source,
        "Latest parity dir": parity_info.get("parity_dir", ""),
        "META rows": len(meta),
        "meta_state": meta_state,
    }

    summary["_prod_history_df"] = prod_history
    html_body = build_html(summary, recs, diffs, meta, positions)
    paths = write_outputs(args.out_dir, recs, diffs, html_body)

    if args.send_email:
        from weinstein_mailer import send_email
        text_body = "\n".join([f"{k}: {v}" for k, v in summary.items() if k != "meta_state"])
        text_body += f"\nMETA F: {meta_state.get('meta_profile','')} / {meta_state.get('meta_reason','')}"
        send_email(
            subject="PROD Account Routing — D Baseline + META F",
            html_body=html_body,
            text_body=text_body,
            cfg_path="config.yaml",
            subject_tag="PROD-Routing",
        )

    print("DONE")
    for k, v in paths.items():
        print(f"{k}: {v}")
    print(f"Recommendations rows: {len(recs)}")
    print(f"Positions loaded: {len(positions)}")
    print(f"META F: {meta_state.get('meta_profile','')} / {meta_state.get('meta_reason','')}")


if __name__ == "__main__":
    main()
