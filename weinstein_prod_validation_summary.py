#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_prod_validation_summary.py

Compares normal PROD intraday output against validation-mode output.

Normal PROD:
  output/intraday_debug.csv

Validation PROD:
  output/intraday_debug_validation.csv

Goal:
- Identify whether BUY/NEAR are blocked by volume pace, pivot/headroom,
  stage/MA, or other gates.
- Produce a compact CSV + HTML report.
- Optionally email the report.

This is an observability tool only. It does not place trades.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


SIGNAL_ORDER = ["BUY", "NEAR", "NONE", "SKIP-STAGE", "SKIP-MA", "SKIP-REGIME", "SELL", "SELL-TRIGGER"]


def read_csv_safe(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def pick_col(df: pd.DataFrame, *names: str) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def normalize_debug(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Source", "Ticker", "Signal", "Reason", "PriceNow", "Pivot", "HeadroomPct",
            "VolPace", "ADX14", "cond_buy_price_ok", "cond_buy_vol_ok",
            "cond_near_pace_gate", "cond_near_now", "buy_confirm",
        ])

    ticker_col = pick_col(df, "Ticker", "ticker")
    signal_col = pick_col(df, "Signal", "signal")
    reason_col = pick_col(df, "Reason", "reason")
    price_col = pick_col(df, "PriceNow", "price")
    pivot_col = pick_col(df, "Pivot", "pivot")
    headroom_col = pick_col(df, "HeadroomPct")
    vol_col = pick_col(df, "VolPace", "pace_full_vs50dma")
    adx_col = pick_col(df, "ADX14")

    out = pd.DataFrame()
    out["Source"] = source
    out["Ticker"] = df[ticker_col].astype(str).str.upper().str.strip() if ticker_col else ""
    out["Signal"] = df[signal_col].astype(str).str.upper().str.strip() if signal_col else ""
    out["Reason"] = df[reason_col].astype(str).str.strip() if reason_col else ""
    out["PriceNow"] = pd.to_numeric(df[price_col], errors="coerce") if price_col else pd.NA
    out["Pivot"] = pd.to_numeric(df[pivot_col], errors="coerce") if pivot_col else pd.NA
    out["HeadroomPct"] = pd.to_numeric(df[headroom_col], errors="coerce") if headroom_col else pd.NA
    out["VolPace"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else pd.NA
    out["ADX14"] = pd.to_numeric(df[adx_col], errors="coerce") if adx_col else pd.NA

    for c in [
        "cond_weekly_stage_ok",
        "cond_rs_ok",
        "cond_ma_ok",
        "cond_pivot_ok",
        "cond_buy_vol_ok",
        "cond_pace_full_gate",
        "cond_near_pace_gate",
        "cond_buy_price_ok",
        "cond_near_now",
        "buy_confirm",
    ]:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = pd.NA

    out = out[out["Ticker"].ne("")]
    return out.reset_index(drop=True)


def signal_counts(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"Source": [label], "Signal": ["NO_FILE"], "Count": [0]})

    vc = df["Signal"].fillna("").replace("", "BLANK").value_counts()
    rows = []
    for sig in SIGNAL_ORDER:
        if sig in vc.index:
            rows.append({"Source": label, "Signal": sig, "Count": int(vc.loc[sig])})
    for sig, cnt in vc.items():
        if sig not in SIGNAL_ORDER:
            rows.append({"Source": label, "Signal": sig, "Count": int(cnt)})
    return pd.DataFrame(rows)


def gate_failure_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["Source", "Gate", "FalseOrMissingCount", "TrueCount"])

    for c in [
        "cond_weekly_stage_ok",
        "cond_rs_ok",
        "cond_ma_ok",
        "cond_pivot_ok",
        "cond_buy_vol_ok",
        "cond_pace_full_gate",
        "cond_near_pace_gate",
        "cond_buy_price_ok",
        "cond_near_now",
        "buy_confirm",
    ]:
        if c not in df.columns:
            continue
        s = df[c]
        true_count = int((s.astype(str).str.lower() == "true").sum())
        false_missing = int(len(s) - true_count)
        rows.append({
            "Source": label,
            "Gate": c,
            "FalseOrMissingCount": false_missing,
            "TrueCount": true_count,
        })
    return pd.DataFrame(rows)


def top_candidates(df: pd.DataFrame, label: str, n: int = 40) -> pd.DataFrame:
    """Show best near-breakout candidates even if they are not confirmed BUY."""
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    x["AbsHeadroomPct"] = x["HeadroomPct"].abs()
    # Prefer actual BUY/NEAR, then closest to pivot, then highest volume pace.
    sig_priority = {"BUY": 0, "NEAR": 1, "NONE": 2, "SKIP-STAGE": 3, "SKIP-MA": 4}
    x["_sig_p"] = x["Signal"].map(sig_priority).fillna(9)
    x = x.sort_values(["_sig_p", "AbsHeadroomPct", "VolPace"], ascending=[True, True, False])
    cols = [
        "Source", "Ticker", "Signal", "Reason", "PriceNow", "Pivot", "HeadroomPct",
        "VolPace", "ADX14", "cond_buy_price_ok", "cond_buy_vol_ok",
        "cond_near_pace_gate", "cond_near_now", "buy_confirm",
    ]
    return x[cols].head(n)


def compare_strict_validation(strict: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    if strict.empty and validation.empty:
        return pd.DataFrame()

    s = strict[["Ticker", "Signal", "Reason", "HeadroomPct", "VolPace"]].copy()
    s.columns = ["Ticker", "StrictSignal", "StrictReason", "StrictHeadroomPct", "StrictVolPace"]

    v = validation[["Ticker", "Signal", "Reason", "HeadroomPct", "VolPace"]].copy()
    v.columns = ["Ticker", "ValidationSignal", "ValidationReason", "ValidationHeadroomPct", "ValidationVolPace"]

    out = pd.merge(s, v, on="Ticker", how="outer")
    out = out[
        (out["StrictSignal"].fillna("") != out["ValidationSignal"].fillna(""))
        | (out["ValidationSignal"].isin(["BUY", "NEAR"]))
    ].copy()

    priority = {"BUY": 0, "NEAR": 1, "NONE": 2, "SKIP-STAGE": 3, "SKIP-MA": 4}
    out["_p"] = out["ValidationSignal"].map(priority).fillna(9)
    return out.sort_values(["_p", "Ticker"]).drop(columns=["_p"])


def html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p><i>No rows.</i></p>"
    return df.head(max_rows).to_html(index=False, escape=True)


def build_html(summary: Dict[str, object], counts: pd.DataFrame, gates: pd.DataFrame, diffs: pd.DataFrame, candidates: pd.DataFrame) -> str:
    parts = [
        "<html><body>",
        "<h2>PROD Validation Mode Summary</h2>",
        "<p>This report compares normal PROD intraday output with TEST-EASE validation output. It does not place trades.</p>",
        "<h3>Summary</h3>",
        "<ul>",
    ]
    for k, v in summary.items():
        parts.append(f"<li><b>{k}</b>: {v}</li>")
    parts.extend([
        "</ul>",
        "<h3>Signal Counts</h3>",
        html_table(counts),
        "<h3>Gate Failure Summary</h3>",
        html_table(gates),
        "<h3>Strict vs Validation Differences</h3>",
        html_table(diffs),
        "<h3>Top Validation Candidates</h3>",
        html_table(candidates),
        "</body></html>",
    ])
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-debug", default="./output/intraday_debug.csv")
    ap.add_argument("--validation-debug", default="./output/intraday_debug_validation.csv")
    ap.add_argument("--out-dir", default="./output/prod_validation")
    ap.add_argument("--send-email", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    strict_raw = read_csv_safe(args.strict_debug)
    validation_raw = read_csv_safe(args.validation_debug)

    strict = normalize_debug(strict_raw, "STRICT_PROD")
    validation = normalize_debug(validation_raw, "VALIDATION_TEST_EASE")

    counts = pd.concat([
        signal_counts(strict, "STRICT_PROD"),
        signal_counts(validation, "VALIDATION_TEST_EASE"),
    ], ignore_index=True)

    gates = pd.concat([
        gate_failure_summary(strict, "STRICT_PROD"),
        gate_failure_summary(validation, "VALIDATION_TEST_EASE"),
    ], ignore_index=True)

    diffs = compare_strict_validation(strict, validation)
    candidates = top_candidates(validation, "VALIDATION_TEST_EASE")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    counts_path = os.path.join(args.out_dir, f"prod_validation_signal_counts_{stamp}.csv")
    gates_path = os.path.join(args.out_dir, f"prod_validation_gate_summary_{stamp}.csv")
    diffs_path = os.path.join(args.out_dir, f"prod_validation_differences_{stamp}.csv")
    cand_path = os.path.join(args.out_dir, f"prod_validation_candidates_{stamp}.csv")
    html_path = os.path.join(args.out_dir, f"prod_validation_summary_{stamp}.html")

    counts.to_csv(counts_path, index=False)
    gates.to_csv(gates_path, index=False)
    diffs.to_csv(diffs_path, index=False)
    candidates.to_csv(cand_path, index=False)

    def cnt(df: pd.DataFrame, sig: str) -> int:
        if df.empty:
            return 0
        return int((df["Signal"] == sig).sum())

    summary = {
        "Strict rows": len(strict),
        "Validation rows": len(validation),
        "Strict BUY": cnt(strict, "BUY"),
        "Strict NEAR": cnt(strict, "NEAR"),
        "Validation BUY": cnt(validation, "BUY"),
        "Validation NEAR": cnt(validation, "NEAR"),
        "Validation NONE": cnt(validation, "NONE"),
        "Differences": len(diffs),
        "Strict debug": args.strict_debug,
        "Validation debug": args.validation_debug,
    }

    html_body = build_html(summary, counts, gates, diffs, candidates)
    Path(html_path).write_text(html_body, encoding="utf-8")

    print("DONE")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"counts: {counts_path}")
    print(f"gates: {gates_path}")
    print(f"differences: {diffs_path}")
    print(f"candidates: {cand_path}")
    print(f"html: {html_path}")

    if args.send_email:
        try:
            from weinstein_mailer import send_email
            text_body = "\n".join([f"{k}: {v}" for k, v in summary.items()])
            send_email(
                subject="PROD Validation Mode Summary",
                html_body=html_body,
                text_body=text_body,
                cfg_path="config.yaml",
                subject_tag="PROD-Validation",
            )
        except Exception as e:
            print(f"WARNING: email failed: {e}")


if __name__ == "__main__":
    main()
