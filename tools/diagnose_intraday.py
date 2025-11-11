#!/usr/bin/env python3
import os, re, sys, csv, glob, argparse
from datetime import datetime
import pandas as pd

NEAR_HTML_GLOB = "./output/intraday_watch_*.html"
INTRADAY_CSV   = "./output/intraday_debug.csv"
OUT_DIR        = "./output"

NEAR_LIST_ITEM_RE = re.compile(
    r"<ol><li><b>\d+\.</b>\s*<b>([A-Z]+)</b>\s*@\s*([0-9.]+)\s*\(pivot\s*([0-9.]+)",
    re.IGNORECASE
)

BUY_GATES_ALL = [
    "cond_weekly_stage_ok",
    "cond_rs_ok",
    "cond_ma_ok",
    "cond_pivot_ok",
    "cond_buy_price_ok",
    "cond_buy_vol_ok",
    "cond_pace_full_gate",
]
NEAR_GATES = [
    "cond_near_pace_gate",   # near gate often separate from full gate
]

SELL_GATES = [
    "cond_sell_price_ok",
    "cond_sell_vol_ok",
]

def load_intraday_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Guard common issues
    essential = [c for c in ["ticker","price","pivot"] if c in df.columns]
    if len(essential) < 3:
        return pd.DataFrame()
    df = df.dropna(subset=["ticker","price","pivot"])
    # bps distance to pivot (positive => price below pivot)
    df["dist_bps"] = (df["pivot"] - df["price"]) / df["pivot"] * 1e4
    # time marker (bigger = later in the day)
    if "elapsed_min" in df.columns:
        df["ts_min"] = df["elapsed_min"]
    else:
        df["ts_min"] = 0
    # Make sure gate columns exist (fill absent with False)
    for col in set(BUY_GATES_ALL + NEAR_GATES + SELL_GATES +
                   ["cond_near_now","cond_sell_near_now","cond_buy_confirm","cond_sell_confirm"]):
        if col not in df.columns:
            df[col] = False
    return df

def parse_near_htmls():
    tickers = []
    rows = []
    for html_path in sorted(glob.glob(NEAR_HTML_GLOB)):
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            for m in NEAR_LIST_ITEM_RE.finditer(text):
                tkr = m.group(1).upper()
                price = float(m.group(2))
                pivot = float(m.group(3))
                rows.append({"file": os.path.basename(html_path), "ticker": tkr, "price": price, "pivot": pivot})
                tickers.append(tkr)
        except Exception as e:
            # Soft-fail and keep going
            pass
    return sorted(set(tickers)), pd.DataFrame(rows)

def fail_reasons(row, gates):
    reasons = []
    for g in gates:
        if g in row and not bool(row[g]):
            reasons.append(g)
    return reasons

def best_latest_snapshot(df, tkr):
    d = df[df["ticker"]==tkr]
    if d.empty:
        return None
    d = d.sort_values("ts_min")
    return d.iloc[-1].to_dict()

def main():
    ap = argparse.ArgumentParser(description="Weinstein intraday diagnostics")
    ap.add_argument("--csv", default=INTRADAY_CSV, help="Path to intraday_debug.csv")
    ap.add_argument("--outdir", default=OUT_DIR, help="Output directory for diagnostics")
    ap.add_argument("--limit", type=int, default=50, help="Max rows to show in console listings")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Build near universe from HTML
    near_tickers, near_rows = parse_near_htmls()
    near_txt = os.path.join(args.outdir, "near_universe.txt")
    with open(near_txt, "w") as f:
        for t in near_tickers:
            f.write(f"{t}\n")

    # 2) Load intraday_debug.csv
    df = load_intraday_csv(args.csv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_buy_near_csv = os.path.join(args.outdir, f"diag_buy_near_{timestamp}.csv")
    out_sell_csv     = os.path.join(args.outdir, f"diag_sell_{timestamp}.csv")
    out_summary_txt  = os.path.join(args.outdir, f"diag_summary_{timestamp}.txt")

    # 3) BUY near-misses: all green except price gate (and we also want weekly/pivot/etc true)
    if not df.empty:
        buy_near_mask = (
            df["cond_weekly_stage_ok"] &
            df["cond_rs_ok"] &
            df["cond_ma_ok"] &
            df["cond_pivot_ok"] &
            df["cond_buy_vol_ok"] &
            df["cond_pace_full_gate"] &
            df["cond_near_pace_gate"] &
            (df["cond_buy_price_ok"] == False)
        )
        buy_near = df[buy_near_mask].copy()

        # Summaries per ticker
        buy_near["scan"] = 1
        agg = (buy_near
               .groupby("ticker", as_index=False)
               .agg(scans=("scan","sum"),
                    min_dist_bps=("dist_bps","min"),
                    last_seen=("ts_min","max"))
               .sort_values(["min_dist_bps","scans"], ascending=[True, False])
        )

        # Latest snapshot + fail reasons
        snapshots = []
        for t in agg["ticker"]:
            snap = best_latest_snapshot(buy_near, t)
            if snap is None:
                continue
            reasons = fail_reasons(snap, BUY_GATES_ALL + NEAR_GATES)
            snapshots.append({
                "ticker": t,
                "price": snap.get("price"),
                "pivot": snap.get("pivot"),
                "dist_bps": snap.get("dist_bps"),
                "last_seen_min": snap.get("ts_min"),
                "scans": int(agg.loc[agg["ticker"]==t, "scans"].values[0]),
                "fail_reasons": ",".join(reasons)
            })
        buy_near_out = pd.DataFrame(snapshots).sort_values(["dist_bps","scans"], ascending=[True, False])
        if not buy_near_out.empty:
            buy_near_out.to_csv(out_buy_near_csv, index=False, quoting=csv.QUOTE_MINIMAL)

        # 4) SELL candidates
        sell_mask = (df["cond_sell_near_now"] | df["cond_sell_confirm"] | (df["cond_sell_price_ok"] & df["cond_sell_vol_ok"]))
        sell_df = df[sell_mask].copy()
        sell_rows = []
        for t in sorted(sell_df["ticker"].unique()):
            snap = best_latest_snapshot(sell_df, t)
            if snap is None: continue
            reasons = fail_reasons(snap, SELL_GATES + ["cond_sell_confirm","cond_sell_near_now"])
            sell_rows.append({
                "ticker": t,
                "price": snap.get("price"),
                "pivot": snap.get("pivot"),
                "last_seen_min": snap.get("ts_min"),
                "sell_reasons_false": ",".join(reasons),
                "sell_confirm": bool(snap.get("cond_sell_confirm", False)),
                "sell_near_now": bool(snap.get("cond_sell_near_now", False)),
            })
        sell_out = pd.DataFrame(sell_rows).sort_values("ticker")
        if not sell_out.empty:
            sell_out.to_csv(out_sell_csv, index=False, quoting=csv.QUOTE_MINIMAL)

        # 5) Human-friendly summary
        with open(out_summary_txt, "w") as f:
            f.write(f"=== Weinstein Intraday Diagnostics @ {timestamp} ===\n")
            f.write(f"Near-universe tickers discovered in HTML: {len(near_tickers)} (saved: {os.path.basename(near_txt)})\n")
            f.write(f"Snapshots in CSV: {len(df)}\n\n")

            # BUY near-misses
            f.write("— Top BUY near-misses (closest to pivot; all green except price):\n")
            if not buy_near_out.empty:
                for _, r in buy_near_out.head(50).iterrows():
                    f.write(f"  {r['ticker']}: price={r['price']:.2f}, pivot={r['pivot']:.2f}, "
                            f"dist_bps={r['dist_bps']:.1f}, scans={r['scans']}, "
                            f"fail={r['fail_reasons']}\n")
                f.write(f"(Full CSV: {os.path.basename(out_buy_near_csv)})\n")
            else:
                f.write("  None.\n")

            # SELL
            f.write("\n— SELL candidates:\n")
            if not sell_out.empty:
                for _, r in sell_out.head(50).iterrows():
                    f.write(f"  {r['ticker']}: price={r['price']:.2f}, "
                            f"sell_confirm={r['sell_confirm']}, sell_near_now={r['sell_near_now']}, "
                            f"reasons_false={r['sell_reasons_false']}\n")
                f.write(f"(Full CSV: {os.path.basename(out_sell_csv)})\n")
            else:
                f.write("  None.\n")

    else:
        # No CSV data, still write near universe
        with open(out_summary_txt, "w") as f:
            f.write(f"=== Weinstein Intraday Diagnostics @ {timestamp} ===\n")
            f.write("No valid intraday_debug.csv found or file empty.\n")
            f.write(f"Near-universe tickers discovered in HTML: {len(near_tickers)} (saved: {os.path.basename(near_txt)})\n")

    # Also persist the parsed near rows with (file, ticker, price, pivot) for traceability
    if not near_rows.empty:
        near_rows.to_csv(os.path.join(args.outdir, f"near_html_rows_{timestamp}.csv"), index=False)

    # Console hints
    print("Diagnostics complete.")
    print(f"- Near-universe: {near_txt}")
    if os.path.exists(out_summary_txt): print(f"- Summary:      {out_summary_txt}")
    if os.path.exists(out_buy_near_csv): print(f"- BUY near CSV: {out_buy_near_csv}")
    if os.path.exists(out_sell_csv):     print(f"- SELL CSV:     {out_sell_csv}")

if __name__ == "__main__":
    main()
