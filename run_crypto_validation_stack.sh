#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

source ~/.weinstein_env 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="./output/crypto_validation/$STAMP"
mkdir -p "$OUTDIR"

run_case () {
  local name="$1"
  shift

  echo
  echo "===== Crypto validation: $name ====="
  date

  python3 weinstein_crypto_watcher.py \
    --config ./config.yaml \
    "$@" \
    2>&1 | tee "$OUTDIR/${name}.log"

  cp -v ./output/crypto_debug.csv "$OUTDIR/${name}_crypto_debug.csv" 2>/dev/null || true

  latest_html="$(ls -1t ./output/crypto_watch_*.html 2>/dev/null | head -1 || true)"
  if [ -n "$latest_html" ]; then
    cp -v "$latest_html" "$OUTDIR/${name}_crypto_watch.html"
  fi
}

run_case PROD_B --profile B
run_case SIM_B  --profile B --sim
run_case SIM_C  --profile C --sim
run_case SIM_D  --profile D --sim
run_case SIM_E  --profile E --sim
run_case SIM_F  --profile F --sim

python3 - <<'PY' "$OUTDIR"
import sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
cases = ["PROD_B", "SIM_B", "SIM_C", "SIM_D", "SIM_E", "SIM_F"]

dfs = {}
summary_rows = []

def load_case(case):
    p = out / f"{case}_crypto_debug.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "Ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})
    return df

def state_series(df):
    if df is None or df.empty:
        return pd.Series(dtype=str)
    state_col = "State" if "State" in df.columns else None
    sell_col = "SellConfirm" if "SellConfirm" in df.columns else None
    rec_col = "PortfolioRecommendation" if "PortfolioRecommendation" in df.columns else None

    out_state = []
    for _, r in df.iterrows():
        state = str(r.get(state_col, "")).upper() if state_col else ""
        sell = str(r.get(sell_col, "")).lower() == "true" if sell_col else False
        rec = str(r.get(rec_col, "")).lower() if rec_col else ""

        if sell or "sell trigger" in rec or "reduce/exit" in rec:
            out_state.append("SELL")
        elif state == "TRIGGERED":
            out_state.append("BUY")
        elif state in {"NEAR", "ARMED"}:
            out_state.append("NEAR")
        else:
            out_state.append("NONE")
    return pd.Series(out_state)

for case in cases:
    df = load_case(case)
    if df is None:
        summary_rows.append({
            "case": case,
            "status": "MISSING",
            "rows": 0,
            "buy": 0,
            "near": 0,
            "sell": 0,
            "none": 0,
        })
        continue

    dfs[case] = df
    s = state_series(df)

    summary_rows.append({
        "case": case,
        "status": "OK",
        "rows": len(df),
        "buy": int((s == "BUY").sum()),
        "near": int((s == "NEAR").sum()),
        "sell": int((s == "SELL").sum()),
        "none": int((s == "NONE").sum()),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(out / "crypto_validation_summary.csv", index=False)

# Build profile comparison matrix by ticker
all_tickers = sorted(set().union(*[
    set(df["Ticker"].astype(str)) for df in dfs.values() if "Ticker" in df.columns
]))

matrix = pd.DataFrame({"Ticker": all_tickers})

for case, df in dfs.items():
    tmp = df.copy()
    tmp["ValidationState"] = state_series(tmp).values
    matrix = matrix.merge(
        tmp[["Ticker", "ValidationState"]].rename(columns={"ValidationState": case}),
        on="Ticker",
        how="left",
    )

for case in cases:
    if case in matrix.columns:
        matrix[case] = matrix[case].fillna("MISSING")

matrix.to_csv(out / "crypto_profile_comparison.csv", index=False)

# PROD_B vs SIM_B parity
diffs = []
if "PROD_B" in matrix.columns and "SIM_B" in matrix.columns:
    for _, r in matrix.iterrows():
        prod = r["PROD_B"]
        sim = r["SIM_B"]
        if prod != sim:
            diffs.append({
                "Ticker": r["Ticker"],
                "PROD_B": prod,
                "SIM_B": sim,
            })

pd.DataFrame(diffs).to_csv(out / "crypto_prod_b_vs_sim_b_differences.csv", index=False)

parity = {
    "comparison": "PROD_B_vs_SIM_B",
    "tickers": len(matrix),
    "differences": len(diffs),
    "match_pct": (100.0 * (len(matrix) - len(diffs)) / len(matrix)) if len(matrix) else 0.0,
    "passed": len(diffs) == 0,
}

pd.DataFrame([parity]).to_csv(out / "crypto_prod_b_vs_sim_b_parity.csv", index=False)

# Minimal HTML
html = f"""
<html>
<head>
<title>Crypto Validation Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
th {{ background: #f4f4f4; }}
.pass {{ color: green; font-weight: bold; }}
.fail {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h1>Crypto Validation Summary</h1>
<p>Output folder: <code>{out}</code></p>
<h2>Gate 1 — PROD B vs SIM B</h2>
<p class="{ 'pass' if parity['passed'] else 'fail' }">
Match: {parity['match_pct']:.2f}% — Differences: {parity['differences']}
</p>
<h2>Profile counts</h2>
{summary.to_html(index=False)}
<h2>Files</h2>
<ul>
<li>crypto_validation_summary.csv</li>
<li>crypto_profile_comparison.csv</li>
<li>crypto_prod_b_vs_sim_b_differences.csv</li>
<li>crypto_prod_b_vs_sim_b_parity.csv</li>
</ul>
</body>
</html>
"""
(out / "crypto_validation_summary.html").write_text(html)

print()
print("Crypto validation summary")
print(summary.to_string(index=False))
print()
print("PROD_B vs SIM_B")
print(pd.DataFrame([parity]).to_string(index=False))
print()
print(f"Done. Results in: {out}")
PY

echo
echo "Done. Results in: $OUTDIR"
