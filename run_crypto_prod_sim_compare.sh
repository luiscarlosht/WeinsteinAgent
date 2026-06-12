#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

source .venv/bin/activate 2>/dev/null || true
source ~/.weinstein_env 2>/dev/null || true

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="./output/crypto_prod_sim_compare/$STAMP"
mkdir -p "$OUTDIR"

run_case () {
  local name="$1"
  shift

  echo
  echo "===== Crypto $name ====="
  date

  python3 weinstein_crypto_watcher.py \
    --config ./config.yaml \
    --force-email false \
    "$@" \
    2>&1 | tee "$OUTDIR/${name}.log"

  cp -v ./output/crypto_debug.csv "$OUTDIR/${name}_crypto_debug.csv" 2>/dev/null || true
  latest_html="$(ls -1t ./output/crypto_watch_*.html 2>/dev/null | head -1 || true)"
  if [ -n "$latest_html" ]; then
    cp -v "$latest_html" "$OUTDIR/${name}_crypto_watch.html"
  fi
}

# PROD candidate B
run_case PROD_B \
  --profile B

# SIM same profile B: should match PROD_B
run_case SIM_B \
  --profile B \
  --sim

# SIM challengers
run_case SIM_D \
  --profile D \
  --sim

run_case SIM_F \
  --profile F \
  --sim

python3 - <<'PY' "$OUTDIR"
import sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
cases = ["PROD_B", "SIM_B", "SIM_D", "SIM_F"]

rows = []
dfs = {}

for case in cases:
    p = out / f"{case}_crypto_debug.csv"
    if not p.exists():
        rows.append({"case": case, "status": "MISSING_CSV"})
        continue

    df = pd.read_csv(p)
    dfs[case] = df

    buy = int((df.get("State", "") == "TRIGGERED").sum()) if "State" in df else 0
    near = int((df.get("State", "").isin(["NEAR", "ARMED"])).sum()) if "State" in df else 0
    sell = int((df.get("SellConfirm", False).astype(str).str.lower() == "true").sum()) if "SellConfirm" in df else 0

    rows.append({
        "case": case,
        "rows": len(df),
        "buy_triggers": buy,
        "near_or_armed": near,
        "sell_confirm": sell,
    })

summary = pd.DataFrame(rows)
summary.to_csv(out / "crypto_prod_sim_summary.csv", index=False)

# PROD_B vs SIM_B parity
parity = []
if "PROD_B" in dfs and "SIM_B" in dfs:
    key = "Ticker" if "Ticker" in dfs["PROD_B"].columns else "ticker"
    compare_cols = [
        c for c in [
            "Ticker", "Price", "Stage", "SMA150", "Pivot",
            "CoreBuySignal", "CoreNearSignal", "State",
            "SellState", "SellConfirm", "SellRiskWatch",
            "PortfolioRecommendation"
        ]
        if c in dfs["PROD_B"].columns and c in dfs["SIM_B"].columns
    ]

    a = dfs["PROD_B"][compare_cols].copy()
    b = dfs["SIM_B"][compare_cols].copy()

    merged = a.merge(b, on=key, suffixes=("_PROD_B", "_SIM_B"), how="outer", indicator=True)

    diffs = []
    for _, r in merged.iterrows():
        ticker = r.get(key)
        if r["_merge"] != "both":
            diffs.append({"ticker": ticker, "field": "_merge", "prod": "", "sim": r["_merge"]})
            continue

        for c in compare_cols:
            if c == key:
                continue
            pv = r.get(f"{c}_PROD_B")
            sv = r.get(f"{c}_SIM_B")
            if str(pv) != str(sv):
                diffs.append({"ticker": ticker, "field": c, "prod": pv, "sim": sv})

    pd.DataFrame(diffs).to_csv(out / "PROD_B_vs_SIM_B_differences.csv", index=False)
    parity.append({
        "comparison": "PROD_B_vs_SIM_B",
        "difference_count": len(diffs),
        "match": len(diffs) == 0,
    })

pd.DataFrame(parity).to_csv(out / "crypto_prod_sim_parity.csv", index=False)

print(f"Summary written to: {out}")
print(summary.to_string(index=False))
PY

echo
echo "Done. Results in: $OUTDIR"
