# Daily SIM vs PROD Parity Layer

## Goal

Add a daily comparison layer so PROD and SIM can be monitored side-by-side.

This keeps the lower Weinstein CORE logic unchanged.

## What it compares

- PROD: `output/intraday_debug.csv`
- SIM D: static PROD profile, regime + exposure scaled
- SIM F: META adaptive profile
- Fidelity account positions
- Account routing:
  - `X48354910` → F META
  - `Z30958579` → D control

## Files

- `account_strategy_profiles.yaml`
- `weinstein_account_profiles.py`
- `weinstein_daily_sim_prod_compare.py`
- `run_daily_sim_vs_prod_compare.sh`

## Install

Copy files into `~/WeinsteinAgent`:

```bash
cd ~/WeinsteinAgent
chmod a+x run_daily_sim_vs_prod_compare.sh
python3 -m py_compile weinstein_account_profiles.py
python3 -m py_compile weinstein_daily_sim_prod_compare.py
```

## Run after market close

```bash
cd ~/WeinsteinAgent
source .venv/bin/activate

POSITIONS_CSV=~/Portfolio_Positions_May-21-2026.csv \
SEND_EMAIL=1 \
UPLOAD_SHEETS=0 \
./run_daily_sim_vs_prod_compare.sh
```

Optional Google Sheet upload:

```bash
UPLOAD_SHEETS=1 ./run_daily_sim_vs_prod_compare.sh
```

## Outputs

Under:

```text
output/daily_parity/<timestamp>/
```

The script creates:

- `sim_D_replay_events.csv`
- `sim_F_base_events.csv`
- `sim_F_meta_equity.csv`
- `daily_prod_sim_signal_comparison_*.csv`
- `daily_account_recommendations_*.csv`
- `daily_prod_sim_summary_*.html`

## Operational use

Use this daily after market close to answer:

- Did PROD and SIM see the same BUY tickers?
- Did PROD and SIM see the same NEAR tickers?
- Did PROD and SIM see the same SELL tickers?
- What does F recommend for the main account?
- What does D recommend for the control account?
- What META mode did F choose recently?
