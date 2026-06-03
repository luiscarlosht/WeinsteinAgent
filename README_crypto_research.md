# Crypto Weinstein A/B/C/D/E/F Research

Research-only crypto A/B/C/D/E/F runner.

This does not change:
- stock PROD
- stock SIM
- stock META F
- stock cron
- live trading execution

## Shorting

For live Fidelity Crypto, assume no shorting unless the broker explicitly supports it.
This first research runner is long-only.

## Validate on VM

```bash
cd ~/WeinsteinAgent
git pull
source .venv/bin/activate
python3 -m py_compile crypto_abcd_ef_research.py
chmod +x run_crypto_research.sh
```

## Fidelity-only universe

```bash
START_DATE=2020-01-01 FIDELITY_ONLY=1 ./run_crypto_research.sh
```

## Broader universe

```bash
START_DATE=2020-01-01 ./run_crypto_research.sh
```

## Output

```text
output/crypto_research/YYYYMMDD_HHMMSS/
```

Key files:
- crypto_profile_summary.csv
- crypto_yearly_returns.csv
- crypto_research_summary_*.html
- crypto_A_events/equity/trades.csv through crypto_F_*.csv
