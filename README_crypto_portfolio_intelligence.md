# Crypto portfolio intelligence

Adds portfolio intelligence fields to the Crypto Watch snapshot while preserving:

- `--force-email`
- CryptoHoldings ownership map
- BTC-USD included in snapshot
- Google service account env override
- newest weekly CSV by modified time

New snapshot fields:

- `unrealized_gain_pct`
- `portfolio_weight_pct`
- `distance_to_ma30_pct`
- `distance_to_pivot_pct`
- `risk_label`
- `portfolio_recommendation`

Test:

```bash
python3 weinstein_crypto_watcher.py \
  --config ./config.yaml \
  --only BTC-USD,ETH-USD,SOL-USD,LTC-USD \
  --force-email
```
