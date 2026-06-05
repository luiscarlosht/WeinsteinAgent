# Crypto portfolio intelligence + force email

This full replacement file includes both:

- Portfolio intelligence fields:
  - unrealized_gain_pct
  - portfolio_weight_pct
  - distance_to_ma30_pct
  - distance_to_pivot_pct
  - risk_label
  - portfolio_recommendation
- Force email CLI:
  - --force-email

Test:

```bash
python3 weinstein_crypto_watcher.py \
  --config ./config.yaml \
  --only BTC-USD,ETH-USD,SOL-USD,LTC-USD \
  --force-email
```
