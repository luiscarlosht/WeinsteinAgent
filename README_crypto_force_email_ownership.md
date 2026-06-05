# Crypto watcher force-email + ownership snapshot

This full replacement file preserves the ownership-aware crypto snapshot and adds:

```bash
--force-email
```

Behavior:

- Existing default behavior remains unchanged.
- Without `--force-email`, email is sent only when BUY or SELLTRIG signals exist.
- With `--force-email`, email is sent even when there are no BUY/SELL triggers.
- Snapshot includes owned crypto symbols, including BTC-USD even when it is used as the benchmark.
- Snapshot includes ownership fields:
  - owned
  - owned_qty
  - current_value
  - cost_basis
  - avg_cost
  - holding_accounts
  - ownership_action

Test:

```bash
python3 weinstein_crypto_watcher.py \
  --config ./config.yaml \
  --only BTC-USD,ETH-USD,SOL-USD,LTC-USD \
  --force-email
```
