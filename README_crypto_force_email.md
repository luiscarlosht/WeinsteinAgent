# Crypto watcher force email

Adds:

```bash
--force-email
```

Usage:

```bash
python3 weinstein_crypto_watcher.py \
  --config ./config.yaml \
  --only BTC-USD,ETH-USD,SOL-USD,LTC-USD \
  --force-email
```

Behavior:
- Existing behavior remains unchanged.
- Without `--force-email`, email is sent only when BUY or SELLTRIG signals exist.
- With `--force-email`, email is sent even when there are no triggers.
- Dry-run still skips email even if `--force-email` is provided.
