# Crypto watcher Google service account env override

This package adds a small patcher script rather than replacing the full watcher file.

Reason:
- `weinstein_crypto_watcher.py` changes frequently.
- A tiny in-place patch is safer than overwriting the full file.
- It preserves all current code and only changes service account path resolution.

## What it changes

In `weinstein_crypto_watcher.py`:

```python
svc_file = google.get("service_account_json")
```

becomes:

```python
svc_file = (
    os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    or google.get("service_account_json")
)
```

## DigitalOcean env

```bash
cat >> ~/.weinstein_env <<'EOF'

# Google Sheets service account path for DigitalOcean
export GOOGLE_SERVICE_ACCOUNT_JSON="/root/WeinsteinAgent/creds/gcp_service_account.json"
export GOOGLE_APPLICATION_CREDENTIALS="/root/WeinsteinAgent/creds/gcp_service_account.json"
EOF
source ~/.weinstein_env
```

## Google VM env

```bash
cat >> ~/.weinstein_env <<'EOF'

# Google Sheets service account path for Google VM
export GOOGLE_SERVICE_ACCOUNT_JSON="/home/luiscarlosht/WeinsteinAgent/creds/gcp_service_account.json"
export GOOGLE_APPLICATION_CREDENTIALS="/home/luiscarlosht/WeinsteinAgent/creds/gcp_service_account.json"
EOF
source ~/.weinstein_env
```
