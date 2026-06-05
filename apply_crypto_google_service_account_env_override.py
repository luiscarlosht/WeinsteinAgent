#!/usr/bin/env python3
"""
apply_crypto_google_service_account_env_override.py

Patches weinstein_crypto_watcher.py so Google service account path is resolved as:

1. GOOGLE_SERVICE_ACCOUNT_JSON
2. GOOGLE_APPLICATION_CREDENTIALS
3. config.yaml google.service_account_json

This lets Google VM and DigitalOcean use the same config.yaml while each machine
points to its local credential path from ~/.weinstein_env.
"""

from pathlib import Path

target = Path("weinstein_crypto_watcher.py")
if not target.exists():
    raise SystemExit("Run this from the WeinsteinAgent repo root. Missing weinstein_crypto_watcher.py")

s = target.read_text()

old = '    svc_file = google.get("service_account_json")'
new = """    svc_file = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or google.get("service_account_json")
    )"""

if old not in s:
    if "GOOGLE_SERVICE_ACCOUNT_JSON" in s and "GOOGLE_APPLICATION_CREDENTIALS" in s:
        print("Already patched: env override exists in weinstein_crypto_watcher.py")
        raise SystemExit(0)
    raise SystemExit("Target line not found. Please inspect load_config() in weinstein_crypto_watcher.py")

target.write_text(s.replace(old, new, 1))
print("Patched weinstein_crypto_watcher.py")
