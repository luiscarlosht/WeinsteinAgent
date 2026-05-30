# Google Sheet Positions as Source of Truth

Goal:
- Stop copying Fidelity CSV into `current_positions.csv`.
- Use the Google Sheet `Holdings` tab as the single source of truth.
- Nightly SIM-vs-PROD and PROD routing should not become stale after a portfolio change.

## Why `Holdings` tab?

Use `Holdings`, not `Open_Positions`.

`Holdings` is the raw Fidelity positions upload and includes account rows such as AAPL, AMD, ANET, DELL, GOOG, GOOGL, MS, SNDK.

`Open_Positions` is a derived performance tab and may only show a subset.

## Install

```bash
cd ~/WeinsteinAgent
unzip ~/weinstein_google_positions_source_package.zip -d .
chmod a+x test_google_sheet_positions_source.sh
python3 -m py_compile weinstein_positions_source.py apply_google_sheet_positions_source_patch.py
```

## Configure

Add to `~/.weinstein_env`:

```bash
export WEINSTEIN_POSITIONS_SHEET_ID="17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4"
export WEINSTEIN_POSITIONS_TAB="Holdings"
```

If your service account file is not the default:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/home/luiscarlosht/WeinsteinAgent/creds/gcp_service_account.json"
```

## Patch account profile loader

```bash
python3 apply_google_sheet_positions_source_patch.py
python3 -m py_compile weinstein_account_profiles.py
```

## Test

```bash
source ~/.weinstein_env
./test_google_sheet_positions_source.sh
```

You should see DELL from the Google Sheet holdings.

## Update cron

Replace static CSV usage with:

```bash
POSITIONS_SOURCE=GOOGLE_SHEET POSITIONS_CSV=GOOGLE_SHEET
```

Recommended active cron lines:

```cron
45 21 * * 1-5 /bin/bash -lc 'source ~/.weinstein_env && cd /home/luiscarlosht/WeinsteinAgent && source .venv/bin/activate && POSITIONS_SOURCE=GOOGLE_SHEET POSITIONS_CSV=GOOGLE_SHEET SEND_EMAIL=1 UPLOAD_SHEETS=0 ./run_daily_sim_vs_prod_compare.sh >> /home/luiscarlosht/WeinsteinAgent/cron_daily_parity.log 2>&1'

15 22 * * 1-5 /bin/bash -lc 'source ~/.weinstein_env && cd /home/luiscarlosht/WeinsteinAgent && source .venv/bin/activate && POSITIONS_SOURCE=GOOGLE_SHEET POSITIONS_CSV=GOOGLE_SHEET SEND_EMAIL=1 ./run_prod_account_routing_email.sh >> /home/luiscarlosht/WeinsteinAgent/cron_prod_account_routing.log 2>&1'
```

## Manual tests

```bash
POSITIONS_SOURCE=GOOGLE_SHEET POSITIONS_CSV=GOOGLE_SHEET SEND_EMAIL=1 ./run_prod_account_routing_email.sh
```

Expected:
- DELL should be detected as owned after you uploaded May 29 holdings to Google Sheets.
