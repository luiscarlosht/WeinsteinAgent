# PROD vs SIM Timing Research + Google Positions Sync

## 1. PROD vs SIM Timing Research

Run:

```bash
cd ~/WeinsteinAgent
chmod a+x run_prod_sim_timing_research.sh
python3 -m py_compile weinstein_prod_sim_timing_research.py

./run_prod_sim_timing_research.sh
```

Outputs:

```text
output/prod_sim_timing_research/
```

This compares:

```text
output/prod_intraday_signal_history.csv
```

against the latest:

```text
output/daily_parity/<latest>/
```

## 2. Google Sheets → current_positions.csv

Add to `~/.weinstein_env`:

```bash
export POSITIONS_GOOGLE_SHEET_ID="YOUR_SHEET_ID"
export POSITIONS_GOOGLE_SHEET_TAB="Open_Positions"
export POSITIONS_CSV_OUT="/home/luiscarlosht/WeinsteinAgent/current_positions.csv"
```

Then test:

```bash
source ~/.weinstein_env
python3 -m py_compile sync_positions_from_google_sheet.py
./run_sync_positions_from_google_sheet.sh
```

If the sheet is private and CSV export fails, publish/share the tab appropriately or use a full CSV export URL:

```bash
export POSITIONS_GOOGLE_CSV_URL="https://docs.google.com/spreadsheets/d/.../gviz/tq?tqx=out:csv&sheet=Open_Positions"
```

## 3. Cron update

Use the synced stable file:

```cron
35 21 * * 1-5 /bin/bash -lc 'source ~/.weinstein_env && cd /home/luiscarlosht/WeinsteinAgent && source .venv/bin/activate && ./run_sync_positions_from_google_sheet.sh >> /home/luiscarlosht/WeinsteinAgent/cron_positions_sync.log 2>&1'

45 21 * * 1-5 /bin/bash -lc 'source ~/.weinstein_env && cd /home/luiscarlosht/WeinsteinAgent && source .venv/bin/activate && POSITIONS_CSV=/home/luiscarlosht/WeinsteinAgent/current_positions.csv SEND_EMAIL=1 UPLOAD_SHEETS=0 ./run_daily_sim_vs_prod_compare.sh >> /home/luiscarlosht/WeinsteinAgent/cron_daily_parity.log 2>&1'

15 22 * * 1-5 /bin/bash -lc 'source ~/.weinstein_env && cd /home/luiscarlosht/WeinsteinAgent && source .venv/bin/activate && POSITIONS_CSV=/home/luiscarlosht/WeinsteinAgent/current_positions.csv SEND_EMAIL=1 ./run_prod_account_routing_email.sh >> /home/luiscarlosht/WeinsteinAgent/cron_prod_account_routing.log 2>&1'

20 22 * * 1-5 /bin/bash -lc 'source ~/.weinstein_env && cd /home/luiscarlosht/WeinsteinAgent && source .venv/bin/activate && ./run_prod_sim_timing_research.sh >> /home/luiscarlosht/WeinsteinAgent/cron_prod_sim_timing_research.log 2>&1'
```
