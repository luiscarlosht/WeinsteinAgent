# PROD Intraday Signal History Fix

This fix preserves transient PROD BUY/NEAR/SELL/SHORT signals that appear during the trading session but disappear before the end-of-day parity/routing jobs run.

## Why

`output/intraday_debug.csv` is the latest snapshot and is overwritten every scan. If PROD generated `NEAR` at 9:27 AM but the ticker no longer qualifies at 5:15 PM, the old reports said `PROD signals: 0`.

## What changed

### New file
- `weinstein_prod_history.py`

### Updated files
- `weinstein_intraday_watcher.py`
- `weinstein_daily_sim_prod_compare.py`
- `weinstein_prod_account_router.py`

## New output

Every intraday scan appends actionable rows to:

```text
output/prod_intraday_signal_history.csv
```

Rows include:

```text
RunUTC, RunCT, RunDateCT, SourceFile, Ticker, Signal, Reason, Price/Pivot/VolPace/etc.
```

## Report behavior after fix

Daily SIM vs PROD now shows:

```text
PROD latest snapshot signals
PROD intraday signals seen
PROD intraday vs D exact ticker/signal matches
PROD intraday vs F exact ticker/signal matches
```

PROD routing now uses:

```text
PROD intraday history if available,
otherwise latest snapshot,
otherwise SIM fallback according to d-source mode.
```

## Test

```bash
cd ~/WeinsteinAgent
python3 -m py_compile weinstein_prod_history.py weinstein_intraday_watcher.py weinstein_daily_sim_prod_compare.py weinstein_prod_account_router.py
```

Run one intraday scan manually and verify:

```bash
./run_cron_short_stack.sh
ls -lh output/prod_intraday_signal_history.csv
```

If there are no BUY/NEAR/SELL/SHORT rows in that scan, the file may not be created yet. It will be created when an actionable signal occurs.
