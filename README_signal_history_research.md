# Weinstein PROD Signal History + HTML Research

Implements:

1. Historical reconstruction from `output/intraday_watch_*.html`.
2. Future durable tracking via `output/intraday_signal_history.csv`.

## Install/test

```bash
cd ~/WeinsteinAgent
chmod a+x run_html_signal_research.sh package_prod_signal_history_outputs.sh
python3 -m py_compile weinstein_html_signal_research.py weinstein_intraday_signal_history.py apply_intraday_signal_history_patch.py
```

## Step 1: Historical HTML research

```bash
DAYS=45 ./run_html_signal_research.sh
```

Outputs:

```text
output/html_signal_research/prod_signal_history_from_html_*.csv
output/html_signal_research/prod_signal_daily_summary_*.csv
output/html_signal_research/prod_signal_lifecycle_*.csv
output/html_signal_research/prod_signal_history_research_*.html
```

The parser ignores generic explanatory HTML like:

```html
<b>BUY:</b> Weekly Stage 2 breakout confirmed...
```

and focuses on real ticker rows/list items such as:

```text
DELL @ 423.87 ... BUY: px=423.87 pivot=420.00 vol=1.73x adx=54.3
```

## Step 2: Durable future tracking

```bash
python3 apply_intraday_signal_history_patch.py
python3 -m py_compile weinstein_intraday_watcher.py
```

After the next intraday run, rows are appended to:

```text
output/intraday_signal_history.csv
```

## Package outputs

```bash
./package_prod_signal_history_outputs.sh
```

## Suggested commit

```bash
git add weinstein_html_signal_research.py weinstein_intraday_signal_history.py apply_intraday_signal_history_patch.py run_html_signal_research.sh package_prod_signal_history_outputs.sh weinstein_intraday_watcher.py
git commit -m "Add PROD intraday signal history and HTML signal research"
git push
```
