# Fidelity Export Sync

This adds a safer all-account ingestion mechanism for the Weinstein Trading Hub.

## Recommended files

Use the all-account Fidelity exports:

- `Portfolio_Positions_All_Accounts_Jun-04-2026.csv`
- `Accounts_History_All_Accounts.csv`

The script classifies rows into:

- Holdings
- CryptoHoldings
- Transactions
- CryptoTransactions

It keeps 401k/retirement rows in the all-account views, but marks them as non-tradable for Weinstein unless they are brokerage equity rows.

## Dry-run

```bash
cd ~/WeinsteinAgent
source .venv/bin/activate

python3 sync_fidelity_exports_to_google_sheet.py \
  --positions-csv ./Portfolio_Positions_All_Accounts_Jun-04-2026.csv \
  --history-csv ./Accounts_History_All_Accounts.csv \
  --dry-run
```

## Write Google Sheet

```bash
python3 sync_fidelity_exports_to_google_sheet.py \
  --positions-csv ./Portfolio_Positions_All_Accounts_Jun-04-2026.csv \
  --history-csv ./Accounts_History_All_Accounts.csv \
  --write-sheet
```

## Dependencies

```bash
pip install pandas numpy pyyaml gspread google-auth
```


## Classification fix

This version classifies:

- `SPAXX`, `SPAXX*`, `SPAXX**`
- money market rows
- `US DOLLARS`
- `FDIC INSURED DEPOSIT`

as `AssetClass = Cash`.

It classifies `Pending activity` rows as:

- `AssetClass = Pending`
- `TradableForWeinstein = False`

This prevents cash/pending balances from becoming false Weinstein equity positions.
