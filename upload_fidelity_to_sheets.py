import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Google API config
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit?usp=drive_link"

def upload_csv_to_sheet(sheet_name, csv_file):
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(SHEET_URL)
    ws = sh.worksheet(sheet_name)

    # Read CSV and replace sheet data
    df = pd.read_csv(csv_file)
    ws.clear()
    ws.update([df.columns.values.tolist()] + df.values.tolist())
    print(f"✅ Uploaded {csv_file} → {sheet_name}")

def main():
    upload_csv_to_sheet("Holdings", "Portfolio_Positions_Oct-23-2025.csv")
    upload_csv_to_sheet("Transactions", "Accounts_History.csv")

if __name__ == "__main__":
    main()
