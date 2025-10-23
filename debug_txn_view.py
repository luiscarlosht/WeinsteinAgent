#!/usr/bin/env python3
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone

SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

def auth():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def get_df(ws):
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame()
    header, data = rows[0], rows[1:]
    data = [r for r in data if any(c.strip() for c in r)]
    return pd.DataFrame(data, columns=header) if data else pd.DataFrame(columns=header)

def main():
    gc = auth()
    sh = gc.open_by_url(SHEET_URL)
    ws = sh.worksheet("Transactions")
    df = get_df(ws)

    # Show last 30 parsed rows with the raw Symbol/Description and Action
    show_cols = [c for c in df.columns if any(k in c.lower() for k in ["symbol","descr","action","type","date","time","price","amount","qty"])]
    print("Columns of interest:", show_cols)
    print(df[show_cols].tail(30).to_string(index=False))

if __name__ == "__main__":
    main()
