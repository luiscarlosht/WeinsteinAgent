import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)
gc = gspread.authorize(creds)

# Use the full URL of the Sheet you just shared
sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit?usp=drive_link")
ws = sh.worksheet("Sheet9")  # or sh.sheet1
ws.update("A1", "Hello from service account!")  # should succeed now
print("OK")
