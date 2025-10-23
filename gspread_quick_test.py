import gspread
from google.oauth2.service_account import Credentials

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Scopes: allow full Sheets + Drive access
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Path to your service account key file
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"

# URL of your shared Google Sheet
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit?usp=drive_link"

# Worksheet name to test
WORKSHEET_NAME = "Mapping"


# ------------------------------------------------------------
# AUTHENTICATION & WRITE TEST
# ------------------------------------------------------------
def main():
    print("🔑 Authorizing service account...")
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)

    print("📄 Opening Google Sheet...")
    sh = gc.open_by_url(SHEET_URL)
    ws = sh.worksheet(WORKSHEET_NAME)

    print("✏️ Writing test message...")
    ws.update("A1", "Hello from service account!")
    print("✅ Success: Data written to Google Sheets!")


if __name__ == "__main__":
    main()
