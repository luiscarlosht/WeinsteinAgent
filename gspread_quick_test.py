import gspread
from google.oauth2.service_account import Credentials

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit?usp=drive_link"
WORKSHEET_NAME = "Mapping"
TEST_VALUE = "Hello from service account!"

# ------------------------------------------------------------
def main():
    print("🔑 Authorizing service account...")
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)

    print(f"📦 gspread version: {gspread.__version__}")
    print("📄 Opening Google Sheet...")
    sh = gc.open_by_url(SHEET_URL)
    ws = sh.worksheet(WORKSHEET_NAME)

    print("✏️ Writing test message to A1...")
    try:
        # Works across versions for a single cell:
        ws.update_acell("A1", TEST_VALUE)
    except Exception:
        # Fallback for newer API shape:
        ws.update(range_name="A1", values=[[TEST_VALUE]])

    # Read back to confirm
    val = ws.acell("A1").value
    print(f"✅ Success: A1 now contains -> {val!r}")

if __name__ == "__main__":
    main()
