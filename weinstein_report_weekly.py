#!/usr/bin/env python3
# weinstein_report_weekly.py
#
# Drop-in replacement that:
#  - Fixes credential discovery (prefers creds/gcp_service_account.json)
#  - Reads SMTP app password from config.yaml at notifications.email.smtp.app_password
#  - Keeps your output & prints the same (Weekly_Report tab + HTML/CSV + optional email)

import os
import sys
import io
import argparse
import datetime as dt
from typing import Tuple, Optional

import pandas as pd

# === optional helper first (keeps your repo compatibility) ===
try:
    from gsheets_sync import authorize_service_account as _auth_helper
    _HAVE_GSHEETS_HELPER = True
except Exception:
    _HAVE_GSHEETS_HELPER = False

# === gspread path as fallback ===
try:
    import gspread
    from google.oauth2.service_account import Credentials as _Creds
except Exception:
    gspread = None
    _Creds = None

# --------------------------------------------------------------------------------------
# Credentials: prefer creds/gcp_service_account.json, then creds/service_account.json,
# or GOOGLE_APPLICATION_CREDENTIALS. Try project helper first if available.
# --------------------------------------------------------------------------------------
def _service_account_client():
    if _HAVE_GSHEETS_HELPER:
        try:
            return _auth_helper()
        except Exception as e:
            print(f"⚠️  gsheets_sync authorize_service_account failed, falling back: {e}")

    if gspread is None or _Creds is None:
        raise RuntimeError("gspread / google-auth not available and helpers failed.")

    root = os.path.dirname(__file__)
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    candidates = [
        env_path,
        os.path.join(root, "creds", "gcp_service_account.json"),  # ← preferred
        os.path.join(root, "creds", "service_account.json"),      # ← legacy fallback
    ]
    sa_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not sa_path:
        tried = "\n".join(f"  - {p}" for p in candidates if p)
        raise FileNotFoundError("Missing Google service account credentials. Tried:\n" + tried)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = _Creds.from_service_account_file(sa_path, scopes=scopes)
    return gspread.authorize(creds)

# --------------------------------------------------------------------------------------
# Read email settings from config.yaml (and fallback env var for app password)
# --------------------------------------------------------------------------------------
def _read_smtp_from_yaml(cfg_path="config.yaml"):
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        e = ((cfg.get("notifications") or {}).get("email") or {})
        smtp = (e.get("smtp") or {})
        return {
            "enabled": bool(e.get("enabled", False)),
            "sender": e.get("sender"),
            "recipients": e.get("recipients") or [],
            "subject_prefix": e.get("subject_prefix") or "",
            "host": smtp.get("host", "smtp.gmail.com"),
            "port": int(smtp.get("port_ssl", 587)),
            "username": smtp.get("username"),
            "password": smtp.get("app_password") or os.environ.get("GMAIL_APP_PASSWORD"),
        }
    except Exception as ex:
        print(f"⚠️  Could not read SMTP config from YAML: {ex}")
        return None

# --------------------------------------------------------------------------------------
# Utility: write or replace a sheet tab with a DataFrame
# --------------------------------------------------------------------------------------
def _write_df(sh, name: str, df: pd.DataFrame):
    try:
        ws = sh.worksheet(name)
        sh.del_worksheet(ws)
    except Exception:
        pass
    ws = sh.add_worksheet(title=name, rows=max(2, len(df) + 1), cols=max(2, len(df.columns)))
    if df.empty:
        ws.update("A1", [[name]])
        return
    ws.update("A1", [list(df.columns)])
    if len(df) > 0:
        ws.update("A2", df.values.tolist())

# --------------------------------------------------------------------------------------
# Build the simple weekly summary you showed in your logs
# --------------------------------------------------------------------------------------
def build_weekly_summary(sheet_url: str) -> Tuple[str, bytes, pd.DataFrame]:
    print("📊 Generating weekly Weinstein report…")
    print("🔑 Authorizing service account…")
    gc = _service_account_client()
    sh = gc.open_by_url(sheet_url)

    # Pull holdings/open positions to build the summary table
    try:
        ws = sh.worksheet("Open_Positions")
    except Exception:
        # fallback to a common alt name if needed
        ws = sh.worksheet("Holdings")
    values = ws.get_all_values()
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)

    # Compute a minimal set of metrics so your email looks like the current format
    # We’ll try to read columns that typically exist; if missing, we’ll be robust.
    def _money(col):
        if col not in df.columns:
            return pd.Series([0.0]*len(df))
        return (df[col]
                .replace({"\\$": "", ",": ""}, regex=True)
                .replace("", "0")
                .astype(float))

    def _num(col):
        if col not in df.columns:
            return pd.Series([0.0]*len(df))
        return pd.to_numeric(df[col].replace("", "0"), errors="coerce").fillna(0.0)

    current_value = _money("Current Value")
    cost_total    = _money("Cost Basis Total")
    gain_dollar   = current_value - cost_total
    gain_pct      = (gain_dollar / cost_total.replace(0, pd.NA)).fillna(0.0) * 100.0

    total_gain = float(gain_dollar.sum())
    # "Portfolio % Gain": total gain divided by total cost
    portfolio_pct = float((gain_dollar.sum() / cost_total.sum() * 100.0) if cost_total.sum() else 0.0)
    # "Average % Gain": mean of per-position pct
    avg_pct = float(gain_pct.mean()) if len(gain_pct) else 0.0

    # Emit the pretty HTML block (current format)
    summary_html = f"""
<h2>Weinstein Weekly - Summary</h2>
<table border="1" cellspacing="0" cellpadding="6">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Gain/Loss ($)</td><td>${total_gain:,.2f}</td></tr>
  <tr><td>Portfolio % Gain</td><td>{portfolio_pct:.2f}%</td></tr>
  <tr><td>Average % Gain</td><td>{avg_pct:.2f}%</td></tr>
</table>
"""

    # Keep your “Per-position Snapshot” table
    # We’ll try to echo the columns you showed; if missing we skip gracefully.
    cols_keep = [
        "Symbol", "Description", "Quantity", "Last Price", "Current Value",
        "Cost Basis Total", "Average Cost Basis", "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent", "Recommendation",
    ]
    present = [c for c in cols_keep if c in df.columns]
    if present:
        # add computed columns if they’re missing so the table stays rich
        if "Total Gain/Loss Dollar" not in df.columns:
            df["Total Gain/Loss Dollar"] = gain_dollar.map(lambda x: f"{x:,.2f}")
        if "Total Gain/Loss Percent" not in df.columns:
            df["Total Gain/Loss Percent"] = gain_pct.map(lambda x: f"{x:.2f}%")

        snap = df[present].copy()
        # format some money/price fields if they exist
        for col in ["Last Price", "Current Value", "Cost Basis Total", "Average Cost Basis"]:
            if col in snap.columns:
                snap[col] = pd.to_numeric(snap[col].replace({"\\$": "", ",": ""}, regex=True), errors="coerce").fillna(0.0)
                snap[col] = snap[col].map(lambda x: f"${x:,.2f}")
        if "Quantity" in snap.columns:
            snap["Quantity"] = pd.to_numeric(snap["Quantity"], errors="coerce").fillna(0.0).map(lambda x: f"{x:,.2f}")

        # HTML table
        snap_html = "<h3>Per-position Snapshot</h3>\n<table border='1' cellspacing='0' cellpadding='4'>\n"
        snap_html += "<tr>" + "".join(f"<th>{c}</th>" for c in snap.columns) + "</tr>\n"
        for _, r in snap.iterrows():
            snap_html += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in snap.columns) + "</tr>\n"
        snap_html += "</table>\n"
    else:
        snap_html = "<p>(No per-position columns found to render a snapshot.)</p>"

    html_body = summary_html + "\n" + snap_html

    # CSV bytes
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # Also write the Weekly_Report tab you showed in logs
    _write_df(sh, "Weekly_Report", pd.DataFrame({
        "Metric": ["Total Gain/Loss ($)", "Portfolio % Gain", "Average % Gain"],
        "Value":  [f"${total_gain:,.2f}", f"{portfolio_pct:.2f}%", f"{avg_pct:.2f}%"],
    }))
    print("✅ Wrote Weekly_Report tab.")
    print("🎯 Done.")
    return html_body, csv_bytes, df

# --------------------------------------------------------------------------------------
# Simple SMTP sender using config.yaml
# --------------------------------------------------------------------------------------
def _send_email_with_smtp(subject: str, html_body: str, csv_bytes: bytes, cfg_path="config.yaml") -> bool:
    smtp_cfg = _read_smtp_from_yaml(cfg_path)
    if not smtp_cfg or not smtp_cfg.get("enabled"):
        print("ℹ️  Email sending is disabled in config.yaml or config not found.")
        return False

    recipients = smtp_cfg["recipients"]
    if not recipients:
        print("⚠️  Email enabled but no recipients configured under notifications.email.recipients")
        return False
    if not smtp_cfg["password"]:
        print("⚠️  Email enabled but no app password found (yaml smtp.app_password or env GMAIL_APP_PASSWORD).")
        return False

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    msg = MIMEMultipart()
    msg["From"] = smtp_cfg["sender"]
    msg["To"] = ", ".join(recipients)
    prefix = smtp_cfg.get("subject_prefix") or ""
    if prefix and not prefix.endswith(" "):
        prefix += " "
    msg["Subject"] = f"{prefix}{subject}"

    msg.attach(MIMEText(html_body, "html"))

    # attach CSV
    part = MIMEBase("application", "octet-stream")
    part.set_payload(csv_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="weekly_positions.csv"')
    msg.attach(part)

    try:
        with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
            server.starttls()
            server.login(smtp_cfg["username"], smtp_cfg["password"])
            server.sendmail(smtp_cfg["sender"], recipients, msg.as_string())
        print("📧 Email sent via SMTP.")
        return True
    except Exception as e:
        print(f"⚠️  SMTP send failed: {e}")
        return False

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="(kept for compatibility; HTML always written to output)")
    parser.add_argument("--email", action="store_true", help="Send email using SMTP settings in config.yaml")
    parser.add_argument("--attach-html", action="store_true", help="(kept for compatibility; body is HTML)")
    parser.add_argument("--sheet-url", required=True, help="Google Sheet URL")
    parser.add_argument("--config", help="(ignored; kept for compatibility)")
    args = parser.parse_args(argv)

    html_body, csv_bytes, df = build_weekly_summary(args.sheet_url)

    # Always write combined output HTML (kept same path used by run_weekly.sh)
    os.makedirs("output", exist_ok=True)
    out_html = os.path.join("output", "combined_weekly_email.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_body)
    print(f"✅ Combined weekly report written: {out_html}")

    if args.email:
        # Subject roughly like before
        subject = f"Weinstein Weekly — {dt.date.today().isoformat()}"
        ok = _send_email_with_smtp(subject, html_body, csv_bytes, cfg_path=os.environ.get("CONFIG_FILE", "config.yaml"))
        if not ok:
            print("⚠️  Email step did not complete.")
    else:
        print("ℹ️  --email not passed; skipping email.")

if __name__ == "__main__":
    main()
