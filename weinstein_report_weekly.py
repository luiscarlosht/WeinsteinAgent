#!/usr/bin/env python3
# weinstein_report_weekly.py
#
# Weekly report generator:
# - Credential discovery prefers creds/gcp_service_account.json
# - SMTP app password read from config.yaml (notifications.email.smtp.app_password)
# - Builds portfolio summary + per-position snapshot from your Google Sheet
# - If ./output/scan_sp500.csv exists, merges the classic Weinstein scan and renders:
#     "Weinstein Weekly — Benchmark: SPY", generated timestamp,
#     Buy/Watch/Avoid counts, and the scan table (robust column matching).
# - Writes output/combined_weekly_email.html
# - Email sending:
#     * --email                  -> send email (new flag)
#     * --attach-html (legacy)   -> send email (compat with run_weekly.sh)
#     * --write (legacy)         -> write HTML (already default; accepted & ignored)
#
# Exit code 0 on success; non-zero on obvious failures.

import os
import io
import sys
import argparse
import datetime as dt
from typing import Tuple, Optional, Dict, List

import pandas as pd

# Optional helper first (repo compatibility)
try:
    from gsheets_sync import authorize_service_account as _auth_helper
    _HAVE_GSHEETS_HELPER = True
except Exception:
    _HAVE_GSHEETS_HELPER = False

# gspread as fallback
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
        os.path.join(root, "creds", "gcp_service_account.json"),  # preferred
        os.path.join(root, "creds", "service_account.json"),      # legacy fallback
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
# Read email settings from config.yaml (with env fallback for app password)
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
# Utility: write/replace a sheet tab with a DataFrame
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
# Portfolio block (unchanged behavior, resilient to missing columns)
# --------------------------------------------------------------------------------------
def _build_portfolio_block(sh) -> Tuple[str, bytes, pd.DataFrame]:
    # Load Open_Positions (fallback to Holdings)
    try:
        ws = sh.worksheet("Open_Positions")
    except Exception:
        ws = sh.worksheet("Holdings")
    values = ws.get_all_values()
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)

    def _money(series: pd.Series) -> pd.Series:
        return (series.replace({"\\$": "", ",": ""}, regex=True)
                      .replace("", "0")
                      .pipe(pd.to_numeric, errors="coerce")
                      .fillna(0.0))

    def _get_money(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([0.0]*len(df))
        return _money(df[col])

    cost_total = _get_money("Cost Basis Total")
    current_value = _get_money("Current Value")
    gain_dollar = current_value - cost_total
    gain_pct = (gain_dollar / cost_total.replace(0, pd.NA)).fillna(0.0) * 100.0

    total_gain = float(gain_dollar.sum())
    portfolio_pct = float((gain_dollar.sum() / cost_total.sum() * 100.0) if cost_total.sum() else 0.0)
    avg_pct = float(gain_pct.mean() if len(gain_pct) else 0.0)

    summary_html = f"""
<h2>Weinstein Weekly - Summary</h2>
<table border="1" cellspacing="0" cellpadding="6">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Gain/Loss ($)</td><td>${total_gain:,.2f}</td></tr>
  <tr><td>Portfolio % Gain</td><td>{portfolio_pct:.2f}%</td></tr>
  <tr><td>Average % Gain</td><td>{avg_pct:.2f}%</td></tr>
</table>
"""

    # Per-position snapshot table (if columns exist)
    cols_keep = [
        "Symbol", "Description", "Quantity", "Last Price", "Current Value",
        "Cost Basis Total", "Average Cost Basis", "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent", "Recommendation",
    ]
    present = [c for c in cols_keep if c in df.columns]
    if present:
        # enrich if missing
        if "Total Gain/Loss Dollar" not in df.columns:
            df["Total Gain/Loss Dollar"] = gain_dollar.map(lambda x: f"{x:,.2f}")
        if "Total Gain/Loss Percent" not in df.columns:
            df["Total Gain/Loss Percent"] = gain_pct.map(lambda x: f"{x:.2f}%")

        snap = df[present].copy()
        for col in ["Last Price", "Current Value", "Cost Basis Total", "Average Cost Basis"]:
            if col in snap.columns:
                snap[col] = pd.to_numeric(
                    snap[col].replace({"\\$": "", ",": ""}, regex=True),
                    errors="coerce"
                ).fillna(0.0).map(lambda x: f"${x:,.2f}")
        if "Quantity" in snap.columns:
            snap["Quantity"] = pd.to_numeric(snap["Quantity"], errors="coerce").fillna(0.0).map(lambda x: f"{x:,.2f}")

        snap_html = "<h3>Per-position Snapshot</h3>\n<table border='1' cellspacing='0' cellpadding='4'>\n"
        snap_html += "<tr>" + "".join(f"<th>{c}</th>" for c in snap.columns) + "</tr>\n"
        for _, r in snap.iterrows():
            snap_html += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in snap.columns) + "</tr>\n"
        snap_html += "</table>\n"
    else:
        snap_html = "<p>(No per-position columns found to render a snapshot.)</p>"

    # CSV for attachment
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # Also write Weekly_Report tab
    _write_df(sh, "Weekly_Report", pd.DataFrame({
        "Metric": ["Total Gain/Loss ($)", "Portfolio % Gain", "Average % Gain"],
        "Value":  [f"${total_gain:,.2f}", f"{portfolio_pct:.2f}%", f"{avg_pct:.2f}%"],
    }))
    print("✅ Wrote Weekly_Report tab.")
    print("🎯 Done.")

    return summary_html + "\n" + snap_html, csv_bytes, df


# --------------------------------------------------------------------------------------
# Classic scan loader + renderer (robust to column naming)
# --------------------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df.columns = [mapping[c] for c in df.columns]
    return df

def _pick_first(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def _label_to_bucket(val: str) -> str:
    s = (val or "").strip().lower()
    if s.startswith("buy") or s == "long" or s == "accumulate":
        return "Buy"
    if s.startswith("watch") or s.startswith("hold") or "basing" in s:
        return "Watch"
    return "Avoid"

def _render_classic_block(csv_path: str, benchmark: str = "SPY") -> Optional[str]:
    if not os.path.exists(csv_path):
        print(f"ℹ️  Classic scan CSV not found at {csv_path}. Skipping classic merge.")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("ℹ️  Classic scan CSV is empty. Skipping classic merge.")
        return None

    df = _normalize_columns(df)

    # identify key columns (robust to names)
    col_ticker = _pick_first(df, ["ticker", "symbol"])
    col_label  = _pick_first(df, ["buy_signal", "signal", "recommendation", "verdict", "label", "rating"])
    col_stage  = _pick_first(df, ["stage", "weinstein_stage"])
    col_price  = _pick_first(df, ["price", "last", "close"])
    col_ma30   = _pick_first(df, ["ma30", "sma_30", "sma30"])
    col_dist   = _pick_first(df, ["dist_ma_pct", "dist_from_ma_pct", "price_to_ma_pct"])
    col_slope  = _pick_first(df, ["ma_slope_per_wk", "ma_slope_weekly", "ma_trend_wk"])
    col_rs     = _pick_first(df, ["rs", "relative_strength"])
    col_rsma   = _pick_first(df, ["rs_ma30", "rs_ma_30", "rs_sma30"])
    col_rs_ab  = _pick_first(df, ["rs_above_ma", "rs_gt_ma", "rs_vs_ma"])
    col_rsslp  = _pick_first(df, ["rs_slope_per_wk", "rs_slope_weekly"])
    col_notes  = _pick_first(df, ["notes", "comment", "reason"])

    # derive counts
    if col_label is None:
        df["__label__"] = "Avoid"
        col_label = "__label__"
    buckets = df[col_label].fillna("").map(_label_to_bucket)
    buy_n = int((buckets == "Buy").sum())
    watch_n = int((buckets == "Watch").sum())
    avoid_n = int((buckets == "Avoid").sum())
    total_n = int(len(df))

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Prepare rendered DF with preferred columns
    out_cols = []
    def _add(col_name: Optional[str], title: str):
        nonlocal out_cols
        if col_name and col_name in df.columns:
            out_cols.append((col_name, title))

    _add(col_ticker, "ticker")

    pretty_label_col = "__pretty_label__"
    df[pretty_label_col] = buckets
    _add(pretty_label_col, "Buy Signal")

    chart_col = "__chart__"
    def _chart_link(t):
        t = (t or "").strip().upper()
        return f'<a href="https://www.tradingview.com/symbols/{t}/" target="_blank">chart</a>' if t else ""
    df[chart_col] = df[col_ticker].map(_chart_link) if col_ticker else ""
    _add(chart_col, "chart")

    _add(col_stage, "stage")
    _add(col_price, "price")
    _add(col_ma30, "ma30")
    _add(col_dist, "dist_ma_pct")
    _add(col_slope, "ma_slope_per_wk")
    _add(col_rs, "rs")
    _add(col_rsma, "rs_ma30")
    _add(col_rs_ab, "rs_above_ma")
    _add(col_rsslp, "rs_slope_per_wk")
    _add(col_notes, "notes")

    if not out_cols:
        out_df = df
        headers = list(out_df.columns)
    else:
        headers = [t for _, t in out_cols]
        out_df = pd.DataFrame({t: df[c] for c, t in out_cols})

    hdr = f"""
<h2>Weinstein Weekly — Benchmark: {benchmark}</h2>
<p><em>Generated {ts}</em></p>
<p><strong>Summary:</strong> ✅ Buy: {buy_n} &nbsp;&nbsp;|&nbsp;&nbsp; 🟡 Watch: {watch_n} &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 Avoid: {avoid_n} &nbsp;&nbsp; (Total: {total_n})</p>
"""
    table = "<table border='1' cellspacing='0' cellpadding='4'>\n"
    table += "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>\n"
    for _, r in out_df.iterrows():
        table += "<tr>" + "".join(f"<td>{r[h]}</td>" for h in headers) + "</tr>\n"
        # no truncation
    table += "</table>\n"
    return hdr + table


# --------------------------------------------------------------------------------------
# Top-level builder
# --------------------------------------------------------------------------------------
def build_weekly_summary(sheet_url: str, classic_csv: str = "./output/scan_sp500.csv") -> Tuple[str, bytes, pd.DataFrame]:
    print("📊 Generating weekly Weinstein report…")
    print("🔑 Authorizing service account…")
    gc = _service_account_client()
    sh = gc.open_by_url(sheet_url)

    portfolio_html, csv_bytes, df_positions = _build_portfolio_block(sh)
    classic_html = _render_classic_block(classic_csv, benchmark="SPY")

    parts = []
    if classic_html:
        parts.append(classic_html)
    parts.append(portfolio_html)
    html_body = "\n<hr/>\n".join(parts)

    return html_body, csv_bytes, df_positions


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
    msg["Subject"] = f"{prefix}Weinstein Weekly — {dt.date.today().isoformat()}"

    msg.attach(MIMEText(html_body, "html"))

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
    # New/primary flags
    parser.add_argument("--email", action="store_true", help="Send email using SMTP settings in config.yaml")
    parser.add_argument("--sheet-url", required=True, help="Google Sheet URL")
    parser.add_argument("--classic-csv", default="./output/scan_sp500.csv", help="Path to classic scan CSV")

    # Legacy flags (accepted for compatibility with run_weekly.sh)
    parser.add_argument("--write", action="store_true", help="(legacy) Write HTML (always done)")
    parser.add_argument("--attach-html", action="store_true", help="(legacy) Also send email")

    args = parser.parse_args(argv)

    html_body, csv_bytes, _ = build_weekly_summary(args.sheet_url, classic_csv=args.classic_csv)

    os.makedirs("output", exist_ok=True)
    out_html = os.path.join("output", "combined_weekly_email.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_body)
    print(f"✅ Combined weekly report written: {out_html}")

    # Determine whether to email
    send_email = bool(args.email or args.attach_html)
    if send_email:
        ok = _send_email_with_smtp(
            subject=f"Weinstein Weekly — {dt.date.today().isoformat()}",
            html_body=html_body,
            csv_bytes=csv_bytes,
            cfg_path=os.environ.get("CONFIG_FILE", "config.yaml"),
        )
        if not ok:
            print("⚠️  Email step did not complete.")
    else:
        print("ℹ️  Email not requested (pass --email or legacy --attach-html to send).")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # propagate argparse exit codes intact
        raise
    except Exception as ex:
        print(f"❌ Fatal error: {ex}")
        sys.exit(1)
