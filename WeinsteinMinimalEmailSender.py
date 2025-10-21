# === Weinstein Minimal Email Sender (auto SSL/STARTTLS by port) ===
import smtplib
import ssl
from email.message import EmailMessage
import yaml
from datetime import datetime

def send_email(subject: str, html_body: str, text_body: str = None, cfg_path="config.yaml"):
    # --- Load YAML config ---
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Support both layouts: notifications.email or top-level email
    email_cfg = cfg.get("notifications", {}).get("email") or cfg.get("email")
    if not email_cfg:
        raise KeyError("email configuration not found under 'notifications.email' or top-level 'email'")

    if not email_cfg.get("enabled", True):
        print("Email notifications are disabled in config.")
        return

    # --- Build message ---
    sender_name = email_cfg.get("from_name") or email_cfg.get("sender", "")
    from_addr   = email_cfg.get("from_email") or email_cfg.get("sender")
    recipients  = email_cfg.get("recipients") or email_cfg.get("to") or []
    if not from_addr or not recipients:
        raise ValueError("Missing sender or recipients in email config.")

    subj_prefix = email_cfg.get("subject_prefix", "").strip()
    final_subject = f"{subj_prefix} {subject}".strip()

    msg = EmailMessage()
    msg["Subject"] = final_subject
    msg["From"] = f"{sender_name} <{from_addr}>" if sender_name else from_addr
    msg["To"] = ", ".join(recipients)
    msg.set_content(text_body or html_body.replace("<br>", "\n"), subtype="plain")
    msg.add_alternative(html_body, subtype="html")

    # --- SMTP settings ---
    smtp_cfg  = email_cfg.get("smtp", {})
    smtp_host = smtp_cfg.get("host", "smtp.gmail.com")
    # Accept common keys, default to 465 (SSL)
    port_ssl  = smtp_cfg.get("port_ssl")
    port_any  = smtp_cfg.get("port") or smtp_cfg.get("smtp_port")
    smtp_port = port_ssl or port_any or 465

    username  = smtp_cfg.get("username") or email_cfg.get("username") or from_addr
    password  = smtp_cfg.get("app_password") or email_cfg.get("app_password")
    if not password:
        raise ValueError("Missing app_password for SMTP login.")

    # --- Choose transport by port: 465 -> SSL, 587 -> STARTTLS, else try sensible default ---
    context = ssl.create_default_context()

    def send_via_ssl():
        with smtplib.SMTP_SSL(smtp_host, 465, context=context) as s:
            s.login(username, password)
            s.send_message(msg)
        print(f"✅ Email sent via SSL: {smtp_host}:465")

    def send_via_starttls():
        with smtplib.SMTP(smtp_host, 587) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(username, password)
            s.send_message(msg)
        print(f"✅ Email sent via STARTTLS: {smtp_host}:587")

    try:
        if smtp_port == 465:
            send_via_ssl()
        elif smtp_port == 587:
            send_via_starttls()
        else:
            # If some other port was given, guess: 465 -> SSL else STARTTLS
            if smtp_port in (25, 587):
                send_via_starttls()
            else:
                send_via_ssl()
    except smtplib.SMTPServerDisconnected as e:
        print(f"Server disconnected: {e}. Falling back to STARTTLS...")
        send_via_starttls()
    except ssl.SSLError as e:
        print(f"SSL error ({e}). Retrying with STARTTLS...")
        send_via_starttls()

if __name__ == "__main__":
    today = datetime.now().strftime("%b %d, %Y")
    send_email(
        subject=f"Test from VM – {today}",
        html_body="<h3>It works ✅</h3><p>This is a test email sent from your VM.</p>",
        text_body="It works - this is a test email sent from your VM."
    )
