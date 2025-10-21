# === Weinstein Minimal Email Sender (supports notifications.email YAML structure) ===
import smtplib
import ssl
from email.message import EmailMessage
import yaml
import os

def send_email(subject: str, html_body: str, text_body: str = None, cfg_path="config.yaml"):
    # --- Load YAML config ---
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Support both layouts: notifications.email or top-level email
    email_cfg = cfg.get("notifications", {}).get("email") or cfg.get("email")
    if not email_cfg:
        raise KeyError("email configuration not found under 'notifications.email' or top-level 'email'")

    # --- Build message ---
    msg = EmailMessage()
    msg["Subject"] = f"{email_cfg.get('subject_prefix','')} {subject}".strip()
    sender_name = email_cfg.get("sender", email_cfg.get("from_name", ""))
    from_addr = email_cfg.get("from_email", email_cfg.get("sender"))
    msg["From"] = f"{sender_name} <{from_addr}>"
    msg["To"] = ", ".join(email_cfg.get("recipients") or email_cfg.get("to") or [])

    msg.set_content(text_body or html_body.replace("<br>", "\n"), subtype="plain")
    msg.add_alternative(html_body, subtype="html")

    # --- Select SMTP credentials ---
    smtp_cfg = email_cfg.get("smtp", {})
    smtp_host = smtp_cfg.get("host", "smtp.gmail.com")
    smtp_port = smtp_cfg.get("port_ssl") or smtp_cfg.get("smtp_port") or 465
    username = smtp_cfg.get("username", email_cfg.get("username"))
    password = smtp_cfg.get("app_password", email_cfg.get("app_password"))

    # --- Connect and send ---
    context = ssl.create_default_context()
    try:
        # SSL (port 465)
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as s:
            s.login(username, password)
            s.send_message(msg)
            print(f"✅ Email sent successfully via {smtp_host}:{smtp_port}")
    except smtplib.SMTPConnectError:
        # Fallback to STARTTLS (port 587)
        with smtplib.SMTP(smtp_host, 587) as s:
            s.ehlo()
            s.starttls(context=context)
            s.login(username, password)
            s.send_message(msg)
            print(f"✅ Email sent successfully via STARTTLS {smtp_host}:587")

if __name__ == "__main__":
    send_email(
        subject="Test from VM",
        html_body="<h3>It works ✅</h3><p>This is a test email sent from your VM.</p>",
        text_body="It works - this is a test email sent from your VM."
    )
