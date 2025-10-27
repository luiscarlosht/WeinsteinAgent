# WeinsteinMinimalEmailSender.py
import os
import smtplib
import mimetypes
from typing import List, Optional
from email.message import EmailMessage

from email_config import load_email_settings

def _attach_files(msg: EmailMessage, filepaths: List[str]) -> None:
    for path in filepaths or []:
        if not path or not os.path.exists(path):
            continue
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=os.path.basename(path),
            )

def send_email_with_yaml(
    subject_suffix: str,
    html_body: str,
    attachments: Optional[List[str]] = None,
    config_path: str = "./config.yaml",
) -> bool:
    """
    Reads notifications.email.* from config.yaml and sends email via SMTP.
    Supports HTML body and optional attachments.
    Returns True on success, False if disabled or misconfigured.
    """
    settings = load_email_settings(config_path)

    if not settings.enabled:
        print("Email disabled in YAML (notifications.email.enabled=false); skipping.")
        return False
    if settings.provider != "smtp":
        print(f"Unsupported email provider '{settings.provider}'. Only 'smtp' is implemented.")
        return False
    if not settings.sender or not settings.recipients:
        print("Email settings missing sender/recipients; skipping.")
        return False
    if not settings.smtp.host or not settings.smtp.port:
        print("SMTP host/port missing; skipping.")
        return False
    if settings.smtp.username and not settings.smtp.password:
        print("SMTP username set but no password (smtp.app_password missing and EMAIL_SMTP_PASS empty); skipping.")
        return False

    subject = f"{settings.subject_prefix} {subject_suffix}".strip()

    msg = EmailMessage()
    msg["From"] = settings.sender
    msg["To"] = ", ".join(settings.recipients)
    msg["Subject"] = subject
    msg.set_content("Your email client does not support HTML.")
    msg.add_alternative(html_body or "", subtype="html")

    _attach_files(msg, attachments or [])

    if settings.smtp.use_ssl:
        with smtplib.SMTP_SSL(settings.smtp.host, settings.smtp.port) as s:
            if settings.smtp.username:
                s.login(settings.smtp.username, settings.smtp.password or "")
            s.send_message(msg)
    else:
        with smtplib.SMTP(settings.smtp.host, settings.smtp.port) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            if settings.smtp.username:
                s.login(settings.smtp.username, settings.smtp.password or "")
            s.send_message(msg)

    print(f"Email sent to: {', '.join(settings.recipients)}")
    return True
