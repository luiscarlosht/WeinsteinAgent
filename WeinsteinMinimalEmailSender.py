# WeinsteinMinimalEmailSender.py
# Lightweight email helper used by weekly/intraday scripts.

from __future__ import annotations

import os
import smtplib
import ssl
import mimetypes
from email.message import EmailMessage
from typing import List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: str = "config.yaml") -> dict:
    cfg = {}
    if yaml is None:
        return cfg
    if not os.path.exists(path):
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _email_cfg() -> dict:
    cfg = _load_config()
    return ((cfg.get("notifications") or {}).get("email") or {})


# ──────────────────────────────────────────────────────────────────────────────
# Message building
# ──────────────────────────────────────────────────────────────────────────────

def _guess_mime_type(path: str) -> Tuple[str, str]:
    mtype, _ = mimetypes.guess_type(path)
    if not mtype:
        return "application", "octet-stream"
    maintype, subtype = mtype.split("/", 1)
    return maintype, subtype


def _make_multipart_message(
    sender: str,
    to: List[str],
    subject: str,
    text_body: Optional[str] = None,
    html_body: Optional[str] = None,
    attachments: Optional[List[str]] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
) -> Tuple[EmailMessage, List[str]]:
    """
    Build an EmailMessage and return it plus the full list of recipients (to+cc+bcc).
    """
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(to) if to else ""
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject or ""

    # Text/HTML alternative
    if html_body:
        if text_body:
            msg.set_content(text_body)
        else:
            # Provide a plaintext fallback if only HTML is provided
            msg.set_content("This message contains HTML content. Please view in an HTML-capable client.")
        msg.add_alternative(html_body, subtype="html")
    else:
        msg.set_content(text_body or "")

    # Attachments
    attachments = attachments or []
    for path in attachments:
        if not path or not os.path.exists(path):
            continue
        maintype, subtype = _guess_mime_type(path)
        with open(path, "rb") as fh:
            data = fh.read()
        filename = os.path.basename(path)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

    all_recipients = list(to or [])
    if cc:
        all_recipients.extend(cc)
    if bcc:
        all_recipients.extend(bcc)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in all_recipients:
        if not r:
            continue
        if r not in seen:
            unique.append(r)
            seen.add(r)

    return msg, unique


# ──────────────────────────────────────────────────────────────────────────────
# Sender backends
# ──────────────────────────────────────────────────────────────────────────────

def _send_via_smtp(
    msg: EmailMessage,
    recipients: List[str],
    host: str,
    port: int,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_starttls: bool = True,
) -> None:
    if use_starttls:
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if username and password:
                server.login(username, password)
            server.send_message(msg, to_addrs=recipients)
    else:
        with smtplib.SMTP(host, port) as server:
            if username and password:
                server.login(username, password)
            server.send_message(msg, to_addrs=recipients)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def send_email(
    subject: str,
    text_body: Optional[str] = None,
    html_body: Optional[str] = None,
    attachments: Optional[List[str]] = None,
    to: Optional[List[str]] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    sender: Optional[str] = None,
) -> None:
    """
    Send an email using settings from config.yaml (notifications.email).
    Parameters can override config values if provided.
    Expected config structure (Option B example):

    notifications:
      email:
        enabled: true
        sender: "you@example.com"
        recipients: ["you@example.com", "other@example.com"]
        subject_prefix: "Weinstein Report READY"
        provider: "smtp"
        smtp:
          host: "smtp.gmail.com"
          port_ssl: 587
          username: "you@example.com"
          app_password: "abcd app password here"
    """
    ecfg = _email_cfg()

    enabled = bool(ecfg.get("enabled", True))
    if not enabled:
        # Quietly no-op if disabled
        return

    # Sender & recipients
    sender_final = sender or ecfg.get("sender") or ecfg.get("from") or ecfg.get("smtp_user")
    if not sender_final:
        raise ValueError("Email FROM is not configured. Set notifications.email.sender or notifications.email.smtp.username in config.yaml")

    to_final: List[str] = to if to is not None else list(ecfg.get("recipients") or [])
    # Optional CC/BCC from config (merged with parameters if both)
    cc_final: List[str] = list(ecfg.get("cc") or [])
    bcc_final: List[str] = list(ecfg.get("bcc") or [])
    if cc:
        cc_final.extend(cc)
    if bcc:
        bcc_final.extend(bcc)

    if not to_final and not cc_final and not bcc_final:
        raise ValueError("No recipients configured. Provide notifications.email.recipients in config.yaml or pass 'to='.")

    # Subject prefix (optional)
    prefix = ecfg.get("subject_prefix", "")
    if prefix and not subject.startswith(prefix):
        subject = f"{prefix} {subject}"

    # Build message
    msg, all_rcpts = _make_multipart_message(
        sender=sender_final,
        to=to_final,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        attachments=attachments,
        cc=cc_final or None,
        bcc=bcc_final or None,
    )

    # Provider route (only SMTP supported here)
    provider = (ecfg.get("provider") or "smtp").lower()
    if provider != "smtp":
        raise ValueError(f"Unsupported email provider '{provider}'. Only 'smtp' is supported in this minimal sender.")

    smtp_cfg = ecfg.get("smtp") or {}
    host = smtp_cfg.get("host") or "smtp.gmail.com"
    port = int(smtp_cfg.get("port_ssl") or smtp_cfg.get("port") or 587)
    username = smtp_cfg.get("username") or sender_final
    password = smtp_cfg.get("app_password") or smtp_cfg.get("password")

    if not password:
        raise ValueError("SMTP password/app_password missing. Set notifications.email.smtp.app_password in config.yaml.")

    _send_via_smtp(
        msg=msg,
        recipients=all_rcpts,
        host=host,
        port=port,
        username=username,
        password=password,
        use_starttls=True,
    )
