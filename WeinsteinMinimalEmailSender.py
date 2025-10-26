#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WeinsteinMinimalEmailSender.py

Minimal email helper that reads config.yaml and sends an email with optional
HTML body and attachments. Supports two config layouts:

A) Legacy:
email:
  from: "you@example.com"
  to: ["a@x.com", "b@y.com"]
  subject_prefix: "Weinstein Report"
  use_tls: true
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  smtp_user: "you@example.com"
  smtp_pass: "app_password"

B) Notifications style (your current file):
notifications:
  email:
    enabled: true
    sender: "you@example.com"
    recipients:
      - "a@x.com"
      - "b@y.com"
    subject_prefix: "Weinstein Report READY"
    provider: "smtp"
    smtp:
      host: "smtp.gmail.com"
      port_ssl: 587
      username: "you@example.com"
      app_password: "app_password"

Usage example (from another script):
    from WeinsteinMinimalEmailSender import send_email
    send_email(
        subject="Weekly report",
        text_body="See attached.",
        html_body="<h1>Weekly</h1><p>See attached.</p>",
        attachments=["/path/to/file.html", "/path/to/file.csv"]
    )
"""

import os
import smtplib
import mimetypes
from typing import Iterable, List, Optional, Tuple

from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_email_cfg(cfg: dict) -> dict:
    """
    Normalize email configuration from either:
      - cfg['email'] {...}
      - or cfg['notifications']['email'] {...} with nested smtp settings.
    Returns a flat dict with keys:
        from, to, smtp_host, smtp_port, smtp_user, smtp_pass, use_tls, subject_prefix, enabled
    """
    e = (cfg or {}).get("email") or {}
    n = ((cfg or {}).get("notifications") or {}).get("email") or {}
    smtp_n = n.get("smtp") or {}

    # Prefer top-level `email` if present
    if e:
        # Allow either lists or comma-separated strings for recipients
        to_val = e.get("to") or e.get("recipients") or []
        if isinstance(to_val, str):
            to_val = [t.strip() for t in to_val.split(",") if t.strip()]

        return {
            "enabled": True if e != {} else bool(n.get("enabled", True)),
            "from": e.get("from") or e.get("smtp_user") or e.get("sender"),
            "to": to_val,
            "smtp_host": e.get("smtp_host") or (e.get("smtp") or {}).get("host") or "smtp.gmail.com",
            "smtp_port": int(
                e.get("smtp_port")
                or (e.get("smtp") or {}).get("port")
                or (e.get("smtp") or {}).get("port_ssl")
                or 587
            ),
            "smtp_user": e.get("smtp_user") or (e.get("smtp") or {}).get("username"),
            "smtp_pass": e.get("smtp_pass") or (e.get("smtp") or {}).get("app_password"),
            "use_tls": bool(e.get("use_tls", True)),
            "subject_prefix": e.get("subject_prefix", ""),
        }

    # Fallback → notifications.email + notifications.email.smtp
    to_val = n.get("recipients") or []
    if isinstance(to_val, str):
        to_val = [t.strip() for t in to_val.split(",") if t.strip()]

    return {
        "enabled": bool(n.get("enabled", True)),
        "from": n.get("sender") or smtp_n.get("username"),
        "to": to_val,
        "smtp_host": smtp_n.get("host", "smtp.gmail.com"),
        "smtp_port": int(smtp_n.get("port_ssl", 587)),
        "smtp_user": smtp_n.get("username"),
        "smtp_pass": smtp_n.get("app_password"),
        "use_tls": True,
        "subject_prefix": n.get("subject_prefix", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core mail send
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _make_multipart_message(
    subject: str,
    from_addr: str,
    to_addrs: Iterable[str],
    text_body: Optional[str] = None,
    html_body: Optional[str] = None,
    attachments: Optional[Iterable[str]] = None,
    cc_addrs: Optional[Iterable[str]] = None,
    bcc_addrs: Optional[Iterable[str]] = None,
) -> Tuple[MIMEMultipart, List[str]]:
    """
    Build a MIME email with alternative (text/html) and attachments.
    Returns (message, all_recipients).
    """
    to_addrs = _ensure_list(to_addrs)
    cc_addrs = _ensure_list(cc_addrs)
    bcc_addrs = _ensure_list(bcc_addrs)
    all_rcpts = list(dict.fromkeys(to_addrs + cc_addrs + bcc_addrs))  # dedupe

    msg_root = MIMEMultipart("mixed")
    msg_root["From"] = from_addr
    msg_root["To"] = ", ".join(to_addrs) if to_addrs else ""
    if cc_addrs:
        msg_root["Cc"] = ", ".join(cc_addrs)
    msg_root["Subject"] = subject
    msg_root["Date"] = formatdate(localtime=True)
    msg_root["Message-ID"] = make_msgid()

    # Create alternative part for text/html bodies
    alt = MIMEMultipart("alternative")
    if text_body:
        alt.attach(MIMEText(text_body, "plain", "utf-8"))
    if html_body:
        alt.attach(MIMEText(html_body, "html", "utf-8"))

    msg_root.attach(alt)

    # Attach files
    for path in (attachments or []):
        if not path:
            continue
        if not os.path.exists(path):
            # Attach a small note as text if missing file; don't fail the whole email.
            note = MIMEText(f"(Attachment missing: {path})\n", "plain", "utf-8")
            note.add_header("Content-Disposition", "attachment", filename=os.path.basename(path) or "missing.txt")
            msg_root.attach(note)
            continue

        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        with open(path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype=subtype)
        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
        msg_root.attach(part)

    return msg_root, all_rcpts


def send_email(
    subject: str,
    text_body: Optional[str] = None,
    html_body: Optional[str] = None,
    attachments: Optional[Iterable[str]] = None,
    cc: Optional[Iterable[str]] = None,
    bcc: Optional[Iterable[str]] = None,
    cfg_path: str = "config.yaml",
) -> bool:
    """
    Send an email based on config.yaml (supports both layouts).
    Returns True on success, False on failure (also raises on certain config errors).
    """
    cfg = load_config(cfg_path)
    em = _resolve_email_cfg(cfg)

    if not em.get("enabled", True):
        print("✉️  Email is disabled by config (notifications.email.enabled = false). Skipping send.")
        return True

    from_addr = em.get("from")
    to_addrs = em.get("to") or []
    host = em.get("smtp_host")
    port = int(em.get("smtp_port") or 587)
    user = em.get("smtp_user")
    pwd = em.get("smtp_pass")
    use_tls = bool(em.get("use_tls", True))
    prefix = em.get("subject_prefix", "")

    if not from_addr or not user:
        raise ValueError("Email FROM is not configured. "
                         "Set either email.from/smtp_user or notifications.email.sender/smtp.username in config.yaml")

    final_subject = f"{prefix} {subject}".strip()

    msg, all_rcpts = _make_multipart_message(
        subject=final_subject,
        from_addr=from_addr,
        to_addrs=to_addrs,
        text_body=text_body,
        html_body=html_body,
        attachments=attachments,
        cc=cc,
        bcc=bcc,
    )

    if not all_rcpts:
        raise ValueError("Email recipient list is empty. "
                         "Add recipients under email.to / email.recipients or notifications.email.recipients.")

    try:
        with smtplib.SMTP(host, port, timeout=30) as s:
            if use_tls:
                s.starttls()
            s.login(user, pwd)
            s.sendmail(from_addr, all_rcpts, msg.as_string())
        print(f"📧 Email sent to {', '.join(all_rcpts)}")
        return True
    except Exception as e:
        print(f"❌ Email send failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper (optional)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simple manual test:
    ok = send_email(
        subject="Weinstein Agent - Test",
        text_body="Plain text body.\nIf you see this, SMTP works.",
        html_body="<h2>Weinstein Agent - Test</h2><p>If you see this, SMTP works.</p>",
        attachments=[],
    )
    raise SystemExit(0 if ok else 2)
