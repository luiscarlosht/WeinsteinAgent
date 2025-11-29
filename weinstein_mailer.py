#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_mailer.py

Lightweight email helper for all Weinstein tools.

Features
--------
- Reads SMTP + recipients from config.yaml
  * supports both:
      notifications.email: {...}
      email: {...}
- Optional on/off toggle: notifications.email.enabled: false
- Optional subject prefix: "[Weinstein]" etc.
- Optional:
    * subject_tag   -> prepended in [TAG] form
    * regime_header -> injected at the top of HTML/text body

Typical usage
-------------
from weinstein_mailer import send_email

send_email(
    subject="Weinstein Intraday – 2025-11-29",
    html_body=html_body,
    text_body=text_body,
    subject_tag="INTRADAY BULL L=True S=False",
    regime_header="Market regime (Ch8): BULL | long_ok=True short_ok=False",
)
"""

import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

import yaml


def _load_email_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Prefer nested notifications.email, fall back to top-level email
    email_cfg = (cfg.get("notifications", {}) or {}).get("email")
    if not email_cfg:
        email_cfg = cfg.get("email", {})

    if not isinstance(email_cfg, dict):
        raise KeyError("Email configuration must be a mapping under 'notifications.email' or 'email'.")

    return email_cfg


def _bool(v, default=False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def send_email(
    subject: str,
    html_body: str,
    text_body: Optional[str] = None,
    *,
    cfg_path: str = "config.yaml",
    subject_tag: Optional[str] = None,
    regime_header: Optional[str] = None,
) -> None:
    """
    Send an email with HTML + optional plain-text.

    Parameters
    ----------
    subject : str
        Base subject (e.g. "Weinstein Intraday – 2025-11-29")
    html_body : str
        Main HTML content (report, watch list, etc.)
    text_body : str, optional
        Plain-text alternative. If omitted, one is synthesized.
    cfg_path : str
        Path to config.yaml (holds SMTP settings).
    subject_tag : str, optional
        Extra tag to prepend like "[INTRADAY BULL L=True S=False]".
    regime_header : str, optional
        Line like "Market regime (Ch8): BULL | long_ok=True short_ok=False"
        that is injected at the very top of both HTML and text bodies.
    """

    email_cfg = _load_email_cfg(cfg_path)

    if not _bool(email_cfg.get("enabled", True), default=True):
        print("Email notifications are disabled in config (notifications.email.enabled = false).")
        return

    # Basic fields
    sender_name = email_cfg.get("from_name") or email_cfg.get("sender_name") or ""
    from_addr = email_cfg.get("from_email") or email_cfg.get("sender")
    recipients = email_cfg.get("recipients") or email_cfg.get("to") or []

    if not from_addr:
        raise ValueError("Missing 'from_email' (or 'sender') in email configuration.")
    if not recipients:
        raise ValueError("No recipients configured under 'recipients' or 'to'.")

    # Subject prefix / tag
    subj_prefix = (email_cfg.get("subject_prefix") or "").strip()
    base_subject = subject.strip()

    final_subject = base_subject
    if subj_prefix:
        # e.g. "[Weinstein]" or "[Weinstein Intraday]"
        final_subject = f"{subj_prefix.strip()} {final_subject}"

    if subject_tag:
        # INTRADAY BULL L=True S=False  -> "[INTRADAY BULL L=True S=False] ..."
        tag = str(subject_tag).strip()
        if tag:
            final_subject = f"[{tag}] {final_subject}"

    # Regime header injected into bodies
    html_out = html_body or ""
    text_out = text_body or ""

    if regime_header:
        regime_line = str(regime_header).strip()
        if regime_line:
            # Prepend a bold paragraph to HTML
            html_out = f"<p><b>{regime_line}</b></p>\n" + html_out

            # Prepend to text
            if text_out:
                text_out = f"{regime_line}\n\n{text_out}"
            else:
                text_out = regime_line

    # If still no text body, make a rough plain-text fallback from HTML
    if not text_out:
        # Very naive strip; good enough as a fallback
        import re

        text_out = re.sub(r"<br\s*/?>", "\n", html_out, flags=re.IGNORECASE)
        text_out = re.sub(r"<[^>]+>", "", text_out)
        text_out = text_out.strip()

    # SMTP params
    smtp_host = email_cfg.get("smtp_host") or email_cfg.get("host") or "smtp.gmail.com"
    smtp_port = int(email_cfg.get("smtp_port") or email_cfg.get("port") or 587)
    use_tls = _bool(email_cfg.get("use_tls", True), default=True)
    username = email_cfg.get("username") or from_addr
    password = email_cfg.get("password")

    if not password:
        raise ValueError("SMTP password not found in email configuration (password).")

    # Build message
    msg = EmailMessage()
    msg["Subject"] = final_subject
    msg["From"] = f"{sender_name} <{from_addr}>" if sender_name else from_addr
    msg["To"] = ", ".join(recipients)

    msg.set_content(text_out or "(no text body)")
    msg.add_alternative(html_out or "<html><body><p>(empty body)</p></body></html>", subtype="html")

    # Send
    if use_tls and smtp_port == 587:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.send_message(msg)
    elif smtp_port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.login(username, password)
            server.send_message(msg)

    print(f"✅ Email sent via {smtp_host}:{smtp_port} with subject: {final_subject}")
