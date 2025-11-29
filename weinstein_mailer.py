#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_mailer.py

Thin email helper used by:
- weinstein_report_weekly.py
- weinstein_intraday_watcher.py
- weinstein_short_watcher.py
- etc.

Goal: be flexible with config.yaml, so we support BOTH:

1) New flat style:

notifications:
  email:
    enabled: true
    sender: "you@gmail.com"
    from_name: "Weinstein Bot"
    from_email: "you@gmail.com"
    recipients: ["a@example.com","b@example.com"]
    subject_prefix: "Weinstein Report READY"
    provider: "smtp"
    host: "smtp.gmail.com"
    port: 587
    use_tls: true
    username: "you@gmail.com"
    password: "APP_PASSWORD_HERE"

2) Your existing nested style (what you pasted):

notifications:
  email:
    enabled: true
    sender: "luiscarlosht@gmail.com"
    recipients:
      - "luiscarloshernandez@hotmail.com"
      - "luiscarlosht@gmail.com"
    subject_prefix: "Weinstein Report READY"
    provider: "smtp"
    smtp:
      host: "smtp.gmail.com"
      port_ssl: 587
      username: "luiscarlosht@gmail.com"
      app_password: "qnamoxtmakhnvlml"

This file will look for BOTH `password` and `app_password`, and will
pull host/port/username from either the flat keys or the nested `smtp` dict.
"""

import ssl
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, Tuple, List, Optional

import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_email_config(cfg_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (email_cfg, smtp_cfg) where:

    email_cfg = cfg["notifications"]["email"] or cfg["email"]
    smtp_cfg  = email_cfg.get("smtp", {})
    """
    cfg = _load_yaml(cfg_path)

    notifications = cfg.get("notifications", {}) or {}
    email_cfg = notifications.get("email") or cfg.get("email") or {}

    if not email_cfg:
        raise KeyError("Email configuration not found under 'notifications.email' or top-level 'email'")

    smtp_cfg = email_cfg.get("smtp", {}) or {}
    return email_cfg, smtp_cfg


def _extract_smtp_settings(email_cfg: Dict[str, Any], smtp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge flat + nested smtp config into a single dict with keys:

    {
      "provider": "smtp" | "console" | ...
      "host": str,
      "port": int,
      "use_tls": bool,
      "use_ssl": bool,
      "username": str,
      "password": str
    }

    - prefers flat keys in email_cfg (host/port/username/password)
    - falls back to smtp_cfg["host"], smtp_cfg["port"], smtp_cfg["port_ssl"], smtp_cfg["username"], smtp_cfg["password"], smtp_cfg["app_password"]
    """
    provider = (email_cfg.get("provider") or "smtp").lower().strip()

    # Host
    host = (
        email_cfg.get("host")
        or smtp_cfg.get("host")
        or "smtp.gmail.com"
    )

    # Port
    port = (
        email_cfg.get("port")
        or smtp_cfg.get("port")
        or smtp_cfg.get("port_ssl")
        or 587
    )
    try:
        port = int(port)
    except Exception:
        port = 587

    # Username
    username = (
        email_cfg.get("username")
        or smtp_cfg.get("username")
        or email_cfg.get("sender")
        or email_cfg.get("from_email")
    )

    # Password: accept several names for compatibility
    password = (
        email_cfg.get("password")
        or email_cfg.get("app_password")
        or smtp_cfg.get("password")
        or smtp_cfg.get("app_password")
    )

    # TLS/SSL flags (reasonable defaults: STARTTLS on 587)
    use_ssl = bool(email_cfg.get("use_ssl", False))
    use_tls = bool(email_cfg.get("use_tls", True))
    if port == 465:
        use_ssl = True
        use_tls = False

    return {
        "provider": provider,
        "host": host,
        "port": port,
        "use_tls": use_tls,
        "use_ssl": use_ssl,
        "username": username,
        "password": password,
    }


def _build_message(
    subject: str,
    html_body: str,
    text_body: Optional[str],
    email_cfg: Dict[str, Any],
) -> EmailMessage:
    """
    Build a multipart/alternative message (text + HTML).
    """

    # Sender / from / recipients
    sender_name = email_cfg.get("from_name") or email_cfg.get("sender", "")
    from_addr = (
        email_cfg.get("from_email")
        or email_cfg.get("sender")
        or email_cfg.get("username")
    )
    raw_recipients = (
        email_cfg.get("recipients")
        or email_cfg.get("to")
        or []
    )

    if isinstance(raw_recipients, str):
        recipients: List[str] = [raw_recipients]
    else:
        recipients = list(raw_recipients or [])

    if not from_addr:
        raise ValueError("Missing sender email (from_email/sender) in email configuration.")
    if not recipients:
        raise ValueError("Missing recipients list in email configuration (recipients/to).")

    subj_prefix = str(email_cfg.get("subject_prefix") or "").strip()
    final_subject = f"{subj_prefix} {subject}".strip() if subj_prefix else subject

    msg = EmailMessage()
    if sender_name:
        msg["From"] = f"{sender_name} <{from_addr}>"
    else:
        msg["From"] = from_addr

    msg["To"] = ", ".join(recipients)
    msg["Subject"] = final_subject

    # Text + HTML parts
    if text_body:
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")
    else:
        # Derive a plain text body from HTML in a simple way
        stripped = (
            html_body.replace("<br>", "\n")
            .replace("<br/>", "\n")
            .replace("</p>", "\n\n")
        )
        msg.set_content(stripped)
        msg.add_alternative(html_body, subtype="html")

    return msg


def send_email(
    subject: str,
    html_body: str,
    text_body: Optional[str] = None,
    cfg_path: str = "config.yaml",
) -> None:
    """
    Main entry point used by the rest of the app.

    - Reads email config from cfg_path
    - Handles both flat + nested smtp configs
    - Sends via SMTP (default) or prints to console if provider="console"
    """

    # 1) Load base config
    email_cfg, smtp_cfg = _resolve_email_config(cfg_path)

    # If disabled, just exit quietly
    if not email_cfg.get("enabled", True):
        print("Email notifications are disabled in config.")
        return

    # 2) Build message
    msg = _build_message(subject, html_body, text_body, email_cfg)

    # 3) Resolve provider + SMTP settings
    smtp_settings = _extract_smtp_settings(email_cfg, smtp_cfg)
    provider = smtp_settings["provider"]

    if provider == "console":
        # Debug mode: just print the subject + "to" and don't send anything
        print("✉️ [console] Would send email:")
        print(f"  Subject: {msg['Subject']}")
        print(f"  To:      {msg['To']}")
        return

    if provider != "smtp":
        raise ValueError(f"Unsupported email provider: {provider}. Only 'smtp' and 'console' are supported.")

    host = smtp_settings["host"]
    port = smtp_settings["port"]
    username = smtp_settings["username"]
    password = smtp_settings["password"]
    use_tls = smtp_settings["use_tls"]
    use_ssl = smtp_settings["use_ssl"]

    if not username:
        raise ValueError("SMTP username not found in email configuration (username/sender/from_email/smtp.username).")
    if not password:
        raise ValueError("SMTP password not found in email configuration (password/app_password/smtp.app_password).")

    # 4) Actually send
    context = ssl.create_default_context()

    if use_ssl:
        # Implicit SSL (port 465 typically)
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(username, password)
            server.send_message(msg)
        print(f"✅ Email sent via SMTP_SSL: {host}:{port}")
    else:
        # STARTTLS (port 587 typically)
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            if use_tls:
                server.starttls(context=context)
                server.ehlo()
            server.login(username, password)
            server.send_message(msg)
        print(f"✅ Email sent via STARTTLS: {host}:{port}")
