#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_mailer.py

Central email helper for all Weinstein tools (weekly, intraday, crypto, shorts).

Goals:
- Read SMTP settings from config.yaml (under notifications.email).
- Use 'app_password' field from config.yaml (Gmail App Password).
- Support a subject_prefix in config.yaml.
- Support optional subject_tag kwarg (e.g. "INTRADAY" → "[INTRADAY]").
- Support optional regime_header kwarg (can be ignored or used by caller).
- Be tolerant of extra keyword arguments so new callers don't break.
"""

import os
import ssl
import smtplib
from email.message import EmailMessage
from typing import Optional

import yaml


# ---------------- Config helpers ----------------

def _load_email_cfg(cfg_path: str) -> dict:
    """
    Load config.yaml and return the email configuration block.

    Expected structure:

    notifications:
      email:
        enabled: true
        sender: "you@gmail.com"
        recipients:
          - "you@gmail.com"
        subject_prefix: "Weinstein Report READY"
        provider: "smtp"
        smtp:
          host: "smtp.gmail.com"
          port_ssl: 587
          username: "you@gmail.com"
          app_password: "your-app-password"
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Prefer nested notifications.email; fall back to top-level 'email'
    email_cfg = (cfg.get("notifications") or {}).get("email") or cfg.get("email")
    if not email_cfg:
        raise KeyError(
            "email configuration not found. Expected under 'notifications.email' or top-level 'email'."
        )
    return email_cfg


def _normalize_addresses(email_cfg: dict):
    """
    Decide From and To addresses from config.
    """
    sender = (
        email_cfg.get("sender")
        or email_cfg.get("from_email")
        or email_cfg.get("from_addr")
    )
    if not sender:
        raise ValueError("Email sender not specified (sender/from_email).")

    recipients = (
        email_cfg.get("recipients")
        or email_cfg.get("to")
        or email_cfg.get("emails")
        or []
    )
    if isinstance(recipients, str):
        recipients = [r.strip() for r in recipients.split(",") if r.strip()]

    if not recipients:
        raise ValueError("No recipient emails specified in config.yaml (recipients/to).")

    from_name = email_cfg.get("from_name") or email_cfg.get("display_name") or sender
    return sender, from_name, recipients


def _smtp_params(email_cfg: dict):
    """
    Extract SMTP provider settings (host, port, username, password/app_password).
    """
    provider = (email_cfg.get("provider") or "smtp").lower()
    if provider != "smtp":
        raise ValueError(f"Unsupported email provider '{provider}'. Only 'smtp' is implemented.")

    smtp_cfg = email_cfg.get("smtp") or {}

    host = smtp_cfg.get("host") or "smtp.gmail.com"
    port = (
        smtp_cfg.get("port_ssl")
        or smtp_cfg.get("port_tls")
        or smtp_cfg.get("port")
        or 587
    )

    username = smtp_cfg.get("username") or email_cfg.get("smtp_username") or email_cfg.get("user")
    if not username:
        raise ValueError("SMTP username not found in email configuration (smtp.username).")

    # Your config.yaml uses 'app_password'
    password = (
        smtp_cfg.get("app_password")
        or smtp_cfg.get("password")
        or email_cfg.get("app_password")
        or email_cfg.get("password")
    )
    if not password:
        raise ValueError("SMTP password not found in email configuration (app_password/password).")

    return host, int(port), username, password


def _build_subject(
    raw_subject: Optional[str],
    email_cfg: dict,
    subject_tag: Optional[str] = None,
) -> str:
    """
    Combine:
      - config subject_prefix
      - subject_tag (e.g. 'INTRADAY' → '[INTRADAY]')
      - raw_subject (e.g. 'Intraday Watch — 2 BUY / 3 NEAR / 1 SELL')
    into a single string.
    """
    prefix = (email_cfg.get("subject_prefix") or "").strip()

    tag_part = ""
    if subject_tag:
        clean_tag = subject_tag.strip()
        # Avoid double brackets
        if not (clean_tag.startswith("[") and clean_tag.endswith("]")):
            clean_tag = f"[{clean_tag}]"
        tag_part = clean_tag

    base_subj = (raw_subject or "").strip() or "Weinstein Report"

    parts = [p for p in [prefix, tag_part, base_subj] if p]
    return " ".join(parts)


# ---------------- Main send_email ----------------

def send_email(
    subject: Optional[str] = None,
    html_body: str = "",
    text_body: Optional[str] = None,
    cfg_path: str = "config.yaml",
    **kwargs,
):
    """
    Main send_email entry point used by:
      - weinstein_report_weekly.py
      - weinstein_intraday_watcher.py
      - weinstein_short_watcher.py
      - weinstein_crypto_watcher.py
      - etc.

    Recognized kwargs:
      - subject_tag: optional string used to tag subject (e.g. 'INTRADAY', 'WEEKLY', 'SHORT')
                     The caller can pass either 'INTRADAY' or '[INTRADAY]'; we normalize.
      - regime_header: optional string the caller may use for regime summary in body.
                       We don't need it here; intraday watcher already includes regime
                       in html_body/text_body, but we accept it to avoid breaking.

    Any other kwargs are safely ignored (forward compatibility).
    """
    subject_tag = kwargs.pop("subject_tag", None)
    regime_header = kwargs.pop("regime_header", None)  # accepted but not required

    # Ignore any unexpected extra kwargs to keep API future-proof
    if kwargs:
        # If you ever want, you could log them here.
        pass

    email_cfg = _load_email_cfg(cfg_path)

    if not bool(email_cfg.get("enabled", True)):
        print("Email notifications are disabled in config.yaml (notifications.email.enabled = false).")
        return

    sender, from_name, recipients = _normalize_addresses(email_cfg)
    host, port, username, password = _smtp_params(email_cfg)

    final_subject = _build_subject(subject, email_cfg, subject_tag=subject_tag)

    # If regime_header is provided and text_body is empty, we could prepend it.
    # But in your intraday/weekly scripts, regime is already baked into text_body/html_body.
    if regime_header and text_body:
        # Make regime line clearly visible at the top of the text version
        text_body = f"{regime_header}\n\n{text_body}"

    # Build message
    msg = EmailMessage()
    msg["Subject"] = final_subject
    msg["From"] = f"{from_name} <{sender}>"
    msg["To"] = ", ".join(recipients)

    if text_body is None:
        text_body = (
            "Your Weinstein report is included as HTML. "
            "Open this email in an HTML-capable client to see the BUY / NEAR / SELL sections, "
            "Structure column, diagnostics table, and charts."
        )

    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    # Send via SMTP/STARTTLS
    context = ssl.create_default_context()
    print(
        f"Connecting to SMTP server {host}:{port} as {username} "
        f"(subject: {final_subject!r})..."
    )
    with smtplib.SMTP(host, port) as server:
        server.ehlo()
        try:
            server.starttls(context=context)
            server.ehlo()
        except smtplib.SMTPException:
            # Some servers might not support STARTTLS, but Gmail does.
            pass

        server.login(username, password)
        server.send_message(msg)

    print("Email sent.")


# ---------------- CLI test helper ----------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test WeinsteinMailer send_email")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--subject", default="Weinstein Mailer Test")
    ap.add_argument("--tag", default="", help="Optional subject tag, e.g. INTRADAY")
    ap.add_argument("--regime", default="", help="Optional regime header line")
    ap.add_argument("--text", default="This is a test email from weinstein_mailer.py")
    args = ap.parse_args()

    send_email(
        subject=args.subject,
        text_body=args.text,
        html_body=f"<h3>{args.subject}</h3><p>{args.text}</p>",
        cfg_path=args.config,
        subject_tag=args.tag or None,
        regime_header=args.regime or None,
    )
