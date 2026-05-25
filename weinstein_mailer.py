#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_mailer.py

Central email helper for all Weinstein tools.

Security model:
- Non-secret defaults can remain in config.yaml.
- Secrets should live in environment variables or local .env.
- .env is loaded automatically if present, but existing environment variables win.

Recommended .env keys:
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=your@gmail.com
  SMTP_APP_PASSWORD=your-new-gmail-app-password
  EMAIL_FROM=your@gmail.com
  EMAIL_TO=recipient1@example.com,recipient2@example.com

Backward compatibility:
- Still supports config.yaml notifications.email.smtp.app_password.
- Prefer env vars over YAML secrets.
"""

from __future__ import annotations

import os
import ssl
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import yaml


_ENV_LOADED = False


def _strip_quotes(v: str) -> str:
    v = str(v or "").strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def load_local_env(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE entries from .env without overwriting real env vars."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    p = Path(env_path)
    if not p.exists():
        return

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = _strip_quotes(val.split(" #", 1)[0].strip())
        if key and key not in os.environ:
            os.environ[key] = val


def _env(*names: str, default: str = "") -> str:
    for name in names:
        v = os.environ.get(name)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def _load_email_cfg(cfg_path: str) -> dict:
    load_local_env()

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    email_cfg = (cfg.get("notifications") or {}).get("email") or cfg.get("email")
    if not email_cfg:
        raise KeyError("email configuration not found. Expected under notifications.email or top-level email.")
    return email_cfg


def _split_addresses(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def _normalize_addresses(email_cfg: dict):
    sender = _env("WEINSTEIN_EMAIL_FROM", "EMAIL_FROM", "SMTP_FROM") or (
        email_cfg.get("sender") or email_cfg.get("from_email") or email_cfg.get("from_addr")
    )
    if not sender:
        raise ValueError("Email sender not specified. Set EMAIL_FROM or notifications.email.sender.")

    recipients_env = _env("WEINSTEIN_EMAIL_TO", "EMAIL_TO", "SMTP_TO")
    recipients = _split_addresses(recipients_env) if recipients_env else _split_addresses(
        email_cfg.get("recipients") or email_cfg.get("to") or email_cfg.get("emails") or []
    )
    if not recipients:
        raise ValueError("No recipient emails specified. Set EMAIL_TO or notifications.email.recipients.")

    from_name = _env("WEINSTEIN_EMAIL_FROM_NAME", "EMAIL_FROM_NAME") or email_cfg.get("from_name") or email_cfg.get("display_name") or sender
    return sender, from_name, recipients


def _resolve_secret_from_env_name(value: str) -> str:
    """Allow YAML values like ${SMTP_APP_PASSWORD} or env:SMTP_APP_PASSWORD."""
    s = str(value or "").strip()
    if not s:
        return ""
    if s.startswith("${") and s.endswith("}"):
        return os.environ.get(s[2:-1], "").strip()
    if s.lower().startswith("env:"):
        return os.environ.get(s.split(":", 1)[1], "").strip()
    return s


def _smtp_params(email_cfg: dict):
    provider = (_env("EMAIL_PROVIDER") or email_cfg.get("provider") or "smtp").lower()
    if provider != "smtp":
        raise ValueError(f"Unsupported email provider {provider!r}. Only smtp is implemented.")

    smtp_cfg = email_cfg.get("smtp") or {}

    host = _env("SMTP_HOST", "EMAIL_SMTP_HOST") or smtp_cfg.get("host") or "smtp.gmail.com"
    port = _env("SMTP_PORT", "EMAIL_SMTP_PORT") or smtp_cfg.get("port_ssl") or smtp_cfg.get("port_tls") or smtp_cfg.get("port") or 587
    username = _env("SMTP_USER", "SMTP_USERNAME", "EMAIL_SMTP_USER", "EMAIL_USERNAME") or smtp_cfg.get("username") or email_cfg.get("smtp_username") or email_cfg.get("user")
    if not username:
        raise ValueError("SMTP username not found. Set SMTP_USER or notifications.email.smtp.username.")

    password = _env(
        "SMTP_APP_PASSWORD",
        "SMTP_PASSWORD",
        "EMAIL_SMTP_PASS",
        "EMAIL_SMTP_PASSWORD",
        "WEINSTEIN_SMTP_PASSWORD",
    )

    if not password:
        password = _resolve_secret_from_env_name(
            smtp_cfg.get("app_password_env") or smtp_cfg.get("password_env") or ""
        )

    # Last-resort backward compatibility only. Avoid committing real values here.
    if not password:
        yaml_password = smtp_cfg.get("app_password") or smtp_cfg.get("password") or email_cfg.get("app_password") or email_cfg.get("password")
        password = _resolve_secret_from_env_name(yaml_password)
        if password:
            print("WARNING: SMTP password came from config.yaml. Move it to .env / environment variables.")

    if not password:
        raise ValueError("SMTP password not found. Set SMTP_APP_PASSWORD in .env or environment.")

    return host, int(port), username, password


def _build_subject(raw_subject: Optional[str], email_cfg: dict, subject_tag: Optional[str] = None) -> str:
    prefix = (email_cfg.get("subject_prefix") or "").strip()
    tag_part = ""
    if subject_tag:
        clean_tag = subject_tag.strip()
        if not (clean_tag.startswith("[") and clean_tag.endswith("]")):
            clean_tag = f"[{clean_tag}]"
        tag_part = clean_tag
    base_subj = (raw_subject or "").strip() or "Weinstein Report"
    return " ".join([p for p in [prefix, tag_part, base_subj] if p])


def send_email(subject: Optional[str] = None, html_body: str = "", text_body: Optional[str] = None, cfg_path: str = "config.yaml", **kwargs):
    subject_tag = kwargs.pop("subject_tag", None)
    regime_header = kwargs.pop("regime_header", None)

    email_cfg = _load_email_cfg(cfg_path)
    if not bool(email_cfg.get("enabled", True)):
        print("Email notifications are disabled.")
        return

    sender, from_name, recipients = _normalize_addresses(email_cfg)
    host, port, username, password = _smtp_params(email_cfg)
    final_subject = _build_subject(subject, email_cfg, subject_tag=subject_tag)

    if regime_header and text_body:
        text_body = f"{regime_header}\n\n{text_body}"

    msg = EmailMessage()
    msg["Subject"] = final_subject
    msg["From"] = f"{from_name} <{sender}>"
    msg["To"] = ", ".join(recipients)

    if text_body is None:
        text_body = "Your Weinstein report is included as HTML."

    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    print(f"Connecting to SMTP server {host}:{port} as {username} (subject: {final_subject!r})...")
    with smtplib.SMTP(host, port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(username, password)
        server.send_message(msg)

    print("Email sent.")
