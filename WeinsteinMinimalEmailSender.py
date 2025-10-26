#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple SMTP mailer with attachment support.

Reads settings from config.yaml:

email:
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  smtp_user: "your_user@gmail.com"
  smtp_pass: "app_password_or_smtp_password"
  from:      "your_user@gmail.com"
  to:
    - "you@example.com"
  cc: []
  bcc: []
  use_tls: true
"""

from __future__ import annotations
import os
import smtplib
import ssl
import mimetypes
from typing import List, Optional, Sequence
from email.message import EmailMessage
import yaml


def _coerce_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    return [s] if s else []


def load_email_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    email_cfg = (cfg.get("email") or {})
    # sane defaults
    email_cfg.setdefault("smtp_host", "smtp.gmail.com")
    email_cfg.setdefault("smtp_port", 587)
    email_cfg.setdefault("use_tls", True)
    email_cfg["to"] = _coerce_list(email_cfg.get("to"))
    email_cfg["cc"] = _coerce_list(email_cfg.get("cc"))
    email_cfg["bcc"] = _coerce_list(email_cfg.get("bcc"))
    return email_cfg


def _attach_file(msg: EmailMessage, path: str) -> None:
    if not os.path.isfile(path):
        return
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    with open(path, "rb") as fp:
        data = fp.read()
    filename = os.path.basename(path)
    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)


def send_email(
    subject: str,
    body_text: str,
    *,
    body_html: Optional[str] = None,
    to: Optional[Sequence[str]] = None,
    cc: Optional[Sequence[str]] = None,
    bcc: Optional[Sequence[str]] = None,
    attachments: Optional[Sequence[str]] = None,
    config_path: str = "config.yaml",
) -> None:
    cfg = load_email_config(config_path)

    from_addr = cfg.get("from") or cfg.get("smtp_user")
    if not from_addr:
        raise ValueError("Email FROM is not configured. Set email.from or email.smtp_user in config.yaml")

    to_all = _coerce_list(to) or cfg.get("to") or []
    cc_all = _coerce_list(cc) or cfg.get("cc") or []
    bcc_all = _coerce_list(bcc) or cfg.get("bcc") or []
    if not to_all and not cc_all and not bcc_all:
        raise ValueError("No recipients. Configure email.to in config.yaml or pass to=[] in send_email()")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    if to_all: msg["To"] = ", ".join(to_all)
    if cc_all: msg["Cc"] = ", ".join(cc_all)

    if body_html:
        # multipart/alternative
        msg.set_content(body_text or "")
        msg.add_alternative(body_html, subtype="html")
    else:
        msg.set_content(body_text or "")

    for path in (attachments or []):
        _attach_file(msg, path)

    smtp_host = cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = int(cfg.get("smtp_port", 587))
    smtp_user = cfg.get("smtp_user")
    smtp_pass = cfg.get("smtp_pass")
    use_tls   = bool(cfg.get("use_tls", True))

    recipients = to_all + cc_all + bcc_all
    if use_tls:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg, from_addr=from_addr, to_addrs=recipients)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg, from_addr=from_addr, to_addrs=recipients)
