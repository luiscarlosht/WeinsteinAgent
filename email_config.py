# email_config.py
import os
import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SMTPConfig:
    host: str
    port: int
    username: str
    password: Optional[str]   # from YAML smtp.app_password OR env EMAIL_SMTP_PASS
    use_ssl: bool             # True => SMTPS (465), False => STARTTLS (587)

@dataclass
class EmailSettings:
    enabled: bool
    sender: str
    recipients: List[str]
    subject_prefix: str
    provider: str
    smtp: SMTPConfig

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]

def load_email_settings(config_path: str = "./config.yaml") -> EmailSettings:
    cfg = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    e = (((cfg.get("notifications") or {}).get("email")) or {})
    smtp = e.get("smtp") or {}

    # Prefer YAML `smtp.app_password`; fallback to env var for convenience
    password = smtp.get("app_password") or os.environ.get("EMAIL_SMTP_PASS")

    # Port rules: 587 -> STARTTLS, 465 -> SSL
    port = int(smtp.get("port_ssl") or smtp.get("port") or 587)
    use_ssl = (port == 465)

    return EmailSettings(
        enabled=bool(e.get("enabled", False)),
        sender=str(e.get("sender") or ""),
        recipients=_as_list(e.get("recipients")),
        subject_prefix=str(e.get("subject_prefix") or ""),
        provider=str(e.get("provider") or "smtp").lower(),
        smtp=SMTPConfig(
            host=str(smtp.get("host") or ""),
            port=port,
            username=str(smtp.get("username") or ""),
            password=password,
            use_ssl=use_ssl,
        ),
    )
