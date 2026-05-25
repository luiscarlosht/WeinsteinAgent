# email_config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


_ENV_LOADED = False


def _strip_quotes(v: str) -> str:
    v = str(v or "").strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def load_local_env(env_path: str = ".env") -> None:
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
        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_quotes(v.split(" #", 1)[0].strip())
        if k and k not in os.environ:
            os.environ[k] = v


def _env(*names: str, default: str = "") -> str:
    load_local_env()
    for name in names:
        v = os.environ.get(name)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


@dataclass
class SMTPConfig:
    host: str
    port: int
    username: str
    password: Optional[str]
    use_ssl: bool


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
        return [str(i).strip() for i in x if str(i).strip()]
    return [i.strip() for i in str(x).replace(";", ",").split(",") if i.strip()]


def _resolve_secret_from_env_name(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    if s.startswith("${") and s.endswith("}"):
        return os.environ.get(s[2:-1], "").strip()
    if s.lower().startswith("env:"):
        return os.environ.get(s.split(":", 1)[1], "").strip()
    return s


def load_email_settings(config_path: str = "./config.yaml") -> EmailSettings:
    load_local_env()
    cfg = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    e = (((cfg.get("notifications") or {}).get("email")) or {})
    smtp = e.get("smtp") or {}

    password = _env("SMTP_APP_PASSWORD", "SMTP_PASSWORD", "EMAIL_SMTP_PASS", "EMAIL_SMTP_PASSWORD")
    if not password:
        password = _resolve_secret_from_env_name(smtp.get("app_password_env") or smtp.get("password_env") or "")
    if not password:
        password = _resolve_secret_from_env_name(smtp.get("app_password") or smtp.get("password") or "")

    port = int(_env("SMTP_PORT", "EMAIL_SMTP_PORT") or smtp.get("port_ssl") or smtp.get("port") or 587)
    use_ssl = port == 465

    recipients_env = _env("WEINSTEIN_EMAIL_TO", "EMAIL_TO", "SMTP_TO")

    return EmailSettings(
        enabled=bool(e.get("enabled", False)),
        sender=str(_env("WEINSTEIN_EMAIL_FROM", "EMAIL_FROM", "SMTP_FROM") or e.get("sender") or ""),
        recipients=_as_list(recipients_env) if recipients_env else _as_list(e.get("recipients")),
        subject_prefix=str(e.get("subject_prefix") or ""),
        provider=str(_env("EMAIL_PROVIDER") or e.get("provider") or "smtp").lower(),
        smtp=SMTPConfig(
            host=str(_env("SMTP_HOST", "EMAIL_SMTP_HOST") or smtp.get("host") or "smtp.gmail.com"),
            port=port,
            username=str(_env("SMTP_USER", "SMTP_USERNAME", "EMAIL_SMTP_USER") or smtp.get("username") or ""),
            password=password,
            use_ssl=use_ssl,
        ),
    )
