#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail if obvious secrets are still committed in config.yaml or tracked files."""
from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = [Path("config.yaml")]
BAD_PATTERNS = [
    ("gmail_app_password", re.compile(r"app_password:\s*['\"]?[a-z0-9]{12,}['\"]?", re.I)),
    ("generic_password", re.compile(r"password:\s*['\"]?(?!\$\{|env:)[^'\"\s]{10,}['\"]?", re.I)),
    ("api_key_literal", re.compile(r"(api_key|apikey|token|secret):\s*['\"]?(?!\$\{|env:)[A-Za-z0-9_\-]{20,}", re.I)),
]

ok = True
for p in FILES:
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    for name, pat in BAD_PATTERNS:
        for m in pat.finditer(text):
            ok = False
            line = text.count("\n", 0, m.start()) + 1
            print(f"POSSIBLE_SECRET {p}:{line} pattern={name}")

if not ok:
    print("\nSecrets may still be present. Move them to .env/environment variables before committing.")
    sys.exit(1)

print("OK: no obvious committed secrets found in checked files.")
