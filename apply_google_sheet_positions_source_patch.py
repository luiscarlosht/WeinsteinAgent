#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_google_sheet_positions_source_patch.py

Patches weinstein_account_profiles.py so the central position loader can read
from Google Sheets when POSITIONS_CSV=GOOGLE_SHEET or POSITIONS_SOURCE=GOOGLE_SHEET.

This is intentionally conservative:
- Creates a .bak file first.
- Adds import for load_positions_dataframe.
- Replaces the first pd.read_csv(...) call in weinstein_account_profiles.py.

If the script cannot safely patch, it exits with guidance.
"""

from __future__ import annotations

import re
from pathlib import Path


TARGET = Path("weinstein_account_profiles.py")
IMPORT_LINE = "from weinstein_positions_source import load_positions_dataframe\n"


def main():
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    original = text

    backup = TARGET.with_suffix(TARGET.suffix + ".bak_google_positions")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    if "load_positions_dataframe" not in text:
        # Add after imports, preferably after pandas import.
        lines = text.splitlines(True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, IMPORT_LINE)
        text = "".join(lines)

    # Replace first read_csv call. The central profile loader should only need one.
    # Handles df = pd.read_csv(path), pd.read_csv(csv_path), pd.read_csv(positions_csv), etc.
    pattern = re.compile(r"pd\.read_csv\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
    matches = list(pattern.finditer(text))

    if not matches:
        raise SystemExit(
            "Could not find a simple pd.read_csv(variable) call to patch. "
            "Open weinstein_account_profiles.py and replace the positions CSV read with "
            "load_positions_dataframe(<same variable>)."
        )

    m = matches[0]
    var_name = m.group(1)
    text = text[:m.start()] + f"load_positions_dataframe({var_name})" + text[m.end():]

    if text == original:
        print("No changes needed.")
        return

    TARGET.write_text(text, encoding="utf-8")
    print(f"Patched {TARGET}")
    print(f"Backup: {backup}")
    print(f"Replaced first pd.read_csv({var_name}) with load_positions_dataframe({var_name})")


if __name__ == "__main__":
    main()
