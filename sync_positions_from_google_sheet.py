#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_positions_from_google_sheet.py

Pull current positions from a Google Sheet tab and write current_positions.csv.

Environment variables:
- POSITIONS_GOOGLE_SHEET_ID
- POSITIONS_GOOGLE_SHEET_TAB  (default: Open_Positions)
- POSITIONS_CSV_OUT           (default: current_positions.csv)

Optional:
- POSITIONS_GOOGLE_CSV_URL    (full CSV export URL overrides sheet id/tab)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def build_csv_url(sheet_id: str, tab: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(tab)}"


def download_csv(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as r:
        data = r.read()
    if not data.strip():
        raise RuntimeError("Downloaded CSV was empty.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", default=os.getenv("POSITIONS_GOOGLE_SHEET_ID", ""))
    ap.add_argument("--tab", default=os.getenv("POSITIONS_GOOGLE_SHEET_TAB", "Open_Positions"))
    ap.add_argument("--csv-url", default=os.getenv("POSITIONS_GOOGLE_CSV_URL", ""))
    ap.add_argument("--out", default=os.getenv("POSITIONS_CSV_OUT", "current_positions.csv"))
    args = ap.parse_args()

    url = args.csv_url.strip()
    if not url:
        if not args.sheet_id.strip():
            raise SystemExit("Missing POSITIONS_GOOGLE_SHEET_ID or POSITIONS_GOOGLE_CSV_URL.")
        url = build_csv_url(args.sheet_id.strip(), args.tab.strip())

    data = download_csv(url)

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)

    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(f"Source tab/url: {args.tab if not args.csv_url else 'custom csv url'}")


if __name__ == "__main__":
    main()
