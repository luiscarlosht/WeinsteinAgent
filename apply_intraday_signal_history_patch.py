#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch weinstein_intraday_watcher.py to append durable intraday_signal_history.csv rows."""
from pathlib import Path
TARGET=Path("weinstein_intraday_watcher.py")
IMPORT_MARKER="from weinstein_intraday_signal_history import append_intraday_signal_history"
CALL_MARKER="append_intraday_signal_history("
IMPORT_BLOCK='''try:\n    from weinstein_intraday_signal_history import append_intraday_signal_history\nexcept Exception:\n    append_intraday_signal_history = None\n'''
CALL_BLOCK='''\n    # Durable intraday signal history: append BUY/NEAR/SELL/WATCH rows before the next run overwrites intraday_debug.csv.\n    try:\n        if append_intraday_signal_history is not None:\n            _regime_label = ""\n            try:\n                _regime_label = getattr(regime_decision, "regime_label", "") if regime_decision is not None else ""\n            except Exception:\n                _regime_label = ""\n            _history_rows = append_intraday_signal_history(\n                diag,\n                timestamp=ts_display,\n                html_path=html_path,\n                out_path=os.path.join(cfg.app.output_dir, "intraday_signal_history.csv"),\n                market_regime=_regime_label,\n                breadth_pct=breadth_pct,\n            )\n            if _history_rows:\n                log(f"Appended intraday signal history rows → {_history_rows}")\n    except Exception as e:\n        log(f"Intraday signal history append skipped: {e}")\n'''

def main():
    if not TARGET.exists(): raise SystemExit("Run this from repo root; missing weinstein_intraday_watcher.py")
    s=TARGET.read_text(encoding="utf-8"); changed=False
    if IMPORT_MARKER not in s:
        needle="import pandas as pd\n"
        if needle in s: s=s.replace(needle, needle+IMPORT_BLOCK, 1)
        else: s=IMPORT_BLOCK+s
        changed=True
    if CALL_MARKER not in s:
        needle='        log(f"Saved HTML → {html_path}")\n'
        if needle not in s: raise SystemExit("Could not find HTML save log insertion point; patch not applied.")
        s=s.replace(needle, needle+CALL_BLOCK+"\n", 1); changed=True
    if not changed:
        print("No changes needed; patch already applied."); return
    backup=TARGET.with_suffix(TARGET.suffix+".bak_signal_history")
    backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET.write_text(s, encoding="utf-8")
    print(f"Patched {TARGET}"); print(f"Backup: {backup}")
if __name__=="__main__": main()
