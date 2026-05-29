#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append durable BUY/NEAR/SELL/WATCH signal history rows after every intraday watcher run."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

DEFAULT_HISTORY_PATH = "output/intraday_signal_history.csv"

def _first(row: pd.Series, *names: str, default: Any = "") -> Any:
    for n in names:
        if n in row.index:
            v = row.get(n)
            if pd.notna(v): return v
    return default

def append_intraday_signal_history(diag: pd.DataFrame, *, timestamp: str, html_path: str = "", out_path: str = DEFAULT_HISTORY_PATH, market_regime: str = "", breadth_pct=None) -> int:
    if diag is None or diag.empty: return 0
    rows=[]
    for _, r in diag.iterrows():
        signal=str(_first(r,"Signal","signal",default="")).upper().strip()
        watch_signal=str(_first(r,"WatchSignal","watch_signal",default="")).upper().strip()
        is_action = signal in {"BUY","NEAR","NEAR_BUY","NEAR-TRIGGER","SELL","SELLTRIG","SELL-TRIGGER"}
        is_watch = watch_signal.startswith("WATCH")
        if not (is_action or is_watch): continue
        norm = "NEAR" if signal.startswith("NEAR") else ("SELL" if signal.startswith("SELL") else signal)
        rows.append({
            "timestamp": timestamp,
            "ticker": str(_first(r,"Ticker","ticker",default="")).upper().strip(),
            "signal": norm,
            "watch_signal": watch_signal,
            "reason": _first(r,"Reason","reason",default=""),
            "watch_reason": _first(r,"WatchReason","watch_reason",default=""),
            "price_now": _first(r,"PriceNow","price_now","close","Close",default=""),
            "pivot": _first(r,"Pivot","pivot",default=""),
            "headroom_pct": _first(r,"HeadroomPct","headroom_pct",default=""),
            "vol_pace": _first(r,"VolPace","vol_pace",default=""),
            "adx14": _first(r,"ADX14","adx14","adx",default=""),
            "stage": _first(r,"Structure","Stage","stage",default=""),
            "market_regime": market_regime,
            "breadth_pct": "" if breadth_pct is None else breadth_pct,
            "html_path": html_path,
        })
    if not rows: return 0
    out=pd.DataFrame(rows); path=Path(out_path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size>0:
        try:
            old=pd.read_csv(path); combo=pd.concat([old,out], ignore_index=True)
            keys=[c for c in ["timestamp","ticker","signal","watch_signal","price_now","pivot"] if c in combo.columns]
            combo.drop_duplicates(subset=keys, keep="last").to_csv(path,index=False)
        except Exception:
            out.to_csv(path, mode="a", header=False, index=False)
    else:
        out.to_csv(path,index=False)
    return len(out)
