#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct historical PROD signal behavior from archived intraday HTML reports.
Parses real ticker BUY/NEAR/WATCH rows from output/intraday_watch_*.html and ignores generic explanatory BUY prose.
"""
from __future__ import annotations

import argparse, html, re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

@dataclass
class Event:
    file: str
    timestamp: str
    date: str
    ticker: str
    signal: str
    price: Optional[float] = None
    pivot: Optional[float] = None
    distance_pct: Optional[float] = None
    headroom_pct: Optional[float] = None
    vol_pace: Optional[float] = None
    adx: Optional[float] = None
    stage: str = ""
    reason: str = ""
    source: str = ""

def ts_from_name(p: Path):
    m = re.search(r"intraday_watch_(20\d{6})_(\d{6})\.html$", p.name)
    if not m: return None
    try: return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")
    except Exception: return None

def num(x):
    try: return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception: return None

def strip_tags(s):
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()

def parse_li_events(text, path, ts):
    events=[]
    for m in re.finditer(r"<li\b[^>]*>(.*?)</li>", text, re.I|re.S):
        raw=m.group(1); plain=strip_tags(raw)
        if "Weekly Stage 2 breakout confirmed" in plain: continue
        if not re.search(r"\b(BUY|NEAR|SELL)\s*:", plain, re.I): continue
        sm=re.search(r"\b(BUY|NEAR|SELL)\s*:", plain, re.I)
        main=re.search(
            r"\b(?P<ticker>[A-Z][A-Z0-9.\-]{0,9})\s*@\s*(?P<price>[0-9.,]+)\s*"
            r"\(\s*pivot\s*(?P<pivot>[0-9.,]+)\s*,\s*distance\s*(?P<distance>-?[0-9.,]+)%\s*,\s*"
            r"vol\s*(?P<vol>[0-9.,]+)x\s*,\s*ADX\s*(?P<adx>[0-9.,]+)\s*,\s*(?P<stage>[^)]*\))",
            plain, re.I)
        if main:
            hm=re.search(r"headroom\s*=\s*(-?[0-9.,]+)%", plain, re.I)
            events.append(Event(str(path), ts.isoformat(sep=" "), ts.date().isoformat(),
                main.group("ticker").upper(), sm.group(1).upper(), num(main.group("price")), num(main.group("pivot")),
                num(main.group("distance")), num(hm.group(1)) if hm else None, num(main.group("vol")), num(main.group("adx")),
                main.group("stage").strip(), plain, "ranked_li"))
            continue
        compact=re.search(r"\b(BUY|NEAR|SELL)\s*:\s*([A-Z][A-Z0-9.\-]{0,9})\s*@\s*([0-9.,]+)", plain, re.I)
        if compact:
            events.append(Event(str(path), ts.isoformat(sep=" "), ts.date().isoformat(), compact.group(2).upper(), compact.group(1).upper(), num(compact.group(3)), reason=plain, source="compact_li"))
    return events

def parse_table_events(text, path, ts):
    events=[]
    for tr in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", text, re.I|re.S):
        cells=[strip_tags(x) for x in re.findall(r"<td\b[^>]*>(.*?)</td>", tr.group(1), re.I|re.S)]
        if len(cells)<3: continue
        row=" | ".join(cells)
        if "Weekly Stage 2 breakout confirmed" in row: continue
        ticker=None
        for c in cells[:10]:
            if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", c): ticker=c.upper(); break
        if not ticker: continue
        signal=None
        for c in cells:
            cu=c.upper().strip()
            if cu in {"BUY","NEAR","NEAR_BUY","NEAR-TRIGGER","SELL","SELLTRIG","SELL-TRIGGER"}:
                signal = "NEAR" if cu.startswith("NEAR") else ("SELL" if cu.startswith("SELL") else "BUY")
                break
            if cu.startswith("WATCH_"):
                signal=cu; break
        if not signal: continue
        def find(pat):
            m=re.search(pat,row,re.I); return num(m.group(1)) if m else None
        events.append(Event(str(path), ts.isoformat(sep=" "), ts.date().isoformat(), ticker, signal,
            price=find(r"(?:price|px)\s*=\s*([0-9.,]+)"), pivot=find(r"pivot\s*=\s*([0-9.,]+)"),
            headroom_pct=find(r"headroom\s*=\s*(-?[0-9.,]+)%"), vol_pace=find(r"vol\s*=\s*([0-9.,]+)x"),
            adx=find(r"adx\s*=\s*([0-9.,]+)"), reason=row[:1000], source="table_row"))
    return events

def parse_file(p):
    ts=ts_from_name(p)
    if not ts: return []
    text=p.read_text(encoding="utf-8", errors="ignore")
    events=parse_li_events(text,p,ts)
    seen={(e.timestamp,e.ticker,e.signal,e.price,e.pivot,e.reason) for e in events}
    for e in parse_table_events(text,p,ts):
        k=(e.timestamp,e.ticker,e.signal,e.price,e.pivot,e.reason)
        if k not in seen:
            events.append(e); seen.add(k)
    return events

def lifecycle(df):
    if df.empty: return pd.DataFrame()
    x=df.copy(); x["dt"]=pd.to_datetime(x["timestamp"], errors="coerce"); rows=[]
    for t,g in x.sort_values("dt").groupby("ticker"):
        s=g["signal"].astype(str).str.upper()
        rows.append({"ticker":t,"first_seen":g.dt.min(),"last_seen":g.dt.max(),"events":len(g),"buy_events":int((s=="BUY").sum()),"near_events":int((s=="NEAR").sum()),"watch_events":int(s.str.startswith("WATCH").sum()),"unique_days":g.date.nunique(),"max_vol_pace":pd.to_numeric(g.vol_pace,errors="coerce").max(),"max_adx":pd.to_numeric(g.adx,errors="coerce").max(),"signal_path":" → ".join(s.drop_duplicates().head(25))})
    return pd.DataFrame(rows).sort_values(["buy_events","near_events","watch_events","events"], ascending=False)

def to_html(df, n=100):
    return "<p><i>No data.</i></p>" if df.empty else df.head(n).to_html(index=False, escape=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--glob",default="output/intraday_watch_*.html"); ap.add_argument("--days",type=int,default=45); ap.add_argument("--out-dir",default="output/html_signal_research"); ap.add_argument("--include-watch",action="store_true",default=True)
    a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); now=datetime.now(); cutoff=now-timedelta(days=a.days) if a.days>0 else None
    files=[]
    for p in sorted(Path(".").glob(a.glob)):
        ts=ts_from_name(p)
        if ts and (not cutoff or ts>=cutoff): files.append(p)
    events=[]
    for p in files: events.extend(parse_file(p))
    df=pd.DataFrame([asdict(e) for e in events])
    if not df.empty:
        df["signal"]=df.signal.astype(str).str.upper().str.strip()
        if not a.include_watch: df=df[~df.signal.str.startswith("WATCH",na=False)]
        df=df.drop_duplicates(subset=["file","ticker","signal","price","pivot","reason"]).sort_values(["timestamp","ticker","signal"])
    sig=df.signal.astype(str).str.upper() if not df.empty else pd.Series(dtype=str)
    daily=pd.DataFrame() if df.empty else df.assign(is_buy=sig.eq("BUY"), is_near=sig.eq("NEAR"), is_watch=sig.str.startswith("WATCH",na=False)).groupby("date").agg(html_runs=("file","nunique"),events=("ticker","count"),unique_tickers=("ticker","nunique"),buy_events=("is_buy","sum"),near_events=("is_near","sum"),watch_events=("is_watch","sum"),max_vol_pace=("vol_pace","max"),avg_vol_pace=("vol_pace","mean")).reset_index()
    life=lifecycle(df)
    stamp=now.strftime("%Y%m%d_%H%M%S")
    hp=out/f"prod_signal_history_from_html_{stamp}.csv"; dp=out/f"prod_signal_daily_summary_{stamp}.csv"; lp=out/f"prod_signal_lifecycle_{stamp}.csv"; htmlp=out/f"prod_signal_history_research_{stamp}.html"
    df.to_csv(hp,index=False); daily.to_csv(dp,index=False); life.to_csv(lp,index=False)
    buys=df[df.signal.isin(["BUY","NEAR"])].copy() if not df.empty else pd.DataFrame()
    page=f"""<html><head><title>Weinstein HTML Signal Research</title><style>body{{font-family:Arial;margin:24px}} table{{border-collapse:collapse;width:100%;font-size:12px}} td,th{{border:1px solid #ddd;padding:5px}} th{{background:#f2f4f8}}</style></head><body><h1>Weinstein HTML Signal Research</h1><p>Generated {now:%Y-%m-%d %H:%M:%S}. Historical reconstruction from archived intraday HTML. No trades. No threshold changes.</p><p><b>HTML files parsed:</b> {len(files)} &nbsp; <b>Events:</b> {len(df)} &nbsp; <b>Unique tickers:</b> {df.ticker.nunique() if not df.empty else 0}</p><h2>Daily Summary</h2>{to_html(daily,80)}<h2>Lifecycle by Ticker</h2>{to_html(life,100)}<h2>Recent BUY/NEAR Events</h2>{to_html(buys.tail(200),200)}</body></html>"""
    htmlp.write_text(page, encoding="utf-8")
    print("DONE HTML signal research"); print(f"HTML files parsed: {len(files)}"); print(f"Events extracted: {len(df)}")
    if not df.empty: print(df.signal.value_counts().to_string())
    print(f"history: {hp}"); print(f"daily: {dp}"); print(f"lifecycle: {lp}"); print(f"html: {htmlp}")
if __name__=="__main__": main()
