#!/usr/bin/env python3
import math, re, argparse
from collections import deque, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
SERVICE_ACCOUNT_FILE = "creds/gcp_service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_URL = "https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit"

TAB_SIGNALS      = "Signals"
TAB_TRANSACTIONS = "Transactions"
TAB_HOLDINGS     = "Holdings"
TAB_REALIZED     = "Realized_Trades"
TAB_OPEN         = "Open_Positions"
TAB_PERF         = "Performance_By_Source"

DEFAULT_EXCHANGE_PREFIX = "NASDAQ: "

# ─────────────────────────────
# UTILITIES
# ─────────────────────────────
def auth_gspread():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(gc, tab):
    sh = gc.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=2000, cols=26)

def read_tab(ws) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)

def to_dt(series): return pd.to_datetime(series, errors="coerce", utc=True)

def to_float(series):
    def conv(x):
        if isinstance(x, str):
            x = x.replace("$","").replace(",","").strip()
        try: return float(x)
        except: return np.nan
    return series.map(conv)

BLACKLIST_TOKENS = {"CASH","USD","INTEREST","DIVIDEND","REINVESTMENT","FEE","WITHDRAWAL","DEPOSIT","TRANSFER","SWEEP"}

def base_symbol_from_string(s):
    if not s or (isinstance(s,float) and math.isnan(s)): return ""
    s = str(s).strip()
    token = s.split()[0].split("-")[0].replace("(","").replace(")","")
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token).upper()
    if not token or token in BLACKLIST_TOKENS or token.isdigit() or (len(token)>8 and token.isalnum()):
        return ""
    return token

def write_tab(ws, df):
    ws.clear()
    if df.empty:
        ws.update(values=[["(empty)"]], range_name="A1")
        return
    rows, cols = df.shape
    ws.resize(rows=max(100, rows+5), cols=max(min(26, cols+2), 8))
    data = [[str(c) for c in df.columns]] + df.astype(str).fillna("").values.tolist()
    chunk, r = 500, 1
    while r <= len(data):
        end = min(r+chunk-1, len(data))
        block = data[r-1:end]
        top = gspread.utils.rowcol_to_a1(r, 1)
        bottom = gspread.utils.rowcol_to_a1(r+len(block)-1, len(df.columns))
        ws.update(values=block, range_name=f"{top}:{bottom}")
        r += len(block)

def read_mapping(gc):
    try:
        df = read_tab(open_ws(gc, "Mapping"))
        out = {}
        if not df.empty and "Ticker" in df.columns:
            for _,row in df.iterrows():
                t = str(row.get("Ticker","")).strip().upper()
                if not t: continue
                out[t] = {
                    "FormulaSym": str(row.get("FormulaSym","")).strip(),
                    "TickerYF": str(row.get("TickerYF","")).strip().upper()
                }
        return out
    except: return {}

def googlefinance_formula_for(ticker, row_idx, mapping):
    base = base_symbol_from_string(ticker)
    mapped = mapping.get(ticker, {}) or mapping.get(base, {}) or {}
    sym = mapped.get("FormulaSym") or (DEFAULT_EXCHANGE_PREFIX + base)
    return f'=IFERROR(GOOGLEFINANCE("{sym}","price"), IFERROR(GOOGLEFINANCE(B{row_idx},"price"), ""))'

# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
def load_signals(df):
    if df.empty: return df
    df["Ticker"] = df[df.columns[df.columns.str.lower().isin(["ticker","symbol"])][0]].map(base_symbol_from_string)
    df["Source"] = df.get("Source","")
    df["Direction"] = df.get("Direction","")
    df["Timeframe"] = df.get("Timeframe","")
    ts = next((c for c in df.columns if c.lower().startswith("timestamp")), None)
    df["TimestampUTC"] = to_dt(df[ts]) if ts else pd.NaT
    df["Price"] = df.get("Price","")
    return df[["TimestampUTC","Ticker","Source","Direction","Price","Timeframe"]]

def _symbol_from_parens(t): m=re.search(r"\(([A-Za-z0-9\.\-]{1,10})\)", t or "");return base_symbol_from_string(m.group(1)) if m else ""

def load_transactions(df, debug=False):
    if df.empty: return df
    symcol=next((c for c in df.columns if c.lower() in ("symbol","security","symbol/cusip")),None)
    typecol=next((c for c in df.columns if c.lower() in ("type","action")),None)
    desccol=next((c for c in df.columns if "description" in c.lower()),None)
    pricecol=next((c for c in df.columns if "price" in c.lower()),None)
    qtycol=next((c for c in df.columns if "quantity" in c.lower()),None)
    amtcol=next((c for c in df.columns if "amount" in c.lower()),None)
    datecol=next((c for c in df.columns if "run date" in c.lower()),None)
    df=df.copy();when=to_dt(df[datecol])
    action=df[typecol].fillna("").astype(str) if typecol else pd.Series("",index=df.index)
    desc=df[desccol].fillna("").astype(str) if desccol else pd.Series("",index=df.index)
    patt=r"(BOUGHT|SOLD|BUY|SELL)"
    mask=action.str.upper().str.contains(patt)|desc.str.upper().str.contains(patt)
    df=df[mask].copy()
    if debug: print(f"• load_transactions: detected {mask.sum()} trade-like rows (of {len(df)})")
    if symcol: sym=df[symcol].map(base_symbol_from_string)
    else: sym=pd.Series("",index=df.index)
    sym=sym.mask(sym.eq(""), action.map(_symbol_from_parens))
    sym=sym.mask(sym.eq(""), desc.map(_symbol_from_parens))
    price=to_float(df[pricecol]) if pricecol else pd.Series(np.nan,index=df.index)
    if qtycol: qty=to_float(df[qtycol])
    elif amtcol and pricecol:
        amt,prc=to_float(df[amtcol]),to_float(df[pricecol])
        qty=pd.Series(np.where((prc!=0)&~np.isnan(prc)&~np.isnan(amt),np.abs(amt)/np.abs(prc),np.nan),index=df.index)
    else: qty=pd.Series(np.nan,index=df.index)
    tx=pd.DataFrame({"When":when,"Type":action.str.upper(),"Symbol":sym.fillna(""),"Qty":pd.to_numeric(qty,errors="coerce"),"Price":pd.to_numeric(price,errors="coerce")})
    tx=tx[(tx.Symbol!="")&tx.When.notna()&(tx.Qty>0)].copy().sort_values("When").reset_index(drop=True)
    if debug and not tx.empty: print(tx.head(8).to_string(index=False))
    return tx

def load_holdings(df):
    if df.empty: return df
    s=next((c for c in df.columns if c.lower() in ("symbol","security","symbol/cusip")),None)
    q=next((c for c in df.columns if "quantity" in c.lower()),None)
    p=next((c for c in df.columns if "price" in c.lower()),None)
    out=pd.DataFrame()
    if s: out["Ticker"]=df[s].map(base_symbol_from_string)
    if q: out["Qty"]=to_float(df[q])
    if p: out["Price"]=to_float(df[p])
    return out

# ─────────────────────────────
# BUILD REALIZED + OPEN
# ─────────────────────────────
def build_realized_and_open(tx, sig):
    if tx.empty: return pd.DataFrame(), pd.DataFrame()
    sig_buy=sig[(sig.Direction.str.upper()=="BUY") & (sig.Ticker!="")].copy().sort_values(["Ticker","TimestampUTC"])
    sig_by=defaultdict(list)
    for _,r in sig_buy.iterrows():
        sig_by[r.Ticker].append((r.TimestampUTC,{"Source":r.get("Source",""),"Timeframe":r.get("Timeframe",""),"SigTime":r.get("TimestampUTC"),"SigPrice":r.get("Price","")}))
    def last_signal_for(t,when):
        for ts,p in reversed(sig_by.get(t,[])):
            if pd.isna(ts) or pd.isna(when): return p
            if ts<=when: return p
        return {"Source":"(unknown)","Timeframe":"","SigTime":pd.NaT,"SigPrice":""}
    lots=defaultdict(deque);real=[]
    for _,r in tx.iterrows():
        tkr,when,ttype,qty,price=r.Symbol,r.When,r.Type,r.Qty,r.Price
        if qty<=0 or pd.isna(when) or not tkr: continue
        is_buy = bool(re.search(r"\b(BOUGHT|BUY)\b", ttype))
        is_sell= bool(re.search(r"\b(SOLD|SELL)\b",  ttype))
        if is_buy and not is_sell:
            s=last_signal_for(tkr,when)
            lots[tkr].append({"qty_left":qty,"entry_price":price,"entry_time":when,"source":s["Source"],"timeframe":s["Timeframe"],"sig_time":s["SigTime"],"sig_price":s["SigPrice"]})
        elif is_sell:
            remaining=qty
            while remaining>0 and lots[tkr]:
                lot=lots[tkr][0];take=min(remaining,lot["qty_left"])
                if take<=0: break
                entry,exitp=lot["entry_price"],price
                ret=((exitp-entry)/entry*100) if entry else np.nan
                held=(when-lot["entry_time"]).days if not pd.isna(lot["entry_time"]) else ""
                real.append({"Ticker":tkr,"Qty":round(take,6),"EntryPrice":entry,"ExitPrice":exitp,"Return%":round(ret,4) if not np.isnan(ret) else "","HoldDays":held,"EntryTimeUTC":lot["entry_time"],"ExitTimeUTC":when,"Source":lot["source"],"Timeframe":lot["timeframe"],"SignalTimeUTC":lot["sig_time"],"SignalPrice":lot["sig_price"]})
                lot["qty_left"]-=take;remaining-=take
                if lot["qty_left"]<=1e-9: lots[tkr].popleft()
    realized=pd.DataFrame(real).sort_values("ExitTimeUTC",ignore_index=True) if real else pd.DataFrame()
    now=pd.Timestamp.now(tz="UTC");open_=[]
    for tkr,q in lots.items():
        for lot in q:
            if lot["qty_left"]<=1e-9: continue
            open_.append({"Ticker":tkr,"OpenQty":round(lot["qty_left"],6),"EntryPrice":lot["entry_price"],"EntryTimeUTC":lot["entry_time"],"DaysOpen":(now-lot["entry_time"]).days if not pd.isna(lot["entry_time"]) else "","Source":lot["source"],"Timeframe":lot["timeframe"],"SignalTimeUTC":lot["sig_time"],"SignalPrice":lot["sig_price"]})
    open_df=pd.DataFrame(open_).sort_values("EntryTimeUTC",ignore_index=True) if open_ else pd.DataFrame()
    return realized,open_df

def add_live_price_formulas(df,mapping):
    if df.empty: return df
    out=df.copy();price_now=[];unreal=[]
    for i,r in out.iterrows():
        sym=r.Ticker;ep=r.get("EntryPrice");row=i+2
        f=googlefinance_formula_for(sym,row,mapping);price_now.append(f)
        try: epf=float(ep);unreal.append(f'=IFERROR(({f}/{epf}-1)*100,"")' if epf>0 else "")
        except: unreal.append("")
    out.insert(out.columns.get_loc("EntryPrice")+1,"PriceNow",price_now)
    out.insert(out.columns.get_loc("PriceNow")+1,"Unrealized%",unreal)
    return out

def build_perf_by_source(realized,open_df,sig):
    if realized.empty:
        rg=pd.DataFrame(columns=["Source","Trades","Wins","WinRate%","AvgReturn%","MedianReturn%"])
    else:
        t=realized.copy();t["ret"]=pd.to_numeric(t["Return%"],errors="coerce");t["is_win"]=t["ret"]>0
        g=t.groupby("Source",dropna=False)
        rg=pd.DataFrame({"Source":g.size().index,"Trades":g.size().values,"Wins":g["is_win"].sum().values,"WinRate%":(g["is_win"].mean().fillna(0)*100).round(2),"AvgReturn%":g["ret"].mean().round(2).values,"MedianReturn%":g["ret"].median().round(2).values})
    oc=open_df.groupby("Source").size().rename("OpenSignals").reset_index() if not open_df.empty else pd.DataFrame(columns=["Source","OpenSignals"])
    p=pd.merge(rg,oc,on="Source",how="outer")
    for c in ["Trades","Wins","OpenSignals"]: p[c]=pd.to_numeric(p[c],errors="coerce").fillna(0).astype(int)
    for c in ["WinRate%","AvgReturn%","MedianReturn%"]: p[c]=pd.to_numeric(p[c],errors="coerce").fillna(0.0)
    return p.sort_values("Source").reset_index(drop=True)

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--debug",action="store_true");args=parser.parse_args()
    print("📊 Building performance dashboard…")
    gc=auth_gspread()
    ws_sig,ws_tx,ws_h=[open_ws(gc,t) for t in (TAB_SIGNALS,TAB_TRANSACTIONS,TAB_HOLDINGS)]
    df_sig,df_tx,df_h=[read_tab(w) for w in (ws_sig,ws_tx,ws_h)]
    print(f"• Loaded: Signals={len(df_sig)} rows, Transactions={len(df_tx)} rows, Holdings={len(df_h)} rows")
    sig=load_signals(df_sig);tx=load_transactions(df_tx,debug=args.debug);_ = load_holdings(df_h)
    realized,open_df=build_realized_and_open(tx,sig)
    mapping=read_mapping(gc);open_df=add_live_price_formulas(open_df,mapping)
    if not realized.empty: realized=realized[["Ticker","Qty","EntryPrice","ExitPrice","Return%","HoldDays","EntryTimeUTC","ExitTimeUTC","Source","Timeframe","SignalTimeUTC","SignalPrice"]]
    if not open_df.empty: open_df=open_df[["Ticker","OpenQty","EntryPrice","PriceNow","Unrealized%","EntryTimeUTC","DaysOpen","Source","Timeframe","SignalTimeUTC","SignalPrice"]]
    perf=build_perf_by_source(realized.copy(),open_df.copy(),sig)
    for tab,df in [(TAB_REALIZED,realized),(TAB_OPEN,open_df),(TAB_PERF,perf)]:
        write_tab(open_ws(gc,tab),df);print(f"✅ Wrote {tab}: {len(df)} rows")
    print("🎯 Done.")

if __name__=="__main__": main()
