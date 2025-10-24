def read_mapping_tables(gc):
    ws = open_ws(gc, "Mapping")
    df = ws_to_df(ws)
    tf_map = {}
    alias_map = {}

    if not df.empty:
        # Default timeframe (A:B)
        if {"Source","DefaultTimeframe"}.issubset(df.columns):
            tmp = df[["Source","DefaultTimeframe"]].dropna()
            for _, r in tmp.iterrows():
                s = str(r["Source"]).strip()
                t = str(r["DefaultTimeframe"]).strip()
                if s and t:
                    tf_map[s] = t

        # Aliases (D:E)
        if {"Alias","Ticker"}.issubset(df.columns):
            tmp = df[["Alias","Ticker"]].dropna()
            for _, r in tmp.iterrows():
                a = str(r["Alias"]).strip().upper()
                t = str(r["Ticker"]).strip().upper()
                if a and t:
                    alias_map[a] = t
    return tf_map, alias_map
