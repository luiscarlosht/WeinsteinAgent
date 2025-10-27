(.venv) luiscarlosht@andree-martin-hs-band-fundraising-2025-bot:~/WeinsteinAgent$ ./run_weekly.sh 
🧾 Using config: ./config.yaml
• Google Sheet: https://docs.google.com/spreadsheets/d/17eYLngeM_SbasWRVSy748J-RltTRli1_4od6mlZnpW4/edit
• Open Positions tab: Open_Positions
• Signals tab:        Signals
• Output dir:         ./output
📊 Building portfolio dashboard (Sheets)…
📊 Building performance dashboard…
🔑 Authorizing service account…
• Loaded: Signals=47 rows, Transactions=174 rows, Holdings=33 rows
• load_transactions: detected 128 trade-like rows (of 174)
✅ Wrote Realized_Trades: 60 rows
✅ Wrote Open_Positions: 45 rows
✅ Wrote Performance_By_Source: 7 rows
✅ Wrote OpenLots_Detail: 45 rows
⚠️ Summary: 5 unmatched SELL events (use --debug to print details).
🔍 Open breakdown:
  - Bo Xu: IFBD → 3 lot(s)
  - ChatGPT: VUG → 2 lot(s)
  - Sarkee Capital: ANET → 2 lot(s)
  - Sarkee Capital: APLD → 2 lot(s)
  - Sarkee Capital: BITF → 2 lot(s)
  - Sarkee Capital: IONQ → 2 lot(s)
  - Sarkee Capital: LAC → 2 lot(s)
  - Sarkee Capital: ROIV → 2 lot(s)
  - Sarkee Capital: UUUU → 2 lot(s)
  - Sarkee Capital: WBD → 2 lot(s)
  - SuperiorStar: CLSK → 2 lot(s)
  - SuperiorStar: CORZ → 2 lot(s)
  - SuperiorStar: CRCL → 1 lot(s)
  - SuperiorStar: CRM → 1 lot(s)
  - SuperiorStar: HOOD → 1 lot(s)
  - SuperiorStar: INTC → 2 lot(s)
  - SuperiorStar: NVDA → 2 lot(s)
  - SuperiorStar: PLTR → 3 lot(s)
  - SuperiorStar: TSM → 1 lot(s)
  - Weinstein: ALB → 2 lot(s)
  - Weinstein: APH → 1 lot(s)
  - Weinstein: EME → 1 lot(s)
  - Weinstein: F → 2 lot(s)
  - Weinstein: GM → 2 lot(s)
  - Weinstein: HCA → 1 lot(s)
🎯 Done.
📰 Generating Weinstein Weekly (portfolio) report…
Universe size: 505 tickers (benchmark: SPY)
Downloading weekly data (Yahoo Finance)…
Traceback (most recent call last):
  File "/home/luiscarlosht/WeinsteinAgent/weinstein_report_weekly.py", line 493, in <module>
    main()
  File "/home/luiscarlosht/WeinsteinAgent/weinstein_report_weekly.py", line 402, in main
    close_w, volume_w = fetch_weekly(tickers, benchmark, weeks=WEEKS_LOOKBACK)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/luiscarlosht/WeinsteinAgent/weinstein_report_weekly.py", line 102, in fetch_weekly
    close = _extract_field(data, "Close", uniq)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/luiscarlosht/WeinsteinAgent/weinstein_report_weekly.py", line 45, in _extract_field
    raise KeyError(f"Field '{field}' not in downloaded data.")
KeyError: "Field 'Close' not in downloaded data."
🔎 Running classic Weinstein scan (sp500, bench SPY)…
ℹ️  Classic scan CSV not found at ./output/scan_sp500.csv. Skipping classic merge.
🧩 Assembling combined weekly HTML…
✅ Combined weekly report written: output/combined_weekly_email.html
Primary email step did not complete (rc=1) or forced. Attempting fallback email sender…
Fallback email not sent: no recipients under email.to in config.yaml
✅ Weekly pipeline finished (Sheets + combined report).
(.venv) luiscarlosht@andree-martin-hs-band-fundraising-2025-bot:~/WeinsteinAgent$ 
