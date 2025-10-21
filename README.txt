# Weinstein Hybrid Starter (VM + Mac iMessage)
Contents:
- config.yaml
- subject_body_template.txt
- cron_examples.txt
- control_tab_layout.txt
- mail_rule_and_shortcut_instructions.txt

Next steps:
1) Edit config.yaml on your VM.
2) Set up cron using cron_examples.txt.
3) Create Google Sheets (Daily_Intake, Holdings, Control).
4) On your Mac, create the Mail rule + Shortcut.
5) Test: ssh user@vm 'python3 /opt/weinstein/run.py --mode quick --tickers NVDA,ANET --send-email --summary short'
