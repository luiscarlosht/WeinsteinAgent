import smtplib, ssl
from email.message import EmailMessage
import yaml

def send_email(subject: str, html_body: str, text_body: str = None, cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)["email"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f'{cfg.get("from_name", "")} <{cfg["from_email"]}>'
    msg["To"] = ", ".join(cfg["to"])
    msg.set_content(text_body or html_body.replace("<br>", "\n"), subtype="plain")
    msg.add_alternative(html_body, subtype="html")

    if cfg.get("use_ssl", True):
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], context=context) as s:
            s.login(cfg["username"], cfg["app_password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(cfg["username"], cfg["app_password"])
            s.send_message(msg)

if __name__ == "__main__":
    send_email(
        subject="Test from VM",
        html_body="<h3>It works ✅</h3><p>This is a test email.</p>",
        text_body="It works - this is a test email."
    )
