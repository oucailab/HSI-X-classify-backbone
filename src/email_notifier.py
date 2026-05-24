import json
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from .config import PROJECT_ROOT


def load_email_config(project_root=PROJECT_ROOT) -> dict:
    config_path = Path(project_root) / "email_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"邮件配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_recipient(email_config: dict, experiment_config) -> str:
    override = (getattr(experiment_config, "email_recipient_override", "") or "").strip()
    if override:
        return override

    recipient = (email_config.get("default_recipient") or "").strip()
    if not recipient:
        raise ValueError("未配置默认收件人")
    return recipient


def extract_report_summary(report_path, max_lines=20) -> str:
    report_file = Path(report_path)
    if not report_file.exists():
        raise FileNotFoundError(f"报告文件不存在: {report_file}")

    lines = report_file.read_text(encoding="utf-8").splitlines()
    report_start = None

    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("-----------------------taskInfo-----------------------"):
            report_start = index
            break

    if report_start is None:
        raise ValueError("报告中未找到本次实验的起始位置")

    summary_lines = []
    for line in lines[report_start:report_start + max_lines]:
        summary_lines.append(line)
        if "Kappa accuracy (%)" in line:
            break

    summary = "\n".join(summary_lines).strip()
    if not summary:
        raise ValueError("报告中未找到本次实验内容")

    return summary


def send_report_email(subject: str, body: str, email_config: dict, recipient: str) -> None:
    sender_email = (email_config.get("sender_email") or "").strip()
    sender_password = (email_config.get("sender_password") or "").strip()
    smtp_host = (email_config.get("smtp_host") or "").strip()
    smtp_port = int(email_config.get("smtp_port") or 465)
    use_ssl = email_config.get("use_ssl", True)

    if not sender_email or not sender_password or not smtp_host:
        raise ValueError("邮件配置缺少 sender_email、sender_password 或 smtp_host")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient
    message.set_content(body)

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        return

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(sender_email, sender_password)
        server.send_message(message)


def notify_experiment_result(experiment_config) -> None:
    email_config = load_email_config()
    if not email_config.get("enabled", False):
        return

    recipient = resolve_recipient(email_config, experiment_config)
    summary = extract_report_summary(experiment_config.report_path)
    subject = (
        f"[{experiment_config.model_name}] "
        f"{experiment_config.experiment_name} - 训练报告"
    )
    send_report_email(subject, summary, email_config, recipient)
