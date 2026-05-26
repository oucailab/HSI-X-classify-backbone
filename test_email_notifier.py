import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.email_notifier import extract_report_summary, resolve_recipient
from src.config import ExperimentConfig


class EmailNotifierTests(unittest.TestCase):
    def test_extract_report_summary_stops_after_kappa_line(self):
        report_text = """-----------------------taskInfo-----------------------\nlr:\t0.0001\nepoch_nums:\t50\n------------------------------------------------------\n\nReport time: 2026-05-21 10:00:00\n\n99.1 Overall accuracy (%)\n98.2 Average accuracy (%)\n97.3 Kappa accuracy (%)\n\nclassification details\n[[1 2] [3 4]]\n"""
        report_path = PROJECT_ROOT / "tmp_test_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        self.addCleanup(lambda: report_path.unlink(missing_ok=True))

        summary = extract_report_summary(report_path)

        self.assertIn("lr:\t0.0001", summary)
        self.assertIn("99.1 Overall accuracy (%)", summary)
        self.assertIn("98.2 Average accuracy (%)", summary)
        self.assertIn("97.3 Kappa accuracy (%)", summary)
        self.assertNotIn("classification details", summary)
        self.assertNotIn("[[1 2] [3 4]]", summary)

    def test_experiment_recipient_overrides_default_recipient(self):
        email_config = {"default_recipient": "default@example.com"}
        experiment_config = ExperimentConfig(email_recipient_override="override@example.com")

        recipient = resolve_recipient(email_config, experiment_config)

        self.assertEqual(recipient, "override@example.com")

    def test_standalone_email_script_uses_default_recipient(self):
        from scripts.test_email import build_test_email

        email_config = {
            "sender_email": "sender@qq.com",
            "default_recipient": "receiver@foxmail.com",
        }

        subject, body, recipient = build_test_email(email_config)

        self.assertEqual(subject, "QQ SMTP 测试邮件")
        self.assertEqual(recipient, "receiver@foxmail.com")
        self.assertIn("这是一封用于验证 SMTP 配置的测试邮件。", body)
        self.assertIn("发件人: sender@qq.com", body)
        self.assertIn("收件人: receiver@foxmail.com", body)


if __name__ == "__main__":
    unittest.main()
