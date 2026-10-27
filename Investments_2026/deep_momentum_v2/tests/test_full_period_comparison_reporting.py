import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dmv2.full_period_comparison_reporting import read_full_period_comparison


class FullPeriodReportTests(unittest.TestCase):
    def test_unverified_report_stops_before_financial_read(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'verification.json').write_text(json.dumps(dict(status='PENDING')))
            with patch('builtins.__import__',wraps=__import__) as imp:
                with self.assertRaises(ValueError):read_full_period_comparison(d)
            self.assertFalse(any(c.args[0] in ('numpy','pandas','pyarrow','sklearn','xgboost','joblib') for c in imp.call_args_list))

    def test_withheld_full_period_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ip=root/'independent.json';ip.write_text(json.dumps(dict(status='PASS',complete_paths=47,withheld_paths=1,monthly_references_verified=16)))
            pin=dict(path=str(ip),size_bytes=ip.stat().st_size,sha256=hashlib.sha256(ip.read_bytes()).hexdigest())
            (root/'verification.json').write_text(json.dumps(dict(status='PASS_VERIFIED_MONTHLY_AND_DAILY_DISPLAY',monthly_tables_verified=16,new_fits=0,new_backtests=0,independent_proof_pin=pin)))
            with self.assertRaises(ValueError):read_full_period_comparison(root)

    def test_changed_proof_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ip=root/'independent.json';ip.write_text('{}')
            wrong=dict(path=str(ip),size_bytes=2,sha256='0'*64)
            (root/'verification.json').write_text(json.dumps(dict(status='PASS_VERIFIED_MONTHLY_AND_DAILY_DISPLAY',monthly_tables_verified=16,new_fits=0,new_backtests=0,independent_proof_pin=wrong)))
            with self.assertRaises(ValueError):read_full_period_comparison(root)
