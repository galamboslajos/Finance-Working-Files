import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dmv2.target_comparison_reporting import read_target_comparison


def save(path, body):
    path.write_text(json.dumps(body, sort_keys=True)+'\n')


def fixture(root):
    root = root.resolve()
    body = dict(expected_fits=20, expected_preprocessors=4, financial_2024_plus_allowed=False)
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    body['plan_sha256'] = 'sha256:' + hashlib.sha256(raw).hexdigest()
    path = root/'read_plan.json'; save(path, body)
    pin = dict(path=str(path), size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(root/'launch.json',dict(model_plan_pin=pin))
    save(root/'ledger.json',dict(model_plan_pin=pin, status='RUNNING', fits_started=4, fits_completed=3))


class PassiveReportingTests(unittest.TestCase):
    def test_missing_run_does_not_create_anything(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'absent'
            with self.assertRaises(FileNotFoundError): read_target_comparison(root)
            self.assertFalse(root.exists())

    def test_running_progress_never_exposes_unverified_results(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fixture(root)
            answer=read_target_comparison(root)
            self.assertEqual(answer['status']['fits_completed'],3)
            self.assertIsNone(answer['comparison'])
            self.assertFalse(answer['backtest_independently_verified'])
            self.assertFalse(answer['automatic_computation'])

    def test_changed_plan_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fixture(root)
            path=root/'read_plan.json'; path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'another comparison'): read_target_comparison(root)

    def test_fit_scope_cannot_be_inflated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fixture(root); path=root/'ledger.json'; body=json.loads(path.read_text())
            body['fits_started']=21; save(path,body)
            with self.assertRaisesRegex(ValueError,'fit/financial scope'): read_target_comparison(root)

    def test_complete_flag_without_independent_proofs_does_not_show_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fixture(root); path=root/'ledger.json'; body=json.loads(path.read_text())
            body['status']='COMPLETE_VERIFIED'; save(path,body)
            with self.assertRaises(FileNotFoundError): read_target_comparison(root)


if __name__=='__main__': unittest.main()
