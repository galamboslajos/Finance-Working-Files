import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dmv2.extension_backtest_reporting import read_extension_backtest


class ReaderTests(unittest.TestCase):
    def test_unverified_completion_never_reads_financial_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'ledger.json').write_text(json.dumps(dict(status='RUNNING',phase='verify_replay',fits_started=0,fits_completed=0)))
            with patch('builtins.__import__',wraps=__import__) as imports:
                with self.assertRaises(ValueError):read_extension_backtest(d)
            self.assertFalse(any(c.args[0] in ('numpy','pandas','pyarrow','sklearn','xgboost','joblib') for c in imports.call_args_list))

    def test_nonzero_fit_claim_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'ledger.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',phase=None,fits_started=1,fits_completed=1)))
            with self.assertRaises(ValueError):read_extension_backtest(d)

    def test_module_loads_without_numeric_or_estimator_imports(self):
        path=Path(__file__).resolve().parents[1]/'src/dmv2/extension_backtest_reporting.py'
        spec=importlib.util.spec_from_file_location('_passive75month',path);m=importlib.util.module_from_spec(spec)
        with patch('builtins.__import__',wraps=__import__) as imports:spec.loader.exec_module(m)
        self.assertFalse(any(c.args[0] in ('numpy','pandas','pyarrow','sklearn','xgboost','joblib') for c in imports.call_args_list))
