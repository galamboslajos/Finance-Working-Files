import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from dmv2.paper15_monthly_reporting import read_paper15_monthly, checked


class MonthlyReader(unittest.TestCase):
    def test_running_progress_imports_no_financial_or_model_libraries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ledger.json').write_text(json.dumps(dict(status='RUNNING',phase='fit-5-2020-rf',fits_completed=0)))
            with patch('builtins.__import__',wraps=__import__) as imported:
                result=read_paper15_monthly(root)
            self.assertEqual(result['status'],'RUNNING')
            self.assertFalse(any(x.args[0] in ('numpy','pandas','joblib','sklearn','pyarrow.parquet') for x in imported.call_args_list))

    def test_false_completion_rejected_before_reading_reports(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ledger.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',phase=None,fits_completed=1,fits_started=1,completed=[])))
            with self.assertRaises(ValueError):read_paper15_monthly(root)

    def test_size_guard_runs_before_reading_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report';path.write_text('x')
            with patch.object(Path,'read_bytes',side_effect=AssertionError('unexpected read')):
                with self.assertRaises(ValueError):checked(dict(path=str(path),size_bytes=100,sha256='x'),maximum=1)


if __name__=='__main__':unittest.main()
