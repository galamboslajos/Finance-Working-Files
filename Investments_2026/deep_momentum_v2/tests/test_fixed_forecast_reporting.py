import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dmv2.fixed_forecast_reporting import read_fixed_forecast_reconciliation,checked


class FixedReaderTests(unittest.TestCase):
    def test_running_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'ledger.json').write_text(json.dumps(dict(status='RUNNING',phase='replay',active_seconds=3)))
            with patch('builtins.__import__',wraps=__import__) as imp:r=read_fixed_forecast_reconciliation(d)
            self.assertEqual(r['maximum_fits'],0);self.assertEqual(r['maximum_scenarios'],28)
            self.assertFalse(any(c.args[0] in ('numpy','pandas','pyarrow.parquet','joblib','sklearn') for c in imp.call_args_list))

    def test_false_complete_or_fit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'ledger.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',phase=None,fits_started=1,fits_completed=1,completed=[])))
            with self.assertRaises(ValueError):read_fixed_forecast_reconciliation(d)

    def test_size_guard_before_content_read(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'x';f.write_text('x')
            with patch.object(Path,'read_bytes',side_effect=AssertionError('unplanned read')):
                with self.assertRaises(ValueError):checked(dict(path=str(f),size_bytes=100,sha256='x'),maximum=1)


if __name__=='__main__':unittest.main()
