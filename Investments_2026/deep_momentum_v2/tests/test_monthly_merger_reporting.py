import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dmv2.monthly_merger_reporting import read_monthly_merger_replay,checked


class ReplayReader(unittest.TestCase):
    def test_running_only_reads_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ledger.json').write_text(json.dumps(dict(status='RUNNING',phase='backtest',fits_started=0,fits_completed=0)))
            with patch('builtins.__import__',wraps=__import__) as imp:r=read_monthly_merger_replay(root)
            self.assertEqual(r['maximum_fits'],0);self.assertEqual(r['status'],'RUNNING')
            self.assertFalse(any(c.args[0] in ('numpy','pandas','pyarrow.parquet','joblib','sklearn') for c in imp.call_args_list))

    def test_fit_or_false_completion_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ledger.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',phase=None,fits_started=1,fits_completed=1,completed=[])))
            with self.assertRaises(ValueError):read_monthly_merger_replay(root)

    def test_small_report_guard_before_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('x')
            with patch.object(Path,'read_bytes',side_effect=AssertionError('unexpected read')):
                with self.assertRaises(ValueError):checked(dict(path=str(p),size_bytes=100,sha256='x'),maximum=1)


if __name__=='__main__':unittest.main()
