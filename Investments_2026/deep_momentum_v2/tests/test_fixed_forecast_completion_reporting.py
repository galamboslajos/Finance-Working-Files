import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dmv2.fixed_forecast_completion_reporting import read_fixed_forecast_completed


class CompletionReaderTests(unittest.TestCase):
    def test_pending_reads_no_tables_or_parent(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'ledger.json').write_text(json.dumps(dict(status='RUNNING',phase='verify_completion')))
            with patch('dmv2.fixed_forecast_completion_reporting.read_fixed_forecast_reconciliation',side_effect=AssertionError('parent opened')):
                r=read_fixed_forecast_completed('/missing',d)
            self.assertEqual(r['status'],'RUNNING');self.assertEqual(r['new_fits'],0)

    def test_fitting_or_false_completion_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for fit in (0,1):
                (Path(d)/'ledger.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',phase=None,fits_started=fit,fits_completed=fit,completed=[])))
                with self.assertRaises(ValueError):read_fixed_forecast_completed('/missing',d)

    def test_wrong_receipt_path_rejected_before_content(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'ledger.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',phase=None,fits_started=0,fits_completed=0,completed=[dict(key='complete',receipt={}),dict(key='verify_completion',receipt={})])))
            with patch('dmv2.fixed_forecast_completion_reporting.checked',return_value=Path(d)/'outside.json'):
                with self.assertRaisesRegex(ValueError,'Unsafe'):read_fixed_forecast_completed('/missing',d)


if __name__=='__main__':unittest.main()
