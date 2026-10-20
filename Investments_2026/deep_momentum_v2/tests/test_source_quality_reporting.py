import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dmv2.source_quality_reporting import read_source_quality_repair
from dmv2.universe import DataContractError


class SourceQualityReportingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root/'configs').mkdir()
        (self.root/'reports').mkdir()
        self.artifact = self.root/'reports/fictional_integrity_record.txt'
        self.artifact.write_text('fictional nonfinancial fixture')
        self.handoff = {'status':'REPAIRED_INPUTS_READY_MODELS_STALE',
            'financial_2024_plus_analysis':False,'new_model_fits':0,'new_backtests':0,
            'existing_runs_modified':False,'summary':{'fixture':True},
            'dependent_fits':[],'artifacts':[self.pin(self.artifact)]}
        self.save()

    def pin(self,path):
        return {'path':str(path.relative_to(self.root)), 'size_bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def save(self):
        self.manifest = self.root/'reports/handoff.json'
        self.manifest.write_text(json.dumps(self.handoff))
        (self.root/'configs/source_quality_repair_20261016.json').write_text(
            json.dumps({'handoff':self.pin(self.manifest)}))

    def test_reports_stale_models_without_loading_them(self):
        result = read_source_quality_repair(self.root)
        self.assertEqual(result['status'],'REPAIRED_INPUTS_READY_MODELS_STALE')
        self.assertEqual(result['verified_files'],1)
        self.assertFalse(result['financial_values_loaded'])
        self.assertFalse(result['models_loaded_or_fitted'])

    def test_tampered_integrity_record_fails(self):
        self.artifact.write_text('changed fixture of different size')
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root)

    def test_changed_handoff_fails_before_report_use(self):
        self.manifest.write_text('changed')
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root)

    def test_path_escape_fails(self):
        self.handoff['artifacts'][0]['path'] = '../outside.txt'
        self.save()
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root)

    def test_file_count_budget_rejects(self):
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root, maximum_files=0)

    def test_total_byte_budget_rejects(self):
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root, maximum_bytes=1)

    def test_duplicate_pins_fail(self):
        self.handoff['artifacts'] *= 2
        self.save()
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root)

    def test_inflated_completion_claim_fails(self):
        self.handoff['new_model_fits'] = 16
        self.save()
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root)

    def test_protected_analysis_claim_fails(self):
        self.handoff['financial_2024_plus_analysis'] = True
        self.save()
        with self.assertRaises(DataContractError):
            read_source_quality_repair(self.root)


if __name__ == '__main__':
    unittest.main()
