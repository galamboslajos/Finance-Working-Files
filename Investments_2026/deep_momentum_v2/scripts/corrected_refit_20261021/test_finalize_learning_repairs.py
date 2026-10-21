import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
import control as c
import learning_repairs as lr
import finalize_learning_repairs as f
from test_learning_repairs import rcm_fixture,disca_fixture


class Finalization(unittest.TestCase):
    def roundtrip(self,reject=False):
        rcm,inv=rcm_fixture();disca,controls=disca_fixture()
        if reject:controls[0]['volume']*=1.001
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'repairs').mkdir()
            pq.write_table(pa.Table.from_pylist(rcm),root/'repairs/rcm_source_full_rows.parquet')
            pq.write_table(pa.Table.from_pylist(disca),root/'repairs/disca_source_raw_all.parquet')
            pq.write_table(pa.Table.from_pylist(controls),root/'controls.parquet')
            c.save_new(root/'repairs/rcm_dated_key_inventory.json',inv)
            control={'schema':'dmv2-learning-repair-finalization-control/v1','disca_target_controls_pin':c.pin(root/'controls.parquet'),'evidence_pins':[]}
            c.save_new(root/'control.json',control)
            # Pin functions are strict; only test-local fixed module dependencies are supplied.
            (root/'control.py').write_text('synthetic');(root/'learning_repairs.py').write_text('synthetic')
            with patch.object(f,'ROOT',root):
                pp=f.prepare_plan(root/'control.json',root/'plan.json')
                launch={'schema':'dmv2-learning-repair-finalization-launch/v1','status':'SEALED_BEFORE_DERIVED_PRICE_READ','plan_pin':pp,'worker_pin':c.pin(f.__file__),'reservation_label':'synthetic'}
                c.save_new(root/'launch.json',launch)
                with patch.object(f,'require_reservation'),contextlib.redirect_stdout(io.StringIO()):f.run(root/'plan.json',root/'launch.json')
            manifest=c.read(root/'repairs/learning_quote_repair_manifest.json')
            return manifest,pq.read_table(root/'repairs/learning_quote_repairs.parquet')
    def test_two_accepted_cases(self):
        manifest,quotes=self.roundtrip();self.assertEqual(manifest['row_count'],165)
        self.assertEqual(quotes.num_columns,20);self.assertEqual(quotes.column_names,list(lr.PRICE_COLUMNS))
        self.assertEqual(manifest['status'],'PASS')
    def test_disca_quality_rejection_still_pipeline_pass(self):
        manifest,quotes=self.roundtrip(True);self.assertEqual(manifest['row_count'],160)
        self.assertEqual(manifest['status'],'PASS');self.assertEqual(manifest['cases']['DISCA']['status'],'REJECTED_SOURCE_QUALITY')
        self.assertTrue(all(x=='sec_zMM1vd' for x in quotes['intrinio_security_id'].to_pylist()))
    def test_guard_future_control_key(self):
        table=pa.Table.from_pylist([{'intrinio_security_id':lr.DISCA_TARGET,'trading_date':'2024-01-01'},{'intrinio_security_id':lr.DISCA_TARGET,'trading_date':'2022-04-11'}])
        with self.assertRaises(ValueError):f.validate_metadata('disca_controls',table,{})
    def test_guard_wrong_source_literal(self):
        rows,_=disca_fixture();rows[0]['query_symbol']='DISCK'
        with self.assertRaises(ValueError):f.validate_metadata('disca',pa.Table.from_pylist(rows),{})

if __name__=='__main__':unittest.main()
