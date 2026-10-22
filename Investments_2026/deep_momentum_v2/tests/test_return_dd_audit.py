"""Controller tests use only invented records and temporary paths."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('ratio_audit_under_test',
    Path(__file__).parents[1] / 'src/dmv2/return_dd_audit.py')
a = importlib.util.module_from_spec(spec); sys.modules[spec.name] = a; spec.loader.exec_module(a)


class Controller(unittest.TestCase):
    def test_pins_authenticate_bytes_and_reject_extra_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'invented.json'; p.write_text('{}')
            item = a.pin(p); self.assertEqual(a.check(item), p.resolve())
            with self.assertRaises(ValueError): a.check({**item, 'extra': 1})
            p.write_text('[]')
            with self.assertRaises(ValueError): a.check(item)

    def test_relative_artifacts_do_not_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); p = root / 'sub' / 'x.json'
            a.save(p, {'invented': True})
            item = a.relative_pin(p, root)
            self.assertEqual(item['path'], 'sub/x.json')
            with self.assertRaises(ValueError): a.relative_pin(p, root / 'unrelated')

    def test_immutable_json_serialization_no_nan_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'x.json'; a.save(p, {'x': 1})
            with self.assertRaises(FileExistsError): a.save(p, {'x': 2})
            self.assertEqual(a.read(p), {'x': 1})
            with self.assertRaises(ValueError): a.canonical({'x': float('nan')})

    def test_closed_resource_policy(self):
        self.assertEqual(a._limits(64*1024**2)['rss_stop_bytes'], 4*1024**3)
        self.assertEqual(a._limits(128*1024**2)['estimator_fits'], 0)
        for invalid in (1, 65*1024**2, True, 4*1024**3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): a._limits(invalid)
        with self.assertRaises(ValueError): a._guard(time.monotonic()-601, a._limits(64*1024**2))
        with patch.object(a, '_rss_bytes', return_value=4*1024**3+1):
            with self.assertRaises(ValueError): a._guard(time.monotonic(), a._limits(64*1024**2))

    def test_protocol_formula_floor_cutoff_and_zero_fit_scope_checked(self):
        value={'schema':'dmv2_return_dd_target_protocol_v1',
               'status':'FROZEN_LABEL_ENGINEERING_AND_AUDIT_ONLY',
               'protocol_id':'signed_executable_return_over_drawdown_20261022_v1',
               'target_family':'return_over_drawdown',
               'formula':'cash_return / max(maximum_drawdown, denominator_floor)',
               'denominator_floor':.01,'observed_through':'2023-12-29',
               'history':['2010-01','2023-12'],
               'path':{'learning_variant':'unstopped','sides':['long','short']},
               'ranking':{'classes':10,'best_class':1,'minimum_observed':10},
               'payoffs':{'calendar_months':120,'required_month_class_cells':1200},
               'training_audit':{'years':[2020,2021,2022,2023]},
               'limits':{**a._limits(64*1024**2),'processes':1,
                         'independent_source_projection_bytes':64*1024**2}}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'protocol.json'; p.write_text(json.dumps(value))
            self.assertEqual(a._protocol(p,a._limits(64*1024**2)),value)
            for key,bad in [('denominator_floor',.02),('observed_through','2024-01-01'),
                            ('formula','cash_return'),('target_family','endpoint')]:
                changed=copy.deepcopy(value);changed[key]=bad;p.write_text(json.dumps(changed))
                with self.subTest(key=key),self.assertRaises(ValueError):a._protocol(p,a._limits(64*1024**2))
            changed=copy.deepcopy(value);changed['limits']['estimator_fits']=1;p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):a._protocol(p,a._limits(64*1024**2))

    def test_full_calendar168_and_financial_tampered_plan_rejected(self):
        self.assertEqual((len(a.month_range()),a.month_range()[0],a.month_range()[-1]),
                         (168,'2010-01','2023-12'))
        with self.assertRaises(ValueError): a.verify_plan({'schema':a.SCHEMA,'status':'RUNNING'})

    def test_projection_counts_only_named_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'invented.parquet'
            pq.write_table(pa.table({'metadata':['a','b'], 'financial':[.1,.2]}),p)
            pf=pq.ParquetFile(p)
            self.assertGreater(a.projection_bytes(pf,['metadata']),0)
            self.assertEqual(a.projection_bytes(pf,[]),0)
            self.assertEqual(a.projection_bytes(pf,['metadata'])+a.projection_bytes(pf,['financial']),
                             a.projection_bytes(pf,['metadata','financial']))
            for invalid in (['metadata','metadata'],['missing']):
                with self.assertRaises(ValueError):a.projection_bytes(pf,invalid)

    def test_training_completion_purge_preserves_original_counts(self):
        fold={'year':2020,'train_start_month':'2010-01','train_end_month':'2019-12','as_of':'2019-12-31'}
        rows=[]
        for i,month in enumerate(('2010-01','2019-11','2019-12','2019-12','2019-12','2020-01')):
            rows.append({'side':'long','outcome_month':month,'prediction_eligible':i!=3,
                         'target_available':i!=4,'label_available_at':'2020-01-02' if i>=2 else '2019-12-02'})
        rows.append({**rows[0],'side':'short'})
        result=a._training_counts(rows,fold,'long')
        self.assertEqual((result['original_rows'],result['prediction_eligible_rows'],result['target_available_rows'],
                          result['fit_rows'],result['incomplete_or_unavailable_rows'],result['uncompleted_boundary_rows']),
                         (5,4,3,2,2,2))
        self.assertFalse(result['fit_performed']);self.assertTrue(result['prediction_eligibility_unchanged'])

    def test_source_projection_excludes_protected_financial_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'invented.parquet'
            window={'signal_date':'2023-11-30','entry_date':'2023-12-01','exit_date':'2024-01-02',
                    'exit_mark':'open','label_available_at':'2024-01-02'}
            rows=[]
            for family in ('executable_return','return_minus_drawdown'):
                for variant in ('unstopped','asset_stop_10pct'):
                    for side in ('long','short'):
                        rows.append({**window,'side':side,'target_family':family,'path_variant':variant,
                                     'prediction_eligible':True,'cash_return':999.,'maximum_drawdown':999.,
                                     'target_class':1})
            pq.write_table(pa.Table.from_pylist(rows),p)
            pf=pq.ParquetFile(p)
            metadata=[name for name in pf.schema_arrow.names if name not in a.NUMERIC]
            month={'source_target_pin':a.pin(p),'metadata_columns':metadata,'value_columns':[],
                   'projected_metadata_bytes':a.projection_bytes(pf,metadata),'projected_value_bytes':0,
                   'original_rows':1,'eligible_rows':1,'source_window':window}
            reservation={'source_projection_bytes':0}
            original=pq.read_table;requested=[]
            def record(*args,**kwargs):
                requested.extend(kwargs['columns'])
                self.assertFalse(set(a.NUMERIC)&set(kwargs['columns']))
                return original(*args,**kwargs)
            with patch.object(pq,'read_table',side_effect=record):table=a._source_month(month,reservation)
            self.assertEqual(table.num_rows,2)
            self.assertEqual(table['cash_return'].to_pylist(),[None,None])
            self.assertEqual(table['maximum_drawdown'].to_pylist(),[None,None])
            self.assertEqual(table['target_class'].to_pylist(),[None,None])
            self.assertEqual(table['target_value'].to_pylist(),[None,None])
            self.assertEqual(reservation['source_projection_bytes'],month['projected_metadata_bytes'])
            self.assertEqual(set(table['target_family'].to_pylist()),{'executable_return'})
            self.assertEqual(set(table['path_variant'].to_pylist()),{'unstopped'})

    def test_wrong_source_admission_and_clock_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'invented.parquet'
            window={'signal_date':'2019-12-31','entry_date':'2020-01-02','exit_date':'2020-02-03',
                    'exit_mark':'open','label_available_at':'2020-02-03'}
            rows=[{**window,'side':side,'target_family':'executable_return','path_variant':'unstopped',
                   'prediction_eligible':side=='long','cash_return':.1,'maximum_drawdown':.2,'target_class':1}
                  for side in ('long','short')]
            pq.write_table(pa.Table.from_pylist(rows),p);pf=pq.ParquetFile(p)
            metadata=[name for name in pf.schema_arrow.names if name not in a.NUMERIC]
            values=list(a.VALUE_COLUMNS)
            month={'source_target_pin':a.pin(p),'metadata_columns':metadata,'value_columns':values,
                   'projected_metadata_bytes':a.projection_bytes(pf,metadata),
                   'projected_value_bytes':a.projection_bytes(pf,values),
                   'original_rows':1,'eligible_rows':1,'source_window':window}
            with self.assertRaisesRegex(ValueError,'admission'):
                a._source_month(month,{'source_projection_bytes':0})


if __name__=='__main__':unittest.main()
