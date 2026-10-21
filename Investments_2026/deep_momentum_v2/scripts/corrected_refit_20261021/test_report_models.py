"""Tiny fabricated saved-score tests; no production reports/data/model loads."""
from pathlib import Path
import importlib.util
import math
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('report_under_test',Path(__file__).with_name('report_models.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def row(sid,month,true,pred,loss=None):
    y,m=map(int,month.split('-'));prev=f'{y-1}-12' if m==1 else f'{y}-{m-1:02d}'
    return {'source_security_id':sid,'signal_date':prev+'-28','outcome_month':month,'outcome_end':month+'-28',
            'label_available':true is not None,'true_class':true,'predicted_class':pred,
            'log_loss':None if true is None else loss,'correct':None if true is None else float(true==pred)}


class Tests(unittest.TestCase):
    def test_pooled_and_equal_month_have_different_explicit_denominators(self):
        rows=[row('A','2020-01',1,1,1),row('A','2020-02',1,1,3),
              row('B','2020-02',2,1,3),row('C','2020-02',2,1,3)]
        result,monthly,classes=g.aggregate(rows,months=['2020-01','2020-02'])
        self.assertEqual(result['pooled_log_loss'],2.5);self.assertEqual(result['mean_monthly_log_loss'],2)
        self.assertEqual(result['pooled_accuracy'],.5);self.assertAlmostEqual(result['mean_monthly_accuracy'],2/3)

    def test_missing_label_stays_in_prediction_coverage_not_score_or_precision(self):
        rows=[row('A','2020-01',1,1,.2),row('B','2020-01',None,1)]
        result,_,classes=g.aggregate(rows,months=['2020-01'])
        self.assertEqual(result['prediction_rows'],2);self.assertEqual(result['labelled_rows'],1)
        self.assertEqual(result['missing_label_rows'],1);self.assertEqual(result['label_coverage'],.5)
        self.assertEqual(result['pooled_log_loss'],.2);self.assertEqual(classes['1']['precision'],1)
        self.assertEqual(classes['1']['predicted_support'],2)
        self.assertEqual(classes['1']['missing_label_predicted_support'],1)

    def test_second_extreme_classes_remain_distinct(self):
        rows=[row('A','2020-01',1,2,1),row('B','2020-01',2,2,1),
              row('C','2020-01',9,10,1),row('D','2020-01',10,10,1)]
        _,_,c=g.aggregate(rows,months=['2020-01'])
        self.assertEqual(set(c),{'1','2','9','10'})
        self.assertEqual(c['2']['precision'],.5);self.assertEqual(c['2']['recall'],1)
        self.assertEqual(c['1']['recall'],0);self.assertIsNone(c['1']['precision'])
        self.assertEqual(c['10']['precision'],.5);self.assertEqual(c['10']['recall'],1)
        self.assertEqual(c['9']['recall'],0);self.assertIsNone(c['9']['precision'])

    def test_unobserved_month_not_imputed_as_zero_loss(self):
        result,monthly,_=g.aggregate([row('A','2020-01',None,1),row('A','2020-02',1,1,4)],months=['2020-01','2020-02'])
        self.assertEqual(result['observed_label_months'],1)
        self.assertIsNone(monthly[0]['log_loss']);self.assertEqual(result['mean_monthly_log_loss'],4)

    def test_saved_score_contract_rejects_invented_labels_and_wrong_correctness(self):
        r=row('A','2020-01',None,1);meta=[{k:r[k] for k in g.META}];values=[{k:r[k] for k in g.VALUES}]
        values[0]['log_loss']=0
        with self.assertRaisesRegex(ValueError,'invented score'):g.merge_scores(meta,values)
        r=row('A','2020-01',2,1,1);meta=[{k:r[k] for k in g.META}];values=[{k:r[k] for k in g.VALUES}]
        values[0]['correct']=1
        with self.assertRaisesRegex(ValueError,'Invalid observed'):g.merge_scores(meta,values)

    def test_metadata_scope_duplicates_and_missing_grid_fail_before_values(self):
        rows=[row('A',f'2020-{m:02d}',1,1,1) for m in range(1,13)]
        rows=[{k:r[k] for k in g.META} for r in rows]
        self.assertEqual(len(g.validate_metadata(rows,2020)),12)
        with self.assertRaisesRegex(ValueError,'Duplicate'):g.validate_metadata(rows+rows[:1],2020)
        with self.assertRaisesRegex(ValueError,'twelve-month'):g.validate_metadata(rows[:-1],2020)
        bad=[dict(r) for r in rows];bad[-1]['outcome_end']='2024-01-01'
        with self.assertRaisesRegex(ValueError,'timing'):g.validate_metadata(bad,2020)

    def test_footer_plan_uses_only_metadata_and_selected_leaf_bytes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'toy.parquet';pq.write_table(pa.Table.from_pylist([{'safe':['A','B'],'other':123.0}]),p)
            with patch.object(pq.ParquetFile,'read',side_effect=AssertionError('No values at planning')):
                z=g.projection(g.pin(p),['safe'])
            footer=pq.ParquetFile(p).metadata.row_group(0)
            expected=sum(footer.column(j).total_compressed_size for j in range(footer.num_columns)
                         if footer.column(j).path_in_schema.split('.')[0]=='safe')
            self.assertEqual(z['projected_compressed_bytes'],expected)
            self.assertEqual(z['columns'],['safe'])

    def test_monthly_and_mode_metadata_protected_grids_precede_numerical_rows(self):
        monthly=[{'outcome_month':f'2020-{m:02d}'} for m in range(1,13)]
        modes=[{'period':p,'class_id':c} for p in ['all',*[r['outcome_month'] for r in monthly]] for c in range(1,11)]
        g.validate_table_grids(monthly,modes,2020)
        with self.assertRaisesRegex(ValueError,'Monthly metadata'):g.validate_table_grids([{'outcome_month':'2024-01'},*monthly[1:]],modes,2020)
        with self.assertRaisesRegex(ValueError,'Mode metadata'):g.validate_table_grids(monthly,[*modes[:-1],modes[0]],2020)

    def test_report_prose_preserves_prior_cv_settings_and_no_portfolio_claim(self):
        s,monthly,c=g.aggregate([row('A','2020-01',1,1,1)],months=['2020-01'])
        fake={'arms':{arm:{'overall':s,'years':{str(y):s for y in g.YEARS},'classes':c} for arm in g.ARMS}}
        md=g.markdown(fake)
        self.assertIn('RF expanded · prior CV settings',md)
        self.assertIn('No new parameter search',md)
        self.assertIn('not top-K basket statistics or portfolio returns',md)
        self.assertIn('rejected by frozen volume controls',md)

    def test_completion_gate_refuses_running_pipeline_before_proof_reads(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);run=root/'run/models';g.save_new(run/'pipeline/progress.json',{'status':'RUNNING'})
            with patch.object(g,'ROOT',root),self.assertRaisesRegex(ValueError,'twenty fits'):
                g.completion_gate(run)


if __name__=='__main__':unittest.main()
