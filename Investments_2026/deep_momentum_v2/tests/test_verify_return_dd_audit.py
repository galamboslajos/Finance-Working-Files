"""Invented-record checks for the independent verifier; no research data reads."""
import copy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

WORKER = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_return_dd_audit.py'
spec = importlib.util.spec_from_file_location('independent_ratio_audit', WORKER)
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def invented(month='2020-01'):
    protected = month == '2023-12'
    window = {'signal_date': '2023-11-30' if protected else '2019-12-31',
        'outcome_month': month, 'entry_date': '2023-12-01' if protected else '2020-01-02',
        'exit_date': '2024-01-02' if protected else '2020-02-03', 'exit_mark': 'open'}
    rows = []
    for side in v.SIDES:
        for i in range(10):
            rows.append({'source_security_id': f'source{i:02d}', 'signal_date': window['signal_date'],
                'side': side, 'path_variant': 'unstopped', 'target_family': 'executable_return',
                **{k: window[k] for k in ('outcome_month', 'entry_date', 'exit_date', 'exit_mark')},
                'label_available_at': window['exit_date'], 'prediction_eligible': True,
                'path_requested': True, 'path_available': not protected,
                'path_status': 'protected_next_open_endpoint_unavailable' if protected else 'available',
                'target_status': 'protected_next_open_endpoint_unavailable' if protected else 'available',
                'target_available': not protected, 'cash_return': None if protected else (10-i) / 100,
                'maximum_drawdown': None if protected else .02,
                'target_value': None if protected else (10-i)/100,
                'target_class': None if protected else i+1,
                'intrinio_security_id': f'native{i:02d}', 'identity_eligible': True})
    return rows, window


class IndependentVerifierTests(unittest.TestCase):
    def test_literal_formula_and_class_order(self):
        rows, window = invented()
        out, summary = v.expected_month(rows, window)
        self.assertEqual([r['target_class'] for r in out[:10]], list(range(1,11)))
        self.assertEqual([r['target_value'] for r in out[:2]], [5.0,4.5])
        self.assertEqual([r['labelled_members'] for r in summary], [10,10])

    def test_literal_negative_return_reversal_remains(self):
        rows, window = invented()
        for i, dd in ((0,.05),(1,.2)):
            rows[i].update(cash_return=-.05, target_value=-.05, maximum_drawdown=dd)
        out, summary = v.expected_month(rows, window)
        self.assertEqual(out[0]['target_value'], -1.)
        self.assertEqual(out[1]['target_value'], -.25)
        self.assertGreater(out[0]['target_class'], out[1]['target_class'])
        self.assertEqual(summary[0]['negative_return_above_floor_count'],2)

    def test_floor_zero_small_and_boundary(self):
        rows, window = invented()
        for i, dd in enumerate((0.,.005,.01)):
            rows[i].update(cash_return=.02,target_value=.02,maximum_drawdown=dd)
        out,_ = v.expected_month(rows,window)
        self.assertEqual([r['target_value'] for r in out[:3]], [2.,2.,2.])
        self.assertEqual([r['ratio_floor_hit'] for r in out[:3]], [True,True,False])
        self.assertEqual(len({r['target_class'] for r in out[:3]}),1)

    def test_all_equal_ties_are_not_arbitrarily_split(self):
        rows,window = invented()
        for row in rows:
            row.update(cash_return=0.,target_value=0.,maximum_drawdown=0.)
        out,summary = v.expected_month(rows,window)
        self.assertTrue(all(r['target_class']==6 for r in out))
        self.assertEqual(summary[0]['empty_classes'],[1,2,3,4,5,7,8,9,10])

    def test_distinct_losing_cash_returns_can_all_tie_at_minus_one(self):
        rows,window=invented()
        for i,row in enumerate(rows):
            loss=-(i%10+1)/100
            row.update(cash_return=loss,target_value=loss,maximum_drawdown=-loss)
        out,summary=v.expected_month(rows,window)
        self.assertTrue(all(r['target_value']==-1. and r['target_class']==6 for r in out))
        self.assertEqual(summary[0]['minus_one_ratio_count'],10)
        self.assertEqual(summary[0]['largest_exact_tie_count'],10)

    def test_dd_above_one_and_large_short_loss_retained(self):
        rows,window=invented()
        rows[10].update(cash_return=-2.,target_value=-2.,maximum_drawdown=2.2)
        out,summary=v.expected_month(rows,window)
        self.assertAlmostEqual(out[10]['target_value'],-2/2.2)
        self.assertEqual(summary[1]['drawdown_above_one_count'],1)

    def test_no_future_row_filter_for_missing_outcome(self):
        rows,window=invented()
        for row in (rows[0],rows[10]):
            row.update(path_available=False,path_status='source_path_unavailable:test',
                target_status='source_path_unavailable:test',target_available=False,
                cash_return=None,maximum_drawdown=None,target_value=None,target_class=None)
        out,summary=v.expected_month(rows,window)
        self.assertEqual(len(out),20)
        self.assertEqual(summary[0]['original_members'],10)
        self.assertEqual(summary[0]['valid_ratio_values'],9)
        self.assertTrue(all(r['target_class'] is None for r in out))

    def test_protected_next_open_is_metadata_only_unavailable(self):
        rows,window=invented('2023-12')
        out,summary=v.expected_month(rows,window)
        self.assertEqual(len(out),20)
        self.assertTrue(all(r['target_value'] is None and r['target_class'] is None for r in out))
        self.assertEqual(summary[0]['rank_status'],'scheduled_endpoint_not_observed')

    def test_protected_values_rejected(self):
        rows,window=invented('2023-12');rows[0]['cash_return']=.1
        with self.assertRaises(ValueError):v.expected_month(rows,window)

    def test_completed_available_route_or_eligibility_change_fails(self):
        rows,window=invented();out,_=v.expected_month(rows,window)
        for field,value in [('intrinio_security_id','wrong_native'),('prediction_eligible',False),('label_available_at','2020-02-04')]:
            altered=copy.deepcopy(out);altered[0][field]=value
            with self.assertRaises(ValueError):v.verify_month(rows,altered,window)

    def test_numerical_and_class_changes_fail(self):
        rows,window=invented();out,_=v.expected_month(rows,window)
        for field,value in [('target_value',99.),('target_class',10),('cash_return',.9),('source_cash_target_class',10)]:
            altered=copy.deepcopy(out);altered[0][field]=value
            with self.assertRaises(ValueError):v.verify_month(rows,altered,window)

    def test_dropped_duplicate_or_extra_cohort_row_fails(self):
        rows,window=invented();out,_=v.expected_month(rows,window)
        for altered in (out[:-1],out+[out[0]],out[1:]+[out[1]]):
            with self.assertRaises(ValueError):v.verify_month(rows,altered,window)

    def test_invalid_source_numbers_fail(self):
        for field,value in [('maximum_drawdown',-.1),('maximum_drawdown',float('nan')),('cash_return',float('inf')),('cash_return',True)]:
            rows,window=invented();rows[0][field]=value
            with self.assertRaises(ValueError):v.expected_month(rows,window)

    def test_foreign_family_or_side_and_completion_clock_fail(self):
        for field,value in [('target_family','utility'),('side','both'),('exit_date','2020-02-04')]:
            rows,window=invented();rows[0][field]=value
            with self.assertRaises(ValueError):v.expected_month(rows,window)

    def test_exact_midrank_is_input_order_independent(self):
        rows,window=invented();out,_=v.expected_month(rows,window)
        reversed_out,_=v.expected_month(list(reversed(rows)),window)
        self.assertEqual({tuple(r[k] for k in v.KEYS):r['target_class'] for r in out},
                         {tuple(r[k] for k in v.KEYS):r['target_class'] for r in reversed_out})


def fixture_aggregates(directory):
    """Production audit helpers are exercised only on invented ten-stock paths."""
    path=WORKER.parents[1]/'src'/'dmv2'/'return_dd_targets.py'
    worker_spec=importlib.util.spec_from_file_location('invented_ratio_worker',path)
    worker=importlib.util.module_from_spec(worker_spec);sys.modules[worker_spec.name]=worker
    worker_spec.loader.exec_module(worker)
    def first_weekday(month):
        date=dt.date.fromisoformat(month+'-01')
        while date.weekday()>4:date+=dt.timedelta(days=1)
        return date.isoformat()
    def last_weekday(month):
        date=dt.date.fromisoformat(v.shift(month,1)+'-01')-dt.timedelta(days=1)
        while date.weekday()>4:date-=dt.timedelta(days=1)
        return date.isoformat()
    all_rows=[];statistics=[];coverage=[];months=[];windows=[]
    for i in range(168):
        month=v.shift('2010-01',i);source,_=invented()
        window={'signal_date':last_weekday(v.shift(month,-1)),'outcome_month':month,
            'entry_date':first_weekday(month),'exit_date':first_weekday(v.shift(month,1)),
            'exit_mark':'open','label_available_at':first_weekday(v.shift(month,1))}
        for row in source:
            row.update(window)
            if month=='2023-12':
                row.update(path_available=False,target_available=False,path_status='protected_next_open_endpoint_unavailable',
                    target_status='protected_next_open_endpoint_unavailable',cash_return=None,maximum_drawdown=None,target_value=None,target_class=None)
        panel=worker.build_return_dd_targets(source,observed_through='2023-12-31')
        verified,diag=v.verify_month(source,panel.table.to_pylist(),window)
        v.same(panel.diagnostics.to_pylist(),diag)
        all_rows.extend(verified);coverage.extend(panel.diagnostics.to_pylist())
        statistics.extend(worker.monthly_class_means(panel));months.append({'outcome_month':month,'source_window':window});windows.append(window)
    class_changes=[]
    for month in [w['outcome_month'] for w in windows]:
        for side in v.SIDES:
            compared=[r for r in all_rows if r['outcome_month']==month and r['side']==side
                      and r['target_available'] and r['source_cash_target_class'] is not None]
            changed=sum(r['target_class']!=r['source_cash_target_class'] for r in compared)
            class_changes.append({'outcome_month':month,'side':side,'class_comparison_rows':len(compared),
                'class_changed_rows':changed,'class_unchanged_rows':len(compared)-changed})
    payloads=[];readiness=[]
    for window in windows[120:]:
        for side in v.SIDES:
            state=worker.payoffs_from_monthly_means(statistics,windows,signal_date=window['signal_date'],side=side,observed_through='2023-12-31')
            payloads.append({'signal_date':window['signal_date'],'side':side,'table':state.table.to_pylist(),
                'provenance':state.provenance,'fingerprint':state.fingerprint})
            available=sum(r['status']=='available' for r in state.monthly.to_pylist())
            readiness.append({k:state.provenance[k] for k in ('signal_date','side','first_outcome_month','last_outcome_month',
                'vector_available','required_calendar_months','required_month_class_cells')})
            readiness[-1].update(available_month_class_cells=available,missing_month_class_cells=1200-available)
    training=[]
    for year in range(2020,2024):
        for side in v.SIDES:
            cohort=[r for r in all_rows if f'{year-10}-01'<=r['outcome_month']<=f'{year-1}-12' and r['side']==side]
            cutoff=last_weekday(f'{year-1}-12')
            nfit=sum(r['target_available'] and r['prediction_eligible'] and r['label_available_at']<=cutoff for r in cohort)
            training.append({'year':year,'side':side,'train_start_month':f'{year-10}-01','train_end_month':f'{year-1}-12',
                'as_of':cutoff,'original_rows':len(cohort),'prediction_eligible_rows':len(cohort),
                'target_available_rows':len(cohort),'fit_rows':nfit,'incomplete_or_unavailable_rows':len(cohort)-nfit,
                'uncompleted_boundary_rows':sum(r['label_available_at']>cutoff for r in cohort),
                'prediction_eligibility_unchanged':True,'fit_performed':False})
    for name,data in [('coverage.json',coverage),('class_changes.json',class_changes),('monthly_class_means.json',statistics),('payoffs.json',payloads),
        ('payoff_readiness.json',readiness),('training_completion.json',training)]:
        (directory/name).write_text(json.dumps(data,allow_nan=False))
    return {'months':months},all_rows,coverage


class AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.directory=Path(cls.temp.name)
        cls.plan,cls.rows,cls.coverage=fixture_aggregates(cls.directory)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_all_invented_168_months_and_96_states(self):
        result=v.verify_aggregates(self.plan,self.directory,self.rows,self.coverage)
        self.assertEqual(result,{'monthly_diagnostics':336,'monthly_class_mean_cells':3360,
            'payoff_states':96,'training_completion_rows':8,'complete_payoff_vectors':94,'class_change_rows':336})
    def test_bogus_first_available_vector_is_rejected(self):
        path=self.directory/'payoff_readiness.json';original=path.read_text()
        data=json.loads(original);data[0]['vector_available']=True;path.write_text(json.dumps(data))
        try:
            with self.assertRaises(ValueError):v.verify_aggregates(self.plan,self.directory,self.rows,self.coverage)
        finally:path.write_text(original)
    def test_training_boundary_was_not_completed_at_annual_cutoff(self):
        path=self.directory/'training_completion.json';original=path.read_text()
        data=json.loads(original);data[0]['fit_rows']+=10;path.write_text(json.dumps(data))
        try:
            with self.assertRaises(ValueError):v.verify_aggregates(self.plan,self.directory,self.rows,self.coverage)
        finally:path.write_text(original)
    def test_cash_payoff_cannot_be_substituted_with_ratio_mean(self):
        path=self.directory/'payoffs.json';original=path.read_text()
        data=json.loads(original);data[2]['table'][0]['class_mean_cash_return']=data[2]['table'][0]['class_mean_ratio'];path.write_text(json.dumps(data))
        try:
            with self.assertRaises(ValueError):v.verify_aggregates(self.plan,self.directory,self.rows,self.coverage)
        finally:path.write_text(original)


if __name__=='__main__':unittest.main()
