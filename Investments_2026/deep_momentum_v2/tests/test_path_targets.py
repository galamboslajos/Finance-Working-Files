"""Synthetic clocks and economic wealth only; no research inputs or replay."""
from pathlib import Path
import copy
import datetime as dt
import importlib.util
import math
import sys
import unittest

import pyarrow as pa
import pyarrow.compute as pc

path = Path(__file__).parents[1] / 'src/dmv2/path_targets.py'
spec = importlib.util.spec_from_file_location('path_targets_under_test', path)
p = importlib.util.module_from_spec(spec); sys.modules[spec.name] = p; spec.loader.exec_module(p)


def fixture(n=11, month='2020-01'):
    year, m = map(int, month.split('-'))
    previous = p._shift(month, -1); following = p._shift(month, 1)
    sessions = [previous+'-30', month+'-02', month+'-03', following+'-03']
    if month == '2020-01': sessions[0] = '2019-12-31'
    window = {'signal_date': sessions[0], 'outcome_month': month, 'entry_date': sessions[1],
              'exit_date': sessions[-1], 'exit_mark': 'open'}
    cohort = [{'source_security_id': f'S{i:02}', 'signal_date': sessions[0],
        'prediction_eligible': True, 'identity_review_required': i == 1,
        'original_native': 'native'+str(i)} for i in range(n)]
    episodes, marks = [], []
    for i, row in enumerate(cohort):
        for side in p.SIDES:
            for variant in p.VARIANTS:
                key = {k: row[k] for k in p.KEYS}; key.update(side=side, path_variant=variant)
                episodes.append({**key, 'requested': True, 'status': 'complete', 'reason': ''})
                change = (.01*i if side == 'long' else -.005*i)
                final = 1 + change if variant == 'unstopped' else 1 - .003*i
                for date, mark, wealth in [(sessions[1], 'open', 1.), (sessions[1], 'close', 1.02),
                        (sessions[2], 'open', .9), (sessions[2], 'close', 1.01), (sessions[-1], 'open', final)]:
                    marks.append({**key, 'date': date, 'mark': mark, 'wealth': wealth})
    return cohort, episodes, marks, [window], sessions


def build(data=None, **kwargs):
    c,e,m,w,s = data or fixture()
    return p.build_path_targets(c,e,m,windows=w,sessions=s,
        observed_through=kwargs.pop('observed_through', w[-1]['exit_date']), **kwargs)


def one(panel, source='S10', family='return_minus_drawdown', side='long', variant='unstopped'):
    return next(r for r in panel.table.to_pylist() if (r['source_security_id'],r['target_family'],r['side'],r['path_variant'])
        == (source,family,side,variant))


def payoff_fixture():
    # 120 complete calendar months; fixed manual classes isolate payoff arithmetic.
    records, windows = [], []
    for i in range(120):
        month = p._shift('2010-01',i); next_month=p._shift(month,1)
        windows.append({'outcome_month':month,'label_available_at':next_month+'-02'})
        for c in p.CLASSES:
            for member in range(1 + (i % 2)):
                records.append({'source_security_id':f'{c}/{member}', 'signal_date':p._shift(month,-1)+'-28',
                    'outcome_month':month, 'entry_date':month+'-02', 'exit_date':next_month+'-02',
                    'exit_mark':'open','label_available_at':next_month+'-02','side':'long','path_variant':'unstopped',
                    'target_family':'return_minus_drawdown','prediction_eligible':True,'path_requested':True,
                    'path_available':True,'target_available':True,'target_status':'available','target_class':c,
                    'cash_return':c/100+i/10000,'maximum_drawdown':.2,'utility':c/100+i/10000-.2,
                    'target_value':c/100+i/10000-.2})
    table=pa.Table.from_pylist(records)
    prov={'policy_id':p.POLICY,'endpoint_policy':'next_open','risk_penalty':1.,'observed_through':'2023-12-31',
        'windows':windows,'metadata_sha256':p._table_hash(table.select(list(p.META))),
        'month_fingerprints':{w['outcome_month']:p._table_hash(table.filter(pc.equal(table['outcome_month'],w['outcome_month']))) for w in windows}}
    return p.PathTargetPanel(table,pa.table({}),prov,p._hash(prov))


class PathTargets(unittest.TestCase):
    def test_manual_cash_drawdown_utility_and_independent_directions(self):
        panel=build(); row=one(panel)
        self.assertAlmostEqual(row['cash_return'],.1)
        self.assertAlmostEqual(row['maximum_drawdown'],(1.02-.9)/1.02)
        self.assertAlmostEqual(row['utility'],.1-(1.02-.9)/1.02)
        self.assertAlmostEqual(one(panel,side='short')['cash_return'],-.05)
        self.assertNotEqual(one(panel,side='short')['utility'],-row['utility'])
        self.assertNotEqual(one(panel,variant='asset_stop_10pct')['cash_return'],row['cash_return'])

    def test_all_original_keys_flags_and_ineligible_rows_retained(self):
        data=fixture();data[0][0]['prediction_eligible']=False; panel=build(data)
        self.assertEqual(panel.table.num_rows,len(data[0])*8)
        for old in data[0]:
            rows=[r for r in panel.table.to_pylist() if r['source_security_id']==old['source_security_id']]
            self.assertEqual(len(rows),8)
            for row in rows:
                for k,v in old.items(): self.assertEqual(row[k],v)
        self.assertFalse(one(panel,'S00')['target_available'])

    def test_pandas_nullable_original_metadata_retained_without_nan_identity(self):
        import pandas as pd
        data=list(fixture())
        data[0][0].update(prediction_eligible=False,original_native=None)
        data[0]=pd.DataFrame(data[0])
        panel=build(data)
        self.assertEqual(panel.table.num_rows,len(data[0])*8)
        original_rows=[row for row in panel.table.to_pylist() if row['source_security_id']=='S00']
        self.assertEqual(len(original_rows),8)
        for row in original_rows:
            self.assertIsNone(row['original_native'])
            self.assertIs(row['prediction_eligible'],False)
            self.assertIs(row['identity_review_required'],False)
            self.assertFalse(row['target_available'])

    def test_future_and_early_stopped_wealth_never_used_before_common_exit(self):
        data=fixture()
        for row in data[2]: row['wealth']=object() # must not be touched
        panel=build(data,observed_through='2020-01-31')
        self.assertFalse(any(panel.table['target_available'].to_pylist()))
        self.assertTrue(all(r['path_status']=='scheduled_endpoint_not_observed' for r in panel.table.to_pylist()))

    def test_requested_unavailable_keeps_unknown_but_not_prediction_filter(self):
        data=fixture()
        for e in data[1]:
            if e['source_security_id']=='S00':e.update(status='unavailable',reason='missing_quote')
        panel=build(data)
        self.assertTrue(one(panel,'S00')['prediction_eligible'])
        self.assertIsNone(one(panel,'S00')['cash_return'])
        self.assertTrue(one(panel,'S10')['target_available']) # ten others still rank

    def test_false_unknown_or_missing_request_withholds_whole_group(self):
        for request in (False,None,'absent'):
            data=fixture()
            if request=='absent': data[1].pop(0)
            else:data[1][0]['requested']=request
            panel=build(data)
            self.assertFalse(one(panel,'S10',variant='unstopped')['target_available'])
            self.assertEqual(one(panel,'S10')['target_status'],'non_exhaustive_original_path_requests')
            self.assertTrue(one(panel,'S10',side='short')['target_available'])

    def test_exact_ties_midrank_and_permutation_stability(self):
        data=fixture(10)
        for row in data[2]:
            if row['date']==data[3][0]['exit_date']: row['wealth']=1.
        panel=build(data)
        self.assertEqual(set(panel.table['target_class'].to_pylist()),{6})
        data[0].reverse();data[1].reverse();data[2].reverse(); shuffled=build(data)
        self.assertEqual(sorted(panel.table.to_pylist(),key=lambda r:tuple(str(r[k]) for k in (*p.EPISODE_KEYS,'target_family'))),
            sorted(shuffled.table.to_pylist(),key=lambda r:tuple(str(r[k]) for k in (*p.EPISODE_KEYS,'target_family'))))

    def test_missing_interior_mark_not_ignored_and_no_stale_fill(self):
        data=fixture();data[2].pop(1);panel=build(data)
        self.assertFalse(one(panel,'S00')['path_available'])
        self.assertEqual(one(panel,'S00')['path_status'],'incomplete_required_path_marks')

    def test_duplicate_unknown_keys_bad_calendar_and_extra_marks_fail_closed(self):
        for case in ('duplicate','foreign','calendar','extra_mark','initial'):
            data=fixture()
            if case=='duplicate':data[2].append(data[2][0].copy())
            elif case=='foreign':data[1][0]['source_security_id']='foreign'
            elif case=='calendar':data[3][0]['exit_date']='2020-02-04'
            elif case=='extra_mark':data[2][0]['date']='2020-01-04'
            else:data[2][0]['wealth']=.5
            with self.subTest(case=case),self.assertRaises(ValueError):build(data)

    def test_negative_short_wealth_is_not_censored(self):
        data=fixture()
        for row in data[2]:
            if row['source_security_id']=='S00' and row['side']=='short' and row['date']=='2020-02-03':row['wealth']=-.5
        row=one(build(data),'S00',side='short')
        self.assertEqual(row['cash_return'],-1.5)
        self.assertGreater(row['maximum_drawdown'],1)
        self.assertTrue(row['nonpositive_wealth_observed'])
        self.assertTrue(row['target_available'])

    def test_december_protected_endpoint_cannot_be_replaced_by_terminal_close(self):
        data=fixture(month='2023-12'); data[4][0]='2023-11-30';data[3][0]['signal_date']='2023-11-30'
        for row in data[0]+data[1]+data[2]:row['signal_date']='2023-11-30'
        # Remove all protected marks: next-open label remains unavailable.
        data[2][:]=[r for r in data[2] if r['date']<'2024-01-01']
        panel=build(data,observed_through='2023-12-31')
        self.assertIsNone(one(panel)['cash_return'])
        self.assertEqual(one(panel)['path_status'],'protected_next_open_endpoint_unavailable')
        data[3][0].update(exit_date='2023-12-03',exit_mark='close')
        with self.assertRaisesRegex(ValueError,'cannot be replaced'):build(data,observed_through='2023-12-31')
        terminal=build(data,observed_through='2023-12-31',endpoint_policy='terminal_close')
        self.assertEqual(terminal.provenance['endpoint_policy'],'terminal_close')

    def test_protected_mark_metadata_rejected_before_poison_value(self):
        data=fixture();data[2][0].update(date='2024-01-02',wealth=object())
        with self.assertRaisesRegex(ValueError,'Protected'):build(data)

    def test_training_excludes_actual_next_open_unfinished_label_values(self):
        panel=build()
        before=p.completed_training_slice(panel,train_start_month='2020-01',train_end_month='2020-01',
            as_of='2020-01-31',target_family='return_minus_drawdown',side='long',path_variant='unstopped')
        self.assertEqual(before.ledger.num_rows,11);self.assertEqual(before.fit_table.num_rows,0)
        after=p.completed_training_slice(panel,train_start_month='2020-01',train_end_month='2020-01',
            as_of='2020-02-03',target_family='return_minus_drawdown',side='long',path_variant='unstopped')
        self.assertEqual(after.fit_table.num_rows,11)

    def test_training_rejects_changed_completed_classes_or_values_but_not_unfinished_values(self):
        original=build()
        for field in ('target_class','target_value','cash_return'):
            records=original.table.to_pylist();records[0][field]=3 if field=='target_class' else 999.
            bad=p.PathTargetPanel(pa.Table.from_pylist(records,schema=original.table.schema),original.diagnostics,
                original.provenance,original.fingerprint)
            args=dict(train_start_month='2020-01',train_end_month='2020-01',target_family='executable_return',side='long',path_variant='unstopped')
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'training month values changed'):
                p.completed_training_slice(bad,as_of='2020-02-03',**args)
            self.assertEqual(p.completed_training_slice(bad,as_of='2020-01-31',**args).fit_table.num_rows,0)


class CausalPayoffs(unittest.TestCase):
    def test_exact_120_equal_calendar_means_and_explicit_score_semantics(self):
        panel=payoff_fixture()
        args=dict(signal_date='2020-01-31',target_family='return_minus_drawdown',side='long',path_variant='unstopped')
        cash=p.build_path_payoffs(panel,payoff_field='cash_return',**args)
        utility=p.build_path_payoffs(panel,payoff_field='utility',**args)
        self.assertTrue(cash.provenance['vector_available']);self.assertEqual(cash.monthly.num_rows,1200)
        self.assertEqual(cash.provenance['last_outcome_month'],'2019-12')
        expected=.01+sum(range(120))/120/10000
        self.assertAlmostEqual(cash.table.to_pylist()[0]['payoff'],expected)
        self.assertAlmostEqual(utility.table.to_pylist()[0]['payoff'],expected-.2)
        score=p.score_path_probabilities([1.]+[0.]*9,cash,class_ids=list(p.CLASSES),signal_date='2020-01-31',as_of='2020-01-31')
        self.assertAlmostEqual(score['score'],expected)
        self.assertEqual(score['score_semantics'],'expected_executable_cash_return')
        self.assertIn('not_original_RET',p.score_path_probabilities([.1]*10,utility,class_ids=list(p.CLASSES),signal_date='2020-01-31',as_of='2020-01-31')['score_semantics'])

    def test_missing_month_or_class_cell_withholds_whole_vector_without_shortening(self):
        base=payoff_fixture()
        for kind in ('month','class'):
            rows=base.table.to_pylist()
            rows=[r for r in rows if not(r['outcome_month']=='2015-01' and (kind=='month' or r['target_class']==1))]
            table=pa.Table.from_pylist(rows);prov=copy.deepcopy(base.provenance)
            prov['metadata_sha256']=p._table_hash(table.select(list(p.META)))
            prov['month_fingerprints']['2015-01']=p._table_hash(table.filter(pc.equal(table['outcome_month'],'2015-01')))
            panel=p.PathTargetPanel(table,pa.table({}),prov,p._hash(prov))
            state=p.build_path_payoffs(panel,signal_date='2020-01-31',target_family='return_minus_drawdown',side='long',path_variant='unstopped',payoff_field='cash_return')
            self.assertFalse(state.provenance['vector_available'])
            self.assertTrue(all(r['payoff'] is None for r in state.table.to_pylist()))
            self.assertEqual(state.provenance['first_outcome_month'],'2010-01')

    def test_future_values_ignored_but_completed_value_tampering_rejected(self):
        panel=payoff_fixture();rows=panel.table.to_pylist();rows[0]['cash_return']=999.
        bad=p.PathTargetPanel(pa.Table.from_pylist(rows),panel.diagnostics,panel.provenance,panel.fingerprint)
        with self.assertRaisesRegex(ValueError,'values changed'):
            p.build_path_payoffs(bad,signal_date='2020-01-31',target_family='return_minus_drawdown',side='long',path_variant='unstopped',payoff_field='cash_return')
        # Latest outcome is not yet completed on its scheduled first-open date's prior day.
        state=p.build_path_payoffs(panel,signal_date='2020-01-01',target_family='return_minus_drawdown',side='long',path_variant='unstopped',payoff_field='cash_return')
        self.assertFalse(state.provenance['vector_available'])
        self.assertEqual(state.provenance['last_outcome_month'],'2019-12')

    def test_probability_alignment_no_renormalization_or_absent_class_padding(self):
        state=p.build_path_payoffs(payoff_fixture(),signal_date='2020-01-31',target_family='return_minus_drawdown',side='long',path_variant='unstopped',payoff_field='cash_return')
        for probs,ids in [([.1]*9,list(range(1,10))),([1.]*10,list(p.CLASSES)),([True]+[0.]*9,list(p.CLASSES))]:
            with self.assertRaises(ValueError):p.score_path_probabilities(probs,state,class_ids=ids,signal_date='2020-01-31',as_of='2020-01-31')
        a=p.score_path_probabilities([1.]+[0.]*9,state,class_ids=list(p.CLASSES),signal_date='2020-01-31',as_of='2020-01-31')
        b=p.score_path_probabilities([0.]*9+[1.],state,class_ids=list(reversed(p.CLASSES)),signal_date='2020-01-31',as_of='2020-01-31')
        self.assertEqual(a,b)

    def test_future_payoff_state_cannot_be_attached_to_an_earlier_prediction(self):
        state=p.build_path_payoffs(payoff_fixture(),signal_date='2020-01-31',target_family='return_minus_drawdown',side='long',path_variant='unstopped',payoff_field='cash_return')
        for signal,cutoff in [('2019-12-31','2019-12-31'),('2020-01-31','2019-12-31'),('2020-02-28','2020-02-28')]:
            with self.subTest(signal=signal,cutoff=cutoff),self.assertRaisesRegex(ValueError,'exact causal'):
                p.score_path_probabilities([.1]*10,state,class_ids=list(p.CLASSES),signal_date=signal,as_of=cutoff)


if __name__ == '__main__':
    unittest.main()
