"""Synthetic verifier regressions: no actual source panels or models."""
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE
SCRIPT = STAGE / 'scripts/verify_rf_tuned_backtest_v2.py'
spec = importlib.util.spec_from_file_location('tuned_verifier', SCRIPT)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
v = audit.load_equations(ROOT)
SESSIONS, WINDOWS = v.independent_clock()

# Production code is permitted only as a generator of synthetic observed
# ledgers. The independent verifier supplies every expected accounting result.
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(ROOT/'src'))
import test_walkforward_accounting_v2 as fixture


def metadata(year=2020):
    windows = WINDOWS.loc[WINDOWS.outcome_month.str[:4].eq(str(year))]
    return pd.DataFrame([{'source_security_id': 'source_A', 'signal_date': row.signal_date,
                         'intrinio_security_id': 'native_A', 'prediction_eligible': True,
                         'paper_prediction_eligible': True, 'identity_eligible': True,
                         'target_available': True, 'outcome_month': row.outcome_month,
                         'outcome_start': row.outcome_start, 'outcome_end': row.outcome_end}
                         for row in windows.itertuples()])


def grid():
    records = [{'arm': arm, 'basket': basket, 'variant': variant, 'cost_bps': fee,
                'complete_proxy_path': True} for arm, basket, variant, fee in
               itertools.product((audit.ARM,), audit.BASKETS, audit.VARIANTS, audit.FEES)]
    comparison = pd.DataFrame(records)
    yearly = pd.DataFrame([{**row, 'year': year} for row in records for year in range(2020, 2024)])
    coverage = pd.DataFrame([{'arm': audit.ARM, 'basket': basket, 'signal_date': row.signal_date}
                            for basket in audit.BASKETS for row in WINDOWS.itertuples()])
    return comparison, yearly, coverage


def selection_fixture(long_count=11, short_count=11):
    signal, entry = '2019-12-31', '2020-01-02'
    records = []
    for side, count, cls in [('long', long_count, 1), ('short', short_count, 10)]:
        for index in range(count):
            score = .30 - index * .001
            probabilities = np.full(10, (1-score)/9); probabilities[cls-1] = score
            records.append({'source_security_id': f'{side}_{index:02d}', 'signal_date': signal,
                            'intrinio_security_id': f'{side}_native_{index:02d}',
                            'prediction_eligible': True, 'paper_prediction_eligible': True,
                            **dict(zip(audit.PROBABILITIES, probabilities))})
    forecasts = pd.DataFrame(records)
    n = min(10, long_count, short_count)
    chosen = forecasts.groupby(forecasts.source_security_id.str[:4], sort=False).head(n).copy() if n else forecasts.iloc[:0].copy()
    chosen['side'] = np.where(chosen.source_security_id.str.startswith('long'), 'long', 'short')
    chosen['execution_date'] = entry
    chosen['selection_score'] = [row.probability_1 if row.side == 'long' else row.probability_10 for row in chosen.itertuples()]
    chosen['intended_weight'] = np.where(chosen.side.eq('long'), .5/max(1,n), -.5/max(1,n))
    ledger = forecasts.copy(); ledger['execution_restricted'] = False; ledger['trade_eligible'] = True
    ledger['predicted_class'] = forecasts[audit.PROBABILITIES].to_numpy().argmax(axis=1)+1
    ledger['selected'] = ledger.source_security_id.isin(chosen.source_security_id)
    monthly = pd.DataFrame([{'signal_date': signal, 'long_candidates': long_count, 'short_candidates': short_count,
                             'long_selected': n, 'short_selected': n, 'selected_names_per_side': n,
                             'effective_per_name_abs_weight': None if n == 0 else .5/n}])
    windows = pd.DataFrame([{'signal_date': signal, 'entry_date': entry}])
    return forecasts, ledger, chosen, monthly, windows


class TunedVerifierTests(unittest.TestCase):
    def _encoding_pair(self,first,second,*,mutation=None):
        columns={'signal_date':pa.array(['2019-12-31','2020-01-31'],type=pa.string()),
                 'source_security_id':pa.array(['A','B'],type=pa.string()),
                 'side':pa.array(['long','short'],type=pa.string()),
                 'value':pa.array([1.,2.],type=pa.float64())}
        a=pa.table(columns);columns['side']=pa.array(['long','short'],type=pa.large_string())
        if mutation=='value':columns['value']=pa.array([1.,3.],type=pa.float64())
        if mutation=='null':columns['side']=pa.array(['long',None],type=pa.large_string())
        if mutation=='other_type':columns['value']=pa.array([1.,2.],type=pa.float32())
        if mutation=='other_string_type':columns['source_security_id']=pa.array(['A','B'],type=pa.large_string())
        b=pa.table(columns)
        if mutation=='drop':b=b.slice(0,1)
        if mutation=='roworder':b=b.take(pa.array([1,0]))
        if mutation=='columnorder':b=b.select(list(reversed(b.column_names)))
        first.parent.mkdir(parents=True,exist_ok=True);second.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(a,first);pq.write_table(b,second)

    def test_exact_side_string_width_reencoding_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for basket in audit.BASKETS:
                for name in ('original_ledger','selected','monthly','rebalances'):
                    rel=f'selections/{audit.ARM}/{basket}/{name}.parquet'
                    self._encoding_pair(root/'preflight'/rel,root/'backtest'/rel)
            budget=audit.ProjectionBudget(v)
            checks=audit.verify_preflight_semantics(v,budget,root)
            self.assertEqual(len(checks),8)
            self.assertTrue(all(x['exact_values_order_nulls_and_dtypes_equal'] for x in checks))

    def test_reencoding_cannot_hide_value_null_drop_or_reorder(self):
        for mutation in ('value','null','drop','roworder','columnorder','other_type','other_string_type'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                for basket in audit.BASKETS:
                    for name in ('original_ledger','selected','monthly','rebalances'):
                        rel=f'selections/{audit.ARM}/{basket}/{name}.parquet'
                        self._encoding_pair(root/'preflight'/rel,root/'backtest'/rel,mutation=mutation)
                with self.assertRaises((v.VerificationError,AssertionError)):
                    audit.verify_preflight_semantics(v,audit.ProjectionBudget(v),root)

    def test_reencoding_budget_charged_before_numeric_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for basket in audit.BASKETS:
                for name in ('original_ledger','selected','monthly','rebalances'):
                    rel=f'selections/{audit.ARM}/{basket}/{name}.parquet'
                    self._encoding_pair(root/'preflight'/rel,root/'backtest'/rel)
            with self.assertRaises(v.VerificationError):
                audit.verify_preflight_semantics(v,audit.ProjectionBudget(v,limit=0),root)
    def test_exact_grid(self):
        audit.verify_grid(v, *grid())

    def test_grid_rejects_missing_scenario(self):
        comparison, yearly, coverage = grid()
        with self.assertRaises(v.VerificationError): audit.verify_grid(v, comparison.iloc[:-1], yearly, coverage)

    def test_grid_rejects_duplicate_or_comparator(self):
        for mode in ('duplicate', 'comparator'):
            comparison, yearly, coverage = grid()
            if mode == 'duplicate': comparison.iloc[0] = comparison.iloc[1]
            else: comparison.loc[0, 'arm'] = 'rf_extended'
            with self.subTest(mode=mode), self.assertRaises(v.VerificationError):
                audit.verify_grid(v, comparison, yearly, coverage)

    def test_grid_rejects_missing_year_or_month(self):
        comparison, yearly, coverage = grid()
        for y, c in [(yearly.iloc[:-1], coverage), (yearly, coverage.iloc[:-1])]:
            with self.assertRaises(v.VerificationError): audit.verify_grid(v, comparison, y, c)

    def test_grid_requires_boolean_complete(self):
        comparison, yearly, coverage = grid(); comparison['complete_proxy_path'] = 1
        with self.assertRaises(v.VerificationError): audit.verify_grid(v, comparison, yearly, coverage)

    def test_metadata_full_year(self):
        self.assertEqual(audit.verify_forecast_metadata(v, metadata(), 2020, WINDOWS)['months'], 12)

    def test_metadata_duplicate_missing_month_and_native_collision(self):
        for kind in ('duplicate', 'month', 'native'):
            frame = metadata()
            if kind == 'duplicate': frame = pd.concat([frame, frame.iloc[:1]])
            elif kind == 'month': frame = frame.iloc[1:]
            else:
                duplicate = frame.iloc[:1].copy(); duplicate['source_security_id'] = 'source_B'
                frame = pd.concat([frame, duplicate])
            with self.subTest(kind=kind), self.assertRaises(v.VerificationError):
                audit.verify_forecast_metadata(v, frame, 2020, WINDOWS)

    def test_metadata_rejects_target_overlap_and_future_year(self):
        for field, value in [('outcome_start', '2019-12-31'), ('signal_date', '2024-01-02')]:
            frame = metadata(); frame.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                audit.verify_forecast_metadata(v, frame, 2020, WINDOWS)

    def test_metadata_strict_bool_and_preserved_paper_cohort(self):
        for field, value in [('prediction_eligible', 1), ('paper_prediction_eligible', False), ('intrinio_security_id', None)]:
            frame = metadata(); frame[field] = frame[field].astype(object); frame.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                audit.verify_forecast_metadata(v, frame, 2020, WINDOWS)

    def test_forecast_projection_never_reads_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'forecast.parquet'; frame = metadata()
            for field in audit.PROBABILITIES: frame[field] = .1
            frame['target_class'] = 7; frame['target_return'] = 999.; frame.to_parquet(path, index=False)
            reads = []
            def read(path, columns):
                reads.append(columns); self.assertNotIn('target_class', columns); self.assertNotIn('target_return', columns)
                return pq.ParquetFile(path).read(columns=columns).to_pandas()
            result, _ = audit.read_forecast(v, path, 2020, WINDOWS, read)
            self.assertEqual(len(result), 12); self.assertEqual(len(reads), 2)

    def test_forecast_future_metadata_blocks_probability_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'forecast.parquet'; frame = metadata(); frame.loc[0, 'signal_date'] = '2024-01-02'
            for field in audit.PROBABILITIES: frame[field] = .1
            frame.to_parquet(path, index=False); reads = []
            def read(path, columns):
                reads.append(columns)
                if 'probability_1' in columns: self.fail('Probability decoded before metadata failure')
                return pq.ParquetFile(path).read(columns=columns).to_pandas()
            with self.assertRaises(v.VerificationError): audit.read_forecast(v, path, 2020, WINDOWS, read)
            self.assertEqual(len(reads), 1)

    def test_forecast_invalid_simplex(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'forecast.parquet'; frame = metadata()
            for field in audit.PROBABILITIES: frame[field] = .2
            frame.to_parquet(path, index=False)
            with self.assertRaises(v.VerificationError):
                audit.read_forecast(v, path, 2020, WINDOWS, lambda p, c: pq.read_table(p, columns=c).to_pandas())

    def test_training_cutoffs(self):
        prediction = {'status':'COMPLETE', 'outer_year':2020, 'candidate_id':'A', 'estimator_fits':0,
                      'preprocessing_fits':0, 'prediction_provenance':{'class_ids':list(range(1,11))}}
        prepare = {'status':'COMPLETE', 'outer_year':2020, 'stage':'outer', 'train_provenance':{
            'train_start':'2010-01-01', 'train_end':'2019-12-31', 'label_observed_through':'2019-12-31',
            'future_financial_rows_converted':False}}
        fit = {'status':'COMPLETE', 'outer_year':2020, 'stage':'outer', 'candidate_id':'A', 'probability_class_ids':list(range(1,11))}
        winner = {'candidate_id':'A'}
        audit.verify_training_manifest(v, prediction, prepare, fit, winner, 2020, WINDOWS)
        for field, value in [('train_start','2009-01-01'),('train_end','2020-12-31'),('label_observed_through','2020-01-02')]:
            changed = deepcopy(prepare); changed['train_provenance'][field] = value
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                audit.verify_training_manifest(v, prediction, changed, fit, winner, 2020, WINDOWS)

    def test_balanced_top_ten_and_nine(self):
        for longs, shorts in [(11,11),(9,37),(0,37)]:
            args = selection_fixture(longs, shorts)
            report = v.verify_selection(*args, basket='class_1_10')
            self.assertEqual(report['selected_slots'], min(10,longs,shorts)*2)

    def test_lowest_class_mode_tie_and_stable_source_rank_tie(self):
        f,l,s,m,w=selection_fixture()
        tied=f.source_security_id.eq('long_00')
        f.loc[tied,audit.PROBABILITIES]=[.3,.3,*([.05]*8)]
        l.loc[tied,audit.PROBABILITIES]=[.3,.3,*([.05]*8)]
        # Boundary long_09/long_10 share the same own-class probability;
        # lexicographically smaller stable source long_09 must remain chosen.
        target=f.source_security_id.eq('long_10');score=float(f.loc[f.source_security_id.eq('long_09'),'probability_1'].iloc[0])
        vector=[score,*([(1-score)/9]*9)]
        f.loc[target,audit.PROBABILITIES]=vector;l.loc[target,audit.PROBABILITIES]=vector
        v.verify_selection(f,l,s,m,w,basket='class_1_10')
        changed=s.copy();changed.loc[changed.source_security_id.eq('long_09'),'source_security_id']='long_10'
        with self.assertRaises(v.VerificationError):v.verify_selection(f,l,changed,m,w,basket='class_1_10')

    def test_selector_rejects_wrong_key_score_weight(self):
        for field in ('source_security_id','selection_score','intended_weight'):
            f,l,s,m,w = selection_fixture()
            s.loc[s.index[0],field] = 'wrong' if field=='source_security_id' else 99.
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                v.verify_selection(f,l,s,m,w,basket='class_1_10')

    def test_projection_budget_reserved_before_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'table.parquet'; pd.DataFrame({'date':['2020-01-02'],'value':[1.]}).to_parquet(path,index=False)
            budget=audit.ProjectionBudget(v,prior=10,limit=10+path.stat().st_size-1)
            with patch.object(audit.pq,'ParquetFile',side_effect=AssertionError('Unreserved decode')):
                with self.assertRaises(v.VerificationError): budget.read(path,['value'])

    def test_key_guard_blocks_protected_numeric_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'table.parquet';pd.DataFrame({'date':['2024-01-02'],'value':[1.]}).to_parquet(path,index=False)
            budget=audit.ProjectionBudget(v); original=budget.read;reads=[]
            def guarded(path,columns=None):
                reads.append(columns)
                if columns is None or 'value' in columns:self.fail('Protected numeric decode')
                return original(path,columns)
            budget.read=guarded
            with self.assertRaises(v.VerificationError): audit.checked_table(v,budget,path,kind='synthetic')
            self.assertEqual(reads,[['date']])

    def test_read_reservation_cumulative_and_no_auto_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'read_usage.json';p.write_text(json.dumps({'schema':'synthetic','reservations':[{'phase':'backtest','projected_compressed_bytes':12}], 'total_projected_compressed_bytes':12,'limit_bytes':512*1024**2}))
            self.assertEqual(audit.reserve_verification(v,root,20),12)
            self.assertEqual(json.loads(p.read_text())['total_projected_compressed_bytes'],32)
            with self.assertRaises(v.VerificationError):audit.reserve_verification(v,root,20)

    def test_read_reservation_rejects_false_prior_and_overflow(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'read_usage.json'
            for total, amount in [(11,20),(12,512*1024**2)]:
                p.write_text(json.dumps({'reservations':[{'phase':'backtest','projected_compressed_bytes':12}], 'total_projected_compressed_bytes':total,'limit_bytes':512*1024**2}))
                with self.assertRaises(v.VerificationError):audit.reserve_verification(v,root,amount)

    def test_no_fit_or_production_import(self):
        text=SCRIPT.read_text()
        for token in ('from dmv2','import sklearn','import xgboost','import joblib','.fit('):self.assertNotIn(token,text)

    def test_independent_cash_math_and_nav_tamper(self):
        s,p,r=fixture.frames(change=False)
        result=fixture.execute(s,p,r,policy=fixture.DailyAccountingPolicy(cost_bps=10,asset_stop_fraction=.1))
        report=v.verify_scenario(result.tables,p,s,r,fixture.DAYS,cost_bps=10,stop_fraction=.1)
        self.assertTrue(report['continuous_cash_units_reconciled'])
        result.daily.loc[0,'nav']+=.001
        with self.assertRaises(v.VerificationError):
            v.verify_scenario(result.tables,p,s,r,fixture.DAYS,cost_bps=10,stop_fraction=.1)

    def test_independent_dropped_holding_and_fee_tamper(self):
        s,p,r=fixture.frames(change=False)
        original=fixture.execute(s,p,r,policy=fixture.DailyAccountingPolicy(cost_bps=10))
        for mutation in ('holding','fee'):
            result=deepcopy(original)
            if mutation=='holding':result.holdings=result.holdings.iloc[1:]
            else:result.trades.loc[0,'cost']+=.001
            with self.subTest(mutation=mutation),self.assertRaises(v.VerificationError):
                v.verify_scenario(result.tables,p,s,r,fixture.DAYS,cost_bps=10)

    def test_independent_failed_exposure_is_preserved(self):
        s,p,r=fixture.frames()
        result=fixture.execute(s,p,r,policy=fixture.DailyAccountingPolicy(cost_bps=10))
        report=v.verify_failed_scenario(result.tables,p,s,fixture.DAYS,cost_bps=10,rebalances=r)
        self.assertFalse(report['economic_performance_available'])
        changed=deepcopy(result);changed.holdings.loc[changed.holdings.date.ge(pd.Timestamp('2020-02-03')),'units']=0.
        with self.assertRaises(v.VerificationError):
            v.verify_failed_scenario(changed.tables,p,s,fixture.DAYS,cost_bps=10,rebalances=r)

    def test_semantic_pin_normalization_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);left={'path':'x','size_bytes':5,'sha256':'a'}
            right={**left,'path':str(root/'x')}
            self.assertTrue(audit.same_pin(v,root,left,right))
            self.assertFalse(audit.same_pin(v,root,left,{**right,'sha256':'b'}))


if __name__=='__main__':unittest.main()
