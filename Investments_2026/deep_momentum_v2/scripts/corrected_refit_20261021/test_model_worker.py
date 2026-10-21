"""Structural and fabricated-data checks; no original financial/model reads."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
import model_worker as w


def winner(year=2020):
    depth, leaf = w.WINNERS[year]
    p = dict(max_depth=depth, min_samples_leaf=leaf, max_features='sqrt', n_estimators=500,
        n_jobs=4, random_state=1729, class_weight=None, bootstrap=True, warm_start=False, oob_score=False)
    return dict(status='COMPLETE', stage='outer', outer_year=year, explicit_parameters=p,
        fit_effective_parameters=p.copy(), saved_inference_effective_parameters={**p, 'n_jobs': 1})


def metadata(year=2020):
    rows=[]; windows={}
    for month in range(1,13):
        prior_year=year-1 if month==1 else year; prior_month=12 if month==1 else month-1
        signal=f'{prior_year}-{prior_month:02d}-28'; target=f'{year}-{month:02d}'
        windows[signal]=dict(outcome_month=target,outcome_start=target+'-01',outcome_end=target+'-28')
        rows.append(dict(source_security_id='A',signal_date=signal,**windows[signal],
            identity_eligible=True,requested=True,paper_prediction_eligible=True,
            prediction_eligible=True,target_available=month!=3))
    return rows,windows


class StructuralTests(unittest.TestCase):
    def test_no_numeric_or_estimator_import_at_module_top_level(self):
        tree=ast.parse(Path(w.__file__).read_text()); modules=[]
        for node in tree.body:
            if isinstance(node,ast.Import): modules.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom): modules.append(node.module)
        self.assertFalse(any(n.startswith(('numpy','pandas','pyarrow','sklearn','xgboost','dmv2')) for n in modules))

    def test_five_arms_twenty_fits_share_eight_bank_states(self):
        self.assertEqual(len(w.ARMS)*len(w.YEARS),20)
        self.assertEqual(len({bank for _,bank in w.ARMS.values()})*len(w.YEARS),8)
        self.assertEqual([a for a,(_,b) in w.ARMS.items() if b=='expanded'],['rf_extended','xgb_extended','rf_tuned_extended'])

    def test_exact_rolling_ten_year_bounds(self):
        self.assertEqual(w.year_window(2023),dict(train_start='2013-01-01',train_end='2022-12-31',prediction_start='2023-01-01',prediction_end='2023-12-31'))
        with self.assertRaises(ValueError): w.year_window(2024)

    def test_missing_labels_keep_prediction_rows(self):
        rows,windows=metadata()
        self.assertEqual(len(w.validate_metadata(2020,rows,rows,rows,rows,windows)),12)
        self.assertTrue(rows[2]['prediction_eligible']); self.assertFalse(rows[2]['target_available'])

    def test_original_denominator_not_filtered_to_observed_targets(self):
        rows,windows=metadata()
        with self.assertRaisesRegex(ValueError,'denominator'):
            w.validate_metadata(2020,rows,rows,rows,[r for r in rows if r['target_available']],windows)

    def test_future_or_overlapping_window_rejected(self):
        rows,windows=metadata();windows[rows[0]['signal_date']]['outcome_end']='2024-01-01'
        with self.assertRaisesRegex(ValueError,'clock'):
            w.validate_metadata(2020,rows,rows,rows,rows,windows)

    def test_changed_admission_rejected(self):
        rows,windows=metadata();target=copy.deepcopy(rows);target[0]['requested']=False
        with self.assertRaisesRegex(ValueError,'admission'):
            w.validate_metadata(2020,rows,rows,rows,target,windows)

    def test_future_label_does_not_make_a_prediction_eligible(self):
        rows,windows=metadata();target=copy.deepcopy(rows);target[0]['prediction_eligible']=False
        with self.assertRaisesRegex(ValueError,'population'):
            w.validate_metadata(2020,rows,rows,rows,target,windows)

    def test_duplicate_original_keys_rejected(self):
        rows,windows=metadata()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            w.validate_metadata(2020,rows+rows[:1],rows,rows,rows,windows)

    def test_fixed_tuned_winners_no_new_search(self):
        for year,expected in w.WINNERS.items():
            p=w.tuned_parameters(year,winner(year));self.assertEqual((p['max_depth'],p['min_samples_leaf']),expected)
            self.assertEqual(p['n_jobs'],4)

    def test_changed_winner_or_saved_inference_parameters_rejected(self):
        z=winner();z['explicit_parameters']['min_samples_leaf']=100
        with self.assertRaisesRegex(ValueError,'winner'):
            w.tuned_parameters(2020,z)
        z=winner();z['saved_inference_effective_parameters']['n_jobs']=4
        with self.assertRaisesRegex(ValueError,'inference'):
            w.tuned_parameters(2020,z)

    def test_all_terminal_months_reviewed_without_label_imputation(self):
        good=dict(status='PASS',events_reviewed=29,target_definition_unchanged=True,missing_terminal_targets_imputed=False)
        w.validate_terminal_guard(good)
        for key,value in [('events_reviewed',28),('missing_terminal_targets_imputed',True)]:
            with self.assertRaises(ValueError): w.validate_terminal_guard({**good,key:value})

    def test_immutable_save_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'x.json';w.save(p,{'first':True})
            with self.assertRaises(FileExistsError):w.save(p,{'first':False})
            self.assertTrue(w.read(p)['first'])

    def test_shared_reservations_exact_categories_and_plan(self):
        import control
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'plan.json';w.save(p,{'pure':True})
            plan={'projected_reads':dict(compressed_source=12,derived_matrix=34)}
            with patch.object(control,'require_reservation') as mock:
                w.require_reservations(control.ROOT/'run',p,plan,dict(compressed_source='c',derived_matrix='d'))
                self.assertEqual(mock.call_args_list[0].args,(control.ROOT,'c',12,w.pin(p),'compressed_source'))
                self.assertEqual(mock.call_args_list[1].args,(control.ROOT,'d',34,w.pin(p),'derived_matrix'))
            with self.assertRaisesRegex(ValueError,'exact reservation'):
                w.require_reservations(control.ROOT/'run',p,plan,dict(compressed_source='c'))

    def test_launch_command_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            protocol_path=Path(d)/'protocol.json';w.save(protocol_path,{'pure':True})
            action=Path(d)/'action.json';w.save(action,{'action':'synthetic'})
            launch=Path(d)/'launch.json';command=['/python',str(w.__file__),'execute','--root','synthetic',
                '--action-plan',str(action),'--action-plan-sha256',w.digest(action)]
            protocol={'python_path':'/python','limits':{'active_seconds_by_category':{'models':14400}}}
            z=dict(schema='dmv2-corrected-refit-phase-launch/v1',protocol_pin=w.pin(protocol_path),
                command=command,code_pins=[w.pin(w.__file__)],maximum_seconds=1200,time_category='models')
            w.save(launch,z)
            with patch.dict(w.os.environ,DMV2_GUARD_LAUNCH_PATH=str(launch),DMV2_GUARD_LAUNCH_SHA256=w.digest(launch)):
                w.require_launch(launch,None,action,None,protocol,protocol_path,command=command)
                with self.assertRaisesRegex(ValueError,'External resource'):
                    w.require_launch(launch,None,action,None,protocol,protocol_path,command=command+['--fit-more'])
            with self.assertRaisesRegex(ValueError,'guard environment'):
                w.require_launch(launch,None,action,None,protocol,protocol_path,command=command)

    def test_array_sha_uses_c_order_and_raw_file_pin_is_separate(self):
        import numpy as np
        a=np.asfortranarray(np.arange(12,dtype=np.float32).reshape(3,4))
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'x.npy';item=w.save_array(p,a)
            self.assertEqual(item['array_sha256'],w.array_sha(np.ascontiguousarray(a)))
            self.assertNotEqual('sha256:'+item['sha256'],item['array_sha256'])
            self.assertEqual(item['shape'],[3,4])

    def test_projection_counts_nested_original_metadata_leaves(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'original.parquet'
            pq.write_table(pa.table({'source_security_id':['A','B'],'indices':[['sp400'],['sp500','sp400']]}),path)
            with patch.object(pq.ParquetFile,'read',side_effect=AssertionError('Footer plan must not read values')):
                spec=w.projection(w.pin(path),['source_security_id','indices'])
            f=pq.ParquetFile(path).metadata
            self.assertEqual(spec['projected_compressed_bytes'],sum(f.row_group(g).column(c).total_compressed_size
                for g in range(f.num_row_groups) for c in range(f.row_group(g).num_columns)))

    def test_incomplete_fit_never_repeats(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'folds/2020/arms/rf_extended/FIT_ATTEMPT.json';w.save(p,{'stopped':True})
            with self.assertRaisesRegex(ValueError,'incomplete fit'):
                w.fit_one(d,{},None,None,2020,'rf_extended',1)

    def test_incomplete_prediction_never_repeats(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'folds/2020/arms/rf_extended/PREDICT_STARTED.json';w.save(p,{'stopped':True})
            with self.assertRaisesRegex(ValueError,'inference attempt'):
                w.predict_one(d,{},2020,'rf_extended')


class InventedPreparationTests(unittest.TestCase):
    def test_two_fresh_training_only_states_and_shared_expanded_artifacts(self):
        import numpy as np
        import pandas as pd
        from dmv2.features import TrainingSlice,registered_feature_specs
        from dmv2.paper_features import PAPER_FEATURE_NAMES
        names=[s['name'] for s in registered_feature_specs() if s.get('model_eligible',True)]
        def fabricated(spec,_):
            year=spec['year']; rows=[]
            for month in range(1,13):
                prev_year=year-1 if month==1 else year; prev_month=12 if month==1 else month-1
                for n in range(10):rows.append((f's{n}',f'{prev_year}-{prev_month:02d}-28',month,n))
            index=pd.MultiIndex.from_tuples([(sid,signal) for sid,signal,_,_ in rows],names=w.KEYS)
            values=np.arange(len(rows),dtype=float)[:,None]+np.arange(len(names),dtype=float)[None,:]+(1000000 if year==2020 else year)
            X=pd.DataFrame(values,index=index,columns=names);y=pd.Series([n+1 for _,_,_,n in rows],index=index,dtype=float)
            ledger=pd.DataFrame(dict(outcome_start=[f'{year}-{m:02d}-01' for _,_,m,_ in rows],outcome_end=[f'{year}-{m:02d}-28' for _,_,m,_ in rows],outcome_month=[f'{year}-{m:02d}' for _,_,m,_ in rows],paper_prediction_eligible=True,target_available=True),index=index)
            if year==2010: y.iloc[0]=np.nan;ledger.iloc[0,ledger.columns.get_loc('target_available')]=False
            return TrainingSlice(X,y,ledger,dict(train_start=f'{year}-01-01',train_end=f'{year}-12-31',label_observed_through=f'{year}-12-28'))
        source=dict(annuals={str(y):{'year':y} for y in range(2010,2021)},registered_specs=[],paper_names=list(PAPER_FEATURE_NAMES),predictor_names=names,
            folds={'2020':{'label_observed_through':'2019-12-28'}})
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d,patch.object(w,'load_annual',side_effect=fabricated):
            item=w.prepare(ROOT,d,source,2020);prepared=w.read(item['path'])
            self.assertEqual(prepared['preprocessor_fits'],2);self.assertEqual(prepared['train_fit_rows'],1199)
            self.assertEqual(prepared['test_prediction_rows'],120)
            self.assertEqual(prepared['schemas']['expanded']['matrix_width'],1430)
            state=w.read(prepared['artifacts']['preprocessors/expanded/manifest.json']['path'])['manifest']
            self.assertLess(max(state['medians'].values()),10000)
            self.assertEqual(state['fit_rows'],1199)
            self.assertEqual(state['train_end'],'2019-12-31')
            self.assertEqual(state['label_observed_through'],'2019-12-28')
            self.assertEqual(len(list((Path(d)/'folds/2020/preprocessors').glob('*/manifest.json'))),2)
            with self.assertRaisesRegex(ValueError,'preprocessing attempt'): w.prepare(ROOT,d,source,2020)


if __name__=='__main__':unittest.main()
