"""Fake-ID regressions only: strict failure and one explicit execution waiver."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import pandas as pd

import dmv2
DMV2=Path(__file__).resolve().parents[1]/'src/dmv2'
if str(DMV2) not in dmv2.__path__:dmv2.__path__.insert(0,str(DMV2))
from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.walkforward_accounting import run_walkforward_accounting as old_run
from dmv2.walkforward_accounting_v2 import run_walkforward_accounting_v2 as run, HANDOVER_POLICY, _prepare
from dmv2.walkforward_backtest_v2 import _attempt_scope, _identity_registry
from dmv2.catalog import canonical_sha256
from dmv2.universe import DataContractError

DAYS=['2020-01-02','2020-01-31','2020-02-03','2020-02-04']
MONTHS=[('2019-12-31','2020-01-02'),('2020-01-31','2020-02-03')]

def frames(change=True,opposite=False):
    rows=[]
    for i,(signal,execution) in enumerate(MONTHS):
        for source,native,side in [('new' if i and change else 'old','A','short' if i and opposite else 'long'),
                ('other','B','long' if i and opposite else 'short')]:
            rows.append(dict(signal_date=signal,execution_date=execution,source_security_id=source,
                intrinio_security_id=native,side=side))
    selected=pd.DataFrame(rows)
    prices=pd.DataFrame([dict(trading_date=date,intrinio_security_id=native,
        open=100.,close=100.,adj_open=100.,adj_close=100.,adj_high=100.,adj_low=100.,
        frequency='daily',intraperiod=False,price_quality_status='accepted')
        for date in DAYS for native in ['A','B','C']])
    schedule=pd.DataFrame(MONTHS,columns=['signal_date','execution_date'])
    return selected,prices,schedule

def waiver():
    return dict(handover_id='fake-exact-handover',old_source_security_id='old',new_source_security_id='new',
        intrinio_security_id='A',effective_date='2020-02-03',signal_date='2020-01-31',
        held_signal_date='2019-12-31',evidence_reference='fake-primary-evidence',verified=False,
        provisionally_admitted=True,execution_evidence_reviewed=True,admission_policy=HANDOVER_POLICY)

def execute(selected=None,prices=None,schedule=None,**kwargs):
    s,p,r=frames()
    return run(s if selected is None else selected,p if prices is None else prices,DAYS,
        rebalances=r if schedule is None else schedule,terminal_date=DAYS[-1],**kwargs)


def approval_files(root,path,manifest):
    import hashlib
    def pin(p):return {'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    receipt=root/'package_receipt.json'
    receipt.write_text(json.dumps({'schema_version':1,'files':[pin(path),pin(root/'primary.json')]}))
    names=['effective_date','held_signal_date','intrinio_security_id','new_source_security_id','old_source_security_id','signal_date']
    decision={'schema_version':1,'status':'APPROVED_FOR_SEPARATE_ACCOUNTING_V2_ONLY',
        'registry_sha256':pin(path)['sha256'],'evidence_package_receipt_sha256':pin(receipt)['sha256'],
        'allowed_transition':{name:manifest['handovers'][0][name] for name in names},
        'identity_status':{'execution_evidence_reviewed':True,'provisionally_admitted':True,
            'source_identity_promoted':False,'source_identity_verified':False,
            'source_review_flags_retained':True,'unified_ids_remain_distinct':True}}
    (root/'decision_record.json').write_text(json.dumps(decision))
    return decision


class AccountingV2Tests(unittest.TestCase):
    def test_strict_collision_withholds_before_quotes_retaining_positions_orders(self):
        s,p,r=frames(); p.loc[p.trading_date.ge('2020-02-03'),'adj_open']=None
        result=execute(s,p,r)
        self.assertEqual(result.status,'failed_closed')
        self.assertEqual(result.failures.reason.tolist(),['selected_provider_held_source_collision'])
        f=result.failures.iloc[0]
        self.assertEqual((f.source_security_id,f.intrinio_security_id,f.role),('new','A','selected'))
        self.assertIsNone(f.price_field)
        blocked=result.daily.date.ge(pd.Timestamp('2020-02-03'))
        self.assertTrue(result.daily.loc[blocked,'nav'].isna().all())
        self.assertIsNone(result.summary['final_nav']);self.assertIsNone(result.summary['total_return'])
        held=result.holdings[result.holdings.date.ge(pd.Timestamp('2020-02-03'))]
        self.assertEqual(set(held.source_security_id),{'old','other'})
        self.assertTrue(held.valuation_price.isna().all())
        self.assertTrue(held.loc[held.source_security_id.eq('old'),'units'].eq(.005).all())
        self.assertFalse(result.trades.date.ge(pd.Timestamp('2020-02-03')).any())
        self.assertTrue(result.orders.loc[result.orders.date.eq(pd.Timestamp('2020-02-03')),'status'].eq('pending_validation').all())
        self.assertTrue(result.identity_handovers.empty)

    def test_v1_rejected_condition_is_preserved_as_explicit_failure(self):
        s,p,r=frames()
        with self.assertRaisesRegex(DataContractError,'Selected provider collides'):
            old_run(s,p,DAYS,rebalances=r,terminal_date=DAYS[-1])
        self.assertEqual(execute(s,p,r).status,'failed_closed')

    def test_exact_long_long_waiver_keeps_lots_and_nets_physical_orders(self):
        result=execute(identity_handovers=[waiver()],policy=DailyAccountingPolicy(cost_bps=10))
        self.assertEqual(result.status,'complete_proxy')
        self.assertEqual(len(result.identity_handovers),1)
        h=result.identity_handovers.iloc[0]
        self.assertEqual((h.old_source_security_id,h.new_source_security_id,h.application_status),('old','new','applied_at_monthly_open'))
        self.assertFalse(h.source_identity_verified);self.assertFalse(h.source_identity_promoted)
        trades=result.trades[result.trades.date.eq(pd.Timestamp('2020-02-03'))&result.trades.intrinio_security_id.eq('A')]
        self.assertEqual(set(trades.source_security_id),{'old','new'})
        self.assertEqual(trades.loc[trades.source_security_id.eq('old'),'units_after'].iloc[0],0.)
        market=result.market_trades[result.market_trades.date.eq(pd.Timestamp('2020-02-03'))&result.market_trades.intrinio_security_id.eq('A')]
        self.assertEqual(len(market),1)
        self.assertAlmostEqual(market.iloc[0].absolute_notional,abs(trades.signed_notional.sum()))
        self.assertAlmostEqual(market.iloc[0].cost,market.iloc[0].absolute_notional*.001)
        self.assertEqual(result.summary['identity_handover_count'],1)

    def test_exact_long_short_waiver_uses_real_reversal_and_new_stop_reference(self):
        s,p,r=frames(opposite=True)
        p.loc[p.trading_date.ge('2020-02-03'),['open','close','adj_open','adj_close','adj_high','adj_low']]=105.
        result=execute(s,p,r,identity_handovers=[waiver()],policy=DailyAccountingPolicy(asset_stop_fraction=.1,cost_bps=10))
        self.assertEqual(result.status,'complete_proxy')
        trades=result.trades[result.trades.date.eq(pd.Timestamp('2020-02-03'))&result.trades.intrinio_security_id.eq('A')]
        self.assertTrue(trades.units_delta.lt(0).all())
        market=result.market_trades[result.market_trades.date.eq(pd.Timestamp('2020-02-03'))&result.market_trades.intrinio_security_id.eq('A')]
        self.assertAlmostEqual(market.iloc[0].absolute_notional,-trades.signed_notional.sum())
        held=result.holdings.query("date=='2020-02-03' and source_security_id=='new'")
        np.testing.assert_allclose(held.entry_reference,105.)
        np.testing.assert_allclose(held.stop_price,115.5)
        self.assertLess(result.identity_handovers.iloc[0].units_after,0.)

    def test_wrong_date_direction_provider_or_held_episode_never_bypasses(self):
        for key,value in [('effective_date','2020-01-02'),('old_source_security_id','new'),
                ('new_source_security_id','wrong'),('intrinio_security_id','C'),
                ('held_signal_date','2019-11-29')]:
            row=waiver();row[key]=value
            with self.subTest(key=key):
                try:result=execute(identity_handovers=[row])
                except DataContractError:continue
                self.assertEqual(result.status,'failed_closed');self.assertTrue(result.identity_handovers.empty)

    def test_nonprovisional_or_unreviewed_or_duplicate_registry_rejected(self):
        for key,value in [('verified',True),('provisionally_admitted',False),('execution_evidence_reviewed',False),('admission_policy','blanket-alias')]:
            row=waiver();row[key]=value
            with self.subTest(key=key),self.assertRaises(DataContractError):execute(identity_handovers=[row])
        with self.assertRaises(DataContractError):execute(identity_handovers=[waiver(),waiver()])

    def test_old_source_in_targets_and_new_provider_nonunique_are_structural_errors(self):
        s,p,r=frames();extra=s.iloc[-2].copy();extra.source_security_id='old'
        s=pd.concat([s,extra.to_frame().T],ignore_index=True)
        with self.assertRaisesRegex(DataContractError,'provider identity collision'):
            execute(s,p,r,identity_handovers=[waiver()])

    def test_global_source_route_changes_still_raise(self):
        s,p,r=frames(change=False);s.loc[s.execution_date.eq('2020-02-03')&s.source_security_id.eq('old'),'intrinio_security_id']='C'
        with self.assertRaisesRegex(DataContractError,'Provider route changes'):
            execute(s,p,r)

    def test_held_route_runtime_branch_withholds_even_with_exact_other_waiver(self):
        # A direct lower-level regression of the defensive runtime branch.
        # Production _prepare rejects this globally; that behavior is tested above.
        s,p,r=frames(change=False);prepared=_prepare(s,p,DAYS,r,DAYS[-1]);changed=prepared[0].copy()
        changed.loc[changed.execution_date.eq(pd.Timestamp('2020-02-03'))&changed.source_security_id.eq('old'),'intrinio_security_id']='C'
        with patch('dmv2.walkforward_accounting_v2._prepare',return_value=(changed,*prepared[1:])):
            result=execute(s,p,r)
        self.assertEqual(result.failures.reason.tolist(),['held_identity_route_changed'])
        self.assertFalse(result.trades.date.ge(pd.Timestamp('2020-02-03')).any())

    def test_existing_action_converted_successor_exception_unchanged(self):
        s,p,r=frames();s.loc[s.source_security_id.eq('new'),'intrinio_security_id']='C'
        action=dict(event_id='fake-action',predecessor_id='A',successor_id='C',stock_ratio=1.,
            cash_per_raw_share=0.,effective_date='2020-01-31',verified=True,evidence_reference='fake-primary-action')
        result=execute(s,p,r,events=[action])
        self.assertEqual(result.status,'complete_proxy');self.assertTrue(result.identity_handovers.empty)
        self.assertEqual(len(result.action_events),1)
        original=old_run(s,p,DAYS,rebalances=r,terminal_date=DAYS[-1],events=[action])
        for name,table in original.tables.items():pd.testing.assert_frame_equal(table,result.tables[name],check_exact=True)

    def test_noncollision_all_tables_numerically_identical_to_v1(self):
        s,p,r=frames(change=False)
        for stop in [None,.1]:
            for cost in [0,5,10,25]:
                policy=DailyAccountingPolicy(cost_bps=cost,asset_stop_fraction=stop)
                old=old_run(s,p,DAYS,rebalances=r,terminal_date=DAYS[-1],policy=policy)
                new=execute(s,p,r,policy=policy)
                for name,table in old.tables.items():pd.testing.assert_frame_equal(table,new.tables[name],check_exact=True)
                self.assertTrue(new.identity_handovers.empty)


class RunnerV2GuardTests(unittest.TestCase):
    def failed(self,root):
        run=root/'outputs/walkforward/fake';prior=run/'backtest';prior.mkdir(parents=True)
        phase={'parent_identity_sha256':'fake','models_fitted':False,'preprocessing_fitted':False}
        (prior/'build_identity.json').write_text(json.dumps(phase))
        (prior/'audit.json').write_text(json.dumps({'status':'STARTED','phase':'walkforward_continuous_backtest_v1',
            'build_identity_sha256':'fake','phase_identity_sha256':canonical_sha256(phase)}))
        (prior/'failure.json').write_text(json.dumps({'exception':'DataContractError','message':'Selected provider collides with a different held source identity'}))
        return run,prior

    def test_explicit_new_output_and_prior_attempt_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,prior=self.failed(root)
            for output,old in [(None,prior),(run/'new',None),(prior,prior),(prior/'new',prior),(run,prior)]:
                with self.subTest(output=output),self.assertRaises(DataContractError):
                    _attempt_scope(root,run,'fake',output_directory=output,failed_attempt_directory=old)
            output,context,pins=_attempt_scope(root,run,'fake',output_directory=run/'new',failed_attempt_directory=prior)
            self.assertEqual(context['file_count'],3);self.assertEqual(len(pins),3);self.assertFalse(output.exists())

    def test_prior_failure_cannot_be_relabelled_as_complete_or_other_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,prior=self.failed(root)
            (prior/'failure.json').write_text(json.dumps({'exception':'DataContractError','message':'unrelated numerical failure'}))
            with self.assertRaises(DataContractError):_attempt_scope(root,run,'fake',output_directory=run/'new',failed_attempt_directory=prior)

    def test_registry_default_empty_and_explicit_evidence_raw_pins_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);s,_,r=frames();schedule=dict(zip(pd.to_datetime(r.execution_date),pd.to_datetime(r.signal_date)))
            self.assertEqual(_identity_registry(root,None,'fake',pd.DatetimeIndex(DAYS),schedule,[]),(None,None,[],None))
            path=root/'registry.json';evidence=root/'primary.json';evidence.write_text('{}')
            import hashlib
            manifest=dict(schema_version=1,registry_id='fake-registry',status='PROVISIONAL_EXECUTION_ONLY',
                root_build_identity_sha256='fake',source_identity_promoted=False,handovers=[waiver()],
                evidence_pins=[dict(path='primary.json',base='registry_directory',sha256=hashlib.sha256(b'{}').hexdigest())])
            path.write_text(json.dumps(manifest));pins=[]
            approval_files(root,path,manifest)
            body,pin,records,approval=_identity_registry(root,path,'fake',pd.DatetimeIndex(DAYS),schedule,pins)
            self.assertEqual(body,manifest);self.assertEqual(len(records),1);self.assertEqual(len(pins),4)
            self.assertEqual(set(approval),{'decision_pin','package_receipt_pin'})
            self.assertEqual(pin['path'],'registry.json')
            evidence.write_text('{"tampered":true}')
            with self.assertRaises(DataContractError):_identity_registry(root,path,'fake',pd.DatetimeIndex(DAYS),schedule,[])



    def test_wrong_approval_registry_hash_or_transition_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();path=root/'registry.json';(root/'primary.json').write_text('{}')
            import hashlib
            manifest=dict(schema_version=1,registry_id='fake',status='PROVISIONAL_EXECUTION_ONLY',
                root_build_identity_sha256='fake',source_identity_promoted=False,handovers=[waiver()],
                evidence_pins=[dict(path='primary.json',base='registry_directory',sha256=hashlib.sha256(b'{}').hexdigest())])
            path.write_text(json.dumps(manifest));decision=approval_files(root,path,manifest)
            _,_,r=frames();schedule=dict(zip(pd.to_datetime(r.execution_date),pd.to_datetime(r.signal_date)))
            for change in ['registry_hash','transition','identity_promotion','package_hash','status']:
                bad=deepcopy(decision)
                if change=='registry_hash':bad['registry_sha256']='0'*64
                elif change=='transition':bad['allowed_transition']['new_source_security_id']='blanket-alias'
                elif change=='identity_promotion':bad['identity_status']['source_identity_promoted']=True
                elif change=='package_hash':bad['evidence_package_receipt_sha256']='0'*64
                else:bad['status']='PENDING'
                (root/'decision_record.json').write_text(json.dumps(bad))
                with self.subTest(change=change),self.assertRaises(DataContractError):
                    _identity_registry(root,path,'fake',pd.DatetimeIndex(DAYS),schedule,[])


    def test_registry_accepts_real_clock_string_sessions_after_metadata_normalization(self):
        from dmv2.walkforward_clock import walkforward_clock
        import hashlib
        sessions,windows,_=walkforward_clock()
        self.assertIsInstance(sessions[0],str)
        schedule=dict(zip(pd.to_datetime(windows.execution_date),pd.to_datetime(windows.signal_date)))
        record=waiver();record.update(effective_date='2021-04-01',signal_date='2021-03-31',held_signal_date='2021-02-26')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();path=root/'registry.json';(root/'primary.json').write_text('{}')
            manifest=dict(schema_version=1,registry_id='fake-real-clock',status='PROVISIONAL_EXECUTION_ONLY',
                root_build_identity_sha256='fake',source_identity_promoted=False,handovers=[record],
                evidence_pins=[dict(path='primary.json',base='registry_directory',sha256=hashlib.sha256(b'{}').hexdigest())])
            path.write_text(json.dumps(manifest));approval_files(root,path,manifest)
            body,pin,records,approval=_identity_registry(root,path,'fake',sessions,schedule,[])
            self.assertEqual(records[0]['effective_date'],pd.Timestamp('2021-04-01'))
            self.assertEqual(body,manifest);self.assertEqual(pin['path'],'registry.json')


class RunnerV2MockIntegrationTests(unittest.TestCase):
    def test_accounting_only_runner_pins_prior_and_writes_all64_handover_ledgers(self):
        """Mock financial engines; exercise source wiring, phase and report plumbing."""
        from contextlib import ExitStack
        from types import SimpleNamespace
        import pyarrow as pa
        import pyarrow.parquet as pq
        import dmv2.walkforward_backtest_v2 as worker
        from dmv2.walkforward_clock import walkforward_clock
        from dmv2.backtest_loading import ARMS
        sessions,windows,clock=walkforward_clock()
        cfg={'backtest':{'baskets':['class_1_10','class_2_9'],
            'variants':{'unstopped':None,'asset_stop_10pct':.1},'cost_bps':[0,5,10,25]},
            'execution':{'backtest_maximum_wall_seconds':900,
                'backtest_maximum_peak_rss_bytes':2*1024**3,'maximum_output_bytes':12*1024**3}}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();run=root/'outputs/walkforward/fake';run.mkdir(parents=True)
            (root/'configs').mkdir();(root/'configs/walkforward_2020_2023.json').write_text(json.dumps(cfg))
            identity={'configuration':cfg};identity_pin=canonical_sha256(identity)
            (run/'build_identity.json').write_text(json.dumps(identity))
            prior=run/'backtest';prior.mkdir()
            oldphase={'parent_identity_sha256':identity_pin,'models_fitted':False,'preprocessing_fitted':False}
            (prior/'build_identity.json').write_text(json.dumps(oldphase))
            (prior/'audit.json').write_text(json.dumps({'status':'STARTED','phase':worker.PHASE,
                'build_identity_sha256':identity_pin,'phase_identity_sha256':canonical_sha256(oldphase)}))
            (prior/'failure.json').write_text(json.dumps({'exception':'DataContractError',
                'message':'Selected provider collides with a different held source identity'}))
            prediction=root/'forecast.parquet'
            pq.write_table(pa.table({f'probability_{k}':[.1] for k in range(1,11)}),prediction)
            price=root/'synthetic_price.parquet'
            values={col:['synthetic'] for col in worker.ACTION_PRICE_COLUMNS}
            values['trading_date']=[sessions[0]];values['intrinio_security_id']=['A']
            pq.write_table(pa.table(values),price)
            metadata=pd.DataFrame({'trading_date':[sessions[0]],'intrinio_security_id':['A']})
            selected=pd.DataFrame([dict(signal_date=pd.Timestamp(row.signal_date),execution_date=pd.Timestamp(row.execution_date),
                source_security_id=source,intrinio_security_id=native,side=side)
                for row in windows.itertuples() for source,native,side in [('a','A','long'),('b','B','short')]])
            monthly=pd.DataFrame({'signal_date':pd.to_datetime(windows.signal_date),'selected_names_per_side':[1]*48})
            rebalances=windows[['signal_date','execution_date']].apply(pd.to_datetime)
            basket=SimpleNamespace(selections=selected,monthly=monthly,rebalances=rebalances,ledger=selected)
            fake_result=SimpleNamespace(status='complete_proxy',tables={'identity_handovers':pd.DataFrame(columns=['handover_id'])},
                failures=pd.DataFrame(),action_events=pd.DataFrame(),orders=pd.DataFrame())
            def summary(result,**kwargs):
                row={k:kwargs[k] for k in ['arm','basket','variant','cost_bps']}
                row.update(complete_proxy_path=True,expected_sessions=1006)
                return row,pd.DataFrame({'month':pd.period_range('2020-01','2023-12',freq='M').astype(str)})
            def yearlies(result,**kwargs):return pd.DataFrame([{**kwargs,'year':year} for year in range(2020,2024)])
            overrides={'_configuration':lambda value:value,'_historical_registry':lambda *args:[],
                '_prediction_plan':lambda *args:({arm:(prediction,[]) for arm in ARMS},0),
                '_price_plan':lambda *args:({2020:price},metadata,{'original_requested_native_ids':['A'],'projected_compressed_bytes':0}),
                '_supplement_plan':lambda *args:([],metadata.iloc[:0],{'projected_compressed_bytes':0}),
                'select_walkforward_basket':lambda *args,**kwargs:basket,
                'validate_walkforward_basket':lambda *args,**kwargs:{'status':'PASS'},
                'preflight_selected_horizons':lambda *args,**kwargs:{'summary':{'ready_for_accounting':True}},
                'summarize_daily':summary,'yearly_breakdown':yearlies}
            with ExitStack() as stack:
                for name,value in overrides.items():stack.enter_context(patch.object(worker,name,value))
                accounting=stack.enter_context(patch.object(worker,'run_walkforward_accounting',return_value=fake_result))
                stack.enter_context(patch('builtins.print'))
                output=worker.run_walkforward_backtest_v2(root,run,cfg,identity_pin,output_directory=run/'backtest_v2',failed_attempt_directory=prior)
            self.assertEqual(accounting.call_count,64)
            self.assertTrue(all(call.kwargs['identity_handovers']==[] for call in accounting.call_args_list))
            audit=json.loads((output/'audit.json').read_text());phase=json.loads((output/'build_identity.json').read_text())
            self.assertEqual(audit['scenarios'],64);self.assertEqual(audit['implementation_revision'],'identity_route_failure_v2')
            self.assertEqual(phase['prior_failed_attempt']['file_count'],3)
            self.assertIsNone(phase['identity_handover_registry']);self.assertIsNone(phase['identity_handover_registry_pin'])
            self.assertTrue({'walkforward_accounting.py','walkforward_backtest.py','walkforward_accounting_v2.py','walkforward_backtest_v2.py'}<=set(phase['workers']))
            self.assertEqual(sum(path.endswith('/identity_handovers.parquet') for path in audit['artifacts']),64)
            self.assertEqual(len(pq.read_table(output/'comparison.parquet')),64)
            self.assertEqual(len(pq.read_table(output/'yearly.parquet')),256)
            self.assertEqual(len(pq.read_table(output/'selection_coverage.parquet')),384)
            self.assertEqual(json.loads((prior/'audit.json').read_text())['status'],'STARTED')


if __name__=='__main__':unittest.main()
