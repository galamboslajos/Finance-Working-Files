"""Synthetic exact-waiver/failure audit regressions. No actual dataset reads."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

B=Path(__file__).resolve().parents[1]
SCRIPT=B/'scripts/verify_walkforward_backtest_v2.py'
if not SCRIPT.exists():SCRIPT=B/'verify_walkforward_backtest_v2.py'
spec=importlib.util.spec_from_file_location('independent_v2',SCRIPT)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
import test_walkforward_accounting_v2 as producer

def pin(root,path):return {'path':str(path.relative_to(root)),'size_bytes':path.stat().st_size,'sha256':v.digest(path)}
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return path
def exact():
    result=producer.waiver();result['evidence_reference']='https://example.com/dated-primary-evidence';return result
def audit(result,s,p,r,*,stop=None,hands=()):
    return v.verify_scenario(result.tables,p,s,r,producer.DAYS,cost_bps=10,stop_fraction=stop,identity_handovers=hands)
def registry_fixture(root,*,record=None,primary_url='https://www.sec.gov/Archives/edgar/data/synthetic/dated-filing.htm'):
    directory=root/'data/derived/synthetic_handover';record=deepcopy(record) if record else exact();evidence=[]
    record.setdefault('evidence_reference','')
    if record['evidence_reference']=='https://example.com/dated-primary-evidence':
        record['evidence_reference']='primary_findings.json; local_identity_evidence.json; all8_selection_transitions.json; stopped_case_identity_status.json'
    names=('primary_findings.json','local_identity_evidence.json','all8_selection_transitions.json','stopped_case_identity_status.json','local_metadata_source_pins.json')
    for index,name in enumerate(names):
        body={'findings':[{'url':primary_url,'issuer':'SEC synthetic primary filing'}]} if name=='primary_findings.json' else {'synthetic':True,'index':index}
        path=save(directory/name,body)
        evidence.append({'path':path.name,'sha256':v.digest(path),'base':'registry_directory'})
    registry={'schema_version':1,'registry_id':'synthetic-one-record','status':'PROVISIONAL_EXECUTION_ONLY',
        'root_build_identity_sha256':'sha256:synthetic-parent','source_identity_promoted':False,
        'evidence_pins':evidence,'handovers':[record]}
    raw=save(directory/'identity_handover_registry.json',registry)
    payloads=[*(directory/name for name in names),raw]
    receipt=save(directory/'package_receipt.json',{'schema_version':1,'files':[
        {'path':p.name,'bytes':p.stat().st_size,'sha256':v.digest(p)} for p in payloads]})
    decision=save(directory/'decision_record.json',{'status':'APPROVED_FOR_SEPARATE_ACCOUNTING_V2_ONLY',
        'registry_sha256':v.digest(raw),'evidence_package_receipt_sha256':v.digest(receipt),
        'allowed_transition':{name:record[name] for name in ('effective_date','held_signal_date','intrinio_security_id','new_source_security_id','old_source_security_id','signal_date')},
        'identity_status':{'execution_evidence_reviewed':True,'provisionally_admitted':True,'source_identity_promoted':False,
            'source_identity_verified':False,'source_review_flags_retained':True,'unified_ids_remain_distinct':True}})
    phase={'parent_identity_sha256':'sha256:synthetic-parent','identity_handover_registry':registry,
        'identity_handover_registry_pin':pin(root,raw),'identity_handover_approval':{
            'decision_pin':pin(root,decision),'package_receipt_pin':pin(root,receipt)},
        'source_pins':[pin(root,p) for p in [*payloads,receipt,decision]]}
    return phase,directory
def repin_decision(root,phase,path):
    phase['identity_handover_approval']['decision_pin']=pin(root,path)
    phase['source_pins']=[pin(root,path) if x['path']==str(path.relative_to(root)) else x for x in phase['source_pins']]

class IndependentIdentityV2Tests(unittest.TestCase):
    def test_exact_long_long_and_long_short_cash_units_costs_stop_reset(self):
        for opposite in (False,True):
            for stop in (None,.1):
                with self.subTest(opposite=opposite,stop=stop):
                    s,p,r=producer.frames(opposite=opposite)
                    p.loc[p.trading_date.ge('2020-02-03'),['open','close','adj_open','adj_close','adj_high','adj_low']]=105.
                    result=producer.execute(s,p,r,identity_handovers=[exact()],
                        policy=producer.DailyAccountingPolicy(cost_bps=10,asset_stop_fraction=stop))
                    report=audit(result,s,p,r,stop=stop,hands=[exact()])
                    self.assertTrue(report['continuous_cash_units_reconciled'])

    def test_missing_extra_reversed_wrongdate_provider_and_unit_records_rejected(self):
        s,p,r=producer.frames();result=producer.execute(s,p,r,identity_handovers=[exact()],policy=producer.DailyAccountingPolicy(cost_bps=10))
        for mutation in ('missing','extra','reversed','date','provider','units','promotion'):
            with self.subTest(mutation=mutation):
                changed=deepcopy(result)
                if mutation=='missing':changed.identity_handovers=changed.identity_handovers.iloc[0:0]
                elif mutation=='extra':changed.identity_handovers=pd.concat([changed.identity_handovers]*2)
                elif mutation=='reversed':changed.identity_handovers.loc[0,['old_source_security_id','new_source_security_id']]=['new','old']
                elif mutation=='date':changed.identity_handovers.loc[0,'date']=pd.Timestamp('2020-02-04')
                elif mutation=='provider':changed.identity_handovers.loc[0,'intrinio_security_id']='C'
                elif mutation=='units':changed.identity_handovers.loc[0,'units_after']+=.01
                else:changed.identity_handovers.loc[0,'source_identity_promoted']=True
                with self.assertRaises(v.VerificationError):audit(changed,s,p,r,hands=[exact()])

    def test_no_registry_or_wrong_scope_cannot_certify_a_complete_alias(self):
        s,p,r=producer.frames();result=producer.execute(s,p,r,identity_handovers=[exact()],policy=producer.DailyAccountingPolicy(cost_bps=10))
        for field,value in [('effective_date','2020-02-04'),('old_source_security_id','wrong'),('intrinio_security_id','C'),('verified',True),('admission_policy','blanket-alias')]:
            hand=exact();hand[field]=value
            with self.subTest(field=field),self.assertRaises(v.VerificationError):audit(result,s,p,r,hands=[hand])
        with self.assertRaises(v.VerificationError):audit(result,s,p,r)

    def test_strict_collision_proved_without_any_failure_date_quote_read(self):
        s,p,r=producer.frames();result=producer.execute(s,p,r,policy=producer.DailyAccountingPolicy(cost_bps=10))
        class BeforeFailureOnly(dict):
            def get(self,key,*args):
                if key[0]>='2020-02-03':raise AssertionError('Identity decision consulted a quote')
                return super().get(key,*args)
            def __getitem__(self,key):
                if key[0]>='2020-02-03':raise AssertionError('Identity decision consulted a quote')
                return super().__getitem__(key)
        q=BeforeFailureOnly({(row['trading_date'],row['intrinio_security_id']):row for row in p.to_dict('records')})
        report=v.verify_failed_scenario(result.tables,p,s,producer.DAYS,cost_bps=10,rebalances=r,prepared_quotes=q)
        self.assertFalse(report['economic_performance_available']);self.assertIsNotNone(report['identity_failure_proof'])
        self.assertEqual(report['valued_prefix_marks_independently_replayed'],4)
        changed=deepcopy(result);changed.orders.loc[changed.orders.date.eq(pd.Timestamp('2020-02-03')),'new_adjusted_units']=1.
        with self.assertRaises(v.VerificationError):v.verify_failed_scenario(changed.tables,p,s,producer.DAYS,cost_bps=10,rebalances=r)

    def test_later_failed_path_still_replays_prior_handover_full_cash_math(self):
        s,p,r=producer.frames(opposite=True)
        p=p.loc[~(p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('A'))]
        result=producer.execute(s,p,r,identity_handovers=[exact()],policy=producer.DailyAccountingPolicy(cost_bps=10,asset_stop_fraction=.1))
        report=v.verify_failed_scenario(result.tables,p,s,producer.DAYS,cost_bps=10,stop_fraction=.1,rebalances=r,identity_handovers=[exact()])
        self.assertFalse(report['economic_performance_available']);self.assertEqual(report['valued_prefix_marks_independently_replayed'],6)
        changed=deepcopy(result);changed.identity_handovers.loc[0,'units_after']+=.01
        with self.assertRaises(v.VerificationError):v.verify_failed_scenario(changed.tables,p,s,producer.DAYS,cost_bps=10,stop_fraction=.1,rebalances=r,identity_handovers=[exact()])

    def test_executed_handover_at_after_cost_failure_is_verified_without_returns(self):
        s,p,r=producer.frames()
        p.loc[p.trading_date.ge('2020-02-03')&p.intrinio_security_id.eq('B'),['open','close','adj_open','adj_close','adj_high','adj_low']]=299.7
        result=producer.execute(s,p,r,identity_handovers=[exact()],policy=producer.DailyAccountingPolicy(cost_bps=10))
        self.assertEqual(result.failures.reason.tolist(),['nonpositive_or_nonfinite_nav_after_cost'])
        self.assertEqual(len(result.identity_handovers),1)
        report=v.verify_failed_scenario(result.tables,p,s,producer.DAYS,cost_bps=10,rebalances=r,identity_handovers=[exact()])
        self.assertFalse(report['economic_performance_available'])
        changed=deepcopy(result);changed.identity_handovers=changed.identity_handovers.iloc[0:0]
        with self.assertRaises(v.VerificationError):v.verify_failed_scenario(changed.tables,p,s,producer.DAYS,cost_bps=10,rebalances=r,identity_handovers=[exact()])

    def test_nonmonthly_gap_stop_after_cost_failure_retains_real_surviving_lot(self):
        s,p,r=producer.frames(change=False)
        p.loc[p.trading_date.ge('2020-01-31')&p.intrinio_security_id.eq('B'),['open','close','adj_open','adj_close','adj_high','adj_low']]=299.6
        result=producer.execute(s,p,r,policy=producer.DailyAccountingPolicy(cost_bps=10,asset_stop_fraction=.1))
        self.assertEqual(result.failures.reason.tolist(),['nonpositive_or_nonfinite_nav_after_cost'])
        self.assertEqual(result.failures.iloc[0]['mark'],'open')
        report=v.verify_failed_scenario(result.tables,p,s,producer.DAYS,cost_bps=10,stop_fraction=.1,rebalances=r)
        self.assertEqual(report['retained_unvalued_lots'],1)
        for mutation in ('fill','allocated_fee','absolute','phase','surviving_reference','surviving_stop'):
            changed=deepcopy(result);day=pd.Timestamp('2020-01-31');failed=changed.trades.date.eq(day)
            if mutation=='fill':changed.trades.loc[failed,'price']+=1.
            elif mutation=='allocated_fee':changed.trades.loc[failed,'cost']+=.01
            elif mutation=='absolute':changed.trades.loc[failed,'absolute_notional']+=.01
            elif mutation=='phase':changed.trades.loc[failed,'phase']='rebalance'
            elif mutation=='surviving_reference':changed.holdings.loc[changed.holdings.date.ge(day),'entry_reference']+=1.
            else:changed.holdings.loc[changed.holdings.date.ge(day),'stop_price']+=1.
            with self.subTest(mutation=mutation),self.assertRaises(v.VerificationError):
                v.verify_failed_scenario(changed.tables,p,s,producer.DAYS,cost_bps=10,stop_fraction=.1,rebalances=r)

    def test_intraday_stop_and_terminal_liquidation_after_cost_failure(self):
        for kind in ('intraday','terminal'):
            s,p,r=producer.frames(change=False)
            if kind=='intraday':
                fee,stop=9000,.1
                p.loc[p.trading_date.eq('2020-01-31')&p.intrinio_security_id.eq('B'),'adj_high']=111.
            else:
                fee,stop=10,None
                p.loc[p.trading_date.eq(producer.DAYS[-1])&p.intrinio_security_id.eq('B'),['open','close','adj_open','adj_close','adj_high','adj_low']]=299.7
            result=producer.execute(s,p,r,policy=producer.DailyAccountingPolicy(cost_bps=fee,asset_stop_fraction=stop))
            self.assertEqual(result.failures.reason.tolist(),['nonpositive_or_nonfinite_nav_after_cost'])
            self.assertEqual(result.failures.iloc[0]['mark'],'close' if kind=='intraday' else 'terminal_close')
            report=v.verify_failed_scenario(result.tables,p,s,producer.DAYS,cost_bps=fee,stop_fraction=stop,rebalances=r)
            self.assertFalse(report['economic_performance_available'])
            for mutation in ('price','fee','phase','stop_missing'):
                changed=deepcopy(result);failed=changed.trades.date.eq(result.failures.iloc[0]['date'])&changed.trades.mark.eq(result.failures.iloc[0]['mark'])
                if mutation=='price':changed.trades.loc[failed,'price']+=.01
                elif mutation=='fee':changed.trades.loc[failed,'cost']+=.01
                elif mutation=='phase':changed.trades.loc[failed,'phase']='rebalance'
                elif kind=='intraday':changed.stops=changed.stops.iloc[0:0]
                else:continue
                with self.subTest(kind=kind,mutation=mutation),self.assertRaises(v.VerificationError):
                    v.verify_failed_scenario(changed.tables,p,s,producer.DAYS,cost_bps=fee,stop_fraction=stop,rebalances=r)

    def test_monthly_after_cost_reference_stop_and_allocated_fee_tampering_rejected(self):
        s,p,r=producer.frames()
        p.loc[p.trading_date.ge('2020-02-03')&p.intrinio_security_id.eq('B'),['open','close','adj_open','adj_close','adj_high','adj_low']]=299.7
        result=producer.execute(s,p,r,identity_handovers=[exact()],policy=producer.DailyAccountingPolicy(cost_bps=10,asset_stop_fraction=.1))
        # Jan31 prices remain below stops; the scheduled Feb open cancels old stops.
        self.assertEqual(result.failures.reason.tolist(),['nonpositive_or_nonfinite_nav_after_cost'])
        v.verify_failed_scenario(result.tables,p,s,producer.DAYS,cost_bps=10,stop_fraction=.1,rebalances=r,identity_handovers=[exact()])
        for mutation in ('entry_reference','stop_price','allocated_fee','phase'):
            changed=deepcopy(result);day=pd.Timestamp('2020-02-03')
            if mutation in ('entry_reference','stop_price'):
                changed.holdings.loc[changed.holdings.date.ge(day),mutation]+=1.
                changed.trades.loc[changed.trades.date.eq(day),mutation]+=1.
            elif mutation=='allocated_fee':changed.trades.loc[changed.trades.date.eq(day),'cost']+=.01
            else:changed.trades.loc[changed.trades.date.eq(day),'phase']='stop_gap'
            with self.subTest(mutation=mutation),self.assertRaises(v.VerificationError):
                v.verify_failed_scenario(changed.tables,p,s,producer.DAYS,cost_bps=10,stop_fraction=.1,rebalances=r,identity_handovers=[exact()])

    def test_protected_held_signal_is_rejected_before_numeric_handover_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'handover.parquet'
            pd.DataFrame({'date':['2020-02-03'],'held_signal_date':['2024-01-02'],'units_after':[1.]}).to_parquet(path,index=False)
            original=pq.ParquetFile;reads=[]
            class Guard:
                def __init__(self,*args,**kwargs):self.pf=original(*args,**kwargs)
                def __getattr__(self,name):return getattr(self.pf,name)
                def read(self,columns=None,*args,**kwargs):
                    reads.append(columns)
                    if columns is None or 'units_after' in columns:raise AssertionError('Protected numeric read')
                    return self.pf.read(columns=columns,*args,**kwargs)
            with patch.object(v.pq,'ParquetFile',side_effect=Guard):
                with self.assertRaisesRegex(v.VerificationError,'Protected period'):v.key_table(path,kind='handover')
            self.assertEqual(reads,[['date','held_signal_date']])

    def test_false_identity_failure_dropped_lot_and_applied_blocked_handover_rejected(self):
        s,p,r=producer.frames();result=producer.execute(s,p,r,policy=producer.DailyAccountingPolicy(cost_bps=10))
        for mutation in ('reason','source','native','held_units','handover','target'):
            changed=deepcopy(result)
            if mutation=='reason':changed.failures.loc[0,'reason']='held_identity_route_changed'
            elif mutation=='source':changed.failures.loc[0,'source_security_id']='other'
            elif mutation=='native':changed.failures.loc[0,'intrinio_security_id']='C'
            elif mutation=='held_units':changed.holdings.loc[changed.holdings.date.ge(pd.Timestamp('2020-02-03')),'units']=0.
            elif mutation=='target':changed.orders.loc[changed.orders.date.eq(pd.Timestamp('2020-02-03')),'target_notional']=1.
            else:
                okay=producer.execute(s,p,r,identity_handovers=[exact()],policy=producer.DailyAccountingPolicy(cost_bps=10))
                changed.identity_handovers=okay.identity_handovers
            with self.subTest(mutation=mutation),self.assertRaises(v.VerificationError):v.verify_failed_scenario(changed.tables,p,s,producer.DAYS,cost_bps=10,rebalances=r)
        # A proven exact exception contradicts a claim that this transition failed strict identity validation.
        with self.assertRaises(v.VerificationError):v.verify_failed_scenario(result.tables,p,s,producer.DAYS,cost_bps=10,rebalances=r,identity_handovers=[exact()])

    def test_held_route_change_has_separate_independent_metadata_proof(self):
        s,p,r=producer.frames(change=False);prepared=producer._prepare(s,p,producer.DAYS,r,producer.DAYS[-1]);changed=prepared[0].copy()
        changed.loc[changed.execution_date.eq(pd.Timestamp('2020-02-03'))&changed.source_security_id.eq('old'),'intrinio_security_id']='C'
        with patch('dmv2.walkforward_accounting_v2._prepare',return_value=(changed,*prepared[1:])):
            result=producer.execute(s,p,r,policy=producer.DailyAccountingPolicy(cost_bps=10))
        report=v.verify_failed_scenario(result.tables,p,changed,producer.DAYS,cost_bps=10,rebalances=r)
        self.assertEqual(report['identity_failure_proof']['reason'],'held_identity_route_changed')

    def test_genuine_conversion_exemption_is_proven_not_a_naked_flag(self):
        selected=pd.DataFrame([{'source_security_id':'old','intrinio_security_id':'ORIGINAL','signal_date':'2019-12-31','execution_date':'2020-01-02','side':'long'}])
        state={'old':{'intrinio_security_id':'A','signal_date':'2019-12-31','units':.1,'action_predecessor_id':'FAKE'}}
        event={'event_id':'legal','effective_date':'2020-01-03','predecessor_id':'ORIGINAL','successor_id':'A','stock_ratio':1.,'verified':True,'evidence_reference':'https://example.com/legal'}
        applied=pd.DataFrame([{'event_id':'legal','date':'2020-01-03','mark':'open','source_security_id':'old','predecessor_id':'ORIGINAL','successor_id':'A'}])
        self.assertEqual(v.genuine_converted_sources(state,selected,[event],applied,'2020-02-03'),{'old'})
        with self.assertRaises(v.VerificationError):v.genuine_converted_sources(state,selected,[event],applied.iloc[0:0],'2020-02-03')
        new={'source_security_id':'new','intrinio_security_id':'A','signal_date':'2020-01-31'}
        issue,_=v.identity_transition(state,[new],[new],'2020-02-03',converted={'old'})
        self.assertIsNone(issue)
        issue,_=v.identity_transition(state,[new],[new],'2020-02-03')
        self.assertEqual(issue['reason'],'selected_provider_held_source_collision')

    def test_old_source_current_or_nonunique_provider_never_gets_waiver(self):
        state={'old':{'intrinio_security_id':'A','signal_date':'2019-12-31','units':.1}}
        new={'source_security_id':'new','intrinio_security_id':'A','signal_date':'2020-01-31'}
        for other in ({**new,'source_security_id':'old','intrinio_security_id':'B'},{**new,'source_security_id':'duplicate'}):
            issue,_=v.identity_transition(state,[new,other],[new,other],'2020-02-03',handovers=[exact()])
            self.assertEqual(issue['reason'],'selected_provider_held_source_collision')

    def test_pinned_registry_and_separate_approval_bind_exact_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);phase,directory=registry_fixture(root)
            self.assertEqual(v.review_handover_registry(root,phase),phase['identity_handover_registry']['handovers'])
            (directory/'local_identity_evidence.json').write_text('tampered')
            with self.assertRaises(v.VerificationError):v.review_handover_registry(root,phase)

    def test_actual_shape_local_references_and_real_clock_metadata(self):
        from dmv2.walkforward_clock import walkforward_clock
        sessions,windows,_=walkforward_clock()
        april=windows.loc[windows.outcome_month.eq('2021-04')].iloc[0]
        march=windows.loc[windows.outcome_month.eq('2021-03')].iloc[0]
        record=exact();record.update(effective_date=april.entry_date,signal_date=april.signal_date,
            held_signal_date=march.signal_date)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);phase,_=registry_fixture(root,record=record)
            actual=v.review_handover_registry(root,phase)[0]
            self.assertIn(actual['effective_date'],sessions)
            self.assertEqual(actual['signal_date'],april.signal_date)
            self.assertEqual(actual['held_signal_date'],march.signal_date)
            self.assertIn('primary_findings.json;',actual['evidence_reference'])

    def test_unpinned_unsafe_duplicate_and_nonprimary_local_reference_rejected(self):
        for reference,url in [
            ('primary_findings.json; unpinned.json',None),
            ('primary_findings.json; ../local_identity_evidence.json',None),
            ('primary_findings.json; primary_findings.json',None),
            ('local_identity_evidence.json',None),
            ('arbitrary nonempty local text',None),
            ('https://www.sec.gov/Archives/not-the-exact-pinned-filing.htm',None),
            (None,'http://www.sec.gov/Archives/edgar/dated-filing.htm'),
            (None,'https://www.sec.gov.evil.example/Archives/edgar/dated-filing.htm'),
            (None,'https://example.com/not-primary'),
        ]:
            with self.subTest(reference=reference,url=url),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);record=exact()
                if reference is not None:record['evidence_reference']=reference
                kwargs={} if url is None else {'primary_url':url}
                phase,_=registry_fixture(root,record=record,**kwargs)
                with self.assertRaises(v.VerificationError):v.review_handover_registry(root,phase)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);url='https://www.sec.gov/Archives/edgar/data/synthetic/dated-filing.htm'
            record=exact();record['evidence_reference']=url
            phase,_=registry_fixture(root,record=record,primary_url=url)
            self.assertEqual(v.review_handover_registry(root,phase)[0]['evidence_reference'],url)

    def test_approval_no_identity_promotion_or_different_transition(self):
        for kind in ('status','registry','package','transition','identity'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);phase,directory=registry_fixture(root);path=directory/'decision_record.json';decision=v.json_file(path)
                if kind=='status':decision['status']='GENERAL_ALIAS_APPROVAL'
                elif kind=='registry':decision['registry_sha256']='0'*64
                elif kind=='package':decision['evidence_package_receipt_sha256']='0'*64
                elif kind=='transition':decision['allowed_transition']['old_source_security_id']='wrong'
                else:decision['identity_status']['source_identity_verified']=True
                save(path,decision);repin_decision(root,phase,path)
                with self.assertRaises(v.VerificationError):v.review_handover_registry(root,phase)

    def test_failed_v1_attempt_is_separate_exhaustively_pinned_and_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();run=root/'outputs/walkforward/synthetic';prior=run/'backtest';new=run/'backtest_v2';new.mkdir(parents=True)
            old={'parent_identity_sha256':'sha256:parent','models_fitted':False,'preprocessing_fitted':False}
            old_pin=v.canonical(old);failure={'exception':'DataContractError','message':'Selected provider collides with a different held source identity'}
            for name,value in [('build_identity.json',old),('failure.json',failure),('audit.json',{'status':'STARTED','phase':'walkforward_continuous_backtest_v1','build_identity_sha256':'sha256:parent','phase_identity_sha256':old_pin})]:save(prior/name,value)
            pins=[pin(root,p) for p in sorted(prior.iterdir())]
            phase={'parent_identity_sha256':'sha256:parent','source_pins':pins,'prior_failed_attempt':{
                'directory':str(prior.relative_to(root)),'file_count':len(pins),'all_existing_files_pinned':True,
                'prior_phase_identity_sha256':old_pin,'failure':failure,'immutable_source_pins':pins}}
            self.assertEqual(v.review_prior_failed_attempt(root,run,new,phase),3)
            save(prior/'unrecorded.json',{'extra':True})
            with self.assertRaises(v.VerificationError):v.review_prior_failed_attempt(root,run,new,phase)

    def test_audit_has_no_production_accounting_or_fit_imports(self):
        text=SCRIPT.read_text()
        for token in ('from dmv2','import sklearn','import xgboost','.fit('):self.assertNotIn(token,text)

if __name__=='__main__':unittest.main()
