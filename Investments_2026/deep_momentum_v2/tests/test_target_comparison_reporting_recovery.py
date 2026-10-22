"""Invented metadata and tiny synthetic aggregates only; no project data reads."""
from pathlib import Path
import datetime as dt
import hashlib
import itertools
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2 import target_comparison_reporting_recovery as r


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + '\n')
    return pin(path)


def pin(path):
    path = path.resolve()
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(parent, *, proof_status='PASS', model_fits=20, scenario_drop=False):
    parent = parent.resolve()
    root = parent / 'run'; source = parent / 'temporary_evidence'; copies = root / 'verification_recovery'
    source.mkdir(); copies.mkdir(parents=True)
    worker = source / 'verifier.py'; worker.write_text('# invented inert verifier fixture\n'); wpin = pin(worker)
    model = dict(expected_fits=20, expected_preprocessors=4, financial_2024_plus_allowed=False)
    model['plan_sha256'] = 'sha256:' + r._canonical(model)
    mpin = save(root / 'read_plan.json', model)
    mvp = save(root / 'verification/models_read_plan.json', dict(phase='models', model_plan_pin=mpin, worker_pin=wpin))
    mproof = dict(status='PASS', phase='models', model_plan_pin=mpin, verification_plan_pin=mvp,
        worker_pin=wpin, verified_estimator_fits=model_fits, verified_shared_preprocessors=4,
        new_estimator_fits=0, financial_2024_plus_values_read=False,
        results=[dict(year=y,head=h) for y,h in itertools.product((2020,2021,2022,2023),r.HEADS)])
    mproofpin = save(root / 'verification/models.json', mproof)
    lpin = save(root / 'ledger.json', dict(status='FAILED_PRESERVED', fits_started=20, fits_completed=20, model_plan_pin=mpin))
    sessions = [(dt.date(2020,2,3)+dt.timedelta(days=i)).isoformat() for i in range(984)] + ['2023-12-29']
    backtest = dict(model_plan_sha256=model['plan_sha256'], scenarios=72, economic_months=47,
        financial_2024_plus_read=False, output_directory=str(root / 'backtest'), sessions=sessions)
    backtest['plan_sha256'] = r._canonical(backtest)
    bpin = save(root / 'backtest_plan.json', backtest)
    cases = sorted(r.CASES)
    comp = [dict(zip(('arm','basket','variant','cost_bps'), c), complete_proxy_path=(i%2==0),
                 total_return=.01) for i,c in enumerate(cases)]
    year = [{**row,'year':y} for row in comp for y in (2020,2021,2022,2023)]
    artifacts = {}
    for name, rows in (('comparison',comp),('yearly',year)):
        path = root / 'backtest' / (name + '.parquet'); path.parent.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows),path); artifacts[name+'.parquet'] = pin(path)
    auditpin = save(root / 'backtest/audit.json', dict(plan_sha256=backtest['plan_sha256'],
        model_verification_pin=mproofpin, status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION', artifacts=artifacts))
    inventory = {i['path']:dict(pin=i) for i in (*artifacts.values(),auditpin)}
    vplan = dict(schema='dmv2-unstopped-target-independent-verification/v5', status='SEALED_BEFORE_FINANCIAL_READ',
        phase='backtest', model_plan_pin=mpin, backtest_plan_pin=bpin, model_verification_pin=mproofpin,
        worker_pin=wpin, run_directory=str(root), receipt_directory=str(source), new_estimator_fits=0,
        financial_2024_plus_authorized=False, derived_read_bytes=1000, artifacts=inventory)
    vpin = save(source / 'read_plan.json',vplan)
    launchpin = save(source / 'launch.json',dict(status='SEALED_VERIFICATION_ONLY',parent_ledger_pin=lpin,
        model_verification_pin=mproofpin,verification_plan_pin=vpin,worker_pin=wpin,new_estimator_fits=0,
        new_backtests=0,original_run_mutations=False,cumulative_reserved_derived_bytes=2000,
        maximum_cumulative_derived_bytes=10000,new_reserved_derived_bytes=1000,
        maximum_new_active_seconds=100,rss_maximum_bytes=1000000))
    scenario_checks = [dict(zip(('arm','basket','variant','cost_bps'), c), complete_proxy_path=comp[i]['complete_proxy_path'])
                       for i,c in enumerate(cases)]
    if scenario_drop:scenario_checks[-1] = scenario_checks[0]
    proofpin = save(source / 'independent_verification.json',dict(schema=vplan['schema'],status=proof_status,
        phase='backtest',verification_plan_pin=vpin,worker_pin=wpin,model_plan_pin=mpin,backtest_plan_pin=bpin,
        model_verification_pin=mproofpin,producer_backtest_audit_pin=auditpin,new_estimator_fits=0,
        financial_2024_plus_values_read=False,source_identity_action_completeness_or_investability_certified=False,
        production_selection_or_accounting_used_as_expected_answer=False,verified_scenarios=72,scenario_checks=scenario_checks))
    cpin = save(source / 'completion.json',dict(status='PASS_VERIFICATION_ONLY',original_run_preserved=True,
        proof_pin=proofpin,new_estimator_fits=0,new_backtests=0,active_seconds=1.,peak_rss_bytes=1000))
    values = dict(independent_backtest_proof_pin=proofpin,verification_plan_pin=vpin,verifier_worker_pin=wpin,
        completion_pin=cpin,launch_pin=launchpin)
    relocations = []
    for field, oldpin in values.items():
        dest=copies / Path(oldpin['path']).name; shutil.copyfile(oldpin['path'],dest)
        relocations.append(dict(original_pin=oldpin,copy_pin=pin(dest)))
    handoff = dict(schema=r.SCHEMA,status='COMPLETE_VERIFIED_RECOVERY',original_run_directory=str(root),
        model_plan_pin=mpin,original_ledger_pin=lpin,model_verification_pin=mproofpin,backtest_plan_pin=bpin,
        producer_backtest_audit_pin=auditpin,evidence_relocations=relocations,new_estimator_fits=0,
        new_backtest_replays=0,financial_2024_plus_values_read=False,
        source_identity_action_completeness_or_investability_certified=False,**values)
    save(copies / 'recovery_handoff.json',handoff)
    return root,source,handoff


class RecoveryReportingTests(unittest.TestCase):
    def test_absent_handoff_fails_without_creating_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'absent'
            with self.assertRaises(FileNotFoundError):r.read_target_comparison_recovery(root)
            self.assertFalse(root.exists())

    def test_complete_proof_opens_only_small_original_aggregates_and_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root,source,_=fixture(Path(d)); before=(root/'ledger.json').read_bytes()
            original=pq.read_table; seen=[]
            def read(path,*args,**kwargs):seen.append(Path(path).name);return original(path,*args,**kwargs)
            with patch('pyarrow.parquet.read_table',side_effect=read):answer=r.read_target_comparison_recovery(root)
            self.assertEqual(seen,['comparison.parquet','yearly.parquet'])
            self.assertEqual((root/'ledger.json').read_bytes(),before)
            self.assertEqual(len(answer['comparison']),72);self.assertEqual(len(answer['yearly']),288)
            self.assertEqual(answer['original_producer_status'],'FAILED_PRESERVED')
            self.assertFalse(answer['automatic_computation']);self.assertFalse(answer['source_identity_or_investability_certified'])

    def test_explicit_byte_exact_relocation_survives_missing_temporary_originals(self):
        with tempfile.TemporaryDirectory() as d:
            root,source,_=fixture(Path(d));shutil.rmtree(source)
            self.assertTrue(r.read_target_comparison_recovery(root)['backtest_independently_verified'])

    def test_changed_copy_fails_before_any_aggregate_decode(self):
        with tempfile.TemporaryDirectory() as d:
            root,_,_=fixture(Path(d));p=root/'verification_recovery/verifier.py';p.write_text('changed')
            with patch('pyarrow.parquet.read_table',side_effect=AssertionError('decoded')):
                with self.assertRaisesRegex(ValueError,'Recovery byte'):r.read_target_comparison_recovery(root)

    def test_nonpass_or_incomplete_proof_fails_before_decode(self):
        for kwargs in ({'proof_status':'FAIL'},{'model_fits':19},{'scenario_drop':True}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as d:
                root,_,_=fixture(Path(d),**kwargs)
                with patch('pyarrow.parquet.read_table',side_effect=AssertionError('decoded')):
                    with self.assertRaises(ValueError):r.read_target_comparison_recovery(root)

    def test_original_producer_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,_,_=fixture(Path(d));p=root/'ledger.json';v=json.loads(p.read_text());v['status']='COMPLETE_VERIFIED';save(p,v)
            with self.assertRaisesRegex(ValueError,'Recovery byte'):r.read_target_comparison_recovery(root)

    def test_duplicate_or_outside_copy_relocations_are_rejected(self):
        for field in ('duplicate','outside','run_artifact'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root,_,handoff=fixture(Path(d));rels=handoff['evidence_relocations']
                if field=='duplicate':rels[-1]=rels[0]
                elif field=='outside':rels[0]['copy_pin']=rels[0]['original_pin']
                else:rels[0]['original_pin']=handoff['original_ledger_pin']
                save(root/'verification_recovery/recovery_handoff.json',handoff)
                with self.assertRaises(ValueError):r.read_target_comparison_recovery(root)

    def test_aggregate_byte_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,_,_=fixture(Path(d));p=root/'backtest/comparison.parquet'
            p.write_bytes(p.read_bytes()+b'changed')
            with patch('pyarrow.parquet.read_table',side_effect=AssertionError('decoded')):
                with self.assertRaisesRegex(ValueError,'Recovery byte'):r.read_target_comparison_recovery(root)


if __name__=='__main__':unittest.main()
