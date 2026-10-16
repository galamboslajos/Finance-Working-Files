"""Synthetic-only freshness, provenance, metadata/PnL separation and replay tests."""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from importlib.util import find_spec
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import pyarrow.parquet as pq

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'src'), str(R / 'tests'), str(Path(__file__).parent)]
import dmv2.source_corrected_backtest as m
from test_walkforward_preflight_runner import fixture as base_fixture, save
from test_walkforward_accounting_v2 import waiver, approval_files


def fixture(root):
    # A deliberately synthetic metadata/reference fixture; never an actual run.
    old_run, cfg, old_id, model_audit = base_fixture(root)
    evidence = root / 'data/derived/original_handover'
    evidence.mkdir(parents=True); (evidence / 'primary.json').write_text('{"synthetic":true}')
    handover = waiver()
    handover.update(effective_date='2021-04-01', signal_date='2021-03-31', held_signal_date='2021-02-26')
    registry = {'schema_version': 1, 'registry_id': 'synthetic-provisional',
        'root_build_identity_sha256': old_id, 'status': 'PROVISIONAL_EXECUTION_ONLY',
        'source_identity_promoted': False, 'handovers': [handover],
        'evidence_pins': [{'base': 'registry_directory', 'path': 'primary.json',
                           'sha256': m._hash(evidence / 'primary.json')}]}
    regpath = evidence / 'identity_handover_registry.json'; save(regpath, registry)
    approval_files(evidence, regpath, registry)
    approval = {'decision_pin': m._pin(root, evidence / 'decision_record.json'),
                'package_receipt_pin': m._pin(root, evidence / 'package_receipt.json')}
    historical = json.loads((root / cfg['historical_retirement_registry']['directory'] / 'build_identity.json').read_text())['events']
    old_phase = {'parent_identity_sha256': old_id, 'phase': m.PHASE,
        'implementation_revision': m.IMPLEMENTATION_REVISION,
        'workers': {name: m._hash(Path(find_spec('dmv2.' + name[:-3]).origin)) for name in m.WORKERS},
        'source_pins': [m._pin(root, old_run / 'build_identity.json')],
        'historical_retirements': historical,
        'event_registry': {'schema': 'dmv2-walkforward-actions/v1', 'build_identity_sha256': old_id,
                           'events': [], 'source_pins': [], 'supplemental_quote_sources': []},
        'identity_handover_registry': registry, 'identity_handover_registry_pin': m._pin(root, regpath),
        'identity_handover_approval': approval}
    reference = old_run / 'backtest_identity_v2'
    save(reference / 'build_identity.json', old_phase)
    old_phase_id = m.canonical_sha256(old_phase)
    save(reference / 'audit.json', {'status': 'COMPLETE', 'phase': m.PHASE,
        'build_identity_sha256': old_id, 'phase_identity_sha256': old_phase_id,
        'implementation_revision': m.IMPLEMENTATION_REVISION, 'models_fitted': False,
        'preprocessing_fitted': False, 'source_pins_unchanged': True,
        'preflight_before_performance': True, 'source_identity_promoted': False,
        'scenarios': 64, 'sessions': 1006, 'outcome_months': 48,
        'artifacts': {'build_identity.json': m._pin(reference, reference / 'build_identity.json')}})
    sources = []
    for year in range(2010, 2024):
        source = old_run / 'years' / str(year)
        if year < 2020:
            save(source / 'audit.json', {'status': 'COMPLETE', 'year': year,
                'build_identity_sha256': old_id, 'artifacts': {}})
        sources.append({'year': year, 'directory': str(source.relative_to(root)),
            'audit_sha256': m._hash(source / 'audit.json'), 'build_identity_sha256': old_id,
            'corrected': year in [2010, 2011, 2012, 2013, 2022, 2023], 'original_rows': 4})
    manifest_path = root / 'outputs/source_quality/corrected_annual_sources.json'
    save(manifest_path, {'status': 'VERIFIED_INPUT_REFERENCES_NOT_MODEL_RESULTS', 'years': sources,
        'context_policy': 'available_cohort_80pct', 'financial_2024_plus_analysis': False})
    manifest_pin = m._pin(root, manifest_path)
    run = root / 'outputs/walkforward_source_corrected/synthetic'
    identity = {'configuration': cfg, 'source_correction_manifest_pin': manifest_pin,
                'source_corrected_synthetic_trial': True}
    identity_sha = m.canonical_sha256(identity)
    save(run / 'build_identity.json', identity)
    shutil.copytree(old_run / 'years', run / 'years')
    for year in range(2020, 2024):
        path = run / 'years' / str(year) / 'audit.json'
        source_item = next(item for item in sources if item['year'] == year)
        original_audit = old_run / 'years' / str(year) / 'audit.json'
        copy = path.parent / 'reference_annual_audit.json'; shutil.copyfile(original_audit, copy)
        value = json.loads(path.read_text()); value['build_identity_sha256'] = identity_sha
        value['artifacts']['reference_annual_audit'] = m._pin(path.parent, copy)
        value['source_corrected_replay_lineage'] = {
            'directory': source_item['directory'], 'audit_pin': m._pin(root, original_audit),
            'build_identity_sha256': source_item['build_identity_sha256'],
            'corrected': source_item['corrected'], 'original_rows': source_item['original_rows'],
            'numerical_artifact_bytes_unchanged': True, 'new_feature_calculations': 0,
            'original_audit_preserved_as': 'reference_annual_audit.json'}
        save(path, value)
    # Detached synthetic model completion metadata. No estimator is constructed.
    shutil.copytree(old_run / 'models', run / 'models')
    new_model_phase = {'parent_identity_sha256': identity_sha}
    new_model_id = m.canonical_sha256(new_model_phase)
    save(run / 'models/build_identity.json', new_model_phase)
    new_model_audit = deepcopy(model_audit)
    new_model_audit.update(build_identity_sha256=identity_sha, model_identity_sha256=new_model_id)
    for arm in m.ARMS:
        path = run / 'models/arms' / arm / 'audit.json'
        value = json.loads(path.read_text())
        value.update(build_identity_sha256=identity_sha, model_identity_sha256=new_model_id)
        save(path, value); new_model_audit['arms'][arm] = m._pin(run / 'models', path)
    save(run / 'models/audit.json', new_model_audit)
    args = {'reference_directory': reference, 'reference_audit_sha256': m._hash(reference / 'audit.json'),
            'reference_phase_sha256': old_phase_id}
    package = m.build_rebound_execution_evidence(root, run, identity_sha, **args,
        output_directory=root / 'data/derived/new_rebound_evidence')
    return SimpleNamespace(root=root, old_run=old_run, cfg=cfg, old_id=old_id, run=run,
        identity=identity_sha, model_audit=new_model_audit, reference=reference,
        args=args, package=package, manifest_pin=manifest_pin, old_phase=old_phase)


def invoke(f, **kwargs):
    return m.run_source_corrected_backtest(f.root, f.run, f.cfg, f.identity,
        **f.args, source_correction_manifest_pin=f.manifest_pin, **f.package,
        output_directory=f.run / 'backtest_source_corrected', **kwargs)


class FreshReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = fixture(Path(self.tmp.name).resolve())

    def test_fresh_scope_needs_completed_reference_and_no_new_failed_attempt(self):
        f = self.f
        directory, context, phase, identity = m._fresh_scope(f.root, f.run, f.identity,
            output_directory=f.run / 'backtest_source_corrected', **f.args)
        self.assertFalse(directory.exists()); self.assertFalse((f.run / 'backtest').exists())
        self.assertTrue(context['fresh_run_requires_no_prior_failed_attempt'])
        self.assertFalse(context['economic_reports_loaded'])

    def test_existing_output_or_old_run_is_never_overwritten(self):
        f = self.f
        for path in [f.run, f.reference, f.run / 'already_exists']:
            path.mkdir(exist_ok=True)
            with self.subTest(path=path), self.assertRaises(m.DataContractError):
                m._fresh_scope(f.root, f.run, f.identity, output_directory=path, **f.args)
        with self.assertRaises(m.DataContractError):
            m._fresh_scope(f.root, f.run, f.identity, output_directory=f.run / 'nested/replay', **f.args)

    def test_old_audit_pin_checked_before_json_decode(self):
        f = self.f
        with patch.object(m, '_json', side_effect=AssertionError('JSON read before outer reference raw pin')):
            with self.assertRaisesRegex(m.DataContractError, 'audit pin changed'):
                m._reference_context(f.root, f.reference, audit_sha256='0' * 64,
                                     phase_sha256=f.args['reference_phase_sha256'])

    def test_original_frozen_worker_change_fails_closed(self):
        f = self.f; original = m._hash
        def changed(path):
            return '0' * 64 if Path(path).name == 'walkforward_accounting_v2.py' else original(path)
        with patch.object(m, '_hash', changed), self.assertRaisesRegex(m.DataContractError, 'code changed'):
            m._reference_context(f.root, f.reference,
                audit_sha256=f.args['reference_audit_sha256'], phase_sha256=f.args['reference_phase_sha256'])

    def test_rebind_preserves_events_handover_evidence_and_old_files(self):
        f = self.f
        for name, pin in f.old_phase['identity_handover_approval'].items(): m._verify_pin(f.root, pin)
        m._verify_pin(f.root, f.old_phase['identity_handover_registry_pin'])
        registry = json.loads(f.package['identity_handover_registry'].read_text())
        events = json.loads(f.package['event_registry'].read_text())
        self.assertEqual(registry['handovers'], f.old_phase['identity_handover_registry']['handovers'])
        self.assertEqual(events['events'], f.old_phase['event_registry']['events'])
        self.assertEqual(registry['root_build_identity_sha256'], f.identity)
        self.assertEqual(events['build_identity_sha256'], f.identity)
        self.assertNotEqual(f.identity, f.old_id)
        self.assertFalse(registry['source_identity_promoted'])
        self.assertEqual((f.package['identity_handover_registry'].parent / 'primary.json').read_bytes(),
                         (f.root / 'data/derived/original_handover/primary.json').read_bytes())

    def test_rebind_cannot_overwrite_package(self):
        f = self.f
        with self.assertRaisesRegex(m.DataContractError, 'Preserve every existing'):
            m.build_rebound_execution_evidence(f.root, f.run, f.identity, **f.args,
                output_directory=f.package['event_registry'].parent)

    def test_original_root_handover_cannot_silently_apply_to_new_root(self):
        f = self.f; sessions, windows, _ = m.walkforward_clock()
        with self.assertRaisesRegex(m.DataContractError, 'bind one reviewed'):
            m._identity_registry(f.root, f.root / f.old_phase['identity_handover_registry_pin']['path'],
                f.identity, sessions, dict(zip(pd.to_datetime(windows.execution_date), pd.to_datetime(windows.signal_date))), [])

    def test_corrected_manifest_and_all_execution_artifact_bytes_are_bound(self):
        f = self.f; main = json.loads((f.run / 'build_identity.json').read_text())
        binding = m._corrected_sources(f.root, f.run, f.identity, main, f.manifest_pin, [])
        self.assertEqual(len(binding['annual_bindings']), 4)
        path = f.run / 'years/2022/audit.json'; value = json.loads(path.read_text())
        value['artifacts']['daily_prices_raw']['sha256'] = '0' * 64; save(path, value)
        with self.assertRaisesRegex(m.DataContractError, 'artifact differs'):
            m._corrected_sources(f.root, f.run, f.identity, main, f.manifest_pin, [])

    def test_manifest_missing_from_new_identity_fails_closed(self):
        f = self.f
        with self.assertRaisesRegex(m.DataContractError, 'explicitly pin'):
            m._corrected_sources(f.root, f.run, f.identity, {'configuration': f.cfg}, f.manifest_pin, [])

    def test_same_byte_detached_artifact_path_alias_is_refused(self):
        f = self.f; main = json.loads((f.run / 'build_identity.json').read_text())
        path = f.run / 'years/2022/audit.json'; original = json.loads(path.read_text())
        for kind in ('daily_prices_raw', 'reference_annual_audit'):
            value = deepcopy(original)
            declared = value['artifacts'][kind]
            source = path.parent / declared['path']
            alias = path.parent / 'alias' / source.name
            alias.parent.mkdir(exist_ok=True); shutil.copyfile(source, alias)
            declared['path'] = str(alias.relative_to(path.parent)); save(path, value)
            with self.subTest(kind=kind), self.assertRaisesRegex(m.DataContractError, 'artifact differs|audit copy'):
                m._corrected_sources(f.root, f.run, f.identity, main, f.manifest_pin, [])
        save(path, original)

    def test_rebound_approval_changes_beyond_root_binding_are_refused(self):
        f = self.f; sessions, windows, _ = m.walkforward_clock()
        manifest, regpin, records, approval = m._identity_registry(f.root,
            f.package['identity_handover_registry'], f.identity, sessions,
            dict(zip(pd.to_datetime(windows.execution_date), pd.to_datetime(windows.signal_date))), [],
            decision_path=f.package['identity_handover_decision'])
        reference, old, _ = m._reference_context(f.root, f.reference,
            audit_sha256=f.args['reference_audit_sha256'], phase_sha256=f.args['reference_phase_sha256'])
        events = json.loads(f.package['event_registry'].read_text())
        m._validate_rebound_execution(reference, old, f.identity, events, manifest, approval, f.root)
        decision = json.loads(f.package['identity_handover_decision'].read_text())
        decision['approval_basis'] = 'Invented new approval'; save(f.package['identity_handover_decision'], decision)
        approval['decision_pin'] = m._pin(f.root, f.package['identity_handover_decision'])
        with self.assertRaisesRegex(m.DataContractError, 'exact original decision'):
            m._validate_rebound_execution(reference, old, f.identity, events, manifest, approval, f.root)

    def test_handover_or_action_change_not_permitted_by_rebind(self):
        f = self.f; reference, old, _ = m._reference_context(f.root, f.reference,
            audit_sha256=f.args['reference_audit_sha256'], phase_sha256=f.args['reference_phase_sha256'])
        events = json.loads(f.package['event_registry'].read_text())
        handover = json.loads(f.package['identity_handover_registry'].read_text())
        for which in ['events', 'handover']:
            e, h = deepcopy(events), deepcopy(handover)
            if which == 'events': e['events'] = [{'fabricated': True}]
            else: h['handovers'][0]['effective_date'] = '2021-04-02'
            with self.subTest(which=which), self.assertRaisesRegex(m.DataContractError, 'exactly the original'):
                m._validate_rebound_execution(reference, old, f.identity, e, h, {}, f.root)

    def test_preflight_all_new_horizons_precedes_financial_execution_decode(self):
        f = self.f; original = pq.ParquetFile
        class Guard:
            def __init__(self, path, *args, **kwargs):
                self.path = Path(path); self.base = original(path, *args, **kwargs)
            def __getattr__(self, name): return getattr(self.base, name)
            def read(self, columns=None, *args, **kwargs):
                if self.path.name == 'daily_prices_raw.parquet' and (columns is None or {'open', 'close', 'adj_open', 'adj_close'} & set(columns)):
                    raise AssertionError('Execution prices decoded during preflight')
                return self.base.read(columns=columns, *args, **kwargs)
        with patch('dmv2.walkforward_models.read_walkforward_models', return_value={'audit': f.model_audit}), \
             patch.object(m, 'run_walkforward_accounting', side_effect=AssertionError('PnL before preflight')), \
             patch('pyarrow.parquet.ParquetFile', side_effect=Guard):
            directory = invoke(f, preflight_only=True)
        summary = pd.read_parquet(directory / 'summary.parquet')
        self.assertEqual(len(summary), 8); self.assertTrue(summary.ready_for_accounting.all())
        self.assertFalse((f.run / 'backtest_source_corrected').exists())
        phase = json.loads((directory / 'build_identity.json').read_text())
        self.assertEqual(phase['fresh_source_replay'], m.SOURCE_REPLAY)
        self.assertNotIn('prior_failed_attempt', phase)
        self.assertTrue(phase['original_completed_reference']['fresh_run_requires_no_prior_failed_attempt'])
        for arm in m.ARMS:
            for basket in ('class_1_10', 'class_2_9'):
                self.assertEqual(len(pd.read_parquet(directory / arm / basket / 'monthly.parquet')), 48)

    def test_missing_selected_quote_blocks_performance_after_saved_preflight(self):
        f = self.f; original = m._price_plan
        def missing(*args):
            plans, metadata, read = original(*args)
            mask = metadata.intrinio_security_id.eq('A') & metadata.trading_date.eq('2020-01-02')
            return plans, metadata.loc[~mask], read
        with patch('dmv2.walkforward_models.read_walkforward_models', return_value={'audit': f.model_audit}), \
             patch.object(m, '_price_plan', side_effect=missing), \
             patch.object(m, 'run_walkforward_accounting', side_effect=AssertionError('Unreviewed gap entered PnL')):
            with self.assertRaisesRegex(m.DataContractError, 'Pre-performance coverage requires repair/review'):
                invoke(f)
        self.assertFalse((f.run / 'backtest_source_corrected').exists())
        self.assertEqual(len(list((f.run / 'preflight').glob('*/summary.parquet'))), 1)

    def test_64_scenarios_use_unchanged_engine_and_exact_grid(self):
        f = self.f
        result = SimpleNamespace(status='complete_proxy',
            tables={'identity_handovers': pd.DataFrame(columns=['handover_id'])},
            failures=pd.DataFrame(), action_events=pd.DataFrame(), orders=pd.DataFrame())
        def summary(value, **kw):
            row = {key: kw[key] for key in ('arm', 'basket', 'variant', 'cost_bps')}
            row.update(complete_proxy_path=True, expected_sessions=1006)
            return row, pd.DataFrame({'month': pd.period_range('2020-01', '2023-12', freq='M').astype(str)})
        def yearlies(value, **kw): return pd.DataFrame([{**kw, 'year': year} for year in range(2020, 2024)])
        with patch('dmv2.walkforward_models.read_walkforward_models', return_value={'audit': f.model_audit}), \
             patch.object(m, 'run_walkforward_accounting', return_value=result) as engine, \
             patch.object(m, 'summarize_daily', side_effect=summary), \
             patch.object(m, 'yearly_breakdown', side_effect=yearlies), patch('builtins.print'):
            directory = invoke(f)
        self.assertEqual(engine.call_count, 64)
        self.assertEqual({c.kwargs['policy'].cost_bps for c in engine.call_args_list}, {0, 5, 10, 25})
        self.assertEqual({c.kwargs['policy'].asset_stop_fraction for c in engine.call_args_list}, {None, .1})
        self.assertTrue(all(c.kwargs['policy'].required_names_per_side is None for c in engine.call_args_list))
        self.assertTrue(all(len(c.args[2]) == 1006 for c in engine.call_args_list))
        self.assertTrue(all(c.kwargs['terminal_date'] == '2023-12-29' for c in engine.call_args_list))
        audit = json.loads((directory / 'audit.json').read_text())
        self.assertEqual(audit['scenarios'], 64); self.assertTrue(audit['old_completed_reference_preserved'])
        self.assertEqual(audit['fresh_source_replay'], m.SOURCE_REPLAY)
        self.assertFalse(audit['models_fitted']); self.assertFalse(audit['preprocessing_fitted'])
        self.assertEqual(len(pd.read_parquet(directory / 'comparison.parquet')), 64)
        self.assertEqual(sum(name.endswith('/identity_handovers.parquet') for name in audit['artifacts']), 64)
        self.assertEqual(len(pd.read_parquet(directory / 'selection_coverage.parquet')), 384)
        saved = m.read_source_corrected_backtest(directory, root=f.root, identity_sha256=f.identity)
        self.assertEqual(len(saved['comparison']), 64)

    def test_accounting_and_metric_functions_are_original_objects(self):
        from dmv2.walkforward_accounting_v2 import run_walkforward_accounting_v2
        from dmv2.daily_backtest import summarize_daily
        from dmv2.walkforward_backtest import yearly_breakdown
        from dmv2.walkforward_selection import select_walkforward_basket
        self.assertIs(m.run_walkforward_accounting, run_walkforward_accounting_v2)
        self.assertIs(m.summarize_daily, summarize_daily)
        self.assertIs(m.yearly_breakdown, yearly_breakdown)
        self.assertIs(m.select_walkforward_basket, select_walkforward_basket)


if __name__ == '__main__': unittest.main()
