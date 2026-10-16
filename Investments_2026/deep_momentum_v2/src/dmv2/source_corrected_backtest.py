"""Fresh source-corrected replay with frozen V2 selection/accounting/metrics.

This additive orchestrator authenticates a completed old reference and explicitly
rebound execution evidence. It never invents a failed prior attempt, fits a
model, fetches a quote, changes selections for missing quotes or edits old runs.
"""
from copy import deepcopy
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import json
import platform
import shutil
import time
import pandas as pd
import pyarrow.parquet as pq
from dmv2.walkforward_backtest import (
    PHASE, REPORTS, MAX_READ_BYTES, _configuration, _historical_registry,
    _prediction_plan, _price_plan, _supplement_plan, yearly_breakdown,
    read_walkforward_backtest, ACTION_PRICE_COLUMNS, DailyAccountingPolicy,
    ARMS, _hash, _json, _pin, _verify_pin, _within, PROBABILITIES,
    canonical_sha256, _rss, _save_table, _write_json, summarize_daily,
    DataContractError, walkforward_clock, validate_current_events,
    select_walkforward_basket, validate_walkforward_basket,
    preflight_selected_horizons, BALANCED_BREADTH_POLICY,
)
from dmv2.walkforward_backtest_v2 import WORKERS, _identity_registry
from dmv2.walkforward_accounting_v2 import (
    IMPLEMENTATION_REVISION,
    run_walkforward_accounting_v2 as run_walkforward_accounting,
)

SOURCE_REPLAY = 'fresh_source_corrected_walkforward_v1'
REBIND_SCHEMA = 'dmv2-source-corrected-execution-evidence-rebind/v1'


def _need(value, message):
    if not value:
        raise DataContractError(message)


def _verify_original_workers(phase):
    expected = phase.get('workers', {})
    _need(set(WORKERS) <= set(expected), 'Reference omits an original frozen V2 worker')
    actual = {name: _hash(Path(find_spec('dmv2.' + name[:-3]).origin)) for name in WORKERS}
    _need(all(actual[name] == expected[name] for name in WORKERS),
          'Original frozen selection/accounting/metrics code changed since the reference')
    return actual


def _reference_context(root, directory, *, audit_sha256, phase_sha256):
    """Authenticate old metadata/evidence, without loading its economic reports."""
    root = Path(root).resolve(); directory = _within(root, Path(directory).resolve())
    ap, bp = directory / 'audit.json', directory / 'build_identity.json'
    _need(ap.is_file() and ap.stat().st_size <= 2 * 1024**2 and _hash(ap) == audit_sha256,
          'Explicit old completed-reference audit pin changed')
    audit = _json(ap)
    _need(bp.is_file() and bp.stat().st_size <= 2 * 1024**2,
          'Old completed-reference phase exceeds metadata budget')
    phase_file_pin = audit.get('artifacts', {}).get('build_identity.json')
    _need(isinstance(phase_file_pin, dict), 'Old reference must raw-pin its exact phase file')
    _verify_pin(directory, phase_file_pin)
    phase = _json(bp)
    _need(canonical_sha256(phase) == phase_sha256 == audit.get('phase_identity_sha256')
          and audit.get('status') == 'COMPLETE' and audit.get('phase') == PHASE
          and audit.get('implementation_revision') == IMPLEMENTATION_REVISION
          and phase.get('implementation_revision') == IMPLEMENTATION_REVISION
          and audit.get('scenarios') == 64 and audit.get('sessions') == 1006
          and audit.get('outcome_months') == 48
          and audit.get('models_fitted') is False and audit.get('preprocessing_fitted') is False
          and audit.get('source_pins_unchanged') is True
          and audit.get('preflight_before_performance') is True
          and audit.get('source_identity_promoted') is False,
          'Old reference is not the explicitly pinned completed V2 48-month replay')
    original_workers = _verify_original_workers(phase)
    old_run = directory.parent
    root_pin = next((item for item in phase.get('source_pins', [])
                     if item.get('path') == str((old_run / 'build_identity.json').relative_to(root))), None)
    _need(root_pin is not None, 'Old reference must raw-pin its original root identity')
    _verify_pin(root, root_pin)
    original_identity = _json(old_run / 'build_identity.json')
    old_identity = canonical_sha256(original_identity)
    _need(old_identity == audit.get('build_identity_sha256') == phase.get('parent_identity_sha256'),
          'Old reference lost its original parent root identity')
    pins = [_pin(root, path) for path in (ap, bp, old_run / 'build_identity.json')]
    for key in ('identity_handover_registry_pin',):
        _need(isinstance(phase.get(key), dict), 'Old reference must retain its explicit handover evidence')
        _verify_pin(root, phase[key])
    approval = phase.get('identity_handover_approval')
    _need(isinstance(approval, dict), 'Old handover decision/receipt bindings are missing')
    sessions, windows, _ = walkforward_clock()
    manifest, regpin, records, checked_approval = _identity_registry(
        root, root / phase['identity_handover_registry_pin']['path'], old_identity, sessions,
        dict(zip(pd.to_datetime(windows.execution_date), pd.to_datetime(windows.signal_date))), pins,
        decision_path=root / approval['decision_pin']['path'])
    _need(manifest == phase.get('identity_handover_registry') and checked_approval == approval,
          'Old handover registry/approval differs from the completed reference')
    registry = phase.get('event_registry')
    _need(isinstance(registry, dict) and registry.get('schema') == 'dmv2-walkforward-actions/v1'
          and registry.get('build_identity_sha256') == old_identity,
          'Old current-event registry lost its original root binding')
    validate_current_events(registry['events'], sessions)
    for item in registry.get('source_pins', []):
        _verify_pin(root, item)
        if item not in pins: pins.append(item)
    context = {'directory': str(directory.relative_to(root)), 'audit_pin': _pin(root, ap),
               'phase_file_pin': _pin(root, bp), 'phase_identity_sha256': phase_sha256,
               'root_identity_sha256': old_identity, 'workers': original_workers,
               'economic_reports_loaded': False, 'immutable_source_pins': pins,
               'fresh_run_requires_no_prior_failed_attempt': True}
    return context, phase, original_identity


def _fresh_scope(root, run, identity, *, output_directory, reference_directory,
                 reference_audit_sha256, reference_phase_sha256):
    _need(output_directory is not None and reference_directory is not None,
          'Fresh replay requires explicit new output and completed old reference')
    directory = _within(run, Path(output_directory).resolve())
    reference = _within(root, Path(reference_directory).resolve())
    _need(directory != run and directory.parent == run and not directory.exists() and not run.is_relative_to(reference.parent)
          and not reference.is_relative_to(run),
          'Fresh replay must preserve the old run and use a separate new output')
    context, phase, old_identity = _reference_context(root, reference,
        audit_sha256=reference_audit_sha256, phase_sha256=reference_phase_sha256)
    _need(identity != context['root_identity_sha256'], 'Source correction requires a fresh root identity')
    return directory, context, phase, old_identity


def _corrected_sources(root, run, identity, main_identity, manifest_pin, pins):
    _need(isinstance(manifest_pin, dict)
          and main_identity.get('source_correction_manifest_pin') == manifest_pin,
          'New root must explicitly pin its exact corrected-annual source manifest')
    path = _verify_pin(root, manifest_pin)
    _need(path.stat().st_size <= 256 * 1024, 'Corrected source manifest exceeds metadata budget')
    manifest = _json(path)
    entries = manifest.get('years')
    _need(manifest.get('status') == 'VERIFIED_INPUT_REFERENCES_NOT_MODEL_RESULTS'
          and manifest.get('context_policy') == 'available_cohort_80pct'
          and manifest.get('financial_2024_plus_analysis') is False
          and isinstance(entries, list) and len(entries) == 14
          and [x.get('year') for x in entries] == list(range(2010, 2024))
          and all(type(x.get('corrected')) is bool for x in entries)
          and {x['year'] for x in entries if x['corrected']} == {2010, 2011, 2012, 2013, 2022, 2023},
          'Source manifest lost its reviewed six-year correction and fourteen-year scope')
    pins.append(manifest_pin)
    annual_bindings = []
    for item in entries:
        source = _within(root, root / item['directory'])
        audit_path = source / 'audit.json'
        _need(audit_path.is_file() and _hash(audit_path) == item['audit_sha256'],
              'Corrected/retained annual audit pin changed')
        source_audit = _json(audit_path)
        _need(source_audit.get('status') == 'COMPLETE' and source_audit.get('year') == item['year']
              and source_audit.get('build_identity_sha256') == item['build_identity_sha256'],
              'Corrected/retained annual identity is not its reviewed input reference')
        pins.append(_pin(root, audit_path))
        if item['year'] < 2020: continue
        detached = run / 'years' / str(item['year'])
        detached_audit = _json(detached / 'audit.json')
        _need(detached_audit.get('status') == 'COMPLETE' and detached_audit.get('year') == item['year']
              and detached_audit.get('build_identity_sha256') == identity,
              'New execution annual must explicitly bind the new root identity')
        expected, actual = source_audit.get('artifacts', {}), detached_audit.get('artifacts', {})
        _need(expected and set(actual) == set(expected) | {'reference_annual_audit'},
              'Detached annual artifact scope differs beyond the explicit source-audit copy')
        reference_copy = _verify_pin(detached, actual['reference_annual_audit'])
        _need(actual['reference_annual_audit'].get('path') == 'reference_annual_audit.json'
              and reference_copy == detached / 'reference_annual_audit.json'
              and actual['reference_annual_audit']['sha256'] == item['audit_sha256']
              and actual['reference_annual_audit']['size_bytes'] == audit_path.stat().st_size,
              'Detached annual source-audit copy does not match the reviewed source')
        expected_lineage = {'directory': item['directory'], 'audit_pin': _pin(root, audit_path),
            'build_identity_sha256': item['build_identity_sha256'], 'corrected': item['corrected'],
            'original_rows': item['original_rows'], 'numerical_artifact_bytes_unchanged': True,
            'new_feature_calculations': 0, 'original_audit_preserved_as': 'reference_annual_audit.json'}
        _need(detached_audit.get('source_corrected_replay_lineage') == expected_lineage,
              'Detached annual must explicitly preserve its source-corrected lineage')
        for name in expected:
            _need(expected[name] == actual[name],
                  'Detached execution/feature artifact differs from its reviewed annual: ' + name)
        annual_bindings.append({'year': item['year'], 'original_audit_pin': _pin(root, audit_path),
                                'detached_audit_pin': _pin(root, detached / 'audit.json'),
                                'all_artifact_bytes_match_reviewed_source': True})
    return {'manifest_pin': manifest_pin, 'annual_bindings': annual_bindings,
            'original_annual_sources_unchanged': True}


def _rebind_context(reference, new_identity):
    return {'schema': REBIND_SCHEMA, 'original_root_identity_sha256': reference['root_identity_sha256'],
            'new_root_identity_sha256': new_identity, 'original_reference_audit_pin': reference['audit_pin'],
            'original_reference_phase_identity_sha256': reference['phase_identity_sha256'],
            'economic_terms_and_dated_handover_conditions_unchanged': True,
            'original_evidence_bytes_unchanged': True, 'source_identity_promoted': False,
            'authority': 'Explicit user authorization to refit all four arms and replay corrected inputs under unchanged rules.'}


def build_rebound_execution_evidence(root, run_directory, identity_sha256, *,
        reference_directory, reference_audit_sha256, reference_phase_sha256, output_directory):
    """Copy original approved metadata; explicitly rebind only the new root.

    Original decisions/evidence remain immutable. This does not approve a new
    identity transition or merger, and reads no forecast probability or price.
    """
    root = Path(root).resolve(); run = _within(root, Path(run_directory).resolve())
    directory = _within(root / 'data/derived', Path(output_directory).resolve())
    _need(not directory.exists(), 'Preserve every existing registry rebind package')
    identity = _json(run / 'build_identity.json')
    _need(canonical_sha256(identity) == identity_sha256, 'New root identity pin changed before rebind')
    reference, phase, old_identity = _reference_context(root, reference_directory,
        audit_sha256=reference_audit_sha256, phase_sha256=reference_phase_sha256)
    _need(identity_sha256 != reference['root_identity_sha256']
          and identity.get('configuration') == old_identity.get('configuration'),
          'Rebind must target a fresh source-only correction under the identical protocol')
    binding = _rebind_context(reference, identity_sha256)
    old_registry_path = root / phase['identity_handover_registry_pin']['path']
    old_package = _json(root / phase['identity_handover_approval']['package_receipt_pin']['path'])
    old_decision = _json(root / phase['identity_handover_approval']['decision_pin']['path'])
    registry_name = old_registry_path.name
    _need(registry_name == 'identity_handover_registry.json', 'Unexpected original handover registry basename')
    # Authenticate all source evidence before creating the additive output.
    copies = []
    for item in old_package['files']:
        relative = Path(item['path'])
        _need(not relative.is_absolute() and '..' not in relative.parts,
              'Old approved evidence package contains unsafe paths')
        source = _within(old_registry_path.parent, old_registry_path.parent / relative)
        _need(_hash(source) == item['sha256'] and source.stat().st_size == item['bytes'],
              'Original approval package evidence bytes changed')
        if item['path'] != registry_name: copies.append((source, relative))
    directory.mkdir(parents=True)
    for source, relative in copies:
        target = directory / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    registry = deepcopy(phase['identity_handover_registry'])
    registry['root_build_identity_sha256'] = identity_sha256
    registry['source_corrected_rebind'] = binding
    new_registry_path = directory / registry_name
    _write_json(new_registry_path, registry)
    package = deepcopy(old_package)
    package['source_corrected_rebind'] = binding
    package['original_package_receipt_pin'] = phase['identity_handover_approval']['package_receipt_pin']
    for item in package['files']:
        target = directory / item['path']
        item.update(sha256=_hash(target), bytes=target.stat().st_size)
    package_path = directory / 'package_receipt.json'; _write_json(package_path, package)
    decision = deepcopy(old_decision)
    decision.update(root_build_identity_sha256=identity_sha256,
                    registry_sha256=_hash(new_registry_path),
                    evidence_package_receipt_sha256=_hash(package_path),
                    source_corrected_rebind=binding,
                    original_approval_decision_pin=phase['identity_handover_approval']['decision_pin'])
    decision_path = directory / 'decision_record.json'; _write_json(decision_path, decision)
    events = deepcopy(phase['event_registry'])
    events.update(build_identity_sha256=identity_sha256, source_corrected_rebind=binding)
    event_path = directory / 'event_registry.json'; _write_json(event_path, events)
    paths = {'event_registry': event_path, 'identity_handover_registry': new_registry_path,
             'identity_handover_decision': decision_path}
    receipt = {'schema': REBIND_SCHEMA, 'status': 'COMPLETE_EXACT_EVIDENCE_REBIND_NOT_NEW_IDENTITY_APPROVAL',
               'binding': binding, 'original_reference': reference,
               'files': {k: _pin(root, v) for k, v in paths.items()},
               'original_evidence_preserved': True, 'financial_values_read': False,
               'new_identity_or_corporate_action_judgments': 0}
    _write_json(directory / 'rebind_receipt.json', receipt)
    return paths


def _validate_rebound_execution(reference, old_phase, identity, registry, handover, handover_approval, root):
    binding = _rebind_context(reference, identity)
    expected_events = deepcopy(old_phase['event_registry'])
    expected_events.update(build_identity_sha256=identity, source_corrected_rebind=binding)
    expected_handover = deepcopy(old_phase['identity_handover_registry'])
    expected_handover.update(root_build_identity_sha256=identity, source_corrected_rebind=binding)
    _need(registry == expected_events and handover == expected_handover,
          'Fresh registry must explicitly rebind exactly the original events, handover and evidence')
    _need(handover_approval is not None, 'Fresh handover requires its exactly rebound approval packet')
    decision = _json(root / handover_approval['decision_pin']['path'])
    package = _json(root / handover_approval['package_receipt_pin']['path'])
    old_approval = old_phase['identity_handover_approval']
    expected_decision = deepcopy(_json(root / old_approval['decision_pin']['path']))
    expected_decision.update(root_build_identity_sha256=identity,
        registry_sha256=decision['registry_sha256'],
        evidence_package_receipt_sha256=handover_approval['package_receipt_pin']['sha256'],
        source_corrected_rebind=binding, original_approval_decision_pin=old_approval['decision_pin'])
    expected_package = deepcopy(_json(root / old_approval['package_receipt_pin']['path']))
    expected_package.update(source_corrected_rebind=binding,
                            original_package_receipt_pin=old_approval['package_receipt_pin'])
    for item in expected_package['files']:
        if item['path'] == 'identity_handover_registry.json':
            item.update(sha256=decision['registry_sha256'],
                        bytes=(root / handover_approval['decision_pin']['path']).parent.joinpath(item['path']).stat().st_size)
    _need(decision == expected_decision and package == expected_package,
          'Fresh execution approval must retain the exact original decision and evidence package')


def run_source_corrected_backtest(root, run_directory, configuration, identity_sha256, *,
        reference_directory, reference_audit_sha256, reference_phase_sha256, source_correction_manifest_pin,
        event_registry=None, output_directory=None,
        identity_handover_registry=None, identity_handover_decision=None,
        preflight_only=False, allow_withheld_paths=False):
    """Prepare new selections/coverage first, then run the unchanged 64 paths.

    Missing key/source-quality/zero-breadth evidence blocks performance by default.
    An explicitly reviewed ``allow_withheld_paths=True`` retains those exposures
    and lets the engine record failed paths; it never changes their selections.
    A changed action registry produces a new phase identity and must use a new
    output_directory if an older numerical attempt already exists.
    """
    if type(preflight_only) is not bool or type(allow_withheld_paths) is not bool:
        raise DataContractError('Explicit boolean preflight/withheld-path scope required')
    _need(isinstance(event_registry, (str, Path))
          and isinstance(identity_handover_registry, (str, Path))
          and isinstance(identity_handover_decision, (str, Path)),
          'Fresh replay requires explicitly raw-pinned rebound event/handover/decision files')
    root = Path(root).resolve(); run = _within(root, Path(run_directory).resolve())
    directory, reference_context, old_phase, old_identity = _fresh_scope(root, run, identity_sha256,
        output_directory=output_directory, reference_directory=reference_directory,
        reference_audit_sha256=reference_audit_sha256, reference_phase_sha256=reference_phase_sha256)
    cfg = _configuration(configuration); started = time.monotonic(); peak = _rss()
    def check_resources():
        nonlocal peak
        peak=max(peak,_rss())
        if (time.monotonic()-started>cfg['execution']['backtest_maximum_wall_seconds']
                or peak>cfg['execution']['backtest_maximum_peak_rss_bytes']):
            raise DataContractError('Walk-forward backtest exceeds its frozen resource ceiling')
    main_identity = _json(run / 'build_identity.json')
    if canonical_sha256(main_identity) != identity_sha256 or main_identity.get('configuration') != cfg:
        raise DataContractError('Backtest configuration differs from its exact parent build identity')
    _need(main_identity.get('configuration') == old_identity.get('configuration'), 'Source correction must preserve the exact original research protocol')
    pins = [_pin(root, run / 'build_identity.json'), *reference_context['immutable_source_pins']]
    source_corrections = _corrected_sources(root, run, identity_sha256, main_identity, source_correction_manifest_pin, pins)
    cfg_path = root / 'configs/walkforward_2020_2023.json'
    if _json(cfg_path) != cfg: raise DataContractError('Saved walk-forward configuration changed')
    pins.append(_pin(root, cfg_path))
    sessions, windows, clock = walkforward_clock()
    handover_manifest,handover_pin,identity_handovers,handover_approval=_identity_registry(root,
        identity_handover_registry,identity_sha256,sessions,
        dict(zip(pd.to_datetime(windows.execution_date),pd.to_datetime(windows.signal_date))),pins,
        decision_path=identity_handover_decision)
    historical = _historical_registry(root, cfg, pins)
    _need(historical == old_phase.get('historical_retirements'), 'Historical retirement terms changed from original reference')
    registry = {'schema': 'dmv2-walkforward-actions/v1', 'events': []} if event_registry is None else event_registry
    if isinstance(registry, (str, Path)):
        path = _within(root, Path(registry).resolve()); registry = _json(path); pins.append(_pin(root, path))
    if not isinstance(registry, dict) or registry.get('schema') != 'dmv2-walkforward-actions/v1' or not isinstance(registry.get('events'), list):
        raise DataContractError('Explicit independently reviewed action-registry manifest required')
    for item in registry.get('source_pins', []):
        _verify_pin(root,item); pins.append(item)
    _validate_rebound_execution(reference_context, old_phase, identity_sha256, registry,
        handover_manifest, handover_approval, root)
    events = registry['events']; validate_current_events(events, sessions)
    prediction_plan, prediction_bytes = _prediction_plan(root, run, identity_sha256, windows, pins)
    annual_plan, metadata, read_plan = _price_plan(root, run, identity_sha256, pins)
    supplement_paths,supplement_metadata,supplement_stats=_supplement_plan(root,registry,metadata,
        read_plan['original_requested_native_ids'],sessions,pins)
    metadata=pd.concat([metadata,supplement_metadata],ignore_index=True)
    read_plan['supplemental_execution_quotes']=supplement_stats
    if read_plan['projected_compressed_bytes'] + prediction_bytes + supplement_stats['projected_compressed_bytes'] > MAX_READ_BYTES:
        raise DataContractError('Combined execution/forecast projections exceed the explicit read budget')
    if not metadata.trading_date.isin(sessions).all():
        raise DataContractError('Execution metadata contains non-XNYS dates')
    check_resources()
    selections, preflights, validations = {}, {}, []
    for arm, (path, columns) in prediction_plan.items():
        forecasts = pq.ParquetFile(path).read(columns=list(dict.fromkeys([*columns, *PROBABILITIES]))).to_pandas()
        for basket in cfg['backtest']['baskets']:
            selected = select_walkforward_basket(forecasts, basket=basket, sessions=clock['selection_sessions'], configuration=cfg)
            check = validate_walkforward_basket(selected, forecasts, basket=basket, sessions=clock['selection_sessions'], configuration=cfg)
            if len(selected.monthly) != 48: raise DataContractError('Selection lost a walk-forward prediction month')
            selections[(arm, basket)] = selected
            validations.append({'arm': arm, 'basket': basket, **{k:v for k,v in check.items() if k != 'policy'}})
            preflights[(arm, basket)] = preflight_selected_horizons(selected.selections, selected.rebalances, metadata,
                sessions, events=events, historical_retirements=historical)
            check_resources()
    phase = {'schema_version': 1, 'phase': PHASE, 'parent_identity_sha256': identity_sha256,
        'configuration_sha256': canonical_sha256(cfg), 'source_pins': pins, 'clock': clock,
        'implementation_revision': IMPLEMENTATION_REVISION, 'fresh_source_replay': SOURCE_REPLAY, 'original_completed_reference': reference_context,
        'source_corrections': source_corrections, 'orchestration_worker_sha256': _hash(Path(__file__)),
        'identity_handover_registry':handover_manifest, 'identity_handover_registry_pin':handover_pin,
        'identity_handover_approval':handover_approval,
        'event_registry': registry, 'historical_retirements': historical,
        'models_fitted': False, 'preprocessing_fitted': False,
        'basket_policy': BALANCED_BREADTH_POLICY, 'read_plan': read_plan,
        'allow_withheld_paths_after_explicit_review': allow_withheld_paths,
        'workers': {name: _hash(Path(find_spec('dmv2.'+name[:-3]).origin)) for name in WORKERS},
        'runtime': {'python': platform.python_version(), **{name:version(name) for name in ('numpy','pandas','pyarrow','exchange-calendars')}}}
    phase_pin = canonical_sha256(phase)
    preflight_dir = run / 'preflight' / phase_pin[7:23]
    if not preflight_dir.exists():
        preflight_dir.mkdir(parents=True); artifacts = {}
        _write_json(preflight_dir / 'build_identity.json', phase)
        artifacts['build_identity.json'] = _pin(preflight_dir, preflight_dir / 'build_identity.json')
        summaries = []
        for (arm, basket), item in preflights.items():
            selected = selections[(arm, basket)]
            for name, frame in item.items():
                if name != 'summary': _save_table(preflight_dir, f'{arm}/{basket}/{name}.parquet', frame, artifacts)
            for name, frame in (('original_ledger', selected.ledger), ('selected', selected.selections),
                    ('rebalances', selected.rebalances), ('monthly', selected.monthly)):
                _save_table(preflight_dir, f'{arm}/{basket}/{name}.parquet', frame, artifacts)
            summaries.append({'arm':arm,'basket':basket,**item['summary'],
                'zero_breadth_months': int(selected.monthly.selected_names_per_side.eq(0).sum())})
        _save_table(preflight_dir, 'summary.parquet', pd.DataFrame(summaries), artifacts)
        _write_json(preflight_dir / 'audit.json', {'status':'COMPLETE','phase':'walkforward_selected_horizon_preflight_v1',
            'build_identity_sha256':identity_sha256,'phase_identity_sha256':phase_pin,'artifacts':artifacts,
            'selection_changed':False,'performance_computed':False,'maximum_horizons_reviewed':True})
    else:
        saved = _json(preflight_dir / 'audit.json')
        if saved.get('phase_identity_sha256') != phase_pin or saved.get('status') != 'COMPLETE':
            raise DataContractError('Preserve the incomplete or stale preflight attempt')
        for item in saved['artifacts'].values(): _verify_pin(preflight_dir, item)
    if preflight_only: return preflight_dir
    unresolved = any(not item['summary']['ready_for_accounting'] for item in preflights.values())
    zero_breadth = any(selection.monthly.selected_names_per_side.eq(0).any() for selection in selections.values())
    if (unresolved or zero_breadth) and not allow_withheld_paths:
        raise DataContractError('Pre-performance coverage requires repair/review; frozen selections and all gaps saved at '+str(preflight_dir))
    if directory.exists():
        raise DataContractError('Preserve every existing numerical attempt; Fresh replay requires a new output directory')
    directory.mkdir(parents=True); artifacts = {}
    _write_json(directory / 'build_identity.json', phase)
    artifacts['build_identity.json'] = _pin(directory, directory / 'build_identity.json')
    _write_json(directory / 'audit.json', {'status':'STARTED','phase':PHASE,'build_identity_sha256':identity_sha256,'phase_identity_sha256':phase_pin})
    prices = pd.concat([pq.ParquetFile(path).read(columns=list(ACTION_PRICE_COLUMNS)).to_pandas()
        for path in [*annual_plan.values(),*supplement_paths]], ignore_index=True)
    if not prices[['trading_date','intrinio_security_id']].equals(metadata[['trading_date','intrinio_security_id']]):
        raise DataContractError('Execution keys changed between metadata and financial projections')
    comparison, yearly, failures, actions, orders = [], [], [], [], []
    try:
        for (arm, basket), selected in selections.items():
            for name, frame in (('original_ledger',selected.ledger),('selected',selected.selections),
                    ('rebalances',selected.rebalances),('monthly',selected.monthly)):
                _save_table(directory, f'selections/{arm}/{basket}/{name}.parquet', frame, artifacts)
            for variant, stop in cfg['backtest']['variants'].items():
                for cost in cfg['backtest']['cost_bps']:
                    check_resources()
                    result = run_walkforward_accounting(selected.selections, prices, sessions,
                        rebalances=selected.rebalances, terminal_date='2023-12-29', events=events,
                        historical_retirements=historical, identity_handovers=identity_handovers,
                        policy=DailyAccountingPolicy(cost_bps=cost,
                            borrow_annual_bps=0, required_names_per_side=None, asset_stop_fraction=stop))
                    row, monthly = summarize_daily(result, arm=arm, basket=basket, variant=variant,
                        cost_bps=cost, expected_sessions=1006)
                    if len(monthly)!=48: raise DataContractError('Continuous accounting lost a monthly return')
                    row.update(outcome_start='2020-01-01',outcome_end='2023-12-31',outcome_months=48,
                        terminal_date='2023-12-29',annual_liquidation=False,annual_model_fits=4)
                    comparison.append(row)
                    yearly.append(yearly_breakdown(result,arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    relative=f'arms/{arm}/{basket}/{variant}/cost_{cost:02d}bps'
                    for name, table in result.tables.items(): _save_table(directory,relative+'/'+name+'.parquet',table,artifacts)
                    _save_table(directory,relative+'/monthly.parquet',monthly,artifacts)
                    if not result.failures.empty: failures.append(result.failures.assign(arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    if not result.action_events.empty: actions.append(result.action_events.assign(arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    if not result.orders.empty: orders.append(result.orders.assign(arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    print(json.dumps({'phase':PHASE,'arm':arm,'basket':basket,'variant':variant,'cost_bps':cost,
                        'status':result.status,'completed_scenarios':len(comparison)}),flush=True)
                    if sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) > cfg['execution']['maximum_output_bytes']:
                        raise DataContractError('Walk-forward output budget exceeded')
        reports = {'comparison':pd.DataFrame(comparison),'yearly':pd.concat(yearly,ignore_index=True),
            'selection_coverage':pd.concat([value.monthly.assign(arm=arm,basket=basket)
                for (arm,basket),value in selections.items()],ignore_index=True),
            'selection_validation':pd.DataFrame(validations),
            'failures':pd.concat(failures,ignore_index=True) if failures else pd.DataFrame(columns=['date','reason','arm','basket','variant','cost_bps']),
            'action_coverage':pd.concat(actions,ignore_index=True) if actions else pd.DataFrame(columns=['event_id','arm','basket','variant','cost_bps']),
            'order_coverage':pd.concat(orders,ignore_index=True) if orders else pd.DataFrame(columns=['status','arm','basket','variant','cost_bps'])}
        for name,frame in reports.items(): _save_table(directory,name+'.parquet',frame,artifacts)
        for item in pins: _verify_pin(root,item)
        check_resources()
        _write_json(directory/'audit.json',{'status':'COMPLETE','phase':PHASE,'build_identity_sha256':identity_sha256,
            'phase_identity_sha256':phase_pin,'artifacts':artifacts,'scenarios':64,'sessions':1006,'outcome_months':48,
            'implementation_revision':IMPLEMENTATION_REVISION,'old_completed_reference_preserved':True, 'fresh_source_replay':SOURCE_REPLAY,
            'provisional_execution_handover_registry_used':handover_manifest is not None,
            'source_identity_promoted':False,
            'complete_proxy_paths':sum(item['complete_proxy_path'] for item in comparison),
            'failed_closed_paths':sum(not item['complete_proxy_path'] for item in comparison),
            'models_fitted':False,'preprocessing_fitted':False,'financial_2024_plus_read':False,
            'source_pins_unchanged':True,'original_pointers_changed':False,'annual_liquidation':False,
            'RET':'not_evaluated','execution_seconds':time.monotonic()-started,'peak_rss_bytes':max(peak,_rss()),
            'preflight_directory':str(preflight_dir.relative_to(run)), 'preflight_before_performance':True})
    except Exception as error:
        _write_json(directory/'failure.json',{'exception':type(error).__name__,'message':str(error)}); raise
    return directory


def read_source_corrected_backtest(directory, *, root, identity_sha256=None,
                                  phase_identity_sha256=None):
    """Passive strict reader for the new replay; no fits, sources or accounting."""
    root = Path(root).resolve(); directory = _within(root, Path(directory).resolve())
    phase, audit = _json(directory / 'build_identity.json'), _json(directory / 'audit.json')
    _need(phase.get('fresh_source_replay') == SOURCE_REPLAY
          and audit.get('fresh_source_replay') == SOURCE_REPLAY
          and phase.get('implementation_revision') == IMPLEMENTATION_REVISION
          and audit.get('implementation_revision') == IMPLEMENTATION_REVISION
          and audit.get('old_completed_reference_preserved') is True
          and 'prior_failed_attempt' not in phase
          and phase.get('orchestration_worker_sha256') == _hash(Path(__file__)),
          'Saved source-corrected replay lost its fresh-run/code/preservation binding')
    reference = phase.get('original_completed_reference', {})
    _need(reference.get('fresh_run_requires_no_prior_failed_attempt') is True,
          'Saved fresh replay lost its original completed reference')
    checked, old_phase, old_identity = _reference_context(root, root / reference['directory'],
        audit_sha256=reference['audit_pin']['sha256'], phase_sha256=reference['phase_identity_sha256'])
    _need(checked == reference and all(pin in phase['source_pins'] for pin in checked['immutable_source_pins']),
          'Saved reference metadata/evidence lost its exact phase source pins')
    run = directory.parent; main_identity = _json(run / 'build_identity.json')
    identity = canonical_sha256(main_identity)
    _need(identity == phase.get('parent_identity_sha256')
          and main_identity.get('configuration') == old_identity.get('configuration'),
          'Saved corrected root differs from its unchanged parent protocol')
    source_binding = _corrected_sources(root, run, identity, main_identity,
        phase['source_corrections']['manifest_pin'], [])
    _need(source_binding == phase['source_corrections'], 'Saved corrected source bindings changed')
    sessions, windows, _ = walkforward_clock()
    approval = phase['identity_handover_approval']
    manifest, regpin, records, checked_approval = _identity_registry(root,
        root / phase['identity_handover_registry_pin']['path'], identity, sessions,
        dict(zip(pd.to_datetime(windows.execution_date), pd.to_datetime(windows.signal_date))), [],
        decision_path=root / approval['decision_pin']['path'])
    _need(manifest == phase['identity_handover_registry'] and checked_approval == approval,
          'Saved rebound handover approval changed')
    _validate_rebound_execution(reference, old_phase, identity, phase['event_registry'], manifest, approval, root)
    expected = {f'arms/{arm}/{basket}/{variant}/cost_{cost:02d}bps/identity_handovers.parquet'
        for arm in ARMS for basket in ('class_1_10', 'class_2_9')
        for variant in ('unstopped', 'asset_stop_10pct') for cost in (0, 5, 10, 25)}
    _need(expected <= set(audit.get('artifacts', {})), 'Saved corrected replay omits a declared handover ledger')
    return read_walkforward_backtest(directory, identity_sha256=identity_sha256,
        phase_identity_sha256=phase_identity_sha256, root=root)

