"""Additive strict identity-failure rerun; preserves the failed V1 attempt.

The unchanged configuration, source panels, 16 saved fits, predictions,
selection rules and 64-scenario arithmetic are reused. No fitting or fetching.
An explicit new output directory and hash-pinned failed prior attempt are
required before any forecast or execution-price projection.
"""
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import json
import platform
import time
import pandas as pd
import pyarrow.parquet as pq
from .walkforward_backtest import (
    PHASE, WORKERS as V1_WORKERS, REPORTS, MAX_READ_BYTES,
    _configuration, _historical_registry, _prediction_plan, _price_plan,
    _supplement_plan, yearly_breakdown, read_walkforward_backtest,
    ACTION_PRICE_COLUMNS, DailyAccountingPolicy, ARMS, _hash, _json, _pin,
    _verify_pin, _within, PROBABILITIES, canonical_sha256, _rss, _save_table,
    _write_json, summarize_daily, DataContractError, walkforward_clock,
    validate_current_events, select_walkforward_basket, validate_walkforward_basket,
    preflight_selected_horizons, BALANCED_BREADTH_POLICY,
)
from .walkforward_accounting_v2 import (
    IMPLEMENTATION_REVISION, _prepare_handovers, run_walkforward_accounting_v2 as run_walkforward_accounting,
)

WORKERS = (*V1_WORKERS, 'walkforward_accounting_v2.py', 'walkforward_backtest_v2.py')
MAX_PRIOR_ATTEMPT_BYTES = 256 * 1024**2
MAX_PRIOR_ATTEMPT_FILES = 512


def _attempt_scope(root, run, identity_sha256, *, output_directory, failed_attempt_directory):
    root=Path(root).resolve();run=_within(root,Path(run).resolve())
    if output_directory is None or failed_attempt_directory is None:
        raise DataContractError('V2 requires explicit new output and immutable failed-attempt directories')
    directory = _within(run, Path(output_directory).resolve())
    prior = _within(run, Path(failed_attempt_directory).resolve())
    if (directory == run or prior == run or directory == prior
            or directory.is_relative_to(prior) or prior.is_relative_to(directory)
            or directory.exists()):
        raise DataContractError('V2 must preserve its failed prior attempt and use a fresh separate output directory')
    audit = _json(prior/'audit.json'); failure = _json(prior/'failure.json')
    phase = _json(prior/'build_identity.json')
    if (audit.get('status') != 'STARTED' or audit.get('phase') != PHASE
            or audit.get('build_identity_sha256') != identity_sha256
            or audit.get('phase_identity_sha256') != canonical_sha256(phase)
            or phase.get('parent_identity_sha256') != identity_sha256
            or phase.get('models_fitted') is not False or phase.get('preprocessing_fitted') is not False
            or phase.get('implementation_revision') is not None
            or failure.get('exception') != 'DataContractError'
            or failure.get('message') not in {'Held identity route changed',
                'Selected provider collides with a different held source identity'}):
        raise DataContractError('V2 prior-attempt evidence is not the preserved V1 runtime identity failure')
    children=list(prior.rglob('*'))
    if any(path.is_symlink() for path in children):
        raise DataContractError('Prior failed-attempt preservation cannot traverse symlinks')
    paths = sorted(path for path in children if path.is_file())
    if (len(paths) > MAX_PRIOR_ATTEMPT_FILES
            or sum(path.stat().st_size for path in paths) > MAX_PRIOR_ATTEMPT_BYTES
            or any(path.is_symlink() or not path.resolve().is_relative_to(prior) for path in paths)):
        raise DataContractError('Prior failed-attempt preservation exceeds its exact bounded scope')
    pins = [_pin(root,path) for path in paths]
    context = {'directory':str(prior.relative_to(root)), 'file_count':len(pins),
        'all_existing_files_pinned':True,'prior_phase_identity_sha256':audit['phase_identity_sha256'],
        'failure':failure, 'immutable_source_pins':pins}
    return directory, context, pins


def _identity_registry(root, registry, identity_sha256, sessions, schedule, pins, *, decision_path=None):
    root=Path(root).resolve()
    if registry is None:
        if decision_path is not None:
            raise DataContractError('Execution approval cannot exist without its explicit handover registry')
        return None, None, [], None
    if not isinstance(registry,(str,Path)):
        raise DataContractError('Provisional execution handovers require a raw-pinned registry file')
    path=_within(root,Path(registry).resolve())
    if not path.is_file() or path.stat().st_size>1024**2:
        raise DataContractError('Execution handover registry must be an existing bounded JSON file')
    manifest=_json(path)
    if (manifest.get('schema_version')!=1 or type(manifest.get('schema_version')) is not int
            or not isinstance(manifest.get('registry_id'),str) or not manifest['registry_id'].strip()
            or manifest.get('status')!='PROVISIONAL_EXECUTION_ONLY'
            or manifest.get('root_build_identity_sha256')!=identity_sha256
            or manifest.get('source_identity_promoted') is not False
            or not isinstance(manifest.get('handovers'),list) or len(manifest['handovers'])!=1
            or not isinstance(manifest.get('evidence_pins'),list) or not manifest['evidence_pins']):
        raise DataContractError('Execution registry must explicitly bind one reviewed provisional handover to this root')
    records=_prepare_handovers(manifest['handovers'],pd.DatetimeIndex(pd.to_datetime(sessions)),schedule)
    registry_pin=_pin(root,path); pins.append(registry_pin)
    if len(manifest['evidence_pins'])>64:
        raise DataContractError('Execution handover evidence exceeds 64 pinned metadata artifacts')
    seen=set(); total=0
    for declared in manifest['evidence_pins']:
        if not isinstance(declared,dict) or declared.get('base') not in {'root','registry_directory'}:
            raise DataContractError('Execution handover evidence requires an explicit approved path base')
        relative=Path(declared.get('path',''))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise DataContractError('Execution handover evidence path must be safe and relative')
        base=root if declared['base']=='root' else path.parent
        evidence=_within(base,(base/relative).resolve())
        if evidence.is_symlink() or not evidence.is_file() or evidence.stat().st_size>4*1024**2:
            raise DataContractError('Execution handover evidence must be a bounded existing metadata artifact')
        actual=_pin(root,evidence)
        if actual['sha256']!=declared.get('sha256') or actual['path'] in seen:
            raise DataContractError('Execution handover evidence raw pin changed or repeats')
        seen.add(actual['path']);total+=actual['size_bytes'];pins.append(actual)
    if total>16*1024**2:
        raise DataContractError('Execution handover evidence exceeds 16 MiB')
    approval=_approval_binding(root,path,registry_pin,manifest,decision_path,pins)
    return manifest,registry_pin,records,approval


def _approval_binding(root, registry_path, registry_pin, manifest, decision_path, pins):
    decision=_within(root,Path(decision_path).resolve()) if decision_path is not None else registry_path.parent/'decision_record.json'
    receipt=registry_path.parent/'package_receipt.json'
    for path in (decision,receipt):
        if not path.is_file() or path.stat().st_size>1024**2:
            raise DataContractError('Exact execution handover approval and package receipt are required')
    decision_pin=_pin(root,decision);receipt_pin=_pin(root,receipt)
    value=_json(decision);package=_json(receipt)
    transition_names=('effective_date','held_signal_date','intrinio_security_id',
        'new_source_security_id','old_source_security_id','signal_date')
    allowed={name:manifest['handovers'][0][name] for name in transition_names}
    identity_status={'execution_evidence_reviewed':True,'provisionally_admitted':True,
        'source_identity_promoted':False,'source_identity_verified':False,
        'source_review_flags_retained':True,'unified_ids_remain_distinct':True}
    if (value.get('schema_version')!=1 or type(value.get('schema_version')) is not int
            or value.get('status')!='APPROVED_FOR_SEPARATE_ACCOUNTING_V2_ONLY'
            or value.get('registry_sha256')!=registry_pin['sha256']
            or value.get('evidence_package_receipt_sha256')!=receipt_pin['sha256']
            or value.get('allowed_transition')!=allowed or value.get('identity_status')!=identity_status
            or any(type(x) is not bool for x in value.get('identity_status',{}).values())
            or value.get('root_build_identity_sha256',manifest['root_build_identity_sha256'])!=manifest['root_build_identity_sha256']
            or package.get('schema_version')!=1 or type(package.get('schema_version')) is not int
            or not isinstance(package.get('files'),list) or not 1<=len(package['files'])<=64):
        raise DataContractError('Execution handover decision does not approve this exact provisional registry and evidence packet')
    seen=set();total=0;registry_bound=False
    for item in package['files']:
        relative=Path(item.get('path',''))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise DataContractError('Execution approval package paths must be safe relative metadata files')
        path=_within(registry_path.parent,registry_path.parent/relative)
        pin=_pin(root,path)
        if (pin['sha256']!=item.get('sha256') or pin['size_bytes']!=item.get('bytes')
                or pin['size_bytes']>4*1024**2 or pin['path'] in seen):
            raise DataContractError('Execution approval package bytes changed or repeat')
        seen.add(pin['path']);total+=pin['size_bytes']
        registry_bound=registry_bound or pin==registry_pin
        if pin not in pins:pins.append(pin)
    if not registry_bound or total>16*1024**2:
        raise DataContractError('Execution approval packet omits its exact registry or exceeds 16 MiB')
    pins.extend([decision_pin,receipt_pin])
    return {'decision_pin':decision_pin,'package_receipt_pin':receipt_pin}


def run_walkforward_backtest_v2(root, run_directory, configuration, identity_sha256, *,
        event_registry=None, output_directory=None, failed_attempt_directory=None,
        identity_handover_registry=None, identity_handover_decision=None,
        preflight_only=False, allow_withheld_paths=False):
    """Prepare frozen selections/coverage first, then explicitly run 64 paths.

    Missing key/source-quality/zero-breadth evidence blocks performance by default.
    An explicitly reviewed ``allow_withheld_paths=True`` retains those exposures
    and lets the engine record failed paths; it never changes their selections.
    A changed action registry produces a new phase identity and must use a new
    output_directory if an older numerical attempt already exists.
    """
    if type(preflight_only) is not bool or type(allow_withheld_paths) is not bool:
        raise DataContractError('Explicit boolean preflight/withheld-path scope required')
    root = Path(root).resolve(); run = _within(root, Path(run_directory).resolve())
    directory, failed_context, failed_pins = _attempt_scope(root, run, identity_sha256,
        output_directory=output_directory, failed_attempt_directory=failed_attempt_directory)
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
    pins = [_pin(root, run / 'build_identity.json'), *failed_pins]
    cfg_path = root / 'configs/walkforward_2020_2023.json'
    if _json(cfg_path) != cfg: raise DataContractError('Saved walk-forward configuration changed')
    pins.append(_pin(root, cfg_path))
    sessions, windows, clock = walkforward_clock()
    handover_manifest,handover_pin,identity_handovers,handover_approval=_identity_registry(root,
        identity_handover_registry,identity_sha256,sessions,
        dict(zip(pd.to_datetime(windows.execution_date),pd.to_datetime(windows.signal_date))),pins,
        decision_path=identity_handover_decision)
    historical = _historical_registry(root, cfg, pins)
    registry = {'schema': 'dmv2-walkforward-actions/v1', 'events': []} if event_registry is None else event_registry
    if isinstance(registry, (str, Path)):
        path = _within(root, Path(registry).resolve()); registry = _json(path); pins.append(_pin(root, path))
    if not isinstance(registry, dict) or registry.get('schema') != 'dmv2-walkforward-actions/v1' or not isinstance(registry.get('events'), list):
        raise DataContractError('Explicit independently reviewed action-registry manifest required')
    for item in registry.get('source_pins', []):
        _verify_pin(root,item); pins.append(item)
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
        'implementation_revision': IMPLEMENTATION_REVISION, 'prior_failed_attempt': failed_context,
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
        raise DataContractError('Preserve every existing numerical attempt; V2 requires a new output directory')
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
            'implementation_revision':IMPLEMENTATION_REVISION,'prior_failed_attempt_preserved':True,
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



def read_walkforward_backtest_v2(directory, *, identity_sha256=None, phase_identity_sha256=None, root=None):
    """Authenticate the V2 phase before the unchanged saved-report reader."""
    directory=Path(directory).resolve(); phase=_json(directory/'build_identity.json'); audit=_json(directory/'audit.json')
    prior=phase.get('prior_failed_attempt', {})
    if (phase.get('implementation_revision') != IMPLEMENTATION_REVISION
            or audit.get('implementation_revision') != IMPLEMENTATION_REVISION
            or audit.get('prior_failed_attempt_preserved') is not True
            or prior.get('all_existing_files_pinned') is not True
            or not isinstance(prior.get('immutable_source_pins'),list)
            or prior.get('file_count') != len(prior['immutable_source_pins'])
            or not prior['immutable_source_pins']
            or not {'walkforward_accounting.py','walkforward_backtest.py',
                'walkforward_accounting_v2.py','walkforward_backtest_v2.py'} <= set(phase.get('workers',{}))):
        raise DataContractError('Saved V2 backtest lost its implementation or prior-attempt preservation binding')
    if any(item not in phase.get('source_pins',[]) for item in prior['immutable_source_pins']):
        raise DataContractError('Saved V2 prior-attempt files lost their phase source pins')
    registry=phase.get('identity_handover_registry');registry_pin=phase.get('identity_handover_registry_pin')
    approval=phase.get('identity_handover_approval')
    if ((registry is None)!=(registry_pin is None) or (registry is None)!=(approval is None)
            or audit.get('source_identity_promoted') is not False):
        raise DataContractError('Saved V2 provisional execution registry binding changed')
    if registry is not None and (registry_pin not in phase.get('source_pins',[])
            or registry.get('root_build_identity_sha256')!=phase.get('parent_identity_sha256')
            or registry.get('status')!='PROVISIONAL_EXECUTION_ONLY'
            or registry.get('source_identity_promoted') is not False):
        raise DataContractError('Saved V2 execution registry lost its raw pin or provisional scope')
    expected_tables={f'arms/{arm}/{basket}/{variant}/cost_{cost:02d}bps/identity_handovers.parquet'
        for arm in ARMS for basket in ('class_1_10','class_2_9')
        for variant in ('unstopped','asset_stop_10pct') for cost in (0,5,10,25)}
    if not expected_tables<=set(audit.get('artifacts',{})):
        raise DataContractError('Saved V2 handover ledgers omit a declared scenario')
    if root is not None:
        root=Path(root).resolve()
        for item in prior['immutable_source_pins']:_verify_pin(root,item)
        if registry is not None:
            raw=_verify_pin(root,registry_pin)
            if _json(raw)!=registry:
                raise DataContractError('Saved V2 registry body differs from its pinned raw file')
            if (not isinstance(approval,dict) or set(approval)!={'decision_pin','package_receipt_pin'}
                    or any(item not in phase['source_pins'] for item in approval.values())):
                raise DataContractError('Saved V2 execution approval lost its explicit phase source pins')
            for item in approval.values():_verify_pin(root,item)
            rebound=_approval_binding(root,raw,registry_pin,registry,root/approval['decision_pin']['path'],[])
            if rebound!=approval:
                raise DataContractError('Saved V2 execution approval changed its exact raw bindings')
            for item in registry['evidence_pins']:
                base=root if item['base']=='root' else raw.parent
                evidence=_within(base,base/item['path']);pin=_pin(root,evidence)
                if pin['sha256']!=item['sha256'] or pin not in phase['source_pins']:
                    raise DataContractError('Saved V2 execution evidence lost its exact phase pin')
    return read_walkforward_backtest(directory,identity_sha256=identity_sha256,
        phase_identity_sha256=phase_identity_sha256,root=root)
