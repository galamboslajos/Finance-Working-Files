"""Bounded saved views of the separately completed eleven-action backtest."""
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .independent_reporting import read_independent_run
from .fallback_reporting import _json, _pinned_json, _receipt, MAX_COMPARISON_BYTES
from .full_panel import _hash, _verify_artifacts, _within
from .universe import DataContractError
from .vvc_action import VVC_EVENT
from .tss_action import TSS_EVENT

ARMS = {'rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended'}


def read_completed_run(root, directory, *, identity_sha256, expected_receipts,
                        verify_sources=False):
    """Verify original and completed receipts without running any numerical work."""
    if (not isinstance(expected_receipts, dict)
            or set(expected_receipts) != {'features', 'models', 'backtest', 'original_backtest'}
            or any(not isinstance(value, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', value)
                for value in expected_receipts.values())):
        raise DataContractError('Completed saved report requires all four explicit final receipts')
    base_receipts = {key: expected_receipts[key] for key in ('features', 'models')}
    base_receipts['backtest'] = expected_receipts['original_backtest']
    saved = read_independent_run(root, directory, identity_sha256=identity_sha256,
        expected_receipts=base_receipts, verify_sources=verify_sources)
    root, run = Path(root).resolve(), saved['directory']
    complete = run / 'backtest_completed'
    _receipt(complete / 'audit.json', expected_receipts['backtest'], 'completed backtest')
    audit = _json(complete / 'audit.json')
    if (audit.get('status') != 'COMPLETE'
            or audit.get('phase') != 'independent_quote_action_completion_v1'
            or audit.get('build_identity_sha256') != identity_sha256
            or audit.get('scenarios') != 64 or audit.get('models_fitted') is not False
            or audit.get('financial_2020_plus_read') is not False):
        raise DataContractError('Completed backtest saved scope changed')
    phase = _pinned_json(complete, audit['artifacts']['build_identity.json'],
        expected_path=complete / 'build_identity.json')
    original_phase = _json(run / 'backtest/build_identity.json')
    if (canonical_sha256(phase) != audit.get('phase_identity_sha256')
            or phase.get('parent_identity_sha256') != identity_sha256
            or phase.get('phase') != 'independent_quote_action_completion_v1'
            or phase.get('models_fitted') is not False or phase.get('preprocessing_fitted') is not False
            or len(phase.get('events', [])) != 11
            or phase['events'][:9] != original_phase.get('events')
            or phase['events'][-2:] != [VVC_EVENT, TSS_EVENT]):
        raise DataContractError('Completed action phase or unchanged first nine events differs')
    configs = {}
    for name in ('independent_backtest.json', 'independent_backtest_completion.json'):
        path = root / 'configs' / name
        item = next((item for item in phase.get('source_pins', [])
            if item.get('path') == str(path.relative_to(root))), None)
        configs[name] = _pinned_json(root, item, expected_path=path)
    before, after = configs['independent_backtest.json'], configs['independent_backtest_completion.json']
    if (canonical_sha256(before) != phase.get('parent_configuration_sha256')
            or canonical_sha256(after) != phase.get('configuration_sha256')
            or after.get('parent_configuration_sha256') != canonical_sha256(before)
            or after.get('protocol_id') != 'dmv2-independent-quote-action-completion-2018-2019-v1'
            or _json(run / 'build_identity.json').get('configuration') != before):
        raise DataContractError('Completed action phase differs from its pinned original/completion configurations')
    for name in set(before) - {'protocol_id', 'authority', 'status', 'backtest'}:
        if before[name] != after.get(name):
            raise DataContractError('Completed configuration changed a frozen research rule')
    if ({k: v for k, v in before['backtest'].items() if k != 'events'}
            != {k: v for k, v in after['backtest'].items() if k != 'events'}):
        raise DataContractError('Completed configuration changed an execution rule beyond event completion')
    original_pin = next((item for item in phase.get('source_pins', [])
        if item.get('path') == str((run / 'backtest/audit.json').relative_to(root))), None)
    _pinned_json(root, original_pin, expected_path=run / 'backtest/audit.json')
    if 'sha256:' + original_pin['sha256'].removeprefix('sha256:') != expected_receipts['original_backtest']:
        raise DataContractError('Completed report refers to a different original quote-repair run')
    reports = {}
    for name, rows in (('comparison', 64), ('selection_coverage', 192)):
        item = audit['artifacts'][name + '.parquet']
        path = _within(complete, complete / item['path'])
        if path != complete / (name + '.parquet') or item['size_bytes'] > MAX_COMPARISON_BYTES:
            raise DataContractError('Completed saved report exceeds its bounded scope')
        _verify_artifacts(complete, {item['path']: item})
        footer = pq.ParquetFile(path)
        if footer.metadata.num_rows != rows:
            raise DataContractError('Completed saved report lost its full scenario/month grid')
        reports[name] = footer.read().to_pandas()
    comparison = reports['comparison']
    if (comparison.duplicated(['arm', 'basket', 'variant', 'cost_bps']).any()
            or set(comparison.arm) != ARMS or set(comparison.basket) != {'class_1_10', 'class_2_9'}
            or set(comparison.variant) != {'unstopped', 'asset_stop_10pct'}
            or set(comparison.cost_bps) != {0, 5, 10, 25}
            or not comparison.expected_sessions.eq(503).all()):
        raise DataContractError('Completed saved backtest scenario grid changed')
    coverage = reports['selection_coverage']
    needed = {'arm', 'basket', 'signal_date', 'long_candidates', 'short_candidates',
        'selected_names_per_side', 'long_selected', 'short_selected', 'effective_per_name_abs_weight'}
    if (not needed <= set(coverage)
            or coverage.duplicated(['arm', 'basket', 'signal_date']).any()
            or set(coverage.arm) != ARMS or set(coverage.basket) != {'class_1_10', 'class_2_9'}
            or not coverage.signal_date.map(lambda day: isinstance(day, str)
                and '2017-12-01' <= day <= '2019-11-30').all()
            or not coverage.groupby(['arm', 'basket']).size().eq(24).all()):
        raise DataContractError('Completed selection coverage lost its original 24-month strategy grid')
    n = np.minimum(np.minimum(10, coverage.long_candidates), coverage.short_candidates)
    if (not coverage.selected_names_per_side.eq(n).all()
            or not coverage.long_selected.eq(n).all() or not coverage.short_selected.eq(n).all()
            or (n < 0).any()
            or not np.allclose(coverage.loc[n > 0, 'effective_per_name_abs_weight'],
                .5 / n[n > 0], rtol=0., atol=1e-14)):
        raise DataContractError('Completed selection coverage changed balanced breadth or 50/50 weights')
    if verify_sources:
        _verify_artifacts(complete, audit['artifacts'])
        _verify_artifacts(root, {item['path']: item for item in phase['source_pins']})
        for name, pin in phase.get('workers', {}).items():
            if Path(name).name != name or _hash(Path(__file__).with_name(name)) != pin:
                raise DataContractError('Completed backtest worker changed')
    saved['original_backtest_summary'] = saved['backtest_summary']
    saved['backtest_summary'] = {name: audit[name] for name in ('status', 'scenarios',
        'complete_proxy_paths', 'failed_closed_paths', 'execution_seconds')}
    saved['backtest_comparison'] = comparison
    saved['selection_coverage'] = reports['selection_coverage']
    saved['backtest_directory'] = complete
    saved['summary'].update(backtest_status=audit['status'], completed_action_count=11,
        prior_56_complete_8_withheld_preserved=True, forecasts_and_selections_reused=True)
    return saved
