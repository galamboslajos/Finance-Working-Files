#!/usr/bin/env python3
"""Build the external archive scope after the parent stops every writer.

Standard library only. Parse declared catalog/plan/control metadata, never
financial proof/report bodies. All other pin checks are opaque SHA/size reads.
Does not inventory/seal staging, reserve reads, install, or perform Git work.
The installer later reserves proof JSON before checking its PASS contents.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
REPO = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
CONTROL = Path('/private/tmp/dmv2-corrected-refit-install-20261021')
DESTINATION = REPO / 'outputs/corrected_refit/2020-2023-20261021-v1'
HASH_BYTES = 0
RAW_KEYS = ('path', 'size_bytes', 'sha256')
METADATA = {
    'catalog': 'input_catalog.json', 'model_protocol': 'model_protocol.json',
    'model_source_plan': 'run/models/read_plan.json',
    'backtest_source_plan': 'backtest_source_plan.json',
    'backtest_protocol': 'backtest_protocol.json',
    'progress': 'run/models/pipeline/progress.json',
}
GATES = {
    'progress': 'run/models/pipeline/progress.json',
    'models': 'run/verification/models.json',
    'backtest': 'run/verification/backtest.json',
    'model_report': 'run/reports/models_v1/receipt.json',
}
BACKTEST = {
    'source_plan_pin': 'backtest_source_plan.json',
    'backtest_protocol_pin': 'backtest_protocol.json',
    'selection_audit_pin': 'run/backtest/selection/audit.json',
    'preflight_audit_pin': 'run/backtest/preflight/audit.json',
    'replay_audit_pin': 'run/backtest/replay/audit.json',
}
ARTIFACTS = {
    **METADATA,
    'model_verification': 'run/verification/models.json',
    'backtest_verification': 'run/verification/backtest.json',
    'fixture_verification': 'run/verification/backtest_fixture.json',
    'input_verification': 'run/input_verification.json',
    'terminal_guard': 'run/terminal_guard_v2.json',
    'model_report_receipt': 'run/reports/models_v1/receipt.json',
    'model_report_json': 'run/reports/models_v1/comparison.json',
    'model_report_markdown': 'run/reports/models_v1/comparison.md',
    'backtest_report_receipt': 'run/reports/backtest_v1/receipt.json',
    'backtest_report_json': 'run/reports/backtest_v1/report.json',
    'backtest_report_markdown': 'run/reports/backtest_v1/report.md',
    'selection_audit': 'run/backtest/selection/audit.json',
    'preflight_audit': 'run/backtest/preflight/audit.json',
    'replay_audit': 'run/backtest/replay/audit.json',
    'final_readme': 'README.md',
    'installer': 'install_corrected_refit.py',
    'installer_interface': 'ARCHIVE_INSTALL_INTERFACE.md',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    p = Path(path)
    need(p.is_absolute() and p == p.resolve()
         and not any(q.is_symlink() for q in (p, *p.parents))
         and p.is_file() and stat.S_ISREG(p.stat().st_mode),
         'Canonical absolute regular file required: ' + str(p))
    return p


def pin(path):
    global HASH_BYTES
    p = regular(path)
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
            HASH_BYTES += len(block)
    return {'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': digest.hexdigest()}


def normalized_pin(value):
    need(all(key in value for key in RAW_KEYS), 'Incomplete raw pin')
    result = {key: value[key] for key in RAW_KEYS}
    path = Path(result['path'])
    need(path.is_absolute() and path == path.resolve()
         and type(result['size_bytes']) is int and result['size_bytes'] >= 0
         and type(result['sha256']) is str and len(result['sha256']) == 64
         and all(c in '0123456789abcdef' for c in result['sha256']), 'Malformed raw pin')
    return result


def embedded_pins(values):
    """Collect embedded pin records; never follow/parse a pinned JSON body."""
    unique = {}
    def visit(value):
        if isinstance(value, dict):
            if set(RAW_KEYS) <= set(value):
                item = normalized_pin(value)
                old = unique.setdefault(item['path'], item)
                need(old == item, 'Conflicting pins for one canonical path: ' + item['path'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(values)
    return unique


def read_metadata(path):
    return json.loads(regular(path).read_text())


def save_external(path, value):
    p = Path(path).absolute()
    need(p == p.resolve() and p.is_relative_to(CONTROL) and not p.is_relative_to(SOURCE)
         and not any(q.is_symlink() for q in (p, *p.parents)) and not p.exists(),
         'Fresh external archive-control output required')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return pin(p)


def validate_metadata(values, pins):
    catalog, protocol, model, backtest, progress = (
        values[key] for key in ('catalog', 'model_protocol', 'model_source_plan',
                               'backtest_source_plan', 'progress'))
    need(protocol['schema'] == 'dmv2-corrected-refit-protocol/v1'
         and protocol['catalog_pin'] == pins['catalog']
         and protocol['catalog_schema'] == catalog['schema']
         and protocol['catalog_status'] == catalog['status'] == 'VERIFIED_PROVISIONAL_FROZEN'
         and protocol['outcome_years'] == [2020, 2021, 2022, 2023]
         and protocol['parameter_search'] is False, 'Frozen model/catalog scope changed')
    years = [item['year'] for item in catalog['annuals']]
    signals = [item['signal_date'] for item in catalog['RET_states']]
    need(len(years) == 14 and all(type(y) is int for y in years)
         and set(years) == set(range(2010, 2024)) and len(signals) == len(set(signals)) == 48,
         'Exact fourteen annuals and forty-eight unique RET states required')
    need(model['schema'] == 'dmv2-corrected-refit-source-plan/v1'
         and model['catalog_pin'] == pins['catalog'] and model['protocol_pin'] == pins['model_protocol'],
         'Model source plan detached from catalog/protocol')
    need(backtest['schema'] == 'dmv2-corrected-forecasts-backtest-source-plan/v1'
         and backtest['catalog_pin'] == pins['catalog']
         and backtest['model_protocol_pin'] == pins['model_protocol']
         and backtest['scenario_count'] == 120 and backtest['selection_sets'] == 15,
         'Backtest source plan detached from model/catalog scope')
    portfolio = values['backtest_protocol']
    need(portfolio['schema'] == 'dmv2-corrected-forecasts-backtest-protocol/v1'
         and portfolio['source_plan_pin'] == pins['backtest_source_plan']
         and portfolio['model_protocol_pin'] == pins['model_protocol']
         and portfolio['scenarios'] == 120 and portfolio['selection_sets'] == 15
         and portfolio['sessions'] == 1006 and portfolio['outcome_months'] == 48
         and portfolio['new_estimator_fits'] == portfolio['new_inference_calls'] == 0
         and portfolio['changed_portfolio_rules'] is False
         and portfolio['changed_stop_or_cost_rules'] is False,
         'Backtest protocol detached from exact source/rule scope')
    need(progress['status'] == 'COMPLETE_VERIFIED'
         and progress['completed_estimator_fits'] == 20 and progress['completed_preprocessors'] == 8
         and progress['completed_phases'] == 50 and progress['completed_annual_verifications'] == 4
         and progress['final_verification_pin'] == pins['models'], 'Terminal model progress required')


def build(output, *, all_writers_stopped=False, public_document='README.md'):
    need(all_writers_stopped is True, 'Parent must explicitly acknowledge all writers stopped')
    need(not DESTINATION.exists() and not DESTINATION.is_symlink(), 'Archive destination must be fresh')
    relative = Path(public_document)
    need(not relative.is_absolute() and '..' not in relative.parts and relative.suffix == '.md',
         'Final public document must be a source-relative Markdown path')
    pins = {key: pin(SOURCE / relative) for key, relative in METADATA.items()}
    pins['models'] = pin(SOURCE / GATES['models'])
    values = {key: read_metadata(SOURCE / relative) for key, relative in METADATA.items()}
    validate_metadata(values, pins)
    # Safe resource-control metadata only; no logs or numerical proof content.
    guards = {}
    for path in sorted((SOURCE / 'run/control').glob('*.json')):
        if path.name == 'active_time.json':
            continue
        guard = read_metadata(path)
        need(guard.get('status') != 'RUNNING', 'Resource guard is still RUNNING')
        guards[path.name] = pin(path)
    declared = embedded_pins(values)
    originals = [item for path, item in sorted(declared.items()) if Path(path).is_relative_to(REPO)]
    need(originals and not any(Path(p['path']).is_relative_to(DESTINATION) for p in originals),
         'Nonempty original repository pin closure required')
    for item in originals:
        need(pin(item['path']) == item, 'Original repository input changed: ' + item['path'])
    artifacts = {name: pin(SOURCE / relative) for name, relative in ARTIFACTS.items()}
    artifacts['public_document'] = pin(SOURCE / relative)
    artifacts['scope_builder'] = pin(SOURCE / 'build_install_scope.py')
    expected = {name: pin(SOURCE / relative) for name, relative in BACKTEST.items()}
    # Inputs/metadata may not move while assembling the external scope.
    for name, relative in METADATA.items():
        need(pin(SOURCE / relative) == pins[name], 'Metadata changed during scope construction')
    body = {
        'schema': 'dmv2-corrected-refit-archive-scope/v1',
        'status': 'SEALED_AFTER_NUMERICAL_COMPLETION_BEFORE_ARCHIVE',
        'source_root': str(SOURCE), 'destination': str(DESTINATION),
        'all_numerical_stages_terminal': True, 'parent_all_writers_stopped_acknowledged': True,
        'gates': {name: pin(SOURCE / relative) for name, relative in GATES.items()},
        'model_protocol_pin': pins['model_protocol'], 'backtest_expected_pins': expected,
        'artifacts': artifacts, 'original_input_pins': originals,
        'builder_metadata_pins': {name: pins[name] for name in METADATA},
        'terminal_guard_control_pins': guards,
        'original_pin_scope': 'Deduplicated embedded repository pins from catalog, model source plan/protocol, backtest source plan/protocol and safe progress metadata; pinned JSON bodies are not followed or parsed.',
        'archive_scope': 'Installer must inventory every regular source file and directory, including failed attempts/checkpoints, not only named artifacts.',
        'proof_and_report_bodies_parsed': False, 'source_tables_decoded': 0,
        'model_deserializations': 0, 'reservations_created': 0,
        'opaque_hash_bytes_before_scope_write': HASH_BYTES,
        'independent_PASS_checks_deferred_to_precharged_installer_prepare': True,
    }
    return save_external(output, body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--all-writers-stopped', action='store_true')
    parser.add_argument('--public-document', default='README.md')
    args = parser.parse_args()
    print(json.dumps(build(args.output, all_writers_stopped=args.all_writers_stopped,
                           public_document=args.public_document), sort_keys=True))


if __name__ == '__main__':
    main()
