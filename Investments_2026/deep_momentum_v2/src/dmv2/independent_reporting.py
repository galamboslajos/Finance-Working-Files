"""Passive saved reports for the independent quote repair; no compute on import."""
from pathlib import Path
import re

import pandas as pd

from .catalog import canonical_sha256
from .fallback_reporting import _json, _pinned_json, _receipt, _backtest_report
from .full_panel import _hash, _verify_artifacts, _within
from .universe import DataContractError

ARMS = {'rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended'}


def read_independent_run(root, directory, *, identity_sha256, expected_receipts,
                         verify_sources=False):
    """Read bounded audit JSON and comparison reports, never panels or models.

    Explicit final audit receipts prevent a mutable output lookup. Deep source
    hashing is optional and does not decode financial source values.
    """
    if (type(verify_sources) is not bool or not isinstance(expected_receipts, dict)
            or set(expected_receipts) != {'features', 'models', 'backtest'}
            or any(not isinstance(value, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', value)
                for value in expected_receipts.values())):
        raise DataContractError('All three final receipts and explicit verification scope required')
    root = Path(root).resolve()
    run = _within(root / 'outputs/independent_backtest', Path(directory).resolve())
    if run.name != '2018-2019-' + identity_sha256[7:23]:
        raise DataContractError('Independent saved directory differs from its explicit identity')
    identity = _json(run / 'build_identity.json')
    if (canonical_sha256(identity) != identity_sha256
            or identity.get('phase') != 'independent_validation_feature_repair_v1'
            or set(identity.get('years', {})) != {'2018', '2019'}
            or identity.get('configuration', {}).get('execution', {}).get('maximum_new_model_fits') != 0):
        raise DataContractError('Independent repair saved scope changed')
    for label, relative in (('features', 'audit.json'), ('models', 'models/audit.json'),
                            ('backtest', 'backtest/audit.json')):
        _receipt(run / relative, expected_receipts[label], label)
    audit = _json(run / 'audit.json')
    if (audit.get('status') != 'COMPLETE' or audit.get('build_identity_sha256') != identity_sha256
            or set(audit.get('years', {})) != {'2018', '2019'}):
        raise DataContractError('Independent validation annual build is incomplete')
    annuals = []
    for year, pin in audit['years'].items():
        annual = _pinned_json(run, pin, expected_path=run / 'years' / year / 'audit.json')
        if (annual.get('status') != 'COMPLETE' or annual.get('year') != int(year)
                or annual.get('build_identity_sha256') != identity_sha256):
            raise DataContractError('Independent annual receipt changed')
        if verify_sources:
            _verify_artifacts(run / 'years' / year, annual['artifacts'])
        annuals.append({'year': int(year), **annual['summary']})
    models = run / 'models'
    model = _json(models / 'audit.json')
    if (model.get('phase') != 'validation_reprediction_v1'
            or model.get('status') != 'COMPLETE'
            or model.get('build_identity_sha256') != identity_sha256
            or model.get('summary', {}).get('newly_fitted_arms') != 0
            or model.get('summary', {}).get('reused_fitted_arms') != 4
            or model.get('summary', {}).get('preprocessing_fits') != 0
            or set(model.get('arms', {})) != ARMS | {'uniform', 'train_prior'}):
        raise DataContractError('Independent inference lost its four reused fits or zero-fit scope')
    phase = _pinned_json(models, model['artifacts']['build_identity.json'],
        expected_path=models / 'build_identity.json')
    if (canonical_sha256(phase) != model.get('model_identity_sha256')
            or phase.get('parent_identity_sha256') != identity_sha256
            or phase.get('new_estimator_fits') != 0 or phase.get('preprocessing_fits') != 0):
        raise DataContractError('Independent inference phase changed')
    plan = phase.get('data_plan', {})
    authority = identity['configuration'].get('validation_reprediction', {})
    if (plan.get('fitted_parent_build_identity_sha256') != authority.get('fitted_parent_build_identity_sha256')
            or plan.get('fitted_parent_model_identity_sha256') != model['summary'].get('fitted_parent_model_identity_sha256')
            or plan.get('new_estimator_fits') != 0 or plan.get('preprocessing_fits') != 0):
        raise DataContractError('Independent inference changed its fitted parent or zero-fit authority')
    parent = _within(root, root / authority['fitted_parent_directory'])
    small_parents = {}
    for relative in ('audit.json', 'build_identity.json', 'models/audit.json', 'models/build_identity.json'):
        path = parent / relative
        key = str(path.relative_to(root))
        pin = plan.get('source_pins', {}).get(key)
        small_parents[relative] = _pinned_json(root, pin, expected_path=path)
    if (canonical_sha256(small_parents['build_identity.json']) != authority['fitted_parent_build_identity_sha256']
            or canonical_sha256(small_parents['models/build_identity.json']) != plan['fitted_parent_model_identity_sha256']
            or small_parents['audit.json'].get('status') != 'COMPLETE'
            or small_parents['models/audit.json'].get('status') != 'COMPLETE'):
        raise DataContractError('Independent inference parent receipts differ from the completed original fits')
    comparisons = []
    for name, pin in model['arms'].items():
        path = models / ('arms' if name in ARMS else 'refs') / name / 'audit.json'
        arm = _pinned_json(models, pin, expected_path=path)
        if (arm.get('build_identity_sha256') != identity_sha256
                or arm.get('model_identity_sha256') != model['model_identity_sha256']
                or arm.get('metadata', {}).get('new_fit_count') != 0):
            raise DataContractError('Independent saved arm no longer references the frozen inference')
        comparisons.append({'arm_id': name, **arm['summary']})
    bt = _json(run / 'backtest/audit.json')
    if bt.get('phase') != 'independent_quote_backtest_v1':
        raise DataContractError('Independent backtest phase changed')
    backtest_summary, backtest_comparison = _backtest_report(
        run, identity_sha256, expected_receipts['backtest'], verify_sources)
    if verify_sources:
        from .validation_reprediction import read_validation_reprediction
        read_validation_reprediction(models, root=root, identity_sha256=identity_sha256)
        _verify_artifacts(root, _json(run / 'source_preservation_manifest.json'))
        for name, pin in identity['workers'].items():
            if Path(name).name != name or _hash(Path(__file__).with_name(name)) != pin:
                raise DataContractError('Independent feature worker changed')
    return {'directory': run, 'year_summaries': pd.DataFrame(annuals).sort_values('year'),
        'model_comparison': pd.DataFrame(comparisons), 'model_summary': model['summary'],
        'backtest_comparison': backtest_comparison, 'backtest_summary': backtest_summary,
        'summary': {'feature_status': 'COMPLETE', 'model_status': 'COMPLETE',
            'backtest_status': backtest_summary['status'], 'outcome_years': [2018, 2019],
            'new_model_fits': 0, 'reused_fitted_arms': 4,
            'research_status': 'DATA_LIMITED_RESTRICTED_UNIVERSE',
            'financial_2020_plus_analysis': False, 'quotes_compiled': False,
            'models_deserialized': False, 'numerical_work_started': False,
            'full_source_bytes_verified': verify_sources}}
