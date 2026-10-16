"""Versioned 2018--2019 quote repair; reuse, never refit, the prior models.

The old ten-year reconstruction and all pointers remain immutable. Only the
two validation annual panels are numerically rebuilt. Training inputs are
authenticated by hashes and referenced rather than relabelled or recopied.
"""
from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import time

import pyarrow as pa

from .catalog import canonical_sha256
from .classifier_pilot import _ResourceGuard
from .data_limited_fallback import _save_year, _verify_year, _preservation
from .fallback_features import build_fallback_year
from .fallback_loading import NUMERICAL_WORKERS, plan_fallback_year, load_fallback_year
from .full_panel import _artifact, _hash, _json, _progress, _specs, _verify_artifacts, _within
from .identity_amendment import load_identity_amendment
from .universe import DataContractError

YEARS = (2018, 2019)
PARENT = "sha256:50aa6e09f11b687c36cd3cf52ed8eb51fdde591fb2391cd58162d02edbb29b10"
WORKERS = tuple(sorted({*NUMERICAL_WORKERS, 'independent_rebuild.py',
    'data_limited_fallback.py', 'fallback_features.py', 'fallback_loading.py',
    'fallback_quotes.py', 'independent_quotes.py', 'composed_quotes.py', 'identity_amendment.py',
    'targets.py', 'ret.py', 'full_target_panel.py', 'bracket_ohlc_repairs.py',
    'ohlc_repairs.py', 'supplemental_bars.py'}))


def validate_configuration(cfg):
    if (cfg.get('protocol_id') != 'dmv2-independent-quote-repair-2018-2019-v1'
            or cfg.get('feature_outcome_years') != list(YEARS)
            or cfg.get('research_status') != 'DATA_LIMITED_RESTRICTED_UNIVERSE'
            or cfg.get('execution', {}).get('maximum_new_model_fits') != 0
            or cfg.get('validation_reprediction', {}).get('fitted_parent_build_identity_sha256') != PARENT
            or cfg.get('validation_reprediction', {}).get('financial_outcome_years') != list(YEARS)
            or set(cfg.get('execution_restrictions', {}).get('native_ids', {})) != {'sec_gjP4xM', 'sec_zmYNnz'}
            or cfg.get('missing_route_fallback', {}).get('replacement_native_id') is not None):
        raise DataContractError('Independent repair must retain the fixed two-year, zero-fit scope')
    return cfg


def plan_independent_rebuild(root, cfg, quotes):
    root = Path(root).resolve()
    validate_configuration(cfg)
    quotes.verify(root)
    parent = _within(root, root / cfg['validation_reprediction']['fitted_parent_directory'])
    parent_identity = json.loads((parent / 'build_identity.json').read_text())
    parent_audit = json.loads((parent / 'audit.json').read_text())
    if canonical_sha256(parent_identity) != PARENT or parent_audit.get('status') != 'COMPLETE':
        raise DataContractError('Prior completed reference reconstruction changed')
    amendment = load_identity_amendment(root / cfg['identity_amendment'])
    plans = {}
    for year in YEARS:
        part = plan_fallback_year(root, year, amendment=amendment, approved_missing_route_fallback=True)
        plans[str(year)] = {'plan_fingerprint': part.fingerprint,
            'read_plan': part.read_plan, 'source_provenance': part.provenance}
        del part
        gc.collect()
        pa.default_memory_pool().release_unused()
    projected = sum(item['read_plan']['projected_compressed_bytes'] for item in plans.values())
    if projected > cfg['execution']['feature_maximum_projected_read_bytes']:
        raise DataContractError('Two-year repair exceeds the frozen read budget')
    identity = {'schema_version': 1, 'phase': 'independent_validation_feature_repair_v1',
        'configuration': cfg, 'configuration_sha256': canonical_sha256(cfg),
        'quote_amendments_fingerprint': quotes.fingerprint,
        'quote_amendments': quotes.provenance, 'years': plans,
        'fitted_parent_directory': str(parent.relative_to(root)),
        'fitted_parent_build_identity_sha256': PARENT,
        'parent_identity_pin': _artifact(root, parent / 'build_identity.json'),
        'training_years_referenced_without_financial_decode': list(range(2010, 2018)),
        'workers': {name: _hash(Path(__file__).with_name(name)) for name in WORKERS},
        'runtime': {'python': platform.python_version(), **{name: version(name)
            for name in ('numpy', 'pandas', 'pyarrow', 'exchange-calendars')}},
        'feature_dictionary_sha256': canonical_sha256(_specs()),
        'projected_compressed_bytes': projected,
        'original_stock_months': sum(item['read_plan']['original_monthly_rows'] for item in plans.values()),
        'new_model_fits': 0, 'new_preprocessor_fits': 0,
        'historical_identity_certified': False, 'survivorship_bias_removed': False,
        'financial_2020_plus_analysis': False, 'original_pointers_changed': False}
    return identity, canonical_sha256(identity)


def build_independent_validation(root, cfg, quotes):
    root = Path(root).resolve()
    identity, pin = plan_independent_rebuild(root, cfg, quotes)
    destination = _within(root, root / 'outputs/independent_backtest' / ('2018-2019-' + pin[7:23]))
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / 'build_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise DataContractError('Existing independent repair identity changed')
    else:
        _json(identity_path, identity)
        preservation = _preservation(root)
        parent = root / cfg['validation_reprediction']['fitted_parent_directory']
        for path in parent.rglob('*'):
            if path.is_file():
                preservation[str(path.relative_to(root))] = _artifact(root, path)
        _json(destination / 'source_preservation_manifest.json', preservation)
    preservation = json.loads((destination / 'source_preservation_manifest.json').read_text())
    _verify_artifacts(root, preservation)
    if (destination / 'audit.json').exists():
        complete = json.loads((destination / 'audit.json').read_text())
        if (complete.get('status') != 'COMPLETE'
                or complete.get('build_identity_sha256') != pin
                or set(complete.get('years', {})) != {'2018', '2019'}):
            raise DataContractError('Preserve unexpected completed root checkpoint for inspection')
        _verify_artifacts(destination, complete['years'])
        for year in YEARS:
            _verify_year(destination / 'years' / str(year), pin, year)
        return destination
    (destination / 'years').mkdir(exist_ok=True)
    started = time.monotonic()
    guards = {'execution': {'maximum_wall_seconds': cfg['execution']['feature_maximum_wall_seconds'],
        'maximum_peak_rss_bytes': cfg['execution']['feature_maximum_peak_rss_bytes']}}
    _progress('independent_feature_plan', build_identity_sha256=pin, output=str(destination),
        projected_compressed_bytes=identity['projected_compressed_bytes'],
        original_stock_months=identity['original_stock_months'], new_model_fits=0)
    with _ResourceGuard(guards, started, destination):
        for year in YEARS:
            directory = destination / 'years' / str(year)
            if (directory / 'audit.json').exists():
                _verify_year(directory, pin, year)
                continue
            if directory.exists():
                raise DataContractError('Preserve incomplete annual attempt: ' + str(directory))
            directory.mkdir()
            _json(directory / 'ATTEMPT.json', {'build_identity_sha256': pin, 'year': year})
            _progress('independent_year_started', year=year)
            year_started = time.monotonic()
            amendment = load_identity_amendment(root / cfg['identity_amendment'])
            plan = plan_fallback_year(root, year, amendment=amendment, approved_missing_route_fallback=True)
            if plan.fingerprint != identity['years'][str(year)]['plan_fingerprint']:
                raise DataContractError('Annual read plan changed after freezing the independent repair')
            loaded = load_fallback_year(root, plan=plan, quote_amendments=quotes)
            result = build_fallback_year(loaded)
            audit = _save_year(directory, result, identity, pin, time.monotonic() - year_started)
            _progress('independent_year_completed', year=year, **audit['summary'])
            del result, loaded, plan
            gc.collect()
            pa.default_memory_pool().release_unused()
            if sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) > cfg['execution']['maximum_output_bytes']:
                raise DataContractError('Independent repair output budget exceeded')
    _verify_artifacts(root, preservation)
    quotes.verify(root)
    annual = {str(year): _artifact(destination, destination / 'years' / str(year) / 'audit.json') for year in YEARS}
    _json(destination / 'audit.json', {'status': 'COMPLETE', 'phase': 'independent_features_and_targets',
        'research_status': cfg['research_status'], 'build_identity_sha256': pin, 'years': annual,
        'source_preservation_verified': True, 'new_model_fits': 0,
        'historical_identity_certified': False, 'survivorship_bias_removed': False,
        'financial_2020_plus_analysis': False, 'elapsed_seconds': time.monotonic() - started})
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'build', 'predict', 'backtest'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--quote-cache', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cfg = validate_configuration(json.loads((args.configuration or root / 'configs/independent_backtest.json').read_text()))
    # Locally compiled object only; verify its exact amendment/source receipts.
    with args.quote_cache.open('rb') as stream:
        quotes = pickle.load(stream)
    quotes.verify(root)
    if args.phase == 'plan':
        identity, pin = plan_independent_rebuild(root, cfg, quotes)
        _progress('independent_plan_verified', build_identity_sha256=pin,
            projected_compressed_bytes=identity['projected_compressed_bytes'])
        return
    if args.phase == 'build':
        result = build_independent_validation(root, cfg, quotes)
    else:
        if args.run_directory is None:
            parser.error('--run-directory required for an explicit separate phase')
        run = _within(root, args.run_directory.resolve())
        identity = json.loads((run / 'build_identity.json').read_text())
        pin = canonical_sha256(identity)
        if identity['configuration'] != cfg:
            raise DataContractError('Frozen reconstruction configuration differs')
        if args.phase == 'predict':
            from .validation_reprediction import run_validation_reprediction
            result = run_validation_reprediction(root, run, cfg, pin,
                fitted_run_directory=root / cfg['validation_reprediction']['fitted_parent_directory'])
        else:
            from .independent_backtest import run_independent_backtest
            result = run_independent_backtest(root, run, cfg, pin, quote_amendments=quotes,
                sni_evidence_directory=root / cfg['sni_evidence_directory'])
    _progress('independent_phase_completed', phase=args.phase, output=str(result))


if __name__ == '__main__':
    main()
