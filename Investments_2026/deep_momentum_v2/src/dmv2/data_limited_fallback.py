"""Explicit, versioned cache-only development rebuild; never notebook Run All.

The BBBY route remains missing. Two fixed retrospective restrictions affect
trading only. This is not a fully repaired or survivorship-free experiment.
Each numerical phase has its own immutable worker/input identity.
"""
from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import resource
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .classifier_pilot import _ResourceGuard
from .fallback_features import build_fallback_year
from .fallback_loading import NUMERICAL_WORKERS, plan_fallback_year, load_fallback_year
from .full_panel import (_artifact, _hash, _json, _progress, _specs,
                         _verify_artifacts, _write_table, _within)
from .full_target_panel import _save_target_state
from .identity_amendment import load_identity_amendment
from .universe import DataContractError

YEARS = tuple(range(2010, 2020))
WORKERS = tuple(sorted({*NUMERICAL_WORKERS, 'data_limited_fallback.py',
    'fallback_loading.py', 'fallback_features.py', 'fallback_quotes.py',
    'identity_amendment.py', 'targets.py', 'ret.py', 'full_target_panel.py',
    'bracket_ohlc_repairs.py', 'ohlc_repairs.py', 'supplemental_bars.py',
    'action_loading.py', 'action_backtest.py'}))
FROZEN_CONFIG_SHA = '4ef51067ff2b0e8ca3c79dbed0183601fafe5457e9ee16ff310317ec4973a846'


def _configuration(path):
    path = Path(path)
    if _hash(path) != FROZEN_CONFIG_SHA:
        raise DataContractError('Explicit frozen fallback configuration bytes changed')
    cfg = json.loads(path.read_text())
    if (cfg['feature_outcome_years'] != list(YEARS)
            or cfg['research_status'] != 'DATA_LIMITED_RESTRICTED_UNIVERSE'
            or cfg['missing_route_fallback']['replacement_native_id'] is not None
            or set(cfg['execution_restrictions']['native_ids']) != {'sec_gjP4xM', 'sec_zmYNnz'}):
        raise DataContractError('Fallback scope changed')
    return cfg


def _quotes(root, path=None):
    if path is None:
        from .fallback_quotes import compile_fallback_quote_amendments
        bundle = compile_fallback_quote_amendments(root)
    else:
        # Only our locally compiled cache is supported. Every returned source
        # pin and amendment fingerprint is independently checked before use.
        with Path(path).open('rb') as stream:
            bundle = pickle.load(stream)
    bundle.verify(root)
    return bundle


def plan_rebuild(root, cfg, quotes):
    root = Path(root).resolve()
    amendment = load_identity_amendment(root / cfg['identity_amendment'])
    # Full daily membership tables are annual working data, not a ten-year
    # in-memory panel. Retain only authenticated small plan receipts here.
    plans = {}
    for year in YEARS:
        part = plan_fallback_year(root, year, amendment=amendment,
            approved_missing_route_fallback=True)
        plans[year] = {'plan_fingerprint': part.fingerprint,
            'read_plan': part.read_plan, 'source_provenance': part.provenance}
        del part
        gc.collect()
        pa.default_memory_pool().release_unused()
    total_read = sum(p['read_plan']['projected_compressed_bytes'] for p in plans.values())
    if total_read > cfg['execution']['feature_maximum_projected_read_bytes']:
        raise DataContractError('Measured ten-year cache projection exceeds frozen read budget')
    identity = {'schema_version': 1, 'phase': 'data_limited_feature_target_rebuild_v1',
        'configuration': cfg, 'configuration_raw_sha256': 'sha256:' + FROZEN_CONFIG_SHA,
        'quote_amendments_fingerprint': quotes.fingerprint,
        'quote_amendments': quotes.provenance,
        'years': {str(y): p for y, p in plans.items()},
        'workers': {name: _hash(Path(__file__).with_name(name)) for name in WORKERS},
        'runtime': {'python': platform.python_version(), **{name: version(name)
            for name in ('numpy', 'pandas', 'pyarrow', 'exchange-calendars')}},
        'feature_dictionary_sha256': canonical_sha256(_specs()),
        'projected_compressed_bytes': total_read,
        'original_stock_months': sum(p['read_plan']['original_monthly_rows'] for p in plans.values()),
        'historical_identity_certified': False, 'survivorship_bias_removed': False,
        'financial_2020_plus_analysis': False,
        'original_pointers_changed': False}
    return plans, identity, canonical_sha256(identity)


def _preservation(root):
    """Pin original runs/pointers without decoding any financial values."""
    paths = set((root / 'configs').glob('*.json'))
    for relative in ('outputs/model_pilot/2010-2017__2018-2019-a2cd7669fde3f582',
                     'outputs/daily_backtest/2018-2019-3018cad95c61c1da',
                     'outputs/action_backtest/2018-2019-b088d2328a8aea03'):
        paths.update(p for p in (root / relative).rglob('*') if p.is_file())
    paths.update(p for p in (root / 'outputs').rglob('LATEST*') if p.is_file())
    return {str(p.relative_to(root)): _artifact(root, p) for p in sorted(paths)}


def _usage():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def _directory(root, pin):
    return _within(root, root / 'outputs/data_limited_backtest' / ('2010-2019-' + pin[7:23]))


def _verify_year(directory, pin, year):
    audit = json.loads((directory / 'audit.json').read_text())
    if (audit.get('status') != 'COMPLETE' or audit.get('year') != year
            or audit.get('build_identity_sha256') != pin):
        raise DataContractError('New annual checkpoint identity/year changed')
    _verify_artifacts(directory, audit['artifacts'])
    return audit


def _save_year(directory, result, identity, pin, elapsed):
    artifacts = {}
    loaded, features, monthly, targets = result.loaded, result.features, result.monthly, result.targets
    for name, table in (('features', features.table), ('feature_coverage', features.coverage),
                        ('original_cohort', loaded.plan.original_cohort),
                        ('monthly_returns', monthly.table),
                        ('daily_prices_raw', loaded.outcome_prices.filter(pc.greater_equal(
                            loaded.outcome_prices['trading_date'], f'{loaded.plan.year}-01-01'))),
                        ('daily_history_raw', loaded.outcome_prices),
                        ('exchange_calendar', loaded.plan.calendar)):
        _write_table(directory, name, table, artifacts)
    _save_target_state(directory, targets, artifacts)
    for name, value in (('feature_provenance', features.provenance),
                        ('monthly_provenance', monthly.provenance),
                        ('loading_provenance', result.provenance),
                        ('feature_summary', features.summary),
                        ('monthly_summary', monthly.summary)):
        _json(directory / (name + '.json'), value)
        artifacts[name] = _artifact(directory, directory / (name + '.json'))
    coverage = features.coverage
    summary = {'original_rows': features.table.num_rows,
        'prediction_eligible_rows': int(pc.sum(coverage['paper_prediction_eligible']).as_py()),
        'labelled_rows': int(pc.sum(targets.table['target_available']).as_py()),
        'monthly_wrong_route_rows': loaded.plan.read_plan['monthly_wrong_route_rows'],
        'daily_wrong_route_rows': loaded.plan.read_plan['daily_wrong_route_rows'],
        'feature_history_rows': loaded.feature_prices.num_rows,
        'separate_outcome_daily_rows': loaded.outcome_prices.num_rows,
        'elapsed_seconds': elapsed, 'process_peak_rss_bytes': _usage(),
        'financial_2020_plus_analysis': False, 'all_original_keys_preserved': True}
    audit = {'status': 'COMPLETE', 'year': loaded.plan.year,
        'build_identity_sha256': pin, 'research_status': identity['configuration']['research_status'],
        'quote_amendments_fingerprint': identity['quote_amendments_fingerprint'],
        'feature_provenance': features.provenance,
        'target_artifact_sha256': targets.artifact_sha256,
        'summary': summary, 'artifacts': artifacts}
    _json(directory / 'audit.json', audit)
    _verify_year(directory, pin, loaded.plan.year)
    return audit


def build_reconstruction(root, cfg, quotes, *, selected_years=None):
    root = Path(root).resolve()
    plans, identity, pin = plan_rebuild(root, cfg, quotes)
    selected = YEARS if selected_years is None else tuple(selected_years)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(YEARS):
        raise DataContractError('Explicit unique 2010--2019 annual checkpoints required')
    destination = _directory(root, pin)
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / 'build_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise DataContractError('Existing feature rebuild identity changed')
    else:
        _json(identity_path, identity)
        _json(destination / 'source_preservation_manifest.json', _preservation(root))
    preservation = json.loads((destination / 'source_preservation_manifest.json').read_text())
    _verify_artifacts(root, preservation)
    (destination / 'years').mkdir(exist_ok=True)
    started = time.monotonic()
    guards = {'execution': {'maximum_wall_seconds': cfg['execution']['feature_maximum_wall_seconds'],
              'maximum_peak_rss_bytes': cfg['execution']['feature_maximum_peak_rss_bytes']}}
    _progress('fallback_feature_plan', build_identity_sha256=pin, output=str(destination),
        selected_years=list(selected), projected_compressed_bytes=identity['projected_compressed_bytes'],
        original_stock_months=identity['original_stock_months'])
    with _ResourceGuard(guards, started, destination):
        for year in selected:
            directory = destination / 'years' / str(year)
            if (directory / 'audit.json').exists():
                audit = _verify_year(directory, pin, year)
                _progress('fallback_year_reused', year=year, **audit['summary'])
                continue
            if directory.exists():
                raise DataContractError('Preserve incomplete annual checkpoint; inspect before retry: ' + str(directory))
            directory.mkdir()
            _json(directory / 'ATTEMPT.json', {'build_identity_sha256': pin, 'year': year})
            year_started = time.monotonic()
            _progress('fallback_year_started', year=year)
            amendment = load_identity_amendment(root / cfg['identity_amendment'])
            plan = plan_fallback_year(root, year, amendment=amendment,
                approved_missing_route_fallback=True)
            if plan.fingerprint != plans[year]['plan_fingerprint']:
                raise DataContractError('Annual metadata changed after frozen read planning')
            loaded = load_fallback_year(root, plan=plan, quote_amendments=quotes)
            result = build_fallback_year(loaded)
            audit = _save_year(directory, result, identity, pin, time.monotonic() - year_started)
            _progress('fallback_year_completed', year=year, **audit['summary'])
            del result, loaded, plan
            gc.collect()
            pa.default_memory_pool().release_unused()
            if sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) > cfg['execution']['maximum_output_bytes']:
                raise DataContractError('Fallback output-byte budget exceeded')
    _verify_artifacts(root, preservation)
    annual = {str(y): _artifact(destination, destination / 'years' / str(y) / 'audit.json')
              for y in YEARS if (destination / 'years' / str(y) / 'audit.json').exists()}
    _json(destination / 'audit.json', {'status': 'COMPLETE' if len(annual) == len(YEARS) else 'PARTIAL',
        'phase': 'features_and_targets', 'research_status': cfg['research_status'],
        'build_identity_sha256': pin, 'years': annual,
        'source_preservation_verified': True, 'historical_identity_certified': False,
        'survivorship_bias_removed': False, 'financial_2020_plus_analysis': False,
        'elapsed_seconds_this_invocation': time.monotonic() - started,
        'process_peak_rss_bytes': _usage()})
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'build', 'models', 'backtest'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--quote-cache', type=Path)
    parser.add_argument('--years', type=int, nargs='+')
    parser.add_argument('--run-directory', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cfg = _configuration(args.configuration or root / 'configs/data_limited_fallback.json')
    if args.phase in ('plan', 'build'):
        quotes = _quotes(root, args.quote_cache)
        if args.phase == 'plan':
            plans, identity, pin = plan_rebuild(root, cfg, quotes)
            _progress('fallback_plan_verified', build_identity_sha256=pin,
                directory=str(_directory(root, pin)), projected_compressed_bytes=identity['projected_compressed_bytes'],
                original_stock_months=identity['original_stock_months'],
                annual_source_financial_values_read=False,
                reviewed_quote_amendment_values_loaded=True)
            return
        result = build_reconstruction(root, cfg, quotes, selected_years=args.years)
    else:
        if args.run_directory is None:
            parser.error('--run-directory is mandatory for a separate numerical phase')
        run = _within(root, args.run_directory)
        identity = json.loads((run / 'build_identity.json').read_text())
        pin = canonical_sha256(identity)
        if identity.get('configuration') != cfg:
            raise DataContractError('New phase configuration differs from feature identity')
        audit = json.loads((run / 'audit.json').read_text())
        if audit.get('status') != 'COMPLETE' or audit.get('build_identity_sha256') != pin:
            raise DataContractError('All ten new feature/target years must finish before further phases')
        _verify_artifacts(run, audit['years'])
        if args.phase == 'models':
            from .fallback_models import run_fallback_models
            result = run_fallback_models(root, run, cfg, pin)
        else:
            from .fallback_backtest import run_fallback_backtest
            result = run_fallback_backtest(root, run, cfg, pin,
                quote_amendments=_quotes(root, args.quote_cache))
    _progress('fallback_phase_completed', phase=args.phase, directory=str(result))


if __name__ == '__main__':
    main()
