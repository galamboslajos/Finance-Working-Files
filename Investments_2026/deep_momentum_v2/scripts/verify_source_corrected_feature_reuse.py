"""Independent streaming-byte verification of reviewed annual input reuse.

Standard-library only: no Parquet financial decoding, formulas, fitted models,
probabilities, inference, fitting, source acquisition or accounting is invoked.
PASS means the exact reviewed inputs and prior bounded formula proofs were
authenticated and reused; it is not a new feature-formula recomputation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import sys
import time

SCHEMA = 'dmv2-source-corrected-feature-input-reuse-verification/v1'
REPAIR_YEARS = (2010, 2011, 2012, 2013, 2022, 2023)
OLD_FEATURE_HELPER_SHA = '2b4e1ad81e27f408934ee30e71ecfc52bbf96a122683de2cd2083463fff1208b'
# Recorded before the refits in source_corrected_reporting_candidate/old_reference_before.json.
OLD_FEATURE_RECEIPT_SHA = '61078970a3929b53896b590da31b4fac9d3fa5bc6068c3c0e0a8b96733e85c26'
MAX_JSON_BYTES = 4 * 1024**2
MAX_UNIQUE_FILES = 1024
MAX_UNIQUE_BYTES = 4 * 1024**3
MAX_HASH_PASSES = 2
MAX_WALL_SECONDS = 600
MAX_RSS_BYTES = 256 * 1024**2


def require(condition, message):
    if not condition: raise AssertionError(message)


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def local(base, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and '..' not in Path(relative).parts, 'Closed relative artifact path required')
    base = Path(base).resolve(); raw = base / relative
    require(not any(p.is_symlink() for p in [raw, *raw.parents]), 'Artifact symlinks refused')
    resolved = raw.resolve()
    require(resolved.is_relative_to(base), 'Artifact escapes its explicit base')
    return resolved


def read(path):
    path = Path(path)
    require(path.is_file() and 0 < path.stat().st_size <= MAX_JSON_BYTES, 'Bounded JSON metadata required')
    return json.loads(path.read_text())


def raw_pin(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), 'Pin must remain inside the research root')
    return {'path': str(path.relative_to(root)), 'sha256': digest(path), 'size_bytes': path.stat().st_size}


def metadata(root, pin):
    path = local(root, pin['path'])
    require(path.stat().st_size <= MAX_JSON_BYTES and path.stat().st_size == pin['size_bytes']
            and digest(path) == pin['sha256'].removeprefix('sha256:'), 'Pinned JSON metadata changed')
    return path, read(path)


def prepare(root, run, expected_identity):
    """Freeze all expected byte pins before the streaming data-file hash pass."""
    root, run = Path(root).resolve(), Path(run).resolve()
    require(run.is_relative_to(root / 'outputs/walkforward_source_corrected'), 'Separate corrected-source run required')
    identity_path = run / 'build_identity.json'; identity = read(identity_path)
    require(canonical(identity) == expected_identity and identity.get('phase') == 'source_corrected_walkforward_2020_2023_v1',
            'Explicit corrected root identity changed')
    cfg = identity['configuration']; audit_path = run / 'audit.json'; audit = read(audit_path)
    require(audit.get('status') == 'COMPLETE' and audit.get('phase') == 'reviewed_annual_input_reuse_only'
            and audit.get('build_identity_sha256') == expected_identity
            and audit.get('financial_2024_plus_analysis') is False
            and audit.get('original_sources_and_runs_preserved') is True
            and audit.get('model_fits') == 0 and audit.get('backtest_runs') == 0
            and set(audit.get('years', {})) == {'2020', '2021', '2022', '2023'},
            'Completed reuse-only input audit changed')
    require(identity.get('configuration_sha256') == canonical(cfg)
            and cfg == read(root / 'configs/walkforward_2020_2023.json')
            and cfg.get('context_policy') == 'available_cohort_80pct'
            and cfg['execution'].get('financial_2024_plus_analysis') is False,
            'Frozen feature/context/protected-scope contract changed')
    files = {}
    def add(pin):
        require(isinstance(pin, dict) and set(pin) >= {'path', 'sha256', 'size_bytes'}, 'Complete raw artifact pin required')
        path = local(root, pin['path'])
        require(path.is_file() and path.stat().st_size == pin['size_bytes']
                and type(pin['size_bytes']) is int and 0 < pin['size_bytes'] <= MAX_UNIQUE_BYTES,
                'Artifact size/type changed before hash plan')
        normalized = {'path': pin['path'], 'sha256': pin['sha256'].removeprefix('sha256:'), 'size_bytes': pin['size_bytes']}
        require(pin['path'] not in files or files[pin['path']] == normalized, 'Same path has conflicting expected pins')
        files[pin['path']] = normalized
    add(raw_pin(root, identity_path)); add(raw_pin(root, audit_path)); add(raw_pin(root, root / 'configs/walkforward_2020_2023.json'))
    for pin in identity['preserved_reference_pins'].values(): add(pin)
    for name, expected in identity['workers'].items():
        path = local(root, 'src/dmv2/' + name)
        require(digest(path) == expected, 'Frozen original worker changed')
        add(raw_pin(root, path))
    manifest_path, manifest = metadata(root, identity['source_correction_manifest_pin'])
    handoff_path, handoff = metadata(root, identity['source_correction_handoff_pin'])
    review_path, review = metadata(root, identity['independent_corrected_input_review_pin'])
    for name in ('source_correction_manifest_pin', 'source_correction_handoff_pin', 'independent_corrected_input_review_pin'): add(identity[name])
    require(handoff.get('status') == 'REPAIRED_INPUTS_READY_MODELS_STALE'
            and handoff.get('existing_runs_modified') is False
            and handoff.get('financial_2024_plus_analysis') is False
            and len(handoff.get('artifacts', [])) == 412,
            'Original 412-artifact reviewed source handoff changed')
    require(len({pin['path'] for pin in handoff['artifacts']}) == 412,
            'Source handoff must preserve 412 distinct artifact paths')
    for pin in handoff['artifacts']: add(pin)
    require(identity['source_correction_manifest_pin'] in handoff['artifacts']
            and identity['independent_corrected_input_review_pin'] in handoff['artifacts'],
            'Source manifest/review escaped the exact handoff')
    require(review.get('status') == 'PASS_BOUNDED_SIX_ANNUAL_INPUT_RECOMPUTATION'
            and review.get('years_checked') == list(REPAIR_YEARS)
            and review.get('model_fits') == 0 and review.get('backtest_runs') == 0,
            'Corrected-source upstream mathematical review changed')
    require(all(review.get('checks', {}).get(name) == 'PASS' for name in (
        'all15_paper_formulas_and_common_population_normalization',
        'exact_monthly_endpoints_and_missing_return_policy',
        'ten_class_exact_tie_rank_and_missing_label_policy', 'no2024_financial_rows_returned')),
        'Upstream corrected-source mathematical/scope claims are incomplete')
    entries = manifest.get('years')
    require(manifest.get('status') == 'VERIFIED_INPUT_REFERENCES_NOT_MODEL_RESULTS'
            and manifest.get('context_policy') == 'available_cohort_80pct'
            and manifest.get('financial_2024_plus_analysis') is False
            and isinstance(entries, list) and [i.get('year') for i in entries] == list(range(2010, 2024))
            and {i['year'] for i in entries if i.get('corrected')} == set(REPAIR_YEARS),
            'Reviewed fourteen-year annual source map changed')
    review_dir = review_path.parent
    annuals, detached, mathematical_proofs = [], [], {}
    for item in entries:
        year = item['year']; source = local(root, item['directory']); source_audit = source / 'audit.json'
        require(digest(source_audit) == item['audit_sha256'], 'Reviewed annual audit raw hash changed')
        a = read(source_audit); add(raw_pin(root, source_audit))
        expected_ref = {'directory': item['directory'], 'audit_pin': raw_pin(root, source_audit),
            'build_identity_sha256': item['build_identity_sha256'], 'corrected': item['corrected'],
            'original_rows': item['original_rows']}
        require(identity['reviewed_source_annuals'][str(year)] == expected_ref
                and a.get('status') == 'COMPLETE' and a.get('year') == year
                and a.get('build_identity_sha256') == item['build_identity_sha256'],
                'Annual source/root identity binding changed')
        require({'features', 'monthly_returns', 'targets', 'daily_history_raw', 'daily_prices_raw',
                 'original_cohort', 'feature_provenance', 'exchange_calendar'} <= set(a['artifacts']),
                'Annual source omitted required separate feature/outcome/membership artifacts')
        require(a['artifacts']['features']['path'] != a['artifacts']['targets']['path']
                and a['artifacts']['daily_history_raw']['path'] != a['artifacts']['daily_prices_raw']['path'],
                'Separate feature/label/daily input roles aliased')
        for p in a['artifacts'].values(): add({'path': str(local(source, p['path']).relative_to(root)), 'sha256': p['sha256'], 'size_bytes': p['size_bytes']})
        annuals.append({'year': year, **expected_ref, 'artifact_count': len(a['artifacts']),
                        'artifact_bytes': sum(p['size_bytes'] for p in a['artifacts'].values())})
        route = identity['annual_sources'][str(year)]
        if year < 2020:
            require(route.get('directory') == item['directory'] and route.get('directory_scope') == 'root'
                    and route.get('audit_pin') == expected_ref['audit_pin']
                    and route.get('build_identity_sha256') == item['build_identity_sha256']
                    and route.get('reused_without_numeric_rebuild') is True,
                    'Historical model input source resolver changed')
            for role in ('parent_identity_pin', 'parent_audit_pin'):
                path, value = metadata(root, route[role]); add(route[role])
                require((canonical(value) if role == 'parent_identity_pin' else value.get('build_identity_sha256'))
                        == item['build_identity_sha256'], 'Historical source parent identity changed')
        else:
            require(route == {'directory': f'years/{year}', 'directory_scope': 'run',
                'parent_reference': 'THIS_BUILD', 'audit_pin': None}, 'Detached model source resolver changed')
            target = run / 'years' / str(year)
            declared = audit['years'][str(year)]
            require(declared['path'] == f'years/{year}/audit.json', 'Root annual audit path redirected')
            target_audit_path, b = metadata(run, declared)
            require(target_audit_path == target / 'audit.json' and b.get('status') == 'COMPLETE'
                    and b.get('year') == year and b.get('build_identity_sha256') == expected_identity,
                    'Detached annual identity or exact path changed')
            add(raw_pin(root, target_audit_path))
            require(set(b['artifacts']) == set(a['artifacts']) | {'reference_annual_audit'},
                    'Detached annual scope differs beyond its explicit reference audit')
            for name, p in a['artifacts'].items():
                require(p == b['artifacts'][name], 'Detached annual full artifact pin differs: ' + name)
                add({'path': str(local(target, p['path']).relative_to(root)), 'sha256': p['sha256'], 'size_bytes': p['size_bytes']})
            reference_pin = b['artifacts']['reference_annual_audit']
            require(reference_pin == {'path': 'reference_annual_audit.json', 'size_bytes': source_audit.stat().st_size,
                                     'sha256': item['audit_sha256']}, 'Preserved source-audit copy redirected or changed')
            add({'path': str((target / reference_pin['path']).relative_to(root)),
                 'sha256': reference_pin['sha256'], 'size_bytes': reference_pin['size_bytes']})
            lineage = {**expected_ref, 'numerical_artifact_bytes_unchanged': True,
                       'new_feature_calculations': 0, 'original_audit_preserved_as': 'reference_annual_audit.json'}
            require(b.get('source_corrected_replay_lineage') == lineage, 'Detached source provenance changed')
            detached.append({'year': year, 'audit_pin': raw_pin(root, target_audit_path),
                             'source_audit_pin': expected_ref['audit_pin'], 'artifact_count': len(b['artifacts']),
                             'copied_bytes': sum(p['size_bytes'] for p in a['artifacts'].values()),
                             'preserved_source_audit_bytes': reference_pin['size_bytes']})
        if year in REPAIR_YEARS:
            rp = review_dir / f'{year}_review.json'
            original_review_pin = review['annual_review_files'][str(year)]
            require(rp.name == Path(original_review_pin['path']).name and digest(rp) == original_review_pin['sha256'],
                    'Relocated corrected annual mathematical review differs from original bytes')
            proof = read(rp); add(raw_pin(root, rp))
            require(proof.get('status') == 'PASS_BOUNDED_INDEPENDENT_INPUT_RECOMPUTATION'
                    and proof.get('year') == year and proof.get('build_identity_sha256') == item['build_identity_sha256']
                    and proof.get('annual_audit_sha256') == item['audit_sha256']
                    and proof.get('helper_sha256') == OLD_FEATURE_HELPER_SHA
                    and proof.get('paper15', {}).get('rows') == item['original_rows']
                    and proof.get('paper15', {}).get('values_checked') == item['original_rows'] * 15
                    and proof.get('classes', {}).get('original_rows') == item['original_rows']
                    and proof.get('no2024_numeric_rows') == 'PASS'
                    and isinstance(proof.get('monthly_returns'), dict) and bool(proof['monthly_returns'])
                    and proof.get('input_artifact_sha256') == {k: p['sha256'] for k, p in a['artifacts'].items()},
                    'Corrected mathematical proof does not bind these exact annual artifact values')
            mathematical_proofs[str(year)] = {'kind': 'UPSTREAM_CORRECTED_BOUNDED_RECOMPUTATION',
                'receipt_pin': raw_pin(root, rp), 'all_paper15_monthly_returns_classes_recomputed_upstream': True,
                'expanded_formulas_exhaustively_certified': False}
    require(sum(x['copied_bytes'] for x in detached) == audit['copied_input_bytes'], 'Detached copy-byte total changed')
    old = root / 'outputs/walkforward/2020-2023-aacd7579fcf21e6a'
    old_audit_pin = identity['preserved_reference_pins'][str((old / 'audit.json').relative_to(root))]
    _, old_audit = metadata(root, old_audit_pin)
    old_identity_pin = identity['preserved_reference_pins'][str((old / 'build_identity.json').relative_to(root))]
    _, old_identity = metadata(root, old_identity_pin)
    require(canonical(old_identity) == old_audit.get('build_identity_sha256')
            and old_identity.get('configuration') == cfg and old_audit.get('status') == 'COMPLETE',
            'Preserved original feature root or protocol binding changed')
    old_proof_path = old / 'verification/features.json'
    require(digest(old_proof_path) == OLD_FEATURE_RECEIPT_SHA,
            'Pre-existing original feature-verification raw receipt pin changed')
    old_proof = read(old_proof_path)
    add(raw_pin(root, old_proof_path))
    helper = root / 'scripts/verify_walkforward_features_v2.py'
    require(digest(helper) == OLD_FEATURE_HELPER_SHA, 'Original independent feature-equation helper changed')
    add(raw_pin(root, helper))
    require(old_proof.get('status') == 'PASS' and old_proof.get('models_fitted') == 0
            and old_proof.get('financial_2024_plus_read') is False
            and old_proof.get('root_audit_sha256') == old_audit_pin['sha256']
            and old_proof.get('build_identity_sha256') == old_audit.get('build_identity_sha256')
            and old_proof.get('verifier_contract_revision') == 'v2-explicit-shared-market-quarantine'
            and old_proof.get('months') == 48 and set(old_proof.get('years', {})) == {'2020', '2021', '2022', '2023'}
            and old_proof.get('verifier_source_sha256') == OLD_FEATURE_HELPER_SHA
            and old_proof.get('verifier_sha256') == OLD_FEATURE_HELPER_SHA,
            'Preserved original feature mathematical receipt changed')
    for year in (2020, 2021):
        source = next(i for i in entries if i['year'] == year)
        old_year_pin = old_audit['years'][str(year)]
        old_year_path, old_year = metadata(old, old_year_pin)
        require(old_year_pin['path'] == f'years/{year}/audit.json'
                and old_year_path == local(root, source['directory']) / 'audit.json'
                and old_year_pin['sha256'] == source['audit_sha256']
                and old_year.get('build_identity_sha256') == old_audit['build_identity_sha256'],
                'Unchanged old annual is not the mathematical receipt source')
        proof = old_proof['years'][str(year)]
        require(proof.get('original_rows') == source['original_rows']
                and proof.get('paper15', {}).get('values_checked') == source['original_rows'] * 15
                and proof.get('no2024_price_rows_returned') is True
                and proof.get('feature_and_outcome_daily_views_separate') is True,
                'Original unchanged-year mathematical scope changed')
        mathematical_proofs[str(year)] = {'kind': 'UPSTREAM_ORIGINAL_UNCHANGED_RECOMPUTATION',
            'receipt_pin': raw_pin(root, old_proof_path), 'annual_audit_pin': old_audit['years'][str(year)],
            'all_paper15_monthly_returns_classes_recomputed_upstream': True,
            'expanded_formulas_exhaustively_certified': False}
    for year in range(2014, 2020):
        mathematical_proofs[str(year)] = {'kind': 'PINNED_EXISTING_INPUT_REUSE',
            'new_mathematical_recomputation_in_this_pass': False,
            'independent_formula_certification_extended_by_this_receipt': False}
    total = sum(p['size_bytes'] for p in files.values())
    require(len(files) <= MAX_UNIQUE_FILES and total <= MAX_UNIQUE_BYTES, 'Frozen streaming-hash budget exceeded')
    return {'schema': SCHEMA + '-plan', 'project_root': str(root), 'run_directory': str(run.relative_to(root)),
        'build_identity_sha256': expected_identity, 'root_audit_sha256': digest(audit_path),
        'root_identity_pin': raw_pin(root, identity_path), 'root_audit_pin': raw_pin(root, audit_path),
        'source_correction_handoff_pin': identity['source_correction_handoff_pin'],
        'independent_corrected_input_review_pin': identity['independent_corrected_input_review_pin'],
        'source_correction_manifest_pin': identity['source_correction_manifest_pin'],
        'old_feature_verification_pin': raw_pin(root, old_proof_path),
        'old_feature_verification_prefit_raw_sha256': OLD_FEATURE_RECEIPT_SHA,
        'original_handoff_artifacts': handoff['artifacts'], 'annuals': annuals, 'detached_annuals': detached,
        'upstream_mathematical_proof_scope_by_year': mathematical_proofs,
        'input_pins': sorted(files.values(), key=lambda p: p['path']),
        'unique_files': len(files), 'unique_bytes': total,
        'limits': {'maximum_unique_files': MAX_UNIQUE_FILES, 'maximum_unique_bytes': MAX_UNIQUE_BYTES,
            'hash_passes': MAX_HASH_PASSES, 'maximum_wall_seconds': MAX_WALL_SECONDS, 'maximum_peak_rss_bytes': MAX_RSS_BYTES},
        'verifier_sha256': digest(__file__), 'financial_columns_decoded': False,
        'new_formula_computation': False, 'models_or_probabilities_read': False}


def verify(root, run, expected_identity, output_directory):
    started = time.monotonic(); root = Path(root).resolve(); out = Path(output_directory).resolve()
    require(not out.exists(), 'Preserve every existing reuse-verification attempt')
    plan = prepare(root, run, expected_identity)
    out.mkdir(parents=True)
    plan_path = out / 'plan.json'; plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2) + '\n')
    hashed_bytes = 0
    for stage in ('before', 'after'):
        for pin in plan['input_pins']:
            path = local(root, pin['path'])
            require(path.stat().st_size == pin['size_bytes'] and digest(path) == pin['sha256'],
                    'Actual input bytes changed during ' + stage + ' verification: ' + pin['path'])
            hashed_bytes += pin['size_bytes']
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            require(time.monotonic() - started <= MAX_WALL_SECONDS and peak <= MAX_RSS_BYTES,
                    'Streaming verification exceeded frozen time/RSS bounds')
        require(digest(Path(__file__)) == plan['verifier_sha256'], 'Verifier code changed during proof')
    result = {'schema': SCHEMA, 'status': 'PASS', 'verification_kind': 'REVIEWED_ANNUAL_INPUT_BYTE_REUSE',
        'verification_scope': 'new_root_input_reference_reuse_and_prior_formula_evidence',
        'build_identity_sha256': expected_identity, 'root_audit_sha256': plan['root_audit_sha256'],
        'root_identity_pin': plan['root_identity_pin'], 'root_audit_pin': plan['root_audit_pin'],
        'plan_sha256': digest(plan_path), 'verifier_sha256': plan['verifier_sha256'],
        'source_correction_handoff_pin': plan['source_correction_handoff_pin'],
        'independent_corrected_input_review_pin': plan['independent_corrected_input_review_pin'],
        'source_correction_manifest_pin': plan['source_correction_manifest_pin'],
        'old_feature_verification_pin': plan['old_feature_verification_pin'],
        'original_handoff_artifacts_preserved': 412,
        'original_handoff_artifact_bytes': sum(p['size_bytes'] for p in plan['original_handoff_artifacts']),
        'all_original_handoff_pins_authenticated_before_and_after': True,
        'annual_reference_years': [p['year'] for p in plan['annuals']], 'annual_reference_count': 14,
        'annual_reference_artifact_count': sum(p['artifact_count'] for p in plan['annuals']),
        'detached_annual_years': [p['year'] for p in plan['detached_annuals']],
        'detached_annual_count': 4, 'detached_copy_bytes': sum(p['copied_bytes'] for p in plan['detached_annuals']),
        'preserved_reference_audit_copy_bytes': sum(p['preserved_source_audit_bytes'] for p in plan['detached_annuals']),
        'all_detached_full_artifact_pins_and_reference_audit_paths_match': True,
        'upstream_mathematical_proof_scope_by_year': plan['upstream_mathematical_proof_scope_by_year'],
        'new_feature_formula_calculations': 0, 'new_monthly_return_calculations': 0,
        'new_target_class_calculations': 0, 'new_expanded_formula_certification': False,
        'financial_columns_decoded': False, 'financial_2024_plus_read': False,
        'models_fitted': 0, 'model_fits': 0, 'model_files_loaded': 0, 'probability_columns_read': 0,
        'source_downloads': 0, 'backtest_runs': 0, 'input_files_changed': False,
        'execution': {'unique_files_hashed': plan['unique_files'], 'unique_file_bytes': plan['unique_bytes'],
            'complete_hash_passes': 2, 'total_file_bytes_hashed': hashed_bytes,
            'elapsed_seconds': time.monotonic() - started, 'peak_rss_bytes': peak, 'limits': plan['limits']},
        'limitations': [
            'PASS authenticates reviewed input reuse and upstream bounded mathematical proofs; no feature, monthly-return or label arithmetic was newly computed in this pass.',
            'Six repaired annuals inherit their exact pinned all-Paper15/endpoint/class recomputation plus sampled daily formulas. Unchanged2020/21 inherit their original exact feature-verification receipt.',
            'The2014-2019 annuals are authenticated existing inputs; this receipt does not extend independent formula certification for those years.',
            'Neither upstream receipt exhaustively certifies all715 expanded formulas; original daily formula sampling limits remain.',
            'Whole-file cryptographic hashing reads bytes without decoding financial or probability values. No protected2024+ financial values were returned or analyzed.',
            'Historical identity, knowledge time, corporate-action/delisting completeness, survivorship and investability remain uncertified. Model/inference/accounting verification are separate gates.',
        ]}
    receipt = out / 'receipt.json'
    with receipt.open('x') as stream: json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-identity-sha256', required=True); parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.run, args.expected_identity_sha256, args.output_directory)
    print(json.dumps({k: result[k] for k in ('status', 'verification_kind', 'annual_reference_count',
        'detached_annual_count', 'original_handoff_artifacts_preserved', 'execution')}, sort_keys=True))
