"""Freeze the unchanged verified BK/XOM panels after rejected quote patches."""
from pathlib import Path
import json
from control import ROOT, check, pin, read, save_new, need


def main():
    candidate_path = ROOT / 'candidate_catalog.json'
    candidate_pin = pin(candidate_path)
    candidate = read(candidate_path)
    base_path = check(candidate['base_catalog_pin'])
    base = read(base_path)
    inherited = read(check(base['independent_verification']))
    repair_path = check(candidate['repair_verification_pin'])
    repair = read(repair_path)
    manifest = read(check(candidate['repair_manifest_pin']))
    terminal_path = ROOT / 'run/terminal_guard_v2.json'
    terminal_pin = pin(terminal_path)
    terminal = read(terminal_path)
    need(inherited['status'] == repair['status'] == terminal['status'] == 'PASS', 'Verification gate')
    need(terminal['candidate_catalog_pin'] == candidate_pin
         and terminal['base_catalog_pin'] == candidate['base_catalog_pin'], 'Terminal guard catalog binding')
    need(terminal['events_reviewed'] == terminal['missing_terminal_overlap_rows'] == 29
         and terminal['post_effective_original_membership_rows'] == 0
         and terminal['target_definition_unchanged'] is True
         and terminal['missing_terminal_targets_imputed'] is False, 'Terminal guard policy differs')
    need(repair['manifest_pin'] == candidate['repair_manifest_pin'], 'Independent repair decision binding')
    for result in (repair, terminal):
        check(result['plan_pin'])
        check(result['worker_pin'])
    need(manifest['row_count'] == repair['admitted_rows'] == candidate['new_learning_quote_rows'] == 0,
         'This freeze only supports zero admitted quote patches')
    need(all(case['status'] == 'REJECTED_SOURCE_QUALITY' and case['admitted_repair_rows'] == 0
             for case in manifest['cases'].values()), 'No rejected partial learning repairs')
    need(candidate['annuals'] == base['annuals'] and candidate['RET_states'] == base['RET_states'],
         'Unchanged annual/RET byte identities required')
    need(len(candidate['RET_states']) == 48
         and len({state['signal_date'] for state in candidate['RET_states']}) == 48,
         'Exactly 48 unique RET signal states required')
    need(len(candidate['annuals']) == 14 and {x['year'] for x in candidate['annuals']} == set(range(2010, 2024)),
         'Exact fourteen years required')
    # Integrity byte reads do not deserialize financial values or fitted states.
    total = 0
    seen = {}
    def verify(item):
        nonlocal total
        raw = {key: item[key] for key in ('path', 'sha256', 'size_bytes')}
        if raw['path'] in seen:
            need(raw == seen[raw['path']], 'Conflicting shared-path pins')
        else:
            check(raw)
            total += raw['size_bytes']
            seen[raw['path']] = raw
    for annual in candidate['annuals']:
        verify(annual['audit'])
        for item in annual['artifacts'].values():
            verify(item)
    for state in candidate['RET_states']:
        for item in state['artifacts'].values():
            verify(item)
    expected = {Path(item['path']).parent.name: item for item in inherited['annual_audit_pins']}
    need(all(expected[str(a['year'])]['sha256'] == a['audit']['sha256']
             and expected[str(a['year'])]['size_bytes'] == a['audit']['size_bytes']
             for a in candidate['annuals']), 'Inherited numerical proof differs from installed annual bytes')
    for item in (candidate_pin, candidate['base_catalog_pin'], base['independent_verification'],
            candidate['repair_verification_pin'], candidate['repair_manifest_pin'], terminal_pin):
        check(item)
    receipt = {'schema': 'dmv2-corrected-refit-input-verification/v1', 'status': 'PASS',
        'candidate_catalog_pin': candidate_pin, 'base_catalog_pin': candidate['base_catalog_pin'],
        'inherited_numerical_verification_pin': base['independent_verification'],
        'repair_verification_pin': pin(repair_path), 'repair_manifest_pin': candidate['repair_manifest_pin'],
        'terminal_guard_pin': terminal_pin, 'annuals_reused_exactly': 14,
        'RET_states_reused_exactly': 48, 'new_quote_rows': 0, 'original_constituent_months': 152307,
        'byte_integrity_bytes_hashed_separately': total,
        'scope': 'Inherited 14 annual and RET numerical PASS for unchanged byte-identical inputs; fresh source rejection and full-event metadata guard',
        'unresolved': ['RCM/DISCA learning volume compatibility did not pass; original missing bars retained',
            'Conditional observed-endpoint labels, not certified terminal economic returns',
            'Provisional identity, adjustment vintage and knowledge-time limitations retained'],
        'model_fits': 0, 'preprocessor_fits': 0, 'inference': 0, 'backtests': 0}
    proof = save_new(ROOT / 'run/input_verification.json', receipt)
    final = {**candidate, 'status': 'VERIFIED_PROVISIONAL_FROZEN', 'independent_verification': proof,
        'terminal_guard_pin': terminal_pin, 'canonical_identity_promoted': False,
        'active_pointer_promoted': False, 'frozen_for_fixed_setting_models': True}
    print(json.dumps(save_new(ROOT / 'input_catalog.json', final)))


if __name__ == '__main__':
    main()
