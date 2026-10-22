"""Passive, standard-library reader for independently checked ratio-label audits.

Opens only bounded JSON receipts and aggregate reports. No Parquet, feature,
price, model or estimator import is possible through this module.
"""
import hashlib
import json
from pathlib import Path


def _need(value, message):
    if not value:
        raise ValueError(message)


def _json(path):
    p = Path(path)
    _need(p.stat().st_size <= 16 * 1024**2, 'Saved report exceeds JSON ceiling')
    return json.loads(p.read_text())


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _pin(item, root=None):
    _need(set(item) == {'path','size_bytes','sha256'}, 'Closed saved pin schema required')
    _need(type(item['size_bytes']) is int and 0 <= item['size_bytes'] <= 16 * 1024**2, 'Pinned report exceeds metadata ceiling')
    p = Path(item['path'])
    if root is not None:
        _need(not p.is_absolute(), 'Relative saved report path required')
        p = (root / p).resolve()
        _need(p.is_relative_to(root), 'Unsafe saved report path')
    _need(p.is_absolute(), 'Absolute saved evidence path required')
    _need(p.is_file() and p.stat().st_size == item['size_bytes'] and _sha(p) == item['sha256'],
          'Saved verification/report bytes changed: ' + str(p))
    return p


def read_return_dd_audit(directory):
    """Authenticate independent proof before opening five small aggregate tables."""
    root = Path(directory).resolve()
    proof = _json(root / 'independent_verification.json')
    _need(proof.get('schema') == 'dmv2-return-dd-independent-verification/v1'
          and proof.get('status') == 'PASS' and proof.get('months_checked') == 168
          and proof.get('directional_rows_checked') == 304614
          and proof.get('models_fitted') == 0
          and proof.get('portfolio_backtest_run') is False
          and proof.get('financial_2024_plus_read') is False,
          'Complete independent ratio-label verification required')
    vp = _json(_pin(proof['verification_plan_pin']))
    _pin(proof['plan_pin'])
    _need(vp['schema'] == proof['schema'] and vp['worker_pin'] == proof['worker_pin'],
          'Independent verifier identity changed')
    _pin(proof['worker_pin'])
    _need(vp['run_directory'] == str(root) and vp['plan_pin'] == proof['plan_pin'],
          'Independent proof belongs to a different run')
    audit_path = _pin(vp['artifacts']['audit.json'])
    _need(audit_path == root / 'audit.json', 'Wrong audited result path')
    audit = _json(audit_path)
    _need(audit['status'] == 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
          and audit['reviewed_plan_pin'] == proof['plan_pin']
          and audit['observed_through'] == '2023-12-29'
          and audit['denominator_floor'] == .01
          and audit['months'] == 168 and audit['original_rows'] == 152307
          and audit['direction_rows'] == 304614
          and audit['financial_2024_plus_read'] is False
          and audit['models_fitted'] is False and audit['portfolio_backtest_run'] is False,
          'Producer audit scope differs from independent proof')
    result = {'status': 'INDEPENDENTLY_VERIFIED_LABEL_AUDIT', 'audit': audit, 'verification': proof}
    for key, name, count in [('coverage','coverage.json',336),
                             ('class_changes','class_changes.json',336),
                             ('payoff_readiness','payoff_readiness.json',96),
                             ('training_completion','training_completion.json',8)]:
        p = _pin(audit['artifacts'][name], root)
        _need(p == _pin(vp['artifacts'][name]), 'Report not bound by independent proof')
        rows = _json(p)
        _need(type(rows) is list and len(rows) == count, 'Truncated saved aggregate: ' + name)
        result[key] = rows
    summary = []
    for side in ('long', 'short'):
        rows = [r for r in result['coverage'] if r['side'] == side]
        changes = [r for r in result['class_changes'] if r['side'] == side]
        states = [r for r in result['payoff_readiness'] if r['side'] == side]
        summary.append({'side': side, 'original_stock_months': sum(r['original_members'] for r in rows),
            'prediction_eligible_stock_months': sum(r['prediction_eligible_members'] for r in rows),
            'observed_ratio_labels': sum(r['labelled_members'] for r in rows),
            'floor_hit_count': sum(r['floor_hit_count'] for r in rows),
            'negative_return_count': sum(r['negative_return_count'] for r in rows),
            'exact_minus_one_count': sum(r['minus_one_ratio_count'] for r in rows),
            'largest_exact_tie_count': max(r['largest_exact_tie_count'] for r in rows),
            'completed_months_with_empty_classes': sum(bool(r['empty_classes']) and r['rank_status']=='available' for r in rows),
            'class_comparison_rows': sum(r['class_comparison_rows'] for r in changes),
            'class_changed_rows': sum(r['class_changed_rows'] for r in changes),
            'payoff_vectors_available': sum(r['vector_available'] for r in states),
            'payoff_vectors_requested': len(states)})
    result['summary'] = summary
    return result
