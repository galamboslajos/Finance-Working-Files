"""Independent bounded numerical check; never reads source prices or fits models."""
from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(directory):
    directory = Path(directory).resolve()
    receipt = directory / 'independent_formula_verification.json'
    need(not receipt.exists(), 'Preserve previous verification receipts')
    audit_path = directory / 'audit.json'
    need(audit_path.stat().st_size <= 1024 * 1024, 'Audit metadata budget')
    audit = json.loads(audit_path.read_text())
    need(audit['status'] == 'COMPLETE_BOUNDED_AUDIT' and not audit['models_fitted']
         and not audit['financial_2024_plus_read'], 'Wrong audit scope')
    plan_path = directory / 'reviewed_plan.json'
    need(plan_path.stat().st_size <= 8 * 1024**2, 'Reviewed plan metadata budget')
    plan = json.loads(plan_path.read_text())
    need(plan['plan_sha256'] == audit['plan_sha256'], 'Audit/plan identity differs')
    grids = {w['signal_date']: [(m['date'], m['mark']) for m in w['required_marks']]
             for w in plan['windows']}
    paths = {}
    for name in ('marks.parquet', 'targets.parquet', 'episodes.parquet'):
        pin = audit['artifacts'][name]
        path = (directory / pin['path']).resolve()
        need(path.is_relative_to(directory) and path.stat().st_size == pin['size_bytes']
             and sha(path) == pin['sha256'].removeprefix('sha256:'), 'Artifact pin mismatch')
        paths[name] = path
    need(sum(p.stat().st_size for p in paths.values()) < 32 * 1024**2, 'Derived read byte ceiling')
    keys = ['source_security_id', 'signal_date', 'side', 'path_variant']
    meta = pq.read_table(paths['marks.parquet'], columns=keys + ['date', 'mark'])
    need(meta.num_rows <= 400000, 'Derived path row ceiling')
    for row in meta.to_pylist():
        need(row['signal_date'] in ('2019-12-31', '2020-09-30') and '2020-01-01' <= row['date'] < '2021-01-01'
             and row['mark'] in ('open', 'close'), 'Derived path leaves approved dates')
    target_meta = pq.read_table(paths['targets.parquet'], columns=keys + ['outcome_month', 'label_available_at'])
    need(target_meta.num_rows <= 15000, 'Derived target row ceiling')
    for row in target_meta.to_pylist():
        expected = '2020-02-03' if row['outcome_month'] == '2020-01' else '2020-11-02'
        need(row['outcome_month'] in ('2020-01', '2020-10') and row['label_available_at'] == expected,
             'Wrong scheduled completion clock')
    marks = pq.read_table(paths['marks.parquet'], columns=keys + ['date', 'mark', 'wealth']).to_pylist()
    groups = defaultdict(list)
    for row in marks:
        groups[tuple(row[k] for k in keys)].append(row)
    statistics = {}
    for key, rows in groups.items():
        rows.sort(key=lambda r: (r['date'], 0 if r['mark'] == 'open' else 1))
        need(len({(r['date'], r['mark']) for r in rows}) == len(rows), 'Duplicate mark')
        need(abs(rows[0]['wealth'] - 1) <= 1e-12, 'Unit capital mismatch')
        high, worst = 1.0, 0.0
        for row in rows:
            value = row['wealth']
            need(isinstance(value, (int, float)) and math.isfinite(value), 'Nonfinite observed wealth')
            high = max(high, value)
            worst = max(worst, (high - value) / high)
        cash = rows[-1]['wealth'] - 1
        statistics[key] = (cash, worst, cash - worst)
    episodes = pq.read_table(paths['episodes.parquet']).to_pylist()
    for row in episodes:
        if row['status'] == 'complete':
            key = tuple(row[k] for k in keys)
            ticks = [(m['date'], m['mark']) for m in groups.get(key, [])]
            need(ticks == grids[row['signal_date']], 'Completed episode grid differs from reviewed calendar')
    targets = pq.read_table(paths['targets.parquet']).to_pylist()
    ranked = defaultdict(list)
    checked = 0
    for row in targets:
        key = tuple(row[k] for k in keys)
        if row['path_available']:
            expected = statistics[key]
            for field, value in zip(('cash_return', 'maximum_drawdown', 'utility'), expected):
                need(math.isclose(row[field], value, rel_tol=1e-12, abs_tol=1e-12), 'Independent '+field+' differs')
            value = expected[0] if row['target_family'] == 'executable_return' else expected[2]
            need(math.isclose(row['target_value'], value, rel_tol=1e-12, abs_tol=1e-12), 'Wrong target family value')
            if row['target_available']:
                ranked[(row['signal_date'], row['side'], row['path_variant'], row['target_family'])].append(row)
            checked += 1
        else:
            need(all(row[f] is None for f in ('cash_return', 'maximum_drawdown', 'utility', 'target_value', 'target_class')),
                 'Unavailable path gained a numerical target')
    for group in ranked.values():
        group.sort(key=lambda r: r['target_value'], reverse=True)
        n = len(group)
        for row in group:
            lower = sum(other['target_value'] > row['target_value'] for other in group)
            tied = sum(other['target_value'] == row['target_value'] for other in group)
            expected = min(10, 1 + (10 * (2 * lower + tied)) // (2 * n))
            need(row['target_class'] == expected, 'Independent exact midrank differs')
    need(len(episodes) == audit['original_rows'] * 4, 'Original episode denominator changed')
    need(len(targets) == audit['original_rows'] * 8, 'Original target denominator changed')
    need(sum(r['status'] == 'complete' for r in episodes) == audit['complete_episode_count'], 'Coverage count mismatch')
    result = {'status': 'PASS_BOUNDED_FORMULAS_AND_CLOCK', 'audit_sha256': sha(audit_path),
              'source_prices_read': False, 'model_fits': 0, 'full_training_history_certified': False,
              'target_rows_checked': len(targets), 'available_target_values_checked': checked,
              'class_groups_checked': len(ranked), 'independent_action_entitlement_certification': False,
              'worker_sha256': sha(Path(__file__))}
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
