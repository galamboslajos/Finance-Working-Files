"""Stream independent formula/rank/clock checks over generated full-audit ledgers.

No source prices, features, estimators, predictions or portfolio NAV are read.
The full audit's own receipt chain is checked separately by its passive reader.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import pyarrow.parquet as pq


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read_json(path, cap=8 * 1024**2):
    need(path.stat().st_size <= cap, 'Independent metadata byte ceiling')
    return json.loads(path.read_text())


def month_shift(month, offset):
    year, number = map(int, month.split('-'))
    count = year * 12 + number - 1 + offset
    return f'{count // 12:04d}-{count % 12 + 1:02d}'


KEYS = ['source_security_id', 'signal_date', 'side', 'path_variant']
GROUP = ['side', 'path_variant', 'target_family']
FINANCIAL = ['cash_return', 'maximum_drawdown', 'utility', 'target_value', 'target_class']
SIDES = ('long', 'short')
VARIANTS = ('unstopped', 'asset_stop_10pct')
FAMILIES = ('executable_return', 'return_minus_drawdown')
EXPECTED_GROUPS = {(s, v, f) for s in SIDES for v in VARIANTS for f in FAMILIES}


def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def safe_artifact(directory, name, report):
    item = report['artifacts'][name]
    path = (directory / item['path']).resolve()
    need(path.is_relative_to(directory.resolve()) and path.stat().st_size == item['size_bytes']
         and sha(path) == item['sha256'].removeprefix('sha256:'), 'Generated artifact pin mismatch: ' + name)
    return path


def verify_month(directory, report, window, expected_month=None):
    target_path = safe_artifact(directory, 'targets.parquet', report)
    mark_path = safe_artifact(directory, 'marks.parquet', report)
    episode_path = safe_artifact(directory, 'episodes.parquet', report)
    need(sum(p.stat().st_size for p in (target_path, mark_path, episode_path)) <= 32 * 1024**2,
         'Independent monthly derived-byte ceiling')
    target_meta = pq.read_table(target_path, columns=KEYS + ['outcome_month', 'entry_date', 'exit_date',
        'label_available_at', 'target_family', 'prediction_eligible', 'path_available', 'target_available'])
    mark_meta = pq.read_table(mark_path, columns=KEYS + ['date', 'mark'])
    need(target_meta.num_rows <= 16000 and mark_meta.num_rows <= 250000, 'Independent monthly row ceiling')
    expected_grid = [(r['date'], r['mark']) for r in window['required_marks']]
    expected_ticks = set(expected_grid)
    meta = target_meta.to_pylist()
    originals, eligible = set(), set()
    observed_group = defaultdict(list)
    for row in meta:
        key = tuple(row[k] for k in KEYS[:2])
        originals.add(key)
        if row['prediction_eligible']:
            eligible.add(key)
        need(row['signal_date'] == window['signal_date'] and row['outcome_month'] == window['outcome_month']
             and row['entry_date'] == window['entry_date'] and row['exit_date'] == window['exit_date']
             and row['label_available_at'] == window['exit_date'], 'Target completion/calendar mismatch')
        if window['exit_date'] >= '2024-01-01':
            need(not row['path_available'] and not row['target_available'], 'Protected next-open target became observed')
        observed_group[tuple(row[k] for k in GROUP)].append(row)
    need(len(originals) == report['original_rows'] and len(eligible) == report['prediction_eligible_rows'],
         'Original or eligible denominator changed')
    for name, keys in [('original', originals), ('eligible', eligible)]:
        measured = canonical(sorted([list(key) for key in keys]))
        need(measured == report[name + '_key_sha256'], 'Original population key identity changed')
        if expected_month is not None:
            need(measured == expected_month[name + '_key_sha256'], 'Population differs from metadata-first plan')
    need(target_meta.num_rows == len(originals) * 8 and set(observed_group) == EXPECTED_GROUPS,
         'Missing original directional/family rows')
    expected_episodes = {(*key, side, variant) for key in originals for side in SIDES for variant in VARIANTS}
    for rows in observed_group.values():
        need({(r['source_security_id'], r['signal_date']) for r in rows} == originals
             and len(rows) == len(originals), 'Future availability filtered an original cohort')
        need({(r['source_security_id'], r['signal_date']) for r in rows if r['prediction_eligible']} == eligible,
             'Group-specific eligibility changed')
    for row in mark_meta.to_pylist():
        need(row['signal_date'] == window['signal_date'] and row['date'] < '2024-01-01'
             and (row['date'], row['mark']) in expected_ticks and tuple(row[k] for k in KEYS) in expected_episodes,
             'Invalid/protected mark before wealth conversion')
    marks = pq.read_table(mark_path, columns=KEYS + ['date', 'mark', 'wealth']).to_pylist()
    paths = defaultdict(list)
    for row in marks:
        paths[tuple(row[k] for k in KEYS)].append(row)
    statistics = {}
    for key, rows in paths.items():
        rows.sort(key=lambda r: (r['date'], 0 if r['mark'] == 'open' else 1))
        need(len({(r['date'], r['mark']) for r in rows}) == len(rows), 'Duplicate standalone economic mark')
        high, worst = 1.0, 0.0
        for row in rows:
            value = row['wealth']
            need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                 'Nonfinite economic wealth')
            high = max(high, value)
            worst = max(worst, (high - value) / high)
        need(abs(rows[0]['wealth'] - 1.0) <= 1e-12, 'Wrong initial unit capital')
        result = rows[-1]['wealth'] - 1.0
        statistics[key] = (result, worst, result - worst)
    episodes = pq.read_table(episode_path).to_pylist()
    need(len(episodes) == len(originals) * 4, 'Episode denominator differs')
    episode_map = {tuple(row[k] for k in KEYS): row for row in episodes}
    need(len(episode_map) == len(episodes) and set(episode_map) == expected_episodes,
         'Duplicate, missing or foreign original episode')
    for key, row in episode_map.items():
        if row['status'] == 'complete':
            need([(r['date'], r['mark']) for r in paths.get(key, [])] == expected_grid,
                 'Complete episode has an incomplete mark grid')
    for row in meta:
        key = tuple(row[k] for k in KEYS)
        episode = episode_map[key]
        expected_path = (row['prediction_eligible'] and episode['requested'] is True
                         and episode['status'] == 'complete' and window['exit_date'] < '2024-01-01'
                         and key in statistics)
        need(row['path_available'] == expected_path, 'Economic path availability differs from eligible requested completed evidence')
    values = pq.read_table(target_path, columns=KEYS + ['target_family', 'path_available', 'target_available'] + FINANCIAL).to_pylist()
    ranks = defaultdict(list)
    available = 0
    for row in values:
        key = tuple(row[k] for k in KEYS)
        if row['path_available']:
            need(episode_map[key]['status'] == 'complete', 'Unavailable source episode gained a label')
            cash, drawdown, utility = statistics[key]
            for name, expected in zip(('cash_return', 'maximum_drawdown', 'utility'), (cash, drawdown, utility)):
                need(math.isclose(row[name], expected, rel_tol=1e-12, abs_tol=1e-12), 'Independent ' + name + ' differs')
            expected = cash if row['target_family'] == 'executable_return' else utility
            need(math.isclose(row['target_value'], expected, rel_tol=1e-12, abs_tol=1e-12), 'Wrong target family')
            available += 1
        else:
            need(all(row[name] is None for name in FINANCIAL), 'Unavailable outcome was filled')
        if row['path_available']:
            ranks[tuple(row[k] for k in GROUP)].append(row)
        if not row['target_available']:
            need(row['target_class'] is None, 'Withheld class is not null')
    grid = {}
    for group, rows in observed_group.items():
        candidates = ranks.get(group, [])
        exhaustive = all(episode_map[tuple(r[k] for k in KEYS)]['requested'] is True
                         for r in rows if r['prediction_eligible'])
        rank_ready = exhaustive and window['exit_date'] < '2024-01-01' and len(candidates) >= 10
        for row in rows:
            need(row['target_available'] == (rank_ready and row['path_available']),
                 'Class availability differs from full eligible observed ranking pool')
        ranked = candidates if rank_ready else []
        counts = Counter()
        def expected_value(row):
            measured = statistics[tuple(row[k] for k in KEYS)]
            return measured[0] if row['target_family'] == 'executable_return' else measured[2]
        ordered = sorted(ranked, key=expected_value, reverse=True)
        lower, n = 0, len(ordered)
        while lower < n:
            upper = lower + 1
            while upper < n and expected_value(ordered[upper]) == expected_value(ordered[lower]):
                upper += 1
            class_id = min(10, 1 + (10 * (lower + upper)) // (2 * n))
            for row in ordered[lower:upper]:
                need(row['target_class'] == class_id, 'Independent exact-midrank class differs')
                counts[class_id] += 1
            lower = upper
        for class_id in range(1, 11):
            grid[(window['outcome_month'], *group, class_id)] = counts[class_id]
    expected_status = Counter(row['status'] for row in episodes)
    need(dict(expected_status) == report['episode_status_counts'], 'Monthly status summary differs')
    need(sum(r['members'] for r in report['class_grid']) == sum(sum(v['target_available'] for v in rows) for rows in observed_group.values()),
         'Class coverage summary differs')
    reported_grid = {(window['outcome_month'], r['side'], r['path_variant'], r['target_family'], r['class_id']): r['members']
                     for r in report['class_grid']}
    need(len(reported_grid) == len(report['class_grid']) == 80 and reported_grid == grid,
         'Individual class coverage cells differ')
    groups = {group: sum(r['target_available'] for r in rows) for group, rows in observed_group.items()}
    return {'original_rows': len(originals), 'eligible_rows': len(eligible), 'target_rows': len(values),
            'available_values': available, 'grid': grid, 'groups': groups,
            'derived_bytes': sum(p.stat().st_size for p in (target_path, mark_path, episode_path))}


def verify(directory):
    directory = Path(directory).resolve()
    destination = directory / 'independent_full_verification.json'
    need(not destination.exists(), 'Preserve existing verification receipts')
    plan = read_json(directory / 'reviewed_plan.json')
    audit = read_json(directory / 'audit.json')
    need(audit['status'] == 'COMPLETE_FULL_PATH_AUDIT' and audit['plan_sha256'] == plan['plan_sha256'], 'Full audit incomplete or mismatched')
    need(not audit['models_fitted'] and not audit['financial_2024_plus_read'], 'Unexpected fitting/holdout scope')
    totals = Counter()
    grid = {}
    monthly = []
    started = time.monotonic()
    for annual in plan['annuals']:
        for window in annual['windows']:
            parent = directory / 'years' / str(annual['year']) / 'months' / window['outcome_month']
            checkpoint = read_json(parent / 'checkpoint.json')
            attempt = (parent / checkpoint['attempt']).resolve()
            need(attempt.parent == parent.resolve() and checkpoint['plan_sha256'] == plan['plan_sha256'], 'Unsafe/mismatched monthly checkpoint')
            report_path = attempt / 'audit.json'
            need(sha(report_path) == checkpoint['audit_sha256'], 'Monthly audit hash changed')
            report = read_json(report_path)
            need(report['plan_sha256'] == plan['plan_sha256'] and report['outcome_month'] == window['outcome_month'], 'Wrong monthly audit')
            expected_month = next(m for m in annual['monthly'] if m['outcome_month'] == window['outcome_month'])
            checked = verify_month(attempt, report, window, expected_month)
            grid.update(checked.pop('grid'))
            groups = checked.pop('groups')
            totals.update(checked)
            monthly.append((report, window, groups))
        print(json.dumps({'stage': 'independent_year_verified', 'year': annual['year'], 'target_rows': totals['target_rows']}), flush=True)
    need(len(monthly) == 168 and totals['original_rows'] == plan['read_plan']['original_rows']
         and totals['eligible_rows'] == plan['read_plan']['prediction_eligible_rows'], 'Full original coverage differs')
    readiness = pq.read_table(directory / 'payoff_readiness.parquet').to_pylist()
    expected_readiness = {(w['signal_date'], *g) for _, w, _ in monthly for g in EXPECTED_GROUPS}
    actual_readiness = {(r['signal_date'], r['side'], r['path_variant'], r['target_family']) for r in readiness}
    need(len(readiness) == len(actual_readiness) == 168 * 8 and actual_readiness == expected_readiness,
         'Payoff readiness grid truncated or duplicated')
    for row in readiness:
        signal = row['signal_date']
        end = month_shift(signal[:7], -1)
        months = [month_shift(end, offset) for offset in range(-119, 1)]
        missing = sum(grid.get((m, row['side'], row['path_variant'], row['target_family'], c), 0) == 0
                      for m in months for c in range(1, 11))
        need(row['required_calendar_months'] == 120 and row['required_class_cells'] == 1200
             and row['window_start'] == months[0] and row['window_end'] == end and row['missing_class_cells'] == missing
             and row['status'] == ('available' if missing == 0 else 'withheld_incomplete_120_month_class_grid'),
             'Causal 120-completed-month readiness differs')
    training = pq.read_table(directory / 'training_completion.parquet').to_pylist()
    need(len(training) == 32, 'Training completion grid truncated')
    seen = set()
    for row in training:
        group = (row['side'], row['path_variant'], row['target_family'])
        key = (row['test_year'], *group)
        need(key not in seen and 2020 <= row['test_year'] <= 2023, 'Invalid/duplicate training group')
        seen.add(key)
        start, end = f"{row['test_year']-10}-01", f"{row['test_year']-1}-12"
        cutoff = next(w['signal_date'] for _, w, _ in monthly if w['outcome_month'] == f"{row['test_year']}-01")
        selected = [(r, w, groups[group]) for r, w, groups in monthly if start <= w['outcome_month'] <= end]
        expected = {
            'original_training_rows': sum(r['original_rows'] for r, _, _ in selected),
            'signal_eligible_training_rows': sum(r['prediction_eligible_rows'] for r, _, _ in selected),
            'completed_supervised_rows': sum(n for _, w, n in selected if w['exit_date'] <= cutoff),
            'purged_not_yet_available_label_rows': sum(n for _, w, n in selected if w['exit_date'] > cutoff),
            'missing_path_or_rank_label_rows': sum(r['prediction_eligible_rows']-n for r, _, n in selected),
        }
        need(row['as_of'] == cutoff and row['train_start_month'] == start and row['train_end_month'] == end
             and all(row[k] == value for k, value in expected.items()), 'Independent training completion purge differs')
    result = {'status': 'PASS_FULL_FORMULAS_RANKS_CLOCK_AND_ORIGINAL_DENOMINATORS', 'audit_sha256': sha(directory / 'audit.json'),
        'plan_sha256': plan['plan_sha256'], 'months_checked': len(monthly), **dict(totals),
        'payoff_readiness_rows_checked': len(readiness), 'training_completion_rows_checked': len(training),
        'source_prices_read': False, 'feature_values_read': False,
        'models_fitted': False, 'financial_2024_plus_read': False, 'source_identity_and_action_certification': False,
        'wall_seconds': time.monotonic() - started, 'worker_sha256': sha(Path(__file__))}
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
