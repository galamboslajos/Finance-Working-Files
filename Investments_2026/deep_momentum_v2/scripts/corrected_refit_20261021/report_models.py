#!/usr/bin/env python3
"""Passive saved-report comparison, only after all twenty fits independently pass.

No estimator, source panel, probability array or historical model is loaded.
Planning uses safe control JSON and Parquet footers. Numerical receipt/table
reads require a sealed exact plan, shared reservation and resource supervisor.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys

from control import ROOT, check, need, pin, read, require_reservation, save_new

YEARS = (2020, 2021, 2022, 2023)
ARMS = ('rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended', 'rf_tuned_extended')
LABELS = ('RF · Paper15', 'XGBoost · Paper15', 'RF · expanded', 'XGBoost · expanded', 'RF expanded · prior CV settings')
CLASSES = (1, 2, 9, 10)
KEYS = ('source_security_id', 'signal_date')
META = (*KEYS, 'outcome_month', 'outcome_end', 'label_available')
VALUES = (*KEYS, 'true_class', 'predicted_class', 'log_loss', 'correct')
MONTHLY = ('outcome_month', 'prediction_rows', 'labelled_rows', 'missing_label_rows',
           'label_coverage', 'log_loss', 'accuracy', 'status')
MODES = ('period', 'class_id', 'prediction_rows', 'predicted_support', 'labelled_predicted_support',
         'missing_label_predicted_support', 'actual_support', 'true_positive_count', 'precision', 'recall', 'status')
MAX_READ = 128 * 1024**2


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def projection(item, columns):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(check(item)); columns = list(columns)
    need(set(columns) <= set(f.schema_arrow.names), 'Saved report projection missing fields')
    sizes = []
    for i in range(f.metadata.num_row_groups):
        group = f.metadata.row_group(i); found = collections.defaultdict(int)
        for j in range(group.num_columns):
            field = group.column(j)
            need(field.total_compressed_size >= 0, 'Invalid footer byte count')
            found[field.path_in_schema.split('.')[0]] += field.total_compressed_size
        need(set(columns) <= set(found), 'Requested root absent from footer leaves')
        sizes.append(sum(found[c] for c in columns))
    return {'pin': item, 'columns': columns, 'rows': f.metadata.num_rows,
            'schema_fields': {c: str(f.schema_arrow.field(c).type) for c in columns},
            'projected_compressed_bytes': sum(sizes)}


def completion_gate(run):
    """Safe controller metadata only; numerical proof is parsed after charging."""
    run = Path(run).resolve()
    need(run == ROOT / 'run/models', 'Unexpected model namespace')
    pp = pin(run / 'pipeline/progress.json'); p = read(pp['path'])
    need(p['status'] == 'COMPLETE_VERIFIED' and p['completed_estimator_fits'] == 20
         and p['completed_preprocessors'] == 8 and p['completed_phases'] == 50
         and p['completed_annual_verifications'] == 4, 'All twenty fits and final verification must finish first')
    final = pin(run.parent / 'verification/models.json')
    need(p['final_verification_pin'] == final, 'Final proof differs from completed controller')
    need(len(list(run.glob('folds/*/arms/*/FIT_ATTEMPT.json'))) == 20
         and len(list(run.glob('folds/*/arms/*/FIT_COMPLETE.json'))) == 20
         and len(list(run.glob('folds/*/arms/*/PREDICT_COMPLETE.json'))) == 20,
         'Exactly twenty consumed/completed fit and prediction artifacts required')
    for item in p['checkpoint_pins'].values(): check(item)
    return pp, final


def make_plan(protocol_path, output):
    need(Path(output).resolve().is_relative_to(ROOT / 'plans'), 'Report plan must use the new scoped plan namespace')
    protocol_path = Path(protocol_path).resolve(); p = read(protocol_path)
    need(p['schema'] == 'dmv2-corrected-refit-protocol/v1' and p['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and p['outcome_years'] == list(YEARS) and p['parameter_search'] is False
         and p['protected_2024_plus'] is True, 'Frozen no-search protocol required')
    run = Path(p['run_directory']).resolve(); progress_pin, final_pin = completion_gate(run)
    for name in ('worker_pin', 'verification_worker_pin', 'control_pin', 'supervisor_pin', 'catalog_pin'):
        check(p[name])
    jsons = {'final_proof': final_pin}; annuals = {}; fits = {}
    for year in YEARS:
        jsons[f'proof_{year}'] = pin(run.parent / 'verification' / f'models_{year}.json')
        arms = {}
        for arm in ARMS:
            base = run / 'folds' / str(year) / 'arms' / arm; prefix = f'{year}/{arm}'
            fits[prefix] = pin(base / 'FIT_COMPLETE.json')
            jsons[prefix + '/prediction'] = pin(base / 'PREDICT_COMPLETE.json')
            jsons[prefix + '/summary'] = pin(base / 'reports/summary.json')
            item = pin(base / 'reports/row_scores.parquet')
            arms[arm] = {'metadata': projection(item, META), 'values': projection(item, VALUES),
                'monthly_metadata': projection(pin(base / 'reports/monthly.parquet'), ('outcome_month',)),
                'mode_metadata': projection(pin(base / 'reports/mode_class_performance.parquet'), ('period','class_id')),
                'monthly': projection(pin(base / 'reports/monthly.parquet'), MONTHLY),
                'modes': projection(pin(base / 'reports/mode_class_performance.parquet'), MODES)}
            need(arms[arm]['metadata']['rows'] <= 20000 and arms[arm]['monthly']['rows'] == 12
                 and arms[arm]['modes']['rows'] == 130, 'Saved report annual envelope changed')
        annuals[str(year)] = arms
    cost = sum(x['size_bytes'] for x in jsons.values()) + sum(
        z['projected_compressed_bytes'] for arms in annuals.values() for specs in arms.values() for z in specs.values())
    need(0 < cost <= MAX_READ, 'Passive report read ceiling exceeded')
    return save_new(output, {'schema': 'dmv2-corrected-model-report-plan/v1', 'status': 'SEALED_BEFORE_REPORT_READ',
        'protocol_pin': pin(protocol_path), 'catalog_pin': p['catalog_pin'], 'source_plan_pin': pin(run / 'read_plan.json'),
        'progress_pin': progress_pin, 'worker_pin': pin(__file__), 'control_pin': pin(ROOT / 'control.py'),
        'supervisor_pin': pin(ROOT / 'supervise.py'), 'run': str(run), 'jsons': jsons, 'fit_pins': fits,
        'annuals': annuals, 'projected_reads': {'compressed_source': cost, 'derived_matrix': 0},
        'model_fits': 0, 'inference_calls': 0, 'old_model_reads': 0,
        'block_decode_disclosure': 'Approved new saved-report column chunks may decode whole blocks; no source panels or probability arrays are opened.'})


def read_table(spec):
    import pyarrow.parquet as pq
    need(projection(spec['pin'], spec['columns']) == spec, 'Report footer/pin changed')
    return pq.ParquetFile(spec['pin']['path']).read(columns=spec['columns'], use_threads=False).to_pylist()


def validate_metadata(rows, year):
    keys = []
    for r in rows:
        need(isinstance(r['source_security_id'], str) and r['source_security_id'].strip(), 'Invalid source key')
        month, signal, end = (r[k] for k in ('outcome_month', 'signal_date', 'outcome_end'))
        need(isinstance(month, str) and re.fullmatch(r'\d{4}-\d{2}', month) and month[:4] == str(year)
             and 1 <= int(month[-2:]) <= 12, 'Report month outside approved year')
        import datetime as dt
        dt.date.fromisoformat(signal); dt.date.fromisoformat(end)
        m = int(month[-2:]); previous = f'{year-1}-12' if m == 1 else f'{year}-{m-1:02d}'
        need(signal[:7] == previous and signal < month + '-01' and end[:7] == month
             and end < '2024-01-01' and type(r['label_available']) is bool, 'Report timing/availability invalid')
        keys.append(tuple(r[c] for c in KEYS))
    need(len(keys) == len(set(keys)) and {r['outcome_month'] for r in rows} == {f'{year}-{m:02d}' for m in range(1,13)},
         'Duplicate keys or incomplete twelve-month report grid')
    return [(r['source_security_id'], r['signal_date'], r['outcome_month'], r['label_available']) for r in rows]


def validate_table_grids(monthly, modes, year):
    months=[f'{year}-{m:02d}' for m in range(1,13)]
    need([r['outcome_month'] for r in monthly]==months, 'Monthly metadata must match twelve approved ordered months')
    grid=[(r['period'],r['class_id']) for r in modes]
    need(len(grid)==130 and len(set(grid))==130
         and set(grid)=={(period,c) for period in ['all',*months] for c in range(1,11)},
         'Mode metadata must match the exact thirteen-period ten-class grid')


def close(a, b, label):
    if a is None or b is None: need(a is None and b is None, label + ': missingness differs')
    else: need(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= 2e-12,
               label + ': numerical value differs')


def aggregate(rows, *, months):
    """Aggregate saved row scores; does not access probabilities or create labels."""
    monthly = []
    for month in months:
        group = [r for r in rows if r['outcome_month'] == month]
        observed = [r for r in group if r['label_available']]
        monthly.append({'outcome_month': month, 'prediction_rows': len(group), 'labelled_rows': len(observed),
            'missing_label_rows': len(group)-len(observed), 'label_coverage': len(observed)/len(group) if group else None,
            'log_loss': math.fsum(r['log_loss'] for r in observed)/len(observed) if observed else None,
            'accuracy': math.fsum(r['correct'] for r in observed)/len(observed) if observed else None,
            'status': 'available' if observed else 'no_observed_labels'})
    seen = [r for r in rows if r['label_available']]
    found = [m for m in monthly if m['labelled_rows']]
    result = {'prediction_rows': len(rows), 'labelled_rows': len(seen), 'missing_label_rows': len(rows)-len(seen),
        'label_coverage': len(seen)/len(rows) if rows else None, 'outcome_months': len(months),
        'observed_label_months': len(found),
        'pooled_log_loss': math.fsum(r['log_loss'] for r in seen)/len(seen) if seen else None,
        'pooled_accuracy': math.fsum(r['correct'] for r in seen)/len(seen) if seen else None,
        'mean_monthly_log_loss': math.fsum(m['log_loss'] for m in found)/len(found) if found else None,
        'mean_monthly_accuracy': math.fsum(m['accuracy'] for m in found)/len(found) if found else None}
    classes = {}
    for c in CLASSES:
        predicted = [r for r in rows if r['predicted_class'] == c]
        observed = [r for r in predicted if r['label_available']]
        actual = [r for r in seen if r['true_class'] == c]
        hits = sum(r['true_class'] == c for r in observed)
        classes[str(c)] = {'predicted_support': len(predicted), 'labelled_predicted_support': len(observed),
            'missing_label_predicted_support': len(predicted)-len(observed), 'actual_support': len(actual),
            'true_positive_count': hits, 'precision': hits/len(observed) if observed else None,
            'recall': hits/len(actual) if actual else None}
    return result, monthly, classes


def merge_scores(meta, values):
    need([[r[k] for k in KEYS] for r in meta] == [[r[k] for k in KEYS] for r in values], 'Saved score key/order mismatch')
    merged = []
    for m, v in zip(meta, values):
        pred = v['predicted_class']; true = v['true_class']
        need(type(pred) is int and 1 <= pred <= 10, 'Invalid saved mode class')
        if m['label_available']:
            need(type(true) is int and 1 <= true <= 10 and v['log_loss'] is not None
                 and math.isfinite(v['log_loss']) and v['log_loss'] >= 0
                 and v['correct'] in (0, 1) and v['correct'] == (true == pred), 'Invalid observed saved score')
        else:
            need(true is None and v['log_loss'] is None and v['correct'] is None, 'Missing label acquired an invented score')
        merged.append({**m, **v})
    return merged


def assert_summary(actual, expected):
    for k, v in expected.items(): close(actual[k], v, k)


def require_guard(plan_pin, plan):
    path = os.environ.get('DMV2_GUARD_LAUNCH_PATH'); sha = os.environ.get('DMV2_GUARD_LAUNCH_SHA256')
    need(path and sha and pin(path)['sha256'] == sha, 'Authenticated resource supervisor required')
    z = read(path)
    need(z['schema'] == 'dmv2-corrected-refit-phase-launch/v1' and z['protocol_pin'] == plan['protocol_pin']
         and z['action_plan_pin'] == plan_pin and plan['worker_pin'] in z['code_pins']
         and plan['control_pin'] in z['code_pins'] and plan['supervisor_pin'] in z['code_pins']
         and z['command'] == [sys.executable, *sys.argv], 'Resource launch differs from exact passive-report command')


def markdown(report):
    pct = lambda x: '—' if x is None else f'{x:.2%}'
    num = lambda x: '—' if x is None else f'{x:.6f}'
    lines = ['Verified saved prediction diagnostics for January2020–December2023 (48 outcome months).',
        'All five arms use the same prediction and observed-label source/signal keys. Monthly return classes are unchanged; class1 is best and class10 worst.', '',
        '| Model | Pooled log loss | Equal-month log loss | Pooled accuracy | Equal-month accuracy | Label coverage |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS):
        s=report['arms'][arm]['overall']; lines.append(f"| {label} | {num(s['pooled_log_loss'])} | {num(s['mean_monthly_log_loss'])} | {pct(s['pooled_accuracy'])} | {pct(s['mean_monthly_accuracy'])} | {pct(s['label_coverage'])} |")
    lines += ['', 'Pooled weights each observed stock-month equally. Equal-month weights each observed month equally. Label coverage is observed outcomes divided by prediction-eligible stock-months; missing outcomes stay in that denominator.', '',
        '| Model | Year | Observed / predicted | Pooled log loss | Equal-month log loss | Accuracy |', '|---|---:|---:|---:|---:|---:|']
    for arm, label in zip(ARMS,LABELS):
        for year in YEARS:
            s=report['arms'][arm]['years'][str(year)]
            lines.append(f"| {label} | {year} | {s['labelled_rows']:,} / {s['prediction_rows']:,} | {num(s['pooled_log_loss'])} | {num(s['mean_monthly_log_loss'])} | {pct(s['pooled_accuracy'])} |")
    lines += ['', '| Model | Class1 P / R (actual n) | Class2 P / R (actual n) | Class9 P / R (actual n) | Class10 P / R (actual n) |', '|---|---:|---:|---:|---:|']
    for arm,label in zip(ARMS,LABELS):
        cells=[]
        for c in CLASSES:
            z=report['arms'][arm]['classes'][str(c)];cells.append(f"{pct(z['precision'])} / {pct(z['recall'])} ({z['actual_support']:,})")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', 'These are ten-class mode precision and recall, with separate classes and no merged tails. They are not top-K basket statistics or portfolio returns.',
        'Tail precision/recall are aggregated from pinned saved row scores and cross-checked against saved mode tables, following the independently verified upstream probability lineage. This reporter does not reconstruct per-row scores from probabilities.',
        'The fifth arm uses prior annual CV settings, newly fitted to the repaired inputs. No new parameter search or repaired-input CV selection was performed.',
        'Inputs retain the provisional BK/XOM provider-adjusted endpoint policy. The proposed RCM/DISCA learning repairs were rejected by frozen volume controls; earlier execution repairs remain separate.',
        'Historical knowledge-time, provider adjustment vintage, terminal-return and survivorship limitations remain provisional. Passing these checks does not certify investable performance.',
        'Old runs are preserved. This report makes no paired-cohort improvement claim against historical runs and launches no RET scoring, model fitting or backtest.', '']
    return '\n'.join(lines)


def execute(plan_path, expected_sha, reservation, output):
    plan_pin = pin(plan_path); need(plan_pin['sha256'] == expected_sha, 'Report plan changed')
    z = read(plan_path)
    need(z['schema'] == 'dmv2-corrected-model-report-plan/v1' and z['worker_pin'] == pin(__file__)
         and z['projected_reads']['derived_matrix'] == 0
         and 0 < z['projected_reads']['compressed_source'] <= MAX_READ, 'Passive report plan/code invalid')
    require_reservation(ROOT, reservation, z['projected_reads']['compressed_source'], plan_pin, 'compressed_source')
    require_guard(plan_pin, z); run = Path(z['run']); output = Path(output).resolve()
    need(output.is_relative_to(ROOT / 'run') and not output.exists(), 'Fresh scoped report directory required')
    save_new(str(output)+'.STARTED.json',{'status':'ATTEMPT_CONSUMED','plan_pin':plan_pin,'model_fits':0,'inference_calls':0})
    progress_pin, final_pin = completion_gate(run)
    need(progress_pin == z['progress_pin'] and final_pin == z['jsons']['final_proof'], 'Completion identity changed')
    for k in ('protocol_pin','catalog_pin','source_plan_pin','control_pin','supervisor_pin'): check(z[k])
    protocol = read(check(z['protocol_pin']))
    need(protocol['catalog_pin']==z['catalog_pin'] and protocol['run_directory']==str(run)
         and protocol['verification_worker_pin']['path']==str(ROOT/'verify_models.py'),
         'Report plan differs from the sealed input/runtime identity')
    proofs = {name: read(check(item)) for name,item in z['jsons'].items()}
    final = proofs['final_proof']
    need(final['schema'] == 'dmv2-independent-corrected-model-final-verification/v1' and final['status'] == 'PASS'
         and final['verified_estimator_fits'] == 20 and final['verified_training_only_preprocessing_states'] == 8
         and final['outcome_months'] == 48 and final['outcome_years'] == list(YEARS)
         and final['same_cohort_all_arms'] is True and final['source_plan_pin'] == z['source_plan_pin']
         and final['protocol_pin'] == z['protocol_pin'] and final['catalog_pin'] == z['catalog_pin']
         and final['worker_pin']==protocol['verification_worker_pin']
         and set(final['comparative_prediction_metrics'])==set(ARMS), 'Final independent proof is not the approved twenty-fit PASS')
    for year in YEARS:
        q=proofs[f'proof_{year}']
        need(q['status']=='PASS' and q['year']==year and q['protocol_pin']==z['protocol_pin']
             and q['source_plan_pin']==z['source_plan_pin'] and set(q['arms'])==set(ARMS)
             and q['catalog_pin']==z['catalog_pin'] and q['worker_pin']==protocol['verification_worker_pin']
             and final['annual_verification_pins'][str(year)]==z['jsons'][f'proof_{year}'], 'Annual proof mismatch')
        for arm in ARMS:
            prefix=f'{year}/{arm}';a=q['arms'][arm]; receipt=proofs[prefix+'/prediction'];spec=z['annuals'][str(year)][arm]
            need(a['fit_receipt_pin']==z['fit_pins'][prefix] and a['prediction_receipt_pin']==z['jsons'][prefix+'/prediction']
                 and receipt['status']=='COMPLETE' and receipt['year']==year and receipt['arm']==arm,
                 'Report receipt is not the independently verified prediction')
            check(z['fit_pins'][prefix])
            for name,key in (('row_scores','metadata'),('monthly','monthly'),('mode_class_performance','modes')):
                need(receipt['artifacts'][name]==spec[key]['pin'], 'Saved report artifact pin differs')
            need(receipt['artifacts']['summary']==z['jsons'][prefix+'/summary'], 'Saved summary pin differs')
    # All dates/keys/availability across every arm are checked before score values.
    metadata={}; reference={}
    for year in YEARS:
        for arm,spec in z['annuals'][str(year)].items():
            rows=read_table(spec['metadata']); observed=validate_metadata(rows,year)
            validate_table_grids(read_table(spec['monthly_metadata']),read_table(spec['mode_metadata']),year)
            if year in reference: need(observed==reference[year], 'Five arms do not share observed prediction keysets')
            else: reference[year]=observed
            metadata[(year,arm)]=rows
    result={'schema':'dmv2-corrected-model-report/v1','status':'COMPLETE_FROM_INDEPENDENTLY_VERIFIED_OUTPUTS',
        'plan_pin':plan_pin,'final_verification_pin':final_pin,'protocol_pin':z['protocol_pin'],'catalog_pin':z['catalog_pin'],
        'same_observed_keys_all_arms':True,'model_fits':0,'inference_calls':0,'backtests':0,'arms':{},
        'observed_key_fingerprints':{str(y):fingerprint([r for r in reference[y] if r[-1]]) for y in YEARS},
        'prediction_key_fingerprints':{str(y):fingerprint(reference[y]) for y in YEARS},
        'historical_runs_used_as_paired_comparison':False}
    truth_reference={}
    for arm in ARMS:
        all_rows=[];yearly={}
        for year in YEARS:
            spec=z['annuals'][str(year)][arm]
            rows=merge_scores(metadata[(year,arm)],read_table(spec['values']));months=[f'{year}-{m:02d}' for m in range(1,13)]
            truth=[r['true_class'] for r in rows]
            if year in truth_reference:need(truth==truth_reference[year],'Saved reports do not share identical observed classes')
            else:truth_reference[year]=truth
            summary,monthly,classes=aggregate(rows,months=months)
            assert_summary(proofs[f'{year}/{arm}/summary'],summary)
            assert_summary(proofs[f'proof_{year}']['arms'][arm]['metric_summary'],summary)
            stored_monthly=read_table(spec['monthly'])
            need([m['outcome_month'] for m in stored_monthly]==months, 'Saved monthly order/grid differs')
            for actual,expected in zip(stored_monthly,monthly):
                for k,v in expected.items():
                    if k in ('outcome_month','status'): need(actual[k]==v,'Saved monthly status/month differs')
                    else: close(actual[k],v,'Saved monthly '+k)
            modes=[r for r in read_table(spec['modes']) if r['period']=='all' and r['class_id'] in CLASSES]
            need({r['class_id'] for r in modes}==set(CLASSES) and len(modes)==4, 'Separate tail-class rows missing/duplicate')
            for r in modes:
                for k,v in classes[str(r['class_id'])].items(): close(r[k],v,'Saved class '+k)
            yearly[str(year)]=summary;all_rows.extend(rows)
        overall,monthly,classes=aggregate(all_rows,months=[f'{y}-{m:02d}' for y in YEARS for m in range(1,13)])
        assert_summary(final['comparative_prediction_metrics'][arm],overall)
        result['arms'][arm]={'overall':overall,'years':yearly,'classes':classes,'monthly':monthly}
    for item in [*z['jsons'].values(),*z['fit_pins'].values()]:check(item)
    for annual in z['annuals'].values():
        for specs in annual.values():
            for spec in specs.values():check(spec['pin'])
    for name in ('protocol_pin','progress_pin','worker_pin','control_pin','supervisor_pin','source_plan_pin','catalog_pin'):check(z[name])
    output.mkdir(parents=True);save_new(output/'comparison.json',result)
    with (output/'comparison.md').open('x') as f:f.write(markdown(result))
    return save_new(output/'receipt.json',{'status':'COMPLETE','plan_pin':plan_pin,'comparison_pin':pin(output/'comparison.json'),
        'markdown_pin':pin(output/'comparison.md'),'final_verification_pin':final_pin,'model_fits':0,'inference_calls':0,
        'reserved_compressed_source_bytes':z['projected_reads']['compressed_source']})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['plan','run'])
    ap.add_argument('--protocol');ap.add_argument('--plan');ap.add_argument('--plan-sha256')
    ap.add_argument('--reservation');ap.add_argument('--output',required=True)
    a=ap.parse_args();value=(make_plan(a.protocol,a.output) if a.command=='plan' else execute(a.plan,a.plan_sha256,a.reservation,a.output))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
