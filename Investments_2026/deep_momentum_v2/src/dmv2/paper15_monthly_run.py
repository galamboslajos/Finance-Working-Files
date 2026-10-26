"""Bounded cache-only preparation, 16 annual fits and monthly-return scenarios."""
from pathlib import Path
import itertools, json
from .paper15_baseline import FEATURES, PROBS, KEYS, need
from .paper15_baseline_data import pin, check, read, save, table_save, canonical

SOURCE_COLUMNS = [*KEYS, 'intrinio_security_id', 'identity_eligible', 'identity_status',
                  'identity_review_required', 'resolution_status', 'prediction_eligible',
                  'prediction_status', 'outcome_month', 'outcome_start', 'outcome_end',
                  'history_years', 'history_complete', 'expected_history_sessions',
                  'valid_history_sessions', *FEATURES]
CODE = ['src/dmv2/paper15_monthly.py', 'src/dmv2/paper15_monthly_run.py',
        'scripts/run_paper15_monthly.py', 'scripts/verify_paper15_monthly.py']


def seal(code_root, data_root, protocol, output, destination):
    import pyarrow.parquet as pq
    code, data = Path(code_root).resolve(), Path(data_root).resolve()
    cfg = read(protocol)
    need(cfg['schema'] == 'dmv2_paper15_monthly_baseline_v1' and cfg['maximum_fits'] == 16, 'Frozen monthly scope')
    parent = data/'outputs/paper15_ret_baseline/20261026-v3'
    ledger = read(parent/'ledger.json'); need(ledger['status'] == 'COMPLETE_VERIFIED', 'Verified executable parent')
    parent_proof_pin = next(read(check(x['receipt']))['result'] for x in ledger['completed'] if x['key'] == 'verify_data')
    proof = read(check(parent_proof_pin)); prep_pin = proof['prepared_receipt_pin']; prep = read(check(prep_pin))
    need(proof['status'] == 'PASS' and proof['financial_2024_plus_read'] is False and proof['plan_pin'] == ledger['plan_pin'], 'Parent source proof')
    vp_pin = proof['verification_read_plan_pin']; vp = read(check(vp_pin))
    inputs = {k: prep['artifacts'][k] for k in ('index', 'month_closes', 'panel5', 'panel10')}
    for k, item in inputs.items():
        need(item == vp['prepared_artifacts'][k], 'Unverified source input'); check(item)
    idx = read(check(inputs['index']))
    need(idx['months'][-1] == '2023-12' and max(idx['sessions']) == '2023-12-29', 'No protected values')
    specs = {}; total = inputs['month_closes']['size_bytes']
    for n in (5, 10):
        pf = pq.ParquetFile(check(inputs[f'panel{n}']))
        need(pf.metadata.num_rows == 152307 and set(SOURCE_COLUMNS) <= set(pf.schema_arrow.names), 'Full source population')
        cost = sum(pf.metadata.row_group(g).column(c).total_compressed_size
                   for g in range(pf.metadata.num_row_groups) for c in range(pf.metadata.row_group(g).num_columns)
                   if pf.metadata.row_group(g).column(c).path_in_schema in SOURCE_COLUMNS)
        specs[str(n)] = dict(columns=SOURCE_COLUMNS, rows=152307, projected_compressed_bytes=cost)
        total += cost
    need(total <= cfg['budgets']['source_projection_bytes_per_pass'], 'Monthly read reservation')
    p = dict(schema=cfg['schema'], status='SEALED_BEFORE_NUMERIC_READ', code_root=str(code), data_root=str(data),
             output_directory=str(Path(output).resolve()), protocol_pin=pin(protocol),
             source_binding_pins=[pin(parent/'ledger.json'), parent_proof_pin, prep_pin, vp_pin, ledger['plan_pin']],
             source_inputs=inputs, source_projections=specs, source_projection_bytes=total,
             code_pins={n: pin(code/n) for n in CODE},
             helper_pins={n: pin(code/'src/dmv2'/n) for n in ('paper15_baseline.py', 'paper15_baseline_data.py', 'pilot_models.py', 'walkforward_metrics.py', 'ret.py')},
             max_fits=16, financial_2024_plus_read=False, daily_prices_read=False, expanded_features_read=False)
    p['plan_sha256'] = canonical(p); save(destination, p); return p


def validate(plan_path):
    p = read(plan_path)
    need(p['plan_sha256'] == canonical({k: v for k, v in p.items() if k != 'plan_sha256'}), 'Monthly plan changed')
    for item in [p['protocol_pin'], *p['source_binding_pins'], *p['source_inputs'].values(), *p['code_pins'].values(), *p['helper_pins'].values()]:
        check(item)
    need(p['max_fits'] == 16 and not p['financial_2024_plus_read'] and not p['daily_prices_read'], 'Monthly scope changed')
    return p


def prepare(plan_path, guard=lambda: None):
    import numpy as np, pandas as pd, pyarrow.parquet as pq
    from .paper15_monthly import monthly_panel, monthly_payoffs
    p = validate(plan_path); root = Path(p['output_directory']); out = root/'prepared'
    need(not out.exists(), 'Preserve consumed preparation'); out.mkdir(parents=True)
    save(out/'STARTED.json', {'plan_pin': pin(plan_path), 'source_projection_bytes': p['source_projection_bytes']})
    idx = read(check(p['source_inputs']['index']))
    closes = np.load(check(p['source_inputs']['month_closes']), mmap_mode='r', allow_pickle=False)
    need(closes.shape == (len(idx['months']), len(idx['natives'])), 'Monthly close array dimensions')
    artifacts = {}; summaries = []
    for n in (5, 10):
        guard(); source = pq.read_table(check(p['source_inputs'][f'panel{n}']), columns=SOURCE_COLUMNS).to_pandas()
        need(len(source) == 152307 and not source.duplicated(list(KEYS)).any(), 'Original rows retained')
        panel = monthly_panel(source, closes, idx['months'], idx['natives'], idx['sessions'])
        artifacts[f'panel{n}'] = table_save(out/f'panel{n}.parquet', panel)
        coverage = panel.assign(admitted_complete_history=panel.identity_eligible & panel.history_complete).groupby(['history_years', 'signal_date', 'outcome_month'], sort=True).agg(
            original_members=('source_security_id', 'size'), identity_admitted=('identity_eligible', 'sum'),
            complete_history=('admitted_complete_history', 'sum'), prediction_eligible=('prediction_eligible', 'sum'),
            labelled=('target_available', 'sum')).reset_index()
        coverage['missing_targets'] = coverage.prediction_eligible - coverage.labelled
        artifacts[f'coverage{n}'] = table_save(out/f'coverage{n}.parquet', coverage)
        payoffs = {}; cells = []
        for signal in sorted(panel.loc[panel.outcome_month.ge('2020-01'), 'signal_date'].unique()):
            mu, grid = monthly_payoffs(panel, signal, n); payoffs[signal] = mu; cells.extend(grid)
        artifacts[f'payoffs{n}'] = save(out/f'payoffs{n}.json', payoffs)
        artifacts[f'payoff_grid{n}'] = table_save(out/f'payoff_grid{n}.parquet', pd.DataFrame(cells))
        summaries.append(dict(history_years=n, original_rows=len(panel), prediction_eligible=int(panel.prediction_eligible.sum()),
                              labelled=int(panel.target_available.sum()), complete_RET_vectors=sum(v is not None for v in payoffs.values())))
    return save(out/'receipt.json', dict(status='PREPARED_PENDING_INDEPENDENT_CHECK', plan_pin=pin(plan_path),
                                        artifacts=artifacts, summary=summaries, daily_values_read=False,
                                        financial_2024_plus_read=False, new_fits=0))


def arrays(plan, years, year):
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    root = Path(plan['output_directory']); receipt = read(root/'prepared/receipt.json')
    need(receipt['plan_pin'] == pin(plan['_path']), 'Wrong prepared plan')
    t = pq.read_table(check(receipt['artifacts'][f'panel{years}']))
    meta = t.select([*KEYS, 'outcome_month', 'outcome_end', 'label_available_at', 'prediction_eligible', 'target_available']).to_pandas()
    need(meta.outcome_end.max() < '2024-01-01', 'Protected targets')
    train = (meta.outcome_month.between(f'{year-years}-01', f'{year-1}-12') & meta.label_available_at.le(f'{year-1}-12-31') & meta.prediction_eligible & meta.target_available).to_numpy(bool)
    test = meta.outcome_month.str.startswith(str(year)).to_numpy(bool); qualified = test & meta.prediction_eligible.to_numpy(bool)
    X = t.select(list(FEATURES)).filter(pa.array(train)).to_pandas().to_numpy(dtype=np.float32)
    y = t.select(['target_class']).filter(pa.array(train)).to_pandas().target_class.to_numpy(dtype=np.int64)
    XP = t.select(list(FEATURES)).filter(pa.array(qualified)).to_pandas().to_numpy(dtype=np.float32)
    cols = [*KEYS, 'intrinio_security_id', 'identity_eligible', 'identity_status', 'identity_review_required', 'resolution_status',
            'prediction_eligible', 'prediction_status', 'outcome_month', 'outcome_start', 'outcome_end']
    predictions = t.select(cols).filter(pa.array(test)).to_pandas().reset_index(drop=True)
    return t, meta, train, test, qualified, X, y, XP, predictions


def backtest(plan_path, guard=lambda: None):
    import pandas as pd, pyarrow.parquet as pq
    from .paper15_monthly import select_monthly, monthly_account, monthly_statistics
    p = validate(plan_path); cfg = read(check(p['protocol_pin'])); root = Path(p['output_directory'])
    proof = read(root/'verification/models.json')
    need(proof['status'] == 'PASS' and proof['verified_fits'] == 16, 'Independent model check required')
    prep = read(root/'prepared/receipt.json'); out = root/'backtest'; need(not out.exists(), 'Preserve prior monthly replay')
    out.mkdir(parents=True); save(out/'STARTED.json', dict(plan_pin=pin(plan_path), new_fits=0))
    artifacts = {}; comparison = []
    for years, kind in itertools.product((5, 10), ('rf', 'xgb')):
        guard(); arm = f'{kind}_paper15_{years}y'; parts = []
        for year in (2020, 2021, 2022, 2023):
            item = proof['forecasts'][f'{years}/{year}/{kind}']; parts.append(pq.read_table(check(item)).to_pandas())
        pred = pd.concat(parts, ignore_index=True); payoffs = read(check(prep['artifacts'][f'payoffs{years}']))
        outcomes = pq.read_table(check(prep['artifacts'][f'panel{years}']), columns=[*KEYS, 'label_return']).to_pandas()
        ledger, selected = select_monthly(pred, payoffs, restricted=cfg['backtest']['execution_restrictions'])
        for name, frame in [('original_ledger', ledger), ('selected', selected)]:
            artifacts[f'{arm}/selection/{name}'] = table_save(out/arm/'selection'/(name+'.parquet'), frame)
        for period, first in [('full48', '2020-01'), ('matched47', '2020-02')]:
            months = sorted(pred.loc[pred.outcome_month.ge(first), 'outcome_month'].unique())
            need(len(months) == (48 if period == 'full48' else 47), 'Full fixed monthly grid')
            for fee in cfg['backtest']['cost_bps']:
                guard(); monthly, positions, failures = monthly_account(selected, outcomes, months, fee)
                complete = len(monthly) == len(months) and failures.empty
                row = dict(arm=arm, period=period, cost_bps=fee, complete_monthly_path=complete, expected_months=len(months),
                           observed_months=len(monthly), first_failure_month=None if failures.empty else failures.iloc[0].outcome_month,
                           **monthly_statistics(monthly, complete))
                base = f'{arm}/{period}/cost_{fee:02d}bps/'
                annual = []
                for year in (2020, 2021, 2022, 2023):
                    g = monthly.loc[monthly.outcome_month.str.startswith(str(year))] if not monthly.empty else monthly
                    annual.append(dict(arm=arm, period=period, cost_bps=fee, year=year, observed_months=len(g),
                                       total_return=float((1+g.net_return).prod()-1) if complete and len(g) else None))
                for name, frame in [('monthly', monthly), ('positions', positions), ('failures', failures), ('annual', pd.DataFrame(annual))]:
                    artifacts[base+name] = table_save(out/(base+name+'.parquet'), frame)
                comparison.append(row)
                save(out/'checkpoints'/(base+'complete.json'), dict(status='COMPLETE_PENDING_VERIFICATION', comparison=row))
        print(json.dumps(dict(phase='monthly_backtest', arm=arm, scenarios_completed=len(comparison), maximum=32)), flush=True)
    artifacts['comparison'] = table_save(out/'comparison.parquet', pd.DataFrame(comparison))
    return save(out/'audit.json', dict(status='COMPLETE_PENDING_INDEPENDENT_CHECK', plan_pin=pin(plan_path),
                                       model_proof_pin=pin(root/'verification/models.json'), artifacts=artifacts, scenarios=32,
                                       complete_paths=sum(r['complete_monthly_path'] for r in comparison),
                                       withheld_paths=sum(not r['complete_monthly_path'] for r in comparison),
                                       daily_values_read=False, financial_2024_plus_read=False, new_fits=0))


import signal, time

def fit(plan_path,years,year,kind,guard=lambda:None):
    import joblib,numpy as np,pyarrow as pa
    from .pilot_models import fit_predict_pilot_arm
    from .walkforward_metrics import evaluate_probabilities
    p=validate(plan_path);p['_path']=str(Path(plan_path).resolve());cfg=read(check(p['protocol_pin']))
    root=Path(p['output_directory']);verified=read(root/'verification/data.json')
    need(verified['status']=='PASS' and verified['plan_pin']==pin(plan_path) and verified['prepared_receipt_pin']==pin(root/'prepared/receipt.json'),'Independent data check before fitting')
    out=root/f'folds/{years}y/{year}/{kind}';need(not out.exists(),'Preserve consumed estimator attempt');out.mkdir(parents=True)
    save(out/'FIT_STARTED.json',{'plan_pin':pin(plan_path),'years':years,'year':year,'kind':kind,'fit_attempt_consumed':True})
    t,meta,train,test,qualified,X,y,XP,forecast=arrays(p,years,year)
    need(X.shape[1]==XP.shape[1]==15 and np.isfinite(X).all() and np.isfinite(XP).all(),'Exactly15finitepaperinputs;noimputation')
    guard();start=time.monotonic();before=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fit time ceiling consumed')))
    signal.alarm(cfg['budgets']['per_fit_seconds'])
    try:result=fit_predict_pilot_arm(kind,X,y,XP,parameters=cfg['parameters'][kind],package_versions=cfg['packages'])
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,before)
    guard();model_path=out/'model.joblib';joblib.dump(result.estimator,model_path,compress=3)
    eligible=forecast.prediction_eligible.to_numpy(dtype=bool)
    for j,n in enumerate(PROBS):forecast[n]=np.nan;forecast.loc[eligible,n]=result.probabilities[:,j]
    artifact=table_save(out/'predictions.parquet',forecast)
    # No future label is supplied to fitting/inference; evaluation follows saved predictions.
    labels=t.select(['target_class']).filter(pa.array(qualified)).to_pandas().target_class
    md=forecast.loc[eligible].reset_index(drop=True);labels=labels.reset_index(drop=True)
    summaries=[];metrics={};prior=np.bincount(y,minlength=11)[1:]/len(y)
    for name,prob in [('model',result.probabilities),('uniform',np.full((len(md),10),.1)),('train_prior',np.tile(prior,(len(md),1)))]:
        evaluation=evaluate_probabilities(prob,labels,class_ids=list(range(1,11)),row_metadata=md,first_month=f'{year}-01',last_month=f'{year}-12')
        summary={**evaluation.summary,'history_years':years,'year':year,'estimator':kind,'reference':name};summaries.append(summary)
        for key,frame in evaluation.tables.items():metrics[name+'/'+key]=table_save(out/'metrics'/name/(key+'.parquet'),frame)
    summary_pin=save(out/'metrics/summary.json',summaries)
    return save(out/'FIT_COMPLETE.json',{'status':'FIT_AND_PREDICT_COMPLETE','plan_pin':pin(plan_path),'years':years,'year':year,'kind':kind,
      'model':pin(model_path),'predictions':artifact,'metrics':metrics,'summary':summary_pin,'fit_metadata':result.metadata,
      'fit_seconds':time.monotonic()-start,'train_outcome_start':f'{year-years}-01','train_outcome_end':f'{year-1}-12',
      'actual_label_cutoff':f'{year-1}-12-31','no_target_or_future_quote_prediction_filter':True,'new_tuning':False})
