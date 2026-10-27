"""Fixed Paper15 extension: twelve annual fits, inference and RET selections.

Reads only verified old/new monthly panels. No raw price reads, source changes,
tuning or trading replay. Existing 2020--2023 models/forecasts remain immutable.
"""
from pathlib import Path
import hashlib
import itertools
import signal
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .paper15_baseline import FEATURES, PROBS, KEYS, need
from .paper15_baseline_data import pin, check, read, save, table_save, canonical

META = [*KEYS, 'intrinio_security_id', 'identity_eligible', 'identity_status',
        'identity_review_required', 'resolution_status', 'prediction_eligible',
        'prediction_status', 'outcome_month', 'outcome_start', 'outcome_end',
        'history_years', 'history_complete', 'expected_history_sessions', 'valid_history_sessions']
TARGET = ['label_return', 'target_class', 'target_available', 'label_available_at',
          'target_entry_date', 'target_exit_date']
PANEL_COLUMNS = [*META, *FEATURES, *TARGET]
FORECAST_META = META[:12]
CODE = ('src/dmv2/extension_models.py', 'src/dmv2/extension_model_metrics.py',
        'scripts/run_extension_models.py', 'scripts/verify_extension_models.py')


def array_sha(a):
    return 'sha256:'+hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def masks(panel, years, year):
    """Training is completed prior outcomes; prediction never uses target state."""
    need(years in (5, 10) and year in (2024, 2025, 2026), 'Frozen annual extension window')
    need(not panel.duplicated(list(KEYS)).any(), 'Unique original signal keys')
    train = (panel.outcome_month.between(f'{year-years}-01', f'{year-1}-12')
             & panel.label_available_at.le(f'{year-1}-12-31')
             & panel.prediction_eligible & panel.target_available).to_numpy(bool)
    test = panel.outcome_month.between(f'{year}-01', min(f'{year}-12', '2026-04')).to_numpy(bool)
    qualified = test & panel.prediction_eligible.to_numpy(bool)
    need(not (train & test).any(), 'Training and prediction outcomes overlap')
    return train, test, qualified


def matrices(panel, years, year):
    train, test, qualified = masks(panel, years, year)
    X = panel.loc[train, list(FEATURES)].to_numpy(np.float32)
    y = panel.loc[train, 'target_class'].to_numpy(np.int64)
    XP = panel.loc[qualified, list(FEATURES)].to_numpy(np.float32)
    need(X.shape[1] == XP.shape[1] == 15 and np.isfinite(X).all() and np.isfinite(XP).all(), 'Finite Paper15; no imputation')
    need(set(y) == set(range(1,11)), 'Complete canonical training classes')
    return train, test, qualified, X, y, XP


def seal(code_root, data_root, protocol, output, destination):
    code, data = Path(code_root).resolve(), Path(data_root).resolve(); cfg = read(protocol)
    need(cfg['schema'] == 'dmv2_paper15_extension_models_v1' and cfg['maximum_fits'] == 12
         and cfg['maximum_backtests'] == 0 and cfg['annual_years'] == [2024,2025,2026]
         and cfg['windows_years'] == [5,10] and cfg['features'] == list(FEATURES)
         and not cfg['parameter_search'] and not cfg['source_upgrade'], 'Fixed twelve-fit model-only extension')
    baseline = code/'configs/paper15_monthly_baseline_20261026.json'; oldcfg = read(baseline)
    for k in ('parameters', 'packages', 'features', 'RET'):
        need(cfg[k] == oldcfg[k], 'Baseline '+k+' changed')
    need(cfg['execution_restrictions'] == oldcfg['backtest']['execution_restrictions'], 'Trading restrictions changed')
    parent = data/'outputs/walkforward_extension_inputs/20261027-v1';ledger=read(parent/'ledger.json')
    need(ledger['status'] == 'COMPLETE_VERIFIED_INPUTS' and ledger['fits_started'] == 0, 'Verified input phase required')
    phase_pin=next(x['receipt'] for x in ledger['completed'] if x['key']=='verify_data')
    proof_pin=read(check(phase_pin))['result'];proof=read(check(proof_pin))
    need(proof['status']=='PASS_INPUT_SOURCE_ARRAYS_AND_MONTHLY_FORMULAS', 'Independent source/feature proof required')
    ip=read(check(ledger['plan_pin']));prep=read(check(proof['prepared_receipt_pin']))
    need(ip['protocol_pin'] == pin(code/'configs/walkforward_extension_inputs_20261027.json'), 'Input protocol drift')
    binding=[pin(baseline),pin(parent/'ledger.json'),ledger['plan_pin'],phase_pin,proof_pin,proof['prepared_receipt_pin']]
    report_pin=pin(parent/'report/verification.json');report=read(check(report_pin))
    need(report['status']=='PASS_AGGREGATE_RECONCILIATION' and report['source_proof_pin']==proof_pin,'Coverage proof')
    binding += [report_pin, report['summary_pin'], report['read_plan_pin']]
    sources={};specs={};cost=0
    for years in (5,10):
        for kind,item in [('old',ip['parent_inputs'][f'panel{years}']),('new',prep['artifacts'][f'panel{years}'])]:
            pf=pq.ParquetFile(check(item));need(set(PANEL_COLUMNS)<=set(pf.schema_arrow.names),'Verified monthly input schema')
            total=sum(pf.metadata.row_group(g).column(c).total_compressed_size for g in range(pf.metadata.num_row_groups)
                      for c in range(pf.metadata.row_group(g).num_columns) if pf.metadata.row_group(g).column(c).path_in_schema in PANEL_COLUMNS)
            key=f'{years}/{kind}';sources[key]=item;specs[key]=dict(columns=PANEL_COLUMNS,rows=pf.metadata.num_rows,compressed_bytes=total);cost+=total
        for name in ('payoffs','payoff_grid'):
            item=prep['artifacts'][f'{name}{years}'];check(item)
            # The small payoff vectors are sufficient; grid stays an input proof reference.
            if name=='payoffs':sources[f'{years}/payoffs_new']=item;cost+=item['size_bytes']
    oldroot=data/'outputs/paper15_monthly_baseline/20261026-v1';ol=read(oldroot/'ledger.json')
    need(ol['status']=='COMPLETE_VERIFIED','Old monthly model proof required')
    oldproof_pin=read(check(next(x['receipt'] for x in ol['completed'] if x['key']=='verify_models')))['result']
    oldproof=read(check(oldproof_pin));need(oldproof['status']=='PASS' and oldproof['verified_fits']==16,'Old16model proof')
    binding += [pin(oldroot/'ledger.json'),oldproof_pin]
    oldprep=read(oldroot/'prepared/receipt.json');binding.append(pin(oldroot/'prepared/receipt.json'))
    olddata=read(check(oldproof['data_proof_pin']));need(olddata['prepared_receipt_pin']==pin(oldroot/'prepared/receipt.json'),'Old learning source proof')
    binding.append(oldproof['data_proof_pin'])
    oldforecasts=oldproof['forecasts']
    for item in oldforecasts.values():check(item);cost+=item['size_bytes']
    oldaudit=read(oldroot/'backtest/audit.json');binding.append(pin(oldroot/'backtest/audit.json'))
    oldbp=read(check(next(x['receipt'] for x in ol['completed'] if x['key']=='verify_backtest')))['result']
    oldbproof=read(check(oldbp));need(oldbproof['status']=='PASS' and oldbproof['audit_pin']==pin(oldroot/'backtest/audit.json'),'Old selection audit proof')
    binding.append(oldbp)
    oldselected={f'{kind}_paper15_{n}y':oldaudit['artifacts'][f'{kind}_paper15_{n}y/selection/selected'] for n,kind in itertools.product((5,10),('rf','xgb'))}
    for years in (5,10):sources[f'{years}/payoffs_old']=oldprep['artifacts'][f'payoffs{years}'];check(sources[f'{years}/payoffs_old']);cost+=sources[f'{years}/payoffs_old']['size_bytes']
    for item in oldselected.values():check(item);cost+=item['size_bytes']
    need(cost<=cfg['budgets']['source_projection_bytes_per_pass'],'Model-stage read ceiling')
    need(not Path(output).exists(),'New immutable model run required')
    p=dict(schema=cfg['schema'],status='SEALED_BEFORE_MODEL_PANEL_READ_OR_FIT',code_root=str(code),data_root=str(data),
        output_directory=str(Path(output).resolve()),protocol_pin=pin(protocol),source_binding_pins=binding,
        source_inputs=sources,source_projections=specs,old_forecasts=oldforecasts,old_selected=oldselected,
        source_projection_bytes=cost,max_fits=12,max_backtests=0,input_version='walkforward_extension_inputs/20261027-v1',
        code_pins={n:pin(code/n) for n in CODE},
        helper_pins={n:pin(code/'src/dmv2'/n) for n in ('paper15_baseline.py','paper15_baseline_data.py','pilot_models.py','ret.py','paper15_monthly.py','walkforward_metrics.py')})
    p['plan_sha256']=canonical(p);save(destination,p);return p


def validate(plan_path):
    p=read(plan_path);need(p['plan_sha256']==canonical({k:v for k,v in p.items() if k!='plan_sha256'}),'Model plan changed')
    for item in [p['protocol_pin'],*p['source_binding_pins'],*p['source_inputs'].values(),*p['old_forecasts'].values(),*p['old_selected'].values(),*p['code_pins'].values(),*p['helper_pins'].values()]:check(item)
    need(p['max_fits']==12 and p['max_backtests']==0,'Fixed phase scope changed');return p


def prepare(plan_path,guard=lambda:None):
    p=validate(plan_path);root=Path(p['output_directory']);out=root/'prepared';need(not out.exists(),'Preserve consumed preparation');out.mkdir(parents=True)
    save(out/'STARTED.json',dict(plan_pin=pin(plan_path),new_fits=0));artifacts={}
    for n in (5,10):
        guard();old=pq.read_table(check(p['source_inputs'][f'{n}/old']),columns=PANEL_COLUMNS).to_pandas()
        new=pq.read_table(check(p['source_inputs'][f'{n}/new']),columns=PANEL_COLUMNS).to_pandas()
        need(len(old)==152307 and old.outcome_month.max()=='2023-12' and len(new)==25460
             and new.outcome_month.min()=='2024-01' and new.outcome_month.max()=='2026-04','Exact old/new chronology')
        panel=pd.concat([old,new],ignore_index=True)
        need(not panel.duplicated(list(KEYS)).any() and len(panel)==177767,'Original full denominator')
        artifacts[f'panel{n}']=table_save(out/f'panel{n}.parquet',panel)
        mu=read(check(p['source_inputs'][f'{n}/payoffs_old']));extra=read(check(p['source_inputs'][f'{n}/payoffs_new']))
        need(not set(mu)&set(extra) and len(mu)==48 and len(extra)==28,'Exact old/new RET dates')
        mu.update(extra);need(len(mu)==76 and all(v is not None for v in mu.values()),'Full76RETgrid states')
        artifacts[f'payoffs{n}']=save(out/f'payoffs{n}.json',mu)
    return save(out/'receipt.json',dict(status='PREPARED_PENDING_INDEPENDENT_CHECK',plan_pin=pin(plan_path),artifacts=artifacts,new_fits=0,
        original_stock_months_per_arm=177767,raw_daily_reads=False,source_or_target_changes=False))


def fit(plan_path,years,year,kind,guard=lambda:None):
    import joblib
    from .pilot_models import fit_predict_pilot_arm
    from .extension_model_metrics import evaluate_probabilities
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);proof=read(root/'verification/data.json')
    need(proof['status']=='PASS' and proof['plan_pin']==pin(plan_path) and proof['prepared_receipt_pin']==pin(root/'prepared/receipt.json'),'Independent inputs before fitting')
    out=root/f'folds/{years}y/{year}/{kind}';need(not out.exists(),'Preserve consumed fit attempt');out.mkdir(parents=True)
    save(out/'FIT_STARTED.json',dict(plan_pin=pin(plan_path),years=years,year=year,kind=kind,fit_attempt_consumed=True))
    prep=read(check(proof['prepared_receipt_pin']));panel=pq.read_table(check(prep['artifacts'][f'panel{years}'])).to_pandas()
    train,test,qualified,X,y,XP=matrices(panel,years,year);expected=proof['matrix_proofs'][f'{years}/{year}']
    need(array_sha(X)==expected['X'] and array_sha(y)==expected['y'] and array_sha(XP)==expected['XP'],'Before-fit matrix proof')
    forecast=panel.loc[test,FORECAST_META].reset_index(drop=True)
    guard();started=time.monotonic();before=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fit ceiling consumed')))
    signal.alarm(cfg['budgets']['per_fit_seconds'])
    try:result=fit_predict_pilot_arm(kind,X,y,XP,parameters=cfg['parameters'][kind],package_versions=cfg['packages'])
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,before)
    guard();model_path=out/'model.joblib';joblib.dump(result.estimator,model_path,compress=3)
    eligible=forecast.prediction_eligible.to_numpy(bool)
    for j,name in enumerate(PROBS):forecast[name]=np.nan;forecast.loc[eligible,name]=result.probabilities[:,j]
    fp=table_save(out/'predictions.parquet',forecast)
    labels=panel.loc[qualified,'target_class'].reset_index(drop=True)
    md=forecast.loc[eligible].reset_index(drop=True);prior=np.bincount(y,minlength=11)[1:]/len(y);metrics={};summaries=[]
    last=min(f'{year}-12','2026-04')
    for ref,values in [('model',result.probabilities),('uniform',np.full(result.probabilities.shape,.1)),('train_prior',np.tile(prior,(len(XP),1)))]:
        evaluation=evaluate_probabilities(values,labels,class_ids=list(range(1,11)),row_metadata=md,first_month=f'{year}-01',last_month=last)
        summaries.append(dict(**evaluation.summary,history_years=years,year=year,estimator=kind,reference=ref))
        for name,frame in evaluation.tables.items():metrics[ref+'/'+name]=table_save(out/'metrics'/ref/(name+'.parquet'),frame)
    sp=save(out/'metrics/summary.json',summaries)
    return save(out/'FIT_COMPLETE.json',dict(status='FIT_AND_PREDICT_COMPLETE',plan_pin=pin(plan_path),years=years,year=year,kind=kind,
        model=pin(model_path),predictions=fp,metrics=metrics,summary=sp,fit_metadata=result.metadata,fit_seconds=time.monotonic()-started,
        train_outcome_start=f'{year-years}-01',train_outcome_end=f'{year-1}-12',actual_label_cutoff=f'{year-1}-12-31',
        test_last_outcome=last,no_target_or_future_quote_prediction_filter=True,new_tuning=False))


def selection(plan_path,guard=lambda:None):
    from .paper15_monthly import select_monthly
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);proof=read(root/'verification/models.json')
    need(proof['status']=='PASS' and proof['verified_fits']==12,'Independent twelve-model inference before selection')
    prep=read(root/'prepared/receipt.json');out=root/'selection';need(not out.exists(),'Preserve selection attempt');out.mkdir(parents=True)
    save(out/'STARTED.json',dict(plan_pin=pin(plan_path),new_fits=0,new_backtests=0));artifacts={};rows=[];gaps=[]
    for n,kind in itertools.product((5,10),('rf','xgb')):
        guard();arm=f'{kind}_paper15_{n}y'
        pins=[p['old_forecasts'][f'{n}/{y}/{kind}'] for y in range(2020,2024)]+[proof['forecasts'][f'{n}/{y}/{kind}'] for y in range(2024,2027)]
        pred=pd.concat([pq.read_table(check(item)).to_pandas()[FORECAST_META+list(PROBS)] for item in pins],ignore_index=True)
        need(pred.outcome_month.nunique()==76 and not pred.duplicated(list(KEYS)).any(),'Full forecast chronology')
        payoff=read(check(prep['artifacts'][f'payoffs{n}']));ledger,picks=select_monthly(pred,payoff,restricted=cfg['execution_restrictions'])
        old=pq.read_table(check(p['old_selected'][arm])).to_pandas()
        compare=['source_security_id','signal_date','side','intended_weight','RET']
        left=picks.loc[picks.outcome_month.le('2023-12'),compare].sort_values(list(KEYS)).reset_index(drop=True)
        right=old[compare].sort_values(list(KEYS)).reset_index(drop=True)
        pd.testing.assert_frame_equal(left,right,check_exact=True)
        panel=pq.read_table(check(prep['artifacts'][f'panel{n}']),columns=[*KEYS,'label_return','target_available']).to_pandas()
        joined=picks.merge(panel,on=list(KEYS),how='left',validate='one_to_one')
        missing=joined.loc[joined.outcome_month.ge('2024-01') & ~joined.target_available].copy()
        missing['arm']=arm;gaps.append(missing[['arm',*KEYS,'outcome_month','intrinio_security_id','side','RET']])
        for name,frame in [('forecasts',pred),('original_ledger',ledger),('selected',picks)]:artifacts[arm+'/'+name]=table_save(out/arm/(name+'.parquet'),frame)
        rows.append(dict(arm=arm,added_months=28,added_original_rows=int(pred.outcome_month.ge('2024-01').sum()),
            added_predictions=int((pred.outcome_month.ge('2024-01') & pred.prediction_eligible).sum()),
            added_selected_rows=int(picks.outcome_month.ge('2024-01').sum()),added_selected_monthly_endpoint_gaps=len(missing),
            full_forecast_months=76,common_candidate_months=75,old_selected_rows_byte_values_unchanged=True))
    artifacts['coverage']=table_save(out/'coverage.parquet',pd.DataFrame(rows))
    artifacts['selected_endpoint_gaps']=table_save(out/'selected_endpoint_gaps.parquet',pd.concat(gaps,ignore_index=True))
    return save(out/'audit.json',dict(status='SELECTION_PENDING_INDEPENDENT_CHECK',plan_pin=pin(plan_path),model_proof_pin=pin(root/'verification/models.json'),
        artifacts=artifacts,summary=rows,new_fits=0,new_backtests=0,no_future_quote_selection_filter=True,old_selections_preserved=True,
        endpoint_gap_inventory_is_not_an_execution_or_corporate_action_audit=True))
