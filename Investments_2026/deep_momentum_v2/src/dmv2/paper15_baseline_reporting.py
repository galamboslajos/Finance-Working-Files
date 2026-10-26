"""Passive authenticated small baseline reports; no model/source-panel imports."""
from pathlib import Path
import hashlib,json


def need(v,m):
    if not v:raise ValueError(m)

def metadata(path):
    p=Path(path);need(p.stat().st_size<=32*1024**2,'Metadata ceiling');return json.loads(p.read_text())

def checked(item,maximum=32*1024**2):
    p=Path(item['path']).resolve();need(p.stat().st_size==item['size_bytes']<=maximum,'Small report size changed')
    need(hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Small report bytes changed');return p


def read_paper15_baseline(directory):
    root=Path(directory).resolve();ledger=metadata(root/'ledger.json');result={'status':ledger['status'],'phase':ledger['phase'],
      'fits_completed':ledger['fits_completed'],'maximum_fits':16,'active_seconds':ledger['active_seconds'],'new_tuning':False}
    cp=root/'resource_checkpoint.json'
    if cp.exists():result['resource_checkpoint']=metadata(cp)
    if ledger['status']!='COMPLETE_VERIFIED':result['error']=ledger.get('error');return result
    need(ledger['fits_started']==ledger['fits_completed']==16 and len(ledger['completed'])==21 and ledger['phase'] is None,'Incomplete fixed fit/check schedule')
    phases={}
    for item in ledger['completed']:
        path=checked(item['receipt']);need(path.parent==root/'phases','Unsafe phase report path')
        record=metadata(path);need(record['status']=='COMPLETE_PHASE','Incomplete checkpoint');phases[item['key']]=record
    plan=metadata(checked(ledger['plan_pin']));proofs={}
    for phase,name in [('verify_data','data'),('verify_models','models'),('verify_backtest','backtest')]:
        path=checked(phases[phase]['result']);need(path==root/'verification'/(name+'.json'),'Wrong independent proof path')
        proof=metadata(path);need(proof['status']=='PASS' and proof['plan_pin']==ledger['plan_pin'] and proof['financial_2024_plus_read'] is False,'Independent scope changed');proofs[name]=proof
    need(proofs['models']['verified_fits']==16 and proofs['backtest']['verified_scenarios']==32,'Wrong independent scenario/fit count')
    checked(proofs['models']['data_proof_pin']);checked(proofs['backtest']['model_proof_pin'])
    need(proofs['backtest']['model_proof_pin']['path']==str(root/'verification/models.json'),'Wrong model binding')
    audit_path=checked(proofs['backtest']['audit_pin']);audit=metadata(audit_path);vp=metadata(checked(proofs['backtest']['verification_read_plan_pin']))
    need(audit['plan_pin']==ledger['plan_pin'] and audit['sessions']==985 and audit['months']==47 and audit['scenarios']==32,'Wrong economic clock')
    import pandas as pd
    import pyarrow.parquet as pq
    frames={}
    for name,count in [('comparison',32),('annual',128)]:
        item=audit['artifacts'][name];need(item==vp['artifacts'][name],'Unverified aggregate')
        path=checked(item,maximum=1024**2);need(path.parent==root/'backtest','Unsafe aggregate path');pf=pq.ParquetFile(path);need(pf.metadata.num_rows==count,'Truncated economic scenario grid')
        frames[name]=pf.read().to_pandas()
    prep_path=checked(proofs['data']['prepared_receipt_pin']);prep=metadata(prep_path);dv=metadata(checked(proofs['data']['verification_read_plan_pin']))
    coverage=[]
    for n in (5,10):
        item=prep['artifacts'][f'coverage{n}'];need(item==dv['prepared_artifacts'][f'coverage{n}'],'Coverage not independently bound')
        pf=pq.ParquetFile(checked(item,maximum=1024**2));need(pf.metadata.num_rows==168,'Incomplete monthly eligibility reports');coverage.append(pf.read().to_pandas())
    metrics=metadata(checked(proofs['models']['metrics_pin']));need(len(metrics)==48,'16fits times3metric references required')
    metric_rows=[]
    for (n,kind,reference),group in pd.DataFrame(metrics).groupby(['history_years','estimator','reference'],sort=True):
        weights=group.observed_label_months.to_numpy(dtype=float);total=float(weights.sum());need(len(group)==4 and total>0,'Four annual observed metric rows')
        row={'arm':f'{kind}_paper15_{n}y','history_years':int(n),'estimator':kind,'reference':reference,
          'forecast_rows':int(group.prediction_rows.sum()),'observed_labels':int(group.labelled_rows.sum()),'observed_label_months':int(total)}
        for name in ('mean_monthly_log_loss','mean_monthly_accuracy','mean_monthly_brier_sum10'):row[name]=float((group[name].to_numpy(dtype=float)*weights).sum()/total)
        for name in ('pooled_log_loss','pooled_accuracy','pooled_brier_sum10'):row[name]=float((group[name]*group.labelled_rows).sum()/group.labelled_rows.sum())
        metric_rows.append(row)
    class_parts=[]
    for key,item in proofs['models']['fit_receipts'].items():
        done=metadata(checked(item));fp=done['metrics']['model/mode_class_performance'];pf=pq.ParquetFile(checked(fp,maximum=1024**2));need(pf.metadata.num_rows==130,'Class report monthly grid')
        frame=pf.read().to_pandas();frame=frame.loc[frame.period.eq('all')].copy();n,y,kind=key.split('/');frame['arm']=f'{kind}_paper15_{n}y';class_parts.append(frame)
    classes=pd.concat(class_parts,ignore_index=True).groupby(['arm','class_id'],as_index=False)[['predicted_support','labelled_predicted_support','missing_label_predicted_support','actual_support','true_positive_count']].sum()
    classes['precision']=classes.true_positive_count.div(classes.labelled_predicted_support.where(classes.labelled_predicted_support.ne(0)))
    classes['recall']=classes.true_positive_count.div(classes.actual_support.where(classes.actual_support.ne(0)))
    primary=[];comp=frames['comparison']
    for arm in sorted(comp.arm.unique()):
        stopped=comp.loc[comp.arm.eq(arm)&comp.variant.eq('asset_stop_10pct')];g=stopped.loc[stopped.cost_bps.eq(0)].iloc[0];n=stopped.loc[stopped.cost_bps.eq(10)].iloc[0]
        primary.append({'arm':arm,'complete_proxy_path':bool(n.complete_proxy_path),'gross_total_return':g.total_return,'net_total_return':n.total_return,
          'net_cagr':n.cagr,'net_annualized_volatility':n.annualized_volatility,'net_sharpe':n.sharpe_zero_cash_rate,'net_maximum_drawdown':n.maximum_drawdown,
          'stop_count':n.stop_count,'first_failure_date':n.first_failure_date})
    result.update(primary=pd.DataFrame(primary),comparison=comp,annual=frames['annual'],prediction_metrics=pd.DataFrame(metric_rows),
      class_performance=classes,coverage=pd.concat(coverage,ignore_index=True),complete_paths=audit['complete_paths'],withheld_paths=audit['withheld_paths'],
      economic_period='3 February2020–29December2023;47months/985sessions',prediction_period='48forecastsJanuary2020–December2023;observed label monthsJanuary2020–November2023',
      source_certification=False,protocol=metadata(checked(plan['protocol_pin'])))
    return result
