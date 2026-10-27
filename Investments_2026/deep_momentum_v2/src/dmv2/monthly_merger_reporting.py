"""Passive, independently bound small reports for the zero-fit merger replay."""
from pathlib import Path
import hashlib,json


def need(v,m):
    if not v:raise ValueError(m)


def metadata(path):
    p=Path(path);need(p.stat().st_size<=32*1024**2,'Metadata ceiling');return json.loads(p.read_text())


def checked(item,maximum=32*1024**2):
    p=Path(item['path']).resolve();need(p.stat().st_size==item['size_bytes']<=maximum,'Small report size changed')
    need(hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Small report bytes changed');return p


def read_monthly_merger_replay(directory):
    root=Path(directory).resolve();ledger=metadata(root/'ledger.json')
    result={k:ledger.get(k) for k in ('status','phase','fits_started','fits_completed','active_seconds','error')}
    result['maximum_fits']=0
    if ledger['status']!='COMPLETE_VERIFIED':return result
    need(ledger['fits_started']==ledger['fits_completed']==0 and ledger['phase'] is None and len(ledger['completed'])==4,'Zero-fit phase schedule')
    phases={}
    for item in ledger['completed']:
        path=checked(item['receipt']);need(path.parent==root/'phases','Wrong phase directory')
        record=metadata(path);need(record['status']=='COMPLETE_PHASE','Incomplete replay phase');phases[item['key']]=record
    need(set(phases)=={'amendments','verify_amendments','backtest','verify_backtest'},'Unexpected replay phase')
    plan=metadata(checked(ledger['plan_pin']));proofs={}
    for phase,name in [('verify_amendments','amendments'),('verify_backtest','backtest')]:
        path=checked(phases[phase]['result']);need(path==root/'verification'/(name+'.json'),'Wrong proof path')
        proof=metadata(path);need(proof['status']=='PASS' and proof['plan_pin']==ledger['plan_pin'] and proof['new_fits']==0
             and proof['financial_2024_plus_read'] is False,'Independent replay proof');proofs[name]=proof
    need(proofs['amendments']['verified_stock_months']==4 and proofs['amendments']['verified_quote_points']==12,'Wrong correction scope')
    bp=proofs['backtest'];need(bp['verified_scenarios']==bp['complete_paths']==32 and bp['withheld_paths']==0
        and bp['forecasts_and_selection_byte_identical'] and bp['labels_features_and_RET_byte_identical'] and bp['common14month_results_preserved'],
        'Incomplete comparison or upstream mutation')
    need(checked(bp['amendment_proof_pin'])==root/'verification/amendments.json','Broken proof chain')
    audit=metadata(checked(bp['audit_pin']));vp=metadata(checked(bp['verification_read_plan_pin']))
    need(audit['plan_pin']==ledger['plan_pin'] and audit['new_fits']==0 and audit['complete_paths']==32
         and audit['upstream_retention']==plan['retention'] and audit['selected']==plan['selected'],'Replay preservation binding')
    import pandas as pd
    import pyarrow.parquet as pq
    def report(key,rows):
        item=audit['artifacts'][key];need(item==vp['backtest_artifacts'][key],'Unverified report')
        path=checked(item,maximum=2*1024**2);need(path.is_relative_to(root/'backtest'),'Wrong report directory')
        pf=pq.ParquetFile(path);need(pf.metadata.num_rows==rows,'Incomplete report grid');return pf.read().to_pandas()
    comparison=report('comparison',32);annual=report('annual',128)
    primary=[];curves=[]
    for (arm,period),g in comparison.groupby(['arm','period'],sort=True):
        gross=g.loc[g.cost_bps.eq(0)].iloc[0];net=g.loc[g.cost_bps.eq(10)].iloc[0]
        primary.append(dict(arm=arm,period=period,gross_total_return=gross.total_return,net_total_return=net.total_return,
            net_cagr=net.cagr,net_annualized_volatility=net.annualized_volatility,net_sharpe=net.sharpe_zero_cash_rate,
            net_monthly_maximum_drawdown=net.monthly_maximum_drawdown,complete_monthly_path=bool(net.complete_monthly_path)))
        if period=='full48':
            f=report(f'{arm}/full48/cost_10bps/monthly',48);f['arm']=arm;curves.append(f)
    receipt=metadata(checked(proofs['amendments']['amendment_receipt_pin']))
    corrections=metadata(checked(receipt['artifacts']['corrections']));need(len(corrections)==4,'Four correction records')
    scope=metadata(checked(receipt['artifacts']['learning_scope_audit']))
    need(all(not r['promoted_to_learning'] and r['features_modified']==r['labels_modified']==r['RET_modified']==0 for r in scope),'Learning scope changed')
    result.update(primary=pd.DataFrame(primary),comparison=comparison,annual=annual,monthly=pd.concat(curves,ignore_index=True),
        corrections=corrections,learning_scope_audit=scope,complete_paths=32,withheld_paths=0,
        full_period='January2020–December2023;48monthlyreturns',matched_period='February2020–December2023;47monthlyreturns',
        existing_models=16,new_fits=0,forecasts_changed=False,RET_changed=False,labels_changed=False,
        metric_frequency='monthly;drawdown_observed_at_month_ends',protocol=metadata(checked(plan['protocol_pin'])))
    presentation=root/'reports/presentation_v1/manifest.json'
    if presentation.exists():
        m=metadata(presentation)
        proof_pin=phases['verify_backtest']['result']
        need(m['status']=='VERIFIED_AGGREGATE_PRESENTATION' and m['parent_proof_pin']==proof_pin
             and m['new_fits']==m['new_backtests']==0 and m['cost_bps']==10 and m['stops'] is False
             and m['drawdown_frequency']=='monthly','Presentation scope changed')
        pr=metadata(checked(m['read_plan_pin']))
        need(pr['parent_proof_pin']==proof_pin and pr['financial_2024_plus_read'] is False,'Presentation proof chain')
        expected={k:v for k,v in audit['artifacts'].items() if k.endswith('/full48/cost_10bps/monthly') or k in ('comparison','annual')}
        need(pr['input_artifacts']==expected,'Presentation inputs changed')
        figure=checked(m['figure_pin'],maximum=8*1024**2)
        need(figure==presentation.parent/'full48_monthly_nav_drawdown.png','Wrong presentation figure')
        result['figure_path']=str(figure)
    return result
