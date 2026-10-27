"""Passive, independently authenticated small fixed-forecast reports only."""
from pathlib import Path
import hashlib,json


def need(v,m):
    if not v:raise ValueError(m)


def metadata(path):
    p=Path(path);need(p.stat().st_size<=16*1024**2,'Small metadata ceiling')
    return json.loads(p.read_text())


def checked(item,maximum=16*1024**2):
    p=Path(item['path']).resolve();need(p.stat().st_size==item['size_bytes']<=maximum,'Small report size changed')
    need(hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Small report fingerprint changed');return p


def check_monthly_curve(monthly,daily,clock):
    """Bind displayed small curves to the independently checked daily NAV ledger."""
    import pandas as pd
    d=daily.loc[daily.mark.ne('open') & daily.date.ge(pd.Timestamp('2020-02-03'))].copy()
    need(len(d)==985 and d.status.eq('ok').all(),'Verified complete daily reporting calendar')
    d['wealth']=d['nav']
    if clock=='signal_close':d.loc[d.rebalance,'wealth']=d.loc[d.rebalance,'nav_before_cost']
    expected=d.groupby(d.date.dt.strftime('%Y-%m'),sort=True).tail(1)[['date','wealth']].copy()
    expected['outcome_month']=expected.date.dt.strftime('%Y-%m')
    expected['return']=expected.wealth.div(expected.wealth.shift(fill_value=1.))-1
    pd.testing.assert_frame_equal(monthly.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False,check_exact=True)


def read_fixed_forecast_reconciliation(directory):
    root=Path(directory).resolve();ledger=metadata(root/'ledger.json')
    result={k:ledger.get(k) for k in ('status','phase','active_seconds','peak_worker_rss_bytes','error')}
    result.update(maximum_fits=0,maximum_scenarios=28,new_fits=0)
    if ledger['status']!='COMPLETE_VERIFIED':return result
    need(ledger['phase'] is None and ledger['fits_started']==ledger['fits_completed']==0
        and len(ledger['completed'])==4,'Zero-fit completed phase schedule')
    phases={}
    for row in ledger['completed']:
        path=checked(row['receipt']);need(path.parent==root/'phases','Unsafe phase receipt')
        phases[row['key']]=metadata(path)
    need(set(phases)=={'prepare','verify_inputs','replay','verify_replay'} and all(x['status']=='COMPLETE_PHASE' for x in phases.values()),'Complete input/replay proof schedule')
    proofs={}
    for key,name in [('verify_inputs','inputs'),('verify_replay','replay')]:
        path=checked(phases[key]['result']);need(path==root/'verification'/(name+'.json'),'Wrong independent proof path')
        v=metadata(path);need(v['status']=='PASS' and v['plan_pin']==ledger['plan_pin'] and v['new_fits']==0
            and v['financial_2024_plus_read'] is False,'Independent zero-fit proof scope');proofs[name]=v
    rp=proofs['replay'];need(rp['verified_scenarios']==28 and rp['producer_accounting_imported'] is False
        and rp['forecasts_and_selections_unchanged'] and rp['learning_inputs_unchanged'],'Independent accounting/preservation proof')
    need(checked(rp['prepared_input_proof_pin'])==root/'verification/inputs.json','Broken input proof chain')
    plan=metadata(checked(ledger['plan_pin']));audit=metadata(checked(rp['audit_pin']))
    vp=metadata(checked(rp['verification_read_plan_pin']))
    need(audit['plan_pin']==ledger['plan_pin'] and audit['retention']==plan['retention']
       and audit['selected']==plan['selected'] and audit['scenarios']==28,'Frozen forecast binding')
    import pandas as pd
    import pyarrow.parquet as pq
    def report(key,rows=None):
        item=audit['artifacts'][key];need(item==vp['artifacts'][key],'Report not independently bound')
        path=checked(item,maximum=2*1024**2);need(path.is_relative_to(root/'replay'),'Wrong report directory')
        pf=pq.ParquetFile(path)
        if rows is not None:need(pf.metadata.num_rows==rows,'Incomplete aggregate grid')
        return pf.read().to_pandas()
    comp=report('comparison',28);recon=report('reconciliation')
    need(int(comp.complete_path.sum())==rp['complete_paths'] and 28-int(comp.complete_path.sum())==rp['withheld_paths'],'Coverage grid mismatch')
    curves=[]
    for row in comp.itertuples():
        f=report(row.arm+'/'+row.stage+'/monthly',47 if row.complete_path else 0)
        if not f.empty:
            daily=report(row.arm+'/'+row.stage+'/daily')
            check_monthly_curve(f,daily,row.execution_clock)
            f['arm']=row.arm;f['stage']=row.stage;curves.append(f)
    protocol=metadata(checked(plan['protocol_pin']))
    result.update(comparison=comp,reconciliation=recon,monthly=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(),
       complete_paths=rp['complete_paths'],withheld_paths=rp['withheld_paths'],
       all_close_controls_reconcile=rp['all_close_controls_reconcile'],
       period='February2020–December2023;47monthlyoutcomes/985subsequentsessions;initialclose2020-01-31',
       forecasts_changed=False,labels_changed=False,RET_changed=False,selection_changed=False,
       original_runs_preserved=True,protocol=protocol,source_certification=False)
    return result
