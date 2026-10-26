"""Passive small monthly reports authenticated by exact independent receipts."""
from pathlib import Path
import hashlib, json


def need(value, message):
    if not value: raise ValueError(message)


def metadata(path):
    p=Path(path); need(p.stat().st_size <= 32*1024**2, 'Metadata ceiling')
    return json.loads(p.read_text())


def checked(item, maximum=32*1024**2):
    p=Path(item['path']).resolve(); need(p.stat().st_size == item['size_bytes'] <= maximum, 'Small report size changed')
    need(hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], 'Small report bytes changed')
    return p


def read_paper15_monthly(directory):
    root=Path(directory).resolve(); ledger=metadata(root/'ledger.json')
    result={k:ledger.get(k) for k in ('status','phase','fits_completed','active_seconds','error')}
    result['maximum_fits']=16
    if ledger['status'] != 'COMPLETE_VERIFIED': return result
    need(ledger['fits_started'] == ledger['fits_completed'] == 16 and len(ledger['completed']) == 21 and ledger['phase'] is None, 'Incomplete monthly schedule')
    phases={}
    for item in ledger['completed']:
        path=checked(item['receipt']); need(path.parent == root/'phases', 'Wrong phase path')
        record=metadata(path); need(record['status'] == 'COMPLETE_PHASE', 'Incomplete phase')
        phases[item['key']]=record
    plan=metadata(checked(ledger['plan_pin'])); proofs={}
    for phase,name in [('verify_data','data'),('verify_models','models'),('verify_backtest','backtest')]:
        path=checked(phases[phase]['result']); need(path == root/'verification'/(name+'.json'), 'Wrong proof path')
        proof=metadata(path); need(proof['status'] == 'PASS' and proof['plan_pin'] == ledger['plan_pin'] and proof['financial_2024_plus_read'] is False, 'Independent monthly proof')
        proofs[name]=proof
    need(proofs['models']['verified_fits'] == 16 and proofs['backtest']['verified_scenarios'] == 32, 'Wrong verified fit/scenario count')
    need(checked(proofs['models']['data_proof_pin']) == root/'verification/data.json' and checked(proofs['backtest']['model_proof_pin']) == root/'verification/models.json', 'Broken independent proof chain')
    audit=metadata(checked(proofs['backtest']['audit_pin'])); bv=metadata(checked(proofs['backtest']['verification_read_plan_pin']))
    need(audit['plan_pin'] == ledger['plan_pin'] and audit['scenarios'] == 32 and audit['daily_values_read'] is False, 'Monthly audit scope')
    import pandas as pd
    import pyarrow.parquet as pq
    def report(key, rows=None):
        item=audit['artifacts'][key]; need(item == bv['inventory']['backtest/'+key], 'Unverified monthly report')
        path=checked(item, maximum=2*1024**2); need(path.is_relative_to(root/'backtest'), 'Unsafe monthly report path')
        pf=pq.ParquetFile(path)
        if rows is not None: need(pf.metadata.num_rows == rows, 'Incomplete report grid')
        return pf.read().to_pandas()
    comparison=report('comparison',32)
    need(not comparison.duplicated(['arm','period','cost_bps']).any(), 'Duplicate monthly scenarios')
    prep=metadata(checked(proofs['data']['prepared_receipt_pin'])); dv=metadata(checked(proofs['data']['verification_read_plan_pin']))
    coverage=[]
    for n in (5,10):
        item=prep['artifacts'][f'coverage{n}']; need(item == dv['prepared_artifacts'][f'coverage{n}'], 'Unverified eligibility coverage')
        pf=pq.ParquetFile(checked(item,maximum=1024**2)); need(pf.metadata.num_rows == 168, 'Incomplete historical coverage')
        coverage.append(pf.read().to_pandas())
    metrics=metadata(checked(proofs['models']['metrics_pin'])); need(len(metrics) == 48, 'Four annual metrics per model/reference')
    aggregates=[]
    for (n,kind,reference),g in pd.DataFrame(metrics).groupby(['history_years','estimator','reference'],sort=True):
        need(len(g)==4, 'Four annual folds')
        row=dict(arm=f'{kind}_paper15_{n}y',history_years=int(n),estimator=kind,reference=reference,
                 forecast_rows=int(g.prediction_rows.sum()),observed_labels=int(g.labelled_rows.sum()),
                 observed_label_months=int(g.observed_label_months.sum()))
        for name in ('mean_monthly_log_loss','mean_monthly_accuracy','mean_monthly_brier_sum10'):
            row[name]=float((g[name]*g.observed_label_months).sum()/g.observed_label_months.sum())
        for name in ('pooled_log_loss','pooled_accuracy','pooled_brier_sum10'):
            row[name]=float((g[name]*g.labelled_rows).sum()/g.labelled_rows.sum())
        aggregates.append(row)
    classes=[]
    for key,item in proofs['models']['fit_receipts'].items():
        done=metadata(checked(item)); frame=pq.read_table(checked(done['metrics']['model/mode_class_performance'],maximum=1024**2)).to_pandas()
        frame=frame.loc[frame.period.eq('all')].copy(); need(len(frame)==10,'Ten separate classes')
        n,year,kind=key.split('/');frame['arm']=f'{kind}_paper15_{n}y';classes.append(frame)
    classes=pd.concat(classes,ignore_index=True).groupby(['arm','class_id'],as_index=False)[['actual_support','labelled_predicted_support','true_positive_count','missing_label_predicted_support']].sum()
    classes['precision']=classes.true_positive_count.div(classes.labelled_predicted_support.where(classes.labelled_predicted_support.ne(0)))
    classes['recall']=classes.true_positive_count.div(classes.actual_support.where(classes.actual_support.ne(0)))
    annual=[]; gaps=[]
    for row in comparison.itertuples():
        base=f'{row.arm}/{row.period}/cost_{row.cost_bps:02d}bps/'
        annual.append(report(base+'annual',4))
        if row.cost_bps == 10 and not row.complete_monthly_path:
            f=report(base+'failures').copy(); f['arm']=row.arm;f['period']=row.period;gaps.append(f)
    primary=[]
    for (arm,period),g in comparison.groupby(['arm','period'],sort=True):
        net=g.loc[g.cost_bps.eq(10)].iloc[0];gross=g.loc[g.cost_bps.eq(0)].iloc[0]
        primary.append(dict(arm=arm,period=period,complete_monthly_path=bool(net.complete_monthly_path),
            gross_total_return=gross.total_return,net_total_return=net.total_return,net_cagr=net.cagr,
            net_annualized_volatility=net.annualized_volatility,net_sharpe=net.sharpe_zero_cash_rate,
            net_monthly_maximum_drawdown=net.monthly_maximum_drawdown,first_failure_month=net.first_failure_month))
    result.update(primary=pd.DataFrame(primary),comparison=comparison,annual=pd.concat(annual,ignore_index=True),
        prediction_metrics=pd.DataFrame(aggregates),class_performance=classes,coverage=pd.concat(coverage,ignore_index=True),
        gaps=pd.concat(gaps,ignore_index=True) if gaps else pd.DataFrame(),complete_paths=audit['complete_paths'],withheld_paths=audit['withheld_paths'],
        full_period='January2020–December2023;48monthlyreturns',matched_period='February2020–December2023;47monthlyreturns;startsflat',
        metric_frequency='monthly;drawdown_at_month_ends_only',same_close_execution_is_idealized=True,
        protocol=metadata(checked(plan['protocol_pin'])))
    common_root=root/'reports/common_period_v1'
    if (common_root/'verification.json').exists():
        proof=metadata(common_root/'verification.json')
        need(proof['status']=='PASS_BOUND_SOURCE_AND_TWO_FORMULAS' and proof['scenarios']==16
             and proof['full_period_results_remain_withheld'] is True and proof['new_fits']==proof['new_backtests']==proof['imputed_returns']==0,
             'Common period scope changed')
        need(checked(proof['parent_backtest_proof_pin'])==root/'verification/backtest.json','Common report parent proof')
        cp=metadata(checked(proof['read_plan_pin']))
        need(cp['parent_backtest_proof_pin']==proof['parent_backtest_proof_pin'] and cp['parent_audit_pin']==proofs['backtest']['audit_pin'],
             'Common report source binding')
        failure_months=comparison.loc[comparison.period.eq('full48'),'first_failure_month'].dropna()
        expected_end=str(pd.Period(failure_months.min(),freq='M')-1) if len(failure_months) else '2023-12'
        need(proof['first_month']==cp['first_month']=='2020-01' and proof['last_month']==cp['last_month']==expected_end
             and proof['months']==cp['months']==len(pd.period_range('2020-01',expected_end,freq='M')),
             'Common period must be selected from coverage alone')
        common_frames={}
        for name,count in [('comparison',16),('annual',16*(int(proof['last_month'][:4])-int(proof['first_month'][:4])+1)),('monthly',16*proof['months'])]:
            path=checked(proof['artifacts'][name],maximum=2*1024**2);need(path.parent==common_root,'Wrong common aggregate path')
            pf=pq.ParquetFile(path);need(pf.metadata.num_rows==count,'Incomplete common aggregate');common_frames[name]=pf.read().to_pandas()
        cc=common_frames['comparison'];need(not cc.duplicated(['arm','cost_bps']).any(),'Duplicate common scenarios')
        rows=[]
        for arm,g in cc.groupby('arm',sort=True):
            gross=g.loc[g.cost_bps.eq(0)].iloc[0];net=g.loc[g.cost_bps.eq(10)].iloc[0]
            rows.append(dict(arm=arm,gross_total_return=gross.total_return,net_total_return=net.total_return,net_cagr=net.cagr,
                net_annualized_volatility=net.annualized_volatility,net_sharpe=net.sharpe_zero_cash_rate,
                net_monthly_maximum_drawdown=net.monthly_maximum_drawdown,
                long_pnl_per_initial_capital=net.long_pnl_per_initial_capital,short_pnl_per_initial_capital=net.short_pnl_per_initial_capital,
                trading_cost_paid=net.trading_cost_paid))
        result.update(common_primary=pd.DataFrame(rows),common_comparison=cc,common_annual=common_frames['annual'],common_monthly=common_frames['monthly'],
            common_period=dict(first_month=proof['first_month'],last_month=proof['last_month'],months=proof['months'],
                interpretation='post-run coverage-limited diagnostic, not full2020-2023 performance'))
    return result
