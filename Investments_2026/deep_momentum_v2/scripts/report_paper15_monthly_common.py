#!/usr/bin/env python3
"""Coverage-chosen common-prefix diagnostic from verified monthly ledgers only.

User26October2026 explicitly chose strict endpoints and a common available
period after the full run. No fits, selection changes, data fills or replays.
"""
from pathlib import Path
import argparse, json, math


def main():
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    from dmv2.paper15_baseline_data import pin, check, read, save, table_save
    from dmv2.paper15_monthly_reporting import read_paper15_monthly
    from dmv2.paper15_monthly import monthly_statistics
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    root=Path(args.run).resolve();r=read_paper15_monthly(root)
    if r['status']!='COMPLETE_VERIFIED':raise ValueError('Verified parent required')
    proof=read(root/'verification/backtest.json');audit=read(check(proof['audit_pin']))
    comp=r['comparison'];full=comp.loc[comp.period.eq('full48')]
    # Date chosen from missingness metadata, before opening monthly return values.
    missing=full.first_failure_month.dropna()
    end=(str(pd.Period(min(missing),freq='M')-1) if len(missing) else '2023-12')
    months=list(pd.period_range('2020-01',end,freq='M').astype(str))
    if not months:raise ValueError('No common complete monthly prefix')
    out=root/'reports/common_period_v1'
    if out.exists():raise ValueError('Preserve completed diagnostic')
    out.mkdir(parents=True)
    columns=['outcome_month','nav_start','nav_end','gross_return','net_return','long_contribution',
             'short_contribution','trading_cost','cost_contribution','one_way_turnover']
    inputs={}
    for row in full.itertuples():
        key=f'{row.arm}/full48/cost_{row.cost_bps:02d}bps/monthly';item=audit['artifacts'][key]
        check(item)
        if item['size_bytes']>1024**2:raise ValueError('Small ledger ceiling')
        inputs[key]=item
    plan=save(out/'read_plan.json',dict(status='SEALED_BEFORE_PREFIX_FINANCIAL_READ',
        parent_backtest_proof_pin=pin(root/'verification/backtest.json'),parent_audit_pin=proof['audit_pin'],
        first_month=months[0],last_month=months[-1],months=len(months),
        period_selection='all-arm common leading prefix before earliest selected missing return;chosen from failure dates only',
        interpretation='post-run coverage-limited diagnostic;not the full2020-2023 result or an untouched test',
        source_artifacts=inputs,columns=columns,worker_pin=pin(__file__),new_fits=0,new_backtests=0,imputed_returns=0))
    records=[];annual=[];paths=[]
    for row in full.itertuples():
        key=f'{row.arm}/full48/cost_{row.cost_bps:02d}bps/monthly'
        table=pq.read_table(check(inputs[key]),columns=columns)
        mask=pc.is_in(table['outcome_month'],value_set=pa.array(months))
        f=table.filter(mask).to_pandas().reset_index(drop=True)
        if f.outcome_month.tolist()!=months or abs(f.nav_start.iloc[0]-1)>1e-15:raise ValueError('Same full common prefix required')
        values=monthly_statistics(f,True)
        # Distinct formulas check the reporting calculation against the already
        # independently reconstructed cash/share monthly source ledger.
        ret=f.net_return.to_numpy();mean=math.fsum(ret)/len(ret)
        sd=math.sqrt(math.fsum((x-mean)**2 for x in ret)/(len(ret)-1));wealth=[1.]
        for value in ret:wealth.append(wealth[-1]*(1+value))
        peak=1.;dd=0.
        for value in wealth:peak=max(peak,value);dd=min(dd,value/peak-1)
        oracle=dict(total_return=math.prod(1+x for x in ret)-1,
            cagr=math.exp(12/len(ret)*math.fsum(math.log1p(x) for x in ret))-1,
            annualized_arithmetic_mean=12*mean,annualized_volatility=sd*math.sqrt(12),
            sharpe_zero_cash_rate=mean/sd*math.sqrt(12) if sd else None,monthly_maximum_drawdown=dd,
            total_one_way_turnover=math.fsum(f.one_way_turnover),trading_cost_paid=math.fsum(f.trading_cost))
        for name,value in oracle.items():
            if value is None:
                if values[name] is not None:raise ValueError('Zero-volatility metric')
            elif abs(values[name]-value)>5e-11:raise ValueError('Common-prefix formula '+name)
        long=math.fsum(f.nav_start*f.long_contribution);short=math.fsum(f.nav_start*f.short_contribution)
        if abs(long+short-values['trading_cost_paid']-values['total_return'])>5e-11:raise ValueError('Additive long/short/cost contribution')
        records.append(dict(arm=row.arm,cost_bps=row.cost_bps,first_month=months[0],last_month=months[-1],months=len(months),
                            long_pnl_per_initial_capital=long,short_pnl_per_initial_capital=short,**values))
        for year,g in f.groupby(f.outcome_month.str[:4],sort=True):
            annual.append(dict(arm=row.arm,cost_bps=row.cost_bps,year=int(year),months=len(g),total_return=float(math.prod(1+x for x in g.net_return)-1)))
        f['arm']=row.arm;f['cost_bps']=row.cost_bps;paths.append(f)
    artifacts=dict(comparison=table_save(out/'comparison.parquet',pd.DataFrame(records)),
                   annual=table_save(out/'annual.parquet',pd.DataFrame(annual)),
                   monthly=table_save(out/'monthly.parquet',pd.concat(paths,ignore_index=True)))
    receipt=save(out/'verification.json',dict(status='PASS_BOUND_SOURCE_AND_TWO_FORMULAS',read_plan_pin=plan,
        parent_backtest_proof_pin=pin(root/'verification/backtest.json'),artifacts=artifacts,scenarios=16,
        months=len(months),first_month=months[0],last_month=months[-1],full_period_results_remain_withheld=True,
        new_fits=0,new_backtests=0,imputed_returns=0,financial_2024_plus_read=False))
    print(json.dumps(dict(status='COMMON_PREFIX_VERIFIED',receipt=receipt,months=len(months),first=months[0],last=months[-1])))


if __name__=='__main__':main()
