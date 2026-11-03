#!/usr/bin/env python3
"""Replay both assumed portfolio days from original holdings and cash/right terms."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from build_momentum_features import write_json
from evaluate_walk_forward import ARMS
from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent


def check_path_metrics(row, returns):
    values = np.asarray(returns, float)
    wealth = np.r_[100., 100. * np.cumprod(1 + values)]
    cagr = (wealth[-1] / 100.) ** (252. / len(values)) - 1
    dd = wealth / np.maximum.accumulate(wealth) - 1
    for name, expected in [('cagr', cagr), ('total_return', wealth[-1]/100. - 1),
                           ('maximum_drawdown', dd.min()), ('annualized_mean', 252 * values.mean()),
                           ('annualized_volatility', np.sqrt(252) * values.std(ddof=1)),
                           ('sharpe_complete_dates', np.sqrt(252) * values.mean() / values.std(ddof=1))]:
        np.testing.assert_allclose(row[name], expected, rtol=1e-11, atol=1e-12)
    return wealth, dd


def verify(results, source, original, audit, output):
    require(not output.exists(), 'Verification output exists; choose a new version')
    manifest = json.loads((results / 'manifest.json').read_text())
    parent = json.loads((source / 'manifest.json').read_text())
    source_proof = json.loads((ROOT / 'data/walk_forward/valuation_verification_v1/report.json').read_text())
    require(source_proof['status'] == 'passed' and source_proof['result_manifest_sha256'] == file_hash(source / 'manifest.json')
            == manifest['source_result_manifest_sha256'], 'Source verification differs')
    for directory, record in [(results, manifest), (source, parent)]:
        for item in record['outputs']:
            require(file_hash(directory / item['path']) == item['sha256'], 'Verified output changed')
    old_manifest = json.loads((original / 'manifest.json').read_text())
    require(file_hash(original / 'manifest.json') == parent['original_result_manifest_sha256'], 'Original experiment differs')
    paths = []
    for item in old_manifest['outputs']:
        require(file_hash(original / item['path']) == item['sha256'], 'Original artifact changed')
        if item['path'].startswith('predictions/'):
            paths.append(str(original / item['path']))
    policy = json.loads((results / 'policy.json').read_text())
    require(policy['long_right_mark'] == 'zero' and policy['short_right_mark'] == 'full_contractual_payout_cap'
            and policy['future_right_cashflows_credited'] is False and policy['future_reserve_releases_credited'] is False,
            'Unexpected accounting policy')
    dates = pd.to_datetime([e['return_date'] for e in policy['events']])
    original_rows = ds.dataset(paths, format='parquet').to_table(filter=ds.field('label_date').isin(
        pa.array(dates, type=pa.timestamp('us')))).to_pandas()
    ledger = pd.read_parquet(source / 'valuation_ledger.parquet')
    claims = ledger.loc[ledger.status.eq('unresolved_contingent_right')].set_index('label_date')
    context_path = audit / 'original_native_context.parquet'
    pin = next(x['sha256'] for x in parent['inputs'] if x['path'].endswith('/original_native_context.parquet'))
    require(file_hash(context_path) == pin, 'Entry price evidence differs')
    context = pd.read_parquet(context_path).set_index(['intrinio_security_id', 'date'])
    base = pd.read_csv(source / 'daily_paper_proxy_returns.csv', parse_dates=['date','return_date']).sort_values('return_date').reset_index(drop=True)
    daily = pd.read_csv(results / 'daily_paper_proxy_returns.csv', parse_dates=['date','return_date']).sort_values('return_date').reset_index(drop=True)
    register = pd.read_csv(results / 'claim_register.csv', parse_dates=['date','return_date'])
    require(len(register) == len(policy['events']) * len(ARMS) and not register.duplicated(['return_date','strategy']).any(), 'Register scope differs')
    require(register.right_sold.eq(False).all() and register.subsequent_claim_pnl_credited.eq(False).all(), 'Unimplemented subsequent cash flow/sale')
    unchanged = [c for c in base.columns if c not in [*ARMS,'complete']]
    pd.testing.assert_frame_equal(base[unchanged], daily[unchanged])
    np.testing.assert_array_equal(base.complete, daily.observed_or_recovered_complete)
    np.testing.assert_array_equal(~base.complete, daily.has_contingent_right_assumption)
    np.testing.assert_array_equal(base.loc[base.complete,list(ARMS)], daily.loc[base.complete,list(ARMS)])
    require(daily.complete.all() and np.isfinite(daily[list(ARMS)]).all().all(), 'Calculation gap remains')
    expected_returns = base[list(ARMS)].to_numpy().copy()
    alternatives = {bits: base[list(ARMS)].to_numpy().copy() for bits in product((0,1),repeat=len(policy['events']))}
    max_error = 0.
    for event_index, event in enumerate(policy['events']):
        date = pd.Timestamp(event['return_date'])
        stock = claims.loc[date]
        require(stock.membership_ticker == event['ticker'] and stock.cash_per_old_share == event['cash_per_old_share']
                and stock.contingent_right_max == event['right_payout_cap'], 'Term evidence differs')
        block = original_rows.loc[original_rows.label_date.eq(date)]
        missing = block.source_security_id.eq(stock.source_security_id).to_numpy()
        finite = np.isfinite(block.forward_return.to_numpy())
        require(missing.sum() == 1 and np.array_equal(missing, ~finite), 'Unexpected unobserved holding')
        entry = context.loc[stock.intrinio_security_id,stock.date]
        require(entry.quote_status == 'ok' and entry.close == stock.prior_raw_close, 'Entry units differ')
        daily_index = np.flatnonzero(daily.return_date.eq(date))
        require(len(daily_index) == 1 and daily.nodes.iloc[daily_index[0]] == len(block), 'Portfolio denominator differs')
        i = daily_index[0]
        for j, arm in enumerate(ARMS):
            weights = block[arm+'_position'].to_numpy() * .15 / (np.sqrt(252) * block.daily_volatility.to_numpy() * len(block))
            w = weights[missing][0]
            known = float(np.dot(weights[finite], block.forward_return.to_numpy()[finite]))
            q = float(w / entry.close)
            mark = event['right_payout_cap'] if w < 0 else 0.
            expected = known + w * ((event['cash_per_old_share'] + mark) / entry.close - 1)
            expected_returns[i,j] = expected
            max_error = max(max_error, abs(expected - daily[arm].iloc[i]))
            np.testing.assert_allclose(expected, daily[arm].iloc[i], rtol=0, atol=2e-15)
            row = register.loc[register.return_date.eq(date) & register.strategy.eq(arm)].iloc[0]
            for actual, reference in [(row.position_weight,w), (row.signed_rights_per_unit_entry_nav,q),
                                      (row.right_mark_per_unit,mark), (row.portfolio_return,expected),
                                      (row.short_reserve_per_unit_entry_nav,-q*mark if w<0 else 0.)]:
                np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=2e-15)
            for bits, alternative in alternatives.items():
                alternative[i,j] = known + w * ((event['cash_per_old_share'] + bits[event_index]*event['right_payout_cap']) / entry.close - 1)
    np.testing.assert_allclose(daily[list(ARMS)],expected_returns,rtol=0,atol=2e-15)
    sessions = xcals.get_calendar('XNYS',start=daily.return_date.min(),end=daily.return_date.max()).sessions.tz_localize(None)
    require(pd.DatetimeIndex(daily.return_date).equals(sessions), 'Scheduled date missing')
    curves = pd.read_csv(results / 'equity_drawdown.csv',parse_dates=['date'])
    metrics = pd.read_csv(results / 'paper_proxy_metrics.csv')
    monthly = pd.read_csv(results / 'monthly_returns.csv')
    require(monthly.complete.all() and len(monthly) == len(daily.return_date.dt.to_period('M').unique()), 'Monthly gap')
    metric_checks = 0
    for arm in ARMS:
        row = metrics.loc[metrics.period.eq('all') & metrics.strategy.eq(arm)].iloc[0]
        wealth, dd = check_path_metrics(row,daily[arm])
        curve = curves.loc[curves.strategy.eq(arm)].sort_values('date')
        require(len(curve)==len(daily)+1 and curve.segment.nunique()==1 and curve.is_initial_capital.sum()==1, 'Curve resets or loses dates')
        np.testing.assert_array_equal(curve.date.iloc[1:].to_numpy(),daily.return_date.to_numpy())
        np.testing.assert_allclose(curve.equity,wealth,rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(curve.drawdown,dd,rtol=1e-10,atol=1e-12)
        for year, block in daily.groupby('test_year'):
            check_path_metrics(metrics.loc[metrics.period.eq(str(year)) & metrics.strategy.eq(arm)].iloc[0],block[arm]);metric_checks += 1
        metric_checks += 1
        for month, block in daily.groupby(daily.return_date.dt.to_period('M')):
            expected = np.prod(1+block[arm])-1
            actual = monthly.loc[monthly.month.eq(str(month)),arm].iloc[0]
            np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
    scenario_metrics = pd.read_csv(results / 'claim_mark_scenario_metrics.csv')
    envelope = pd.read_csv(results / 'claim_mark_sensitivity_envelope.csv',parse_dates=['date'])
    for j, arm in enumerate(ARMS):
        scenario_wealth = []
        for bits, values in alternatives.items():
            label = '_'.join('face' if b else 'zero' for b in bits)
            row = scenario_metrics.loc[scenario_metrics.scenario.eq(label) & scenario_metrics.strategy.eq(arm)].iloc[0]
            wealth, dd = check_path_metrics(row,values[:,j]); scenario_wealth.append(wealth)
        e = envelope.loc[envelope.strategy.eq(arm)].sort_values('date')
        np.testing.assert_allclose(e.lower_equity,np.min(scenario_wealth,axis=0),rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(e.upper_equity,np.max(scenario_wealth,axis=0),rtol=1e-12,atol=1e-10)
    result = {'status':'passed', 'result_manifest_sha256':file_hash(results/'manifest.json'),
        'source_result_manifest_sha256':file_hash(source/'manifest.json'), 'script_sha256':file_hash(__file__),
        'original_finite_portfolio_days_preserved':int(base.complete.sum()), 'assumed_stock_events_checked':len(policy['events']),
        'strategy_event_returns_rederived_from_holdings':len(policy['events'])*len(ARMS),
        'maximum_assumed_return_error':max_error, 'scheduled_dates':len(daily), 'calculation_gaps':0,
        'monthly_strategy_returns_checked':len(monthly)*len(ARMS), 'annual_and_aggregate_metric_rows_checked':metric_checks,
        'single_equity_run_for_every_strategy':True, 'scenario_combinations_checked':len(alternatives),
        'source_stock_labels_remain_unobserved':True, 'future_rights_cashflows_are_modeled':False}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result);print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'data/walk_forward/continuous_results_v1')
    p.add_argument('--source',type=Path,default=ROOT/'data/walk_forward/valuation_results_v1')
    p.add_argument('--original',type=Path,default=ROOT/'data/walk_forward/annual_results_v1')
    p.add_argument('--audit',type=Path,default=ROOT/'data/performance_gap_audit/v1')
    p.add_argument('--output',type=Path,default=ROOT/'data/walk_forward/continuous_verification_v1/report.json')
    a=p.parse_args();verify(a.results.resolve(),a.source.resolve(),a.original.resolve(),a.audit.resolve(),a.output.resolve())
