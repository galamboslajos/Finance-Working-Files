#!/usr/bin/env python3
"""Independently replay the frozen-forecast valuation overlay and portfolio sums."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from build_momentum_features import write_json
from evaluate_walk_forward import ARMS
from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent


def verify(results, original, audit, output):
    require(not output.exists(), 'Verification output exists; use a new version')
    record = json.loads((results / 'manifest.json').read_text())
    old = json.loads((original / 'manifest.json').read_text())
    old_check_path = ROOT / 'data/walk_forward/verification_v1/report.json'
    old_check = json.loads(old_check_path.read_text())
    require(old_check['status'] == 'passed' and old_check['result_manifest_sha256'] == file_hash(original / 'manifest.json')
            == record['original_result_manifest_sha256'], 'Original WFO numerical verification does not match')
    for item in record['inputs']:
        require(file_hash(item['path']) == item['sha256'], 'Repair evidence changed')
    for item in record['outputs']:
        require(file_hash(results / item['path']) == item['sha256'], 'Repair output changed')
    paths = []
    for item in old['outputs']:
        require(file_hash(original / item['path']) == item['sha256'], 'Original artifact changed')
        if item['path'].startswith('predictions/'):
            paths.append(original / item['path'])
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values(['date', 'source_security_id']).reset_index(drop=True)
    ledger = pd.read_parquet(results / 'valuation_ledger.parquet')
    keys = ['date', 'source_security_id']
    require(not ledger.duplicated(keys).any(), 'Duplicate ledger key')
    expected = frame.loc[frame.forward_return.isna(), keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, ledger[keys].sort_values(keys).reset_index(drop=True))
    independent_path = audit / 'independent_daily_bars.parquet'
    require(file_hash(independent_path) == record['independent_object']['sha256'], 'Independent object changed')
    dates = sorted({str(d.date()) for d in ledger.label_date})
    sip = ds.dataset(independent_path).to_table(filter=(ds.field('trading_date').isin(dates))
        & (ds.field('adjustment') == 'raw')).to_pandas()
    history = pd.read_parquet(audit / 'original_native_context.parquet').set_index(['intrinio_security_id', 'date'])
    prices = pd.read_parquet(audit / 'event_date_source_prices.parquet').set_index(['intrinio_security_id', 'trading_date'])
    events = {(e['ticker'], e['label_date']):e for e in json.loads((ROOT / 'configs/valuation_events_v1.json').read_text())['events']}
    manual = {(q['ticker'], q['date']):q for q in json.loads((ROOT / 'configs/valuation_manual_quotes_v1.json').read_text())['quotes']}
    replacements = {}
    for row in ledger.itertuples():
        entry = history.loc[row.intrinio_security_id, row.date]
        require(entry.quote_status == 'ok' and entry.close == row.prior_raw_close, 'Entry quote differs')
        key = row.membership_ticker, str(row.label_date.date())
        if row.method == 'mandatory_merger_entitlement':
            e = events[key]
            require(pd.Timestamp(e['effective_date']) <= row.label_date, 'Not yet effective')
            if e.get('contingent_right_max', 0):
                require(row.status == 'unresolved_contingent_right' and pd.isna(row.repaired_forward_return), 'Unknown right was filled')
                continue
            mark = e['cash']
            if e['ratio']:
                if 'native' in e:
                    quote = prices.loc[e['native'], key[1]]
                    require(quote.price_quality_status == 'accepted', 'Invalid acquirer quote')
                    raw = quote.close
                else:
                    quote = sip.loc[sip.security_id.eq(e['independent_security_id']) & sip.trading_date.eq(key[1])]
                    require(len(quote) == 1, 'Independent acquirer quote absent')
                    raw = quote.close.iloc[0]
                mark += e['ratio'] * raw
            value = mark / entry.close - 1.
        else:
            if row.method == 'independent_sip_raw_close':
                q = sip.loc[sip.security_id.eq(row.source_security_id) & sip.trading_date.eq(key[1])]
                require(len(q) == 1 and q.feed.iloc[0] == 'sip' and q.timeframe.iloc[0] == '1Day', 'Independent quote absent')
                raw = q.close.iloc[0]
            else:
                raw = manual[key]['close']
            value = raw / entry.close / row.adjustment_factor - 1.
        require(row.status == 'repaired', 'Finite repair status differs')
        np.testing.assert_allclose(value, row.repaired_forward_return, rtol=0, atol=2e-15)
        replacements[row.date, row.source_security_id] = value
    original_values = frame.forward_return.to_numpy().copy()
    realized = original_values.copy()
    for i, key in enumerate(zip(frame.date, frame.source_security_id)):
        if key in replacements:
            require(not np.isfinite(realized[i]), 'Observed label overwritten')
            realized[i] = replacements[key]
    np.testing.assert_array_equal(realized[np.isfinite(original_values)], original_values[np.isfinite(original_values)])
    frame['checked_realized'] = realized
    saved = pd.read_csv(results / 'daily_paper_proxy_returns.csv', parse_dates=['date', 'return_date']).set_index('date')
    errors = []
    for day, block in frame.groupby('date', sort=True):
        line = saved.loc[day]
        count = len(block)
        finite = np.isfinite(block.checked_realized.to_numpy())
        require(line.nodes == count and line.available_labels == finite.sum() and line.complete == finite.all(), 'Cohort availability changed')
        require(block.label_date.nunique() == 1 and line.return_date == block.label_date.iloc[0], 'Realization date differs')
        for arm in ARMS:
            weights = .15 * block[arm + '_position'].to_numpy() / (np.sqrt(252.) * block.daily_volatility.to_numpy() * count)
            np.testing.assert_allclose(line[arm + '_net_weight'], weights.sum(), rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(line[arm + '_gross_weight'], abs(weights).sum(), rtol=1e-13, atol=1e-14)
            if finite.all():
                value = np.dot(weights, block.checked_realized.to_numpy())
                errors.append(abs(value - line[arm]))
                np.testing.assert_allclose(value, line[arm], rtol=1e-12, atol=1e-14)
            else:
                require(pd.isna(line[arm]), 'Incomplete cohort return published')
    sessions = xcals.get_calendar('XNYS', start=saved.return_date.min(), end=saved.return_date.max()).sessions.tz_localize(None)
    require(pd.DatetimeIndex(saved.return_date).equals(sessions), 'Scheduled date absent')
    old_daily = pd.read_csv(original / 'daily_paper_proxy_returns.csv', parse_dates=['date']).set_index('date')
    for arm in ARMS:
        np.testing.assert_allclose(saved.loc[old_daily.complete, arm], old_daily.loc[old_daily.complete, arm], rtol=1e-12, atol=1e-14)
    result = {'status':'passed', 'result_manifest_sha256':file_hash(results / 'manifest.json'),
        'original_verification_sha256':file_hash(old_check_path), 'original_models_and_forecasts_preserved':True,
        'original_artifacts_hash_checked':len(old['outputs']), 'forecast_rows_preserved':len(frame),
        'unavailable_stock_labels_checked':len(ledger), 'repaired_stock_labels_checked':len(replacements),
        'unresolved_contingent_rights':int(ledger.status.eq('unresolved_contingent_right').sum()),
        'portfolio_dates_replayed':len(saved), 'strategy_date_returns_replayed':len(saved)*len(ARMS),
        'maximum_portfolio_absolute_error':max(errors), 'complete_dates':int(saved.complete.sum()),
        'unavailable_dates':int((~saved.complete).sum()), 'script_sha256':file_hash(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT / 'data/walk_forward/valuation_results_v1')
    p.add_argument('--original',type=Path,default=ROOT / 'data/walk_forward/annual_results_v1')
    p.add_argument('--audit',type=Path,default=ROOT / 'data/performance_gap_audit/v1')
    p.add_argument('--output',type=Path,default=ROOT / 'data/walk_forward/valuation_verification_v1/report.json')
    a = p.parse_args(); verify(a.results.resolve(),a.original.resolve(),a.audit.resolve(),a.output.resolve())
