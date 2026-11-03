#!/usr/bin/env python3
"""Build continuous WFO accounting scenarios with explicit contingent-right marks.

Raw stock returns remain missing for the two unquoted rights. Their portfolio
contributions are evaluated with the declared long-zero/short-cap convention.
No entire portfolio day is zero-filled, dropped, interpolated or reweighted.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_walk_forward import extended_metrics, monthly_returns, rolling_statistics, segment_curves
from build_momentum_features import write_json
from evaluate_walk_forward import ARMS
from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent


def conservative_mark(position_weight, cap):
    require(np.isfinite(position_weight) and np.isfinite(cap) and cap >= 0, 'Invalid rights exposure')
    return float(cap) if position_weight < 0 else 0.


def apply_policy(base, sensitivity, ledger, policy):
    """Portfolio-level assumptions; no fabricated stock label is constructed."""
    out = base.copy().sort_values('return_date').reset_index(drop=True)
    require(not out.return_date.duplicated().any(), 'Duplicate portfolio date')
    require(np.array_equal(out[list(ARMS)].notna().to_numpy(),
                           np.repeat(out.complete.to_numpy()[:, None], len(ARMS), axis=1)), 'Inconsistent source coverage')
    out['observed_or_recovered_complete'] = out.complete
    out['has_contingent_right_assumption'] = ~out.complete
    out['valuation_basis'] = np.where(out.complete, 'original_or_recovered', 'conservative_contingent_right_scenario')
    events = policy['events']
    expected = {pd.Timestamp(e['return_date']) for e in events}
    require(len(expected) == len(events) and set(out.loc[~out.complete, 'return_date']) == expected,
            'Policy must cover exactly the unavailable portfolio dates')
    require(not sensitivity.duplicated(['return_date', 'strategy']).any(), 'Duplicate sensitivity key')
    require(len(sensitivity) == len(events) * len(ARMS), 'Unexpected sensitivity scope')
    claims = ledger.loc[ledger.status.eq('unresolved_contingent_right')].set_index('label_date')
    scenario_frames = {bits: base.copy().sort_values('return_date').reset_index(drop=True)
                       for bits in product((0, 1), repeat=len(events))}
    register = []
    for event_index, e in enumerate(events):
        day = pd.Timestamp(e['return_date'])
        row = claims.loc[day]
        require(row.membership_ticker == e['ticker'] and row.cash_per_old_share == e['cash_per_old_share']
                and row.contingent_right_max == e['right_payout_cap'], 'Configured consideration differs from evidence')
        mask = out.return_date.eq(day)
        require(mask.sum() == 1, 'Unknown event date')
        for arm in ARMS:
            s = sensitivity.loc[sensitivity.return_date.eq(day) & sensitivity.strategy.eq(arm)]
            require(len(s) == 1 and s.iloc[0].ticker == e['ticker'], 'Wrong sensitivity route')
            s = s.iloc[0]
            require(s.nodes == out.loc[mask, 'nodes'].iloc[0], 'Cohort denominator changed')
            w = float(s.position_weight)
            mark = conservative_mark(w, e['right_payout_cap'])
            chosen = float(s.portfolio_return_claim_face_mark if w < 0 else s.portfolio_return_claim_zero_mark)
            require(np.isfinite(chosen) and chosen > -1, 'Invalid scenario portfolio return')
            np.testing.assert_allclose(chosen, s.one_day_return_lower, rtol=0, atol=1e-14)
            out.loc[mask, arm] = chosen
            for bits, alternative in scenario_frames.items():
                alternative.loc[mask, arm] = float(s.portfolio_return_claim_face_mark if bits[event_index]
                                                  else s.portfolio_return_claim_zero_mark)
            register.append({'ticker':e['ticker'], 'strategy':arm, 'date':row.date, 'return_date':day,
                'source_security_id':row.source_security_id, 'prior_raw_close':row.prior_raw_close,
                'cash_per_old_share':e['cash_per_old_share'], 'right_payout_cap':e['right_payout_cap'],
                'position_weight':w, 'signed_rights_per_unit_entry_nav':w / row.prior_raw_close,
                'right_mark_per_unit':mark, 'conservative_mark_rule':'short_full_cap' if w < 0 else 'long_zero' if w > 0 else 'no_exposure',
                'short_reserve_per_unit_entry_nav':-w / row.prior_raw_close * mark if w < 0 else 0.,
                'long_uncredited_face_per_unit_entry_nav':w / row.prior_raw_close * e['right_payout_cap'] if w > 0 else 0.,
                'portfolio_return':chosen, 'alternative_zero_mark_return':s.portfolio_return_claim_zero_mark,
                'alternative_face_mark_return':s.portfolio_return_claim_face_mark, 'source':e['source'],
                'subsequent_claim_pnl_credited':False, 'right_sold':False,
                'interpretation':'Accounting assumption; excludes later payouts, reserve releases and financing; no observed market price'})
    require(np.isfinite(out[list(ARMS)]).all().all(), 'Unresolved calculation gap')
    out['complete'] = True  # Defined under the scenario, not newly observed source labels.
    unchanged = [c for c in base.columns if c not in [*ARMS, 'complete']]
    pd.testing.assert_frame_equal(out[unchanged], base.sort_values('return_date').reset_index(drop=True)[unchanged])
    originally_known = out.observed_or_recovered_complete
    np.testing.assert_array_equal(out.loc[originally_known, list(ARMS)],
        base.sort_values('return_date').reset_index(drop=True).loc[originally_known, list(ARMS)])
    for f in scenario_frames.values():
        require(np.isfinite(f[list(ARMS)]).all().all(), 'Sensitivity calculation gap')
        f['complete'] = True
    return out, pd.DataFrame(register), scenario_frames


def path_metrics(frame, arm):
    r = frame[arm].to_numpy(float)
    require(np.isfinite(r).all() and (r > -1).all(), 'Continuous path requires every calculated return')
    wealth = np.r_[1., np.exp(np.cumsum(np.log1p(r))) ]
    dd = wealth / np.maximum.accumulate(wealth) - 1
    out = extended_metrics(frame, arm)
    out['total_return'] = float(wealth[-1] - 1)
    out['calmar'] = float(out['cagr'] / -dd.min()) if dd.min() < 0 else None
    longest, run = 0, 0
    for value in dd[1:]:
        run = run + 1 if value < -1e-14 else 0
        longest = max(longest, run)
    out['longest_underwater_sessions'] = longest
    return out


def scenario_tables(alternatives):
    metrics, paths = [], []
    for bits, frame in alternatives.items():
        label = '_'.join('face' if bit else 'zero' for bit in bits)
        for arm in ARMS:
            metrics.append({'scenario':label, 'strategy':arm, **path_metrics(frame, arm)})
        curves, segments = segment_curves(frame)
        require(len(segments) == 1, 'Sensitivity scenario resets equity')
        curves['scenario'] = label
        paths.append(curves)
    paths = pd.concat(paths, ignore_index=True)
    envelope = paths.groupby(['strategy', 'date'], as_index=False).agg(
        lower_equity=('segment_equity', 'min'), upper_equity=('segment_equity', 'max'),
        lower_drawdown=('segment_drawdown', 'min'), upper_drawdown=('segment_drawdown', 'max'))
    return pd.DataFrame(metrics), envelope


def build(source, source_verification, output, config):
    require(not output.exists(), 'Continuous output exists; choose a new version')
    source_manifest = json.loads((source / 'manifest.json').read_text())
    proof = json.loads(source_verification.read_text())
    policy = json.loads(config.read_text())
    source_hash = file_hash(source / 'manifest.json')
    require(source_hash == policy['source_result_manifest_sha256'] == proof['result_manifest_sha256']
            and proof['status'] == 'passed', 'Wrong or unverified source result version')
    objects = {x['path']:x for x in source_manifest['outputs']}
    names = ['daily_paper_proxy_returns.csv', 'contingent_claim_sensitivity.csv', 'valuation_ledger.parquet']
    for name in names:
        require(file_hash(source / name) == objects[name]['sha256'], 'Source output changed')
    base = pd.read_csv(source / names[0], parse_dates=['date', 'return_date'])
    sensitivity = pd.read_csv(source / names[1], parse_dates=['return_date'])
    ledger = pd.read_parquet(source / names[2])
    require(policy['long_right_mark'] == 'zero' and policy['short_right_mark'] == 'full_contractual_payout_cap'
            and policy['future_right_cashflows_credited'] is False and policy['future_reserve_releases_credited'] is False,
            'Unimplemented accounting convention')
    frame, register, alternatives = apply_policy(base, sensitivity, ledger, policy)
    curves, segments = segment_curves(frame)
    require(len(segments) == 1 and int(segments.sessions.iloc[0]) == len(frame), 'Primary equity resets')
    monthly = monthly_returns(frame)
    require(monthly.complete.all(), 'Monthly performance gap')
    metrics = pd.DataFrame([{'period':period, 'strategy':arm, **path_metrics(block, arm)}
        for period, block in [('all', frame), *[(str(y), b) for y,b in frame.groupby('test_year')]] for arm in ARMS])
    sensitivity_metrics, envelope = scenario_tables(alternatives)
    output.mkdir(parents=True)
    tables = {'daily_paper_proxy_returns':frame, 'claim_register':register, 'paper_proxy_metrics':metrics,
        'equity_drawdown':curves.rename(columns={'segment_equity':'equity', 'segment_drawdown':'drawdown'}),
        'complete_segments':segments, 'monthly_returns':monthly, 'rolling_63_sessions':rolling_statistics(frame),
        'claim_mark_scenario_metrics':sensitivity_metrics, 'claim_mark_sensitivity_envelope':envelope}
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    write_json(output / 'policy.json', policy)
    result = {'status':'complete', 'purpose':'Continuous accounting scenario on frozen original WFO',
        'accounting_scenario':True, 'policy_id':policy['id'], 'sessions':len(frame),
        'observed_or_recovered_portfolio_dates':int(frame.observed_or_recovered_complete.sum()),
        'dates_with_explicit_right_mark_assumptions':int(frame.has_contingent_right_assumption.sum()),
        'unavailable_calculated_dates':int((~frame.complete).sum()), 'complete_months':int(monthly.complete.sum()),
        'months':len(monthly), 'equity_runs':len(segments), 'source_stock_labels_still_unobserved':2,
        'stock_labels_imputed':False, 'portfolio_zero_return_imputation':False, 'future_claim_pnl_credited':False,
        'models_refitted':0, 'forecasts_changed':0, 'original_positions_changed':0,
        'scenario_is_observed_market_data':False, 'full_path_metrics_are_conditional_on_policy':True,
        'source_result_directory':str(source), 'source_result_manifest_sha256':source_hash,
        'source_verification_sha256':file_hash(source_verification), 'config_sha256':file_hash(config),
        'script_sha256':file_hash(__file__),
        'outputs':[{'path':p.name, 'sha256':file_hash(p), 'size_bytes':p.stat().st_size}
                   for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / 'manifest.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'outputs'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT / 'data/walk_forward/valuation_results_v1')
    p.add_argument('--source-verification',type=Path,default=ROOT / 'data/walk_forward/valuation_verification_v1/report.json')
    p.add_argument('--output',type=Path,default=ROOT / 'data/walk_forward/continuous_results_v1')
    p.add_argument('--config',type=Path,default=ROOT / 'configs/contingent_rights_policy_v1.json')
    a=p.parse_args(); build(a.source.resolve(),a.source_verification.resolve(),a.output.resolve(),a.config.resolve())
