"""Monthly close-to-close target/RET and idealized monthly portfolio accounting.

No daily execution, stop inference, target imputation or signal-time future filter.
"""
import math
import numpy as np
import pandas as pd
from .paper15_baseline import FEATURES, PROBS, KEYS, deciles, need, shift


def monthly_panel(source, closes, months, natives, sessions):
    """Keep verified signal inputs unchanged; replace only forward target fields."""
    out = source.copy(deep=True)
    need(not out.duplicated(list(KEYS)).any(), 'Unique historical original rows')
    ni = {n: i for i, n in enumerate(natives)}
    mi = {m: i for i, m in enumerate(months)}
    ends = {m: max(d for d in sessions if d.startswith(m)) for m in months}
    values = np.full(len(out), np.nan)
    status = np.full(len(out), 'not_prediction_eligible', dtype=object)
    starts, finishes = [], []
    for j, row in enumerate(out.to_dict('records')):
        sm, om = row['signal_date'][:7], row['outcome_month']
        need(om == shift(sm, 1) and '2010-01' <= om <= '2023-12', 'Protected monthly clock')
        need(row['signal_date'] == ends[sm] and row['outcome_end'] == ends[om], 'Exact month ends')
        starts.append(ends[sm]); finishes.append(ends[om])
        if not row['prediction_eligible']:
            continue
        native = ni.get(row['intrinio_security_id'])
        if native is None:
            status[j] = 'missing_native'; continue
        a, b = float(closes[mi[sm], native]), float(closes[mi[om], native])
        if not math.isfinite(a) or a <= 0:
            status[j] = 'missing_or_invalid_start'; continue
        if not math.isfinite(b) or b <= 0:
            status[j] = 'missing_or_invalid_end'; continue
        values[j] = b / a - 1.0
        status[j] = 'ok'
    out['label_return'] = values
    out['return_status'] = status
    out['target_entry_date'] = starts
    out['target_exit_date'] = finishes
    out['label_available_at'] = finishes
    out['target_class'] = np.nan
    for _, part in out.groupby('signal_date', sort=True):
        out.loc[part.index, 'target_class'] = deciles(part.label_return.to_numpy())
    out['target_available'] = out.target_class.notna()
    return out


def monthly_payoffs(panel, signal, years):
    """Include signal month's now-complete close-to-close outcome in RET."""
    need(years in (5, 10) and signal < '2024-01-01', 'Approved payoff clock')
    months = [shift(signal[:7], j) for j in range(-12 * years + 1, 1)]
    past = panel.loc[panel.outcome_month.isin(months) & panel.label_available_at.le(signal)]
    grid = []
    for month in months:
        g = past.loc[past.outcome_month.eq(month)]
        for k in range(1, 11):
            v = g.loc[g.target_class.eq(k), 'label_return'].to_numpy(dtype=float)
            grid.append(dict(signal_date=signal, history_years=years, outcome_month=month,
                             class_id=k, count=len(v), mean_return=math.fsum(v) / len(v) if len(v) else None))
    mu = ([math.fsum(c['mean_return'] for c in grid if c['class_id'] == k) / (12 * years)
           for k in range(1, 11)] if all(c['count'] for c in grid) else None)
    return mu, grid


def select_monthly(predictions, payoffs, restricted=()):
    """Accept only signal metadata and probabilities, never realized returns."""
    need(not any(str(c).startswith(('label', 'target', 'future', 'realized')) for c in predictions),
         'Forward outcomes prohibited in selection')
    need({*KEYS, 'intrinio_security_id', 'prediction_eligible', *PROBS} <= set(predictions), 'Signal schema')
    ledger = predictions.drop(columns=list(PROBS)).copy(deep=True)
    ledger['RET'] = np.nan; ledger['side'] = None; ledger['selected'] = False
    ledger['intended_weight'] = 0.; ledger['selection_status'] = 'not_selected'
    for signal, part in predictions.groupby('signal_date', sort=True):
        mu = payoffs[signal]
        if mu is None:
            ledger.loc[part.index, 'selection_status'] = 'withheld_incomplete_RET'; continue
        need(len(mu) == 10 and np.isfinite(mu).all(), 'Complete finite payoff vector')
        eligible = part.loc[part.prediction_eligible]
        p = eligible[list(PROBS)].to_numpy(dtype=float)
        need(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8), 'Probabilities')
        ledger.loc[eligible.index, 'RET'] = p @ np.asarray(mu)
        candidates = ledger.loc[eligible.index].loc[lambda x: ~x.intrinio_security_id.isin(restricted)]
        need(not candidates.intrinio_security_id.duplicated().any(), 'Native collision')
        n = min(10, len(candidates) // 2)
        if not n:
            ledger.loc[part.index, 'selection_status'] = 'withheld_zero_candidates'; continue
        long = candidates.sort_values(['RET', 'source_security_id'], ascending=[False, True], kind='stable').index[:n]
        short = candidates.drop(index=long).sort_values(['RET', 'source_security_id'], kind='stable').index[:n]
        for side, chosen, weight in [('long', long, .5 / n), ('short', short, -.5 / n)]:
            ledger.loc[chosen, 'selected'] = True; ledger.loc[chosen, 'side'] = side
            ledger.loc[chosen, 'intended_weight'] = weight; ledger.loc[chosen, 'selection_status'] = 'selected'
    return ledger, ledger.loc[ledger.selected].copy()


def monthly_account(selected, outcomes, months, cost_bps):
    """Signed return notionals, drift-aware turnover; no fabricated missing return.

    Targets use pre-trade NAV, as in the saved daily comparison. Cost reduces
    cash rather than retrospectively resizing orders. Terminal NAV is marked,
    with no forced liquidation. Every scenario starts flat at the first signal.
    """
    need(cost_bps in (0, 5, 10, 25), 'Frozen cost sensitivity')
    need(not outcomes.duplicated(list(KEYS)).any(), 'Unique monthly outcome keys')
    lookup = outcomes.set_index(list(KEYS)).label_return.to_dict()
    holdings = {}; nav = 1.; rows = []; positions = []; failures = []
    for month in months:
        pick = selected.loc[selected.outcome_month.eq(month)]
        if not len(pick):
            failures.append(dict(outcome_month=month, reason='no_complete_RET_selection')); break
        need(pick.signal_date.nunique() == 1 and pick.source_security_id.is_unique, 'One signal/unique basket')
        weights = dict(zip(pick.source_security_id, pick.intended_weight))
        long = math.fsum(v for v in weights.values() if v > 0)
        short = math.fsum(v for v in weights.values() if v < 0)
        need(abs(long - .5) < 1e-12 and abs(short + .5) < 1e-12, 'Balanced gross-one allocation')
        realized = {}
        for row in pick.to_dict('records'):
            value = lookup.get((row['source_security_id'], row['signal_date']), np.nan)
            if not math.isfinite(value) or value < -1:
                failures.append(dict(outcome_month=month, signal_date=row['signal_date'],
                                     source_security_id=row['source_security_id'],
                                     intrinio_security_id=row['intrinio_security_id'], side=row['side'],
                                     reason='missing_selected_monthly_return'))
            else:
                realized[row['source_security_id']] = float(value)
        if failures:
            break
        targets = {k: nav * w for k, w in weights.items()}
        trades = {k: targets.get(k, 0.) - holdings.get(k, 0.) for k in set(targets) | set(holdings)}
        turnover = math.fsum(abs(v) for v in trades.values()) / nav
        fee = nav * turnover * cost_bps / 10000
        lp = math.fsum(targets[k] * realized[k] for k, w in weights.items() if w > 0)
        sp = math.fsum(targets[k] * realized[k] for k, w in weights.items() if w < 0)
        finish = nav + lp + sp - fee
        end_holdings = {k: targets[k] * (1 + realized[k]) for k in targets}
        rows.append(dict(outcome_month=month, signal_date=pick.signal_date.iloc[0],
                         nav_start=nav, nav_end=finish, gross_return=(lp + sp) / nav,
                         net_return=(finish / nav - 1), long_contribution=lp / nav,
                         short_contribution=sp / nav, trading_cost=fee, cost_contribution=-fee / nav,
                         one_way_turnover=turnover, selected_long=len(pick.loc[pick.side.eq('long')]),
                         selected_short=len(pick.loc[pick.side.eq('short')]),
                         cash_end=nav - math.fsum(targets.values()) - fee))
        for k, notional in targets.items():
            positions.append(dict(outcome_month=month, source_security_id=k, target_notional=notional,
                                  monthly_return=realized[k], pnl=notional * realized[k], end_notional=end_holdings[k]))
        if finish <= 0 or not math.isfinite(finish):
            failures.append(dict(outcome_month=month, reason='nonpositive_portfolio_wealth')); break
        holdings, nav = end_holdings, finish
    columns = ['outcome_month', 'signal_date', 'nav_start', 'nav_end', 'gross_return', 'net_return',
               'long_contribution', 'short_contribution', 'trading_cost', 'cost_contribution',
               'one_way_turnover', 'selected_long', 'selected_short', 'cash_end']
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(positions), pd.DataFrame(failures)


def monthly_statistics(frame, complete):
    fields = ('total_return', 'cagr', 'annualized_arithmetic_mean', 'annualized_volatility',
              'sharpe_zero_cash_rate', 'monthly_maximum_drawdown', 'total_one_way_turnover', 'trading_cost_paid')
    if not complete:
        return dict.fromkeys(fields)
    returns = frame.net_return.to_numpy(dtype=float)
    vol = float(np.std(returns, ddof=1))
    nav = np.r_[1., frame.nav_end.to_numpy(dtype=float)]
    return dict(total_return=float(nav[-1] - 1), cagr=float(nav[-1] ** (12 / len(returns)) - 1),
                annualized_arithmetic_mean=float(12 * returns.mean()), annualized_volatility=vol * math.sqrt(12),
                sharpe_zero_cash_rate=float(returns.mean() / vol * math.sqrt(12)) if vol > 0 else None,
                monthly_maximum_drawdown=float(np.min(nav / np.maximum.accumulate(nav) - 1)),
                total_one_way_turnover=float(frame.one_way_turnover.sum()), trading_cost_paid=float(frame.trading_cost.sum()))
