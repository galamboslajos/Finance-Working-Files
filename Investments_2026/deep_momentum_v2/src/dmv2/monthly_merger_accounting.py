"""Frozen-selection monthly cash/share merger payoff accounting; no fitting.

Return corrections never enter predictors, labels, RET or selection. Mandatory
conversions generate signed cash/successor holdings, not fictitious old-stock
month-end prices or voluntary market trades.
"""
import math
import numpy as np
import pandas as pd


def need(value, message):
    if not value: raise ValueError(message)


def merger_components(start_adjusted_close, predecessor_raw_close, predecessor_adjusted_close,
                      cash_per_raw_share, stock_ratio, successor_raw_close=None,
                      successor_adjusted_close=None, successor_end_adjusted_close=None):
    """Wealth per unit initial capital using paired observed close factors.

    Predecessor basis is the last trading close before legal conversion.
    Successor factor is measured at the effective session's close; it converts
    raw shares to adjusted units and is not a trade/fill or a sale at that close.
    Ordinary adjusted-price distributions are not paid a second time.
    """
    need(all(math.isfinite(x) and x > 0 for x in (start_adjusted_close,predecessor_raw_close,predecessor_adjusted_close)), 'Observed positive predecessor bases')
    need(math.isfinite(cash_per_raw_share) and cash_per_raw_share >= 0 and math.isfinite(stock_ratio) and stock_ratio >= 0
         and (cash_per_raw_share > 0 or stock_ratio > 0), 'Positive legal entitlement')
    adjusted_units = 1.0 / start_adjusted_close
    predecessor_factor = predecessor_adjusted_close / predecessor_raw_close
    raw_shares = adjusted_units * predecessor_factor
    cash = raw_shares * cash_per_raw_share
    successor_units = 0.; successor_value = 0.; successor_factor = None
    if stock_ratio:
        need(all(x is not None and math.isfinite(x) and x > 0 for x in
                 (successor_raw_close,successor_adjusted_close,successor_end_adjusted_close)), 'Observed successor conversion and end closes')
        successor_factor = successor_adjusted_close / successor_raw_close
        successor_units = raw_shares * stock_ratio / successor_factor
        successor_value = successor_units * successor_end_adjusted_close
    return dict(asset_return=cash+successor_value-1.,cash_component=cash,successor_component=successor_value,
                predecessor_raw_shares_per_initial_capital=raw_shares,predecessor_close_factor=predecessor_factor,
                successor_close_factor=successor_factor,successor_adjusted_units_per_initial_capital=successor_units)


def replay_monthly_mergers(selected, outcomes, months, cost_bps, amendments):
    """Keep fixed baskets; earn supported returns and apply mandatory conversions.

    Normal monthly return notionals remain the prior adjusted-return proxy.
    Post-conversion cash stays cash, and successor holdings net with any other
    holdings in that native security. Costs apply only to voluntary rebalance
    trades against the resulting holdings, never to the legal conversion itself.
    """
    need(cost_bps in (0,5,10,25), 'Fixed cost sensitivity')
    need(not outcomes.duplicated(['source_security_id','signal_date']).any(), 'Unique original outcome keys')
    returns=outcomes.set_index(['source_security_id','signal_date']).label_return.to_dict()
    fix={(r['source_security_id'],r['signal_date']):r for r in amendments}
    need(len(fix)==len(amendments), 'Duplicate correction')
    cash=1.; holdings={}; nav=1.; monthly=[]; positions=[]; actions=[]; trades=[]; failures=[]
    for month in months:
        g=selected.loc[selected.outcome_month.eq(month)]
        if g.empty:
            failures.append(dict(outcome_month=month,reason='no_complete_RET_selection'));break
        need(g.source_security_id.is_unique and g.intrinio_security_id.is_unique and g.signal_date.nunique()==1,'Frozen unique basket')
        need(abs(g.loc[g.side.eq('long'),'intended_weight'].sum()-.5)<1e-12
             and abs(g.loc[g.side.eq('short'),'intended_weight'].sum()+.5)<1e-12,'50/50 original allocation')
        values=[]
        for row in g.to_dict('records'):
            key=(row['source_security_id'],row['signal_date']);correction=fix.get(key)
            value=correction['asset_return'] if correction is not None else returns.get(key,np.nan)
            if not math.isfinite(value) or value < -1:
                failures.append(dict(outcome_month=month,source_security_id=row['source_security_id'],
                                     intrinio_security_id=row['intrinio_security_id'],reason='unresolved_selected_monthly_return'))
            if correction is not None:
                need(correction['predecessor_id']==row['intrinio_security_id'] and correction['outcome_month']==month,
                     'Correction security/month mismatch')
                need(row['signal_date']<correction['effective_date']<=row['outcome_end'],'Must be held before conversion')
            values.append((row,value,correction))
        if failures:break
        start=cash+math.fsum(holdings.values());need(abs(start-nav)<1e-10 and start>0,'Opening portfolio wealth')
        targets={r['intrinio_security_id']:start*r['intended_weight'] for r,_,_ in values}
        delta={n:targets.get(n,0)-holdings.get(n,0) for n in set(targets)|set(holdings)}
        turnover=math.fsum(abs(v) for v in delta.values())/start
        fee=start*turnover*cost_bps/10000
        cash-=math.fsum(delta.values())+fee
        for native in sorted(delta):
            trades.append(dict(outcome_month=month,intrinio_security_id=native,
                               pretrade_notional=holdings.get(native,0.),target_notional=targets.get(native,0.),
                               signed_trade_notional=delta[native],trading_cost=abs(delta[native])*cost_bps/10000))
        end={};lp=0.;sp=0.
        for row,value,correction in values:
            native=row['intrinio_security_id'];notional=targets[native];pnl=notional*value
            if row['side']=='long':lp+=pnl
            else:sp+=pnl
            if correction is None:
                remaining=notional*(1+value);end[native]=end.get(native,0.)+remaining
                cash_entitlement=0.;successor_id=None;successor_notional=0.
            else:
                cash_entitlement=notional*correction['cash_component'];cash+=cash_entitlement
                successor_id=correction['successor_id'];successor_notional=notional*correction['successor_component']
                if successor_id is not None:end[successor_id]=end.get(successor_id,0.)+successor_notional
                remaining=0.
                actions.append(dict(outcome_month=month,source_security_id=row['source_security_id'],
                    side=row['side'],event_id=correction['event_id'],predecessor_id=native,successor_id=successor_id,
                    signed_initial_notional=notional,signed_cash_entitlement=cash_entitlement,
                    signed_successor_end_notional=successor_notional,mandatory_conversion_market_cost=0.))
            positions.append(dict(outcome_month=month,source_security_id=row['source_security_id'],intrinio_security_id=native,
                side=row['side'],target_notional=notional,asset_return=value,pnl=pnl,predecessor_end_notional=remaining,
                signed_cash_entitlement=cash_entitlement,successor_id=successor_id,signed_successor_end_notional=successor_notional,
                return_source='verified_merger_entitlement' if correction is not None else 'original_adjusted_close_ratio'))
        nav=cash+math.fsum(end.values())
        need(abs(nav-(start+lp+sp-fee))<1e-10,'Signed cash/share conservation')
        monthly.append(dict(outcome_month=month,signal_date=g.signal_date.iloc[0],nav_start=start,nav_end=nav,
            gross_return=(lp+sp)/start,net_return=nav/start-1,long_contribution=lp/start,short_contribution=sp/start,
            trading_cost=fee,cost_contribution=-fee/start,one_way_turnover=turnover,cash_end=cash,
            selected_long=int(g.side.eq('long').sum()),selected_short=int(g.side.eq('short').sum()),
            month_end_native_positions=len(end)))
        holdings=end
        if nav<=0 or not math.isfinite(nav):failures.append(dict(outcome_month=month,reason='nonpositive_portfolio_wealth'));break
    return dict(monthly=pd.DataFrame(monthly),positions=pd.DataFrame(positions),actions=pd.DataFrame(actions),
                trades=pd.DataFrame(trades),failures=pd.DataFrame(failures))
