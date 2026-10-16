"""Independent signed-unit/cash replay of a saved 2018–2019 action rerun.

Never fits, selects, or calls the accounting engine. Source loading uses the
saved bounded plan; only the separate reviewed quote preparation is shared.
"""
from collections import defaultdict
import json
from math import isclose
from pathlib import Path
import sys

import pandas as pd

from dmv2.action_backtest import read_action_backtest, plan_action_backtest, _prepare_prices
from dmv2.action_loading import load_action_inputs
from dmv2.supplemental_bars import admit_reviewed_missing_bars


def close(a,b):
    if not isclose(float(a),float(b),rel_tol=2e-9,abs_tol=2e-11):
        raise AssertionError(f"Independent reconciliation differs: {a} versus {b}")


def grouped(frame):
    result=defaultdict(list)
    for row in frame.to_dict('records'):
        result[(pd.Timestamp(row['date']),row['mark'])].append(row)
    return result


def verify(root):
    root=Path(root).resolve()
    saved=read_action_backtest(root)
    plan=plan_action_backtest(root)
    bundle=load_action_inputs(root,plan=plan['input_plan'],
        unquoted_successor_policy=plan['configuration']['corporate_actions']['unquoted_successor_policy'])
    prices,_,_=_prepare_prices(bundle,plan['reviewed_manifest'],plan['configuration'],plan['evidence_dir'],
        additional_manifests=plan['additional_manifests'])
    manifest=plan['reviewed_manifest']
    prices=admit_reviewed_missing_bars(prices,evidence_root=plan['evidence_dir'],
        **{name:manifest[name] for name in ('projection_evidence','source_provenance','native_mapping','review_notes')}).prices
    assert pd.to_datetime(prices.trading_date).between('2018-01-01','2019-12-31').all()
    quotes={(pd.Timestamp(r['trading_date']),r['intrinio_security_id']):r for r in prices.to_dict('records')}
    sessions=list(pd.to_datetime(bundle.sessions))
    previous={sessions[i]:sessions[i-1] for i in range(1,len(sessions))}
    totals=defaultdict(int)
    for scenario in saved.comparison.to_dict('records'):
        folder=saved.path/'scenarios'/scenario['arm']/scenario['basket']/scenario['variant']/f"{int(scenario['cost_bps'])}bps"
        daily, trades, market, actions, stops, orders=[pd.read_parquet(folder/f'{name}.parquet')
            for name in ('daily','trades','market_trades','action_events','stops','orders')]
        by_trade,by_market,by_action=map(grouped,(trades,market,actions))
        units={};cash=nav=1.;failed=False
        for row in daily.to_dict('records'):
            key=(pd.Timestamp(row['date']),row['mark'])
            if row['status']!='ok':
                failed=True
                assert pd.isna(row['nav']) and pd.isna(row['return'])
                assert not by_trade[key] and not by_market[key]
                continue
            assert not failed
            for event in by_action[key]:
                lot=event['source_security_id'];old_native,amount=units[lot]
                assert old_native==event['predecessor_id']
                old=quotes[(previous[key[0]],old_native)]
                raw=amount*old['adj_close']/old['close']
                close(raw,event['signed_raw_entitlement'])
                entitlement=raw*event['cash_per_raw_share']
                close(entitlement,event['signed_cash']);cash+=entitlement
                if event['stock_ratio']:
                    successor=quotes[(key[0],event['successor_id'])]
                    new_amount=raw*event['stock_ratio']*successor['open']/successor['adj_open']
                    close(new_amount,event['new_adjusted_units'])
                    units[lot]=(event['successor_id'],new_amount)
                    if pd.notna(event['old_stop_price']):
                        close((amount*event['old_stop_price']-entitlement)/new_amount,event['new_stop_price'])
                else:del units[lot]
                totals['entitlements_reconciled']+=1
            instructions=by_trade[key]
            expected_groups=defaultdict(list)
            for trade in instructions:
                lot=trade['source_security_id'];native=trade['intrinio_security_id']
                old_native,old_amount=units.get(lot,(native,0.))
                assert old_native==native
                close(old_amount,trade['units_before'])
                close(trade['units_after']-old_amount,trade['units_delta'])
                close(trade['units_delta']*trade['price'],trade['signed_notional'])
                if trade['phase'] not in ('stop_gap','stop_intraday'):
                    field='adj_open' if key[1]=='open' else 'adj_close'
                    close(trade['price'],quotes[(key[0],native)][field])
                bucket=lot if trade['phase']=='stop_intraday' and key[1]!='open' else 'simultaneous_mark'
                expected_groups[(native,float(trade['price']),bucket)].append(trade)
                units[lot]=(native,trade['units_after'])
                totals['lot_trades_reconciled']+=1
            physical=by_market[key]
            actual_groups={(t['intrinio_security_id'],float(t['price']),t['execution_bucket']):t for t in physical}
            assert len(actual_groups)==len(physical) and set(actual_groups)==set(expected_groups)
            for group,items in expected_groups.items():
                close(sum(t['signed_notional'] for t in items),actual_groups[group]['signed_notional'])
                close(sum(abs(t['signed_notional']) for t in items),actual_groups[group]['lot_instruction_notional'])
            close(sum(t['signed_notional'] for t in instructions),sum(t['signed_notional'] for t in physical))
            close(sum(t['cost'] for t in instructions),sum(t['cost'] for t in physical))
            for trade in physical:
                close(abs(trade['signed_notional']),trade['absolute_notional'])
                close(trade['absolute_notional']*scenario['cost_bps']/10000,trade['cost'])
                cash-=trade['signed_notional']+trade['cost']
                totals['market_orders_reconciled']+=1
            field='adj_open' if key[1]=='open' else 'adj_close'
            long=short=0.
            for native,amount in units.values():
                if not amount:continue
                value=amount*quotes[(key[0],native)][field]
                if amount>0:long+=value
                else:short+=value
            close(cash,row['cash']);close(long,row['long_market_value']);close(short,row['short_market_value'])
            current=cash+long+short
            close(current,row['nav']);close(current/nav-1,row['return']);close(current-nav,row['net_pnl'])
            close(row['borrow_cost'],0.);nav=current
            totals['valid_marks_reconciled']+=1
        for stop in stops.to_dict('records'):
            quote=quotes[(pd.Timestamp(stop['date']),stop['intrinio_security_id'])]
            threshold=stop['stop_price']
            if stop['fill_type']=='gap_open':
                close(quote['adj_open'],stop['fill_price'])
                assert quote['adj_open']<=threshold if stop['side']=='long' else quote['adj_open']>=threshold
            else:
                close(threshold,stop['fill_price'])
                assert quote['adj_low']<=threshold if stop['side']=='long' else quote['adj_high']>=threshold
            totals['stop_fill_checks']+=1
        if not orders.empty:
            close(orders.intended_weight.abs().min(),.05);close(orders.intended_weight.abs().max(),.05)
            valid_opens={pd.Timestamp(r['date']):r for r in daily.to_dict('records')
                         if r['mark']=='open' and r['status']=='ok'}
            for order in orders[orders.status.eq('executed')].to_dict('records'):
                day=pd.Timestamp(order['date']);opening=quotes[(day,order['intrinio_security_id'])]['adj_open']
                weight=.05 if order['side']=='long' else -.05
                close(order['new_adjusted_units']*opening,weight*valid_opens[day]['nav_before_cost'])
                totals['fixed_weight_order_checks']+=1
            retired=orders[orders.status.eq('unfilled_retired_security')]
            assert retired.new_adjusted_units.eq(0).all()
            totals['retired_order_cash_slots']+=len(retired)
        if failed:
            assert pd.isna(scenario['total_return']) and pd.isna(scenario['sharpe_zero_cash_rate'])
        else:close(nav-1,scenario['total_return'])
        totals['scenarios_reconciled']+=1
    assert saved.ret_readiness.status.eq('withheld_incomplete_120_month_class_grid').all()
    assert totals['scenarios_reconciled']==64
    return {'status':'PASS','saved_run':str(saved.path),**totals,'model_fits':0,
        'excluded_financial_values_returned_or_used':False,'RET_signals_withheld':24}


if __name__=='__main__':
    result=verify(sys.argv[1] if len(sys.argv)>1 else Path.cwd())
    print(json.dumps(result,indent=2))
