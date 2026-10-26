#!/usr/bin/env python3
"""Independent monthly formulas, labels, inference, selection and accounting.

Does not import the producer's monthly rules, fitting or portfolio functions.
Prior source/feature correctness is inherited through exact independent pins;
all new target/RET/portfolio formulas are independently reconstructed here.
"""
from pathlib import Path
import hashlib, itertools, json, math

FEATURES = tuple(p + str(n) for p in ('zMOM', 'meanMOM', 'stdMOM') for n in (1, 3, 6, 9, 12))
PROBS = tuple('probability_' + str(k) for k in range(1, 11))
KEYS = ['source_security_id', 'signal_date']
SOURCE_COLUMNS = [*KEYS, 'intrinio_security_id', 'identity_eligible', 'identity_status',
                  'identity_review_required', 'resolution_status', 'prediction_eligible',
                  'prediction_status', 'outcome_month', 'outcome_start', 'outcome_end',
                  'history_years', 'history_complete', 'expected_history_sessions',
                  'valid_history_sessions', *FEATURES]


def need(value, message):
    if not value: raise ValueError(message)


def pin(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=h.hexdigest())


def check(item):
    need(pin(item['path']) == {k: item[k] for k in ('path', 'size_bytes', 'sha256')}, 'Independent byte binding')
    return Path(item['path'])


def read(path):
    need(Path(path).stat().st_size <= 32 * 1024**2, 'Independent metadata ceiling')
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream: json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return pin(p)


def shift(month, offset):
    y, m = map(int, month.split('-')); t = 12*y + m - 1 + offset
    return f'{t//12:04d}-{t%12+1:02d}'


def context(plan_path, phase):
    p = read(plan_path)
    need(hashlib.sha256(json.dumps({k:v for k,v in p.items() if k != 'plan_sha256'}, sort_keys=True,
                                  separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest() == p['plan_sha256'], 'Plan digest')
    for item in [p['protocol_pin'], *p['source_binding_pins'], *p['source_inputs'].values(), *p['code_pins'].values(), *p['helper_pins'].values()]: check(item)
    cfg = read(check(p['protocol_pin'])); root = Path(p['output_directory']); prep = read(root/'prepared/receipt.json')
    need(prep['plan_pin'] == pin(plan_path) and p['max_fits'] == 16 and not p['financial_2024_plus_read'] and not p['daily_prices_read'], 'Independent monthly scope')
    inventory = dict(prep['artifacts'])
    if phase == 'models':
        for n, y, k in itertools.product((5,10), (2020,2021,2022,2023), ('rf','xgb')):
            fp = root/f'folds/{n}y/{y}/{k}/FIT_COMPLETE.json'; done = read(fp)
            inventory[f'fit/{n}/{y}/{k}'] = pin(fp)
            for name in ('model', 'predictions', 'summary'): inventory[f'{n}/{y}/{k}/{name}'] = done[name]
            inventory.update({f'{n}/{y}/{k}/metrics/{name}': item for name,item in done['metrics'].items()})
    if phase == 'backtest':
        ap = root/'backtest/audit.json'; audit = read(ap); inventory['backtest_audit'] = pin(ap)
        inventory.update({'backtest/'+k:v for k,v in audit['artifacts'].items()})
    need(sum(v['size_bytes'] for v in inventory.values()) <= cfg['budgets']['derived_read_bytes_per_phase'], 'Independent derived read ceiling')
    for item in inventory.values(): check(item)
    vplan = save(root/'verification'/('read_plan_'+phase+'.json'), dict(status='SEALED_BEFORE_NUMERIC_READ', plan_pin=pin(plan_path),
        prepared_receipt_pin=pin(root/'prepared/receipt.json'), prepared_artifacts=prep['artifacts'], inventory=inventory,
        source_inputs=p['source_inputs'], source_projection_bytes=p['source_projection_bytes'], worker_pin=pin(__file__), new_fits=0))
    return p, cfg, root, prep, vplan


def rank_oracle(values):
    import numpy as np, pandas as pd
    a = np.asarray(values, dtype=float); result = np.full(len(a), np.nan); good = np.isfinite(a); n = int(good.sum())
    if n >= 10:
        twice_rank = (2 * pd.Series(a[good]).rank(ascending=False, method='average')).to_numpy(dtype=np.int64)
        result[good] = np.clip(1 + (10*(twice_rank-1)) // (2*n), 1, 10)
    return result


def verify_data(plan_path, guard=lambda: None):
    import numpy as np, pandas as pd, pyarrow.parquet as pq
    p, cfg, root, prep, vplan = context(plan_path, 'data')
    idx = read(check(p['source_inputs']['index'])); dates = idx['sessions']
    need(dates[-1] == '2023-12-29', 'No protected financial dates')
    ends = {m: max(d for d in dates if d.startswith(m)) for m in idx['months']}
    mi, ni = {m:i for i,m in enumerate(idx['months'])}, {n:i for i,n in enumerate(idx['natives'])}
    close = np.load(check(p['source_inputs']['month_closes']), mmap_mode='r', allow_pickle=False)
    need(close.shape == (len(mi), len(ni)), 'Source monthly array shape')
    summary = []
    for n in (5,10):
        guard(); source = pq.read_table(check(p['source_inputs'][f'panel{n}']), columns=SOURCE_COLUMNS).to_pandas()
        panel = pq.read_table(check(prep['artifacts'][f'panel{n}'])).to_pandas()
        need(len(panel) == len(source) == 152307 and not panel.duplicated(KEYS).any(), 'Original cohorts retained')
        pd.testing.assert_frame_equal(panel[SOURCE_COLUMNS], source[SOURCE_COLUMNS], check_exact=True)
        expected = np.full(len(panel), np.nan); classes = np.full(len(panel), np.nan)
        for signal, g in source.groupby('signal_date', sort=True):
            om = shift(signal[:7], 1)
            need(signal == ends[signal[:7]] and om == g.outcome_month.iloc[0] and '2010-01' <= om <= '2023-12', 'Independent calendar')
            for row in g.loc[g.prediction_eligible].itertuples():
                j = ni.get(row.intrinio_security_id)
                if j is not None:
                    start, finish = close[mi[signal[:7]], j], close[mi[om], j]
                    if np.isfinite([start, finish]).all() and min(start, finish) > 0:
                        expected[row.Index] = finish/start-1
            classes[g.index] = rank_oracle(expected[g.index])
            need(panel.loc[g.index, 'target_entry_date'].eq(signal).all() and panel.loc[g.index, 'label_available_at'].eq(ends[om]).all()
                 and panel.loc[g.index, 'target_exit_date'].eq(ends[om]).all(), 'Exact close completion')
        np.testing.assert_allclose(panel.label_return.to_numpy(), expected, rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(panel.target_class.to_numpy(), classes)
        need(panel.target_available.tolist() == np.isfinite(classes).tolist(), 'No target imputation')
        means = {}; counts = {}
        for (month,k),g in panel.loc[panel.target_available].groupby(['outcome_month','target_class']):
            means[(month,int(k))] = math.fsum(g.label_return)/len(g); counts[(month,int(k))] = len(g)
        got = read(check(prep['artifacts'][f'payoffs{n}'])); grid = pq.read_table(check(prep['artifacts'][f'payoff_grid{n}'])).to_pandas()
        need(len(got) == 48 and len(grid) == 48*12*n*10 and not grid.duplicated(['signal_date','outcome_month','class_id']).any(), 'Exact monthly payoff grid')
        for signal, actual in got.items():
            wanted = [shift(signal[:7], j) for j in range(-12*n+1, 1)]
            expected_mu = ([math.fsum(means[(m,k)] for m in wanted)/(12*n) for k in range(1,11)]
                           if all((m,k) in means for m in wanted for k in range(1,11)) else None)
            if expected_mu is None: need(actual is None, 'Missing monthly cell must withhold vector')
            else: np.testing.assert_allclose(actual, expected_mu, rtol=0, atol=1e-14)
            g = grid.loc[grid.signal_date.eq(signal)]
            need(set(g.outcome_month) == set(wanted) and max(wanted) == signal[:7], 'RET signal-month inclusion')
            for row in g.itertuples():
                key = (row.outcome_month, row.class_id)
                need(row.count == counts.get(key,0), 'Class payoff count')
                if key in means: need(abs(row.mean_return-means[key]) <= 1e-14, 'Class payoff mean')
                else: need(pd.isna(row.mean_return), 'Missing class mean invented')
        coverage = pq.read_table(check(prep['artifacts'][f'coverage{n}'])).to_pandas()
        need(len(coverage) == 168 and int(coverage.original_members.sum()) == 152307, 'Original monthly denominator')
        for row in coverage.itertuples():
            g = panel.loc[panel.signal_date.eq(row.signal_date)]
            need(row.original_members == len(g) and row.prediction_eligible == int(g.prediction_eligible.sum())
                 and row.labelled == int(g.target_available.sum()) and row.missing_targets == int((g.prediction_eligible & ~g.target_available).sum()), 'Coverage counters')
        summary.append(dict(history_years=n, original_rows=len(panel), eligible=int(panel.prediction_eligible.sum()),
                            labelled=int(panel.target_available.sum()), RET_complete_vectors=sum(v is not None for v in got.values())))
    return save(root/'verification/data.json', dict(status='PASS', plan_pin=pin(plan_path), prepared_receipt_pin=pin(root/'prepared/receipt.json'),
                verification_read_plan_pin=vplan, inherited_source_and_features_verified=True,
                new_monthly_returns_classes_RET_and_cutoffs_independently_checked=True, summary=summary,
                worker_pin=pin(__file__), new_fits=0, financial_2024_plus_read=False))


def verify_backtest(plan_path, guard=lambda: None):
    import numpy as np, pandas as pd, pyarrow.parquet as pq
    p, cfg, root, prep, vplan = context(plan_path, 'backtest')
    audit = read(root/'backtest/audit.json'); model_proof = read(root/'verification/models.json')
    need(audit['plan_pin'] == pin(plan_path) and audit['model_proof_pin'] == pin(root/'verification/models.json') and model_proof['status'] == 'PASS', 'Backtest model binding')
    artifacts = audit['artifacts']; comp = pq.read_table(check(artifacts['comparison'])).to_pandas()
    need(len(comp) == 32 and not comp.duplicated(['arm','period','cost_bps']).any(), 'All fixed scenario rows')
    for n, kind in itertools.product((5,10), ('rf','xgb')):
        guard(); arm = f'{kind}_paper15_{n}y'
        forecasts = pd.concat([pq.read_table(check(model_proof['forecasts'][f'{n}/{y}/{kind}'])).to_pandas()
                               for y in (2020,2021,2022,2023)], ignore_index=True)
        need(not any(str(c).startswith(('label','target','future','realized')) for c in forecasts), 'Signal-only selection inputs')
        mu = read(check(prep['artifacts'][f'payoffs{n}'])); panel = pq.read_table(check(prep['artifacts'][f'panel{n}'])).to_pandas()
        ledger = pq.read_table(check(artifacts[f'{arm}/selection/original_ledger'])).to_pandas()
        selected = pq.read_table(check(artifacts[f'{arm}/selection/selected'])).to_pandas()
        need(list(map(tuple, ledger[KEYS].to_numpy())) == list(map(tuple, forecasts[KEYS].to_numpy())), 'Original forecast keys preserved')
        expected_picks = []
        for signal, g in forecasts.groupby('signal_date', sort=True):
            eligible = g.loc[g.prediction_eligible & ~g.intrinio_security_id.isin(cfg['backtest']['execution_restrictions'])]
            if mu[signal] is None: continue
            scored = [(row.source_security_id, math.fsum(float(getattr(row,name))*float(v) for name,v in zip(PROBS,mu[signal]))) for row in eligible.itertuples()]
            num = min(10, len(scored)//2); long = sorted(scored,key=lambda x:(-x[1],x[0]))[:num]
            short = sorted((x for x in scored if x[0] not in {v[0] for v in long}),key=lambda x:(x[1],x[0]))[:num]
            for side, picks, weight in [('long',long,.5/num if num else 0),('short',short,-.5/num if num else 0)]:
                for sid,score in picks: expected_picks.append((sid,signal,side,weight,score))
        need(len(selected) == len(expected_picks), 'Selection count')
        actual_picks = {(row.source_security_id,row.signal_date):(row.side,row.intended_weight,row.RET) for row in selected.itertuples()}
        for sid,signal,side,weight,score in expected_picks:
            got = actual_picks[(sid,signal)]
            need(got[0] == side and abs(got[1]-weight) < 1e-15 and abs(got[2]-score) < 2e-12, 'Independent RET basket')
        # Reconcile all unselected rows and scores too, not only final baskets.
        chosen = set(actual_picks)
        need(ledger.selected.tolist() == [(a,b) in chosen for a,b in ledger[KEYS].itertuples(index=False,name=None)], 'Original selection flags')
        outcomes = {(row.source_security_id,row.signal_date): row.label_return for row in panel.itertuples()}
        for period,first in [('full48','2020-01'),('matched47','2020-02')]:
            months = sorted(forecasts.loc[forecasts.outcome_month.ge(first),'outcome_month'].unique())
            for fee in (0,5,10,25):
                base = f'{arm}/{period}/cost_{fee:02d}bps/'
                monthly = pq.read_table(check(artifacts[base+'monthly'])).to_pandas()
                positions = pq.read_table(check(artifacts[base+'positions'])).to_pandas()
                failures = pq.read_table(check(artifacts[base+'failures'])).to_pandas()
                annual = pq.read_table(check(artifacts[base+'annual'])).to_pandas()
                cash = 1.; liabilities = {}; nav = 1.; records = []; bad = []; wanted_positions = []
                for month in months:
                    g = selected.loc[selected.outcome_month.eq(month)]
                    if g.empty: bad.append(dict(outcome_month=month,reason='no_complete_RET_selection')); break
                    vals = []
                    for row in g.itertuples():
                        v = outcomes.get((row.source_security_id,row.signal_date), np.nan)
                        if not np.isfinite(v) or v < -1:
                            bad.append(dict(outcome_month=month, signal_date=row.signal_date, source_security_id=row.source_security_id,
                                            intrinio_security_id=row.intrinio_security_id, side=row.side,reason='missing_selected_monthly_return'))
                        vals.append((row,v))
                    if bad: break
                    start = cash + math.fsum(liabilities.values()); need(abs(start-nav) < 1e-11, 'Self-financing opening wealth')
                    target = {row.source_security_id: row.intended_weight*start for row,_ in vals}
                    trade = {sid:target.get(sid,0)-liabilities.get(sid,0) for sid in target.keys() | liabilities.keys()}
                    cost = fee/10000*math.fsum(abs(v) for v in trade.values())
                    cash -= math.fsum(trade.values()) + cost
                    lp = math.fsum(target[row.source_security_id]*v for row,v in vals if row.side == 'long')
                    sp = math.fsum(target[row.source_security_id]*v for row,v in vals if row.side == 'short')
                    liabilities = {row.source_security_id:target[row.source_security_id]*(1+v) for row,v in vals}
                    nav = cash + math.fsum(liabilities.values())
                    records.append(dict(outcome_month=month,nav_start=start,nav_end=nav,gross_return=(lp+sp)/start,
                        net_return=nav/start-1,long_contribution=lp/start,short_contribution=sp/start,
                        trading_cost=cost,cost_contribution=-cost/start,one_way_turnover=math.fsum(abs(v) for v in trade.values())/start,cash_end=cash))
                    for row,v in vals:
                        sid=row.source_security_id;wanted_positions.append(dict(outcome_month=month,source_security_id=sid,
                            target_notional=target[sid],monthly_return=v,pnl=target[sid]*v,end_notional=liabilities[sid]))
                    if nav <= 0 or not math.isfinite(nav): bad.append(dict(outcome_month=month,reason='nonpositive_portfolio_wealth')); break
                need(len(monthly) == len(records), 'No omitted monthly path rows')
                for k in ('outcome_month',): need(monthly[k].tolist() == [x[k] for x in records], 'Monthly time grid')
                for k in ('nav_start','nav_end','gross_return','net_return','long_contribution','short_contribution','trading_cost','cost_contribution','one_way_turnover','cash_end'):
                    np.testing.assert_allclose(monthly[k], [x[k] for x in records], rtol=2e-11, atol=2e-12)
                if wanted_positions:
                    pd.testing.assert_frame_equal(positions.sort_values(['outcome_month','source_security_id']).reset_index(drop=True),
                        pd.DataFrame(wanted_positions).sort_values(['outcome_month','source_security_id']).reset_index(drop=True), check_exact=False, rtol=2e-11,atol=2e-12)
                need(failures.to_dict('records') == bad, 'Exact fail-closed missing outcome list')
                row = comp.loc[comp.arm.eq(arm)&comp.period.eq(period)&comp.cost_bps.eq(fee)].iloc[0]
                complete = len(records) == len(months) and not bad
                need(row.complete_monthly_path == complete and row.observed_months == len(records) and row.expected_months == len(months), 'Full-period completeness')
                fields = ('total_return','cagr','annualized_arithmetic_mean','annualized_volatility','sharpe_zero_cash_rate','monthly_maximum_drawdown','total_one_way_turnover','trading_cost_paid')
                if complete:
                    rets = np.array([x['net_return'] for x in records]); wealth = np.r_[1.,np.cumprod(1+rets)]
                    sd = float(np.std(rets,ddof=1)); wanted = dict(total_return=wealth[-1]-1,cagr=wealth[-1]**(12/len(rets))-1,
                        annualized_arithmetic_mean=12*rets.mean(),annualized_volatility=sd*math.sqrt(12),
                        sharpe_zero_cash_rate=math.sqrt(12)*rets.mean()/sd if sd>0 else np.nan,
                        monthly_maximum_drawdown=float((wealth/np.maximum.accumulate(wealth)-1).min()),
                        total_one_way_turnover=sum(x['one_way_turnover'] for x in records),trading_cost_paid=sum(x['trading_cost'] for x in records))
                    for k in fields: need(abs(row[k]-wanted[k]) < 5e-10 or (pd.isna(row[k]) and pd.isna(wanted[k])), 'Independent monthly statistic '+k)
                else: need(all(pd.isna(row[k]) for k in fields), 'Withheld full-period metrics must stay missing')
                need(len(annual)==4 and annual.year.tolist()==[2020,2021,2022,2023], 'Annual aggregate rows')
                for ar in annual.itertuples():
                    g=[x for x in records if x['outcome_month'].startswith(str(ar.year))]
                    need(ar.observed_months==len(g), 'Annual observation denominator')
                    if complete: need(abs(ar.total_return-(math.prod(1+x['net_return'] for x in g)-1))<5e-11, 'Continuous annual returns')
                    else: need(pd.isna(ar.total_return), 'Incomplete annual result')
    need(int(comp.complete_monthly_path.sum()) == audit['complete_paths'] and int((~comp.complete_monthly_path).sum()) == audit['withheld_paths'], 'Scenario coverage counters')
    return save(root/'verification/backtest.json', dict(status='PASS',plan_pin=pin(plan_path),model_proof_pin=pin(root/'verification/models.json'),
        audit_pin=pin(root/'backtest/audit.json'),verification_read_plan_pin=vplan,worker_pin=pin(__file__),verified_scenarios=32,
        complete_paths=audit['complete_paths'],withheld_paths=audit['withheld_paths'],new_fits=0,financial_2024_plus_read=False,
        metric_frequency='monthly',same_close_execution_is_idealized=True))


def verify_models(plan_path,guard=lambda:None):
    import joblib,numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,prep,vplan=context(plan_path,'models');data=read(root/'verification/data.json')
    need(data['status']=='PASS' and data['prepared_receipt_pin']==pin(root/'prepared/receipt.json'),'Independent prepared source proof')
    receipts={};forecasts={};summaries=[];all_metrics=[]
    for n in (5,10):
        panel=pq.read_table(check(prep['artifacts'][f'panel{n}'])).to_pandas()
        for year,kind in itertools.product((2020,2021,2022,2023),('rf','xgb')):
            guard();directory=root/f'folds/{n}y/{year}/{kind}';done=read(directory/'FIT_COMPLETE.json')
            need(done['plan_pin']==pin(plan_path) and done['status']=='FIT_AND_PREDICT_COMPLETE','Wrong saved model scope')
            mask=panel.prediction_eligible & panel.target_available & panel.outcome_month.between(f'{year-n}-01',f'{year-1}-12') & panel.label_available_at.le(f'{year-1}-12-31')
            test=panel.outcome_month.str.startswith(str(year));qualified=test&panel.prediction_eligible
            X=panel.loc[mask,list(FEATURES)].to_numpy(dtype=np.float32);y=panel.loc[mask,'target_class'].to_numpy(dtype=np.int64);XP=panel.loc[qualified,list(FEATURES)].to_numpy(dtype=np.float32)
            metadata=done['fit_metadata']
            def array_sha(a):return 'sha256:'+hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
            encoded=y if kind=='rf' else y-1
            need(array_sha(X)==metadata['prepared_train_matrix_sha256'] and array_sha(XP)==metadata['prepared_prediction_matrix_sha256']
              and array_sha(encoded)==metadata['encoded_training_labels_sha256'],'Independent ordered fit/predict matrices and labels')
            need(metadata['matrix_width']==15 and metadata['fit_count']==1 and metadata['explicit_parameters']==cfg['parameters'][kind],'Feature width/settings drift')
            estimator=joblib.load(check(done['model']));params=estimator.get_params(deep=False)
            need(all(params[k]==v for k,v in cfg['parameters'][kind].items()),'Saved estimator settings differ')
            need(list(estimator.classes_)==(list(range(1,11)) if kind=='rf' else list(range(10))),'Native class mapping')
            if kind=='rf':prob=estimator.predict_proba(XP)
            else:
                z=estimator.predict(XP,output_margin=True).astype(np.float64);z-=z.max(axis=1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True)
            forecast=pq.read_table(check(done['predictions'])).to_pandas();expected=panel.loc[test]
            need(len(forecast)==len(expected) and list(map(tuple,forecast[['source_security_id','signal_date']].to_numpy()))==list(map(tuple,expected[['source_security_id','signal_date']].to_numpy())),'Original forecast keys dropped/reordered')
            for name in ('prediction_eligible','intrinio_security_id','outcome_month','outcome_start','outcome_end'):
                need(forecast[name].fillna('').tolist()==expected[name].fillna('').tolist(),'Forecast original identity/clock '+name)
            got=forecast.loc[forecast.prediction_eligible,list(PROBS)].to_numpy(dtype=float)
            need(np.allclose(prob,got,rtol=2e-12,atol=2e-12) and np.isfinite(got).all() and (got>=0).all() and np.allclose(got.sum(axis=1),1,rtol=0,atol=1e-8),'Independent saved-model inference')
            need(forecast.loc[~forecast.prediction_eligible,list(PROBS)].isna().all().all(),'Ineligible predictions invented')
            actual_labels=panel.loc[qualified,'target_class'].to_numpy(dtype=float);observed=np.isfinite(actual_labels);months=panel.loc[qualified,'outcome_month'].to_numpy()
            prior=np.bincount(y,minlength=11)[1:]/len(y);saved_summaries=read(check(done['summary']))
            for ref,values in [('model',prob),('uniform',np.full(prob.shape,.1)),('train_prior',np.tile(prior,(len(prob),1)))]:
                s=next(r for r in saved_summaries if r['reference']==ref);lab=actual_labels[observed].astype(int);v=values[observed];mo=months[observed]
                loss=-np.log(np.maximum(v[np.arange(len(lab)),lab-1],1e-15));acc=(v.argmax(axis=1)+1==lab).astype(float);errors=v.copy();errors[np.arange(len(lab)),lab-1]-=1;brier=np.square(errors).sum(axis=1)
                for key,val in [('pooled_log_loss',loss.mean()),('pooled_accuracy',acc.mean()),('pooled_brier_sum10',brier.mean()),
                   ('mean_monthly_log_loss',np.mean([loss[mo==m].mean() for m in sorted(set(mo))])),('mean_monthly_accuracy',np.mean([acc[mo==m].mean() for m in sorted(set(mo))])),
                   ('mean_monthly_brier_sum10',np.mean([brier[mo==m].mean() for m in sorted(set(mo))]))]:
                    need(abs(s[key]-val)<=2e-12,'Independent probability metric '+key)
                need(s['prediction_rows']==len(prob) and s['labelled_rows']==len(lab),'Metric denominators')
                all_metrics.append(s)
                if ref=='model':
                    # Independently reconcile each class's precision/recall and
                    # confusion counts, preserving zero-support classes.
                    cp=pq.read_table(check(done['metrics'][ref+'/mode_class_performance'])).to_pandas();cf=pq.read_table(check(done['metrics'][ref+'/confusion'])).to_pandas();pred=v.argmax(axis=1)+1
                    for k in range(1,11):
                        r=cp.loc[cp.period.eq('all')&cp.class_id.eq(k)].iloc[0];hits=int(((lab==k)&(pred==k)).sum());pn=int((pred==k).sum());an=int((lab==k).sum())
                        need(r.true_positive_count==hits and r.actual_support==an and r.labelled_predicted_support==pn,'Class supports/hits')
                        if pn:need(abs(r.precision-hits/pn)<=2e-12,'Class precision')
                        if an:need(abs(r.recall-hits/an)<=2e-12,'Class recall')
                        for j in range(1,11):need(int(cf.loc[cf.period.eq('all')&cf.actual_class.eq(k)&cf.predicted_class.eq(j),'count'].iloc[0])==int(((lab==k)&(pred==j)).sum()),'Confusion count')
            forecasts[f'{n}/{year}/{kind}']=done['predictions'];receipts[f'{n}/{year}/{kind}']=pin(directory/'FIT_COMPLETE.json');summaries.append({'years':n,'year':year,'kind':kind,'train_rows':len(X),'forecasts':len(prob),'observed_labels':int(observed.sum())})
            del estimator;guard()
    metrics_pin=save(root/'verification/prediction_metrics.json',all_metrics)
    return save(root/'verification/models.json',{'status':'PASS','plan_pin':pin(plan_path),'data_proof_pin':pin(root/'verification/data.json'),'verification_read_plan_pin':vplan,
       'worker_pin':pin(__file__),'verified_fits':16,'fit_receipts':receipts,'forecasts':forecasts,'metrics_pin':metrics_pin,'summary':summaries,'new_fits':0,'financial_2024_plus_read':False})
