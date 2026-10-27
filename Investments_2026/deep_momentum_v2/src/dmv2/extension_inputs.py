"""Additive, explicitly scoped Paper15 input audit through April 2026.

This module cannot fit, select portfolios or replay trades. Old holdout guards
and old artifacts stay intact. Dates and price columns are sealed before reads.
"""
from collections import Counter
from pathlib import Path
import math
import time
import resource

import numpy as np
import pandas as pd

from .paper15_baseline import FEATURES, KEYS, deciles, need, shift
from .paper15_baseline_data import pin, check, read, save, table_save, canonical
from .data import MEMBERSHIP_COLUMNS, BRIDGE_COLUMNS, LINK_COLUMNS
from .universe import build_snapshot

META = ('trading_date', 'intrinio_security_id', 'frequency', 'intraperiod',
        'price_quality_status', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid')
OHLC = ('open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close')
COHORT = ('source_security_id', 'signal_date', 'indices', 'membership_tickers',
          'unified_security_id', 'intrinio_security_id', 'identity_status',
          'identity_review_required', 'resolution_status', 'source_identity_status',
          'identity_eligible', 'identity_eligibility_policy', 'identity_eligibility_reason')
# These are demonstrated conflicts, not exclusions selected from future prices.
QUARANTINED = {'siblis_ticker_company_00326c7d9ad13903e5e3',
               'siblis_isin_c297a5f81a23c9f90503'}
INTERVAL_LIMITED = {'siblis_ticker_company_1c21a7b293fd137c59c9',
                    'siblis_isin_b284bb7497616f5afba0'}
CODE = ('src/dmv2/extension_inputs.py', 'scripts/run_extension_inputs.py',
        'scripts/verify_extension_inputs.py')


def scope(cfg):
    need(cfg['schema'] == 'dmv2_paper15_extension_inputs_v1', 'Explicit extension protocol required')
    need(cfg['maximum_fits'] == cfg['maximum_backtests'] == 0, 'Input audit cannot fit or backtest')
    need(cfg['first_outcome'] == '2024-01' and cfg['last_outcome'] == '2026-04', 'Frozen outcome bounds')
    need(cfg['financial_start'] == '2000-01-01' and cfg['financial_end'] == '2026-04-30', 'Frozen financial scope')
    need(cfg['features'] == list(FEATURES) and cfg['windows_years'] == [5, 10], 'Unchanged Paper15 arms')
    need(cfg['parameter_search'] is False and cfg['source_upgrade'] is False, 'No tuning or source promotion')


def qualify(records):
    """Same structural provisional policy in the new explicit date scope.

    Approved BK/XOM shadow price intervals end in 2023. Do not fall back to
    demonstrated wrong original routes or pretend those intervals were extended.
    Keep their original rows and mark the concrete route dependency separately.
    """
    out = []; seen = set()
    for original in records:
        r = dict(original); day = r['signal_date']; sid = r['source_security_id']
        need('2023-12-01' <= day <= '2026-03-31' and sid, 'New cohort outside scoped signal dates')
        need((sid, day) not in seen, 'Duplicate original member'); seen.add((sid, day))
        r['source_identity_status'] = r['identity_status']
        out.append(r)
    for r in out:
        sid, native = r['source_security_id'], r.get('intrinio_security_id')
        status = r['identity_status']
        if sid in INTERVAL_LIMITED and r['signal_date'] <= '2023-12-29':
            from .bk_xom_shadow_sources import ROUTES
            unified, old, key, provider, start, end = ROUTES[sid]
            need(r['unified_security_id'] == unified and native == old, 'Reviewed route metadata changed')
            r['intrinio_security_id'] = native = key
            if status == 'target_collision':
                status = 'resolved' if r['identity_review_required'] is False and r['resolution_status'] == 'resolved' else 'needs_review'
        if status == 'resolved' and (r.get('identity_review_required') is not False or r.get('resolution_status') != 'resolved'):
            status = 'needs_review'
        admitted = (status in ('resolved', 'needs_review') and isinstance(native, str) and bool(native.strip())
                    and isinstance(r.get('unified_security_id'), str) and bool(r['unified_security_id'].strip())
                    and type(r.get('identity_review_required')) is bool and r.get('resolution_status') == 'resolved'
                    and (status == 'resolved' or r['identity_review_required'] is True))
        reason = ('provisional_review_flag_accepted' if r.get('identity_review_required') is True
                  else 'unique_unflagged_link') if admitted else 'blocked_' + status
        if sid in QUARANTINED:
            status, admitted, reason = 'known_wrong_route_quarantined', False, 'blocked_known_BBBY_conflict'
        elif sid in INTERVAL_LIMITED and r['signal_date'] > '2023-12-29':
            status, admitted, reason = 'reviewed_route_interval_ended', False, 'BK_XOM_post2023_route_not_yet_approved'
        r.update(identity_status=status, identity_eligible=bool(admitted),
                 identity_eligibility_policy='provisional_unique_resolved_link', identity_eligibility_reason=reason)
    # Recheck collisions after computational-route changes, including all rows.
    counts = Counter((r['signal_date'], r['intrinio_security_id']) for r in out if r.get('intrinio_security_id'))
    for r in out:
        if r.get('intrinio_security_id') and counts[(r['signal_date'], r['intrinio_security_id'])] > 1:
            r.update(identity_status='target_collision', identity_eligible=False, identity_eligibility_reason='blocked_target_collision')
    return [{k: r.get(k) for k in COHORT} for r in out]


def projection(item, columns, start=None, end=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(check(item)); need(set(columns) <= set(pf.schema_arrow.names), 'Projection schema')
    groups = []; cost = 0; rows = 0
    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        if start:
            s = rg.column(pf.schema_arrow.names.index('trading_date')).statistics
            if s and s.has_min_max and (s.max < start or s.min > end): continue
        groups.append(i); rows += rg.num_rows
        cost += sum(rg.column(j).total_compressed_size for j in range(rg.num_columns)
                    if rg.column(j).path_in_schema in columns)
    return dict(pin=item, columns=list(columns), row_groups=groups, start=start, end=end,
                projected_compressed_bytes=cost, candidate_rows=rows)


def parent_proof(root, name):
    folder = root/'outputs'/name
    ledger_pin = pin(folder/'ledger.json'); ledger = read(check(ledger_pin))
    need(ledger['status'] == 'COMPLETE_VERIFIED', 'Verified parent required')
    phase_pin = next(x['receipt'] for x in ledger['completed'] if x['key'] == 'verify_data')
    phase = read(check(phase_pin)); proof_pin = phase['result']; proof = read(check(proof_pin))
    need(proof['status'] == 'PASS' and proof['plan_pin'] == ledger['plan_pin'], 'Parent proof binding')
    prep_pin = proof['prepared_receipt_pin']; prep = read(check(prep_pin))
    return prep, [ledger_pin, phase_pin, proof_pin, prep_pin, ledger['plan_pin']]


def seal(code_root, data_root, protocol, output, destination):
    import exchange_calendars as xc
    from .catalog import SourceCatalog
    code, data = Path(code_root).resolve(), Path(data_root).resolve(); cfg = read(protocol); scope(cfg)
    need(not Path(output).exists(), 'Preserve old attempts; choose a new output directory')
    history, hp = parent_proof(data, 'paper15_ret_baseline/20261026-v3')
    monthly, mp = parent_proof(data, 'paper15_monthly_baseline/20261026-v1')
    old = {k: history['artifacts'][k] for k in ('index', 'valid', 'present', 'month_closes')}
    old.update({f'panel{n}': monthly['artifacts'][f'panel{n}'] for n in (5, 10)})
    for item in old.values(): check(item)
    index = read(check(old['index'])); need(index['sessions'][-1] == '2023-12-29', 'Pinned parent end')
    catalog = SourceCatalog(data); source_pins = [pin(data/'configs/data_sources.local.json')]
    for s in catalog: source_pins.append(pin(s.base/'MANIFEST.json'))
    specs = []
    for src, table, cols in [('siblis', 'membership_by_index_by_date', MEMBERSHIP_COLUMNS),
                             ('identity', 'security_identity_bridge', BRIDGE_COLUMNS),
                             ('intrinio', 'intrinio_identity_links', LINK_COLUMNS)]:
        for obj in catalog.table_objects(src, table):
            obj.check(hashes=True)
            specs.append(dict(kind=src, table=table, **projection(pin(obj.path), cols,
                '2023-12-01' if src == 'siblis' else None, '2026-03-31' if src == 'siblis' else None)))
    prices = []
    for obj in catalog.table_objects('intrinio', 'intrinio_daily_prices'):
        obj.check(hashes=True)
        spec = projection(pin(obj.path), (*META, *OHLC), cfg['financial_start'], cfg['financial_end'])
        if spec['row_groups']: prices.append(spec)
    cal = xc.get_calendar('XNYS', start='2000-01-01', end='2026-04-30')
    sessions = list(cal.sessions.strftime('%Y-%m-%d'))
    need(sessions[:len(index['sessions'])] == index['sessions'], 'Calendar must preserve all historical sessions')
    ends = {d[:7]: d for d in sessions}
    signals = [ends[shift(m, -1)] for m in ends if cfg['first_outcome'] <= m <= cfg['last_outcome']]
    cost = sum(x['projected_compressed_bytes'] for x in prices+specs) + sum(x['size_bytes'] for x in old.values())
    need(cost <= cfg['budgets']['source_projection_bytes_per_pass'], 'Measured projection exceeds frozen ceiling')
    p = dict(schema=cfg['schema'], status='SEALED_BEFORE_EXTENSION_VALUE_READ', output_directory=str(Path(output).resolve()),
             code_root=str(code), data_root=str(data), protocol_pin=pin(protocol), parent_pins=hp+mp,
             source_pins=source_pins, source_metadata=specs, price_specs=prices, parent_inputs=old,
             sessions=sessions, signals=signals, projected_bytes_per_pass=cost, max_fits=0, max_backtests=0,
             helper_pins={n:pin(code/'src/dmv2'/n) for n in ('paper15_baseline.py', 'paper15_baseline_data.py',
                         'universe.py', 'data.py', 'bk_xom_shadow_sources.py')},
             code_pins={n:pin(code/n) for n in CODE})
    p['plan_sha256'] = canonical(p); save(destination, p); return p


def validate(path):
    p = read(path); need(p['plan_sha256'] == canonical({k:v for k,v in p.items() if k != 'plan_sha256'}), 'Plan changed')
    for item in [p['protocol_pin'], *p['parent_pins'], *p['source_pins'], *p['code_pins'].values(),
                 *p['helper_pins'].values(), *p['parent_inputs'].values()]: check(item)
    cfg = read(check(p['protocol_pin'])); scope(cfg)
    need(p['max_fits'] == p['max_backtests'] == 0 and len(p['signals']) == 28, 'Input-only scope changed')
    return p


def read_metadata(p):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    tables = {}
    for spec in p['source_metadata']:
        pf = pq.ParquetFile(check(spec['pin'])); parts = []
        for g in spec['row_groups']:
            t = pf.read_row_group(g, columns=spec['columns'], use_threads=False)
            if spec['kind'] == 'siblis':
                mask = pc.and_(pc.is_in(t['trading_date'], value_set=pa.array(p['signals'])),
                               pc.is_in(t['index_id'], value_set=pa.array(['sp500', 'sp400'])))
                t = t.filter(mask)
            parts.append(t)
        tables.setdefault(spec['kind'], []).extend(parts)
    members, bridge, links = (pa.concat_tables(tables[k]) for k in ('siblis', 'identity', 'intrinio'))
    rows = []; exclusions = []
    for signal in p['signals']:
        t = members.filter(pc.equal(members['trading_date'], signal))
        snapshot = build_snapshot(t, bridge, links, as_of=signal, universe='combined', mode='historical_reconstruction')
        need(snapshot.table.num_rows > 0 and {r['index_id'] for r in t.to_pylist()} == {'sp500', 'sp400'}, 'Both dated sleeves required')
        for r in snapshot.table.to_pylist():
            r['signal_date'] = r.pop('as_of'); rows.append(r)
        exclusions.extend(snapshot.excluded.to_pylist())
    return pd.DataFrame(qualify(rows)).sort_values(list(KEYS), kind='stable').reset_index(drop=True), pd.DataFrame(exclusions)


def quote_blocks(p, natives, missing_history):
    """Date/ID masks precede financial conversion; no values beyond April 2026.

    Mixed Parquet chunks can transiently decode excluded Arrow values. Such
    values never reach Python, returned batches, statistics or saved artifacts.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    for count, spec in enumerate(p['price_specs'], 1):
        pf = pq.ParquetFile(check(spec['pin']))
        for g in spec['row_groups']:
            keys = pf.read_row_group(g, columns=['trading_date', 'intrinio_security_id'], use_threads=False)
            dates, ids = keys['trading_date'], keys['intrinio_security_id']
            need(all(c.null_count == 0 for c in keys.columns), 'Null price keys')
            isnew = pc.greater_equal(dates, '2024-01-01')
            idsok = pc.or_(pc.and_(isnew, pc.is_in(ids, value_set=pa.array(natives, type=pa.string()))),
                          pc.and_(pc.invert(isnew), pc.is_in(ids, value_set=pa.array(missing_history, type=pa.string()))))
            mask = pc.and_(idsok, pc.and_(pc.greater_equal(dates, spec['start']), pc.less_equal(dates, spec['end'])))
            if not pc.any(mask).as_py(): continue
            selected = keys.filter(mask)
            values = pf.read_row_group(g, columns=[c for c in spec['columns'] if c not in keys.column_names], use_threads=False).filter(mask)
            for name in values.column_names: selected = selected.append_column(name, values[name])
            yield selected.to_pandas()
        if count % 10 == 0:
            print('{"phase":"source_projection","files_completed":%d,"files":%d}' % (count, len(p['price_specs'])), flush=True)


def ingest(p, cohorts):
    old = p['parent_inputs']; oi = read(check(old['index'])); old_ids = oi['natives']; old_days = oi['sessions']
    natives = sorted(set(old_ids) | set(cohorts.loc[cohorts.identity_eligible, 'intrinio_security_id']))
    new_ids = sorted(set(natives)-set(old_ids)); days = p['sessions']; months = sorted({d[:7] for d in days})
    ni = pd.Index(natives); di = pd.Index(days); mi = {m:i for i,m in enumerate(months)}; ends = {d[:7]:d for d in days}
    valid = np.zeros((len(days), len(natives)), bool); present = valid.copy()
    closes = np.full((len(months), len(natives)), np.nan)
    oi_pos = ni.get_indexer(old_ids)
    valid[:len(old_days), oi_pos] = np.load(check(old['valid']), allow_pickle=False)
    present[:len(old_days), oi_pos] = np.load(check(old['present']), allow_pickle=False)
    closes[:len(oi['months']), oi_pos] = np.load(check(old['month_closes']), allow_pickle=False)
    extra_days = [d for d in days if d >= '2024-01-01']; ei = pd.Index(extra_days)
    ohlc = np.full((len(extra_days), len(natives), 8), np.nan); qmarks = np.zeros((len(extra_days),len(natives),2), bool)
    seen = np.zeros_like(valid); returned = 0
    for t in quote_blocks(p, natives, new_ids):
        d = di.get_indexer(t.trading_date); n = ni.get_indexer(t.intrinio_security_id)
        ok = (d >= 0) & (n >= 0); t = t.loc[ok].reset_index(drop=True); d,n = d[ok],n[ok]
        need(not seen[d,n].any() and len(set(zip(d,n))) == len(d), 'Duplicate native/session')
        seen[d,n] = True; returned += len(t)
        v = t[list(OHLC)].to_numpy(float)
        accepted = (t.frequency.eq('daily') & t.intraperiod.eq(False) & t.price_quality_status.isin(['accepted','ok'])).to_numpy()
        bounds = (t.raw_ohlc_bounds_valid.eq(True) & t.adjusted_ohlc_bounds_valid.eq(True)).to_numpy()
        good = accepted & bounds & np.isfinite(v).all(axis=1) & (v>0).all(axis=1)
        for offset in (0,4):
            op,hi,lo,cl = v[:,offset:offset+4].T
            good &= (hi>=np.maximum(op,cl)) & (lo<=np.minimum(op,cl)) & (hi>=lo)
        valid[d,n] = good; present[d,n] = True
        end = np.array([date == ends[date[:7]] for date in t.trading_date]); m = np.array([mi[x[:7]] for x in t.trading_date])
        closes[m[end], n[end]] = np.nan
        endgood = end & accepted & np.isfinite(v[:,-1]) & (v[:,-1]>0)
        closes[m[endgood], n[endgood]] = v[endgood,-1]
        e = ei.get_indexer(t.trading_date); fresh = e>=0; ohlc[e[fresh],n[fresh]] = v[fresh]
        for k,cols in ((0,[0,4]),(1,[3,7])):
            marks = accepted & bounds & np.isfinite(v[:,cols]).all(axis=1) & (v[:,cols]>0).all(axis=1)
            qmarks[e[fresh],n[fresh],k] = marks[fresh]
    return dict(valid=valid, present=present, month_closes=closes, extension_ohlc=ohlc, extension_quote_validity=qmarks), \
           dict(sessions=days, months=months, natives=natives, extension_sessions=extra_days,
                ohlc_columns=list(OHLC), new_native_ids=new_ids, returned_source_rows=returned)


def build_panel(cohorts, arrays, index, years):
    """Past-only eligibility/features, strict monthly labels in a separate step."""
    days = np.asarray(index['sessions']); ni = {n:i for i,n in enumerate(index['natives'])}
    mi = {m:i for i,m in enumerate(index['months'])}; ends = {d[:7]:d for d in index['sessions']}
    starts = {m:next(d for d in index['sessions'] if d.startswith(m)) for m in index['months']}
    valid, closes = arrays['valid'], arrays['month_closes']
    invalid = np.vstack([np.zeros((1,valid.shape[1]),np.int32),np.cumsum(~valid,axis=0,dtype=np.int32)])
    blocks = []; coverage = []
    for signal, original in cohorts.groupby('signal_date',sort=True):
        need(signal == ends[signal[:7]], 'Signal is not exact exchange month end')
        b = original.copy(); outcome = shift(signal[:7],1)
        lower = (pd.Timestamp(signal)-pd.DateOffset(years=years)).strftime('%Y-%m-%d')
        lo,hi = np.searchsorted(days,[lower,signal],side='right'); expected = int(hi-lo)
        counts = []; raw = np.full((len(b),5),np.nan)
        for j,r in enumerate(b.to_dict('records')):
            n = ni.get(r['intrinio_security_id']); count = expected-int(invalid[hi,n]-invalid[lo,n]) if n is not None else 0
            counts.append(count)
            if not r['identity_eligible'] or count != expected: continue
            for k,h in enumerate((1,3,6,9,12)):
                required = [shift(signal[:7],-1),signal[:7]] if h==1 else [shift(signal[:7],q) for q in range(-h,0)]
                vals = [closes[mi[m],n] for m in required]
                if np.isfinite(vals).all() and min(vals)>0:
                    raw[j,k] = math.prod(vals[z+1]/vals[z] for z in range(len(vals)-1))-1
        complete = np.array(counts)==expected; admitted = b.identity_eligible.to_numpy(bool)
        common = admitted & complete & np.isfinite(raw).all(axis=1); X = np.full((len(b),15),np.nan)
        if common.sum()>=2:
            pop=raw[common]; mean=np.array([math.fsum(v)/len(v) for v in pop.T])
            sd=np.array([math.sqrt(math.fsum((a-m)**2 for a in v)/(len(v)-1)) for v,m in zip(pop.T,mean)])
            if np.isfinite(sd).all() and min(sd)>0:
                X[common,:5]=(pop-mean)/sd;X[common,5:10]=mean;X[common,10:]=sd
        eligible = common & np.isfinite(X).all(axis=1)
        b['history_years']=years;b['history_start_exclusive']=lower;b['history_end']=signal
        b['expected_history_sessions']=expected;b['valid_history_sessions']=counts;b['history_complete']=complete
        b['prediction_eligible']=eligible;b['prediction_status']=np.where(eligible,'eligible',np.where(~admitted,'identity_blocked',np.where(~complete,'incomplete_past_daily_history','paper_input_or_population_unavailable')))
        b['outcome_month']=outcome;b['outcome_start']=starts[outcome];b['outcome_end']=ends[outcome]
        for k,name in enumerate(FEATURES):b[name]=X[:,k]
        # Forward outcomes are never used above this line.
        ret=np.full(len(b),np.nan);status=np.full(len(b),'not_prediction_eligible',object);miss=[]
        a,z=np.searchsorted(days,[starts[outcome],ends[outcome]])
        for j,r in enumerate(b.to_dict('records')):
            n=ni.get(r['intrinio_security_id']);miss.append(int(z-a+1-valid[a:z+1,n].sum()) if n is not None else int(z-a+1))
            if not eligible[j]:continue
            x,y=closes[mi[signal[:7]],n],closes[mi[outcome],n]
            if not math.isfinite(x) or x<=0:status[j]='missing_or_invalid_start'
            elif not math.isfinite(y) or y<=0:status[j]='missing_or_invalid_end'
            else:ret[j]=y/x-1;status[j]='ok'
        b['label_return']=ret;b['return_status']=status;b['target_class']=deciles(ret);b['target_available']=b.target_class.notna()
        b['label_available_at']=ends[outcome];b['target_entry_date']=signal;b['target_exit_date']=ends[outcome]
        b['outcome_invalid_daily_sessions']=miss
        blocks.append(b)
        coverage.append(dict(history_years=years,signal_date=signal,outcome_month=outcome,original_members=len(b),
            identity_admitted=int(admitted.sum()),complete_history=int((admitted&complete).sum()),
            prediction_eligible=int(eligible.sum()),labelled=int(b.target_available.sum()),missing_targets=int((eligible&~b.target_available.to_numpy()).sum()),
            eligible_with_incomplete_outcome_daily=int((eligible&(np.array(miss)>0)).sum())))
    return pd.concat(blocks,ignore_index=True),pd.DataFrame(coverage)


def payoff_grid(panel,signal,years):
    months=[shift(signal[:7],i) for i in range(-years*12+1,1)]; cells=[]
    historical=panel.loc[panel.outcome_month.isin(months)&panel.label_available_at.le(signal)]
    for m in months:
        g=historical.loc[historical.outcome_month.eq(m)]
        for k in range(1,11):
            v=g.loc[g.target_class.eq(k),'label_return'].to_numpy(float)
            cells.append(dict(signal_date=signal,outcome_month=m,class_id=k,count=len(v),mean_return=math.fsum(v)/len(v) if len(v) else None))
    mu=[math.fsum(c['mean_return'] for c in cells if c['class_id']==k)/(12*years) for k in range(1,11)] if all(c['count'] for c in cells) else None
    return mu,cells


def prepare(plan_path):
    import pyarrow.parquet as pq
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);out=root/'prepared'
    need(not out.exists(),'Preserve consumed input attempt');out.mkdir(parents=True)
    save(out/'STARTED.json',dict(plan_pin=pin(plan_path),maximum_fits=0,maximum_backtests=0))
    start=time.monotonic()
    def guard():
        need(time.monotonic()-start<cfg['budgets']['phase_seconds'],'Input phase time ceiling')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<cfg['budgets']['peak_rss_bytes'],'Input RSS ceiling')
    cohorts,exclusions=read_metadata(p);guard();arrays,index=ingest(p,cohorts);guard();artifacts={}
    artifacts['cohorts']=table_save(out/'cohorts.parquet',cohorts)
    artifacts['membership_exclusions']=table_save(out/'membership_exclusions.parquet',exclusions)
    artifacts['index']=save(out/'index.json',index)
    for name,value in arrays.items():
        path=out/(name+'.npy')
        with path.open('xb') as f:np.save(f,value,allow_pickle=False)
        artifacts[name]=pin(path)
    annual=[];summary=[]
    for years in (5,10):
        panel,coverage=build_panel(cohorts,arrays,index,years);guard()
        artifacts[f'panel{years}']=table_save(out/f'panel{years}.parquet',panel)
        artifacts[f'coverage{years}']=table_save(out/f'coverage{years}.parquet',coverage)
        past=pq.read_table(check(p['parent_inputs'][f'panel{years}'])).to_pandas()
        need(past.outcome_month.max()=='2023-12' and len(past)==152307,'Old learning history changed')
        combined=pd.concat([past,panel],ignore_index=True);payoffs={};grids=[]
        need(not combined.duplicated(list(KEYS)).any(),'Old/new panel overlap')
        for signal in p['signals']:
            mu,cells=payoff_grid(combined,signal,years);payoffs[signal]=mu;grids.extend(cells)
        artifacts[f'payoffs{years}']=save(out/f'payoffs{years}.json',payoffs)
        artifacts[f'payoff_grid{years}']=table_save(out/f'payoff_grid{years}.parquet',pd.DataFrame(grids))
        for year in (2024,2025,2026):
            train=combined.loc[combined.outcome_month.between(f'{year-years}-01',f'{year-1}-12')&combined.label_available_at.le(f'{year-1}-12-31')&combined.prediction_eligible&combined.target_available]
            future=panel.loc[panel.outcome_month.str.startswith(str(year))]
            annual.append(dict(history_years=years,year=year,train_start=f'{year-years}-01',train_end=f'{year-1}-12',training_rows=len(train),
                training_months=int(train.outcome_month.nunique()),training_classes=int(train.target_class.nunique()),prediction_months=int(future.outcome_month.nunique()),
                prediction_eligible=int(future.prediction_eligible.sum()),labelled=int(future.target_available.sum()),missing_targets=int((future.prediction_eligible&~future.target_available).sum())))
        summary.append(dict(history_years=years,original_members=len(panel),identity_admitted=int(panel.identity_eligible.sum()),
            prediction_eligible=int(panel.prediction_eligible.sum()),labelled=int(panel.target_available.sum()),missing_targets=int((panel.prediction_eligible&~panel.target_available).sum()),
            complete_RET_vectors=sum(v is not None for v in payoffs.values()),signal_months=len(payoffs)))
    artifacts['annual_readiness']=table_save(out/'annual_readiness.parquet',pd.DataFrame(annual))
    blocked=cohorts.loc[~cohorts.identity_eligible].groupby(['identity_status','identity_eligibility_reason']).size().reset_index(name='member_months')
    artifacts['identity_blockers']=table_save(out/'identity_blockers.parquet',blocked)
    artifacts['missing_labels']=table_save(out/'missing_labels.parquet',pd.concat([
        pq.read_table(artifacts[f'panel{y}']['path']).to_pandas().loc[lambda t:t.prediction_eligible&~t.target_available,
          ['history_years',*KEYS,'outcome_month','intrinio_security_id','membership_tickers','return_status','outcome_invalid_daily_sessions']] for y in (5,10)],ignore_index=True))
    guard();need(sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<cfg['budgets']['output_bytes'],'Output ceiling')
    return save(out/'receipt.json',dict(status='INPUTS_BUILT_PENDING_INDEPENDENT_CHECK',plan_pin=pin(plan_path),artifacts=artifacts,summary=summary,
        source_projection_bytes_reserved=p['projected_bytes_per_pass'],new_fits=0,new_backtests=0,active_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,financial_end='2026-04-30',
        execution_action_coverage_certified=False,source_identity_certified=False,old_runs_preserved=True))
