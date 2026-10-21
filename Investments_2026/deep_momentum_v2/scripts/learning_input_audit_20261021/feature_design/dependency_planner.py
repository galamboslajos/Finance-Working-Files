"""Pure structural dependency envelopes; consumes calendar/specs, never prices."""
from collections import defaultdict


def need(value,message):
    if not value: raise ValueError(message)


def shift_month(month,amount):
    y,m=map(int,month.split('-')); n=y*12+m-1+amount
    return f'{n//12:04d}-{n%12+1:02d}'


def month_ends(sessions):
    out={}
    for day in sessions: out[day[:7]]=day
    return out


def exact_span_dependencies(sessions,windows,patch_dates,spans):
    need(tuple(sessions)==tuple(sorted(set(sessions))),'Exact unique ordered calendar required')
    positions={d:i for i,d in enumerate(sessions)}
    need(all(d in positions for d in patch_dates),'Patch date outside explicit calendar')
    out=[]
    for window in windows:
        signal=window['signal_date']; need(signal in positions,'Signal outside calendar')
        i=positions[signal]; fields=defaultdict(list)
        for day in patch_dates:
            lag=i-positions[day]
            for field,lo,hi in spans:
                need(isinstance(lo,int) and 0<=lo<=hi,'Past-only inclusive lag span required')
                if lo<=lag<=hi: fields[field].append(day)
        if fields:
            out.append(dict(window,patched_field_dates={f:sorted(set(ds)) for f,ds in sorted(fields.items())}))
    return out


def paper_candidates(sessions,windows,patch_dates):
    ends=month_ends(sessions); patched=set(patch_dates); out=[]
    for window in windows:
        signal=window['signal_date']; t=signal[:7]; hits={}
        for horizon in (1,3,6,9,12):
            months=[shift_month(t,-1),t] if horizon==1 else [shift_month(t,j) for j in range(-horizon,0)]
            need(all(x in ends for x in months),'Paper anchor calendar incomplete')
            touched=sorted(patched.intersection(ends[x] for x in months))
            if touched: hits[str(horizon)]=touched
        if hits:
            out.append(dict(window,raw_momentum_horizons=hits,potential_common_population_change=True,
                potential_affected_paper_features=[f'{prefix}{h}' for prefix in ('zMOM','meanMOM','stdMOM') for h in (1,3,6,9,12)]))
    return out


def context_spans(spec,daily_definitions):
    name=spec['name']; role=spec['role']
    if role=='cross_sectional': return daily_definitions[spec['required_inputs'][0]]['_spans']
    n=spec['lookback_sessions']
    return tuple((field,0,n-1) for field in spec['required_inputs'] if field in ('adj_close','close','volume'))


def classify_patch_fields(dependency,field_status):
    blocked=[field for field in dependency['patched_field_dates'] if field_status.get(field) in ('missing','all_null')]
    unknown=[field for field in dependency['patched_field_dates'] if field_status.get(field) not in ('present_all_nonnull','missing','all_null')]
    return dict(blocked_referenced_patch_fields=blocked,unknown_referenced_patch_fields=unknown,
        execution_patch_supplies_referenced_fields=not blocked and not unknown,
        rest_of_feature_window_verified=False)


def fold_map(windows,signals,folds):
    selected={w['outcome_month'] for w in windows if w['signal_date'] in signals}
    def within(start,end): return sorted(m for m in selected if start[:7]<=m<=end[:7])
    out=[]
    for fold in folds:
        year=fold['outer_outcome_year']
        out.append(dict(outer_outcome_year=year,
            outer_training_hit_months=within(fold['train_outcome_start'],fold['train_outcome_end']),
            outer_prediction_hit_months=sorted(m for m in selected if m[:4]==str(year)),
            inner_splits=[dict(split_id=s['split_id'],train_hit_months=within(s['train_outcome_start'],s['train_outcome_end']),
                validation_hit_months=within(s['validation_outcome_start'],s['validation_outcome_end'])) for s in fold['inner_splits']]))
    return out
