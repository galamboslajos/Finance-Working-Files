"""Metadata/footer-only feature impact envelope; never decodes financial columns."""
from pathlib import Path
import ast,collections,hashlib,importlib.metadata,json,sys
from dependency_planner import exact_span_dependencies,paper_candidates,context_spans,classify_patch_fields,fold_map

R=Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
S=Path('/private/tmp/dmv2-learning-impact-20261021/feature_design')
META_LIMIT=16*1024**2
reads={}

def need(ok,msg):
    if not ok: raise ValueError(msg)

def read(path):
    path=Path(path).resolve(); size=path.stat().st_size
    reads[str(path)]=size; need(size<2*1024**2 and sum(reads.values())<=META_LIMIT,'Metadata source read ceiling')
    return path.read_text()

def obj(path): return json.loads(read(path))

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pin(path):
    p=Path(path).resolve(); return dict(path=str(p.relative_to(R)) if p.is_relative_to(R) else str(p),sha256=digest(p),size_bytes=p.stat().st_size)

def auth(item):
    p=R/item['path']; need(p.stat().st_size==item['size_bytes'] and digest(p)==item['sha256'],'Pinned metadata changed'); return p

def specs_from_source(path,constants,functions):
    tree=ast.parse(read(path)); env={}
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            names=[x.id for x in node.targets if isinstance(x,ast.Name)] if isinstance(node,ast.Assign) else [node.target.id]
            for name in names:
                if name in constants:
                    value=node.value
                    if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='frozenset':
                        env[name]=frozenset(ast.literal_eval(value.args[0]))
                    else: env[name]=ast.literal_eval(value)
    keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions]
    need({n.name for n in keep}==set(functions),'Spec function changed')
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),env)
    return env


def footer(path):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path); rows=pf.metadata.num_rows; columns={}
    for field in pf.schema_arrow:
        nulls=[]
        for rg in range(pf.metadata.num_row_groups):
            meta=pf.metadata.row_group(rg).column(pf.schema_arrow.get_field_index(field.name)); st=meta.statistics
            nulls.append(None if st is None or not st.has_null_count else st.null_count)
        known=all(n is not None for n in nulls); count=sum(nulls) if known else None
        status=('present_all_nonnull' if count==0 else 'all_null' if count==rows else 'present_partial_null' if known else 'null_statistics_unavailable')
        columns[field.name]=dict(type=str(field.type),null_count=count,status=status)
    return dict(path=str(path.relative_to(R)),physical_bytes=path.stat().st_size,rows=rows,
        row_groups=pf.metadata.num_row_groups,footer_serialized_bytes=pf.metadata.serialized_size,columns=columns,
        financial_column_chunks_decoded=False,price_min_max_read=False)


def main():
    output=S/'feature_dependency_audit.json'; need(not output.exists(),'Preserve prior private audit')
    parent_path=R/'outputs/ret_backtest/2020-2023-20261020-v1/build_identity.json'; parent=obj(parent_path)
    need(digest(parent_path)=='4c5fba9f96339052083ba057eb363136001fa458afe86c11e28a492aaaadf3f7','Original RET identity changed')
    windows=[]; state_pins=[]
    for annual in parent['annual_sources']:
        p=auth(annual['target_state_pin']); x=obj(p); windows+=x['provenance']['month_windows']; state_pins.append(annual['target_state_pin'])
    windows=sorted(windows,key=lambda x:x['outcome_month'])
    need(len(windows)==168 and len({w['signal_date'] for w in windows})==168 and windows[0]['signal_date']=='2009-12-31' and windows[-1]['signal_date']=='2023-11-30','Exact supervised signal clock changed')
    need(importlib.metadata.version('exchange-calendars')=='4.13.2','Frozen XNYS metadata package changed')
    import exchange_calendars as xc
    calendar=xc.get_calendar('XNYS',start='2007-01-01',end='2023-12-31')
    sessions=[d.strftime('%Y-%m-%d') for d in calendar.sessions]
    ends={}
    for day in sessions: ends[day[:7]]=day
    need(all(ends[w['signal_date'][:7]]==w['signal_date'] and ends[w['outcome_month']]==w['outcome_end'] for w in windows),'Independent XNYS and saved signal clock disagree')
    daily_env=specs_from_source(R/'src/dmv2/daily_features.py',('ENDPOINTS','WINDOWS','TREND_WINDOWS','OSC_WINDOWS','SKIPS'),('_definitions',))
    daily=daily_env['_definitions'](); by_name={x['name']:x for x in daily}
    ctx_env=specs_from_source(R/'src/dmv2/context_features.py',('SOURCE_COMMIT','MARKET_HORIZONS','RELATIVE_WINDOWS','MARKET_METRICS','STRUCTURAL_CONSTANTS'),('balanced_cross_sectional_bases','_base_lookback','context_feature_specs'))
    context=ctx_env['context_feature_specs']()
    paper_env=specs_from_source(R/'src/dmv2/paper_features.py',('HORIZONS',),('paper_feature_specs',))
    paper=paper_env['paper_feature_specs']('signal_qualified_common_momentum_v1')
    definitions=[*daily,*context,*paper]
    role=obj(R/'configs/context_panel_roles.json'); dict_path=R/role['primary']['directory']/'feature_dictionary.json'
    dictionary=obj(dict_path); dictionary_names={x['name'] for x in dictionary}
    need(len(dictionary)==len(definitions)==728 and dictionary_names=={x['name'] for x in definitions},'Registered dictionary differs from workers')
    actual={x['name']:x for x in dictionary}
    for spec in definitions:
        for field in ('formula','required_inputs','lookback_sessions','lookback_months'):
            if field in spec: need(json.dumps(spec[field])==json.dumps(actual[spec['name']][field]),'Frozen dictionary definition mismatch '+spec['name']+' '+field)
    need(sum(s.get('model_eligible',True) for s in dictionary)==715,'Fixed predictor allowlist count changed')
    manifest_path=R/'data/derived/ret_repairs/20261020-v1/repair_manifest.json'; manifest=obj(manifest_path)
    proofs={p['intrinio_security_id']:p for p in manifest['quote_unit_proofs']}
    field_footers={}
    for proof in proofs.values():
        for key in ('source_quotes_pin','target_controls_pin','paired_source_pin'):
            if proof.get(key):
                path=R/proof[key]['path']; need(path.stat().st_size==proof[key]['size_bytes'],'Quote footer file size changed')
                if str(path) not in field_footers: field_footers[str(path)]=footer(path)
    qpath=R/manifest['quotes_pin']['path']; field_footers[str(qpath)]=footer(qpath)
    need(sum(x['footer_serialized_bytes'] for x in field_footers.values())<1024**2,'Footer metadata ceiling')
    case_defs=[('RCM','sec_zMM1vd'),('DISCA_WBD','sec_Xnq2jn')]
    tuning_path=R/'configs/rf_expanded_tuning_20261020.json'; tuning=obj(tuning_path); folds=tuning['chronology']['folds']
    cases=[]
    for label,native in case_defs:
        dates=sorted(row['trading_date'] for row in manifest['quote_amendments'] if row['intrinio_security_id']==native)
        need(len(dates)==(22 if label=='RCM' else 5),'Exact approved repair dates changed')
        proof=proofs[native]; source_footer=field_footers[str(R/proof['source_quotes_pin']['path'])]
        statuses={f:x['status'] for f,x in source_footer['columns'].items()}
        impacts=[]
        for spec in [*daily,*context]:
            spans=spec['_spans'] if spec['role']=='technical' else context_spans(spec,by_name)
            hits=exact_span_dependencies(sessions,windows,dates,spans)
            if hits:
                for hit in hits: hit.update(classify_patch_fields(hit,statuses))
                impacts.append(dict(name=spec['name'],role=spec['role'],model_eligible=actual[spec['name']].get('model_eligible',True),
                    exact_native_spans=[list(s) for s in spans],signals=hits,
                    propagation=('same_native_only' if spec['role']=='technical' else 'conditional_original_union_or_sleeve_and_benchmark_dependents')))
        paper_hits=paper_candidates(sessions,windows,dates)
        candidate_signals=sorted({h['signal_date'] for item in impacts if item['model_eligible'] for h in item['signals']}|{h['signal_date'] for h in paper_hits})
        supplied_signals=sorted({h['signal_date'] for item in impacts if item['model_eligible'] for h in item['signals'] if h['execution_patch_supplies_referenced_fields']}|{h['signal_date'] for h in paper_hits})
        family={r:sorted({h['signal_date'] for i in impacts if i['role']==r and i['model_eligible'] for h in i['signals']}) for r in sorted({i['role'] for i in impacts})}
        cases.append(dict(case=label,native=native,approved_patch_dates=dates,input_field_status=statuses,
            technical_and_context_dependencies=impacts,paper_common_population_dependencies=paper_hits,
            candidate_signal_dates=candidate_signals,patch_supplied_dependency_signal_dates=supplied_signals,
            candidate_signals_by_role=family,
            candidate_outer_and_inner_fold_map=fold_map(windows,candidate_signals,folds),
            patch_supplied_outer_and_inner_fold_map=fold_map(windows,supplied_signals,folds),
            actual_feature_values_or_membership_checked=False,
            qualification='STRUCTURAL_UPPER_BOUND_REQUIRES_DATED_MEMBERSHIP_ROUTE_AND_REST_OF_HISTORY'))
    source_cfg=obj(R/'configs/data_sources.local.json'); intrinio=next(s for s in source_cfg['sources'] if s['key']=='intrinio')
    intrinio_manifest_path=R/intrinio['manifest']; intrinio_manifest=obj(intrinio_manifest_path)
    table=next(t for t in intrinio_manifest['schema']['tables'] if t['name']=='intrinio_daily_prices')
    source_fields=[c for c in table['columns'] if c['name'] in ('open','high','low','close','adj_open','adj_high','adj_low','adj_close','volume','dividend','split_ratio','factor')]
    route=obj(R/'data/derived/ret_repairs/20261020-v1/related_provider_identity_metadata.json')
    after=obj(R/'data/derived/ret_repairs/20261020-v1/rcm_after_control_metadata.json')
    route_interval=dict(provider_master_rows=[r for r in route['rows'] if r['intrinio_security_id'] in ('sec_yKbEk4','sec_zMM1vd')],
        approved_execution_remap=['2021-11-01','2021-12-01'],later_key_probe_request=after['provenance']['request'],
        later_key_counts={k:len(v) for k,v in after['dates'].items()},later_key_overlap_count=len(after['common_dates']),
        full_2010_to_2022_numeric_or_identity_interval_verified=False,
        distinction='Provider first/last dates and later keys are inventory evidence, not a verified full daily history or authority to extend the22-date remap.')
    report=dict(schema='dmv2-learning-feature-static-dependency-audit/v1',status='COMPLETE_METADATA_AND_DEFINITIONS_ONLY',
        source_signal_clock=dict(windows=windows,target_state_pins=state_pins,independent_calendar_version='4.13.2',
            independent_calendar_name='XNYS',independent_calendar_scope=[sessions[0],sessions[-1]],exact_calendar_sessions=sessions),
        definitions=dict(registered_count=728,predictor_count=715,daily_count=298,paper_count=15,context_count=415,
            max_native_lookback_sessions=max(s['lookback_sessions'] for s in daily),
            max_context_lookback_sessions=max(s['lookback_sessions'] for s in context),
            dictionary_pin=pin(dict_path),daily_native_specs=daily,context_specs=context,paper_specs=paper),
        cases=cases,repair_input_footer_readiness=list(field_footers.values()),
        upstream_daily_field_schema_provenance=dict(manifest_pin=pin(intrinio_manifest_path),declared_source_fields=source_fields,
            per_date_volume_validity_checked=False,raw_volume_never_inferred_from_prices=True),
        RCM_route_interval_readiness=route_interval,
        other_approved_quote_patch=dict(native='execution_cusip:48214T305',rows=13,
            automatically_applied_to_original_feature_native=False,
            reason='New successor execution identity is not an approved canonical feature-route repair; requires separately dated identity and cohort workflow.'),
        claims=dict(financial_column_chunks_decoded=False,financial_feature_values_calculated=False,models_or_predictions_loaded=False,
            production_artifacts_changed=False,new_feature_builds=0,actual_changed_feature_count=None,
            PIT='Every dependency date is at or before its signal; following-month targets remain separate. Shared past lookbacks across signals are causal reuse, not target leakage.',
            context='A contribution/availability change can alter union/sleeve ranks,MAD and shared market state; a changed daily benchmark can propagate through complete relative windows. The80% gate uses original denominators and exhaustive admitted requests.',
            normalization='Common-five Paper15 population changes can alter all15 inputs for all qualifying members at an affected signal, even when only one raw momentum horizon directly uses the repaired endpoint.',
            training='Any actual changed supervised feature/eligibility can affect train-only medians,drop schema,missing indicators and all dependent fitted estimates; a candidate fold hit is not proof of numerical prediction change.',
            route='Do not extend provisional RCM execution mapping from provider first/last inventory alone; unchanged historical routes outside22dates may require a separate fuller repair.'),
        metadata_read_scope=dict(unique_text_files=len(reads),unique_text_bytes=sum(reads.values()),footer_count=len(field_footers),
            footer_serialized_bytes=sum(x['footer_serialized_bytes'] for x in field_footers.values()),ceiling_text_bytes=META_LIMIT,
            footer_stats_used='schema,rows,types,null counts only; no price min/max or financial data chunks'),
        source_and_worker_pins=[pin(p) for p in [parent_path,manifest_path,tuning_path,R/'src/dmv2/daily_features.py',R/'src/dmv2/context_features.py',R/'src/dmv2/paper_features.py',R/'src/dmv2/full_features.py',R/'src/dmv2/features.py']])
    raw=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n'; need(len(raw.encode())<16*1024**2,'Private output bound')
    output.write_text(raw)
    lines=['# Feature dependency and readiness audit — metadata only','',
        'No feature values were recalculated. The exact168 supervised months use signals2009-12-31 through2023-11-30. The728 dictionary definitions and715 eligible names match the actual workers. Maximum individual dependency is503 exchange sessions, driven by a252-session abnormal-volume average.','',
        'A candidate date is a structural upper bound. Dated membership, identity continuity, original-cohort gates, all other required observations and numerical nondegeneracy have not been checked here.','']
    for case in cases:
        lines += ['## '+case['case'],f"Approved dates: {case['approved_patch_dates'][0]} through {case['approved_patch_dates'][-1]} ({len(case['approved_patch_dates'])} sessions).",'',
            f"Candidate signals: {', '.join(case['candidate_signal_dates'])}.",
            f"Signals with at least one dependency whose referenced patch fields are supplied: {', '.join(case['patch_supplied_dependency_signal_dates'])}.",
            f"Paper15 common-population candidate signals: {', '.join(h['signal_date'] for h in case['paper_common_population_dependencies']) or 'none'}.",'']
        for f in case['candidate_outer_and_inner_fold_map']:
            lines.append(f"Outer {f['outer_outcome_year']}: feature-input training months {', '.join(f['outer_training_hit_months']) or 'none'}; prediction months {', '.join(f['outer_prediction_hit_months']) or 'none'}.")
            for s in f['inner_splits']:
                if s['train_hit_months'] or s['validation_hit_months']: lines.append(f"- {s['split_id']}: inner train {', '.join(s['train_hit_months']) or 'none'}; inner validation {', '.join(s['validation_hit_months']) or 'none'}.")
        lines += ['']
    lines += ['## Readiness and limits','',
        'The execution supplement has OHLC and adjusted OHLC but no volume. Footer null counts are recorded per proof source; field presence does not establish valid historical values. A feature can reference a repaired close only at its far boundary without referencing the repaired dates’ missing volume, so the JSON checks exact field/date spans instead of declaring every volume-family feature unusable.',
        'RCM provider first/last dates are2007-02-23–2022-06-21(old) and2017-03-15–2024-11-18(new). The separate post-gap key probe returns138 old-route dates and zero overlap; that proves neither full2010history nor quote validity. The remap remains confined to22approved sessions pending a fuller dated route/history decision.',
        'Paper15 can ripple through its common-five population; daily/context ranks,original-denominator80%gates and relative daily benchmarks can ripple to other cohort members. The parent full-cohort audit must decide actual membership and observed input impacts before any rebuild or refit.',
        'The frozen four outer windows and twelve time-ordered inner splits are mapped by the outcome month belonging to each affected feature signal. No labels,RET payoffs,features,preprocessors,models,selections or backtests were changed.']
    (S/'FEATURE_DEPENDENCY_AUDIT.md').write_text('\n\n'.join(lines)+'\n')
    (S/'audit_receipt.json').write_text(json.dumps(dict(status=report['status'],artifacts=[pin(output),pin(S/'FEATURE_DEPENDENCY_AUDIT.md')],metadata_read_scope=report['metadata_read_scope']),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status=report['status'],output=str(output),bytes=len(raw.encode()),metadata=report['metadata_read_scope'],
        summary=[dict(case=c['case'],candidate_signals=c['candidate_signal_dates'],paper_signals=[h['signal_date'] for h in c['paper_common_population_dependencies']]) for c in cases]),indent=2))

if __name__=='__main__': main()
