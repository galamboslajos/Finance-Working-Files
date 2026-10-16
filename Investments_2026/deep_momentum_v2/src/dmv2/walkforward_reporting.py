"""Receipt-bound, bounded saved walk-forward reports; no numerical workflow.

This reader never opens feature panels, execution quotes, prepared matrices or
serialized models. Optional source verification hashes bytes without decoding
financial values. Independent receipts bind the reports to completed audits.
"""
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .universe import DataContractError

ARMS = ('rf_paper15','xgb_paper15','rf_extended','xgb_extended')
REFERENCES = ('uniform','train_prior')
YEARS = (2020,2021,2022,2023)
BASKETS = ('class_1_10','class_2_9')
VARIANTS = ('unstopped','asset_stop_10pct')
RECEIPTS = {'features','models','backtest','feature_verification','model_verification','backtest_verification'}
MAX_JSON_BYTES = 16 * 1024**2
MAX_REPORT_BYTES = 16 * 1024**2


def _need(ok,message):
    if not ok: raise DataContractError(message)


def _hash(path):
    state=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):state.update(block)
    return state.hexdigest()


def _within(base,path):
    base=Path(base).resolve(); path=Path(path).resolve()
    _need(path.is_relative_to(base),'Saved report path leaves its declared workspace')
    return path


def _json(path):
    path=Path(path)
    _need(path.stat().st_size<=MAX_JSON_BYTES,'Saved report JSON exceeds its bounded size')
    return json.loads(path.read_text())


def _receipt(path,expected):
    _need('sha256:'+_hash(path)==expected,'Saved report or independent verification receipt changed: '+str(path))


def _pin(base,item,*,expected_path=None):
    _need(isinstance(item,dict) and {'path','size_bytes','sha256'}<=set(item),'Missing saved report artifact pin')
    path=_within(base,Path(base)/item['path'])
    _need(expected_path is None or path==Path(expected_path).resolve(),'Saved report manifest redirects a fixed path')
    _need(path.stat().st_size==item['size_bytes'] and _hash(path)==item['sha256'].removeprefix('sha256:'),
        'Saved report artifact changed: '+str(path))
    return path


def _pins(base,items):
    for item in (items.values() if isinstance(items,dict) else items):_pin(base,item)


def _pinned_json(base,item,expected_path):
    return _json(_pin(base,item,expected_path=expected_path))


def _report(base,item,relative,rows,*,metadata_guard=None):
    path=_pin(base,item,expected_path=Path(base)/relative)
    _need(item['size_bytes']<=MAX_REPORT_BYTES,'Saved report exceeds its bounded byte ceiling')
    file=pq.ParquetFile(path)
    _need(file.metadata.num_rows==rows,'Saved report lost its complete row grid: '+relative)
    if metadata_guard is not None:metadata_guard(file)
    return file.read().to_pandas()


def _months():
    return [str(month) for month in pd.period_range('2020-01','2023-12',freq='M')]


def _verification_receipts(root,directory,expected,identity,raw,model_phase,backtest_phase,backtest_directory):
    result={}
    for label,filename in (('feature_verification','features.json'),('model_verification','models.json'),
            ('backtest_verification','backtest.json')):
        path=directory/filename; _receipt(path,expected[label]); receipt=_json(path)
        _need(receipt.get('status') in {'PASS','PASSED'} and receipt.get('build_identity_sha256')==identity,
            'Independent verification does not certify this exact completed research run: '+label)
        # Every independent verifier authenticates the exact feature audit. The
        # explicit raw output-audit receipts above also reject changed reports.
        _need(receipt.get('root_audit_sha256')==raw['features'].removeprefix('sha256:'),
            'Independent verification refers to a different feature audit: '+label)
        if label=='model_verification':
            _need(receipt.get('model_identity_sha256')==model_phase
                and receipt.get('models_audit_sha256')==raw['models'].removeprefix('sha256:')
                and receipt.get('models_fitted_by_verifier')==0
                and receipt.get('preprocessors_fitted_by_verifier')==0
                and receipt.get('annual_fits_reconciled')==16
                and receipt.get('stitched_prediction_months')==48
                and receipt.get('financial_2024_plus_decoded') is False,
                'Independent model verification changed fit count, audit or protected scope')
        elif label=='backtest_verification':
            _need(receipt.get('backtest_phase_identity_sha256')==backtest_phase
                and receipt.get('backtest_audit_sha256')==raw['backtest'].removeprefix('sha256:')
                and receipt.get('models_fitted')==0 and receipt.get('preprocessors_fitted')==0
                and receipt.get('source_pins_unchanged') is True
                and receipt.get('financial_2024_plus_read') is False
                and receipt.get('sessions')==1006 and receipt.get('outcome_months')==48
                and receipt.get('scenarios')==64,
                'Independent backtest verification changed audit, continuous horizon or protected scope')
            if 'backtest_directory' in receipt:
                _need((root/receipt['backtest_directory']).resolve()==backtest_directory,
                    'Independent verification refers to another backtest directory')
        else:
            _need(receipt.get('financial_2024_plus_read',False) is False
                and receipt.get('models_fitted',0)==0,'Independent feature verification changed financial/fit scope')
        result[label]=receipt
    return result


def _deep_hash(root,run,identity,feature,models,model,model_phase,backtest,bt,bt_phase):
    """Optional full source-byte authentication; no data/model deserialization."""
    preservation=_json(run/'source_preservation_manifest.json')
    _pins(root,preservation.get('artifacts',preservation))
    for name,sha in identity.get('workers',{}).items():
        _need(Path(name).name==name and _hash(root/'src/dmv2'/name)==sha.removeprefix('sha256:'),
            'Frozen feature worker changed')
    for year,item in feature['years'].items():
        path=_pin(run,item); _pins(path.parent,_json(path)['artifacts'])
    _pins(root,model_phase['annual_source_audit_pins'])
    _pins(models,model['artifacts'])
    for item in model['folds'].values():
        path=_pin(models,item); annual=_json(path)
        _pins(path.parent,annual['artifacts'])
        plan=_json(path.parent/'read_plan.json')
        for source in plan['years'].values():_pins(root,source['source_pins'])
        for arm_item in annual['arms'].values():
            arm_path=_pin(path.parent,arm_item); _pins(arm_path.parent,_json(arm_path)['artifacts'])
    for item in model['arms'].values():
        path=_pin(models,item); _pins(path.parent,_json(path)['artifacts'])
    for phase in (model_phase,bt_phase):
        for name,sha in phase['workers'].items():
            _need(Path(name).name==name and _hash(root/'src/dmv2'/name)==sha.removeprefix('sha256:'),
                'Frozen model/backtest worker changed')
    _pins(root,bt_phase['source_pins']); _pins(backtest,bt['artifacts'])


def read_walkforward_run(root,directory,*,identity_sha256,expected_receipts,
        verification_directory=None,verify_sources=False):
    """Return only bounded saved summaries after six explicit receipt checks.

    Missing saved artifacts raise FileNotFoundError. An incomplete, changed or
    unverified run raises DataContractError; no build or retry occurs. The
    optional verification directory may hold copied receipt files outside the
    run, but their byte hashes and exact research-audit bindings remain required.
    """
    _need(type(verify_sources) is bool and isinstance(expected_receipts,dict)
        and set(expected_receipts)==RECEIPTS
        and all(isinstance(value,str) and re.fullmatch(r'sha256:[0-9a-f]{64}',value) for value in expected_receipts.values())
        and isinstance(identity_sha256,str) and re.fullmatch(r'sha256:[0-9a-f]{64}',identity_sha256),
        'All six final receipts, exact identity and explicit verification scope are required')
    root=Path(root).resolve(); run=_within(root/'outputs/walkforward',Path(directory))
    _need(run.name=='2020-2023-'+identity_sha256[7:23],'Saved walk-forward directory differs from its explicit identity')
    verification=(run/'verification') if verification_directory is None else Path(verification_directory).resolve()
    # An explicitly hash-bound verifier may refer to an additive action-complete
    # backtest folder. No mutable LATEST lookup or original pointer is consulted.
    _receipt(verification/'backtest.json',expected_receipts['backtest_verification'])
    bound_backtest=_json(verification/'backtest.json').get('backtest_directory')
    backtest=run/'backtest' if bound_backtest is None else _within(run,root/bound_backtest)
    for label,path in (('features',run/'audit.json'),('models',run/'models/audit.json'),('backtest',backtest/'audit.json')):
        _receipt(path,expected_receipts[label])
    identity=_json(run/'build_identity.json'); feature=_json(run/'audit.json')
    _need(canonical_sha256(identity)==identity_sha256
        and identity.get('phase')=='walkforward_features_and_targets_2020_2023_v1'
        and feature.get('status')=='COMPLETE' and feature.get('phase')=='walkforward_features_and_targets'
        and feature.get('build_identity_sha256')==identity_sha256
        and set(feature.get('years',{}))=={str(year) for year in YEARS},'Saved four-year feature/target scope changed')
    cfg=identity['configuration']; execution=cfg.get('execution',{})
    _need(cfg.get('protocol_id')=='dmv2-walkforward-2020-2023-v1'
        and cfg.get('test_years')==list(YEARS) and execution.get('maximum_new_model_fits')==16
        and execution.get('maximum_new_preprocessor_fits')==8
        and execution.get('financial_2024_plus_analysis') is False,'Saved rolling outer protocol changed')
    models=run/'models'; model=_json(models/'audit.json')
    _need(model.get('status')=='COMPLETE' and model.get('phase')=='walkforward_models_v1'
        and model.get('build_identity_sha256')==identity_sha256
        and model.get('summary',{}).get('fitted_arms')==16
        and model['summary'].get('preprocessor_fits')==8 and model['summary'].get('months')==48
        and model['summary'].get('outer_years')==list(YEARS)
        and set(model.get('arms',{}))=={*ARMS,*REFERENCES}
        and set(model.get('folds',{}))=={str(year) for year in YEARS},'Saved sixteen-fit model scope changed')
    model_phase=_pinned_json(models,model['artifacts']['build_identity.json'],models/'build_identity.json')
    _need(canonical_sha256(model_phase)==model.get('model_identity_sha256')
        and model_phase.get('parent_identity_sha256')==identity_sha256
        and model_phase.get('configuration_sha256')==canonical_sha256(cfg),'Saved model phase changed')
    bt=_json(backtest/'audit.json')
    _need(bt.get('status')=='COMPLETE' and bt.get('phase')=='walkforward_continuous_backtest_v1'
        and bt.get('build_identity_sha256')==identity_sha256 and bt.get('scenarios')==64
        and bt.get('sessions')==1006 and bt.get('outcome_months')==48
        and type(bt.get('complete_proxy_paths')) is int and type(bt.get('failed_closed_paths')) is int
        and bt['complete_proxy_paths']>=0 and bt['failed_closed_paths']>=0
        and bt['complete_proxy_paths']+bt['failed_closed_paths']==64
        and bt.get('models_fitted') is False and bt.get('preprocessing_fitted') is False
        and bt.get('annual_liquidation') is False and bt.get('financial_2024_plus_read') is False,
        'Saved backtest lacks the complete continuous64-scenario audit grid')
    bt_phase=_pinned_json(backtest,bt['artifacts']['build_identity.json'],backtest/'build_identity.json')
    _need(canonical_sha256(bt_phase)==bt.get('phase_identity_sha256')
        and bt_phase.get('parent_identity_sha256')==identity_sha256
        and bt_phase.get('configuration_sha256')==canonical_sha256(cfg),'Saved backtest phase changed')
    receipts=_verification_receipts(root,verification,expected_receipts,identity_sha256,expected_receipts,
        model['model_identity_sha256'],bt['phase_identity_sha256'],backtest)
    # All independent receipts pass before any report financial values are read.
    annuals=[]
    for year,item in feature['years'].items():
        annual=_pinned_json(run,item,run/'years'/year/'audit.json')
        _need(annual.get('status')=='COMPLETE' and annual.get('year')==int(year)
            and annual.get('build_identity_sha256')==identity_sha256,'Saved annual feature receipt changed')
        annuals.append({'year':int(year),**annual.get('summary',{})})
    comparison=_report(models,model['artifacts']['comparison'],'reports/comparison.parquet',6)
    _need('arm_id' in comparison and not comparison.arm_id.duplicated().any()
        and set(comparison.arm_id)=={*ARMS,*REFERENCES},'Saved six-row model comparison grid changed')
    class_parts=[]
    for arm in (*ARMS,*REFERENCES):
        directory=models/('arms' if arm in ARMS else 'refs')/arm
        state=_pinned_json(models,model['arms'][arm],directory/'audit.json')
        _need(state.get('status')=='COMPLETE' and state.get('build_identity_sha256')==identity_sha256
            and state.get('model_identity_sha256')==model['model_identity_sha256']
            and state.get('metadata',{}).get('new_fits_during_stitching')==0,'Saved aggregate model arm changed')
        item=state.get('artifacts',{}).get('mode_class_performance')
        if item is not None:
            frame=_report(directory,item,'reports/mode_class_performance.parquet',490)
            _need(not frame.duplicated(['period','class_id']).any()
                and set(frame.period)=={'all',*_months()} and set(frame.class_id)==set(range(1,11)),
                'Saved class precision/recall report lost its48-month grid')
            selected=frame.loc[frame.period.eq('all')&frame.class_id.isin([1,2,9,10])].copy()
            _need(len(selected)==4,'Saved pooled extreme/second-extreme class report changed')
            class_parts.append(selected.assign(arm_id=arm))
    reports={}
    for name,rows in (('comparison',64),('yearly',256),('selection_coverage',384)):
        reports[name]=_report(backtest,bt['artifacts'][name+'.parquet'],name+'.parquet',rows)
    bt_comparison=reports['comparison']; grid=['arm','basket','variant','cost_bps']
    _need(not bt_comparison.duplicated(grid).any()
        and set(map(tuple,bt_comparison[grid].values))==set(itertools.product(ARMS,BASKETS,VARIANTS,[0,5,10,25]))
        and bt_comparison.expected_sessions.eq(1006).all()
        and int(bt_comparison.complete_proxy_path.sum())==bt['complete_proxy_paths'],
        'Saved backtest scenario grid changed')
    yearly=reports['yearly']
    _need(not yearly.duplicated([*grid,'year']).any()
        and set(map(tuple,yearly[[*grid,'year']].values))==set(itertools.product(ARMS,BASKETS,VARIANTS,[0,5,10,25],YEARS)),
        'Saved annual breakdown lost continuous scenario/year coverage')
    coverage=reports['selection_coverage']
    _need(not coverage.duplicated(['arm','basket','signal_date']).any()
        and set(coverage.arm)==set(ARMS) and set(coverage.basket)==set(BASKETS)
        and coverage.groupby(['arm','basket']).size().eq(48).all()
        and coverage.signal_date.map(lambda day:isinstance(day,str) and '2019-12-01'<=day<='2023-11-30').all(),
        'Saved selection coverage lost its original48-month grid')
    breadth=np.minimum(np.minimum(10,coverage.long_candidates),coverage.short_candidates)
    _need((breadth>=0).all() and coverage.selected_names_per_side.eq(breadth).all()
        and coverage.long_selected.eq(breadth).all() and coverage.short_selected.eq(breadth).all()
        and np.allclose(coverage.loc[breadth>0,'effective_per_name_abs_weight'],.5/breadth[breadth>0],rtol=0,atol=1e-14)
        and coverage.loc[breadth==0,'effective_per_name_abs_weight'].isna().all(),
        'Saved breadth changed balanced min(10,long,short) or50/50 weights')
    monthly_parts=[]
    def month_guard(file):
        _need(set(file.schema_arrow.names)=={'outcome_month','nav','return'},'Monthly report contains unexpected source fields')
        meta=file.read(columns=['outcome_month']).to_pandas()
        _need(meta.outcome_month.tolist()==_months(),'Saved monthly return table exceeds its48-month bounds')
    for arm,basket,variant in itertools.product(ARMS,BASKETS,VARIANTS):
        relative=f'arms/{arm}/{basket}/{variant}/cost_10bps/monthly.parquet'
        item=bt['artifacts'].get(relative)
        _need(item is not None,'Saved10bps monthly report is absent')
        frame=_report(backtest,item,relative,48,metadata_guard=month_guard)
        monthly_parts.append(frame.assign(arm=arm,basket=basket,variant=variant,cost_bps=10))
    if verify_sources:_deep_hash(root,run,identity,feature,models,model,model_phase,backtest,bt,bt_phase)
    for label,path in (('features',run/'audit.json'),('models',models/'audit.json'),('backtest',backtest/'audit.json')):
        _receipt(path,expected_receipts[label])
    for label,filename in (('feature_verification','features.json'),('model_verification','models.json'),('backtest_verification','backtest.json')):
        _receipt(verification/filename,expected_receipts[label])
    return {'directory':run,'backtest_directory':backtest,'verification_directory':verification,
        'year_summaries':pd.DataFrame(annuals).sort_values('year'),
        'model_comparison':comparison,'model_summary':model['summary'],
        'model_classes_1_2_9_10':pd.concat(class_parts,ignore_index=True) if class_parts else pd.DataFrame(),
        'backtest_comparison':bt_comparison,'backtest_yearly':yearly,'selection_coverage':coverage,
        'monthly_10bps':pd.concat(monthly_parts,ignore_index=True),'verification_receipts':receipts,
        'backtest_summary':{key:bt[key] for key in ('status','scenarios','sessions','outcome_months','complete_proxy_paths','failed_closed_paths')},
        'summary':{'feature_status':'COMPLETE','model_status':'COMPLETE','backtest_status':'COMPLETE',
            'outer_years':list(YEARS),'prediction_months':48,'daily_sessions':1006,'annual_model_fits':16,
            'train_only_preprocessor_states':8,'models_deserialized':False,'numerical_work_started':False,
            'features_or_prices_decoded':False,'source_bytes_fully_hashed':verify_sources,
            'independent_verification_passed':True,'financial_2024_plus_analysis':False,
            'research_status':'DATA_LIMITED_RESTRICTED_UNIVERSE','identity_policy':'provisional_unique_resolved_link'}}
