"""Recompute frozen formulas and shared classes on the explicit fallback view."""
from dataclasses import dataclass

import pyarrow as pa

from .walkforward_loading import (FALLBACK_POLICY, KEYS, LoadedWalkforwardYear,
                               _fingerprint, _need)
from .full_features import build_full_feature_panel
from .identity_amendment import SOURCE_ROUTES
from .identity_policy import PROVISIONAL_IDENTITY_POLICY
from .monthly import MonthlyReturnPanel, build_monthly_returns
from .paper_features import COMMON_MOMENTUM_POLICY
from .targets import MonthlyTargetPanel, build_monthly_targets


@dataclass(frozen=True)
class WalkforwardFeatureYear:
    features: object
    monthly: MonthlyReturnPanel
    targets: MonthlyTargetPanel
    loaded: LoadedWalkforwardYear
    provenance: dict


def build_walkforward_year(loaded):
    """Keep every original key; recompute cross sections, benchmarks and ranks.

    The wrong BBBY route contributes no own price features, shared price
    statistics, label or RET payoff. Missing rows remain in all original
    denominators. Other valid histories, including DF/RDC, remain inputs.
    No model is fitted and no future-year source is opened by this worker.
    """
    _need(isinstance(loaded, LoadedWalkforwardYear)
          and _fingerprint(loaded.plan) == loaded.plan.fingerprint,
          "Unchanged explicitly approved fallback inputs required")
    plan = loaded.plan
    _need(loaded.provenance.get("policy") == FALLBACK_POLICY
          and loaded.provenance.get("approved_missing_route_fallback") is True
          and loaded.provenance.get("fully_identity_repaired") is False,
          "Fallback authority cannot become a complete identity repair claim")
    features = build_full_feature_panel(loaded.feature_prices, plan.calendar, plan.snapshots,
        requested_security_ids=plan.requested_ids, daily_cohorts=plan.daily_cohorts,
        mode="historical_reconstruction", identity_policy=PROVISIONAL_IDENTITY_POLICY,
        normalization_policy=COMMON_MOMENTUM_POLICY, context_policy="available_cohort_80pct")
    monthly = build_monthly_returns(loaded.outcome_prices, plan.calendar, plan.snapshots,
        start_month=f"{plan.year}-01", end_month=f"{plan.year}-12",
        requested_security_ids=plan.requested_ids, mode="historical_reconstruction",
        identity_policy=PROVISIONAL_IDENTITY_POLICY)
    targets = build_monthly_targets(features.coverage, monthly,
        label_observed_through=plan.windows[-1].outcome_end)
    original = plan.original_cohort.select(list(KEYS)).to_pylist()
    for table in (features.table, features.coverage, monthly.table, targets.table):
        _need(table.select(list(KEYS)).to_pylist() == original,
              "Fallback computation changed ordered original cohort denominators")
    coverage = features.coverage.to_pylist();labels = targets.table.to_pylist()
    for feature, target in zip(coverage, labels, strict=True):
        if feature["source_security_id"] in SOURCE_ROUTES:
            _need(feature["identity_eligible"] is False and feature["requested"] is False
                  and feature["paper_prediction_eligible"] is False
                  and target["target_available"] is False,
                  "Quarantined old BBBY route acquired a feature/label admission")
    provenance = {**loaded.provenance, "features_calculated": True,
        "monthly_returns_recomputed": True, "target_classes_recomputed": True,
        "all_original_rows_and_denominators_retained": True,
        "paper15_common_population_recomputed": True,
        "daily_historical_benchmarks_recomputed": True,
        "ret_payoff_source_is_new_shared_target_panel": True,
        "old_features_targets_or_model_predictions_reused": False,
        "model_fitted": False, "daily_backtest_run": False,
        "reconstruction_not_knowledge_time_certification": True}
    return WalkforwardFeatureYear(features, monthly, targets, loaded, provenance)


# Explicit feature phase. No notebook or import calls this builder implicitly.
import gc,json,platform,time
from pathlib import Path
from importlib.metadata import version
import pyarrow.compute as pc
from .catalog import canonical_sha256
from .classifier_pilot import _ResourceGuard
from .full_panel import _artifact,_hash,_json,_progress,_specs,_verify_artifacts,_within,_write_table
from .full_target_panel import _save_target_state
from .data_limited_fallback import _preservation,_usage,_verify_year
from .identity_amendment import load_identity_amendment
from .walkforward_loading import NUMERICAL_WORKERS,plan_walkforward_year,load_walkforward_year,wrap_walkforward_quotes
from .universe import DataContractError

YEARS=(2020,2021,2022,2023)
WORKERS=tuple(sorted({*NUMERICAL_WORKERS,'walkforward_loading.py','walkforward_panel.py','walkforward_quotes.py',
 'identity_amendment.py','targets.py','full_target_panel.py','ret.py',
 'fallback_loading.py','fallback_quotes.py','independent_quotes.py','composed_quotes.py',
 'ohlc_repairs.py','bracket_ohlc_repairs.py','supplemental_bars.py'}))


def validate_walkforward_configuration(cfg):
    if (cfg.get('protocol_id')!='dmv2-walkforward-2020-2023-v1'
        or cfg.get('feature_outcome_years')!=list(YEARS)
        or cfg.get('context_policy')!='available_cohort_80pct'
        or cfg.get('research_status')!='DATA_LIMITED_RESTRICTED_UNIVERSE'
        or cfg.get('execution',{}).get('financial_2020_2023_analysis_authorized') is not True
        or cfg.get('execution',{}).get('financial_2024_plus_analysis') is not False
        or set(cfg.get('execution_restrictions',{}).get('native_ids',{}))!={'sec_gjP4xM','sec_zmYNnz'}):
        raise DataContractError('Walk-forward configuration must retain frozen2020–2023 scope,80%context and restrictions')
    return cfg


def _reference_sources(root,cfg):
    """Pin all historical reused annual audits without opening feature values."""
    annual={}
    for prefix,years in [('parent_training',range(2010,2018)),('corrected_validation',(2018,2019))]:
        directory=_within(root,root/cfg[prefix+'_directory'])
        parent_identity=json.loads((directory/'build_identity.json').read_text())
        expected=cfg[prefix+'_build_identity_sha256']
        if canonical_sha256(parent_identity)!=expected:raise DataContractError('Reused parent identity changed')
        parent_audit=json.loads((directory/'audit.json').read_text())
        if parent_audit.get('status')!='COMPLETE' or parent_audit.get('build_identity_sha256')!=expected:
            raise DataContractError('Reused historical feature phase is incomplete')
        for year in years:
            audit_path=directory/'years'/str(year)/'audit.json'
            annual_audit=json.loads(audit_path.read_text())
            if annual_audit.get('status')!='COMPLETE' or annual_audit.get('year')!=year or annual_audit.get('build_identity_sha256')!=expected:
                raise DataContractError('Historical annual reference year or identity changed')
            parent_pin=parent_audit.get('years',{}).get(str(year))
            if not parent_pin:raise DataContractError('Historical parent omits annual checkpoint pin')
            _verify_artifacts(directory,{str(year):parent_pin})
            annual[str(year)]={'directory':str(audit_path.parent.relative_to(root)),'directory_scope':'root',
                'build_identity_sha256':expected,'audit_pin':_artifact(root,audit_path),
                'parent_identity_pin':_artifact(root,directory/'build_identity.json'),
                'parent_audit_pin':_artifact(root,directory/'audit.json'),'reused_without_financial_decode':True}
    for year in YEARS:
        annual[str(year)]={'directory':'years/'+str(year),'directory_scope':'run',
            'parent_reference':'THIS_BUILD','audit_pin':None,'reused_without_financial_decode':False}
    return annual


def plan_walkforward_panel(root,cfg,quotes):
    """Plan and pin four new years plus ten reused years using metadata only."""
    root=Path(root).resolve();validate_walkforward_configuration(cfg)
    quotes=wrap_walkforward_quotes(quotes);quotes.verify(root)
    if quotes.fingerprint!=cfg.get('quote_cache_fingerprint'):
        raise DataContractError('Reviewed quote bundle differs from frozen configuration fingerprint')
    amendment=load_identity_amendment(root/cfg['identity_amendment'])
    annual_sources=_reference_sources(root,cfg);plans={}
    for year in YEARS:
        plan=plan_walkforward_year(root,year,amendment=amendment,approved_missing_route_fallback=True)
        plans[str(year)]={'plan_fingerprint':plan.fingerprint,'read_plan':plan.read_plan,'source_provenance':plan.provenance}
        del plan;gc.collect();pa.default_memory_pool().release_unused()
    projected=sum(p['read_plan']['projected_compressed_bytes'] for p in plans.values())
    if projected>cfg['execution']['feature_maximum_projected_read_bytes']:
        raise DataContractError('Four-year walk-forward read plan exceeds frozen source-projection budget')
    identity={'schema_version':1,'phase':'walkforward_features_and_targets_2020_2023_v1',
        'configuration':cfg,'configuration_sha256':canonical_sha256(cfg),
        'quote_amendments_fingerprint':quotes.fingerprint,'quote_amendments':quotes.provenance,
        'years':plans,'annual_sources':annual_sources,
        'workers':{name:_hash(Path(__file__).with_name(name)) for name in WORKERS},
        'runtime':{'python':platform.python_version(),**{name:version(name) for name in ('numpy','pandas','pyarrow','exchange-calendars')}},
        'feature_dictionary_sha256':canonical_sha256(_specs()),'projected_compressed_bytes':projected,
        'original_stock_months':sum(p['read_plan']['original_monthly_rows'] for p in plans.values()),
        'metadata_quote_preflight_before_financial_decode':True,
        'new_model_fits_in_feature_phase':0,'source_price_cache_tables_decoded_by_planner':False,
        'compiled_quote_overlay_integrity_verified':True,
        'financial_2024_plus_analysis':False,'original_run_pointers_changed':False,
        'historical_identity_certified':False,'survivorship_bias_removed':False}
    return identity,canonical_sha256(identity)


def _save_walkforward_year(directory,result,identity,pin,elapsed):
    artifacts={};loaded,features,monthly,targets=result.loaded,result.features,result.monthly,result.targets
    for name,table in [('features',features.table),('feature_coverage',features.coverage),
        ('original_cohort',loaded.plan.original_cohort),('daily_cohorts',loaded.plan.daily_cohorts),
        ('monthly_returns',monthly.table),('feature_history_daily',loaded.feature_prices),
        ('daily_history_raw',loaded.outcome_prices),('exchange_calendar',loaded.plan.calendar),
        ('daily_prices_raw',loaded.outcome_prices.filter(pc.greater_equal(loaded.outcome_prices['trading_date'],f'{loaded.plan.year}-01-01')))]:
        _write_table(directory,name,table,artifacts)
    _save_target_state(directory,targets,artifacts)
    for name,value in [('feature_provenance',features.provenance),('monthly_provenance',monthly.provenance),
        ('loading_provenance',result.provenance),('feature_summary',features.summary),('monthly_summary',monthly.summary)]:
        _json(directory/(name+'.json'),value);artifacts[name]=_artifact(directory,directory/(name+'.json'))
    summary={'original_rows':features.table.num_rows,
        'prediction_eligible_rows':int(pc.sum(features.coverage['paper_prediction_eligible']).as_py()),
        'labelled_rows':int(pc.sum(targets.table['target_available']).as_py()),
        'monthly_wrong_route_rows':loaded.plan.read_plan['monthly_wrong_route_rows'],
        'daily_wrong_route_rows':loaded.plan.read_plan['daily_wrong_route_rows'],
        'feature_history_rows':loaded.feature_prices.num_rows,'separate_outcome_daily_rows':loaded.outcome_prices.num_rows,
        'elapsed_seconds':elapsed,'process_peak_rss_bytes':_usage(),'financial_2020_2023_analysis':True,
        'financial_2024_plus_analysis':False,'all_original_keys_preserved':True}
    audit={'status':'COMPLETE','year':loaded.plan.year,'build_identity_sha256':pin,
        'research_status':identity['configuration']['research_status'],
        'quote_amendments_fingerprint':identity['quote_amendments_fingerprint'],
        'feature_provenance':features.provenance,'target_artifact_sha256':targets.artifact_sha256,
        'summary':summary,'artifacts':artifacts}
    _json(directory/'audit.json',audit);_verify_year(directory,pin,loaded.plan.year);return audit


def _walkforward_preservation(root,cfg):
    # Hash-only extension: latest reconstructed training and validation parents
    # are immutable evidence as well as the older three reference runs.
    records=_preservation(root)
    for name in ('parent_training_directory','corrected_validation_directory'):
        directory=_within(root,root/cfg[name])
        for path in sorted(directory.rglob('*')):
            if path.is_file():records[str(path.relative_to(root))]=_artifact(root,path)
    return records


def build_walkforward_panel(root,cfg,quotes,*,approved_financial_read=False):
    """Explicit, bounded, checkpointed four-year build after protocol freezing."""
    if approved_financial_read is not True:raise DataContractError('Freeze and approve protocol before future price decoding')
    root=Path(root).resolve();identity,pin=plan_walkforward_panel(root,cfg,quotes);quotes=wrap_walkforward_quotes(quotes)
    destination=_within(root,root/'outputs/walkforward'/('2020-2023-'+pin[7:23]));destination.mkdir(parents=True,exist_ok=True)
    identity_path=destination/'build_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=identity:raise DataContractError('Existing walk-forward feature identity changed')
    else:
        _json(identity_path,identity);_json(destination/'source_preservation_manifest.json',_walkforward_preservation(root,cfg))
    preservation=json.loads((destination/'source_preservation_manifest.json').read_text());_verify_artifacts(root,preservation)
    if (destination/'audit.json').exists():
        audit=json.loads((destination/'audit.json').read_text())
        if audit.get('status')!='COMPLETE' or audit.get('build_identity_sha256')!=pin or set(audit.get('years',{}))!={str(y) for y in YEARS}:
            raise DataContractError('Preserve unexpected completed feature checkpoint')
        _verify_artifacts(destination,audit['years'])
        for year in YEARS:_verify_year(destination/'years'/str(year),pin,year)
        return destination
    (destination/'years').mkdir(exist_ok=True);started=time.monotonic()
    guards={'execution':{'maximum_wall_seconds':cfg['execution']['feature_maximum_wall_seconds'],
        'maximum_peak_rss_bytes':cfg['execution']['feature_maximum_peak_rss_bytes']}}
    _progress('walkforward_feature_plan',build_identity_sha256=pin,output=str(destination),projected_compressed_bytes=identity['projected_compressed_bytes'])
    with _ResourceGuard(guards,started,destination):
        for year in YEARS:
            directory=destination/'years'/str(year)
            if (directory/'audit.json').exists():_verify_year(directory,pin,year);continue
            if directory.exists():raise DataContractError('Preserve incomplete annual walk-forward attempt for review')
            directory.mkdir();_json(directory/'ATTEMPT.json',{'build_identity_sha256':pin,'year':year});_progress('walkforward_year_started',year=year)
            t=time.monotonic();amendment=load_identity_amendment(root/cfg['identity_amendment'])
            plan=plan_walkforward_year(root,year,amendment=amendment,approved_missing_route_fallback=True)
            if plan.fingerprint!=identity['years'][str(year)]['plan_fingerprint']:raise DataContractError('Annual walk-forward plan changed after freezing')
            loaded=load_walkforward_year(root,plan=plan,quote_amendments=quotes,approved_financial_read=True)
            result=build_walkforward_year(loaded);audit=_save_walkforward_year(directory,result,identity,pin,time.monotonic()-t)
            _progress('walkforward_year_completed',year=year,**audit['summary']);del result,loaded,plan;gc.collect();pa.default_memory_pool().release_unused()
            if sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())>cfg['execution']['maximum_output_bytes']:
                raise DataContractError('Walk-forward feature outputs exceed frozen budget')
    _verify_artifacts(root,preservation);quotes.verify(root)
    _json(destination/'audit.json',{'status':'COMPLETE','phase':'walkforward_features_and_targets',
        'research_status':cfg['research_status'],'build_identity_sha256':pin,
        'years':{str(y):_artifact(destination,destination/'years'/str(y)/'audit.json') for y in YEARS},
        'annual_sources':identity['annual_sources'],'source_preservation_verified':True,
        'new_model_fits':0,'financial_2020_2023_analysis':True,'financial_2024_plus_analysis':False,
        'elapsed_seconds':time.monotonic()-started})
    return destination
