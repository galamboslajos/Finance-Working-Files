"""Prepare a fresh walk-forward identity from reviewed corrected annuals.

This worker copies pinned bytes, never recomputes features or changes frozen
workers. Model fitting remains an explicit separate action.
"""
from pathlib import Path
import hashlib
import json
import shutil
from importlib.metadata import version
import platform

from .catalog import canonical_sha256
from .source_quality_reporting import read_source_quality_repair
from .universe import DataContractError
from .walkforward_models import WORKERS as MODEL_WORKERS, _configuration
from .full_panel import _specs

YEARS=tuple(range(2010,2024))
REBUILT={2010,2011,2012,2013,2022,2023}


def need(condition,message):
    if not condition:raise DataContractError(message)


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pin(root,path):
    root,path=Path(root).resolve(),Path(path).resolve()
    need(path.is_relative_to(root) and path.is_file(),'Pinned path must be inside the workspace')
    return {'path':str(path.relative_to(root)),'size_bytes':path.stat().st_size,'sha256':hash_file(path)}


def verify(root,item):
    rel=Path(item['path']);need(not rel.is_absolute() and '..' not in rel.parts,'Safe relative pin required')
    path=Path(root)/rel
    need(path.resolve().is_relative_to(Path(root).resolve()) and path.is_file()
         and path.stat().st_size==item['size_bytes'] and hash_file(path)==item['sha256'],
         'Pinned artifact changed: '+str(rel))
    return path


def js(path):return json.loads(Path(path).read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def prepare_source_corrected_run(root,run_directory,*,approved_replay=False):
    """Freeze all fourteen source references and copy four execution annuals.

    The new outcome-year audits explicitly label unchanged numerical reuse and
    pin the original reviewed audit. No audit is represented as a fresh feature
    calculation. Existing source files, runs and pointers are never replaced.
    """
    need(approved_replay is True,'Explicit user-authorized corrected replay required')
    root=Path(root).resolve();run=Path(run_directory).resolve()
    need(run.is_relative_to(root/'outputs/walkforward_source_corrected') and not run.exists(),
         'Use a fresh private source-corrected run directory')
    cfg_path=root/'configs/walkforward_2020_2023.json';cfg=js(cfg_path)
    _configuration(root,cfg)
    handoff=read_source_quality_repair(root)
    need(handoff['summary']['rebuilt_outcome_years']==sorted(REBUILT),'Unexpected repair scope')
    repair=root/'outputs/source_quality/confirmed-repairs-20261016-v1'
    manifest_path=repair/'corrected_annual_sources.json';manifest=js(manifest_path)
    entries=manifest['years']
    need([x['year'] for x in entries]==list(YEARS) and {x['year'] for x in entries if x['corrected']}==REBUILT,
         'Complete ordered corrected annual source map required')
    independent=repair/'input_independent_review/review_receipt.json'
    need(js(independent)['status']=='PASS_BOUNDED_SIX_ANNUAL_INPUT_RECOMPUTATION',
         'Independent actual corrected input audit required')
    annual_sources={};sources={};copy_bytes=0
    for item in entries:
        y=item['year'];directory=(root/item['directory']).resolve()
        need(directory.is_relative_to(root/'outputs'),'Annual source leaves private outputs')
        ap=directory/'audit.json';a=js(ap)
        need(hash_file(ap)==item['audit_sha256'] and a['status']=='COMPLETE' and a['year']==y
             and a['build_identity_sha256']==item['build_identity_sha256'],'Annual source audit differs')
        for artifact in a['artifacts'].values():verify(directory,artifact)
        sources[str(y)]={'directory':item['directory'],'audit_pin':pin(root,ap),
                         'build_identity_sha256':item['build_identity_sha256'],
                         'corrected':item['corrected'],'original_rows':item['original_rows']}
        if y>=2020:
            annual_sources[str(y)]={'directory':f'years/{y}','directory_scope':'run',
                                    'parent_reference':'THIS_BUILD','audit_pin':None}
            copy_bytes+=sum(v['size_bytes'] for v in a['artifacts'].values())
            continue
        if y<=2013:parent=repair/'el_paso_input_rebuild'
        elif y<=2017:parent=root/cfg['parent_training_directory']
        else:parent=root/cfg['corrected_validation_directory']
        identity_path=parent/'build_identity.json';parent_audit=parent/'audit.json'
        need(canonical_sha256(js(identity_path))==item['build_identity_sha256']
             and js(parent_audit)['status']=='COMPLETE'
             and js(parent_audit)['build_identity_sha256']==item['build_identity_sha256'],
             'Reused annual parent identity/completion mismatch')
        annual_sources[str(y)]={'directory':item['directory'],'directory_scope':'root',
            'build_identity_sha256':item['build_identity_sha256'],'audit_pin':pin(root,ap),
            'parent_identity_pin':pin(root,identity_path),'parent_audit_pin':pin(root,parent_audit),
            'reused_without_numeric_rebuild':True}
    need(copy_bytes<1024**3,'Copied annual input ceiling exceeded')
    old=root/'outputs/walkforward/2020-2023-aacd7579fcf21e6a'
    preserved={str(p.relative_to(root)):pin(root,p) for p in [
        old/'build_identity.json',old/'audit.json',old/'models/audit.json',
        old/'models/build_identity.json',old/'backtest_identity_v2/audit.json',
        old/'backtest_identity_v2/build_identity.json',cfg_path]}
    identity={'schema_version':1,'phase':'source_corrected_walkforward_2020_2023_v1',
        'authority':'User explicitly authorized refitting all four model arms and rerunning2020–2023 walk-forward on corrected inputs with unchanged rules on2026-10-16.',
        'configuration':cfg,'configuration_sha256':canonical_sha256(cfg),
        'feature_dictionary_sha256':canonical_sha256(_specs()),
        'annual_sources':annual_sources,'reviewed_source_annuals':sources,
        'source_correction_manifest_pin':pin(root,manifest_path),
        'source_correction_handoff_pin':pin(root,repair/'handoff.json'),
        'independent_corrected_input_review_pin':pin(root,independent),
        'preserved_reference_pins':preserved,
        'quote_amendments_fingerprint':canonical_sha256({'base':cfg['quote_cache_fingerprint'],
            'reviewed_corrected_sources':pin(root,manifest_path),'input_review':pin(root,independent)}),
        'workers':{name:hash_file(Path(__file__).with_name(name)) for name in MODEL_WORKERS},
        'source_replay_worker_sha256':hash_file(Path(__file__)),
        'runtime':{'python':platform.python_version(),**{p:version(p) for p in
                   ('numpy','pandas','pyarrow','scikit-learn','xgboost','exchange-calendars')}},
        'scope':{'new_estimator_fits':16,'new_preprocessor_fits':8,'parameter_search':False,
                 'primary_context_coverage':.8,'sensitivity_90pct_fits':0,
                 'new2018_2019_development_fits':0,'financial_2024_plus_analysis':False,
                 'new_cloud_reads_or_mutations':False,'existing_runs_changed':False},
        'model_input_rebuild':'Reviewed fixed annual values reused; four outcome-year caches copied byte for byte under new identity.',
        'historical_identity_or_survivorship_certified':False}
    fingerprint=canonical_sha256(identity)
    run.mkdir(parents=True)
    save(run/'build_identity.json',identity)
    save(run/'source_preservation_manifest.json',preserved)
    audits={}
    for year in range(2020,2024):
        source=root/sources[str(year)]['directory'];a=js(source/'audit.json')
        dest=run/'years'/str(year);dest.mkdir(parents=True)
        for item in a['artifacts'].values():
            source_path=verify(source,item);target=dest/item['path']
            need(target.resolve().is_relative_to(dest),'Artifact copy escapes annual directory')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_path,target)
            verify(dest,item)
        shutil.copy2(source/'audit.json',dest/'reference_annual_audit.json')
        a={**a,'build_identity_sha256':fingerprint,
           'source_corrected_replay_lineage':{**sources[str(year)],
             'numerical_artifact_bytes_unchanged':True,'new_feature_calculations':0,
             'original_audit_preserved_as':'reference_annual_audit.json'},
           'artifacts':{**a['artifacts'],'reference_annual_audit':pin(dest,dest/'reference_annual_audit.json')}}
        save(dest/'audit.json',a);audits[str(year)]=pin(run,dest/'audit.json')
    for item in preserved.values():verify(root,item)
    save(run/'audit.json',{'status':'COMPLETE','phase':'reviewed_annual_input_reuse_only',
        'build_identity_sha256':fingerprint,'years':audits,'model_fits':0,'backtest_runs':0,
        'copied_input_bytes':copy_bytes,'original_sources_and_runs_preserved':True,
        'financial_2024_plus_analysis':False})
    return run,fingerprint
