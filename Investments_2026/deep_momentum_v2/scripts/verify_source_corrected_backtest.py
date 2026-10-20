"""Independent fresh-source scope adapter; frozen V2 equations are unchanged."""
from pathlib import Path
from copy import deepcopy
import argparse
import importlib.util
import json
import os
import sys

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'

BASE_SHA='16c3651289258bed24a83d500f6b8589fac3bcb4d358e516fe981acec527a794'
FRESH='fresh_source_corrected_walkforward_v1'
_base=None


def load_equations(root):
    import hashlib
    path=Path(root)/'scripts/verify_walkforward_backtest_v2.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==BASE_SHA,'Independent equation script changed'
    spec=importlib.util.spec_from_file_location('_frozen_source_replay_equations',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def review_fresh_completed_reference(root,run,backtest,phase):
    """Replace only the old failed-attempt authority check, before value reads."""
    v=_base;require=v.require
    require(phase.get('fresh_source_replay')==FRESH and 'prior_failed_attempt' not in phase,
            'Fresh corrected replay cannot fabricate a prior failure')
    root_identity=v.json_file(run/'build_identity.json');identity=v.canonical(root_identity)
    require(root_identity['phase']=='source_corrected_walkforward_2020_2023_v1'
            and phase['parent_identity_sha256']==identity,'Wrong fresh source root')
    context=phase.get('original_completed_reference')
    require(isinstance(context,dict) and context.get('economic_reports_loaded') is False
            and context.get('fresh_run_requires_no_prior_failed_attempt') is True,
            'Exact completed-reference authority required')
    prior=v.local(root,context['directory'])
    require(not prior.is_relative_to(run) and not run.is_relative_to(prior),
            'Fresh replay overlaps the completed old reference')
    pins=context['immutable_source_pins']
    require(isinstance(pins,list) and 1<=len(pins)<=512
            and sum(x['size_bytes'] for x in pins)<=256*1024**2,
            'Bounded completed reference metadata pins required')
    for item in pins:
        require(item in phase['source_pins'],'Old reference pin missing from new phase')
        v.pinned(root,item)
    audit_path=v.pinned(root,context['audit_pin']);phase_path=v.pinned(root,context['phase_file_pin'])
    require(audit_path==prior/'audit.json' and phase_path==prior/'build_identity.json',
            'Completed reference redirects its audit/phase')
    old_audit=v.json_file(audit_path);old=v.json_file(phase_path)
    preserved=root_identity['preserved_reference_pins']
    require(preserved.get(context['audit_pin']['path'])==context['audit_pin']
            and preserved.get(context['phase_file_pin']['path'])==context['phase_file_pin'],
            'Old completed reference differs from pre-fit preservation pins')
    require(old_audit['status']=='COMPLETE' and old_audit['phase']=='walkforward_continuous_backtest_v1'
            and old.get('implementation_revision')=='identity_route_failure_v2'
            and old_audit['phase_identity_sha256']==v.canonical(old)==context['phase_identity_sha256']
            and old_audit['build_identity_sha256']==old['parent_identity_sha256']==context['root_identity_sha256']
            and old['parent_identity_sha256']!=identity,'Old completed authority changed')
    require(context['workers']==old['workers'] and old['configuration_sha256']==phase['configuration_sha256'],
            'Fresh replay changed the original accounting protocol')
    for name,value in old['workers'].items():
        require(phase['workers'].get(name)==value,'Frozen numerical accounting worker changed: '+name)
    binding={'schema':'dmv2-source-corrected-execution-evidence-rebind/v1',
        'original_root_identity_sha256':context['root_identity_sha256'],
        'new_root_identity_sha256':identity,'original_reference_audit_pin':context['audit_pin'],
        'original_reference_phase_identity_sha256':context['phase_identity_sha256'],
        'economic_terms_and_dated_handover_conditions_unchanged':True,
        'original_evidence_bytes_unchanged':True,'source_identity_promoted':False,
        'authority':'Explicit user authorization to refit all four arms and replay corrected inputs under unchanged rules.'}
    events=deepcopy(old['event_registry']);events.update(build_identity_sha256=identity,source_corrected_rebind=binding)
    handover=deepcopy(old['identity_handover_registry']);handover.update(root_build_identity_sha256=identity,source_corrected_rebind=binding)
    require(phase['event_registry']==events and phase['identity_handover_registry']==handover,
            'Economic action terms or dated handover conditions changed')
    old_approval=old['identity_handover_approval'];new_approval=phase['identity_handover_approval']
    old_decision=v.json_file(v.pinned(root,old_approval['decision_pin']))
    old_package=v.json_file(v.pinned(root,old_approval['package_receipt_pin']))
    new_decision_path=v.pinned(root,new_approval['decision_pin'])
    new_package_path=v.pinned(root,new_approval['package_receipt_pin'])
    new_registry_path=v.pinned(root,phase['identity_handover_registry_pin'])
    expected_package=deepcopy(old_package)
    expected_package.update(source_corrected_rebind=binding,
                            original_package_receipt_pin=old_approval['package_receipt_pin'])
    for item in expected_package['files']:
        payload=v.local(new_package_path.parent,item['path'])
        if payload!=new_registry_path:
            require(payload.stat().st_size==item['bytes'] and v.digest(payload)==item['sha256'],
                    'Original nonbinding evidence bytes changed')
        item.update(sha256=v.digest(payload),bytes=payload.stat().st_size)
    require(v.json_file(new_package_path)==expected_package,'Rebound package changed beyond root-binding bytes')
    expected_decision=deepcopy(old_decision)
    expected_decision.update(root_build_identity_sha256=identity,registry_sha256=v.digest(new_registry_path),
        evidence_package_receipt_sha256=v.digest(new_package_path),source_corrected_rebind=binding,
        original_approval_decision_pin=old_approval['decision_pin'])
    require(v.json_file(new_decision_path)==expected_decision,'Rebound approval changed beyond exact root binding')
    corrections=phase.get('source_corrections',{});manifest_pin=corrections['manifest_pin']
    require(manifest_pin==root_identity['source_correction_manifest_pin'] and manifest_pin in phase['source_pins'],
            'Fresh source manifest differs from pre-fit identity')
    handoff_pin=root_identity['source_correction_handoff_pin'];handoff=v.json_file(v.pinned(root,handoff_pin))
    require(handoff['status']=='REPAIRED_INPUTS_READY_MODELS_STALE'
            and manifest_pin in handoff['artifacts']
            and root_identity['independent_corrected_input_review_pin'] in handoff['artifacts'],
            'Source manifest/independent review escaped the reviewed repair package')
    review=v.json_file(v.pinned(root,root_identity['independent_corrected_input_review_pin']))
    require(review['status']=='PASS_BOUNDED_SIX_ANNUAL_INPUT_RECOMPUTATION','Actual corrected input audit absent')
    manifest=v.json_file(v.pinned(root,manifest_pin));entries=manifest['years']
    require([x['year'] for x in entries]==list(range(2010,2024))
            and {x['year'] for x in entries if x['corrected']}=={2010,2011,2012,2013,2022,2023},
            'Fresh source map changed its complete fourteen-year repair scope')
    bindings=[]
    for item in entries:
        source=v.local(root,item['directory']);source_audit=source/'audit.json'
        require(v.digest(source_audit)==item['audit_sha256'],'Original annual audit pin changed')
        a=v.json_file(source_audit)
        require(a['status']=='COMPLETE' and a['year']==item['year']
                and a['build_identity_sha256']==item['build_identity_sha256'],'Wrong source annual identity')
        if item['year']<2020:continue
        target=run/'years'/str(item['year']);b=v.json_file(target/'audit.json')
        require(b['build_identity_sha256']==identity and b['year']==item['year'] and b['status']=='COMPLETE',
                'Detached execution annual not bound to fresh identity')
        require(set(b['artifacts'])==set(a['artifacts'])|{'reference_annual_audit'},'Unexpected detached artifact scope')
        for name,source_pin in a['artifacts'].items():
            v.pinned(source,source_pin);v.pinned(target,b['artifacts'][name])
            require(source_pin==b['artifacts'][name],
                    'Detached annual numerical bytes differ from reviewed source')
        reference=v.pinned(target,b['artifacts']['reference_annual_audit'])
        require(b['artifacts']['reference_annual_audit']['path']=='reference_annual_audit.json'
                and reference==target/'reference_annual_audit.json'
                and v.digest(reference)==item['audit_sha256'],'Detached original audit copy differs')
        original_pin={'path':str(source_audit.relative_to(root)),'size_bytes':source_audit.stat().st_size,'sha256':v.digest(source_audit)}
        expected_lineage={'directory':item['directory'],'audit_pin':original_pin,
            'build_identity_sha256':item['build_identity_sha256'],'corrected':item['corrected'],
            'original_rows':item['original_rows'],'numerical_artifact_bytes_unchanged':True,
            'new_feature_calculations':0,'original_audit_preserved_as':'reference_annual_audit.json'}
        require(b.get('source_corrected_replay_lineage')==expected_lineage,
                'Detached annual lost its exact reviewed source lineage')
        detached=target/'audit.json'
        detached_pin={'path':str(detached.relative_to(root)),'size_bytes':detached.stat().st_size,'sha256':v.digest(detached)}
        bindings.append({'year':item['year'],'original_audit_pin':original_pin,'detached_audit_pin':detached_pin,
                         'all_artifact_bytes_match_reviewed_source':True})
    require(corrections=={'manifest_pin':manifest_pin,'annual_bindings':bindings,'original_annual_sources_unchanged':True},
            'Saved corrected-source bindings differ from independent byte checks')
    return len(pins)


def verify_saved_run(root,run,backtest,*,output):
    global _base
    root,run,backtest,output=map(lambda p:Path(p).resolve(),[root,run,backtest,output])
    _base=load_equations(root)
    _base.require(output.is_relative_to(run) and not output.exists(),'Use a new audit under the fresh run')
    _base.review_prior_failed_attempt=review_fresh_completed_reference
    result=_base.verify_saved_run(root,run,backtest,output=None)
    count=result.pop('prior_failed_attempt_preserved_files')
    result.update(phase='independent_source_corrected_backtest_verification_v1',
        verifier_contract_revision='fresh_completed_reference_scope_with_unchanged_v2_equations',
        completed_reference_preserved_files=count,
        independent_equations_script_sha256=BASE_SHA,script_sha256=_base.digest(Path(__file__)),
        source_corrections_and_exact_evidence_rebind_independently_verified=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','run','backtest','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=verify_saved_run(a.root,a.run,a.backtest,output=a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ['selections','scenario_checks']},sort_keys=True))
