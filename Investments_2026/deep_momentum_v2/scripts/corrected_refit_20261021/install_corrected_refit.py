#!/usr/bin/env python3
"""Additive immutable archive installer. Opaque copies only; never loads models.

All real invocations belong to the parent/root after terminal numerical proof.
Gate JSON reads are charged before parsing; archive hashing/copying is opaque
byte I/O. External control plans avoid circular self-inventory and source edits.
Destination writes require the calling tool's normal filesystem approval.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import stat
import sys

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parent
REPO=Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
CONTROL=Path('/private/tmp/dmv2-corrected-refit-install-20261021')
DESTINATION=REPO/'outputs/corrected_refit/2020-2023-20261021-v1'
MAX_BYTES=20*1024**3
GENERATED=('archive_installation_plan.json','archive_gate_plan.json','archive_scope.json',
           'relocation_index.json','installed_catalog.json','archive_receipt.json')
BACKTEST_PINS=('source_plan_pin','backtest_protocol_pin','selection_audit_pin','preflight_audit_pin','replay_audit_pin')
HASH_BYTES=0
COPY_BYTES=0


def need(value,message):
    if not value:raise ValueError(message)


def encode(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def digest(path):
    global HASH_BYTES
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk);HASH_BYTES+=len(chunk)
    return h.hexdigest()


def pin(path):
    p=Path(path)
    need(p.is_absolute() and p==p.resolve() and not p.is_symlink() and p.is_file(),'Absolute regular pinned file required')
    return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':digest(p)}


def check(item):
    p=Path(item['path'])
    need(pin(p)=={k:item[k] for k in ('path','size_bytes','sha256')},'Pinned source bytes changed: '+str(p))
    return p


def read(path):return json.loads(Path(path).read_text())


def save_new(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(encode(value));f.flush();os.fsync(f.fileno())
    return pin(p)


def external_output(path):
    original=Path(path).absolute()
    need(not any(q.is_symlink() for q in [original,*original.parents]),'Symlinked archive control directory')
    p=original.resolve()
    need(p.is_relative_to(CONTROL) and not p.is_relative_to(SOURCE) and not p.exists(),
         'Fresh external archive-control path required')
    return p


def source_file(item):
    p=Path(item['path'])
    need(p.is_relative_to(SOURCE) and not p.is_symlink(),'Gate/artifact must belong to this concrete source attempt')
    return check(item)


def validate_scope(scope):
    need(scope['schema']=='dmv2-corrected-refit-archive-scope/v1'
         and scope['status']=='SEALED_AFTER_NUMERICAL_COMPLETION_BEFORE_ARCHIVE'
         and scope['source_root']==str(SOURCE) and scope['destination']==str(DESTINATION)
         and scope['all_numerical_stages_terminal'] is True,'Concrete terminal archive scope required')
    need(Path(scope['gates']['progress']['path'])==SOURCE/'run/models/pipeline/progress.json'
         and Path(scope['gates']['models']['path'])==SOURCE/'run/verification/models.json'
         and Path(scope['gates']['backtest']['path'])==SOURCE/'run/verification/backtest.json'
         and Path(scope['gates']['model_report']['path'])==SOURCE/'run/reports/models_v1/receipt.json',
         'Unexpected terminal gate path')
    need(set(scope['gates'])=={'progress','models','backtest','model_report'},'Exact four terminal gates required')
    need(set(scope['backtest_expected_pins'])==set(BACKTEST_PINS),'Closed backtest provenance pins required')
    need(type(scope['artifacts']) is dict and scope['artifacts'],'Explicit artifact locator map required')
    for name,item in scope['artifacts'].items():
        need(isinstance(name,str) and name and Path(item['path']).is_relative_to(SOURCE),
             'Explicit new-attempt artifact pins required')
    need(type(scope['original_input_pins']) is list,'Original input pin list required (may be empty)')
    for item in scope['original_input_pins']:
        need(Path(item['path']).is_relative_to(REPO) and not Path(item['path']).is_relative_to(DESTINATION),
             'Original input preservation pin outside repository')
    return scope


def make_gate_plan(scope_path,output):
    """Pin gates/metadata only; never parse metric-bearing gate proof bodies."""
    scope_path=Path(scope_path).absolute();scope=validate_scope(read(scope_path))
    for item in scope['gates'].values():source_file(item)
    source_file(scope['model_protocol_pin'])
    for item in scope['backtest_expected_pins'].values():source_file(item)
    output=external_output(output)
    return save_new(output,{'schema':'dmv2-corrected-refit-archive-gate-plan/v1',
        'status':'SEALED_BEFORE_GATE_JSON_READ','scope_pin':pin(scope_path),
        'installer_pin':pin(Path(__file__).resolve()),'control_pin':pin(SOURCE/'control.py'),
        'gates':scope['gates'],'projected_compressed_source_bytes':sum(p['size_bytes'] for p in scope['gates'].values()),
        'model_deserializations':0,'financial_table_reads':0,'opaque_hash_bytes_at_plan':HASH_BYTES})


def require_gate_reservation(label,amount,plan_pin):
    # Import only after disabling bytecode writes; shared ledger remains immutable
    # from the inventory seal onward. This check itself does not write the ledger.
    from control import ROOT,require_reservation
    need(ROOT==SOURCE,'Unexpected source ledger root')
    return require_reservation(SOURCE,label,amount,plan_pin,'compressed_source')


def gate_summary(scope,proofs):
    p,m,b,r=(proofs[k] for k in ('progress','models','backtest','model_report'))
    need(p['status']=='COMPLETE_VERIFIED' and p['completed_estimator_fits']==20
         and p['completed_preprocessors']==8 and p['completed_phases']==50
         and p['completed_annual_verifications']==4 and p['final_verification_pin']==scope['gates']['models'],
         'Model pipeline is not completely verified')
    need(m['schema']=='dmv2-independent-corrected-model-final-verification/v1' and m['status']=='PASS'
         and m['verified_estimator_fits']==20 and m['verified_training_only_preprocessing_states']==8
         and m['outcome_years']==[2020,2021,2022,2023] and m['outcome_months']==48
         and m['same_cohort_all_arms'] is True and m['protocol_pin']==scope['model_protocol_pin'],
         'Independent model PASS or source identity missing')
    need(b['schema']=='dmv2-independent-corrected-backtest-verification/v1' and b['status']=='PASS'
         and b['scenarios']==120 and b['sessions']==1006 and b['outcome_months']==48 and b['selection_sets']==15
         and b['all_scenarios_retained'] is True and b['original_source_pins_unchanged'] is True
         and b['missing_selected_positions_dropped'] is False
         and b['new_estimator_fits']==0 and b['new_inference_calls']==0,'Independent complete-scope backtest PASS missing')
    need(type(b['complete_proxy_paths']) is int and type(b['withheld_paths']) is int
         and b['complete_proxy_paths']>=0 and b['withheld_paths']>=0
         and b['complete_proxy_paths']+b['withheld_paths']==120,'Backtest scenarios disappeared')
    for name,item in scope['backtest_expected_pins'].items():need(b[name]==item,'Backtest proof provenance differs: '+name)
    need(r['status']=='COMPLETE' and r['final_verification_pin']==scope['gates']['models']
         and r['model_fits']==0 and r['inference_calls']==0,'Passive model comparison is incomplete')
    return {'models':'COMPLETE_VERIFIED','estimator_fits':20,'preprocessing_states':8,'backtest':'PASS',
        'scenarios':120,'complete_proxy_paths':b['complete_proxy_paths'],'withheld_paths':b['withheld_paths'],
        'all_scenarios_retained':True,'missing_selected_positions_dropped':False,
        'verified_withheld_paths_preserved':True,'all_numerical_stages_terminal':True}


def inventory(root):
    root=Path(root);need(root==SOURCE and not root.is_symlink() and root.is_dir(),'Concrete regular source directory required')
    files={};directories=[]
    for parent,dirs,names in os.walk(root,followlinks=False):
        parent=Path(parent)
        for name in sorted(dirs):
            p=parent/name;need(not p.is_symlink() and p.is_dir(),'Symlink/non-directory in source tree')
            directories.append(p.relative_to(root).as_posix())
        for name in sorted(names):
            p=parent/name;need(not p.is_symlink() and stat.S_ISREG(p.lstat().st_mode),'Nonregular/symlink source entry')
            relative=p.relative_to(root).as_posix()
            need(relative not in GENERATED,'Source collides with reserved archive metadata filename')
            files[relative]=pin(p)
    need(files,'Empty source archive')
    need(sum(p['size_bytes'] for p in files.values())<=MAX_BYTES,'Archive source exceeds20GiB')
    return {'files':dict(sorted(files.items())),'directories':sorted(directories),
            'source_bytes':sum(p['size_bytes'] for p in files.values())}


def no_running_metadata(scope):
    """Inspect only safe guard status JSON, never logs or numerical reports."""
    for p in (SOURCE/'run/control').glob('*.json'):
        if p.name=='active_time.json':continue
        need(not p.is_symlink(),'Symlink guard status')
        z=read(p)
        if 'status' in z:need(z['status']!='RUNNING','Numerical/resource guard still running')


def prepare(gate_plan_path,expected_sha,reservation,output):
    gp=pin(Path(gate_plan_path).absolute());need(gp['sha256']==expected_sha,'Gate plan changed')
    z=read(check(gp));need(z['schema']=='dmv2-corrected-refit-archive-gate-plan/v1'
        and z['installer_pin']==pin(Path(__file__).resolve()),'Gate plan/installer identity changed')
    check(z['control_pin']);scope=validate_scope(read(check(z['scope_pin'])))
    need(z['gates']==scope['gates'],'Frozen gate scope differs')
    require_gate_reservation(reservation,z['projected_compressed_source_bytes'],gp)
    output=external_output(output)
    # A reservation cannot be reused to parse numerical proof JSON twice.
    consumed=CONTROL/('gate_read_'+hashlib.sha256(reservation.encode()).hexdigest()+'.json')
    save_new(consumed,{'status':'GATE_READ_ATTEMPT_CONSUMED','gate_plan_pin':gp,'reservation':reservation})
    source_file(scope['model_protocol_pin'])
    for item in scope['backtest_expected_pins'].values():source_file(item)
    proofs={name:read(source_file(item)) for name,item in z['gates'].items()}
    summary=gate_summary(scope,proofs);no_running_metadata(scope)
    for item in scope['artifacts'].values():source_file(item)
    for item in scope['original_input_pins']:check(item)
    need(not DESTINATION.exists(),'Preserve existing archive destination')
    body=inventory(SOURCE)
    for item in scope['gates'].values():source_file(item)
    check(z['scope_pin']);check(z['installer_pin']);check(z['control_pin'])
    return save_new(output,{'schema':'dmv2-corrected-refit-archive-install-plan/v1','status':'SEALED_SOURCE_INVENTORY',
        'source_root':str(SOURCE),'destination':str(DESTINATION),'gate_plan_pin':gp,'scope_pin':z['scope_pin'],
        'installer_pin':z['installer_pin'],'gate_summary':summary,'artifacts':scope['artifacts'],
        'original_input_pins':scope['original_input_pins'],**body,'maximum_archive_bytes':MAX_BYTES,
        'source_inventory_sha256':hashlib.sha256(encode(body)).hexdigest(),
        'financial_table_reads':0,'model_deserializations':0,'hash_io_bytes_at_seal':HASH_BYTES,
        'embedded_legacy_paths_rewritten':False})


def relocated(item,root=None):
    root=DESTINATION if root is None else root
    p=Path(item['path']);need(p.is_relative_to(SOURCE),'Only source-attempt paths relocate')
    return {**item,'path':str(root/p.relative_to(SOURCE))}


def copy_new(source,destination,expected):
    global COPY_BYTES
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    need(not destination.exists() and not destination.is_symlink(),'Refuse archive overwrite')
    h=hashlib.sha256();size=0
    with Path(source).open('rb') as src,destination.open('xb') as dst:
        for chunk in iter(lambda:src.read(4*1024**2),b''):
            dst.write(chunk);h.update(chunk);size+=len(chunk);COPY_BYTES+=len(chunk)
        dst.flush();os.fsync(dst.fileno())
    need(h.hexdigest()==expected['sha256'] and size==expected['size_bytes'],'Source changed during byte copy')
    item=pin(destination)
    need(item['sha256']==expected['sha256'] and item['size_bytes']==expected['size_bytes'],'Copied file hash differs')
    return item


def execute(plan_path,expected_sha):
    pp=pin(Path(plan_path).absolute());need(pp['sha256']==expected_sha,'Install plan changed')
    z=read(check(pp))
    need(z['schema']=='dmv2-corrected-refit-archive-install-plan/v1' and z['status']=='SEALED_SOURCE_INVENTORY'
         and z['source_root']==str(SOURCE) and z['destination']==str(DESTINATION)
         and z['installer_pin']==pin(Path(__file__).resolve()) and z['maximum_archive_bytes']==MAX_BYTES,
         'Concrete sealed installation plan required')
    need(not DESTINATION.exists() and not DESTINATION.is_symlink(),'Preserve existing archive destination')
    need(not any(p.is_symlink() for p in DESTINATION.parents),'Symlinked destination parent')
    for item in z['original_input_pins']:check(item)
    before=inventory(SOURCE)
    need(all(before[k]==z[k] for k in ('files','directories','source_bytes')),'Source tree changed after inventory seal')
    need(hashlib.sha256(encode(before)).hexdigest()==z['source_inventory_sha256'],'Sealed source inventory fingerprint differs')
    controls={'archive_installation_plan.json':pp,'archive_gate_plan.json':z['gate_plan_pin'],'archive_scope.json':z['scope_pin']}
    for item in controls.values():check(item)
    reloc={'schema':'dmv2-corrected-refit-archive-relocation/v1','source_root':str(SOURCE),'destination':str(DESTINATION),
        'embedded_legacy_paths_rewritten':False,'files':{item['path']:str(DESTINATION/relative) for relative,item in z['files'].items()},
        'external_control_copies':{item['path']:str(DESTINATION/name) for name,item in controls.items()}}
    catalog={'schema':'dmv2-corrected-refit-installed-catalog/v1','status':'VERIFIED_ARCHIVE_NO_ACTIVE_POINTER_PROMOTION',
        'package_root':str(DESTINATION),'source_root':str(SOURCE),'gate_summary':z['gate_summary'],
        'artifacts':{name:relocated(item) for name,item in z['artifacts'].items()},
        'files':{relative:relocated(item) for relative,item in z['files'].items()},
        'original_input_pins':z['original_input_pins'],'embedded_legacy_paths_preserved':True,
        'legacy_path_resolution':'Translate only original source-root prefix using relocation_index.json; copied original receipts remain byte-identical.',
        'archive_plan_pin':{**pp,'path':str(DESTINATION/'archive_installation_plan.json')}}
    # Compute all output bytes before creating the fresh destination.
    metadata={'relocation_index.json':reloc,'installed_catalog.json':catalog}
    catalog_pin={'path':str(DESTINATION/'installed_catalog.json'),'size_bytes':len(encode(catalog)),
                 'sha256':hashlib.sha256(encode(catalog)).hexdigest()}
    receipt={'schema':'dmv2-corrected-refit-archive-receipt/v1','status':'COMPLETE_INSTALLED',
        'source_root':str(SOURCE),'destination':str(DESTINATION),'source_inventory_sha256':z['source_inventory_sha256'],
        'installed_catalog_pin':catalog_pin,'gate_summary':z['gate_summary'],'all_copied_bytes_verified':True,
        'source_tree_unchanged_before_after':True,'original_input_pins_unchanged':True,
        'copied_source_file_count':len(z['files']),'source_bytes':z['source_bytes'],
        'active_pointer_promoted':False,'financial_table_reads':0,'model_deserializations':0}
    metadata['archive_receipt.json']=receipt
    total=z['source_bytes']+sum(p['size_bytes'] for p in controls.values())+sum(len(encode(v)) for v in metadata.values())
    need(total<=MAX_BYTES,'Archive plus generated metadata exceeds20GiB')
    DESTINATION.parent.mkdir(parents=True,exist_ok=True);DESTINATION.mkdir()
    # Failed partial installations remain; no retries or destination replacement.
    for relative in z['directories']:(DESTINATION/relative).mkdir(parents=True,exist_ok=True)
    for relative,item in z['files'].items():copy_new(item['path'],DESTINATION/relative,item)
    for name,item in controls.items():copy_new(item['path'],DESTINATION/name,item)
    after=inventory(SOURCE)
    need(after==before,'Source changed while archiving')
    for item in controls.values():check(item)
    for item in z['original_input_pins']:check(item)
    for relative,item in z['files'].items():
        got=pin(DESTINATION/relative)
        need(got==relocated(item),'Copied source bytes changed before completion')
    for name,value in metadata.items():save_new(DESTINATION/name,value)
    actual=sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file())
    need(actual==total and actual<=MAX_BYTES,'Final archive byte total differs')
    return pin(DESTINATION/'archive_receipt.json')


def publish_doc(plan_path,expected_sha,source_relative,destination):
    pp=pin(Path(plan_path).absolute());need(pp['sha256']==expected_sha,'Install plan changed')
    z=read(check(pp));relative=Path(source_relative)
    need(not relative.is_absolute() and '..' not in relative.parts and relative.suffix=='.md'
         and relative.as_posix() in z['files'],'Document not in sealed source archive')
    dest=Path(destination).absolute()
    need(dest.parent==REPO/'docs' and dest.suffix=='.md' and not dest.exists() and not dest.is_symlink(),
         'Explicit fresh repository docs Markdown destination required')
    need(not any(p.is_symlink() for p in dest.parents),'Symlinked document destination parent')
    receipt=read(DESTINATION/'archive_receipt.json')
    need(receipt['status']=='COMPLETE_INSTALLED' and receipt['source_inventory_sha256']==z['source_inventory_sha256'],
         'Successful matching archive required before optional document publication')
    item=z['files'][relative.as_posix()]
    return copy_new(check(relocated(item)),dest,item)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['plan-gates','prepare','execute','publish-doc'])
    ap.add_argument('--scope');ap.add_argument('--plan');ap.add_argument('--plan-sha256');ap.add_argument('--reservation')
    ap.add_argument('--output');ap.add_argument('--source-relative');ap.add_argument('--destination')
    a=ap.parse_args()
    if a.command=='plan-gates':value=make_gate_plan(a.scope,a.output)
    elif a.command=='prepare':value=prepare(a.plan,a.plan_sha256,a.reservation,a.output)
    elif a.command=='execute':value=execute(a.plan,a.plan_sha256)
    else:value=publish_doc(a.plan,a.plan_sha256,a.source_relative,a.destination)
    print(json.dumps({'result_pin':value,'opaque_hash_read_bytes':HASH_BYTES,
                      'opaque_copy_read_bytes':COPY_BYTES,'opaque_copy_write_bytes':COPY_BYTES},sort_keys=True))


if __name__=='__main__':main()
