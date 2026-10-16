"""Freeze source-only action proposals, then finalize only after pinned unit PASS.

No source panel, forecast, model object or portfolio ledger is decoded here.
All original drafts and prior registries remain immutable. Legal facts, local
identity routing and economic execution conventions are explicitly separated.
"""
from copy import deepcopy
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

PLAN='dmv2-walkforward-action-registry-proposal-plan/v1'
PROPOSAL='dmv2-walkforward-action-registry-proposal/v1'
CONVENTIONS={
    'fractional_successor_units':'Theoretical fractional economic units; actual cash-in-lieu amount/settlement is not invented.',
    'cash_settlement':'Signed face-value cash receivable/liability at effective open, inherited zero funding; not physical settlement certification.',
    'source_adjusted_proxy':'Observed paired raw/adjusted bases; source dividends represented in adjusted prices are not added again. Source action-feed completeness is not certified.',
    'separate_distributions':'Documented separate dividends remain separately documented; none is added to merger cash by this helper.',
    'retired_new_orders':'Unfilled cash slot, no prior-holder entitlement, replacement, exclusion or reweighting.',
    'identity':'Frozen provisional identities retain original review flags. execution_cusip/isin routes are explicit local execution-only keys, not invented Intrinio identities.',
    'scope':'2020–2023 current events only; original11preperiod actions remain in the separately pinned historical-retirement registry.',
}


class RegistryError(ValueError):pass
def need(ok,message):
    if not ok:raise RegistryError(message)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def read(path):
    need(Path(path).stat().st_size<=4*1024*1024,'Bounded JSON receipt exceeds4MiB')
    return json.loads(Path(path).read_text())
def canonical(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def local_pin(root,path):
    path=Path(path).resolve();need(path.is_relative_to(root),'Evidence must be installed under its project root')
    return {'path':str(path.relative_to(root)),'sha256':sha(path),'size_bytes':path.stat().st_size}
def verify(root,pin):
    need(isinstance(pin,dict) and {'path','sha256','size_bytes'}<=set(pin),'Exact root-relative path/hash/size pin required')
    path=(root/pin['path']).resolve();need(path.is_relative_to(root),'Evidence pin escapes project root')
    need(path.is_file() and path.stat().st_size==pin['size_bytes'] and sha(path)==pin['sha256'].removeprefix('sha256:'),'Pinned evidence changed: '+str(path))
    return path
def unique_pins(root,pins):
    result={}
    for item in pins:
        path=verify(root,item);clean=local_pin(root,path)
        need(clean['path'] not in result or result[clean['path']]==clean,'Conflicting source pins')
        result[clean['path']]=clean
    return [result[key] for key in sorted(result)]
def write_new(root,path,value):
    path=Path(path).resolve();need(path.is_relative_to(root/'data/derived'),'Registry/proposal output must be private project data/derived')
    need(not path.exists(),'Preserve original proposal/registry output')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return path
def environment(plan):
    root=Path(plan['project_root']).resolve();run=(root/plan['run_directory']).resolve()
    need(run.is_relative_to(root),'Run escapes project root')
    identity=read(run/'build_identity.json');need(canonical(identity)==plan['build_identity_sha256'],'Parent build identity changed')
    need(read(verify(root,plan['configuration']))==identity['configuration'],'Frozen configuration changed')
    return root,run


def normalize(draft):
    """Proposal normalization never carries a source's ready flag forward."""
    need(isinstance(draft,dict),'Source event must be a mapping')
    need('action_type' not in draft or draft['action_type'] in {'cash_merger','stock_merger','mixed_merger'},
        'Explicit unsupported action_type requires a separately reviewed economic ledger')
    date=draft.get('effective_date')
    need(isinstance(date,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',date)
        and '2020-01-01'<=date<='2023-12-31','Current event is outside2020–2023')
    datetime.strptime(date,'%Y-%m-%d')
    need(draft.get('verified') is True or draft.get('primary_event_terms_verified') is True
        or draft.get('legal_terms_verified') is True,'Primary legal terms must already be reviewed')
    for name in ('event_id','predecessor_id','evidence_reference'):
        need(isinstance(draft.get(name),str) and draft[name].strip(),'Explicit event/source identity required')
    need(draft['evidence_reference'].startswith('https://'),'Primary evidence URL required')
    for name in ('cash_per_raw_share','stock_ratio'):
        value=draft.get(name)
        need(type(value) in (int,float) and math.isfinite(value) and value>=0,'Finite nonnegative legal terms required')
    cash,ratio=draft['cash_per_raw_share'],draft['stock_ratio'];need(cash>0 or ratio>0,'No fabricated zero retirement')
    successor=draft.get('successor_id')
    if ratio:
        need(isinstance(successor,str) and successor and successor!=draft['predecessor_id'],'Distinct successor route required')
        if not successor.startswith('sec_'):
            namespace=draft.get('identity_namespace',draft.get('successor_identity_namespace'))
            identifier=draft.get('permanent_identifier',draft.get('successor_permanent_identifier'))
            patterns={'execution_cusip':r'[A-Z0-9]{9}','execution_isin':r'[A-Z]{2}[A-Z0-9]{10}'}
            need(namespace in patterns and isinstance(identifier,str) and re.fullmatch(patterns[namespace],identifier)
                and successor==namespace+':'+identifier and draft.get('successor_provider_identity_asserted') is False,
                'External successor requires explicit nonprovider permanent-identifier namespace')
    else:need(successor is None,'Cash retirement cannot invent successor stock')
    result={key:deepcopy(draft[key]) for key in ('event_id','predecessor_id','successor_id','effective_date',
        'cash_per_raw_share','stock_ratio','evidence_reference')}
    result.update(action_type='mixed_merger' if cash and ratio else 'stock_merger' if ratio else 'cash_merger',
        verified=False,primary_legal_terms_reviewed=True,paired_unit_review_status='PENDING',
        legal_effective_date=draft.get('legal_effective_date',draft.get('legal_closing_date',draft.get('legal_date',date))),
        source_documented_event=deepcopy(draft),execution_review_conventions=deepcopy(CONVENTIONS))
    for key in ('identity_namespace','permanent_identifier','successor_identity_namespace','successor_permanent_identifier','successor_provider_identity_asserted'):
        if key in draft:result[key]=draft[key]
    return result


def propose(plan_path,plan_sha256,output):
    plan_path=Path(plan_path).resolve();need(sha(plan_path)==plan_sha256.removeprefix('sha256:'),'Proposal input plan changed')
    plan=read(plan_path);need(plan.get('schema')==PLAN,'Unknown proposal plan schema')
    root,run=environment(plan);reviewer=verify(root,plan['unit_reviewer_script'])
    need(isinstance(plan.get('events'),list) and 0<len(plan['events'])<=32,'One to32 finite source events required')
    source_pins=[local_pin(root,plan_path),local_pin(root,run/'build_identity.json'),plan['configuration'],plan['unit_reviewer_script']]
    events=[];sources=[]
    for source in plan['events']:
        draft_path=verify(root,source['draft']);draft=read(draft_path);event=normalize(draft)
        need(isinstance(source.get('evidence'),list) and source['evidence'],'Pinned primary-evidence package required')
        for item in source['evidence']:verify(root,item)
        sources.append({'event_id':event['event_id'],'draft':local_pin(root,draft_path),'evidence':source['evidence']})
        events.append(event);source_pins.extend([source['draft'],*source['evidence']])
    need(len({event['event_id'] for event in events})==len(events)
        and len({(event['effective_date'],event['predecessor_id']) for event in events})==len(events),'Duplicate/conflicting action identities')
    proposal={'schema':PROPOSAL,'status':'SOURCE_PROPOSAL_UNITS_PENDING','project_root':str(root),
        'run_directory':plan['run_directory'],'build_identity_sha256':plan['build_identity_sha256'],
        'configuration':plan['configuration'],'unit_reviewer_script':local_pin(root,reviewer),
        'normalizer_script_sha256':sha(__file__),'events':events,'event_sources':sources,
        'source_pins':unique_pins(root,source_pins),'supplemental_quote_sources':plan.get('supplemental_quote_sources',[]),
        'execution_review_conventions':deepcopy(CONVENTIONS),'source_panels_or_targets_or_pnl_decoded':False,
        'drafts_changed_or_readiness_promoted':False,'model_or_preprocessor_fits':0}
    write_new(root,output,proposal);return proposal


def finalize(proposal_path,proposal_sha256,unit_receipts,output):
    proposal_path=Path(proposal_path).resolve();need(sha(proposal_path)==proposal_sha256.removeprefix('sha256:'),'Frozen proposal changed')
    proposal=read(proposal_path);need(proposal.get('schema')==PROPOSAL and proposal.get('status')=='SOURCE_PROPOSAL_UNITS_PENDING','Unknown/incomplete source proposal')
    root,run=environment(proposal);source_pins=unique_pins(root,proposal['source_pins'])
    need(proposal.get('normalizer_script_sha256')==sha(__file__),'Normalization helper changed after proposal freezing')
    model_path=run/'models/audit.json';model=read(model_path)
    need(model.get('status')=='COMPLETE' and model.get('phase')=='walkforward_models_v1'
        and model.get('build_identity_sha256')==proposal['build_identity_sha256']
        and model.get('summary',{}).get('fitted_arms')==16 and model['summary'].get('preprocessor_fits')==8
        and model['summary'].get('months')==48 and model['summary'].get('outer_years')==[2020,2021,2022,2023],
        'All16completed fits are required before reading numerical unit-review receipts')
    reviewer=verify(root,proposal['unit_reviewer_script']);expected_drafts={source['event_id']:source['draft'] for source in proposal['event_sources']}
    expected_events={event['event_id']:event for event in proposal['events']}
    need(set(expected_events)==set(expected_drafts) and all(event.get('verified') is False for event in expected_events.values()),'Proposal readiness flags or event set changed')
    reviewed={};receipt_sources=[]
    need(isinstance(unit_receipts,list) and 0<len(unit_receipts)<=32,'Pinned PASS unit receipts required')
    for receipt_pin in unit_receipts:
        receipt_path=verify(root,receipt_pin);receipt=read(receipt_path)
        need(receipt.get('schema')=='dmv2-postfit-action-unit-review/v1' and receipt.get('status')=='PASS'
            and receipt.get('build_identity_sha256')==proposal['build_identity_sha256']
            and receipt.get('reviewer_script_sha256')==sha(reviewer)
            and receipt.get('paired_absolute_adjusted_price_residual_gate')==.0002
            and receipt.get('metadata_dates_keys_and_quality_guarded_before_numeric_projection') is True
            and receipt.get('portfolio_performance_computed') is False
            and receipt.get('targets_or_model_objects_decoded') is False
            and receipt.get('model_fits')==0 and receipt.get('event_drafts_changed_or_promoted') is False,'Unit receipt is not an exact bounded zero-fit PASS')
        budgets=receipt['budgets']
        need(0<budgets['requested_distinct_keys']<=64 and 0<budgets['returned_financial_rows']<=128
            and 0<=budgets['projected_compressed_bytes']<=64*1024*1024,'Unit-review budgets changed')
        completion=unique_pins(root,receipt['completion_and_source_pins'])
        need(local_pin(root,model_path) in completion,'Unit PASS refers to a different finalized model audit')
        receipt_sources.extend([receipt_pin,receipt['plan'],receipt['paired_bar_evidence'],*completion,*receipt['original_event_draft_pins']])
        draft_pins=unique_pins(root,receipt['original_event_draft_pins'])
        for review in receipt['event_reviews']:
            event_id=review['event_id'];need(event_id in expected_events and event_id not in reviewed,'Unknown or duplicate unit-reviewed action')
            need(review.get('status')=='PASS_UNIT_ALGEBRA' and review.get('draft_event_promoted') is False
                and review.get('portfolio_pnl_computed') is False,'Action has no passing signed unit review')
            for basis in (review['predecessor_basis'],review.get('successor_basis')):
                if basis is None:continue
                need(basis.get('pass') is True and basis.get('threshold_absolute_adjusted_price')==.0002
                    and type(basis.get('maximum_absolute_residual')) in (int,float)
                    and 0<=basis['maximum_absolute_residual']<=.0002,'Paired-unit gate was widened or failed')
            need(expected_drafts[event_id] in draft_pins,'Unit receipt reviewed a different draft source')
            reviewed[event_id]=local_pin(root,receipt_path)
    need(set(reviewed)==set(expected_events),'Every proposed event needs its own matching unit PASS')
    events=[]
    for event_id,event in expected_events.items():
        original=read(verify(root,expected_drafts[event_id]));need(normalize(original)==event,'Legal terms changed after proposal freezing')
        ready=deepcopy(event);ready.update(verified=True,paired_unit_review_status='PASS',unit_review_receipt=reviewed[event_id])
        events.append(ready)
    sources=unique_pins(root,[*source_pins,*receipt_sources,local_pin(root,proposal_path)])
    registry={'schema':'dmv2-walkforward-actions/v1','events':events,'source_pins':sources,
        'supplemental_quote_sources':proposal['supplemental_quote_sources'],
        'execution_review_conventions':deepcopy(CONVENTIONS),'build_identity_sha256':proposal['build_identity_sha256'],
        'proposal':local_pin(root,proposal_path),'normalizer_script_sha256':sha(__file__),
        'finalized_at_utc':datetime.now(timezone.utc).isoformat(),'models_or_preprocessors_fitted':False,
        'original_drafts_or_registries_or_selections_changed':False,'portfolio_performance_certified':False,
        'physical_settlement_or_identity_or_source_action_completeness_certified':False}
    write_new(root,output,registry);return registry


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    first=sub.add_parser('propose');first.add_argument('--plan',required=True);first.add_argument('--plan-sha256',required=True)
    first.add_argument('--output',required=True)
    second=sub.add_parser('finalize');second.add_argument('--proposal',required=True);second.add_argument('--proposal-sha256',required=True)
    second.add_argument('--unit-receipts',required=True,help='JSON list of root-relative path/hash/size pins');second.add_argument('--output',required=True)
    args=parser.parse_args()
    result=propose(args.plan,args.plan_sha256,args.output) if args.mode=='propose' else finalize(args.proposal,
        args.proposal_sha256,read(args.unit_receipts),args.output)
    print(json.dumps({'schema':result['schema'],'events':len(result['events']),'all_verified':all(event['verified'] for event in result['events'])}))
if __name__=='__main__':main()
