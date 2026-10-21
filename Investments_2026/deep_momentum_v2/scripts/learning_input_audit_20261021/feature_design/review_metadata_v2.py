"""Independent saved-JSON review; never opens production Parquet or prices."""
from pathlib import Path
from collections import Counter
import hashlib,json

S=Path('/private/tmp/dmv2-learning-impact-20261021')
M=S/'metadata'; D=M/'result-v2'
def j(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,message):
 if not ok: raise ValueError(message)
audit=j(D/'audit.json'); plan=j(M/'plan.json'); cfg=j(M/'config-v2.json'); reservation=j(D/'read_reservation.json')
queue=[json.loads(line) for line in (D/'eligible_missing_target_review_queue.jsonl').read_text().splitlines()]
events=j(D/'known_event_overlaps.json'); collisions=j(D/'route_collisions.json'); windows=j(D/'source_month_windows.json')
need(audit['schema']=='dmv2-learning-metadata-audit/v1' and audit['status']=='PASS_METADATA_AUDIT','Completed metadata-v2 receipt required')
need(audit['plan_sha256']==sha(M/'plan.json')==cfg['plan_sha256'],'Frozen plan binding')
need(audit['config_sha256']==sha(M/'config-v2.json') and audit['worker_sha256']==sha(M/'scripts/audit_learning_metadata.py')==cfg['worker_sha256'],'Worker/config binding')
need(audit['months']==168 and len(audit['monthly'])==168 and len(audit['annual'])==14 and len(windows)==168,'Full calendar retained')
need(audit['financial_columns_decoded']==[] and audit['financial_2024_plus_decoded'] is False and audit['new_fits']==0 and audit['models_or_inference_loaded'] is False,'Read/compute scope')
need(len(plan['years'])==14 and all(len(y['tables'])==4 for y in plan['years']) and audit['source_table_count']==56,'Complete four-table original cohorts')
projection=sum(t['projected_compressed_bytes'] for y in plan['years'] for t in y['tables'])
need(projection==audit['projected_compressed_bytes']==reservation['projected_compressed_bytes']<=cfg['projected_read_limit_bytes'],'Bounded exact projections')
for y in plan['years']:
 for t in y['tables']:
  columns=set(t['columns'])
  need(not any(c in columns for c in ('target_class','label_return','provider_adjusted_return','open','high','low','close','adj_open','adj_close','volume','factor','start_adj_close','end_adj_close')),'Forbidden financial values in metadata projection')
  need(t['rows']<=15000,'Bounded annual table')
summary=audit['summary']
for group in ('monthly','annual'):
 for field,value in summary.items():
  if isinstance(value,int) and not isinstance(value,bool): need(sum(row[field] for row in audit[group])==value,'Additive '+group+' count '+field)
  elif field=='boolean_flag_counts':
   for flag,states in value.items():
    for state,total in states.items(): need(sum(row[field][flag][state] for row in audit[group])==total,'Additive Boolean '+flag+' '+state)
for row in [summary,*audit['monthly'],*audit['annual']]:
 for flag,states in row['boolean_flag_counts'].items(): need(set(states)=={'true','false','unknown'} and sum(states.values())==row['original_rows'],'Nullable flag partition '+flag)
need(summary['original_rows']==152307 and summary['prediction_eligible_rows']==140680 and summary['target_available_rows']==140387,'Corrected cohort totals')
need(len(queue)==293==summary['prediction_eligible_rows']-summary['target_available_rows'],'Complete eligible-missing queue')
need(len({(r['source_security_id'],r['signal_date']) for r in queue})==len(queue),'Unique retained missing keys')
need(all(r['prediction_eligible'] is True and r['identity_eligible'] is True and r['target_available'] is False and r['terminal_cause_inferred'] is False for r in queue),'No manufactured admission/terminal causes')
clock={w['outcome_month']:w for w in windows}
for r in queue:
 need(all(r[k]==clock[r['outcome_month']][k] for k in ('signal_date','outcome_start','outcome_end')),'Queue source clock exact')
 matched=sorted(e['event_id'] for e in plan['known_events'] if e['predecessor_id']==r['intrinio_security_id'] and r['signal_date']<e['effective_date']<=r['outcome_end'])
 need(matched==r['known_dated_event_overlap_ids'],'Independent native/date event overlap reproduction')
need(len(events)==len(plan['known_events'])==29 and len(collisions)==audit['route_collision_signal_native_count']==622,'All event/collision diagnostics retained')
for collision in collisions:
 need(len(set(collision['source_security_ids']))>1 and collision['all_original_rows_preserved'] and collision['identity_promoted'] is False,'Nonpromoting dated collision')
for e in events:
 need(e['candidate_dependency_only'] is True and e['new_terminal_classification_performed'] is False,'Known event causal scope')
 need(len(e['overlap_rows'])==e['event_outcome_overlap_rows'],'Event overlap row count')
 for row in e['overlap_rows']:
  need(row['intrinio_security_id']==e['predecessor_id'] and row['signal_date']<e['effective_date']<=clock[row['outcome_month']]['outcome_end'],'Event original dated overlap')
counts={'queue_rows':len(queue),'missing_end_rows':sum('missing_end' in r['endpoint_failure_reasons'] for r in queue),
 'missing_start_rows':sum('missing_start' in r['endpoint_failure_reasons'] for r in queue),
 'known_event_overlap_rows':sum(bool(r['known_dated_event_overlap_ids']) for r in queue),
 'unmatched_missing_end_rows':sum('missing_end' in r['endpoint_failure_reasons'] and not r['known_dated_event_overlap_ids'] for r in queue),
 'endpoint_reasons':dict(Counter(reason for r in queue for reason in r['endpoint_failure_reasons'])),
 'return_status':dict(Counter(r['return_status'] for r in queue))}
paths=[M/'plan.json',M/'config-v2.json',M/'scripts/audit_learning_metadata.py',D/'audit.json',D/'read_reservation.json',D/'eligible_missing_target_review_queue.jsonl',D/'known_event_overlaps.json',D/'route_collisions.json',D/'source_month_windows.json']
receipt=dict(schema='dmv2-learning-metadata-independent-json-review/v1',status='PASS_SAVED_METADATA_RECONCILIATION',
 summary=counts,original_rows=152307,prediction_eligible=140680,available_labels=140387,months=168,known_events=29,route_collision_groups=622,
 nullable_identity_review_unknown_rows=summary['boolean_flag_counts']['identity_review_required']['unknown'],
 production_table_decodes=0,financial_values_or_classes_or_prices_read=False,new_fits=0,
 limits='Reconciles authenticated saved metadata counts/queue/known-date joins; does not independently decode all152307source rows or certify economic causes.',
 reviewed_sources=[dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size) for p in paths])
(S/'feature_design/metadata_v2_independent_review.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
