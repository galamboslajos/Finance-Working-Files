"""Shadow targets, causal RET and measured data dependencies; import-safe stdlib.

No fitting, inference or backtest API is exposed. Financial/Arrow imports occur
only inside execution functions after caller-supplied seals are checked.
"""
from collections import Counter
import base64
from datetime import date
import hashlib
import json
import math
from pathlib import Path

KEYS = ('source_security_id','signal_date')
PAPER15 = tuple(f'{p}{h}' for p in ('zMOM','meanMOM','stdMOM') for h in (1,3,6,9,12))
TARGET_COMPARE = ('identity_eligible','requested','paper_prediction_eligible','prediction_eligible',
 'outcome_requested','observed_return_available','target_available','target_status','return_observation_status',
 'return_available','source_return_status','label_return','target_class','outcome_month','outcome_start','outcome_end')
COVERAGE_COMPARE = ('identity_eligible','requested','paper_prediction_eligible','intrinio_security_id',
 'identity_status','source_identity_status','identity_eligibility_policy','identity_eligibility_reason',
 'identity_review_required','resolution_status','feature_status')


def need(value,message):
 if not value:raise ValueError(message)


def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):digest.update(chunk)
 return digest.hexdigest()


def pin(path):
 p=Path(path).resolve();return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)}


def checked(item):
 p=Path(item['path']);need(p.stat().st_size==item['size_bytes'] and sha(p)==item['sha256'],'Pinned artifact changed');return p


def publish(path,value):
 p=Path(path);need(not p.exists(),'Preserve prior shadow artifact');p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 return pin(p)


def canonical(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def execution_seal(authorization,action):
 need(isinstance(authorization,dict) and authorization.get('schema')=='dmv2-shadow-target-impact-execution/v1'
      and authorization.get('status')=='SEALED_BEFORE_READ' and authorization.get('module_sha256')==sha(__file__)
      and action in authorization.get('allowed_actions',[]),'Unsealed shadow target/impact execution')
 need(all(authorization.get(k) is False for k in ('preprocessor_fitting','model_fitting','inference','backtesting')),
      'Shadow target/impact cannot fit, infer or backtest')
 for item in authorization.get('pins',[]):checked(item)
 return True


def day(value):
 need(isinstance(value,str),'Date string required');d=date.fromisoformat(value)
 need(d.isoformat()==value,'Canonical ISO date required');return value


def step_month(value,offset):
 y,m=map(int,value.split('-'));n=y*12+m-1+offset;return f'{n//12:04d}-{n%12+1:02d}'


def rows_by_key(rows):
 result={}
 for row in rows:
  key=tuple(row[k] for k in KEYS);need(isinstance(key[0],str) and key[0].strip()==key[0] and key[0],'Bad source key')
  need(day(key[1])<'2024-01-01' and key not in result,'Duplicate/protected original key')
  result[key]=row
 return result


def explicit_bool(value,name):
 need(type(value) is bool,name+' must remain an explicit boolean');return value


def validate_target_metadata(rows,state,year):
 """All timing, keys and independent windows BEFORE financial Python use."""
 lookup=rows_by_key(rows);provenance=state['provenance'];observed=day(provenance['label_observed_through'])
 need(observed<'2024-01-01','Protected target observation cutoff')
 windows={}
 for window in provenance['month_windows']:
  signal=day(window['signal_date']);month=window['outcome_month']
  start,end=day(window['outcome_start']),day(window['outcome_end'])
  need(signal not in windows and month==step_month(signal[:7],1) and month.startswith(str(year)+'-')
       and start[:7]==end[:7]==month and signal<start<=end<=observed,'Invalid independent annual target window')
  windows[signal]=window
 need(sorted(w['outcome_month'] for w in windows.values())==[f'{year}-{m:02d}' for m in range(1,13)],
      'Annual target source lost a complete outcome month')
 need({key[1] for key in lookup}==set(windows),'Target metadata changed independent signal calendar')
 for key,row in lookup.items():
  need(row['outcome_month']==step_month(key[1][:7],1),'Misaligned target outcome calendar month')
  for field in ('outcome_month','outcome_start','outcome_end'):
   need(row[field]==windows[key[1]][field],'Target metadata disagrees with pinned independent window')
  for field in ('identity_eligible','requested','paper_prediction_eligible','prediction_eligible','return_available','observed_return_available','target_available'):
   explicit_bool(row[field],field)
 return lookup


def finite(value):
 return type(value) in (float,int) and math.isfinite(value)


def same_value(left,right):
 # Missing numeric states are equivalent only for feature values; masks and
 # source-status fields are compared separately without null->false coercion.
 if finite(left) or finite(right):return finite(left) and finite(right) and left==right
 if left is None or isinstance(left,float):
  return right is None or isinstance(right,float) and not math.isfinite(right)
 return type(left) is type(right) and left==right


def compare_vector(old,new):
 need(len(old)==len(new),'Feature vectors changed original denominator')
 changed=[];counts=Counter();maximum=0.;overflow=0
 for index,(a,b) in enumerate(zip(old,new)):
  af,bf=finite(a),finite(b)
  if af and bf:
   counts['shared_finite']+=1
   if a!=b:
    changed.append(index);counts['shared_finite_changed']+=1;delta=abs(b-a)
    if math.isfinite(delta):maximum=max(maximum,delta)
    else:overflow+=1
  elif af:
   counts['finite_to_missing']+=1;changed.append(index)
  elif bf:
   counts['missing_to_finite']+=1;changed.append(index)
  else:counts['both_nonfinite']+=1
 return {'rows':len(old),**{k:counts[k] for k in ('shared_finite','shared_finite_changed','finite_to_missing','missing_to_finite','both_nonfinite')},
         'max_absolute_shared_finite_change':maximum,'overflowing_absolute_differences':overflow},changed


def compare_target_rows(old_rows,new_rows):
 old,new=rows_by_key(old_rows),rows_by_key(new_rows)
 need(set(old)==set(new),'Shadow targets changed original source/signal denominator')
 effects=[];counts=Counter()
 for key in sorted(old,key=lambda k:(k[1],k[0])):
  a,b=old[key],new[key]
  need(all(a[k]==b[k] for k in ('outcome_month','outcome_start','outcome_end')),'Shadow target windows changed')
  need('2010-01'<=a['outcome_month']<='2023-12' and day(a['outcome_end'])<'2024-01-01','Target outside development scope')
  for row in (a,b):
   for field in ('identity_eligible','requested','paper_prediction_eligible','prediction_eligible','return_available','observed_return_available','target_available'):
    explicit_bool(row[field],field)
   need(row['prediction_eligible']==(row['identity_eligible'] and row['paper_prediction_eligible']),'Target identity/Paper15 relation changed')
   need(not row['target_available'] or row['prediction_eligible'],'Label outside signal prediction cohort')
   need(row['target_available']==(type(row['target_class']) is int and 1<=row['target_class']<=10),'Canonical class/availability mismatch')
   need(not row['observed_return_available'] or finite(row['label_return']) and row['label_return']>=-1,'Observed class return must be finite and valid')
  changed=[field for field in TARGET_COMPARE if not same_value(a.get(field),b.get(field))]
  effects.append({'source_security_id':key[0],'signal_date':key[1],'outcome_month':a['outcome_month'],
    'outcome_start':a['outcome_start'],'outcome_end':a['outcome_end'],
    'old_prediction_eligible':a['prediction_eligible'],'new_prediction_eligible':b['prediction_eligible'],
    'old_target_available':a['target_available'],'new_target_available':b['target_available'],
    'old_fit_mask':a['prediction_eligible'] and a['target_available'],'new_fit_mask':b['prediction_eligible'] and b['target_available'],
    'prediction_mask_changed':a['prediction_eligible']!=b['prediction_eligible'],
    'fit_mask_changed':(a['prediction_eligible'] and a['target_available'])!=(b['prediction_eligible'] and b['target_available']),
    'target_class_changed':not same_value(a['target_class'],b['target_class']),
    'label_return_changed':not same_value(a['label_return'],b['label_return']),
    'target_metadata_changed':bool(changed),'changed_target_fields':changed,
    'paper15_X_changed':False,'expanded_X_changed':False,'registered_feature_cells_changed':0})
  counts.update(changed)
 return effects,{'original_rows':len(effects),'target_rows_changed':sum(r['target_metadata_changed'] for r in effects),
                'changed_field_counts':dict(sorted(counts.items()))}


def _key_digest(rows,predicate):
 return canonical([[r[k] for k in KEYS] for r in rows if predicate(r)])


def partition_impact(effects,*,start,end,feature_set,role,annual_summaries=None,require_complete_month_grid=False):
 need(feature_set in ('paper15','expanded') and role in ('training','prediction'),'Unknown model input role')
 start,end=day(start),day(end);need(start<=end<'2024-01-01','Protected/reversed fold bounds')
 rows=[r for r in effects if start<=r['outcome_start'] and r['outcome_end']<=end]
 rows.sort(key=lambda r:(r['signal_date'],r['source_security_id']))
 need(rows,'No original rows for fold partition')
 if require_complete_month_grid:
  expected=[];month=start[:7]
  while month<=end[:7]:expected.append(month);month=step_month(month,1)
  need(sorted({r['outcome_month'] for r in rows})==expected,'Fold lost complete outcome-month cross-sections')
 prefix='paper15' if feature_set=='paper15' else 'expanded'
 oldmask='old_fit_mask' if role=='training' else 'old_prediction_eligible'
 newmask='new_fit_mask' if role=='training' else 'new_prediction_eligible'
 shared=[r for r in rows if r[oldmask] and r[newmask]]
 masks_changed=sum(r[oldmask]!=r[newmask] for r in rows)
 x_changed=sum(r[prefix+'_X_changed'] for r in shared)
 y_changed=sum(r['target_class_changed'] for r in rows if r['old_fit_mask'] and r['new_fit_mask'])
 score_changed=sum(r['old_fit_mask']!=r['new_fit_mask'] for r in rows)
 result={'role':role,'feature_set':feature_set,'start':start,'end':end,'original_rows':len(rows),
  'minimum_signal_date':min(r['signal_date'] for r in rows),'minimum_outcome_start':min(r['outcome_start'] for r in rows),
  'maximum_outcome_end':max(r['outcome_end'] for r in rows),
  'original_keys_sha256':_key_digest(rows,lambda r:True),
  'old_selected_rows':sum(r[oldmask] for r in rows),'new_selected_rows':sum(r[newmask] for r in rows),
  'old_selected_keys_sha256':_key_digest(rows,lambda r:r[oldmask]),'new_selected_keys_sha256':_key_digest(rows,lambda r:r[newmask]),
  'selected_mask_changed_rows':masks_changed,'shared_selected_rows':len(shared),'shared_selected_X_changed_rows':x_changed,
  'shared_observed_y_changed_rows':y_changed,'score_mask_changed_rows':score_changed,
  'selected_inputs_changed':bool(masks_changed or x_changed or role=='training' and y_changed),
  'evaluation_labels_changed':bool(score_changed or y_changed),
  'preprocessor_or_estimator_fitted':False}
 if annual_summaries is not None:
  years=list(range(int(start[:4]),int(end[:4])+1))
  need(start.endswith('-01-01') and end.endswith('-12-31'),'Annual compositional hashes require complete outcome years')
  for side in ('old','new'):
   def annual(y):
    item=annual_summaries.get(y,annual_summaries.get(str(y)));need(item is not None,'Missing annual input hash');return item
   result[side+'_raw_X_compositional_sha256']=canonical([(y,annual(y)['raw_input_hashes'][side][role][feature_set]) for y in years])
   result[side+'_observed_y_compositional_sha256']=canonical([(y,annual(y)['raw_input_hashes'][side]['training_y']) for y in years])
  result['raw_X_fingerprint_changed']=result['old_raw_X_compositional_sha256']!=result['new_raw_X_compositional_sha256']
  result['observed_y_fingerprint_changed']=result['old_observed_y_compositional_sha256']!=result['new_observed_y_compositional_sha256']
 return result


def fold_impacts(effects,chronology,*,annual_summaries=None):
 rf=[];baseline=[]
 for fold in chronology['folds']:
  year=fold['outer_outcome_year'];need(year in (2020,2021,2022,2023),'Unknown outer year')
  partitions=[('outer',f'{year}_outer',fold,{'start':f'{year}-01-01','end':f'{year}-12-31'})]
  partitions += [('inner',s['split_id'],s,{'start':s['validation_outcome_start'],'end':s['validation_outcome_end']}) for s in fold['inner_splits']]
  for stage,split,train,predict in partitions:
   tr=partition_impact(effects,start=train['train_outcome_start'],end=train['train_outcome_end'],feature_set='expanded',role='training',annual_summaries=annual_summaries,require_complete_month_grid=True)
   pr=partition_impact(effects,**predict,feature_set='expanded',role='prediction',annual_summaries=annual_summaries,require_complete_month_grid=True)
   need(tr['end']<pr['start'],'Train and evaluation outcomes overlap')
   need(tr['maximum_outcome_end']<=pr['minimum_signal_date'] and tr['maximum_outcome_end']<pr['minimum_outcome_start'],
        'Training labels overlap prediction signal or outcome interval')
   for result in (tr,pr):rf.append({'outer_year':year,'stage':stage,'split_id':split,**result})
  for estimator in ('rf','xgboost'):
   for feature_set in ('paper15','expanded'):
    for role,start,end in (('training',fold['train_outcome_start'],fold['train_outcome_end']),('prediction',f'{year}-01-01',f'{year}-12-31')):
     baseline.append({'outer_year':year,'estimator':estimator,'arm':estimator+'_'+feature_set,
       **partition_impact(effects,start=start,end=end,feature_set=feature_set,role=role,annual_summaries=annual_summaries,require_complete_month_grid=True)})
 return {'rf_tuning_partitions':rf,'baseline_outer_partitions':baseline,'preprocessor_fits':0,'estimator_fits':0,'inference_runs':0,'backtests':0}


def build_targets(feature_panel,monthly_returns,directory,*,authorization):
 execution_seal(authorization,'targets')
 from dmv2.targets import build_monthly_targets,verify_target_panel
 import pyarrow.parquet as pq
 cutoff=max(w.outcome_end for w in monthly_returns.windows)
 need(cutoff<'2024-01-01','Protected target build cutoff')
 result=build_monthly_targets(feature_panel.coverage,monthly_returns,label_observed_through=cutoff)
 verify_target_panel(result)
 destination=Path(directory);destination.mkdir(parents=True,exist_ok=True);artifacts={}
 for name,table in (('targets',result.table),('target_diagnostics',result.diagnostics)):
  path=destination/(name+'.parquet');need(not path.exists(),'Preserve prior target table');pq.write_table(table,path,compression='zstd');artifacts[name]=pin(path)
 state={'provenance':result.provenance,'artifact_sha256':result.artifact_sha256,
        'schemas':{name:base64.b64encode(table.schema.serialize()).decode('ascii') for name,table in (('targets',result.table),('target_diagnostics',result.diagnostics))}}
 artifacts['target_state']=publish(destination/'target_state.json',state)
 execution_seal(authorization,'targets')
 return result,artifacts


def build_ret(panels,source_year_pins,signals,directory,*,authorization):
 execution_seal(authorization,'ret')
 import pyarrow as pa
 import pyarrow.compute as pc
 import pyarrow.parquet as pq
 from dmv2.targets import verify_target_panel,target_month_fingerprint
 from dmv2.ret_backtest import build_shared_payoffs
 months={};windows={};fingerprints={};diagnostic_pins={};policy=None;expected_signals=[]
 for panel in panels:
  verify_target_panel(panel,through=min(panel.provenance['label_observed_through'],'2023-11-30'))
  need(policy in (None,panel.provenance['policy_sha256']),'Target policy differs across shadow years');policy=panel.provenance['policy_sha256']
  for window in panel.provenance['month_windows']:
   month=window['outcome_month']
   if '2020-01'<=month<='2023-12':expected_signals.append(window['signal_date'])
   if month>'2023-11':continue
   need(month not in months and window['outcome_end']<='2023-11-30','Duplicate/uncompleted RET source month')
   selected=panel.table.filter(pc.equal(panel.table['outcome_month'],month))
   diagnostic=panel.diagnostics.filter(pc.equal(panel.diagnostics['outcome_month'],month))
   need(target_month_fingerprint(selected)==panel.provenance['month_target_fingerprints'][month]
        and target_month_fingerprint(diagnostic)==panel.provenance['month_diagnostic_fingerprints'][month],'Shadow month fingerprint changed')
   groups={k:[] for k in range(1,11)}
   for row in selected.select(['target_available','target_class','label_return']).to_pylist():
    if row['target_available']:
     need(type(row['target_class']) is int and row['target_class'] in groups and finite(row['label_return']) and row['label_return']>=-1,'Invalid released RET class return')
     groups[row['target_class']].append(row['label_return'])
   eligible=int(pc.sum(pc.cast(selected['prediction_eligible'],pa.int64())).as_py())
   months[month]=[{'outcome_month':month,'outcome_end':window['outcome_end'],'class_id':k,
     'original_member_count':selected.num_rows,'prediction_eligible_count':eligible,'member_count':len(values),
     'class_mean_return':math.fsum(v/len(values) for v in values) if values else None,
     'status':'available' if values else 'empty_realized_class'} for k,values in groups.items()]
   windows[month]=window;fingerprints[month]=panel.provenance['month_target_fingerprints'][month];diagnostic_pins[month]=panel.provenance['month_diagnostic_fingerprints'][month]
 need(sorted(months)==[step_month('2010-01',i) for i in range(167)],'RET source must contain exact167 completed months')
 need(len(expected_signals)==len(set(expected_signals))==48 and sorted(signals)==sorted(expected_signals),
      'RET signals must equal the exact48 independently pinned nominal target windows')
 source={'monthly':months,'windows':windows,'target_fingerprints':fingerprints,'diagnostic_fingerprints':diagnostic_pins,'source_target_policy_sha256':policy}
 payoffs=build_shared_payoffs(source,signals)
 dest=Path(directory);need(not dest.exists(),'Preserve prior shadow RET');dest.mkdir(parents=True)
 artifacts={};grid=[];values=[]
 for signal,panel in sorted(payoffs.items()):
  sub=dest/signal;sub.mkdir()
  for name,table in (('payoffs',panel.table),('monthly',panel.monthly)):
   path=sub/(name+'.parquet');pq.write_table(table,path,compression='zstd');artifacts[str(path.relative_to(dest))]=pin(path)
  artifacts[str((sub/'payoff_state.json').relative_to(dest))]=publish(sub/'payoff_state.json',{'provenance':panel.provenance,'fingerprint':panel.fingerprint})
  grid.extend({'signal_date':signal,**r} for r in panel.monthly.to_pylist());values.extend({'signal_date':signal,**r} for r in panel.table.to_pylist())
 means_path=dest/'source_month_class_means.parquet';pq.write_table(pa.Table.from_pylist([r for m in sorted(months) for r in months[m]]),means_path,compression='zstd');artifacts[means_path.name]=pin(means_path)
 summary={'schema':'dmv2-shadow-ret-build/v1','signals':len(payoffs),'source_months':167,'grid_cells':len(grid),'payoffs':len(values),
  'available_vectors':sum(p.provenance['vector_available'] for p in payoffs.values()),'source_year_pins':source_year_pins,
  'source_month_target_fingerprints':fingerprints,'preprocessor_fits':0,'estimator_fits':0,'inference_runs':0,'backtests':0}
 artifacts['summary.json']=publish(dest/'summary.json',summary)
 execution_seal(authorization,'ret')
 return payoffs,artifacts,summary


def value_digest(values,indices):
 import struct
 encoded=[]
 for value in values:
  need(value is None or type(value) in (float,int),'Predictors must remain numeric or missing, never booleans/text')
  encoded.append(b'\x01'+struct.pack('>d',0.0 if value==0 else float(value)) if finite(value) else b'\x00'+b'\0'*8)
 return hashlib.sha256(b''.join(encoded[i] for i in indices)).hexdigest()


def plan_annual_comparison(year,old,new,registered_names,predictor_names,*,authorization):
 """Footer-only plan. Parent seals this exact result before any value read."""
 execution_seal(authorization,'footer_plan')
 import pyarrow.parquet as pq
 need(type(year) is int and 2010<=year<=2023,'Annual comparison year outside scope')
 registered=list(registered_names);predictors=list(predictor_names)
 need(len(registered)==len(set(registered)) and len(predictors)==len(set(predictors))==715
      and set(PAPER15)<=set(predictors)<=set(registered),'Frozen predictor/feature dictionary differs')
 sources={};cost=0;hash_cost=0;footer_cost=0
 projections={'features':[*KEYS,*registered],'feature_coverage':[*KEYS,*COVERAGE_COMPARE],
              'targets':[*KEYS,*TARGET_COMPARE]}
 for label,paths in (('old',old),('new',new)):
  sources[label]={}
  for name,columns in projections.items():
   path=Path(paths[name]).resolve();item=pin(path);pf=pq.ParquetFile(path)
   need(set(columns)<=set(pf.schema_arrow.names),'Annual source lacks fixed comparison fields')
   if name=='features':need(set(pf.schema_arrow.names)==set(columns),'Feature dictionary/schema names changed')
   projected=sum(pf.metadata.row_group(rg).column(ci).total_compressed_size
      for rg in range(pf.metadata.num_row_groups) for ci,n in enumerate(pf.schema_arrow.names) if n in columns)
   footer_upper=65536+pf.metadata.serialized_size+8
   sources[label][name]={'pin':item,'columns':columns,'rows':pf.metadata.num_rows,
     'types':{n:str(pf.schema_arrow.field(n).type) for n in columns},
     'projected_compressed_bytes':projected,'footer_upper_bytes':footer_upper}
   cost+=projected;hash_cost+=2*item['size_bytes'];footer_cost+=footer_upper
  state_path=Path(paths['target_state']).resolve();sources[label]['target_state']={'pin':pin(state_path)}
  state=json.loads(state_path.read_text());need(state['provenance']['policy_id']=='monthly_paper_qualified_midrank_deciles_v1','Unexpected target policy')
  hash_cost+=2*state_path.stat().st_size
 return {'schema':'dmv2-shadow-annual-impact-read-plan/v1','year':year,'sources':sources,
  'registered_names':registered,'predictor_names':predictors,'paper15_names':list(PAPER15),
  'projected_compressed_bytes':cost,'hash_integrity_bytes':hash_cost,'metadata_footer_upper_bytes':footer_cost,
  'module_sha256':sha(__file__),'preprocessor_fits':0,'estimator_fits':0,'inference_runs':0,'backtests':0}


def compare_annual(plan,*,authorization):
 execution_seal(authorization,'impact')
 need(plan['schema']=='dmv2-shadow-annual-impact-read-plan/v1' and plan['module_sha256']==sha(__file__)
      and authorization.get('annual_plan_sha256')==canonical(plan),'Annual impact read plan differs from sealed reservation')
 import pyarrow.parquet as pq
 import pyarrow as pa
 source=plan['sources'];loaded={};physical=0;keys_order=None;coverage_changes=Counter()
 for side in ('old','new'):
  specs=source[side]
  for record in specs.values():checked(record['pin'])
  tables={name:pq.ParquetFile(specs[name]['pin']['path']).read(columns=specs[name]['columns'])
          for name in ('feature_coverage','targets')}
  for name in ('feature_coverage','targets'):physical+=specs[name]['projected_compressed_bytes']
  target_meta=tables['targets'].select([n for n in specs['targets']['columns'] if n not in ('label_return','target_class')]).to_pylist()
  state=json.loads(checked(specs['target_state']['pin']).read_text())
  target_lookup=validate_target_metadata(target_meta,state,plan['year'])
  coverage=tables['feature_coverage'].to_pylist()
  feature_pf=pq.ParquetFile(specs['features']['pin']['path'])
  feature_keys=feature_pf.read(columns=list(KEYS)).to_pylist()
  cover_lookup=rows_by_key(coverage);feature_lookup=rows_by_key(feature_keys)
  need(set(target_lookup)==set(cover_lookup)==set(feature_lookup),'Annual feature/target/cohort denominators disagree')
  order=sorted(target_lookup,key=lambda k:(k[1],k[0]))
  if keys_order is None:keys_order=order
  need(keys_order==order,'Parent/shadow original membership keys differ')
  for key in order:
   need(all(target_lookup[key][f]==cover_lookup[key][f] for f in ('identity_eligible','requested','paper_prediction_eligible')),
        'Feature and target signal-time eligibility differ')
  loaded[side]={'target_table':tables['targets'],'coverage':cover_lookup,'feature_positions':{k:i for i,k in enumerate(tuple(r[k] for k in KEYS) for r in feature_keys)},
                'feature_pf':feature_pf}
 # Both source versions' complete metadata, calendars and cross-table keysets
 # are now validated. Only now convert these admitted annual financial rows.
 for side in ('old','new'):
  targets=loaded[side].pop('target_table').to_pylist()
  loaded[side]['target_rows']=targets;loaded[side]['target_lookup']=rows_by_key(targets)
 effects,summary=compare_target_rows(loaded['old']['target_rows'],loaded['new']['target_rows'])
 need([tuple(r[k] for k in KEYS) for r in effects]==keys_order,'Impact row order differs')
 for effect,key in zip(effects,keys_order):
  fields=[f for f in COVERAGE_COMPARE if not same_value(loaded['old']['coverage'][key].get(f),loaded['new']['coverage'][key].get(f))]
  effect['changed_coverage_fields']=fields;coverage_changes.update(fields)
  effect['coverage_metadata_changed']=bool(fields)
  effect['old_intrinio_security_id']=loaded['old']['coverage'][key]['intrinio_security_id']
  effect['new_intrinio_security_id']=loaded['new']['coverage'][key]['intrinio_security_id']
 masks={side:{'training':[i for i,r in enumerate(effects) if r[side+'_fit_mask']],
              'prediction':[i for i,r in enumerate(effects) if r[side+'_prediction_eligible']]}
        for side in ('old','new')}
 columns=[];column_hashes={side:{role:{} for role in ('training','prediction')} for side in ('old','new')}
 predictors=set(plan['predictor_names']);paper=set(PAPER15)
 for name in plan['registered_names']:
  vectors={}
  for side in ('old','new'):
   raw=loaded[side]['feature_pf'].read(columns=[name]).column(0).to_pylist()
   vectors[side]=[raw[loaded[side]['feature_positions'][key]] for key in keys_order]
   # Physical projection charged once in the frozen full-feature read; each
   # feature column is read exactly once, keys separately exactly once.
   if name in predictors:
    for role in ('training','prediction'):
     column_hashes[side][role][name]=value_digest(vectors[side],masks[side][role])
  stats,changed=compare_vector(vectors['old'],vectors['new'])
  columns.append({'year':plan['year'],'feature':name,
   'old_type':source['old']['features']['types'][name],'new_type':source['new']['features']['types'][name],**stats})
  for index in changed:
   effects[index]['registered_feature_cells_changed']+=1
   if name in predictors:effects[index]['expanded_X_changed']=True
   if name in paper:effects[index]['paper15_X_changed']=True
 for side in ('old','new'):physical+=source[side]['features']['projected_compressed_bytes']
 need(physical==plan['projected_compressed_bytes'],'Comparison read projection differs from frozen reservation')
 annual_hashes={}
 for side in ('old','new'):
  annual_hashes[side]={}
  for role in ('training','prediction'):
   selected_keys=[[effects[i][k] for k in KEYS] for i in masks[side][role]]
   annual_hashes[side][role]={}
   for feature_set,names in (('paper15',list(PAPER15)),('expanded',plan['predictor_names'])):
    annual_hashes[side][role][feature_set]=canonical({'keys':selected_keys,'columns':names,
      'column_values':[(name,column_hashes[side][role][name]) for name in names]})
  y=[loaded[side]['target_lookup'][keys_order[i]]['target_class'] for i in masks[side]['training']]
  annual_hashes[side]['training_y']=canonical({'keys':[[effects[i][k] for k in KEYS] for i in masks[side]['training']], 'classes':y})
 summary.update(year=plan['year'],feature_columns=len(columns),features_with_any_value_change=sum(any(r[k] for k in ('shared_finite_changed','finite_to_missing','missing_to_finite')) for r in columns),
  paper15_X_changed_rows=sum(r['paper15_X_changed'] for r in effects),expanded_X_changed_rows=sum(r['expanded_X_changed'] for r in effects),
  coverage_changed_field_counts=dict(sorted(coverage_changes.items())),projected_compressed_bytes=physical,
  raw_input_hashes=annual_hashes,hash_interpretation='Compositional exact raw numeric inputs and explicit masks; not a fitted/preprocessed matrix or proof that predictions would change.',
  schema_changes=[{'table':name,'old':source['old'][name]['types'],'new':source['new'][name]['types']} for name in ('features','feature_coverage','targets')
    if source['old'][name]['types']!=source['new'][name]['types']])
 for side in ('old','new'):
  for record in source[side].values():checked(record['pin'])
 execution_seal(authorization,'impact')
 return {'summary':summary,'row_effects':effects,'feature_changes':columns}


def validate_ret_rows(grids,payoffs,*,expected_signals=48,expected_signal_dates=None):
 grouped={};vectors={}
 for row in grids:
  signal=day(row['signal_date']);need('2020-01'<=step_month(signal[:7],1)<='2023-12','RET state outside nominal outcomes')
  need(type(row['class_id']) is int and 1<=row['class_id']<=10,'Invalid RET class ID')
  grouped.setdefault(signal,[]).append(row)
 for row in payoffs:vectors.setdefault(day(row['signal_date']),[]).append(row)
 need(len(grouped)==expected_signals and set(grouped)==set(vectors),'RET state count/key mismatch')
 need(len({step_month(s[:7],1) for s in grouped})==len(grouped),'Multiple RET signals for one outcome month')
 if expected_signal_dates is not None:
  need(set(grouped)==set(expected_signal_dates) and len(expected_signal_dates)==expected_signals,
       'RET signals differ from pinned independent nominal dates')
 if expected_signals==48:
  need(sorted(step_month(s[:7],1) for s in grouped)==[step_month('2020-01',i) for i in range(48)],
       'RET signals lost a nominal outcome calendar month')
 for signal,rows in grouped.items():
  expected=[(step_month(signal[:7],-119+i),k) for i in range(120) for k in range(1,11)]
  rows.sort(key=lambda r:(r['outcome_month'],r['class_id']))
  need([(r['outcome_month'],r['class_id']) for r in rows]==expected,'RET must retain exact120×10 monthly grid')
  available=all(r['status']=='available' for r in rows)
  for row in rows:
   if row['outcome_end'] is not None:
    need(day(row['outcome_end'])[:7]==row['outcome_month'],'RET endpoint is outside its outcome calendar month')
   if row['status']=='available':
    need(row['outcome_end'] is not None and day(row['outcome_end'])<=signal and row['member_count']>0
         and finite(row['class_mean_return']),'Future/nonfinite/empty RET cell')
   else:need(row['class_mean_return'] is None,'Unavailable RET mean cannot be filled')
  values=sorted(vectors[signal],key=lambda r:r['class_id'])
  need([r['class_id'] for r in values]==list(range(1,11)),'RET vector missing/duplicate class')
  for row in values:
   if available:
    expected_value=math.fsum(r['class_mean_return']/120 for r in rows if r['class_id']==row['class_id'])
    need(row['status']=='available' and row['payoff']==expected_value,'RET differs from equal-calendar-month payoff')
   else:need(row['status']=='withheld_incomplete_120_month_class_grid' and row['payoff'] is None,'Entire incomplete RET vector must be withheld')
 return grouped,vectors


def compare_ret_rows(old_grids,new_grids,old_payoffs,new_payoffs,*,expected_signals=48,expected_signal_dates=None):
 old,ov=validate_ret_rows(old_grids,old_payoffs,expected_signals=expected_signals,expected_signal_dates=expected_signal_dates)
 new,nv=validate_ret_rows(new_grids,new_payoffs,expected_signals=expected_signals,expected_signal_dates=expected_signal_dates)
 need(set(old)==set(new),'Shadow RET changed nominal signals')
 signals=[];changed_cells=[];changed_payoffs=[]
 for signal in sorted(old):
  cell_changes=0;mean_changes=0
  for a,b in zip(old[signal],new[signal]):
   fields=[k for k in ('outcome_end','original_member_count','prediction_eligible_count','member_count','class_mean_return','status') if not same_value(a.get(k),b.get(k))]
   if fields:
    cell_changes+=1;mean_changes+=int('class_mean_return' in fields)
    changed_cells.append({'signal_date':signal,'outcome_month':a['outcome_month'],'class_id':a['class_id'],'changed_fields':fields,
      'old_class_mean_return':a['class_mean_return'],'new_class_mean_return':b['class_mean_return'],
      'old_member_count':a['member_count'],'new_member_count':b['member_count']})
  vector_changes=0
  for a,b in zip(sorted(ov[signal],key=lambda r:r['class_id']),sorted(nv[signal],key=lambda r:r['class_id'])):
   if a['status']!=b['status'] or not same_value(a['payoff'],b['payoff']):
    vector_changes+=1;changed_payoffs.append({'signal_date':signal,'class_id':a['class_id'],
      'old_payoff':a['payoff'],'new_payoff':b['payoff'],'old_status':a['status'],'new_status':b['status']})
  signals.append({'signal_date':signal,'grid_cells':1200,'changed_cells':cell_changes,'class_mean_value_changed_cells':mean_changes,
   'changed_class_payoffs':vector_changes,'original_available':all(r['status']=='available' for r in ov[signal]),
   'shadow_available':all(r['status']=='available' for r in nv[signal]),
   'original_first_month':old[signal][0]['outcome_month'],'last_completed_month':signal[:7],
   'effect_on_fitted_classifier':'none_from_RET_alone','probabilities_or_portfolios_recomputed':False})
 return {'schema':'dmv2-shadow-RET-impact/v1','signals':signals,'changed_grid_cells':changed_cells,
  'changed_payoffs':changed_payoffs,'summary':{'signal_states':len(signals),'grid_cells':1200*len(signals),
   'signals_with_changed_payoffs':sum(r['changed_class_payoffs']>0 for r in signals),
   'changed_class_payoffs':len(changed_payoffs),'changed_grid_cells':len(changed_cells),
   'source_return_policy_changed':False,'predictions_recomputed':False,'backtests':0}}
