"""Pure BK candidate diagnostics; no I/O, promotion, features or models."""
from collections import Counter
from datetime import date
import math
from numbers import Real

NATIVE='sec_zq1J05';START='2007-07-02';END='2023-12-31'
KEYS=('intrinio_security_id','trading_date')
RAW=('open','high','low','close');ADJ=tuple('adj_'+x for x in RAW)
FINANCIAL=(*RAW,'volume',*ADJ,'adj_volume','factor','split_ratio','dividend')
META=('intrinio_security_id','company_id','requested_identifier','ticker','composite_ticker',
 'figi','composite_figi','share_class_figi','exchange_mic','trading_date','frequency','intraperiod',
 'raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid','price_quality_status','retrieved_at_utc',
 'source_run_id','source_page_index','source_page_hash','source_relative_path')
COLUMNS=(*META,*FINANCIAL)
TOLERANCE=0.0002

class AuditError(ValueError):pass

def need(value,message):
 if not bool(value):raise AuditError(message)

def day(value):
 need(isinstance(value,str),'Explicit date string required')
 try:d=date.fromisoformat(value)
 except ValueError as error:raise AuditError('Invalid date') from error
 need(d.isoformat()==value,'Noncanonical date');return value

def number(value,positive=False,nonnegative=False):
 if not isinstance(value,Real) or isinstance(value,bool) or not math.isfinite(value):return None
 if positive and value<=0 or nonnegative and value<0:return None
 return float(value)

def relative(a,b):return abs(a/b-1) if a is not None and b is not None and b!=0 else None

def bracket(row,fields):
 values=[number(row[k],positive=True) for k in fields]
 if any(v is None for v in values):return False
 o,h,l,c=values;return l<=min(o,c)<=max(o,c)<=h

def audit_rows(rows,sessions):
 """Only already-filtered postmerger XNYS rows; metadata precedes numeric use."""
 sessions=[day(d) for d in sessions];need(sessions==sorted(set(sessions)),'Exact ordered unique XNYS calendar required')
 permitted=[d for d in sessions if START<=d<=END];known=set(permitted);seen=set()
 for row in rows:
  need(set(COLUMNS)<=set(row),'Frozen source columns missing')
  d=day(row['trading_date']);key=(row['intrinio_security_id'],d)
  need(key[0]==NATIVE and START<=d<=END and d in known,'Unapproved native/date/non-session before numeric conversion')
  need(key not in seen,'Duplicate native/date key');seen.add(key)
 ordered=sorted(rows,key=lambda r:r['trading_date']);quality=[];lookup={r['trading_date']:r for r in ordered}
 for row in ordered:
  flags=[];raw_ok=bracket(row,RAW);adj_ok=bracket(row,ADJ)
  if not raw_ok:flags.append('raw_price_or_OHLC_bracket_invalid')
  if not adj_ok:flags.append('adjusted_price_or_OHLC_bracket_invalid')
  for field in ('volume','adj_volume'):
   if number(row[field],nonnegative=True) is None:flags.append(field+'_invalid')
  if row['frequency']!='daily' or row['intraperiod'] is not False:flags.append('frequency_or_completion_invalid')
  if row['price_quality_status'] not in ('accepted','ok'):flags.append('source_quality_not_accepted')
  if row['raw_ohlc_bounds_valid'] is not raw_ok:flags.append('source_raw_bounds_flag_disagrees')
  if row['adjusted_ohlc_bounds_valid'] is not adj_ok:flags.append('source_adjusted_bounds_flag_disagrees')
  c=number(row['close'],positive=True);a=number(row['adj_close'],positive=True)
  scale=a/c if a is not None and c is not None else None
  residuals=[relative(number(row['adj_'+f],positive=True)/number(row[f],positive=True),scale)
      if number(row['adj_'+f],positive=True) is not None and number(row[f],positive=True) is not None else None for f in RAW]
  residual=max(residuals) if all(r is not None for r in residuals) else None
  if residual is None or residual>TOLERANCE:flags.append('raw_adjusted_OHLC_scales_disagree')
  factor=number(row['factor'],positive=True)
  if factor is None:flags.append('source_factor_missing_or_invalid')
  for field in ('split_ratio','dividend'):
   if row[field] is not None and number(row[field]) is None:flags.append(field+'_nonfinite_or_invalid')
  quality.append({'intrinio_security_id':NATIVE,'trading_date':row['trading_date'],'flags':flags,
   'field_flags_count':len(flags),'derived_adj_over_raw_close':scale,'OHLC_max_scale_relative_residual':residual,
   'source_factor_direct_relative_residual':relative(factor,scale),
   'source_factor_inverse_relative_residual':relative(factor,1/scale if scale else None),
   'raw_volume':number(row['volume'],nonnegative=True),'adjusted_volume':number(row['adj_volume'],nonnegative=True),
   'split_ratio_source':number(row['split_ratio']),'dividend_source':number(row['dividend']),
   'split_ratio_is_null':row['split_ratio'] is None,'dividend_is_null':row['dividend'] is None})
 jumps=[]
 for previous,current in zip(permitted,permitted[1:]):
  if previous not in lookup or current not in lookup:continue # never bridge a missing true session
  p,r=lookup[previous],lookup[current];praw=number(p['close'],positive=True);raw=number(r['close'],positive=True)
  padj=number(p['adj_close'],positive=True);adj=number(r['adj_close'],positive=True)
  rr=raw/praw-1 if praw is not None and raw is not None else None
  ar=adj/padj-1 if padj is not None and adj is not None else None
  ps=padj/praw if padj and praw else None;cs=adj/raw if adj and raw else None
  scale_change=cs/ps-1 if ps and cs else None
  split=number(r['split_ratio']);div=number(r['dividend'])
  if (rr is not None and abs(rr)>.5) or (ar is not None and abs(ar)>.5) or (scale_change is not None and abs(scale_change)>.05):
   jumps.append({'intrinio_security_id':NATIVE,'previous_session':previous,'trading_date':current,
    'raw_close_change':rr,'adjusted_close_change':ar,'derived_price_scale_change':scale_change,
    'split_ratio_source':split,'dividend_source':div,'source_action_marker_present':(split not in (None,0.,1.)) or (div not in (None,0.)),
    'verified_economic_explanation':False,'synthetic_fill_or_exclusion':False})
 def maximum(field):
  values=[r[field] for r in quality if r[field] is not None];return max(values) if values else None
 counts=Counter(flag for row in quality for flag in row['flags'])
 summary={'financial_rows':len(rows),'first_financial_date':ordered[0]['trading_date'] if ordered else None,
  'last_financial_date':ordered[-1]['trading_date'] if ordered else None,'expected_postmerger_XNYS_sessions':len(permitted),
  'missing_postmerger_sessions':[d for d in permitted if d not in lookup],
  'rows_with_field_flags':sum(bool(r['flags']) for r in quality),'field_flag_counts':dict(sorted(counts.items())),
  'max_OHLC_common_scale_relative_residual':maximum('OHLC_max_scale_relative_residual'),
  'source_factor_direct_max_relative_residual':maximum('source_factor_direct_relative_residual'),
  'source_factor_inverse_max_relative_residual':maximum('source_factor_inverse_relative_residual'),
  'adjusted_volume_assumed_inverse_of_dividend_inclusive_price_factor':False,
  'source_factor_semantics_certified':False,'split_ratio_null_rows':sum(r['split_ratio_is_null'] for r in quality),
  'dividend_null_rows':sum(r['dividend_is_null'] for r in quality),'flagged_consecutive_session_discontinuities':len(jumps),
  'discontinuity_thresholds':{'absolute_raw_or_adjusted_close_change':.5,'absolute_derived_price_scale_change':.05},
  'retrieval_identity_metadata_is_historical_admission':False,
  'provider_metadata_distinct_values':{f:dict(sorted(Counter('<null>' if r[f] is None else str(r[f]) for r in ordered).items()))
      for f in ('company_id','requested_identifier','ticker','composite_ticker','figi','composite_figi','share_class_figi','exchange_mic','source_run_id')},
  'canonical_mapping_promoted':False,'features_labels_models_backtests_built':False,'new_derived_target_or_RET':False}
 return summary,quality,jumps

def key_and_warmup_summary(keys,sessions):
 need(keys==sorted(set(keys)) and all('2007-01-01'<=day(d)<=END for d in keys),'Original bounded BK key ledger changed')
 known=set(sessions);post=[d for d in sessions if START<=d<='2009-12-31']
 return {'full2007_2023_candidate_keys':len(keys),'premerger_keys_metadata_only':[d for d in keys if d<START],
  'provider_non_XNYS_keys_metadata_only':[d for d in keys if d not in known],
  'postmerger_sessions_to_first2010outcome_forecast_signal':len(post),'first2010outcome_forecast_signal':'2009-12-31',
  '503_session_warmup_start':post[-503] if len(post)>=503 else None,
  '503_session_warmup_end':'2009-12-31','premerger_financial_values_needed_for503warmup':len(post)<503,
  'warmup_key_coverage_complete':len(post)>=503 and all(d in set(keys) for d in post[-503:]),
  'price_field_quality_separately_audited_not_implied_by_key_presence':True}
