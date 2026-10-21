"""Pure paired SIP OHLCV diagnostics; no I/O or research inputs/model changes."""
from collections import Counter
from datetime import date
import math
from numbers import Real

SOURCE_ID='siblis_isin_b284bb7497616f5afba0';SYMBOL='XOM'
START='2016-01-04';END='2023-12-31';LABELS=('raw','all')
KEYS=('security_id','trading_date','adjustment');OHLC=('open','high','low','close')
FIELDS=(*KEYS,'query_symbol',*OHLC,'volume')
PRICE_SCALE_TOLERANCE=.0002
FORBIDDEN={'vwap','trade_count','dividend','split_ratio','factor','target','label_return','target_class'}

class AuditError(ValueError):pass

def need(value,message):
 if not bool(value):raise AuditError(message)

def day(value):
 need(isinstance(value,str),'Explicit date string required')
 try:d=date.fromisoformat(value)
 except ValueError as error:raise AuditError('Invalid date') from error
 need(value==d.isoformat(),'Noncanonical date');return value

def numeric(value,positive=False,nonnegative=False):
 if not isinstance(value,Real) or isinstance(value,bool) or not math.isfinite(value):return None
 if positive and value<=0 or nonnegative and value<0:return None
 return float(value)

def ratio(a,b):
 if a is None or b is None or b<=0:return None
 return numeric(a/b,positive=True)

def row_flags(row):
 flags=[];prices=[numeric(row[f],positive=True) for f in OHLC]
 if any(v is None for v in prices):flags.append('OHLC_nonpositive_or_nonfinite')
 else:
  o,h,l,c=prices
  if not l<=min(o,c)<=max(o,c)<=h:flags.append('OHLC_bracket_invalid')
 if numeric(row['volume'],nonnegative=True) is None:flags.append('volume_negative_or_nonfinite')
 return flags

def audit_rows(rows,sessions):
 """Return(summary,rowquality) against independently pinned caller sessions.

    All source/date/adjustment/prohibited-field metadata is validated before any
    financial scalar is converted or used. Missing raw/all pairs remain missing;
    output quality retains every supplied expected session. No fill or inferred
    dividend/split cash flow is produced.
    """
 calendar=[day(d) for d in sessions]
 need(calendar and calendar==sorted(set(calendar)),'Ordered unique independent XNYS calendar required')
 grid=[d for d in calendar if START<=d<=END];need(grid,'Empty authorized independent session grid')
 known=set(grid);lookup={};metadata_counts={f:Counter() for f in ('feed','timeframe','query_symbol')}
 for row in rows:
  need(set(FIELDS)<=set(row) and not set(row)&FORBIDDEN,'Incomplete or unapproved source fields')
  d=day(row['trading_date'])
  need(row['security_id']==SOURCE_ID and row['query_symbol']==SYMBOL
       and row['adjustment'] in LABELS and START<=d<=END and d in known,
       'Unapproved source/query/adjustment/date/non-session before numeric use')
  if 'feed' in row:need(row['feed']=='sip','Unexpected source feed')
  if 'timeframe' in row:need(row['timeframe']=='1Day','Unexpected source timeframe')
  key=(row['security_id'],d,row['adjustment']);need(key not in lookup,'Duplicate source/date/adjustment key')
  lookup[key]=row
  for field in metadata_counts:
   if field in row:metadata_counts[field][str(row[field])]+=1
 quality=[];paired={};counts=Counter()
 for d in grid:
  raw=lookup.get((SOURCE_ID,d,'raw'));adjusted=lookup.get((SOURCE_ID,d,'all'))
  flags=[];raw_flags=row_flags(raw) if raw else [];all_flags=row_flags(adjusted) if adjusted else []
  if raw is None:flags.append('missing_raw_pair_row')
  if adjusted is None:flags.append('missing_all_pair_row')
  flags.extend('raw_'+flag for flag in raw_flags);flags.extend('all_'+flag for flag in all_flags)
  scale=None;residual=None;raw_volume=None;all_volume=None;volume_difference=None;volume_ratio=None
  if raw is not None and adjusted is not None:
   rc=numeric(raw['close'],positive=True);ac=numeric(adjusted['close'],positive=True)
   if rc is not None and ac is not None:
    scale=ratio(ac,rc);errors=[]
    for field in OHLC:
     r=numeric(raw[field],positive=True);a=numeric(adjusted[field],positive=True)
     observed=ratio(a,r);relative=ratio(observed,scale)
     if relative is None:errors=[];break
     errors.append(abs(relative-1))
    if errors:residual=max(errors)
   if residual is None or residual>PRICE_SCALE_TOLERANCE:flags.append('raw_all_OHLC_price_scales_disagree')
   raw_volume=numeric(raw['volume'],nonnegative=True);all_volume=numeric(adjusted['volume'],nonnegative=True)
   if raw_volume is not None and all_volume is not None:
    volume_difference=all_volume-raw_volume
    volume_ratio=ratio(all_volume,raw_volume)
   paired[d]={'raw_close':rc,'all_close':ac,'scale':scale}
  counts.update(flags)
  quality.append({'security_id':SOURCE_ID,'trading_date':d,'raw_present':raw is not None,'all_present':adjusted is not None,
   'flags':flags,'raw_flags':raw_flags,'all_flags':all_flags,'paired_all_over_raw_close':scale,
   'OHLC_max_common_scale_relative_residual':residual,'raw_volume':raw_volume,'all_volume':all_volume,
   'volume_difference_all_minus_raw':volume_difference,'all_over_raw_volume_ratio':volume_ratio,
   'economic_return_or_identity_certified':False})
 jumps=[]
 for previous,current in zip(grid,grid[1:]):
  # Inspect each side only across adjacent expected sessions; never skip gaps.
  def close(d,label):
   row=lookup.get((SOURCE_ID,d,label));return numeric(row['close'],positive=True) if row else None
  rr=ratio(close(current,'raw'),close(previous,'raw'));ar=ratio(close(current,'all'),close(previous,'all'))
  fc=ratio(paired.get(current,{}).get('scale'),paired.get(previous,{}).get('scale'))
  raw_change=rr-1 if rr is not None else None
  all_change=ar-1 if ar is not None else None
  factor_change=fc-1 if fc is not None else None
  if (raw_change is not None and abs(raw_change)>.5) or (all_change is not None and abs(all_change)>.5) or (factor_change is not None and abs(factor_change)>.05):
   jumps.append({'security_id':SOURCE_ID,'previous_session':previous,'trading_date':current,
    'raw_close_change':raw_change,'all_close_change':all_change,'paired_price_factor_change':factor_change,
    'explicit_dividend_split_action_fields_available':False,'verified_economic_explanation':False,
    'synthetic_fill_or_automatic_exclusion':False})
 residuals=[r['OHLC_max_common_scale_relative_residual'] for r in quality if r['OHLC_max_common_scale_relative_residual'] is not None]
 differing=[r for r in quality if r['volume_difference_all_minus_raw'] not in (None,0.)]
 summary={'schema':'dmv2-XOM-SIP-paired-field-diagnostic/v1','security_id':SOURCE_ID,'query_symbol':SYMBOL,
  'scope_start':START,'scope_end':END,'input_rows':len(rows),'expected_XNYS_sessions':len(grid),
  'raw_rows':sum(r['raw_present'] for r in quality),'all_rows':sum(r['all_present'] for r in quality),
  'complete_raw_all_pairs':len(paired),'missing_raw_dates':[r['trading_date'] for r in quality if not r['raw_present']],
  'missing_all_dates':[r['trading_date'] for r in quality if not r['all_present']],
  'dates_with_quality_flags':sum(bool(r['flags']) for r in quality),'flag_counts':dict(sorted(counts.items())),
  'OHLC_common_price_scale_relative_tolerance':PRICE_SCALE_TOLERANCE,
  'max_OHLC_common_scale_relative_residual':max(residuals) if residuals else None,
  'volume_difference_pair_dates':len(differing),'volume_assumed_inverse_of_price_adjustment_factor':False,
  'null_action_semantics':'No explicit dividend/split/action fields supplied by this SIP source; unavailable, not zero.',
  'explicit_dividend_split_action_fields_available':False,'economic_total_return_certified':False,
  'discontinuity_thresholds':{'absolute_raw_or_all_close_change':.5,'absolute_price_factor_change':.05},
  'flagged_consecutive_session_discontinuities':jumps,'calendar_independently_pinned_by_caller_required':True,
  'source_metadata_counts':{k:dict(sorted(v.items())) for k,v in metadata_counts.items()},
  'source_alias_identity_flags_used_as_predictors':False,'canonical_mapping_promoted':False,
  'features_labels_models_or_backtests_built':False,'missing_values_filled':False,
  'limitations':['OHLCV pair checks and smoothness do not independently prove instrument identity or action/total-return completeness.',
   'Missing dates and paired rows stay visible; no future coverage admission or source row exclusion is authorized.',
   'Source provenance/aliases are audit evidence only; the caller retains originals and separately verifies read scope and hashes.']}
 return summary,quality
