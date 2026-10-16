"""Finite independent daily-quote review, separate from every frozen run.

Full bad bars require full raw overlap controls. Missing bars follow the prior
append contract's close controls. Both use only preceding accepted paired units;
following observations validate units and never determine a repaired value.
No source row, volume, factor or action is changed in an existing quote.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
from math import isfinite
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .catalog import canonical_sha256
from .fallback_quotes import AMENDMENT_SCHEMA, REPLACEMENT_FIELDS, KEYS, _day, _safe
from .ohlc_repairs import OHLC, ORIGINAL_FIELDS, original_bar_sha256
from .universe import DataContractError
import pyarrow.compute as pc
from .walkforward_loading import wrap_walkforward_quotes

POLICY_ID='dmv2-global-preflight-independent-sip-2020-2023-v1'
MAX_CLOSE_ABSOLUTE=.05
MAX_CLOSE_RELATIVE=.005
MAX_PAIRED_RESIDUAL=.0002
SOURCE_URI='gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/daily_bars/part-00000.parquet'
SOURCE_GENERATION=1783344954247444
SOURCE_SIZE=187908505
SOURCE_SHA='sha256:4a4919d48128d5c573ddbca64649d8cc78783f7bbb818219a3ad5f34728e38a0'
MANIFEST_CANONICAL_SHA='67119e4939e031cbe2c4aad752b8f99a7a72e9bc1ca43595e3c1efc5b40d7f03'

def _need(ok,message):
    if not ok: raise DataContractError(message)

def _number(value,name,nonnegative=False):
    _need(not isinstance(value,bool) and isinstance(value,(int,float)) and isfinite(value)
          and (value>=0 if nonnegative else value>0),name+' is invalid')
    return float(value)

def _raw(row,prefix=''):
    values={field:_number(row[prefix+field],prefix+field) for field in OHLC}
    _need(values['low']<=min(values['open'],values['close'])<=max(values['open'],values['close'])<=values['high'],prefix+'OHLC bounds fail')
    return values

def _control(row,*,following):
    _need(row['price_quality_status']=='accepted' and row['frequency']=='daily'
          and row['intraperiod']==False and row['raw_ohlc_bounds_valid']==True
          and row['adjusted_ohlc_bounds_valid']==True,'Original control quality/frequency fails')
    _raw(row);_raw(row,'adj_')
    _need(_number(row['split_ratio'],'Control split ratio')==1.,'Control split requires separately reviewed units')
    dividend=_number(row['dividend'],'Control dividend',True)
    _need(not following or dividend==0.,'Following dividend requires separately reviewed units')
    return dividend

def review_walkforward_quote_case(case, originals, sip):
    """Pure one-route review. Refusals are retained; the caller groups aliases.

    originals is role->original record; sip is exact (source,date)->raw record.
    Missing target action fields remain unknown; constant paired units do not
    assert absence of every corporate action or knowledge-time certification.
    """
    sid,day,source=case['intrinio_security_id'],case['trading_date'],case['source_security_id']
    result={'policy_id':POLICY_ID,'intrinio_security_id':sid,'trading_date':day,
      'source_security_id':source,'member_on_target':case.get('member_on_target'),'status':'REFUSED_UNCHANGED','reason':None,
      'historical_identity_certified':False,'knowledge_time_certified':False,
      'raw_price_interpolation':False,'target_action_feed_complete':False}
    try:
        _need('2020-01-01'<=day<='2023-12-31','Target leaves authorized dates')
        lefts,rights=case['preceding_control'],case['following_control']
        _need(len(lefts)==len(rights)==1,'One accepted preceding and following control required')
        leftday,rightday=lefts[0],rights[0]
        _need(leftday<day<rightday,'Control chronology fails')
        left,right=originals.get('preceding'),originals.get('following')
        _need(left is not None and right is not None,'Original accepted overlap control absent')
        _need(left['trading_date']==leftday and right['trading_date']==rightday
              and left['intrinio_security_id']==right['intrinio_security_id']==sid,'Original control identity/date differs')
        previous_dividend=_control(left,following=False);_control(right,following=True)
        target=originals.get('target');kind='append' if target is None else 'replace'
        if target is not None:
            _need(target['intrinio_security_id']==sid and target['trading_date']==day,'Original target identity/date differs')
            _need(target['price_quality_status'] in ('provider_ohlc_anomaly','provider_incomplete_ohlc')
                  and target['frequency']=='daily' and target['intraperiod']==False,'Existing accepted or unsupported target cannot be replaced')
            _need(_number(target['split_ratio'],'Target split ratio')==1.
                  and _number(target['dividend'],'Target dividend',True)==0.,'Target action requires separate reviewed units')
        bracket={}
        for d in (leftday,day,rightday):
            item=sip.get((source,d));_need(item is not None,'Actual same-source target or overlap quote absent')
            _need(item['query_symbol'] in case['symbol_candidates'],'Independent query symbol differs from dated membership evidence')
            _need(item['feed']=='sip' and item['adjustment']=='raw' and item['timeframe']=='1Day','Independent source frequency/feed/units differ')
            _need(all(isinstance(item.get(k),str) and item[k] for k in ('source_page_hash','source_task_id','source_request_id')),'Independent retained lineage absent')
            bracket[d]=_raw(item)
        multiplier=_number(left['adj_close'],'Preceding adjusted close')/_number(left['close'],'Preceding raw close')
        residuals=[];close_differences=[]
        for d,row in ((leftday,left),(rightday,right)):
            raw=_raw(row);difference=abs(bracket[d]['close']-raw['close'])
            _need(difference<=MAX_CLOSE_ABSOLUTE and difference/raw['close']<=MAX_CLOSE_RELATIVE,'Independent control close conflicts under frozen 0.05/0.005 gate')
            close_differences.append(difference)
            if kind=='replace':
                _need(all(bracket[d][field]==raw[field] for field in OHLC),'Independent full raw overlap OHLC disagrees')
            for field in OHLC:
                residuals.append(abs(_number(row['adj_'+field],'Control adjusted '+field)-raw[field]*multiplier))
        _need(max(residuals)<=MAX_PAIRED_RESIDUAL,'Preceding adjustment units fail following control at 0.0002')
        raw=bracket[day];adjusted={'adj_'+field:raw[field]*multiplier for field in OHLC}
        _need(all(isfinite(x) and x>0 for x in adjusted.values()),'Replacement adjusted OHLC invalid')
        replacement={**raw,**adjusted,'price_quality_status':'accepted','raw_ohlc_bounds_valid':True,'adjusted_ohlc_bounds_valid':True}
        if kind=='append':
            replacement.update(intrinio_security_id=sid,trading_date=day,frequency='daily',intraperiod=False,
                               volume=None,adj_volume=None,factor=None,split_ratio=None,dividend=None)
        result.update(status='ACCEPTED_'+kind.upper(),kind=kind,
          replacement_fields=replacement,original_target_bar_sha256=None if target is None else original_bar_sha256(target),
          preceding_control_bar_sha256=original_bar_sha256(left),following_control_bar_sha256=original_bar_sha256(right),
          original_raw_ohlc=None if target is None else {field:target[field] for field in OHLC},
          original_adjusted_ohlc=None if target is None else {field:target['adj_'+field] for field in OHLC},
          multiplier=multiplier,multiplier_source_date=leftday,following_control_for_qa_only=rightday,
          maximum_paired_residual=max(residuals),maximum_control_close_difference=max(close_differences),
          preceding_dividend_already_in_paired_units=previous_dividend,
          full_raw_control_equality_required=kind=='replace',replacement_volume_changed=False,
          target_source_page_hash=sip[source,day]['source_page_hash'])
    except (DataContractError,KeyError,TypeError,ValueError) as exc:
        result['reason']=str(exc)
    return result


def review_walkforward_quotes(requests,control_rows,sip_rows):
    """Review the fixed metadata preflight population without predictions/P&L."""
    _need(isinstance(requests,list) and 0<len(requests)<=256,'Finite preflight requests required')
    _need(len(sip_rows)<=1024,'Independent projection exceeds1024records')
    sip={}
    for row in sip_rows:
        _need('2020-01-01'<=row['trading_date']<='2023-12-31','Source projection leaves authorized dates')
        key=row['security_id'],row['trading_date'];_need(key not in sip,'Duplicate source quote key');sip[key]=row
    original={}
    for row in control_rows:
        key=row['target_native_id'],row['target_date'],row['role'];_need(key not in original,'Duplicate original target-role key');original[key]=row
    ledger=[]
    for request in requests:
        key=request['intrinio_security_id'],request['trading_date']
        roles={role:original[(*key,role)] for role in ('target','preceding','following') if (*key,role) in original}
        ledger.append(review_walkforward_quote_case(request,roles,sip))
    grouped={}
    for entry in ledger:grouped.setdefault((entry['intrinio_security_id'],entry['trading_date']),[]).append(entry)
    accepted=[];summary=[]
    for (sid,day),entries in sorted(grouped.items()):
        passing=[e for e in entries if e['status'].startswith('ACCEPTED_')]
        if passing and len({canonical_sha256(e['replacement_fields']) for e in passing})==1:
            chosen=sorted(passing,key=lambda e:e['source_security_id'])[0];accepted.append(chosen)
            summary.append({'intrinio_security_id':sid,'trading_date':day,'status':chosen['status'],'reason':None,'accepted_source_routes':len(passing)})
        else:
            reason='Conflicting independently passing source routes' if passing else '; '.join(sorted({e['reason'] or 'Unspecified refusal' for e in entries}))
            summary.append({'intrinio_security_id':sid,'trading_date':day,'status':'REFUSED_UNCHANGED','reason':reason,'accepted_source_routes':0})
    return {'policy_id':POLICY_ID,'accepted':accepted,'route_ledger':ledger,'target_summary':summary,
            'target_count':len(grouped),'accepted_count':len(accepted),'refused_count':len(grouped)-len(accepted)}




def _snapshot(amendments,review,pins,worker_sha):
 return {'policy_id':POLICY_ID,'amendments':amendments.to_pylist(),'review':review,
         'source_pins':list(pins),'worker_sha256':worker_sha}

@dataclass(frozen=True)
class ReviewedWalkforwardQuotes:
 amendments:pa.Table
 review:dict
 source_pins:tuple
 worker_sha256:str
 fingerprint:str
 @property
 def ledger(self):return pd.DataFrame(self.review['route_ledger'])
 @property
 def provenance(self):
  return {'policy_id':POLICY_ID,'fingerprint':self.fingerprint,
   'target_count':self.review['target_count'],'accepted_count':self.review['accepted_count'],
   'refused_count':self.review['refused_count'],'replacement_count':sum(r['kind']=='replace' for r in self.review['accepted']),
   'append_count':sum(r['kind']=='append' for r in self.review['accepted']),
   'source_pins':list(self.source_pins),'worker_sha256':self.worker_sha256,
   'source_files_changed':False,'historical_identity_certified':False,
   'knowledge_time_certified':False,'fabricated_prices':False,
   'existing_volume_action_fields_changed':False,'financial_2024_plus_read':False,
   'source_mixed_row_group_numeric_decode_possible':True,
   'excluded_financial_rows_returned_or_converted_to_python':False}
 def verify(self,root):
  from .backtest_loading import _verify_pin
  _need(self.worker_sha256==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Walk-forward quote worker changed after admission')
  for pin in self.source_pins:_verify_pin(Path(root).resolve(),pin)
  _need(self.fingerprint==canonical_sha256(_snapshot(self.amendments,self.review,self.source_pins,self.worker_sha256)),'Walk-forward quote bundle identity changed')
 def apply(self,table,*,start=None,end=None):
  _need(isinstance(table,pa.Table) and set(KEYS)<=set(table.column_names),'Arrow quote native/date keys required')
  _need(self.fingerprint==canonical_sha256(_snapshot(self.amendments,self.review,self.source_pins,self.worker_sha256)),'Walk-forward quote bundle changed')
  dates=[_day(d) for d in table['trading_date'].to_pylist()];ids=table['intrinio_security_id'].to_pylist()
  _need(all(d<'2024-01-01' for d in dates),'Protected2024 quote metadata refused')
  lower=_day(start) if start else min(dates,default='2020-01-01');upper=_day(end) if end else max(dates,default='2023-12-31')
  _need(lower<=upper<'2024-01-01' and all(lower<=d<=upper for d in dates),'Walk-forward quote input scope mismatch')
  keys=list(zip(ids,dates,strict=True));_need(len(keys)==len(set(keys)),'Duplicate original quote keys');positions={key:i for i,key in enumerate(keys)};changes={};appends=[]
  for item in self.amendments.to_pylist():
   native,day=item['intrinio_security_id'],item['trading_date']
   if not lower<=day<=upper or native not in ids:continue
   fields=json.loads(item['fields_json']);key=native,day
   if key in positions:
    _need(set(ORIGINAL_FIELDS)<=set(table.column_names),'Full original quote fields required for pinned amendments')
    before=table.slice(positions[key],1).select(list(ORIGINAL_FIELDS)).to_pylist()[0];actual=original_bar_sha256(before)
    if actual==item['replacement_bar_sha256']:continue
    _need(item['kind']=='replace' and actual==item['original_bar_sha256'],'Original quote differs from reviewed amendment')
    _need(set(fields)<=set(REPLACEMENT_FIELDS),'Replacement attempts unapproved fields')
    for name,value in fields.items():changes.setdefault(name,{})[positions[key]]=value
   else:
    _need(item['kind']=='append','Pinned original replacement key is absent')
    appends.append({field.name:fields.get(field.name) for field in table.schema})
  output=table
  for name,replacements in changes.items():
   if name not in output.column_names:continue
   mask=pa.array([i in replacements for i in range(len(table))],type=pa.bool_());values=pa.array([replacements[i] for i in sorted(replacements)],type=output.schema.field(name).type)
   output=output.set_column(output.schema.get_field_index(name),output.schema.field(name),pc.replace_with_mask(output[name].combine_chunks(),mask,values))
  if appends:output=pa.concat_tables([output,pa.Table.from_pylist(appends,schema=table.schema)])
  _need(len(output)==len(table)+len(appends) and output.schema==table.schema,'Walk-forward quote application changed schema or original population')
  return output
 def apply_table(self,table,*,start=None,end=None):return self.apply(table,start=start,end=end)


def read_guarded_original_controls(path,requests):
 """Guard every original target/control identity and date before price decode."""
 file=pq.ParquetFile(path);columns=['target_native_id','target_date','role','intrinio_security_id','trading_date']
 _need(set(columns)<=set(file.schema_arrow.names) and 0<file.metadata.num_rows<=1024,
       'Bounded original control metadata required')
 allowed={};kinds={}
 for request in requests:
  target=request['intrinio_security_id'],request['trading_date'];kinds[target]=request['kind']
  roles=allowed.setdefault(target,{'target':set(),'preceding':set(),'following':set()})
  roles['target'].add(request['trading_date']);roles['preceding'].update(request['preceding_control']);roles['following'].update(request['following_control'])
 metadata=file.read(columns=columns).to_pylist();seen=set()
 for row in metadata:
  target=row['target_native_id'],row['target_date'];role=row['role'];signature=(*target,role)
  _need(target in allowed and role in allowed[target] and row['intrinio_security_id']==target[0]
        and row['trading_date'] in allowed[target][role]
        and '2020-01-01'<=row['trading_date']<='2023-12-31',
        'Original control metadata leaves frozen target/native/role/date scope')
  _need(signature not in seen,'Duplicate original target/control role');seen.add(signature)
 for target,kind in kinds.items():
  _need(((*target,'target') in seen)==(kind=='invalid_existing'),
        'Original target presence disagrees with fixed preflight state')
 return file.read().to_pylist()


def compile_walkforward_quotes(root,evidence_root):
 """Reproduce the fixed143-target review from pinned selected observations."""
 from .backtest_loading import _pin,_verify_pin
 root,evidence_root=Path(root).resolve(),Path(evidence_root).resolve();manifest=json.loads((evidence_root/'repair_manifest.json').read_text())
 _need(manifest['schema']=='dmv2-walkforward-quote-repair/v1' and manifest['policy_id']==POLICY_ID,'Walk-forward quote schema/policy differs')
 worker_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();_need(manifest['worker_sha256']==worker_sha,'Walk-forward quote worker differs from reviewed package')
 pins=[]
 for name,expected in manifest['artifacts'].items():
  path=(evidence_root/name).resolve();_need(path.is_relative_to(evidence_root) and path.stat().st_size==expected['size_bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256'],'Walk-forward quote evidence bytes changed')
  pin=_pin(root,path);_verify_pin(root,pin);pins.append(pin)
 pins.append(_pin(root,evidence_root/'repair_manifest.json'))
 plan=json.loads((evidence_root/'acquisition_plan.json').read_text());request=json.loads((evidence_root/'request.json').read_text());receipt=json.loads((evidence_root/'sip_range_receipt.json').read_text())
 _need(hashlib.sha256((evidence_root/'acquisition_plan.json').read_bytes()).hexdigest()=='d1b73db7b7ff56a0eb1b03ea17755a961ad89e90697295b7036f7f837149b148','Frozen acquisition plan differs')
 _need(hashlib.sha256((evidence_root/'request.json').read_bytes()).hexdigest()==plan['request']['sha256'] and request['targets_total']==143 and request['selected_source_date_keys']==308,'Frozen exact source request differs')
 obj=receipt['object'];_need(obj==plan['source_object'] and obj['uri']==SOURCE_URI and obj['generation']==SOURCE_GENERATION and obj['size_bytes']==SOURCE_SIZE and obj['sha256']==SOURCE_SHA,'Pinned SIP source identity differs')
 _need(receipt['publication_manifest_canonical_sha256']==MANIFEST_CANONICAL_SHA and receipt['metadata_before_financial'] is True and receipt['only_exact_requested_financial_rows_returned_or_converted_to_python'] is True and receipt['actual_range_bytes']<=64*1024*1024,'Selected source projection differs')
 publication=json.loads((evidence_root/'publication_manifest.json').read_text());_need(canonical_sha256(publication).removeprefix('sha256:')==MANIFEST_CANONICAL_SHA,'Publication manifest pin differs')
 _need(hashlib.sha256((evidence_root/'sip_brackets.parquet').read_bytes()).hexdigest()==receipt['output_sha256'],'Selected SIP projection pin differs')
 file=pq.ParquetFile(evidence_root/'sip_brackets.parquet');_need(file.metadata.num_rows<=1024,'Selected SIP source exceeds frozen rows')
 wanted={(r['source_security_id'],d) for r in request['requests'] for d in r['dates']};meta=file.read(columns=['security_id','trading_date'])
 _need(all((r['security_id'],r['trading_date']) in wanted and '2020-01-01'<=r['trading_date']<='2023-12-31' for r in meta.to_pylist()),'Selected SIP metadata leaves frozen scope')
 controls=read_guarded_original_controls(evidence_root/'original_target_controls.parquet',request['requests']);review=review_walkforward_quotes(request['requests'],controls,file.read().to_pylist());saved=json.loads((evidence_root/'review.json').read_text());_need(canonical_sha256(review)==canonical_sha256(saved),'Walk-forward quote review no longer reproduces')
 rows=[]
 for entry in review['accepted']:
  fields=entry['replacement_fields'];before=entry['original_target_bar_sha256'];after=original_bar_sha256(fields)
  if entry['kind']=='replace':
   original=next(r for r in controls if r['target_native_id']==entry['intrinio_security_id'] and r['target_date']==entry['trading_date'] and r['role']=='target')
   after=original_bar_sha256({**original,**fields})
  rows.append({'intrinio_security_id':entry['intrinio_security_id'],'trading_date':entry['trading_date'],'kind':entry['kind'],'original_bar_sha256':before,'replacement_bar_sha256':after,'fields_json':json.dumps(fields,sort_keys=True,allow_nan=False),'policy_id':POLICY_ID})
 amendments=pa.Table.from_pylist(rows,schema=AMENDMENT_SCHEMA);fp=canonical_sha256(_snapshot(amendments,review,pins,worker_sha));bundle=ReviewedWalkforwardQuotes(amendments,review,tuple(pins),worker_sha,fp);bundle.verify(root);return bundle


@dataclass(frozen=True)
class CompleteWalkforwardQuotes:
 historical:object
 recent:ReviewedWalkforwardQuotes
 fingerprint:str
 @property
 def provenance(self):return {'policy_id':'old_pre2020_then_reviewed_2020_2023_v1','fingerprint':self.fingerprint,'historical':self.historical.provenance,'recent':self.recent.provenance,'source_files_changed':False,'financial_2024_plus_read':False}
 def verify(self,root):
  self.historical.verify(root);self.recent.verify(root);_need(self.fingerprint==canonical_sha256({'policy_id':'old_pre2020_then_reviewed_2020_2023_v1','historical':self.historical.fingerprint,'recent':self.recent.fingerprint}),'Composed walk-forward quote identity changed')
 def apply(self,table,*,start=None,end=None):return self.recent.apply(self.historical.apply(table,start=start,end=end),start=start,end=end)
 def apply_table(self,table,*,start=None,end=None):return self.apply(table,start=start,end=end)

def compose_complete_walkforward_quotes(historical,recent):
 historical=wrap_walkforward_quotes(historical);_need(isinstance(recent,ReviewedWalkforwardQuotes),'Reviewed recent quote bundle required')
 fp=canonical_sha256({'policy_id':'old_pre2020_then_reviewed_2020_2023_v1','historical':historical.fingerprint,'recent':recent.fingerprint})
 return CompleteWalkforwardQuotes(historical,recent,fp)
