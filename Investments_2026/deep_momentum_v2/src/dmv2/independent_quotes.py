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
from .fallback_quotes import build_quote_amendments
from .ohlc_repairs import OHLC, ORIGINAL_FIELDS, original_bar_sha256
from .universe import DataContractError

POLICY_ID='dmv2-global-preflight-independent-sip-2018-2019-v1'
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

def review_independent_case(case, originals, sip):
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
        _need('2018-01-01'<=day<='2019-12-31','Target leaves authorized dates')
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


def review_independent_quotes(requests,control_rows,sip_rows):
    """Review the fixed metadata preflight population without predictions/P&L."""
    _need(isinstance(requests,list) and 0<len(requests)<=256,'Finite preflight requests required')
    _need(len(sip_rows)<=512,'Independent projection exceeds512records')
    sip={}
    for row in sip_rows:
        _need('2018-01-01'<=row['trading_date']<='2019-12-31','Source projection leaves authorized dates')
        key=row['security_id'],row['trading_date'];_need(key not in sip,'Duplicate source quote key');sip[key]=row
    original={}
    for row in control_rows:
        key=row['target_native_id'],row['target_date'],row['role'];_need(key not in original,'Duplicate original target-role key');original[key]=row
    ledger=[]
    for request in requests:
        key=request['intrinio_security_id'],request['trading_date']
        roles={role:original[(*key,role)] for role in ('target','preceding','following') if (*key,role) in original}
        ledger.append(review_independent_case(request,roles,sip))
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


@dataclass(frozen=True)
class IndependentQuoteAmendments:
    compiled:object
    review:dict
    evidence_pins:tuple
    fingerprint:str
    @property
    def source_pins(self):return self.evidence_pins
    @property
    def sessions(self):return self.compiled.sessions
    @property
    def ledger(self):return pd.DataFrame(self.review['route_ledger'])
    @property
    def amendments(self):return self.compiled.amendments
    @property
    def provenance(self):
        return {'policy_id':POLICY_ID,'fingerprint':self.fingerprint,
          'target_count':self.review['target_count'],'accepted_count':self.review['accepted_count'],
          'refused_count':self.review['refused_count'],'replacement_count':sum(r['kind']=='replace' for r in self.review['accepted']),
          'append_count':sum(r['kind']=='append' for r in self.review['accepted']),
          'evidence_pins':list(self.evidence_pins),'original_source_files_changed':False,
          'replacement_volume_changed':False,'target_action_feed_complete':False,
          'historical_identity_certified':False,'knowledge_time_certified':False,
          'unit_transport_provisional':True,'financial_2020_plus_read':False}
    def verify(self,root):
        from .backtest_loading import _verify_pin
        for pin in self.evidence_pins:_verify_pin(Path(root).resolve(),pin)
        self.compiled.verify(root)
        _need(self.fingerprint==canonical_sha256({'policy_id':POLICY_ID,'compiled':self.compiled.fingerprint,'review':self.review,'pins':list(self.evidence_pins)}),'Independent quote bundle changed')
    def apply(self,raw_table,*,start=None,end=None):return self.compiled.apply(raw_table,start=start,end=end)
    def apply_table(self,raw_table,*,start=None,end=None):return self.apply(raw_table,start=start,end=end)


def compile_independent_quote_amendments(root,evidence_root):
    """Verify immutable private review, then compile only its reviewed deltas.

    Caller applies the frozen parent quote bundle first. This separate bundle
    does not edit any parent workers/artifacts/pointers or redefine past policy.
    """
    from .backtest_loading import _pin,_verify_pin
    root,evidence_root=Path(root).resolve(),Path(evidence_root).resolve()
    manifest=json.loads((evidence_root/'repair_manifest.json').read_text())
    _need(manifest['policy_id']==POLICY_ID and manifest['schema']=='dmv2-independent-quote-repair/v1','Independent repair policy/schema differs')
    _need(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==manifest['independent_worker_sha256'],'Independent review worker differs from frozen package')
    pins=[]
    for relative,expected in manifest['artifacts'].items():
        path=(evidence_root/relative).resolve();_need(path.is_relative_to(evidence_root),'Evidence path escapes package')
        _need(path.stat().st_size==expected['size_bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256'],'Independent evidence bytes differ')
        pin=_pin(root,path);_verify_pin(root,pin);pins.append(pin)
    pins.append(_pin(root,evidence_root/'repair_manifest.json'))
    request=json.loads((evidence_root/'request.json').read_text());receipt=json.loads((evidence_root/'sip_range_receipt.json').read_text())
    obj=receipt['object'];_need(obj['uri']==SOURCE_URI and obj['generation']==SOURCE_GENERATION and obj['size_bytes']==SOURCE_SIZE and obj['sha256']==SOURCE_SHA,'Independent source identity differs')
    _need(receipt['publication_manifest_canonical_sha256']==MANIFEST_CANONICAL_SHA and receipt['metadata_before_financial'] is True and receipt['only_exact_requested_financial_rows_returned_or_converted_to_python'] is True and receipt['actual_range_bytes']<=64*1024*1024,'Independent source projection contract differs')
    publication=json.loads((evidence_root/'publication_manifest.json').read_text());_need(canonical_sha256(publication).removeprefix('sha256:')==MANIFEST_CANONICAL_SHA,'Publication manifest canonical pin differs')
    _need(hashlib.sha256((evidence_root/'sip_brackets.parquet').read_bytes()).hexdigest()==receipt['output_sha256'],'Selected source projection pin differs')
    pf=pq.ParquetFile(evidence_root/'sip_brackets.parquet');_need(pf.metadata.num_rows<=512,'Independent projection too large')
    metadata=pf.read(columns=['security_id','trading_date','query_symbol','adjustment','feed','timeframe'])
    wanted={(r['source_security_id'],d) for r in request['requests'] for d in r['dates']}
    _need(all((row['security_id'],row['trading_date']) in wanted and '2018-01-01'<=row['trading_date']<='2019-12-31' for row in metadata.to_pylist()),'Independent quote table leaves exact request')
    controls=pq.read_table(evidence_root/'parent_target_controls.parquet').to_pylist();sip_rows=pf.read().to_pylist()
    review=review_independent_quotes(request['requests'],controls,sip_rows)
    saved=json.loads((evidence_root/'review.json').read_text());_need(canonical_sha256(review)==canonical_sha256(saved),'Independent repair review no longer reproduces')
    # The controls may be repeated per reviewed target; collapse only exact source keys.
    control_frame=pd.DataFrame(controls);base_columns=[c for c in control_frame.columns if c not in ('target_native_id','target_date','role')]
    original=control_frame[base_columns].drop_duplicates().reset_index(drop=True)
    _need(not original.duplicated(['intrinio_security_id','trading_date']).any(),'Original target controls conflict')
    corrected=original.copy(deep=True);append=[]
    for entry in review['accepted']:
        values=entry['replacement_fields'];key=(entry['intrinio_security_id'],entry['trading_date'])
        if entry['kind']=='replace':
            idx=corrected.index[(corrected.intrinio_security_id==key[0])&(corrected.trading_date==key[1])];_need(len(idx)==1,'Replacement original key missing')
            _need(original_bar_sha256(corrected.loc[idx[0]])==entry['original_target_bar_sha256'],'Original repair target hash differs')
            for field,value in values.items():corrected.at[idx[0],field]=value
        else:
            append.append({field:values.get(field) for field in original.columns})
    if append:corrected=pd.concat([corrected,pd.DataFrame(append)],ignore_index=True)
    compiled=build_quote_amendments(original,corrected,ledger=pd.DataFrame(review['route_ledger']),source_pins=tuple(pins),sessions=sorted({d for r in request['requests'] for d in r['dates']}))
    fp=canonical_sha256({'policy_id':POLICY_ID,'compiled':compiled.fingerprint,'review':review,'pins':list(pins)})
    bundle=IndependentQuoteAmendments(compiled,review,tuple(pins),fp);bundle.verify(root);return bundle
