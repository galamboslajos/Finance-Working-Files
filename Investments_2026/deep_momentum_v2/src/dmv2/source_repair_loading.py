"""Explicit post-load source repairs; preserves frozen annual loaders and runs."""
from __future__ import annotations
from dataclasses import dataclass,replace,asdict
from pathlib import Path
import hashlib,json,math
from collections import Counter
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dmv2.catalog import canonical_sha256
from dmv2.data import session_dates
from dmv2.fallback_loading import LoadedFallbackYear,_fingerprint as fallback_fingerprint,PRICE_COLUMNS
from dmv2.walkforward_loading import LoadedWalkforwardYear,_fingerprint as walkforward_fingerprint
from dmv2.identity_amendment import SOURCE_ROUTES
try:
    from .el_paso_route import ElPasoRouteAmendment,SOURCE as EP_SOURCE,OLD as EP_OLD,NEW as EP_NEW
    from .opch_amendment import ReviewedOpchAmendment
except ImportError:
    from el_paso_route import ElPasoRouteAmendment,SOURCE as EP_SOURCE,OLD as EP_OLD,NEW as EP_NEW
    from opch_amendment import ReviewedOpchAmendment
POLICY='dmv2-joint-el-paso-opch-post-load-repair-v1'
EP_START='2008-01-04';EP_END='2012-05-24';EP_ROWS=1111
class RepairError(ValueError):pass
def need(value,message):
    if not value:raise RepairError(message)
def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return 'sha256:'+h.hexdigest()
def date_guard(value):
    from datetime import date
    need(isinstance(value,str),'Canonical string date required')
    try:d=date.fromisoformat(value)
    except ValueError as exc:raise RepairError('Malformed date')from exc
    need(d.isoformat()==value and value<'2024-01-01','Protected or noncanonical date');return value
@dataclass(frozen=True)
class PinnedFile:
    path:str
    sha256:str
    size_bytes:int
    def verify(self,*,max_bytes=64*1024**2):
        p=Path(self.path);need(p.is_absolute() and p.resolve()==p and not any(q.is_symlink()for q in[p,*p.parents]),'Canonical explicit nonsymlinked pin path required')
        need(type(self.size_bytes)is int and 0<self.size_bytes<=max_bytes,'Pinned file byte limit exceeded')
        need(p.is_file()and p.stat().st_size==self.size_bytes and file_hash(p)==self.sha256,'Pinned file changed');return p
@dataclass(frozen=True)
class CorrectElPasoProjection:
    projection:PinnedFile
    receipt:PinnedFile
    plan:PinnedFile
    @property
    def fingerprint(self):return canonical_sha256({'projection':asdict(self.projection),'receipt':asdict(self.receipt),'plan':asdict(self.plan)})
    @classmethod
    def from_pins(cls,*,projection,receipt,plan):
        state=cls(projection,receipt,plan);state.verify_metadata();return state
    def verify_metadata(self):
        pp=self.projection.verify();rp=self.receipt.verify(max_bytes=256*1024);pl=self.plan.verify(max_bytes=256*1024)
        r=json.loads(rp.read_text());p=json.loads(pl.read_text());need(r.get('status')=='COMPLETE_CORRECT_NATIVE_LOCAL_HISTORY_ACQUIRED'and r.get('plan_sha256')==self.plan.sha256.removeprefix('sha256:'),'Acquisition receipt/plan mismatch')
        output=r.get('output',{});need(output.get('sha256')==self.projection.sha256.removeprefix('sha256:')and output.get('rows')==EP_ROWS and Path(output.get('path','')).name==pp.name,'Projection is not exact acquired output (relocation must preserve basename/hash)')
        need(r.get('whole_source_hash_authenticated_before_and_after')is True and r.get('native_id')==EP_NEW and r.get('figi')=='BBG000DST4Q5'and r.get('provider_lookup_ISIN')=='US28336L1098','Correct native/whole-source receipt guards missing')
        need(r.get('actual_date_min')==EP_START and r.get('actual_date_max')==EP_END and r.get('missing_session_dates')==[] and r.get('no_synthetic_quotes')is True and r.get('financial_2024_plus_returned')is False,'Acquisition dates/completeness/holdout guards differ')
        expected=p.get('expected_provider_metadata',{});need(expected.get('intrinio_security_id')==EP_NEW and expected.get('company_id')=='com_gDvxOz'and expected.get('figi')=='BBG000DST4Q5'and 'query_identifier=US28336L1098-'in expected.get('source_relative_path',''),'Frozen provider identity guard differs')
        objects=p.get('source_objects');need(isinstance(objects,list)and len(objects)==1,'One exact original source object required')
        declared=r.get('provenance',{}).get('objects',[]);need(len(declared)==1,'Acquisition provenance must bind one source object')
        obj=objects[0];need(declared[0]['relative_path']==obj['relative_path']and declared[0]['sha256']==obj['manifest_sha256']and declared[0]['size_bytes']==obj['size_bytes'],'Source object declarations differ')
        source=Path(obj['path']);need(source.is_file()and source.stat().st_size==obj['size_bytes']and file_hash(source)==obj['manifest_sha256'],'Original whole source changed after acquisition')
        f=pq.ParquetFile(pp);need(f.metadata.num_rows==EP_ROWS<=2048,'Projection row cap/count differs')
        meta=['intrinio_security_id','trading_date','company_id','ticker','figi','frequency','intraperiod','source_relative_path'];need(set(meta)<=set(f.schema_arrow.names)and set(PRICE_COLUMNS)<=set(f.schema_arrow.names),'Projection lacks reviewed core metadata')
        t=f.read(columns=meta);rows=t.to_pylist();keys=set()
        for x in rows:
            d=date_guard(x['trading_date']);key=(x['intrinio_security_id'],d);need(key not in keys,'Duplicate corrected-source key');keys.add(key)
            need(EP_START<=d<=EP_END and x['intrinio_security_id']==EP_NEW and x['company_id']=='com_gDvxOz'and x['ticker']=='EP'and x['figi']=='BBG000DST4Q5'and x['frequency']=='daily'and x['intraperiod']is False and'intrinio_security_id=sec_XO6Eab-'in x['source_relative_path'],'Correct-source metadata mismatch')
        need(min(x['trading_date']for x in rows)==EP_START and max(x['trading_date']for x in rows)==EP_END,'Source span differs');return t
    def load(self):
        meta=self.verify_metadata();t=pq.read_table(self.projection.path,columns=list(PRICE_COLUMNS));need(t.select(['intrinio_security_id','trading_date']).equals(meta.select(['intrinio_security_id','trading_date'])),'Projection metadata changed across financial read');return t
@dataclass(frozen=True)
class JointSourceRepairs:
    el_paso_route:ElPasoRouteAmendment|None=None
    el_paso_source:CorrectElPasoProjection|None=None
    opch:ReviewedOpchAmendment|None=None
    @property
    def fingerprint(self):return canonical_sha256({'policy':POLICY,'el_paso_route':getattr(self.el_paso_route,'fingerprint',None),'el_paso_source':getattr(self.el_paso_source,'fingerprint',None),'opch':getattr(self.opch,'fingerprint',None)})
    def verify(self):
        need((self.el_paso_route is None)==(self.el_paso_source is None),'Route and explicit correct source must be admitted together')
        need(self.el_paso_route is not None or self.opch is not None,'At least one explicit repair required')
        if self.el_paso_route is not None:
            need(isinstance(self.el_paso_route,ElPasoRouteAmendment)and isinstance(self.el_paso_source,CorrectElPasoProjection),'Typed reviewed El Paso state required');self.el_paso_route._verify();self.el_paso_source.verify_metadata()
        if self.opch is not None:need(isinstance(self.opch,ReviewedOpchAmendment),'Typed reviewed OPCH state required');self.opch._verify()
def counts(table):
    rows=table.select(['signal_date','source_security_id']).to_pylist();pairs=[(x['signal_date'],x['source_security_id'])for x in rows];need(len(pairs)==len(set(pairs)),'Duplicate original source/date member');return dict(Counter(x[0]for x in pairs))
def guard_prices(table,*,start,end,ids,max_rows):
    need(isinstance(table,pa.Table)and set(PRICE_COLUMNS)<=set(table.column_names)and len(table.column_names)==len(set(table.column_names)),'Complete unique daily schema required')
    need(table.num_rows<=max_rows,'Annual quote row ceiling exceeded');rows=table.select(['intrinio_security_id','trading_date']).to_pylist();keys=set()
    for r in rows:
        d=date_guard(r['trading_date']);native=r['intrinio_security_id'];need(isinstance(native,str)and native in ids and start<=d<=end,'Quote ID/date escapes frozen request');need((native,d)not in keys,'Duplicate native/date quote');keys.add((native,d))
def align(table,schema):
    arrays=[];fields=[]
    for field in schema:
        if field.name in table.column_names:arrays.append(pc.cast(table[field.name],field.type,safe=True));fields.append(field)
        else:arrays.append(pa.nulls(table.num_rows,type=field.type));fields.append(field.with_nullable(True))
    return pa.Table.from_arrays(arrays,schema=pa.schema(fields,metadata=schema.metadata))
def amend_prices(base,source,*,ids,permitted,start,end,max_rows):
    # Preserve all existing retained values/audit columns; remove only no-longer-requested native histories.
    base=base.filter(pc.is_in(base['intrinio_security_id'],value_set=pa.array(ids,type=base['intrinio_security_id'].type)))
    added=0
    if source is not None and EP_NEW in ids:
        extra=source.filter(pc.and_(pc.is_in(source['trading_date'],value_set=pa.array(sorted(permitted),type=source['trading_date'].type)),pc.and_(pc.greater_equal(source['trading_date'],start),pc.less_equal(source['trading_date'],end))))
        need(not pc.any(pc.equal(base['intrinio_security_id'],EP_NEW)).as_py(),'Correct native already present in base; explicit overlap review required')
        extra=align(extra,base.schema);base=pa.concat_tables([base,extra],promote_options='default');added=extra.num_rows
    guard_prices(base,start=start,end=end,ids=set(ids),max_rows=max_rows);return base,added
def apply_source_repairs(loaded,repairs,*,approved_source_repair=False):
    need(approved_source_repair is True and isinstance(repairs,JointSourceRepairs),'Explicit approved typed post-load repairs required')
    if isinstance(loaded,LoadedFallbackYear):fp=fallback_fingerprint;ceiling='2019-12-31'
    elif isinstance(loaded,LoadedWalkforwardYear):fp=walkforward_fingerprint;ceiling='2023-12-31'
    else:raise RepairError('An already base-loaded fallback/walk-forward year is required')
    p=loaded.plan;need(fp(p)==p.fingerprint and loaded.provenance.get('plan_fingerprint')==p.fingerprint,'Base plan fingerprint changed');need(date_guard(p.raw_start)<=date_guard(p.feature_end)<=date_guard(p.raw_end)<=ceiling,'Frozen annual bounds differ')
    repairs.verify()
    if loaded.provenance.get('joint_source_repairs'):
        raise RepairError('Already amended input: reuse the verified result explicitly; automatic reapplication is refused')
    for table in[p.original_cohort,p.daily_cohorts]:
        counts(table)
        need(all(p.raw_start<=date_guard(d)<=p.feature_end for d in set(table['signal_date'].to_pylist())), 'Original cohort dates escape frozen past-feature bounds')
        for r in table.select(['source_security_id','intrinio_security_id','identity_eligible']).to_pylist():
            if r['source_security_id']in SOURCE_ROUTES:need(r['intrinio_security_id']is None and r['identity_eligible']is False,'Existing BBBY quarantine lost')
    guard_prices(loaded.feature_prices,start=p.raw_start,end=p.feature_end,ids=set(p.requested_ids),max_rows=p.limits.max_price_cache_rows)
    guard_prices(loaded.outcome_prices,start=p.raw_start,end=p.raw_end,ids=set(p.requested_ids),max_rows=p.limits.max_price_cache_rows)
    monthly,daily=p.original_cohort,p.daily_cohorts;route_ledgers={};source=None
    if repairs.el_paso_route is not None:
        for name,t in[('monthly',monthly),('daily',daily)]:
            rr=repairs.el_paso_route.apply_with_ledger(t,all_original_members=True,original_counts=counts(t));route_ledgers[name]=rr.ledger
            if name=='monthly':monthly=rr.table
            else:daily=rr.table
    requested=tuple(sorted({r['intrinio_security_id']for t in[monthly,daily]for r in t.select(['intrinio_security_id','identity_eligible']).to_pylist()if r['identity_eligible']}));need(requested,'Empty corrected request')
    new_ids=set(requested)-set(p.requested_ids);need(new_ids<={EP_NEW}and (not new_ids or repairs.el_paso_source is not None),'Unreviewed native source admission')
    need(set(requested)<=set(p.original_requested_ids)|({EP_NEW}if repairs.el_paso_source else set()),'Corrected request escapes original+explicit source allowlist')
    if EP_NEW in requested:source=repairs.el_paso_source.load()
    feature_sessions={x for x in session_dates(p.calendar)if p.raw_start<=x<=p.feature_end}
    outcome_sessions={day for w in p.windows for day in(w.signal_date,*w.expected_outcome_sessions)}
    feature,fa=amend_prices(loaded.feature_prices,source,ids=requested,permitted=feature_sessions,start=p.raw_start,end=p.feature_end,max_rows=p.limits.max_price_cache_rows)
    outcome,oa=amend_prices(loaded.outcome_prices,source,ids=requested,permitted=outcome_sessions,start=p.raw_start,end=p.raw_end,max_rows=p.limits.max_price_cache_rows)
    opch_ledgers={}
    # Always after the completed old composed overlay, including2022 corrections in2023 warmup.
    if repairs.opch is not None and p.raw_start<='2022-03-10'and p.raw_end>='2022-01-28':
        for name,t,end in[('feature',feature,p.feature_end),('outcome',outcome,p.raw_end)]:
            result=repairs.opch.apply_with_ledger(t,start=p.raw_start,end=end);opch_ledgers[name]=result.ledger
            if name=='feature':feature=result.table
            else:outcome=result.table
    snapshots=tuple(replace(s,table=monthly.filter(pc.equal(monthly['signal_date'],s.as_of)),provenance={**s.provenance,'joint_source_repair_state_fingerprint':repairs.fingerprint,'el_paso_source_native_route_corrected':bool(route_ledgers.get('monthly'))})for s in p.snapshots)
    joint={'policy':POLICY,'state_fingerprint':repairs.fingerprint,'base_plan_fingerprint':p.fingerprint,'base_quote_amendments_fingerprint':loaded.provenance.get('quote_amendments_fingerprint'),'el_paso_route_fingerprint':getattr(repairs.el_paso_route,'fingerprint',None),'explicit_el_paso_source_fingerprint':getattr(repairs.el_paso_source,'fingerprint',None),'opch_amendment_fingerprint':getattr(repairs.opch,'fingerprint',None),'el_paso_route_ledger_sha256':canonical_sha256(route_ledgers),'el_paso_route_changed_rows':{k:len(v)for k,v in route_ledgers.items()},'correct_native_added_rows':{'feature':fa,'outcome':oa},'opch_ledger_sha256':canonical_sha256(opch_ledgers),'opch_applied_rows':{k:len(v)for k,v in opch_ledgers.items()},'original_membership_keys_preserved':True,'producer_review_flags_preserved':True,'original_cache_request_ids_preserved':True,'explicit_new_source_ids':sorted(new_ids),'original_provisional_identity_policy_unchanged':True,'opch_after_existing_quote_bundle':True,'opch_volume_imputed':False,'historical_identity_certified':False,'source_files_changed':False}
    read_plan={**p.read_plan,'admitted_request_ids':len(requested),'joint_source_repairs':joint}
    provenance={**p.provenance,'joint_source_repairs':joint,'quote_amendments_fingerprint':canonical_sha256(joint)}
    plan=replace(p,original_cohort=monthly,daily_cohorts=daily,snapshots=snapshots,requested_ids=requested,read_plan=read_plan,provenance=provenance,fingerprint='');plan=replace(plan,fingerprint=fp(plan))
    prov={**loaded.provenance,'joint_source_repairs':joint,'quote_amendments_fingerprint':canonical_sha256(joint),'quote_amendment_sha256':canonical_sha256(joint),'plan_fingerprint':plan.fingerprint}
    stats={**loaded.loading_stats,'feature_returned_rows':feature.num_rows,'outcome_returned_rows':outcome.num_rows,'joint_source_repairs':joint}
    return replace(loaded,feature_prices=feature,outcome_prices=outcome,plan=plan,loading_stats=stats,provenance=prov)
