"""Exact, immutable OPCH amendment. No quote acquisition or implicit repairs."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
import hashlib,json,math
from pathlib import Path,PurePosixPath
from numbers import Real
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

POLICY='dmv2-opch-confirmed15-sip-ohlc-null-volume-v1'
NATIVE='sec_zmox6E'
SOURCE='siblis_ticker_company_ecfc88ae95b19a8b279a'
DATES=('2022-01-28','2022-01-31','2022-02-02','2022-02-03','2022-02-08','2022-02-09','2022-02-11','2022-02-14','2022-02-18','2022-02-22','2022-02-25','2022-03-02','2022-03-04','2022-03-08','2022-03-10')
OHLC=('open','high','low','close');ADJ=tuple('adj_'+c for c in OHLC)
CORE=('intrinio_security_id','trading_date','frequency','intraperiod',*OHLC,'volume',*ADJ,'adj_volume','factor','split_ratio','dividend','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid','price_quality_status')
CHANGED=(*OHLC,*ADJ,'volume','adj_volume')
ARTIFACTS={'proposal','original_projection','original_provenance','original_source','independent_projection','independent_receipt','publication_manifest'}
SOURCE_URI='gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/daily_bars/part-00000.parquet'
SOURCE_GENERATION=1783344954247444
SOURCE_SHA='sha256:4a4919d48128d5c573ddbca64649d8cc78783f7bbb818219a3ad5f34728e38a0'

class AmendmentError(ValueError):pass

def need(ok,message):
    if not ok:raise AmendmentError(message)

def day(v):
    if isinstance(v,datetime):need(v.time()==datetime.min.time(),'Noncanonical session timestamp');v=v.date()
    if isinstance(v,date):v=v.isoformat()
    need(isinstance(v,str),'ISO date required')
    try:d=date.fromisoformat(v)
    except ValueError as e:raise AmendmentError('Invalid date') from e
    need(d.isoformat()==v and d<date(2024,1,1),'Protected or noncanonical date')
    return v

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(obj):return 'sha256:'+hashlib.sha256(canonical(obj)).hexdigest()

def scalar(v):
    if v is None:return None
    if isinstance(v,bool):return v
    if isinstance(v,Real):
        need(math.isfinite(v),'Nonfinite original field');return float(v)
    if isinstance(v,(str,date,datetime)):return day(v) if isinstance(v,(date,datetime)) else v
    raise AmendmentError('Unsupported source field type')

def record(row):
    need(set(CORE)<=set(row),'Full original fields required')
    out={c:scalar(row[c]) for c in CORE};out['trading_date']=day(out['trading_date']);return out

def bars(row,prefix=''):
    out={c:scalar(row[prefix+c]) for c in OHLC}
    need(all(isinstance(v,float) and v>0 for v in out.values()),'Positive finite OHLC required')
    need(out['low']<=min(out['open'],out['close'])<=max(out['open'],out['close'])<=out['high'],'OHLC bounds fail')
    return out

def artifact_pin(root,path,source_uri):
    root=Path(root).resolve();path=Path(path).resolve();need(path.is_relative_to(root),'Artifact escapes evidence root')
    return {'path':path.relative_to(root).as_posix(),'size_bytes':path.stat().st_size,'sha256':'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),'source_uri':source_uri}

def verify_pin(root,pin):
    need(isinstance(pin,dict) and set(pin)=={'path','size_bytes','sha256','source_uri'},'Closed artifact pin required')
    rel=pin['path'];need(isinstance(rel,str) and rel and '\\' not in rel,'Relative POSIX evidence path required');pp=PurePosixPath(rel)
    need(not pp.is_absolute() and all(z not in ('','.','..') for z in rel.split('/')),'Evidence path traversal refused')
    path=Path(root).joinpath(*pp.parts);need(path.resolve().is_relative_to(Path(root).resolve()),'Evidence escapes root')
    need(not any(p.is_symlink() for p in [path,*path.parents]),'Symlinked evidence refused')
    need(type(pin['size_bytes']) is int and 0<pin['size_bytes']<=64*1024*1024,'Small bounded artifact required')
    need(path.is_file() and path.stat().st_size==pin['size_bytes'],'Evidence size or path changed')
    need('sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256'],'Evidence SHA256 changed')
    need(isinstance(pin['source_uri'],str) and bool(pin['source_uri']),'Evidence source reference required')
    return path

def selected_records(path,*,key_column,keys,columns,max_rows=2048):
    f=pq.ParquetFile(path);need(f.metadata.num_rows<=max_rows,'Evidence projection row budget exceeded')
    meta=f.read(columns=[key_column,'trading_date']);rows=meta.to_pylist()
    for r in rows:day(r['trading_date'])
    take=[i for i,r in enumerate(rows) if (r[key_column],r['trading_date']) in keys]
    need(len(take)==len(keys),'Every reviewed target/control requires exactly one evidence row')
    out=f.read(columns=columns).take(pa.array(take,type=pa.int64())).to_pylist()
    need(len({(r[key_column],r['trading_date']) for r in out})==len(keys),'Evidence duplicate key refused')
    return {(r[key_column],r['trading_date']):r for r in out}

def build_manifest(*,evidence_root,artifacts,reviewed=False):
    """Build admission-ready evidence from pinned tiny projections, never prices.

    `reviewed=True` is an explicit caller declaration after review. This function
    cannot discover or acquire replacement quotes. Pin the returned manifest
    separately before `ReviewedOpchAmendment.from_manifest` accepts it.
    """
    root=Path(evidence_root).resolve();need(set(artifacts)==ARTIFACTS,'All seven source/proposal artifacts required')
    paths={k:verify_pin(root,p) for k,p in artifacts.items()}
    proposal=json.loads(paths['proposal'].read_text());need(proposal.get('schema')=='dmv2-opch-confirmed-contamination-repair-evidence/v1','Wrong forensic proposal schema')
    need(proposal.get('target_count')==15 and len(proposal.get('targets',[]))==15,'Exactly fifteen targets required')
    targets=proposal['targets'];need({t.get('trading_date') for t in targets}==set(DATES),'Only exact reviewed OPCH dates admitted')
    control_days={t[k] for t in targets for k in ('preceding_full_ohlc_match_control','following_full_ohlc_match_control')}
    keys={(NATIVE,d) for d in set(DATES)|control_days}
    originals=selected_records(paths['original_projection'],key_column='intrinio_security_id',keys=keys,columns=[*CORE,'source_page_hash'])
    sipcolumns=['security_id','trading_date','query_symbol','feed','adjustment','timeframe','source_page_hash',*OHLC]
    independent=selected_records(paths['independent_projection'],key_column='security_id',keys={(SOURCE,d) for _,d in keys},columns=sipcolumns)
    provenance=json.loads(paths['original_provenance'].read_text());need(provenance.get('source_objects_hash_reread_matched') is True,'Original source whole-object verification required')
    objects=provenance['provenance']['objects'];need(len(objects)==1 and objects[0]['sha256']==artifacts['original_source']['sha256'],'Original source object pin mismatch')
    receipt=json.loads(paths['independent_receipt'].read_text());obj=receipt['object']
    need(obj['uri']==SOURCE_URI and int(obj['generation'])==SOURCE_GENERATION and obj['sha256']==SOURCE_SHA,'SIP object identity/generation/hash pin differs')
    need(receipt['output_sha256']==artifacts['independent_projection']['sha256'].removeprefix('sha256:'),'SIP receipt output pin differs')
    need(receipt.get('metadata_before_financial') is True and receipt.get('no_unrequested_financial_rows_returned_or_converted') is True,'SIP projection guards missing')
    # The publication bytes are independently hashed above. Its declaration must
    # bind this exact object; partial-source limitations remain in the manifest.
    publication=json.loads(paths['publication_manifest'].read_text());pubtext=json.dumps(publication,sort_keys=True)
    need(SOURCE_SHA in pubtext and ('1783344954247444' in pubtext),'Publication does not bind SIP object')
    for (sid,d),r in independent.items():
        need(sid==SOURCE and r['query_symbol']=='OPCH' and r['feed']=='sip' and r['adjustment']=='raw' and r['timeframe']=='1Day','Wrong independent instrument/feed/units')
        bars(r);need(isinstance(r['source_page_hash'],str) and bool(r['source_page_hash']),'Independent page lineage required')
    controls={}
    for d in sorted(control_days):
        r=originals[NATIVE,d];s=independent[SOURCE,d];rr=record(r)
        need(rr['frequency']=='daily' and rr['intraperiod'] is False and rr['split_ratio']==1. and rr['dividend']==0.,'Control frequency/action guard failed')
        need(rr['raw_ohlc_bounds_valid'] is True and rr['adjusted_ohlc_bounds_valid'] is True and rr['price_quality_status']=='accepted','Control original quality guard failed')
        need(bars(rr)==bars(s) and bars(rr,'adj_')==bars(rr),'All four raw and adjusted controls must match multiplier one')
        controls[d]={'original_record':rr,'original_record_sha256':digest(rr),'original_page_hash':r['source_page_hash'],'independent_page_hash':s['source_page_hash']}
    cases=[]
    for t in sorted(targets,key=lambda z:z['trading_date']):
        d=day(t['trading_date']);r=originals[NATIVE,d];s=independent[SOURCE,d];rr=record(r)
        need(t['intrinio_security_id']==NATIVE and t['source_security_id']==SOURCE,'Wrong target native/source identity')
        need(t['paired_multiplier_from_preceding']==1. and t['maximum_following_adjusted_control_residual']==0.,'Only reviewed exact multiplier-one bracket admitted')
        need(t['other_source_fields_unchanged'] is True and t['proposed_volume'] is None and t['proposed_adj_volume'] is None,'Volume policy or field scope differs')
        need(rr['frequency']=='daily' and rr['intraperiod'] is False and rr['split_ratio']==1. and rr['dividend']==0.,'Target frequency/action guard failed')
        need(bars(rr)==t['original_raw_ohlc'] and bars(rr,'adj_')==bars(rr),'Original target differs from forensic proposal')
        need(rr['volume']==t['original_volume'] and rr['adj_volume']==t['original_adj_volume'],'Original volume differs')
        need(r['source_page_hash']==t['original_source_page_hash'] and s['source_page_hash']==t['independent_source_page_hash'],'Target page lineage changed')
        replacement=bars(s);need(replacement==t['independent_raw_ohlc'],'Independent target differs from proposal')
        pre,post=t['preceding_full_ohlc_match_control'],t['following_full_ohlc_match_control'];need(pre<d<post and pre in controls and post in controls,'Dated control bracket differs')
        new=dict(rr,**replacement,**{'adj_'+c:replacement[c] for c in OHLC},volume=None,adj_volume=None)
        cases.append({'intrinio_security_id':NATIVE,'trading_date':d,'original_record':rr,'original_record_sha256':digest(rr),'replacement_record':new,'replacement_record_sha256':digest(new),'preceding_control_date':pre,'following_control_date':post})
    return {'schema':'dmv2-reviewed-opch-amendment/v1','policy_id':POLICY,'status':'REVIEWED_ADMITTED' if reviewed else 'REVIEW_READY_NOT_APPLIED','native_id':NATIVE,'source_id':SOURCE,'artifacts':artifacts,'controls':controls,'cases':cases,
            'limits':['Actual SIP raw OHLC only; multiplier one checked on preceding and following controls.','Volume and adjusted volume are unavailable on corrupt source bars; independent feed volume is not admitted.','Whole SIP source hash and raw response bodies were not reread/authenticated; the published projection and receipt pins are verified.','All original source fields and values persist in this manifest; no existing source file or run is mutated.','Features, labels, normalization, models and predictions require separate causal rebuild-impact review.']}

@dataclass(frozen=True)
class OpchAmendmentResult:
    table: pa.Table
    ledger: tuple
    amendment_sha256: str

@dataclass(frozen=True)
class ReviewedOpchAmendment:
    evidence_root: str
    manifest_relative_path: str
    manifest_sha256: str
    _manifest_json: str

    @property
    def fingerprint(self):return self.manifest_sha256
    @classmethod
    def from_manifest(cls,*,evidence_root,manifest_relative_path,expected_sha256):
        root=Path(evidence_root).resolve();path=root/manifest_relative_path
        pin={'path':manifest_relative_path,'size_bytes':path.stat().st_size,'sha256':expected_sha256,'source_uri':'reviewed-local-opch-amendment'}
        verify_pin(root,pin);m=json.loads(path.read_text())
        need(set(m)=={'schema','policy_id','status','native_id','source_id','artifacts','controls','cases','limits'},'Closed amendment schema required')
        need(m['schema']=='dmv2-reviewed-opch-amendment/v1' and m['policy_id']==POLICY and m['status']=='REVIEWED_ADMITTED' and m['native_id']==NATIVE and m['source_id']==SOURCE,'Amendment identity or admission differs')
        rebuilt=build_manifest(evidence_root=root,artifacts=m['artifacts'],reviewed=True)
        need(canonical(rebuilt)==canonical(m),'Manifest differs from verified actual source/control evidence')
        return cls(str(root),manifest_relative_path,expected_sha256,canonical(m).decode())
    def _verify(self):
        # Reauthenticate evidence each application. Immutable class state cannot
        # turn later changed files into a trusted reusable amendment.
        current=type(self).from_manifest(evidence_root=self.evidence_root,manifest_relative_path=self.manifest_relative_path,expected_sha256=self.manifest_sha256)
        need(current._manifest_json==self._manifest_json,'Loaded amendment state changed')
        return json.loads(self._manifest_json)
    def apply(self,table,*,start,end):return self.apply_with_ledger(table,start=start,end=end).table
    def apply_with_ledger(self,table,*,start,end):
        need(isinstance(table,pa.Table),'Arrow table required');start,end=day(start),day(end);need(start<=end,'Reversed request dates')
        need(len(table.column_names)==len(set(table.column_names)) and set(CORE)<=set(table.column_names),'Unique full original Arrow schema required')
        need(pa.types.is_string(table.schema.field('intrinio_security_id').type) or pa.types.is_large_string(table.schema.field('intrinio_security_id').type),'Native IDs must be strings')
        dates=table['trading_date'].to_pylist();ids=table['intrinio_security_id'].to_pylist()
        need(all(v is not None and isinstance(v,str) and bool(v) for v in ids),'Null or malformed native key')
        dates=[day(v) for v in dates];need(all(start<=v<=end for v in dates),'Input dates outside explicit request')
        need(len(set(zip(ids,dates)))==len(dates),'Duplicate native/date input key')
        manifest=self._verify();cases={c['trading_date']:c for c in manifest['cases']}
        # A present control must still be the exact source used to approve units.
        control_take=[i for i,(sid,d) in enumerate(zip(ids,dates)) if sid==NATIVE and d in manifest['controls']]
        if control_take:
            control_rows=table.select(list(CORE)).take(pa.array(control_take,type=pa.int64())).to_pylist()
            for row in control_rows:
                rr=record(row);spec=manifest['controls'][rr['trading_date']]
                need(digest(rr)==spec['original_record_sha256'] and rr==spec['original_record'],'Present reviewed control source fields changed')
        take=[i for i,(sid,d) in enumerate(zip(ids,dates)) if sid==NATIVE and d in cases]
        if not take:return OpchAmendmentResult(table,(),self.fingerprint)
        original_rows=table.select(list(CORE)).take(pa.array(take,type=pa.int64())).to_pylist()
        edits={c:{} for c in CHANGED};ledger=[]
        for i,row in zip(take,original_rows):
            rr=record(row);case=cases[rr['trading_date']];hh=digest(rr)
            if hh==case['replacement_record_sha256']:
                need(rr==case['replacement_record'],'Replacement hash collision/state differs');state='already_applied_exact'
            else:
                need(hh==case['original_record_sha256'] and rr==case['original_record'],'Original target source fields differ; partial/unreviewed repair refused');state='applied_exact_original'
                for c in CHANGED:edits[c][i]=case['replacement_record'][c]
            ledger.append({'intrinio_security_id':NATIVE,'trading_date':rr['trading_date'],'state':state,'original_record_sha256':case['original_record_sha256'],'replacement_record_sha256':case['replacement_record_sha256'],'amendment_sha256':self.fingerprint,'volume_policy':'unavailable_on_confirmed_corrupt_bar'})
        out=table
        for c,changes in edits.items():
            if not changes:continue
            idx=out.schema.get_field_index(c);field=out.schema.field(idx)
            need((c in OHLC+ADJ and pa.types.is_float64(field.type)) or (c in ('volume','adj_volume') and (pa.types.is_floating(field.type) or pa.types.is_integer(field.type))),'Exact float64 OHLC and numeric volume Arrow schema required')
            # Vectorized Arrow replacement retains all unselected values and
            # physical row order. Null volume requires nullable volume fields.
            mask=pa.array([i in changes for i in range(len(out))]);replacement=pa.array([changes[i] for i in sorted(changes)],type=field.type)
            arr=pc.replace_with_mask(out[c].combine_chunks(),mask,replacement)
            if c in ('volume','adj_volume'):field=field.with_nullable(True)
            out=out.set_column(idx,field,arr)
        need(out.num_rows==table.num_rows and out.column_names==table.column_names,'Row/column preservation failed')
        for c in table.column_names:
            if c not in CHANGED:need(table[c].equals(out[c]),'Unrelated original field changed')
        written=out.select(list(CORE)).take(pa.array(take,type=pa.int64())).to_pylist()
        for row in written:
            rr=record(row);spec=cases[rr['trading_date']]
            need(digest(rr)==spec['replacement_record_sha256'] and rr==spec['replacement_record'],'Replacement cannot be represented exactly in input schema')
        return OpchAmendmentResult(out,tuple(ledger),self.fingerprint)
