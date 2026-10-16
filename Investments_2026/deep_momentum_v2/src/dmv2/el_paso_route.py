"""Exact non-promoting El Paso source-route amendment; no price reads or fits."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date,datetime
import hashlib,json
from pathlib import Path,PurePosixPath
from collections import defaultdict
import pyarrow as pa
import pyarrow.compute as pc

SOURCE='siblis_ticker_company_953e9b44770b9402bccb'
UNIFIED='equity_us_7ddb7953d9db219ea493f217'
OLD='sec_g2q5mL';NEW='sec_XO6Eab'
START='2008-01-04';END='2012-05-24'
AMENDMENT_ID='dmv2-el-paso-empire-route-correction-v1'
EL_PASO_URL='https://www.sec.gov/Archives/edgar/data/225777/000092881611001038/a_43npx.htm'
EMPIRE_URL='https://www.sec.gov/Archives/edgar/data/887396/000107261323000285/empire_form10k123122.htm'
AUDIT_COLUMNS={'el_paso_route_amendment_id':pa.string(),'el_paso_route_amendment_sha256':pa.string(),'el_paso_route_original_native_id':pa.string(),'el_paso_route_replacement_native_id':pa.string(),'el_paso_route_reviewed_cusip':pa.string(),'el_paso_route_lookup_identifier':pa.string(),'el_paso_route_historical_identity_certified':pa.bool_()}
ARTIFACT_ROLES={'proposal','primary_findings','provider_metadata','source_bridge_metadata'}
DEFINITION={'schema':'dmv2-el-paso-route-definition/v1','amendment_id':AMENDMENT_ID,'source_security_id':SOURCE,'unified_security_id':UNIFIED,'original_native_id':OLD,'replacement_native_id':NEW,'minimum_cohort_date':START,'maximum_cohort_date':END,'reviewed_historical_cusip':'28336L109','replacement_cached_lookup_identifier':'US28336L1098','original_cached_lookup_identifier':'US2920343033','replacement_figi':'BBG000DST4Q5','original_figi':'BBG000BBS4G6','effect':'correct_exact_source_native_route_preserve_all_rows_flags_unified_ids','historical_knowledge_time_certified':False,'identity_canonically_promoted':False,'terminal_entitlement_or_exit_supplied':False}
class RouteError(ValueError):pass

def need(ok,msg):
    if not ok:raise RouteError(msg)

def day(v):
    if isinstance(v,datetime):need(v.time()==datetime.min.time() and v.tzinfo is None,'Canonical date-only metadata required');v=v.date()
    if isinstance(v,date):v=v.isoformat()
    need(isinstance(v,str),'ISO cohort date required')
    try:d=date.fromisoformat(v)
    except ValueError as e:raise RouteError('Invalid cohort date') from e
    need(d.isoformat()==v and v<'2024-01-01','Protected or noncanonical cohort date');return v

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):return 'sha256:'+hashlib.sha256(Path(path).read_bytes()).hexdigest()

def artifact_pin(root,path,source_uri):
    root=Path(root).resolve();path=Path(path).resolve();need(path.is_relative_to(root),'Evidence outside root');return {'path':path.relative_to(root).as_posix(),'size_bytes':path.stat().st_size,'sha256':sha(path),'source_uri':source_uri}

def verify_pin(root,pin):
    need(isinstance(pin,dict) and set(pin)=={'path','size_bytes','sha256','source_uri'},'Closed metadata evidence pin required')
    rel=pin['path'];need(isinstance(rel,str) and rel and '\\' not in rel,'Relative POSIX path required');p=PurePosixPath(rel);need(not p.is_absolute() and all(x not in ('','.','..') for x in rel.split('/')),'Evidence path traversal refused');path=Path(root).joinpath(*p.parts)
    need(path.resolve().is_relative_to(Path(root).resolve()) and not any(q.is_symlink() for q in [path,*path.parents]),'Escaped or linked evidence refused')
    need(type(pin['size_bytes']) is int and 0<pin['size_bytes']<=2*1024*1024,'Bounded metadata evidence required');need(path.is_file() and path.stat().st_size==pin['size_bytes'] and sha(path)==pin['sha256'],'Evidence pin changed');need(isinstance(pin['source_uri'],str) and bool(pin['source_uri']),'Source reference required');return json.loads(path.read_text())

def build_route_manifest(*,evidence_root,artifacts,reviewed=False):
    need(set(artifacts)==ARTIFACT_ROLES,'Exactly four metadata/primary evidence roles required');root=Path(evidence_root).resolve();data={k:verify_pin(root,p) for k,p in artifacts.items()};proposal=data['proposal']
    proposal_fields={'schema','status','source_security_id','source_named_issuer','source_membership_labels','original_native_id','original_native_issuer','original_provider_lookup_isin','original_native_figi','candidate_native_id','candidate_native_issuer','candidate_provider_lookup_isin','candidate_native_figi','candidate_security_master_last_quote','applicability','collision_check','evidence','replacement_history_already_downloaded','required_before_new_build','source_incorrect_route_confirmed','newproviderlookup_performed','canonical_identity_promoted','old_artifacts_modified','model_fits'}
    need(set(proposal)==proposal_fields,'Closed exact v1 route proposal required')
    need(proposal.get('schema')=='dmv2-el-paso-identity-route-override-proposal/v1' and proposal.get('source_security_id')==SOURCE and proposal.get('original_native_id')==OLD and proposal.get('candidate_native_id')==NEW,'Wrong closed El Paso proposal identity')
    need(proposal.get('source_incorrect_route_confirmed') is True and proposal.get('canonical_identity_promoted') is False and proposal.get('old_artifacts_modified') is False and proposal.get('model_fits')==0,'Proposal must remain non-promoting and immutable')
    need(proposal.get('candidate_provider_lookup_isin')=='US28336L1098' and proposal.get('original_provider_lookup_isin')=='US2920343033' and proposal.get('candidate_native_figi')==DEFINITION['replacement_figi'] and proposal.get('original_native_figi')==DEFINITION['original_figi'],'Proposal cached lookup metadata differs')
    proof=data['primary_findings'];need(set(proof)=={'schema','status','record_kind','el_paso','empire'},'Closed primary-findings schema required');need(proof['schema']=='dmv2-el-paso-primary-findings/v1' and proof['status']=='REVIEWED_PRIMARY_SOURCE_FACTS' and proof['record_kind']=='documented_primary_source_findings_not_raw_provider_response','Reviewed documented primary findings required')
    for name,url,cusip,issuer in [('el_paso',EL_PASO_URL,'28336L109','El Paso Corporation'),('empire',EMPIRE_URL,'292034303','Empire Petroleum Corporation')]:
        finding=proof[name];need(set(finding)=={'url','issuer','ticker','cusip','fact_date','evidence_scope'},'Closed issuer primary finding required');need(finding['url']==url and finding['issuer']==issuer and finding['ticker']=='EP' and finding['cusip']==cusip and finding['evidence_scope']=='issuer_and_security_identifier_only','Primary issuer/CUSIP facts differ')
    need(proof['el_paso']['fact_date']=='2011-05-17' and proof['empire']['fact_date']=='2022-03-08','Primary evidence date differs')
    metadata=data['provider_metadata'];need(isinstance(metadata,list),'Cached native metadata list required')
    for native,figi,isin in [(OLD,DEFINITION['original_figi'],'US2920343033'),(NEW,DEFINITION['replacement_figi'],'US28336L1098')]:
        rows=[r for r in metadata if r.get('intrinio_security_id')==native];need(len(rows)==1,'Exactly one cached native response required');r=rows[0]
        need(r.get('figi')==figi and r.get('requested_identifier')==native and r.get('ticker')=='EP' and ('query_identifier='+isin+'-') in r.get('source_relative_path',''),'Cached native FIGI/query-literal evidence differs')
        need(isinstance(r.get('source_response_hash'),str) and len(r['source_response_hash'])==64 and isinstance(r.get('source_run_id'),str) and bool(r['source_run_id']),'Cached native response lineage required')
    bridges=data['source_bridge_metadata'];need(isinstance(bridges,list),'Source bridge metadata list required')
    rows=[r for r in bridges if r.get('source_security_id')==SOURCE and r.get('source_product_id')=='sp_composite1500'];need(len(rows)==1,'Exact S&P1500 original source bridge required');r=rows[0]
    need(r.get('unified_security_id')==UNIFIED and r.get('identity_key')=='ticker_company:EP|EL PASO' and r.get('company_name')=='El Paso Corp.' and r.get('last_membership_date')==END and r.get('first_membership_date')<=START,'Original issuer/unified/membership scope differs')
    return {'schema':'dmv2-reviewed-el-paso-route/v1','definition':DEFINITION,'status':'REVIEWED_ADMITTED' if reviewed else 'REVIEW_READY_NOT_APPLIED','artifacts':artifacts,'limits':['Provider lookup identifiers are historical query-literal metadata, not a returned ISIN field or a new provider lookup.','Primary facts distinguish issuers; complete historical identity, knowledge-time and survivorship certification remain unavailable.','Membership rows, producer flags, source/unified identities and terminal missing outcomes remain preserved.','Prices and merger consideration are outside this metadata amendment; corrected native-price requests require explicit separate source admission.']}

@dataclass(frozen=True)
class RouteResult:
    table:pa.Table
    ledger:tuple
    summary:dict

@dataclass(frozen=True)
class ElPasoRouteAmendment:
    evidence_root:str
    manifest_relative_path:str
    manifest_sha256:str
    _manifest_json:str
    @property
    def fingerprint(self):return self.manifest_sha256
    @classmethod
    def from_manifest(cls,*,evidence_root,manifest_relative_path,expected_sha256):
        root=Path(evidence_root).resolve();path=root/manifest_relative_path;pin={'path':manifest_relative_path,'size_bytes':path.stat().st_size,'sha256':expected_sha256,'source_uri':'reviewed-local-el-paso-route'};m=verify_pin(root,pin)
        need(set(m)=={'schema','definition','status','artifacts','limits'} and m['schema']=='dmv2-reviewed-el-paso-route/v1' and m['definition']==DEFINITION and m['status']=='REVIEWED_ADMITTED','Closed admitted route manifest required')
        rebuilt=build_route_manifest(evidence_root=root,artifacts=m['artifacts'],reviewed=True);need(canonical(rebuilt)==canonical(m),'Route manifest differs from verified evidence');return cls(str(root),manifest_relative_path,expected_sha256,canonical(m).decode())
    def _verify(self):
        other=type(self).from_manifest(evidence_root=self.evidence_root,manifest_relative_path=self.manifest_relative_path,expected_sha256=self.fingerprint);need(other._manifest_json==self._manifest_json,'Loaded route state changed')
    def apply(self,table,*,date_column='signal_date',all_original_members=False,original_counts=None):
        return self.apply_with_ledger(table,date_column=date_column,all_original_members=all_original_members,original_counts=original_counts).table
    def apply_with_ledger(self,table,*,date_column='signal_date',all_original_members=False,original_counts=None):
        need(isinstance(table,pa.Table),'Arrow original cohort table required');need(all_original_members is True,'Caller must supply ALL original members, before eligibility/price filters')
        need(date_column in ('signal_date','as_of','trading_date'),'Explicit cohort date column required');need(len(table.column_names)==len(set(table.column_names)),'Duplicate Arrow column names refused')
        required={'source_security_id','intrinio_security_id','unified_security_id','identity_status','identity_review_required','resolution_status',date_column};need(required<=set(table.column_names),'Complete source/native/unified/producer metadata required')
        # Inspect date/identity metadata only; financial columns are never read.
        cols=list(required|({'source_identity_status'}&set(table.column_names))|set(AUDIT_COLUMNS).intersection(table.column_names));rows=table.select(cols).to_pylist();days=[day(r[date_column]) for r in rows]
        for alias in ('signal_date','as_of','trading_date'):
            if alias in table.column_names:
                aliasdays=[day(v) for v in table[alias].to_pylist()];need(aliasdays==days,'Conflicting cohort date aliases')
        sources=[r['source_security_id'] for r in rows];need(all(isinstance(s,str) and bool(s.strip()) for s in sources),'Every original member requires stable source identity')
        routes=[r['intrinio_security_id'] for r in rows];need(all(v is None or (isinstance(v,str)) for v in routes),'Malformed native route')
        perday=defaultdict(set)
        for d,s in zip(days,sources):perday[d].add(s)
        if original_counts is not None:
            need(isinstance(original_counts,dict),'Original denominator mapping required');need(set(original_counts)==set(perday) and all(type(original_counts[d]) is int and original_counts[d]==len(perday[d]) for d in perday),'Original cohort denominators differ; filtered input refused')
        target=[i for i,s in enumerate(sources) if s==SOURCE]
        for i in target:
            r=rows[i];need(START<=days[i]<=END,'El Paso source row outside approved warmup/membership scope');need(r['unified_security_id']==UNIFIED,'Exact original El Paso unified ID changed');need(routes[i] in (OLD,NEW),'Unknown El Paso native route; do not overwrite another correction')
            need(r['identity_status']=='needs_review' and r['identity_review_required'] is True and r['resolution_status']=='resolved','Producer review/resolution metadata differs; no identity promotion allowed')
            if 'source_identity_status' in r:need(r['source_identity_status']=='needs_review','Original source review flag differs')
        self._verify()
        expected={'el_paso_route_amendment_id':AMENDMENT_ID,'el_paso_route_amendment_sha256':self.fingerprint,'el_paso_route_original_native_id':OLD,'el_paso_route_replacement_native_id':NEW,'el_paso_route_reviewed_cusip':'28336L109','el_paso_route_lookup_identifier':'US28336L1098','el_paso_route_historical_identity_certified':False}
        existing=set(AUDIT_COLUMNS).intersection(table.column_names);need(not existing or existing==set(AUDIT_COLUMNS),'Partial route audit schema refused')
        for i in target:
            if routes[i]==NEW:need(all(rows[i].get(c)==v for c,v in expected.items()),'New route lacks exact idempotent amendment audit')
            else:need(all(rows[i].get(c) is None for c in AUDIT_COLUMNS),'Original route has partial/foreign amendment audit')
        for i in range(len(rows)):
            if sources[i]!=SOURCE:need(all(rows[i].get(c) is None for c in AUDIT_COLUMNS),'Amendment audit attached to unrelated source')
        newroutes=list(routes)
        for i in target:newroutes[i]=NEW
        # Count all original source routes, including blocked members. Duplicate
        # sleeve rows for the SAME original source do not create a false collision.
        collisions=defaultdict(set)
        for d,s,p in zip(days,sources,newroutes):
            if p:collisions[d,p].add(s)
        bad=[(d,p,sorted(ss)) for (d,p),ss in collisions.items() if p==NEW and len(ss)>1 and SOURCE in ss]
        need(not bad,'Corrected native collides with another ORIGINAL source; entire amendment refused')
        out=table;idx=out.schema.get_field_index('intrinio_security_id');field=out.schema.field(idx);need(pa.types.is_string(field.type) or pa.types.is_large_string(field.type),'Native route Arrow string schema required');out=out.set_column(idx,field,pa.array(newroutes,type=field.type))
        for c,t in AUDIT_COLUMNS.items():
            values=[expected[c] if s==SOURCE else None for s in sources]
            if c in out.column_names:
                need(out[c].type==t,'Route audit Arrow schema differs');out=out.set_column(out.schema.get_field_index(c),out.schema.field(c),pa.array(values,type=t))
            else:out=out.append_column(c,pa.array(values,type=t))
        need(out.num_rows==table.num_rows,'Original row count changed')
        for c in table.column_names:
            if c not in {'intrinio_security_id',*AUDIT_COLUMNS}:need(out[c].equals(table[c]),'Original producer/source field changed')
        ledger=tuple({'source_security_id':SOURCE,'date':days[i],'original_native_id':OLD,'replacement_native_id':NEW,'state':'already_applied_exact' if routes[i]==NEW else 'applied_exact_old_route','amendment_sha256':self.fingerprint,'all_original_distinct_source_collision_check':'passed'} for i in target)
        return RouteResult(out,ledger,{'retained_original_rows':len(rows),'target_rows':len(target),'distinct_target_source_dates':len({days[i] for i in target}),'all_original_distinct_sources_by_date':{d:len(s) for d,s in sorted(perday.items())},'full_cohort_scope_declared':True,'original_denominators_supplied':original_counts is not None,'producer_flags_changed':False,'canonical_identity_promoted':False,'financial_values_read':False,'historical_knowledge_time_certified':False})
    def requested_ids(self,table,*,date_column='signal_date',all_original_members=False,original_counts=None,admitted_only=True):
        """Derive requests from amended SOURCE cohorts; never globally map OLD→NEW.

        Another genuine Empire source may still require OLD in the same request.
        This helper authorizes no price read or bypass of existing source bounds.
        """
        amended=self.apply(table,date_column=date_column,all_original_members=all_original_members,original_counts=original_counts)
        if admitted_only:
            need('identity_eligible' in amended.column_names,'Explicit identity eligibility required for admitted-only requests');mask=pc.equal(amended['identity_eligible'],True);amended=amended.filter(pc.fill_null(mask,False))
        return tuple(sorted({p for p in amended['intrinio_security_id'].to_pylist() if isinstance(p,str) and p}))
