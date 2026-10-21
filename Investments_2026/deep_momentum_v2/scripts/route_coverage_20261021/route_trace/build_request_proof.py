"""Local JSON/text/hash-only proof: exact native IDs were already requested.

This does not decode any Parquet column or inspect/recompute financial values.
"""
from pathlib import Path
import hashlib,json

R=Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
S=Path(__file__).resolve().parent
CASES={'XOM':{'source_security_id':'siblis_isin_b284bb7497616f5afba0','intrinio_security_id':'sec_z5YpGE'},
       'BK':{'source_security_id':'siblis_ticker_company_1c21a7b293fd137c59c9','intrinio_security_id':'sec_gA3NYQ'}}

def pin(p):
    p=Path(p);return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def check_pin(p,expected):
    actual=pin(p)
    assert (actual['sha256'],actual['size_bytes'])==(expected['sha256'].removeprefix('sha256:'),expected['size_bytes'])
    return actual


def build():
    parent=R/'outputs/ret_backtest/2020-2023-20261020-v1/build_identity.json';identity=json.loads(parent.read_text())
    years=[];all_signals=[]
    for annual in identity['annual_sources']:
        directory=R/annual['directory'];audit=directory/'audit.json'
        auditpin=check_pin(audit,annual['audit_pin']);saved=json.loads(audit.read_text())
        files={}
        for name in ('feature_provenance','monthly_provenance','loading_provenance'):
            item=saved['artifacts'][name];p=directory/item['path'];files[name]=(json.loads(p.read_text()),check_pin(p,item))
        feature=files['feature_provenance'][0];monthly=files['monthly_provenance'][0]
        signals=feature['signal_dates'];all_signals.extend(signals)
        assert len(signals)==len(set(signals))==12 and signals==sorted(signals)
        assert feature['paper']['exhaustive_requested_by_signal']=={d:True for d in signals}
        case_rows={}
        for ticker,case in CASES.items():
            native=case['intrinio_security_id'];flags={
                'feature_requested_ids_contains_exact_native':native in feature['requested_ids'],
                'paper_requested_security_ids_contains_exact_native':native in feature['paper']['requested_security_ids'],
                'technical_requested_security_ids_contains_exact_native':native in feature['technical']['requested_security_ids'],
                'forward_monthly_requested_security_ids_contains_exact_native':native in monthly['requested_security_ids']}
            assert all(flags.values())
            case_rows[ticker]={**case,**flags,'all_twelve_signal_requests_exhaustive':True}
        years.append({'outcome_year':annual['year'],'source_directory':str(directory),
            'original_annual_audit_pin':auditpin,'provenance_pins':{k:v[1] for k,v in files.items()},
            'signal_dates':signals,'first_signal':signals[0],'last_signal':signals[-1],
            'outcome_start_month':monthly['start_month'],'outcome_end_month':monthly['end_month'],
            'feature_requested_native_count':len(feature['requested_ids']),
            'monthly_requested_native_count':len(monthly['requested_security_ids']),
            'cases':case_rows})
    assert [row['outcome_year'] for row in years]==list(range(2010,2024))
    assert len(all_signals)==len(set(all_signals))==168
    code_lines={
        'src/dmv2/data.py':[(104,119,'Load source bridge and native links from their pinned tables, independently of quotes.')],
        'src/dmv2/universe.py':[(146,198,'Exact source_product=sp_composite1500 and source_security_id to unified ID, then exact unified ID to native link; structural statuses preserved.'),
            (239,286,'Quote metadata is exact-date/native; original requested empty native yields missing_price, not_requested remains separate; no alias fallback.')],
        'src/dmv2/full_loading.py':[(347,360,'Union identity-admitted native IDs from original monthly and daily context cohorts BEFORE prices.'),
            (389,410,'PriceLoader receives exact prepared requested native IDs and bounded dates; source quote inventory does not select members.')],
        'src/dmv2/loading.py':[(249,295,'Pinned candidate Parquet row groups use date/native min/max, then exact metadata masks; only requested matches returned.'),
            (375,393,'PriceLoader uses intrinio_daily_prices and exact intrinio_security_id filter, with no ticker/ISIN/FIGI fallback.')],
        'src/dmv2/full_panel.py':[(161,177,'Preserve complete annual warmup separately; outcome-year daily_prices_raw is disjoint, avoiding losing prior December anchors.')],
        'src/dmv2/identity_audit.py':[(185,240,'Source-alias ISIN/ticker agreement and declared lookup query retained separately; independent returned-provider ISIN not certified.')]
    }
    return {'schema':'dmv2-xom-bk-request-and-code-trace/v1','status':'PASS_METADATA_ONLY',
        'master_protocol_pin':pin(S.parent/'protocol.json'),'original_RET_identity_pin':pin(parent),
        'case_native_ids':CASES,'outcome_years':list(range(2010,2024)),
        'signal_dates_count':168,'first_signal_date':min(all_signals),'last_signal_date':max(all_signals),
        'all_four_request_ledgers_contain_each_exact_native_all_fourteen_years':True,
        'quote_absence_is_not_missing_request':True,
        'quote_absence_actual_price_inventory_proven_here':False,
        'prior_full_metadata_audit_claim':'Each case168original months,zero matchingstart/endquotes,zero predictioneligible. This proof establishes explicit requests, not identity correctness or independent price absence.',
        'years':years,'code_trace':[{'pin':pin(R/path),'line_spans':[{'start':a,'end':b,'meaning':text} for a,b,text in ranges]} for path,ranges in code_lines.items()],
        'producer_lookup_evidence_fields':['query_identifier','query_identifier_type','observed_tickers_json','observed_isins_json',
            'http_status','source_response_hash','source_relative_path','source_identity_run_id','source_identity_manifest_hash'],
        'returned_provider_master_fields':['intrinio_security_id','company_id','requested_identifier','ticker','composite_ticker','figi',
            'composite_figi','share_class_figi','primary_security','primary_listing','type','code','exchange_mic','retrieved_at_utc','source_run_id','source_response_hash','source_relative_path'],
        'provider_master_has_ISIN_column':False,
        'interpretation':'A structurally resolved native link can point to no retained price history. Current loader never tries another alias automatically. Stage1 identity metadata and stage2 native/date inventory are needed to distinguish wrong security/shareclass route from acquisition/delivery absence.',
        'financial_columns_decoded':[],'new_cloud_reads':0,'models_fitted':0,'backtests_run':0,
        'mappings_promoted_or_modified':False,'existing_R_files_modified':False}

if __name__=='__main__':
    value=build();p=S/'request_and_code_trace.json'
    assert not p.exists();p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'path':str(p),'sha256':pin(p)['sha256'],'years':len(value['years']),'signals':168,
        'all_exact_natives_requested':True,'financial_columns_decoded':[]}))
