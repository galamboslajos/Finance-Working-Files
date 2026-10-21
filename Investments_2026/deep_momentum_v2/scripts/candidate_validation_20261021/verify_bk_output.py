"""Independent output-only QA; never opens original source price files."""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse
import hashlib
import json
import math
import pyarrow as pa
import pyarrow.parquet as pq

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def valid(x):
    return isinstance(x,(int,float)) and math.isfinite(x)

def verify(table, sessions):
    keys=table.select(['intrinio_security_id','trading_date']).to_pylist()
    assert table.num_rows<=10000
    seen=set()
    for row in keys:
        native=row['intrinio_security_id']
        day=str(row['trading_date'])
        assert native=='sec_zq1J05' and '2007-07-02'<=day<='2023-12-31'
        assert day in sessions and day not in seen
        seen.add(day)
    # Scope is proven with metadata before any financial scalar conversion.
    selected=table.to_pylist()
    selected.sort(key=lambda x:str(x['trading_date']))
    stats={'rows':len(selected),'missing_sessions':sorted(set(sessions)-seen),
           'price_null_or_nonfinite':{},'price_nonpositive':{},
           'volume_null_or_nonfinite':{},'volume_negative':{},'volume_zero':{},
           'raw_OHLC_failures':[],'adjusted_OHLC_failures':[],
           'within_row_adjustment_ratio_failures':[],
           'daily_jump_over_30pct_dates':[],
           'source_factor_vs_adj_close_ratio_mismatch_dates':[],
           'split_ratio_nontrivial_dates':[],'positive_dividend_dates':[]}
    for field in ['open','high','low','close','adj_open','adj_high','adj_low','adj_close']:
        stats['price_null_or_nonfinite'][field]=sum(not valid(r[field]) for r in selected)
        stats['price_nonpositive'][field]=sum(valid(r[field]) and r[field]<=0 for r in selected)
    for field in ['volume','adj_volume']:
        stats['volume_null_or_nonfinite'][field]=sum(not valid(r[field]) for r in selected)
        stats['volume_negative'][field]=sum(valid(r[field]) and r[field]<0 for r in selected)
        stats['volume_zero'][field]=sum(valid(r[field]) and r[field]==0 for r in selected)
    session_positions={d:i for i,d in enumerate(sorted(sessions))}
    last=None
    for row in selected:
        day=str(row['trading_date'])
        for prefix,name in [('', 'raw_OHLC_failures'),('adj_','adjusted_OHLC_failures')]:
            values=[row[prefix+x] for x in ['open','high','low','close']]
            if all(valid(x) for x in values):
                o,h,l,c=values
                eps=1e-8
                if l>min(o,c)+eps or h<max(o,c)-eps or l>h+eps:
                    stats[name].append(day)
        ratios=[]
        for field in ['open','high','low','close']:
            raw,adj=row[field],row['adj_'+field]
            if valid(raw) and raw>0 and valid(adj): ratios.append(adj/raw)
        if len(ratios)==4 and not all(math.isclose(ratios[0],x,rel_tol=1e-7,abs_tol=1e-8) for x in ratios):
            stats['within_row_adjustment_ratio_failures'].append(day)
        if valid(row['close']) and row['close']>0 and valid(row['adj_close']) and valid(row['factor']):
            if not math.isclose(row['factor'],row['adj_close']/row['close'],rel_tol=1e-7,abs_tol=1e-8):
                stats['source_factor_vs_adj_close_ratio_mismatch_dates'].append(day)
        if valid(row['split_ratio']) and row['split_ratio'] not in (0,1): stats['split_ratio_nontrivial_dates'].append(day)
        if valid(row['dividend']) and row['dividend']>0: stats['positive_dividend_dates'].append(day)
        if last is not None and session_positions[day]-session_positions[str(last['trading_date'])]==1:
            if valid(last['adj_close']) and last['adj_close']>0 and valid(row['adj_close']):
                if abs(row['adj_close']/last['adj_close']-1)>.30:
                    stats['daily_jump_over_30pct_dates'].append(day)
        last=row
    return stats

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    config_path=Path(args.config)
    config=json.loads(config_path.read_text())
    expected_root=Path('/private/tmp/dmv2-candidate-validation-20261021/core')
    q=Path(config['quotes_pin']['path']); c=Path(config['calendar_pin']['path']); o=Path(args.output)
    receipt=Path(config['core_run_receipt_pin']['path'])
    assert q.resolve().is_relative_to(expected_root.resolve())
    assert receipt.resolve().is_relative_to(q.parent.resolve())
    assert c.resolve()==expected_root/'calendar_metadata.json'
    assert q.name=='candidate_postmerger_XNYS_rows.parquet' and receipt.name=='receipt.json'
    def check_pins():
        for pin in [config['quotes_pin'],config['calendar_pin'],config['core_run_receipt_pin']]:
            p=Path(pin['path'])
            assert p.is_file() and not p.is_symlink()
            assert p.stat().st_size==pin['size_bytes'] and digest(p)==pin['sha256']
    check_pins()
    core_receipt=json.loads(receipt.read_text())
    assert core_receipt['status']=='COMPLETE_DIAGNOSTIC_AUDIT_NOT_PROMOTED'
    assert core_receipt['candidate_rows_pin']==config['quotes_pin']
    assert not o.exists()
    o.mkdir(parents=True)
    assert q.stat().st_size<16*1024*1024 and c.stat().st_size<1024*1024
    config_digest=digest(config_path)
    plan={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'financial_scope':'New bounded BK output only; no original price reread.','sealed_config_sha256':config_digest,'quotes':config['quotes_pin'],'calendar':config['calendar_pin'],'core_receipt':config['core_run_receipt_pin'],'price_ratio_rtol':1e-7,'price_ratio_atol':1e-8,'OHLC_atol':1e-8,'jump_diagnostic_threshold':.30,'max_output_rows':10000}
    (o/'read_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    payload=json.loads(c.read_text())
    # Exact calendar payload key is set explicitly by root after metadata review.
    days=payload['sessions']
    sessions=[str(d) for d in days if '2007-07-02'<=str(d)<='2023-12-31']
    assert len(sessions)==len(set(sessions))
    table=pq.read_table(q,columns=['intrinio_security_id','trading_date'])
    verify_keys=table.to_pylist()
    assert len(verify_keys)<=10000
    assert all(x['intrinio_security_id']=='sec_zq1J05' and '2007-07-02'<=str(x['trading_date'])<='2023-12-31' and str(x['trading_date']) in sessions for x in verify_keys)
    columns=['intrinio_security_id','trading_date','open','high','low','close','volume','adj_open','adj_high','adj_low','adj_close','adj_volume','factor','split_ratio','dividend']
    table=pq.read_table(q,columns=columns)
    stats=verify(table,set(sessions))
    assert stats['rows']==core_receipt['financial_rows']
    check_pins()
    assert digest(config_path)==config_digest
    result={'schema':'dmv2-BK-independent-output-field-QA/v1','scope':'Saved candidate session-only output; no original financial source reread.','price_ratio_rtol':1e-7,'price_ratio_atol':1e-8,'stats':stats,'mapping_promoted':False,'financial_values_after_2023':False,'source_factor_comparison_is_diagnostic_only':True,'volume_factor_semantics_certified':False}
    (o/'independent_QA.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'OUTPUT_QA_COMPLETED','rows':stats['rows'],'missing_sessions':len(stats['missing_sessions']),'raw_OHLC_failures':len(stats['raw_OHLC_failures']),'adjusted_OHLC_failures':len(stats['adjusted_OHLC_failures']),'ratio_diagnostic_failures':len(stats['within_row_adjustment_ratio_failures'])}))

if __name__=='__main__': main()
