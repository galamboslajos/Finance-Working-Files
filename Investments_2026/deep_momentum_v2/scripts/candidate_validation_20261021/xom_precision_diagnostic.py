"""Post-hoc output-only precision compatibility; preserves frozen QA flags."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import pyarrow.parquet as pq

FIELDS=('open','high','low','close')
SOURCE='siblis_isin_b284bb7497616f5afba0'

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def common_interval(raw,adjusted,*,raw_half_width,adjusted_half_width):
    assert len(raw)==len(adjusted)==4
    assert all(math.isfinite(x) and x>max(raw_half_width,adjusted_half_width) for x in raw+adjusted)
    lows=[(y-adjusted_half_width)/(x+raw_half_width) for x,y in zip(raw,adjusted)]
    highs=[(y+adjusted_half_width)/(x-raw_half_width) for x,y in zip(raw,adjusted)]
    low,high=max(lows),min(highs)
    return {'compatible':low<=high+1e-12,'factor_lower':low,'factor_upper':high}

def on_cent_grid(values):
    return all(math.isclose(x*100,round(x*100),rel_tol=0.,abs_tol=1e-6) for x in values)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());plan_digest=sha(plan_path)
    assert plan['schema']=='dmv2-XOM-output-precision-diagnostic-plan/v1'
    assert sha(Path(__file__))==plan['worker_sha256']
    pins=[plan[k] for k in ['quotes_pin','receipt_pin','summary_pin','calendar_pin','quality_pin']]
    def check():
        for pin in pins:
            p=Path(pin['path']);assert not p.is_symlink()
            assert p.stat().st_size==pin['size_bytes'] and sha(p)==pin['sha256']
    check()
    q=Path(plan['quotes_pin']['path'])
    expected=Path('/private/tmp/dmv2-candidate-validation-20261021/xom/financial-result-v1/quotes.parquet')
    assert q.resolve()==expected and q.stat().st_size<4*1024*1024
    receipt=json.loads(Path(plan['receipt_pin']['path']).read_text())
    assert receipt['status']=='PASS_FIELD_AUDIT_COMPLETED' and receipt['quotes_pin']==plan['quotes_pin']
    keys=['security_id','trading_date','query_symbol','adjustment']
    metadata=pq.read_table(q,columns=keys).to_pylist()
    sessions=set(json.loads(Path(plan['calendar_pin']['path']).read_text())['sessions'])
    assert len(metadata)==receipt['financial_python_rows']<=10000
    assert all(x['security_id']==SOURCE and x['query_symbol']=='XOM' and x['adjustment'] in ('raw','all')
               and '2016-01-04'<=x['trading_date']<='2023-12-31' and x['trading_date'] in sessions for x in metadata)
    rows=pq.read_table(q,columns=keys+list(FIELDS)).to_pylist()
    lookup={(x['trading_date'],x['adjustment']):x for x in rows};assert len(lookup)==len(rows)
    quality=json.loads(Path(plan['quality_pin']['path']).read_text())
    flagged={r['trading_date'] for r in quality if 'raw_all_OHLC_price_scales_disagree' in r['flags']}
    result=[]
    for d in sorted({x['trading_date'] for x in rows}):
        raw=[lookup[d,'raw'][f] for f in FIELDS];adj=[lookup[d,'all'][f] for f in FIELDS]
        result.append({'trading_date':d,'original_2bp_flag':d in flagged,
                       'raw_all_four_on_cent_grid':on_cent_grid(raw),'adjusted_all_four_on_cent_grid':on_cent_grid(adj),
                       'adjusted_only_half_cent':common_interval(raw,adj,raw_half_width=0.,adjusted_half_width=.005),
                       'both_half_cent':common_interval(raw,adj,raw_half_width=.005,adjusted_half_width=.005)})
    assert len(flagged)==69 and len(result)==2012
    check();assert sha(plan_path)==plan_digest
    out=Path(args.output);assert not out.exists();out.mkdir(parents=True)
    summary={'schema':'dmv2-XOM-output-precision-compatibility/v1','status':'POST_HOC_DIAGNOSTIC_NOT_A_QA_OVERRIDE',
        'sessions':len(result),'original_2bp_flags_preserved':len(flagged),
        'raw_cent_grid_sessions':sum(r['raw_all_four_on_cent_grid'] for r in result),
        'adjusted_cent_grid_sessions':sum(r['adjusted_all_four_on_cent_grid'] for r in result),
        'all_sessions_adjusted_only_half_cent_compatible':sum(r['adjusted_only_half_cent']['compatible'] for r in result),
        'all_sessions_both_half_cent_compatible':sum(r['both_half_cent']['compatible'] for r in result),
        'flagged_sessions_adjusted_only_half_cent_compatible':sum(r['original_2bp_flag'] and r['adjusted_only_half_cent']['compatible'] for r in result),
        'flagged_sessions_both_half_cent_compatible':sum(r['original_2bp_flag'] and r['both_half_cent']['compatible'] for r in result),
        'assumption':'Each observed adjusted price may represent an unobserved price in [observed-0.005, observed+0.005]. Separate test also gives raw prices the same interval. Closed bounds and 1e-12 factor slack are conservative numerical conventions.',
        'inference_limit':'Compatibility does not establish provider precision, rounding method, adjustment correctness or the cause of any discrepancy. The original 69 flags and frozen 2-bp criterion remain unchanged.',
        'source_values_changed':False,'original_source_rereads':0,'mapping_promoted':False,'plan_sha256':plan_digest}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'per_session.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,str)}))

if __name__=='__main__':main()
