"""Footer/JSON/hash-only plan generation; never decodes Parquet columns."""
from pathlib import Path
import hashlib
import json
import pyarrow.parquet as pq
from learning_repair_counterfactual import TARGET_FIELDS, MONTH_FIELDS

ROOT=Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
STAGE=Path(__file__).resolve().parent
OLD='outputs/ret_backtest/2020-2023-20261020-v1'
ANNUAL='outputs/walkforward_source_corrected/2020-2023-20261016-v1/years/2021'


def pin(path):
    p=Path(path); p=p if p.is_absolute() else ROOT/p
    digest=hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':digest.hexdigest()}


def projection(name,path,metadata,financial,filters):
    p=ROOT/path; footer=pq.ParquetFile(p); cols=list(footer.schema_arrow.names)
    def cost(selected):
        return sum(footer.metadata.row_group(g).column(cols.index(c)).total_compressed_size
                   for g in range(footer.metadata.num_row_groups) for c in selected)
    return {'name':name,'pin':pin(p),'rows':footer.metadata.num_rows,
        'metadata_columns':metadata,'numeric_columns':financial,'filter':filters,
        'metadata_projected_bytes':cost(metadata),'numeric_projected_bytes':cost(financial),
        'schema':[[c,str(footer.schema_arrow.field(c).type)] for c in cols],
        'numeric_decode_contract':'Validate ALL projected date metadata first; Arrow-filter exact keys before Python financial conversion. Whole selected row-group chunks charged, no discarded financial values analyzed.'}


def build():
    pre=ROOT/OLD/'preflight';audit=json.loads((pre/'audit.json').read_text())
    tables=[projection('targets',ANNUAL+'/targets.parquet',
        [x for x in TARGET_FIELDS if x not in ('target_class','label_return')],list(TARGET_FIELDS),
        {'outcome_month':'2021-11'}),
        projection('class_means',OLD+'/preflight/source_month_class_means.parquet',
        ['outcome_month','outcome_end','class_id'],list(MONTH_FIELDS),{'outcome_month_range':['2010-01','2023-11']}),
        projection('start_quote',ANNUAL+'/daily_prices_raw.parquet',
        ['intrinio_security_id','trading_date','frequency','intraperiod','price_quality_status','adjusted_ohlc_bounds_valid','raw_ohlc_bounds_valid'],
        ['intrinio_security_id','trading_date','adj_close'],{'intrinio_security_id':'sec_zMM1vd','trading_date':'2021-10-29'}),
        projection('end_quote','data/derived/ret_repairs/20261020-v1/reviewed_execution_quotes.parquet',
        ['intrinio_security_id','trading_date','frequency','intraperiod','price_quality_status','adjusted_ohlc_bounds_valid','raw_ohlc_bounds_valid'],
        ['intrinio_security_id','trading_date','adj_close'],{'intrinio_security_id':'sec_zMM1vd','trading_date':'2021-11-30'})]
    states=[]
    for relative,expected in sorted(audit['artifacts'].items()):
        if relative.startswith('payoffs/') and relative.endswith('/payoff_state.json'):
            p=pre/relative;state=json.loads(p.read_text());window=state['provenance']
            pins=[pin(pre/relative),pin(pre/relative.replace('payoff_state.json','payoffs.parquet')),
                pin(pre/relative.replace('payoff_state.json','monthly.parquet'))]
            for oldpin in pins:
                rel=str(Path(oldpin['path']).relative_to(pre)); saved=audit['artifacts'][rel]
                if (oldpin['sha256'],oldpin['size_bytes']) != (saved['sha256'],saved['size_bytes']):raise ValueError('Old payoff bytepin changed')
            states.append({'signal_date':window['signal_date'],'fingerprint':state['fingerprint'],'pins':pins})
    if len(states)!=48:raise ValueError('Expected48 old vectors')
    source_pins=[pin(ROOT/OLD/'build_identity.json'),pin(pre/'audit.json'),
        pin(ROOT/OLD/'verification/ret_backtest.json'),pin(ROOT/ANNUAL/'target_state.json'),
        pin(ROOT/'data/derived/ret_repairs/20261020-v1/repair_manifest.json'),
        pin(ROOT/'configs/rf_expanded_tuning_20261020.json')]
    return {'schema':'dmv2-learning-conditional-numeric-plan/v1','project_root':str(ROOT),
        'master_protocol_pin':pin(STAGE.parent/'protocol.json'),'source_pins':source_pins,
        'tables':tables,'old_vectors':states,'original_source_security_id':'siblis_ticker_company_8500ae2d7e99a52f8713',
        'original_native':'sec_zMM1vd','outcome_month':'2021-11','signal_date':'2021-10-29',
        'outcome_start':'2021-11-01','outcome_end':'2021-11-30',
        'max_projected_bytes':268435456,'max_rss_bytes':4294967296,'max_wall_seconds':600,
        'numeric_threads':1,'automatic_retries':0,
        'projected_compressed_bytes':sum(t['metadata_projected_bytes']+t['numeric_projected_bytes'] for t in tables),
        'numeric_stage_status':'NOT_RUN','raw_payoff_bytes_preserved_not_decoded':True,
        'baseline_contract':'Authenticate original RET PASS and preflight audit; bytehash all48 saved payoff_state.json+payoffs.parquet+monthly.parquet before/after. Authenticate original November Arrow target fingerprint; reproduce every saved old class/availability/status. Reproduce November class means exactly versus saved167-month ledger. Independently reconstruct every old120-month vector and canonical table+monthly+provenance logical fingerprint from saved means+old state JSON; require exact saved fingerprint before any shadow output.',
        'code_pins':[pin(STAGE/name) for name in ('learning_repair_counterfactual.py','run_counterfactual_audit.py')]}

if __name__=='__main__':
    result=build();destination=STAGE/'numeric_read_plan.json'
    if destination.exists():raise SystemExit('Refuse to overwrite frozen plan')
    destination.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'path':str(destination),'projected_compressed_bytes':result['projected_compressed_bytes'],'sha256':pin(destination)['sha256']}))
