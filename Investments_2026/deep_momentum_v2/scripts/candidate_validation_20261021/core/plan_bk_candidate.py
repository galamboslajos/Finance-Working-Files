"""Footer/hash/JSON-only exact BK read plan. This never decodes financial columns."""
from pathlib import Path
import json,hashlib
import pyarrow.parquet as pq
from bk_field_audit import COLUMNS,KEYS,NATIVE,START,END,TOLERANCE
S=Path(__file__).resolve().parent

def pin(path):
 p=Path(path).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':h.hexdigest()}

def build():
 protocol=S.parent/'protocol.json';footer_plan=S/'metadata_inspection_plan.json'
 f=json.loads(footer_plan.read_text());assert len(f['source_objects'])==1;obj=f['source_objects'][0]
 pf=pq.ParquetFile(obj['path']);names=pf.schema_arrow.names
 def cost(cols):return sum(pf.metadata.row_group(i).column(names.index(c)).total_compressed_size for i in range(pf.metadata.num_row_groups) for c in cols)
 prior=Path('/private/tmp/dmv2-route-coverage-20261021/key_inventory/result-v1/date_keys.parquet')
 metadata_plan={'schema':'dmv2-BK-prior-key-footer-metadata-plan/v1','path':str(prior),
  'allowed':'Onlynative/date schema and compressed-key chunk sizes; no column data.'}
 md=S/'prior_key_footer_metadata_plan.json'
 if not md.exists():md.write_text(json.dumps(metadata_plan,sort_keys=True,indent=2)+'\n')
 pp=pq.ParquetFile(prior);prior_cost=sum(pp.metadata.row_group(i).column(pp.schema_arrow.names.index(c)).total_compressed_size for i in range(pp.metadata.num_row_groups) for c in KEYS)
 calendar=S/'calendar_metadata.json';sessions=json.loads(calendar.read_text())['sessions']
 return {'schema':'dmv2-BK-field-plan/v1','status':'FROZEN_SUBPLAN_NO_FINANCIAL_READ_YET',
  'protocol_pin':pin(protocol),'native':NATIVE,'metadata_start':'2007-01-01','metadata_end':END,
  'financial_start':START,'financial_end':END,'financial_XNYS_only':True,
  'source_object':obj,'candidate_rows':pf.metadata.num_rows,'source_files':1,
  'metadata_projection':list(KEYS),'financial_projection':list(COLUMNS),
  'projected_schema':[[c,str(pf.schema_arrow.field(c).type)] for c in COLUMNS],
  'metadata_projected_bytes':cost(KEYS),'financial_projected_bytes':cost(COLUMNS),'prior_key_projected_bytes':prior_cost,
  'total_projected_bytes':cost(KEYS)+cost(COLUMNS)+prior_cost,
  'returned_financial_rows_upper_bound':len([d for d in sessions if START<=d<=END]),
  'calendar_pin':pin(calendar),'footer_plan_pin':pin(footer_plan),'prior_keys_pin':pin(prior),
  'prior_key_receipt_pin':pin(prior.with_name('receipt.json')),
  'OHLC_common_scale_relative_tolerance':TOLERANCE,
  'source_factor_comparisons':'Reportfieldfactor versusobserved adjclose/rawclose AND reciprocal; no semanticreinterpretation orautomaticcertification.',
  'volume_policy':'Finite nonnegative rawandadjusted volumes required; noforcedinverseofdividend-inclusivepricefactor.',
  'action_null_policy':'Preserveunknown null; nosyntheticzero andno doublecountdividend.',
  'discontinuity_policy':'ConsecutiveexpectedXNYS sessionsonly; flag absraw/adjclosechange>.5 orderivedscalechange>.05; nocompressionovermissinggaps, nofills/exclusions.',
  'mixed_block_limit':'Whole33-columnchunks for255799sourceblockrowsmaytransientlydecode othernatives/premerger/non-session/2024+values inArrow. Native/date/postmerger/XNYS filterbeforePython meansnoneoftheseexcludedfinancialvaluesreturned/analyzed.',
  'identity_policy':'Unpromoted candidate; premergerkeysand11provider-onlydatesmetadataonly; retrievalmetadataneverpastadmission.',
  'models_rebuilds_backtests_authorized':False}

if __name__=='__main__':
 plan=build();p=S/'bk_financial_read_plan.json';assert not p.exists();p.write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
 cfg={'schema':'dmv2-BK-field-config/v1','plan_pin':pin(p),'code_pins':[pin(S/n) for n in ('bk_field_audit.py','read_bk_candidate.py')]}
 c=S/'bk_financial_config.json';assert not c.exists();c.write_text(json.dumps(cfg,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'plan_pin':pin(p),'config_pin':pin(c),'code_pins':cfg['code_pins'],'projected_bytes':plan['total_projected_bytes'],
  'returned_rows_upper_bound':plan['returned_financial_rows_upper_bound']}))
