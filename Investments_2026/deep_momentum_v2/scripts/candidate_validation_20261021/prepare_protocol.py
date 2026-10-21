from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R=Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
S=Path('/private/tmp/dmv2-candidate-validation-20261021')
pins=[]
for rel in ['AGENTS.md','docs/PROVISIONAL_IDENTITY_POLICY.md','docs/DATA_FOUNDATIONS.md','docs/IDENTITY_AUDIT.md','docs/ROUTE_COVERAGE_XOM_BK_20261021.md','src/dmv2/loading.py','outputs/route_coverage/20261021-xom-bk-v1/preservation.json','outputs/route_coverage/20261021-xom-bk-v1/handoff.json']:
    p=R/rel
    pins.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
protocol={
 'schema':'dmv2-candidate-field-validation-protocol/v1',
 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'authority':'User approved validating BK candidate actual prices/dated identity, followed by XOM alternative history, before rebuilding affected learning inputs.',
 'status':'FROZEN_SCOPE_REQUIRES_EXACT_SUBPLANS_BEFORE_FINANCIAL_READS',
 'root':str(R),
 'original_pins':pins,
 'scope':[
  {'ticker':'BK','native':'sec_zq1J05','start':'2007-01-01','end':'2023-12-31','role':'unpromoted ordinary-share candidate validation'},
  {'ticker':'XOM','source':'pinned SIP alternative','start':'2016-01-04','end':'2023-12-31','role':'bounded alternative validation; no synthetic earlier history'},
 ],
 'financial_read_requirements':['Exact object/generation/hash or authenticated range pin, explicit schema/projection and date/ID bounds, compressed-byte/row/file reservations before conversion.','Metadata/date/ID masks in Arrow before any financial Python conversion; excluded rows never analyzed or returned. Disclose mixed Parquet block transient decoding; do not claim physical per-cell isolation.','No financial footer min/max statistics used for pruning. Independent XNYS sessions; retain extra keys only in audit/raw views.','Preserve all original cohort denominators and source identity/review flags. No future availability admission screen.'],
 'budget':{'max_BK_projected_bytes':134217728,'max_XOM_projected_bytes':134217728,'max_combined_projected_bytes':268435456,'max_financial_returned_rows_per_case':10000,'max_rss_bytes':2147483648,'max_worker_wall_seconds':600,'threads':1,'max_output_bytes':67108864,'automatic_retries':0},
 'required_checks':['Exact native/date uniqueness and requested coverage; no deduplication or gap fills.','All available raw/adjusted OHLCV, action and quality/provenance fields. Missing action fields explicitly unavailable.','Finite positive price checks, OHLC bracket consistency, nonnegative volume, raw/adjusted OHLC common factor, split/dividend semantics and flagged discontinuities.','Dated ordinary-share identity and source/native evidence separately from platform aliases.','Versioned conditional repair proposal only if supported; no production promotion.'],
 'forbidden':['2024+ financial outputs or analysis','Source/cache/run/pointer overwrites','Cohort removal based on future data or current survival','New features/targets/class/RET panels','Model fitting, inference or backtesting','Changing monthly label objective, basket/stop rules or normalization','Cloud mutations/IAM or messaging other people'],
 'next_after_validation':'Save exact validated intervals, unresolved gaps and rebuild-impact requirements; dependent shadow rebuild/compute gets a separate frozen scope.'
}
with (S/'protocol.json').open('x') as f: json.dump(protocol,f,indent=2,sort_keys=True); f.write('\n')
print(hashlib.sha256((S/'protocol.json').read_bytes()).hexdigest())
