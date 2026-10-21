from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

S=Path('/private/tmp/dmv2-candidate-validation-20261021')
R=Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
rel='outputs/learning_input_impact/20261021-v1/metadata/result-v2/source_native_year_diagnostics.jsonl'
p=R/rel
failure={'status':'STOPPED_BEFORE_CONTENT_READ','reason':'Proposed8MiB saved-metadata ceiling smaller than stat size. No JSON content parsed.','source_size_bytes':p.stat().st_size,'previous_ceiling_bytes':8388608}
with (S/'collision_metadata_initial_preflight.json').open('x') as f: json.dump(failure,f,indent=2)
assert p.stat().st_size<32*1024*1024
with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
plan={'schema':'dmv2-candidate-collision-saved-metadata-read/v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'file':{'path':rel,'size_bytes':p.stat().st_size,'sha256':digest},'scope':'Saved source-native-year diagnostic metadata only; no original Parquet or financial values. Candidate absence here is a diagnostic, not a replacement for rebuilding same-date collisions before a future promotion.','max_input_bytes':33554432,'max_records':30000,'candidate_native':'sec_zq1J05','financial_values_read':False}
with (S/'collision_metadata_plan_v2.json').open('x') as f: json.dump(plan,f,indent=2)
matches=[]
count=0
fields=set()
with p.open() as f:
    for line in f:
        count+=1
        assert count<=plan['max_records']
        row=json.loads(line)
        fields.update(row)
        if 'sec_zq1J05' in json.dumps(row): matches.append(row)
assert len(matches)<1000
out={'schema':'dmv2-candidate-collision-saved-metadata-result/v1','status':'PASS_SAVED_METADATA_DIAGNOSTIC','plan_sha256':hashlib.sha256((S/'collision_metadata_plan_v2.json').read_bytes()).hexdigest(),'records':count,'fields':sorted(fields),'candidate_matches':matches,'financial_values_read':False,'claim_limit':'No existing candidate-native occurrence in this saved all-original source/native/year metadata is evidence against a collision, not a new canonical same-date admission audit.'}
with (S/'collision_metadata_result.json').open('x') as f: json.dump(out,f,indent=2)
print(json.dumps({'records':count,'fields':sorted(fields),'candidate_matches':len(matches)}))
