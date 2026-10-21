from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

S=Path('/private/tmp/dmv2-candidate-validation-20261021')
def pin(p):
    return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

calendar=json.loads((S/'core/calendar_metadata.json').read_text())['sessions']
qa=json.loads((S/'bk_output_review-v1/independent_QA.json').read_text())['stats']
required=[d for d in calendar if '2008-01-04'<=d<='2023-12-29']
assert len(required)==4025
assert not any(d>='2008-01-04' for d in qa['raw_OHLC_failures']+qa['adjusted_OHLC_failures'])
assert not qa['missing_sessions']
for category in ['price_null_or_nonfinite','price_nonpositive','volume_null_or_nonfinite','volume_negative','volume_zero']:
    assert not any(qa[category].values())
proposal={
 'schema':'dmv2-BK-conditional-shadow-route-proposal/v1',
 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'CONDITIONAL_REQUIRED_WINDOW_CANDIDATE_NOT_INSTALLED',
 'production_promoted':False,
 'source_security_id':'siblis_ticker_company_1c21a7b293fd137c59c9',
 'original_unified_id':'equity_us_a2ca9feacdef53087ce87b15',
 'original_native_id':'sec_gA3NYQ',
 'candidate_native_id':'sec_zq1J05',
 'original_route_retained_as_evidence':True,
 'proposed_postmerger_instrument_interval_half_open':['2007-07-02','2023-12-30'],
 'observed_issuer_ISIN_document_date':'2013-03-12',
 'interval_is_provisional_inference_not_certified_ISIN_history':True,
 'required_financial_window_inclusive':['2008-01-04','2023-12-29'],
 'required_XNYS_sessions':len(required),
 'first_forecast_signal':'2009-12-31',
 'first_outcome_month':'2010-01',
 'warmup_sessions':503,
 'known_bad_OHLC_dates_preserved':qa['raw_OHLC_failures'],
 'known_bad_dates_outside_predetermined_required_window':True,
 'required_window_all_raw_adjusted_OHLCV_finite_positive_and_OHLC_brackets_valid':True,
 'adjustment_semantics':'Use published adjusted OHLC/volume as the existing provisional proxy; no multiplication by source factor and no dividend addition. Exact economic/knowledge-time adjustment history remains uncertified.',
 'certification_flags':{'raw_provider_response_verified':False,'independent_provider_returned_ISIN_verified':False,'temporal_identity_verified':False,'knowledge_time_verified':False,'survivorship_bias_free_certified':False},
 'prerequisites_before_any_shadow_consumption':[
  'Bind reviewed source, code, calendar, identity and field-output hashes in a new frozen consumer recipe.',
  'Rebuild each original dated membership cohort before price joins; preserve every original source row, denominator and producer review flag.',
  'Recompute exact same-date native collisions against all original members and applicable context populations before admitting the amended route.',
  'Use independent XNYS sessions and exact existing warmup. Retain all provider-only and known bad dates in audit metadata; never substitute stale prices.',
  'Rebuild affected cross-sectional feature statistics and observed-outcome classes/RET inputs consistently in shadow outputs; do not splice a replacement row into old model matrices.',
  'Measure masks/value/schema/fold changes before authorizing inference or fits. Preserve every old run and objective/basket/stop rule.'
 ],
 'evidence_pins':[pin(S/p) for p in ['protocol.json','core/result-v1/receipt.json','core/result-v1/summary.json','bk_output_review-v1/independent_QA.json','identity/primary_evidence.json','identity/provisional_identity_recommendation.json','collision_metadata_result.json']],
 'no_features_labels_RET_models_or_backtests_built':True,
}
p=S/'BK_conditional_shadow_proposal.json'
with p.open('x') as f: json.dump(proposal,f,indent=2,sort_keys=True); f.write('\n')
print(json.dumps({'status':proposal['status'],'required_sessions':len(required),'proposal_sha256':pin(p)['sha256']}))
