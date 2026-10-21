from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

S=Path('/private/tmp/dmv2-candidate-validation-20261021')
def pin(p):
    return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write_new(p,obj):
    with p.open('x') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')

xom=json.loads((S/'xom/financial-result-v1/field_summary.json').read_text())
xr=json.loads((S/'xom/financial-result-v1/receipt.json').read_text())
precision=json.loads((S/'xom_precision-v1/summary.json').read_text())
assert xom['complete_raw_all_pairs']==2012 and xom['dates_with_quality_flags']==69
assert precision['original_2bp_flags_preserved']==69
xom_proposal={
 'schema':'dmv2-XOM-conditional-shadow-source-proposal/v1',
 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'PARTIAL_HISTORY_CANDIDATE_WITH_PRESERVED_PRECISION_WARNINGS_NOT_INSTALLED',
 'production_promoted':False,
 'source_security_id':'siblis_isin_b284bb7497616f5afba0',
 'original_unified_id':'equity_us_bf463bc8d3ea387578b7bafc',
 'original_intrinio_native_retained_as_evidence':'sec_z5YpGE',
 'proposed_price_route':{'provider':'Alpaca/SIP','source_key_type':'security_id','source_key':'siblis_isin_b284bb7497616f5afba0','product':'sp_composite1500_market_daily/v1','object':xr['object'],'adjustment_for_proposed_learning_proxy':'all','raw_retained_for_QA':True},
 'SIP_rows_must_not_be_relabelled_as_Independently_verified_Intrinio_native_history':True,
 'observed_window_inclusive':['2016-01-04','2023-12-29'],
 'observed_XNYS_sessions':2012,'observed_raw_all_rows':4024,
 'missing_original_2010_2015_history_retained':True,
 'earlier_warmup_not_filled':True,
 'base_OHLC_volume_and_date_checks_passed':True,
 'frozen_common_scale_2bp_warning_dates':69,
 'frozen_common_scale_2bp_warning_flags_cleared':False,
 'maximum_observed_common_scale_residual_bp':xom['max_OHLC_common_scale_relative_residual']*10000,
 'post_hoc_cent_envelope_compatible_warning_dates':precision['flagged_sessions_adjusted_only_half_cent_compatible'],
 'post_hoc_interpretation':'Observed all-adjusted OHLC lie on a cent grid and a common price factor is possible within plus/minus half a cent. This is compatibility under an assumption, not a verified provider rounding rule or cause.',
 'source_prices_changed_or_filled':False,
 'dividends_splits_and_provider_quality_fields_unavailable':True,
 'knowledge_time_identity_or_economic_total_return_certified':False,
 'prerequisites_before_shadow_consumption':[
  'A new frozen shadow consumer recipe must explicitly retain the 69 warnings and state its precision handling; do not overwrite the original 2-bp audit result.',
  'Keep the typed SIP provider/source key and price units separate from the original Intrinio native lineage; no silent provider splice.',
  'Build historical membership before price joins, preserve all original denominators/flags and recompute same-date route conflicts.',
  'Use only exact independent XNYS dates in the observed interval, preserve missing pre-2016 history and exact feature warmup; no stale/synthetic fills.',
  'Use published all-adjusted OHLCV only as the existing provisional proxy; do not add dividends or apply an inferred factor again.',
  'Rebuild affected cross-sectional features, observed-outcome classes and RET coherently in new outputs; measure actual dependency changes before new inference or fitting.'
 ],
 'evidence_pins':[pin(S/p) for p in ['xom/financial-result-v1/receipt.json','xom/financial_result_review.json','identity/xom_output_review-v1/result.json','xom_precision_plan.json','xom_precision-v1/summary.json','xom_identity/xom_identity_note_v2.json','xom_identity/provider_precision_research.json']],
 'no_features_labels_RET_models_or_backtests_built':True,
}
write_new(S/'XOM_conditional_shadow_proposal.json',xom_proposal)

handoff={
 'schema':'dmv2-BK-XOM-field-validation-handoff/v1',
 'status':'VALIDATION_COMPLETE_CONDITIONAL_PROPOSALS_NOT_INSTALLED',
 'BK':{'required_window':['2008-01-04','2023-12-29'],'required_sessions':4025,'required_missing_sessions':0,'required_base_field_failures':0,'postmerger_diagnostic_sessions':4154,'pre_required_window_OHLC_warning_dates':['2007-07-02','2007-07-10','2007-08-17','2007-10-19'],'candidate_native':'sec_zq1J05','identity_status':'PROVISIONAL_POSTMERGER_ISSUER_EVIDENCE_NOT_FULL_CERTIFICATION'},
 'XOM':{'observed_window':['2016-01-04','2023-12-29'],'sessions':2012,'raw_all_rows':4024,'missing_sessions_or_pairs':0,'base_OHLC_volume_failures':0,'frozen_2bp_scale_warning_dates':69,'maximum_scale_residual_bp':xom_proposal['maximum_observed_common_scale_residual_bp'],'post_hoc_cent_envelope_compatible_warnings':69,'provider_precision_policy_verified':False,'pre_2016_history_resolved':False,'typed_provider_source_key_required':True},
 'source_read_budget':{'BK_projected_bytes':12075945,'XOM_cumulative_range_and_footer_payload_bytes':115639849,'combined_conservative_bytes':127715794,'maximum_worker_peak_RSS_bytes':max(xr['peak_rss_bytes'],json.loads((S/'core/result-v1/summary.json').read_text())['peak_rss_bytes'])},
 'root_distinct_implementation_synthetic_tests_passed':61,
 'independent_output_reviews_completed':True,
 'original_production_files_changed':0,
 'production_mapping_or_source_promotion':False,
 'new_feature_target_RET_builds_or_models_or_backtests':0,
 'next_step':'Freeze a separate coherent shadow source/feature/target/RET build that preserves original cohorts and warnings, then compare masks, values and fold hashes before selecting any necessary rescore/refit scope.',
 'remaining_limits':['XOM2010–2015 and earlier warmup remain absent.','Historical identifier knowledge time, provider raw identity responses, source adjustment vintages and economic-return completeness remain unverified.','No new strategy-performance claim or alternative-label experiment is made.'],
}
write_new(S/'handoff.json',handoff)

draft=(S/'CANDIDATE_VALIDATION_20261021.draft.md').read_text()
draft=draft.replace('Working draft. The completed report will distinguish actual field checks from source inventory, identity inference and proposed consumption. No canonical route, production feature/label panel, model, forecast or backtest is changed by this audit.',
 'Status: **field validation complete; separate conditional proposals saved, with no production promotion**. BK has a clean required-history interval. XOM supplies partial history with retained precision warnings. Original source files, cohorts, labels, features, models, forecasts and backtests are preserved. This continues the [route/source audit](ROUTE_COVERAGE_XOM_BK_20261021.md) and [learning-input audit](LEARNING_INPUT_IMPACT_20261021.md).')
start=draft.index('## XOM: validation in progress')
end=draft.index('## Follow-on dependency work',start)
replacement='''## XOM: complete observed 2016–2023 coverage, retained precision warnings

The SIP alternative was read by exact original source ID `siblis_isin_b284bb7497616f5afba0`, literal query `XOM`, independent XNYS date and raw/all adjustment mode. The pinned manifest identifies the underlying retained provider as Alpaca. Primary issuer/SEC filings support the ordinary-share CUSIP/ISIN identity around the interval; this remains a reconstruction rather than a daily knowledge-time/provider-native certificate.

The actual output contains **2,012 expected sessions, with 2,012 raw and 2,012 all-adjusted rows**, from **4 January 2016 through 29 December 2023**. Every expected pair is present, with no extra dates. All individual OHLC and volume checks pass; raw/all volumes agree. No frozen large-move or factor-discontinuity diagnostic fires. The source does not supply explicit dividends, splits, provider adjustment factors or provider quality fields: those are unavailable, not zero.

**69 dates exceed the frozen two-basis-point common-OHLC scale tolerance**, maximum **3.524929 bp**. The independent output checker reproduces every warning and all coverage/field diagnostics. A completed read or a reconciliation PASS is not a clean-source PASS.

A separately sealed, **post-hoc precision diagnostic** tested all 2,012 pairs. Every observed all-adjusted OHLC value lies on a cent grid, while only 1,478 raw OHLC rows have all four prices on that grid. For every date, including all 69 warnings, there exists one positive common factor if each adjusted price is allowed to represent an unobserved price within **±0.005 price units**, even treating raw prices as fixed. The calculation intersects the four admissible factor intervals; it does not round, repair or replace any observation. Closed bounds and a fixed `1e-12` comparison slack are disclosed.

This is **compatibility with an assumed precision envelope**, not proof that Alpaca rounded those bars or that rounding caused the differences. The official API documents raw/all/SIP semantics but supplies no rounding guarantee found in this review. Its numeric examples include sub-cent prices; do not infer a universal cents-only contract. A user forum rounding conjecture is not provider-policy evidence. **The original 69 warnings, original two-bp criterion and original prices remain unchanged.** [Official historical-bars reference](https://docs.alpaca.markets/us/reference/stockbars).

The proposed source stays a **partial-history candidate with precision warnings**. It must carry its SIP provider/source key and units explicitly; do not relabel these prices as independently verified Intrinio-native history or silently splice price levels across providers. Any future shadow recipe must state its handling of the retained warnings before consumption. No source amendment is installed by this report.

The original Intrinio route still has no study-period keys. **2010–2015 and earlier XOM warmup remain unresolved**. The research start stays 2010, original membership denominators stay intact, and no synthetic/stale history fills that gap. Exact feature qualification must be recomputed in the shadow build; observed daily coverage alone does not establish a usable predictor or target at every date.

'''
draft=draft[:start]+replacement+draft[end:]
draft+='''
## Read scope, checks and saved artifacts

BK read one pinned local shard, projecting 12,075,945 compressed bytes including the explicit metadata and prior-key passes. XOM used the exact-generation SP1500 `daily_bars/part-00000.parquet` object: generation **1783344954247444**, source object size **187,908,505 bytes**, manifest SHA-256 `4a4919d48128d5c573ddbca64649d8cc78783f7bbb818219a3ad5f34728e38a0`. Only metadata/footer and explicit column ranges were retrieved: **115,639,849 payload bytes cumulatively**, below the frozen 128 MiB XOM cap. The full object was not downloaded or rehashed. Exact generation, range bounds and each payload digest were checked; these are range-integrity receipts, not a full downloaded-object digest.

The financial pass retained only 4,024 XOM rows. Its peak process RSS was **972,963,840 bytes (0.91 GiB)** and it completed in about 66 seconds. No original compressed source chunks are retained. Private quote outputs remain under ignored `outputs/`; no raw market rows or credentials are published.

Native/source/date and independent-session masks precede financial Python conversion. Selected Parquet column blocks can transiently decode excluded dates or other instruments inside Arrow; this is an output/analysis guard, **not physical per-cell isolation**. No excluded 2024+ financial value was returned, converted to Python or analyzed.

**61 focused implementation tests passed** across BK QA, independent BK checks, footer/range/key guards, XOM financial masking, paired QA and the separate precision diagnostic. Independent readers reconciled both actual bounded outputs. One intermediate synthetic fixture still expected non-session rows after the stricter guard correctly removed them; its expectations and exclusion assertions were corrected, and all final integration tests passed before the real financial read. This is not a claim of a full repository test run or certified economic data. Ninety-five inherited source, request, calendar and policy/code pins were checked unchanged; this does not rehash every model in every old run.

Public audit snapshots are in `scripts/candidate_validation_20261021/`. They retain their executed code hashes and original path plans; replay requires fresh outputs and a newly sealed plan, not notebook Run All. The public proposed configs are `configs/bk_conditional_shadow_20261021.proposed.json` and `configs/xom_conditional_shadow_20261021.proposed.json`; neither is wired into current production readers.

Private evidence is in `outputs/candidate_validation/20261021-bk-xom-v1/`: `handoff.json`, `protocol.json`, `core/result-v1/`, `bk_output_review-v1/`, `identity/`, `xom/footer-result-v1/`, `xom/key-result-v1/`, `xom/financial-result-v1/`, `xom_precision-v1/`, source identity/precision notes, plans, launch seals, test and preservation receipts. The separately proposed route/source amendments retain all original flags and known warnings. No model fitting, inference, backtesting, panel rebuild, cloud write or external message occurred.
'''
p=S/'FINAL_REPORT.md'
with p.open('x') as f:f.write(draft)
print(json.dumps({'report':pin(p),'handoff':pin(S/'handoff.json'),'XOM_proposal':pin(S/'XOM_conditional_shadow_proposal.json')}))
