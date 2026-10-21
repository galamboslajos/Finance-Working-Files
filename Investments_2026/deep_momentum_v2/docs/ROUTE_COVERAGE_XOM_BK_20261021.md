# XOM and BK route/source coverage audit — 21 October 2026

Status: **metadata audit complete; no mapping promoted and no production input changed**. This continues the [learning-input audit](LEARNING_INPUT_IMPACT_20261021.md) before alternative label experiments. Original runs, labels, features, normalization, models, predictions, portfolio rules and notebook behavior remain unchanged.

## What was established

The two original native IDs were explicitly requested in all fourteen corrected annual partitions and all four saved ledgers: feature, Paper15, technical and forward-monthly requests. All 168 signal requests are preserved. An omitted native-ID request does not explain their zero coverage.

The existing code maps exact source → unified → native IDs and filters exact native/date keys. It has no silent ticker fallback or current-master survival filter. The key-only scan authenticated the release manifest and all 105 safe native/date footer ranges, then read three selected shards. It inspected **no OHLCV, return, feature, class or probability values**.

| Route | Observed 2007–2023 date keys | Exact original monthly anchors | Interpretation |
|---|---:|---:|---|
| XOM current `sec_z5YpGE` | 0 | 0 / 169 | Published capture belongs only to a reported July 2026 history |
| BK current `sec_gA3NYQ` | 0 | 0 / 169 | Provider labels this Series I preferred, with zero captured prices |
| BK candidate `sec_zq1J05` | 4,289 | 169 / 169 | Unpromoted ordinary-share candidate with full required date-key coverage |

These are absences/presences in the authenticated selected-native source release and specified interval. They do not prove global market-history absence or valid financial observations. Current and future inventory dates are retrieval-time diagnostics; they never change historical admission in this audit.

## XOM: the selected route cannot supply the study history

The original source ID is `siblis_isin_b284bb7497616f5afba0`, source literal ISIN `US30231G1022`, unified ID `equity_us_bf463bc8d3ea387578b7bafc`.

The published link stores query type `isin` but literal query **`XOM`**, rather than the source ISIN. This is a metadata type/value inconsistency; original network lookup parameters and response bytes were not independently verified. The alias ISIN is source/platform evidence, not a provider-returned ISIN.

The selected provider master describes ordinary shares with first/last stock-price dates **2026-07-06–2026-07-24**; its acquisition record reports **15 rows**. The actual local key scan finds zero keys in the 2007–2023 requested interval. This explains the historical coverage failure of this selected route. It does not establish that Exxon lacked earlier market data or establish a replacement ID.

Older retained Intrinio master/link/acquisition references and the alternative unified identity bridge were compared by exact URI/generation/hash/size. They are the same metadata objects already inspected locally, so rereading them would add no older native mapping.

A separate published SIP source supplies an alternative **capture candidate**. Two small metadata files were read through Google Cloud CLI as `info@galamboscapital.com`, project `trusty-agility-492109-v0`, configuration `deep-momentum-research`:

- `gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/symbol_normalization_evidence/part-00000.parquet`, generation `1783344910405429`.
- `gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/missing_symbol_evidence/part-00000.parquet`, generation `1783344910553470`.

Downloaded sizes and SHA-256 values match the pinned manifests, totaling **372,564 bytes**. The literal XOM/BK normalization and request records were selected in Arrow before Python conversion. Despite the second table's name, inspected request records are **complete with positive reported row counts**. For 2016–2023, 32 unique records cover two symbols × eight years × raw/all adjustment modes; reported raw/all counts agree and range from 250–253 rows/year.

This verifies published query/capture metadata, not exact dated bars, valid OHLCV, share-class continuity or comparable adjustment units. The curated alternative declares coverage beginning **2016-01-04**, leaving **2010–2015 and earlier warmup unresolved**. The research start remains 2010. No daily-bar object or raw-price JSON was downloaded. Python read local metadata files; Python GCS ADC was not tested.

Two failed CLI refresh attempts are retained. After the user completed forced browser authentication, the same two generation-pinned reads succeeded. No storage-permission error or owner IAM action was established. No credentials or tokens are stored in the audit output.

## BK: a concrete ordinary-share candidate is already local

The original source ID is `siblis_ticker_company_1c21a7b293fd137c59c9`. Its membership source has a blank ISIN and a Bank of New York Mellon company/ticker identity. The current native lookup resolves literal `BK` to code `PRF`, type **FXDFR preferred perpetual, Series I**. It reports zero price rows. This is a concrete share-class warning for the intended equity-index research route.

Another published identity record resolves literal ISIN `US0640581007` to native **`sec_zq1J05`**, code `EQS`, ordinary shares, provider ticker `BNY`, with reported acquisition history from 1973 and 13,430 whole-history rows. It remains a candidate: source aliases, current ticker/name, inventory counts and a shared capture cannot independently establish its dated relationship to every original BK membership row.

Its actual local date keys cover **all 4,278 pinned XNYS sessions from 2007–2023**, with zero missing sessions, and all 169 original month-end signal/outcome anchors. There are also **11 non-XNYS date keys**:

`2007-11-22`, `2009-01-19`, `2009-07-03`, `2010-01-18`, `2011-04-22`, `2012-10-30`, `2013-01-21`, `2013-02-18`, `2013-03-29`, `2013-07-04`, `2014-09-01`.

The independent session filters must remain in place. Static review verifies that the full/walk-forward adapters remove non-session records before feature calculation and retain them in raw/audit views. Daily-indicator validation rejects any surviving non-session record before financial conversion, then uses exact session windows. Generic `PriceLoader` and raw files alone do not apply that calendar filter.

Complete date keys do not establish non-null/positive prices, volume coverage, split/dividend semantics, adjustment continuity, liquidity or executable returns. **No BK financial values were inspected or attached to the original source cohort in this audit.**

## Next work before label experiments

1. Freeze a small, field-complete **BK candidate validation**: observed raw/adjusted OHLC, volume and action/quality fields; original source/native provenance; independent session filtering; and dated ordinary-share identity evidence. Record any accepted provisional interval in a separate reviewed conditional amendment. A ticker or a good coverage count alone is insufficient for promotion.
2. Validate the alternative **XOM SIP history from 2016 onward** under an explicit exact-key/OHLCV/adjustment read plan, and locate the pre-2016 ordinary-share route/source. Preserve unresolved history explicitly while that work proceeds. Source product start dates and successful request counts do not substitute for observed historical bars.
3. After source amendments are reviewed, rebuild a **shadow** feature/target/class/120-month RET version with the existing monthly-return objective and original membership denominators. Recovering a constituent can change cross-sectional normalization and other stocks' class boundaries throughout history, so changes are not limited to that stock's row.
4. Compare actual eligibility/value/schema and inner/outer fold hashes. Reuse unchanged artifacts, rescore validation-only changes where possible and refit affected training dependencies only under a later frozen compute scope. No model fitting or automatic repeat of the 100-fit search is authorized by this audit.

The user-accepted provisional identity policy remains active; this work does not require full identity certification before unrelated development. It does distinguish a demonstrated preferred-share route warning from unresolved general review flags. Nothing here certifies knowledge-time identity or a fully survivorship-bias-free universe.

## Reproduction and evidence

Public audit code is in `scripts/route_coverage_20261021/`; their executed paths and immutable content pins are retained in the private evidence. Replays need a fresh output directory and a newly sealed path/read plan; they are not notebook Run All stages.

Private output is `outputs/route_coverage/20261021-xom-bk-v1/`:

- `handoff.json`: machine-readable findings and next tasks.
- `route_trace/request_and_code_trace.json` and `review/request_inclusion.json`: exact request/source-chain proof.
- `source_inventory/result-v1/cases.json`: source bridge, lookup, provider and acquisition evidence.
- `key_inventory/result-v1/date_keys.parquet`, `cases.json`, `receipt.json`: exact bounded date keys, endpoint flags and read reservations.
- `key_inventory/independent_result_review.json`, `review/date_key_v2_result_review.json`, `review/calendar_only/`: independent counts, calendars and limitations.
- `key_inventory/non_session_feature_boundary_static_review_v2.json`: exact code/session boundaries.
- `alternate_source_plan/`: pinned duplicate-source comparisons and alternative-source locators.
- `sip_metadata_download-v1/`, `-v2/`, `-v3/`: preserved failed refreshes and successful pinned downloads.
- `sip_metadata_analysis-v1/` and `review/sip_result_review_ret_inputs.json`: filtered query/capture evidence and independent review.
- `preservation.json`: unchanged inherited production pins.

**34 focused synthetic tests passed** across source candidate handling, key pruning/budgets, empty-route/request distinction and triage. Independent readers reproduced the saved metadata/key/anchor results and calendar comparison. The selected stages projected **8,313,687 compressed metadata-column bytes** and converted **89,382 metadata rows**, across eleven distinct input objects; no financial values were read. Footer counts are distinct-file budgets: the key stage also discloses its repeated pre/post footer inspections rather than claiming only one parse per file. Ninety-five inherited production pins (46,671,297 bytes) were rechecked unchanged. This preservation scope does not rehash every model in every old run.
