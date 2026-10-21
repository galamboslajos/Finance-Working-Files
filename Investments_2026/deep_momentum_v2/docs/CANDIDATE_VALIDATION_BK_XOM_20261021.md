# BK and XOM candidate validation — 21 October 2026

Status: **field validation complete; separate conditional proposals saved, with no production promotion**. BK has a clean required-history interval. XOM supplies partial history with retained precision warnings. Original source files, cohorts, labels, features, models, forecasts and backtests are preserved. This continues the [route/source audit](ROUTE_COVERAGE_XOM_BK_20261021.md) and [learning-input audit](LEARNING_INPUT_IMPACT_20261021.md).

## BK: usable required history, conditional route

The source's original preferred-share route remains preserved. The ordinary-share candidate is Intrinio `sec_zq1J05`, reached through the published literal ISIN query `US0640581007`. Primary dated issuer/SEC evidence supports the post-July-2007 ordinary-share identity. The ISIN itself is explicitly observed in a March 2013 issuer prospectus; extending that identity across the study remains a provisional reconstruction, not a certified dated provider crosswalk. [July 2007 SEC merger filing](https://www.sec.gov/Archives/edgar/data/1390777/000119312507147812/d8k12g3.htm), [March 2013 issuer prospectus, printed page 19](https://www.fsma.be/sites/default/files/prospectus/files/2013/2013-%25282686%2529-EN-SCS20130419-A01-B01-C01-NP-CD12_03.pdf).

The original configured feature warmup is 503 sessions. For the first forecast signal, **31 December 2009**, predicting January 2010, that window begins **4 January 2008**. This boundary was established from the independent calendar before the financial read. Pre-merger history is unnecessary for this configured scope.

Observed field audit:

| Scope | XNYS sessions | Missing sessions | Raw/adjusted OHLC failures | Volume failures |
|---|---:|---:|---:|---:|
| Post-merger diagnostic: 2 July 2007–29 December 2023 | 4,154 | 0 | 4 dates | 0 |
| Predetermined required history: 4 January 2008–29 December 2023 | 4,025 | 0 | 0 | 0 |

All eight raw/adjusted price fields are finite and positive; both volume fields are present, finite and positive. The four genuine OHLC failures are **2 July, 10 July, 17 August and 19 October 2007**. Source quality flags also identify those four. They remain in the evidence; the original stock and membership rows are not dropped. No price is fabricated or repaired to pass this audit. Eleven non-XNYS keys remain date-only diagnostics, excluded before financial Python conversion. Pre-merger price values likewise remain outside selected/converted output.

The fixed paired-OHLC scale tolerance is two basis points. Maximum observed relative residual is **0.071889 bp**, within that preread criterion. A separate, stricter independent diagnostic (`rtol=1e-7`, `atol=1e-8`) reports 4,152 small ratio differences. These thresholds answer different questions; do not relabel those diagnostics as 4,152 corrupt bars or assert rounding as a proven explanation. Raw source fields remain unchanged.

The source `factor` is not assumed equal to that row's cumulative adjusted/raw price ratio. Intrinio describes it as an adjustment applied to prices **before** its date. The audit records pointwise comparisons only; their residuals do not establish incorrect published adjusted prices. Use already published adjusted OHLCV under the existing provisional return convention, without applying the factor again or adding dividends again. This does not certify total-return economics, historical adjustment vintages or volume-adjustment semantics. [Intrinio output-field documentation](https://docs.intrinio.com/documentation/web_api/get_security_stock_prices_v2).

The separate conditional proposal requires pinned consumer inputs and a new same-date collision calculation against all original membership/context rows before any shadow use. Saved original route diagnostics contain no occurrence of this candidate native, but that is not a substitute for the new collision check. Original unified IDs, source rows, flags and denominators remain visible. No mapping is promoted by this document.

## XOM: complete observed 2016–2023 coverage, retained precision warnings

The SIP alternative was read by exact original source ID `siblis_isin_b284bb7497616f5afba0`, literal query `XOM`, independent XNYS date and raw/all adjustment mode. The pinned manifest identifies the underlying retained provider as Alpaca. Primary issuer/SEC filings support the ordinary-share CUSIP/ISIN identity around the interval; this remains a reconstruction rather than a daily knowledge-time/provider-native certificate. [2016 SEC identity record](https://www.sec.gov/Archives/edgar/data/216851/000021685116000070/peonpx06302016.htm), [2023 issuer cover filing](https://www.sec.gov/Archives/edgar/data/34088/000003408823000056/xom-20230930.htm).

The actual output contains **2,012 expected sessions, with 2,012 raw and 2,012 all-adjusted rows**, from **4 January 2016 through 29 December 2023**. Every expected pair is present, with no extra dates. All individual OHLC and volume checks pass; raw/all volumes agree. No frozen large-move or factor-discontinuity diagnostic fires. The source does not supply explicit dividends, splits, provider adjustment factors or provider quality fields: those are unavailable, not zero.

**69 dates exceed the frozen two-basis-point common-OHLC scale tolerance**, maximum **3.524929 bp**. The independent output checker reproduces every warning and all coverage/field diagnostics. A completed read or a reconciliation PASS is not a clean-source PASS.

A separately sealed, **post-hoc precision diagnostic** tested all 2,012 pairs. Every observed all-adjusted OHLC value lies on a cent grid, while only 1,478 raw OHLC rows have all four prices on that grid. For every date, including all 69 warnings, there exists one positive common factor if each adjusted price is allowed to represent an unobserved price within **±0.005 price units**, even treating raw prices as fixed. The calculation intersects the four admissible factor intervals; it does not round, repair or replace any observation. Closed bounds and a fixed `1e-12` comparison slack are disclosed.

This is **compatibility with an assumed precision envelope**, not proof that Alpaca rounded those bars or that rounding caused the differences. The official API documents raw/all/SIP semantics but supplies no rounding guarantee found in this review. Its numeric examples include sub-cent prices; do not infer a universal cents-only contract. A user forum rounding conjecture is not provider-policy evidence. **The original 69 warnings, original two-bp criterion and original prices remain unchanged.** [Official historical-bars reference](https://docs.alpaca.markets/us/reference/stockbars).

The proposed source stays a **partial-history candidate with precision warnings**. It must carry its SIP provider/source key and units explicitly; do not relabel these prices as independently verified Intrinio-native history or silently splice price levels across providers. Any future shadow recipe must state its handling of the retained warnings before consumption. No source amendment is installed by this report.

The original Intrinio route still has no study-period keys. **2010–2015 and earlier XOM warmup remain unresolved**. The research start stays 2010, original membership denominators stay intact, and no synthetic/stale history fills that gap. Exact feature qualification must be recomputed in the shadow build; observed daily coverage alone does not establish a usable predictor or target at every date.

## Follow-on dependency work

After field and conditional-route review, prepare a separate shadow build using the existing monthly-return objective, original membership denominators, 80% primary context rule and exact causal windows. Recompute cross-sectional feature statistics, classes and 120-month RET inputs consistently; adding one constituent can change other stocks' values and class boundaries. Keep daily feature history and daily execution data separate.

Measure the actual row masks, values, schema and inner/outer fold hashes before deciding what can be reused, rescored or refitted. The current audit does not launch a panel rebuild, inference, a new tuning search or a backtest. Existing basket, cost and stop rules remain fixed.

## Read scope, checks and saved artifacts

BK read one pinned local shard, projecting 12,075,945 compressed bytes including the explicit metadata and prior-key passes. XOM used the exact-generation SP1500 `daily_bars/part-00000.parquet` object: generation **1783344954247444**, source object size **187,908,505 bytes**, manifest SHA-256 `4a4919d48128d5c573ddbca64649d8cc78783f7bbb818219a3ad5f34728e38a0`. Only metadata/footer and explicit column ranges were retrieved: **115,639,849 payload bytes cumulatively**, below the frozen 128 MiB XOM cap. The full object was not downloaded or rehashed. Exact generation, range bounds and each payload digest were checked; these are range-integrity receipts, not a full downloaded-object digest.

The exact cloud object is `gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/daily_bars/part-00000.parquet`. Authentication used the dedicated `deep-momentum-research` CLI configuration, account `info@galamboscapital.com` and project `trusty-agility-492109-v0`. The Python range transport used an in-process CLI-issued credential without logging or saving its value. Python GCS Application Default Credentials were not retested; successful range reads establish this specific object access, not broader permissions.

The financial pass retained only 4,024 XOM rows. Its peak process RSS was **972,963,840 bytes (0.91 GiB)** and it completed in about 66 seconds. No original compressed source chunks are retained. Private quote outputs remain under ignored `outputs/`; no raw market rows or credentials are published.

Native/source/date and independent-session masks precede financial Python conversion. Selected Parquet column blocks can transiently decode excluded dates or other instruments inside Arrow; this is an output/analysis guard, **not physical per-cell isolation**. No excluded 2024+ financial value was returned, converted to Python or analyzed.

**61 focused implementation tests passed** across BK QA, independent BK checks, footer/range/key guards, XOM financial masking, paired QA and the separate precision diagnostic. Independent readers reconciled both actual bounded outputs. One intermediate synthetic fixture still expected non-session rows after the stricter guard correctly removed them; its expectations and exclusion assertions were corrected, and all final integration tests passed before the real financial read. This is not a claim of a full repository test run or certified economic data. Ninety-five inherited source, request, calendar and policy/code pins were checked unchanged; this does not rehash every model in every old run.

Public audit snapshots are in `scripts/candidate_validation_20261021/`. They retain their executed code hashes and original path plans; replay requires fresh outputs and a newly sealed plan, not notebook Run All. The public proposed configs are `configs/bk_conditional_shadow_20261021.proposed.json` and `configs/xom_conditional_shadow_20261021.proposed.json`; neither is wired into current production readers.

Private evidence is in `outputs/candidate_validation/20261021-bk-xom-v1/`: `handoff.json`, `protocol.json`, `core/result-v1/`, `bk_output_review-v1/`, `identity/`, `xom/footer-result-v1/`, `xom/key-result-v1/`, `xom/financial-result-v1/`, `xom_precision-v1/`, source identity/precision notes, plans, launch seals, test and preservation receipts. The separately proposed route/source amendments retain all original flags and known warnings. No model fitting, inference, backtesting, panel rebuild, cloud write or external message occurred.
