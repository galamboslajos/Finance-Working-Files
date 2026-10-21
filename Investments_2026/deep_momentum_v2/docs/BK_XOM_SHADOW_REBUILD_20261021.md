# BK/XOM shadow rebuild: 2010–2023

The BK/XOM feature, monthly-target and RET payoff rebuild is complete and
independently verified. It is an additive research shadow for the historical
S&P 500 + S&P 400 union. The original panels, fitted models and backtests remain
reference versions; no active input pointer was promoted.

## Measured coverage and target changes

All 152,307 original constituent-month rows and all 168 outcome months across
2010–2023 are preserved. No original member was removed to improve coverage.

| Measure | Reference | BK/XOM shadow | Change |
| --- | ---: | ---: | ---: |
| Original constituent-month rows | 152,307 | 152,307 | 0 |
| Prediction-eligible rows | 140,680 | 140,931 | +251 |
| Available target labels | 140,387 | 140,638 | +251 |
| Eligible rows without an available target | 293 | 293 | 0 |

There are 892 changed class entries: 251 newly available labels and 641 class
switches among rows with labels in both versions. No existing available label
was lost. The class switches include the effect of reranking a changed monthly
cohort; they do not by themselves establish that each stock's return changed.
The comparison does not certify that every previously observed return value or
return-status field is unchanged.

The definitions remain 728 registered features and 715 predictors. Paper15
formulas, the 80% context rule and the existing normalization rules are unchanged.
Nevertheless, repaired source history and changed cross sections propagate into
feature values. Both Paper15 and expanded raw predictor vectors differ on all
140,680 rows eligible in both versions, including all 140,387 rows selected for
supervised fitting in both versions. Across all original rows, including rows
withheld from prediction, at least one expanded predictor differs on 152,307
rows; at least one Paper15 predictor differs on 140,931 rows. There are
46,818,042 changed registered feature cells. These are raw feature comparisons,
not comparisons of fitted preprocessing output or model predictions.

## RET and fitted-run dependencies

The shadow contains 48 causal RET payoff states for outcome months January 2020
through December 2023, signalled December 2019 through November 2023. Each state
uses the unchanged 120 completed calendar-month gate. All 480 class payoffs are
available and differ from their reference counterparts, so all 48 payoff vectors
change. The largest absolute payoff change is 0.00011656731377902851 return units
(about 1.166 basis points), for class 10 at the 2022-11-30 signal. This is a
change in an estimated class payoff, not a measured portfolio return.

All 57,600 RET grid cells have a comparison difference; this includes changed
metadata such as eligible counts and does not mean every numerical monthly
class mean changed. RET's source-return formula and weighting policy remain
unchanged. No model probabilities were recomputed or multiplied by these new
payoffs, and no RET rankings, baskets or backtests were regenerated.

All 64 audited dependency-role records have changed raw-X fingerprints,
observed-class fingerprints and selected masks:

- 32 RF tuning records cover training and prediction roles for 12 inner splits
  and four outer windows, with outer years 2020–2023.
- 32 baseline records cover training and prediction roles for RF/Paper15,
  RF/expanded, XGBoost/Paper15 and XGBoost/expanded in the same four outer years.

Existing fitted states therefore remain attached to the reference input version.
These comparisons do not establish whether fitted preprocessing would absorb
some changes, whether a tuning winner would change, or whether predictions would
change. They are not proof that every estimator fit in the old tuning search
must be repeated. A coherent new model result requires a separately versioned
refit/rescore plan pinned to this catalog.

## Verification and compute boundary

- All 14 unamended annual reconstructions reproduced their saved reference
  controls exactly before the BK/XOM differences were accepted.
- All 14 independent annual numerical checks passed. Across the full shadow,
  2,284,605 Paper15 cells, monthly endpoint returns and 140,638 available target
  classes were independently recomputed.
- Final independent verification passed for 1,670 source month/class cells,
  57,600 RET grid cells, 480 payoffs and 64 dependency-role records.
- The 298 daily and 415 context feature formulas rely on the frozen workers,
  exact unamended parity and structural tests; each formula was not independently
  reimplemented in the final verifier. The streamed old/new expanded-feature
  comparisons were reviewed separately.
- No estimator fitting, preprocessing fitting, inference or backtesting ran.

The independent receipt is `run/verification/independent_shadow.json` inside the
installed package, SHA-256
`ffb32ca6dbf4307ae52a869618f6e8b9bf9fb85b46d921969da03e19f2e5ab93`.
Its exact annual/aggregation pins and the installer receipt are retained.

## Source and method boundaries

BK uses the reviewed postmerger ordinary-share candidate `sec_zq1J05` only in
the approved interval, with price needs beginning 2008-01-04. XOM uses the
published SIP raw/all pairs beginning 2016-01-04 under
`retain_precision_warned_published_all`; earlier missing history is not filled.
The computational key `dmv2_sip:siblis_isin_b284bb7497616f5afba0:all` is explicit
and does not certify an Intrinio native security identity. Raw OHLCV populates
raw prices, and published all-adjusted OHLCV populates adjusted prices. Unavailable
adjustment-factor, dividend and split fields remain null.

All 69 original XOM precision-warning dates remain in QA/lineage outside the
predictors. Their cause has not been proven to be rounding. With this source
history, the earliest potential Paper15-eligible XOM signal is 2017-01-31 for
February 2017; 503-session features cannot complete before the 2017-12-29 source
session. No earlier missing history is fabricated.

The existing BBBY quarantine, El Paso, independent quote and corrected annual
source policies remain in place. Missing 2014–2019 warmup/context artifacts were
recovered from authenticated original sources using the existing amendments;
every recovered unamended annual control passed exact parity. The bounded old
quote-amendment cache decode and the earlier metadata inspection deviation are
recorded in the protocol addendum. No model pickle was loaded.

The original historical membership keys, monthly endpoint-return target,
ten-class midrank/tie rule, common-five/Paper15 eligibility, 80% context policy,
503-session warmup, daily indicators with month-end snapshots and rolling
ten-year chronology remain unchanged. Market capitalization is not introduced.

Remaining limitations are explicit:

- RCM's 22-date and DISCA's five-date execution supplements were not applied to
  learning inputs. Field-complete OHLCV repair remains outstanding.
- Cash/stock/ADS terminal-event economics for monthly labels remain a separate
  unresolved policy.
- Identity, adjustment vintage, knowledge-time provenance and survivorship
  remain provisional. Arithmetic verification is not a blanket PIT or
  survivorship-bias certification.

## Locate and continue from the installed shadow

The project-relative entry point is
`outputs/bk_xom_shadow/2010-2023-20261021-v1/installed_catalog.json`.
The private output package is ignored by Git; this public document and the two
worker modules do not embed its financial tables. No commit or merge was made
in this round.

The catalog contains verified absolute destination pins for all 14 annual
feature, coverage, target/state, original cohort, daily context, raw daily and
warmup artifacts, plus all 48 RET states. Use those pins in future loaders.
`daily_prices_raw` is the disjoint outcome-year core suitable for later annual
concatenation. `feature_history_daily` and `daily_history_raw` contain overlapping
warmups and must not be blindly concatenated for backtesting.

Copied audits and plans retain their original absolute private staging paths
as immutable provenance. `relocation_index.json` maps these to installed paths
without rewriting their bytes. `installation_receipt.json` records original-input
preservation and byte-identical installation checks. Failed attempts, including
the metadata-only roundtrip attempt, remain archived. The installed catalog is
the destination locator; archived runners are execution records with legacy
paths and require a fresh sealed plan to replay.

Review the measured dependencies and remaining learning-data limitations before
selecting the next refit/rescore scope. Keep current class-basket and stop-loss
rules fixed for any later comparison unless a new experiment explicitly changes
them. Alternative labels are a separate future experiment.

The two public helpers are copied without byte changes to
`src/dmv2/bk_xom_shadow_sources.py` and `src/dmv2/bk_xom_shadow_impact.py`. The two
public unit tests differ from the archived originals only in their imports;
`public_payload/import_adaptations.json` records the replacements and hashes.
