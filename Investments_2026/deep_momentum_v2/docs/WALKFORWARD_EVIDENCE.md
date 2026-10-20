# Walk-forward evidence and outstanding checks

Status snapshot: October 16, 2026. All sixteen authorized model fits and the
initial selected-horizon preflight are complete. Nine new-period merger events
have passed the bounded quote-unit checks and entered a separate final registry.
Independent feature verification V2 has status `PASS` for all 48 months and
43,655 original rows. Independent model verification V2 has status `PASSED` for
all sixteen models, eight training-only preprocessors and 48 forecast months,
with zero verifier fits. The separate additive V2 daily-accounting run and its
independent audit are complete for all 64 declared scenarios: 28 complete proxy
paths and 36 withheld paths. The accounting audit has status `PASS`; it verifies
the complete-path arithmetic and preserves unresolved exposure on withheld
paths, rather than certifying performance for all strategies. This note reports
no portfolio returns or ranking.
The experiment specification is [WALKFORWARD_2020_2023.md](WALKFORWARD_2020_2023.md)
and the frozen configuration is
[`walkforward_2020_2023.json`](../configs/walkforward_2020_2023.json).

## Chronology and evaluation scope

This evaluation uses historical S&P 500 and S&P 400 membership together. Each
annual model uses the preceding ten outcome years and makes twelve monthly
forecasts. January uses the prior December month-end signal. A training label
must be fully observed by the last XNYS session of the prior year.

| Forecast outcome year | Training outcome years | Latest training label observation |
| --- | --- | --- |
| 2020 | 2010–2019 | 2019-12-31 |
| 2021 | 2011–2020 | 2020-12-31 |
| 2022 | 2012–2021 | 2021-12-31 |
| 2023 | 2013–2022 | 2022-12-30 |

The four arms are RF/Paper15, XGBoost/Paper15, RF/expanded and XGBoost/expanded:
four fits per year, exactly sixteen estimator fits in the authorized protocol.
They use frozen parameters and seeds, with no parameter search, early stopping,
calibration or performance-driven refits. Each training-only preprocessor is
shared by the two estimators using its feature set. Earlier evaluation years may
enter a later training window only after their outcomes have become observable.
This is sequential walk-forward evaluation; it is not four years of forecasts
from one unchanged model. Paper15 is the project's no-size adaptation; these
arms do not reproduce the paper author's original sixteen-input, 100-fit method.

The completed accounting grid has 48 monthly decisions and all 1,006 XNYS
sessions from January 2, 2020 through December 29, 2023. The daily portfolio
continues across annual model updates without an annual NAV reset or forced
liquidation. The final authorized liquidation uses December 29, 2023 close.
There is no 2024 execution-price or outcome analysis in this experiment.

Class 1/10 and class 2/9 baskets remain separate, with 50% long and 50% short
capital weights. Names on each side equal the smaller of ten and the available
eligible candidates on the two sides. Ranking uses the corresponding predicted
class probability and the frozen deterministic tie rule. The unstopped and 10%
per-asset stop variants are separate, as are the declared transaction-cost
sensitivities. RET is not evaluated in this class-basket experiment; its existing
120-month eligibility gate is unchanged. The original 2018–2019 development runs
remain preserved references.

## What the independent feature check covers

The standalone zero-fit feature verifier checks original cohort denominators,
dated membership and identity rules, recorded chronology, the BBBY quarantine,
and retention of valid historical rows subject only to execution restrictions.
Monthly targets, outcome daily data and past daily feature history remain
separate. Missing future labels do not remove prediction rows. Recorded
availability clocks are checks on the historical reconstruction, not proof that
all source information was available to a researcher at the historical time.

The numerical scope is explicit: recompute all fifteen Paper15 variables and
the shared monthly target classes, including exact-tie and missing-return rules.
For daily indicators, independently recompute six formulas on a deterministic
sample of the first two eligible distinct securities per month: one-session and
twenty-session simple returns, the corresponding log returns, the twenty-session
moving-average gap and twenty-session realized volatility. These use data no
later than the signal. The check does **not** independently certify every one of
the 715 expanded feature formulas. Matching the feature schema, source pins and
past-only windows is broader structural evidence, not a substitute for formula
recomputation.

The verifier has no model fits and a separate bounded execution budget. Its
actual V2 receipt, `verification/features.json`, reports `PASS` for the stated
checks over all 48 months and 43,655 original rows. Model/preprocessing
verification and daily-accounting verification have their own receipts; a
feature PASS cannot establish model or backtest correctness.

The first actual feature-audit attempt failed a verifier contract check and
emitted no PASS receipt. V1 incorrectly required every non-key value on a
quarantined BBBY row to be null. A bounded diagnosis confirmed all 663
stock-specific or member-specific columns were null on the six quarantined
rows, with identity, request, prediction and label admission all false; the
remaining values were only the 65 shared market-state columns broadcast to all
original same-signal members. V2 permits only those 65 explicitly enumerated
names and requires identical same-signal values and missingness. It retains the
null requirements for every own-price, Paper15, cross-sectional, relative or
unregistered column and the original admission guards. The original verifier,
tests and failure evidence remain preserved. Its 21 synthetic tests passed
before the separately executed full numerical V2 audit passed. That PASS retains
the scope limits above, particularly the six sampled daily formulas rather than
certification of all 715 expanded formulas. No source, feature, label, selection
or fitted model changed for this verifier correction.

The independent model V2 receipt, `verification/models.json`, reports `PASSED`.
It checked the sixteen completed estimator fits and eight preprocessing fits,
training-only preprocessing statistics and prepared matrices, saved-model
inference and monthly classification scores. All reloaded probabilities matched
the saved forecasts exactly. The verifier performed zero estimator or
preprocessor fits; the original training work was not repeated. The first model
verifier had assumed a dictionary file existed at the run root. V2 resolves the
dictionary through the frozen primary-reference audit/report pins and checks
all fourteen annual provenance hashes and footer schemas. This verifier
contract correction changed no source data, features, models or selections.
Neither the feature nor model receipt substitutes for the separate daily-accounting receipt described below.

## Nine legal reviews, unit checks and final event admissions

The following nine merger events have completed primary legal review and bounded
post-fit quote-unit review, followed by admission to the separate final
new-period action registry. The table is a public evidence index; the pinned
private registry and its review receipt govern execution. An application session
is the first suspended or nontrading session used for accounting and can differ
from legal completion.

| Historical predecessor | Legal completion | Admitted application session | Consideration | Primary timing/terms evidence |
| --- | --- | --- | --- | --- |
| Cypress (CY) | 2020-04-16 | 2020-04-16 | Cash | [Nasdaq notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2020-64) |
| Allergan (AGN) | 2020-05-08 | 2020-05-11 | Cash and AbbVie shares | [Closing 8-K](https://www.sec.gov/Archives/edgar/data/1578845/000119312520137717/d927229d8k.htm), [NYSE notice](https://www.sec.gov/Archives/edgar/data/876661/000087666120000384/ruleprovisionnotice.htm) |
| WPX Energy (WPX) | 2021-01-07 | 2021-01-07 | Devon shares | [Closing 8-K](https://www.sec.gov/Archives/edgar/data/1518832/000110465921002045/tm2039568d3_8k.htm) |
| Cantel Medical (CMD) | 2021-06-02 | 2021-06-02 | Cash and STERIS shares | [Closing 8-K](https://www.sec.gov/Archives/edgar/data/19446/000119312521185492/d158762d8k.htm) |
| Cimarex Energy (XEC) | 2021-10-01 | 2021-10-01 | Cabot/Coterra shares | [Closing 8-K](https://www.sec.gov/Archives/edgar/data/1168054/000110465921122062/tm2129003d1_8k.htm) |
| Sanderson Farms (SAFM) | 2022-07-22 | 2022-07-22 | Cash | [Nasdaq notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-143) |
| American Campus Communities (ACC) | 2022-08-09 | 2022-08-10 | Cash | [NYSE full Form 25 submission](https://www.sec.gov/Archives/edgar/data/876661/000087666122000619/0000876661-22-000619.txt) |
| Nielsen (NLSN) | 2022-10-11 | 2022-10-12 | Cash | [Closing 8-K](https://www.sec.gov/Archives/edgar/data/1492633/000119312522260583/d407513d8k.htm) |
| Alleghany (Y) | 2022-10-19 | 2022-10-19 | Cash | [NYSE notice](https://www.sec.gov/Archives/edgar/data/876661/000087666122000857/ruleprovisionnotice.htm) |

The immutable normalized nine-event source proposal retains `verified:false`
and its original pending-review state, preserving the source-document flags.
Its pinned SHA-256 is
`d638fc29591c8bcec22612ceefc3355b5eacfd8cbdec01e2e69517077ed84a8f`.
It was not rewritten or promoted in place. The separate admitted registry
`final_actions_v1.json` has SHA-256
`7513e3f1cd9ce6c1d58b4febe402e7a9dc590f4d4033433afc2203feed028a62`.
A legacy source draft's `verified` field alone cannot replace the completed
post-fit unit-review receipt.

The completed review checked actual accepted predecessor basis observations,
successor identity and required successor observations for stock consideration,
and paired raw/adjusted OHLC consistency against the unchanged absolute residual
ceiling of 0.0002. All nine events passed signed raw-share entitlement, cash
receivable/liability and successor-unit algebra. The bounded review covered 14
exact requested keys and 16 financial rows, including the separate Cypress
diagnostic, within 25,616,731 projected compressed bytes. It computed no portfolio
PnL. Final registry admission occurred separately; the unit helper did not
promote source drafts. This PASS establishes reviewed conversion units, not
complete historical identity, distribution or physical-settlement certification.
An accounting action does not create missing tradable quotes. An order newly
selected after retirement does not inherit a prior holder's merger entitlement
or obtain a replacement security.

Ordinary distributions, closing consideration and fractional-share settlement
are kept distinct. No additional dividend or cash-in-lieu amount is invented.
For example, Alleghany's difference from the initial proposal reflects the
[documented adviser-fee deduction](https://www.sec.gov/Archives/edgar/data/775368/000114036122013997/ny20003698x1_prem14a.htm),
rather than a separate closing dividend.
Cash settlement timing and theoretical fractional units retain their disclosed
simulation limits. None of these legal reviews independently certifies the
provider's full historical adjusted-distribution representation.

## Completed additive accounting and independent verification

The first full-period accounting attempt stopped at a strict cross-source
identity guard. Its partial output and failure evidence remain unchanged. A
separate V2 accounting implementation converts both runtime identity-conflict
exceptions into explicit failures of the affected path, retaining its held lots,
pending orders and withheld metrics while allowing other scenarios to finish.
The original structural identity guards and verified merger-successor exception
remain unchanged.

One separately pinned provisional execution handover permits the exact April 1,
2021 PENN source transition, with the old source absent from current targets and
one uniquely selected provider. It is supported by the
[March 2021 S&P index-migration announcement](https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20210312-1336360/1336360_march2021-5146-shuf-pr.pdf),
the [August 2022 corporate-name amendment](https://www.sec.gov/Archives/edgar/data/921738/000092173822000034/penn-20220804.htm),
and separately reviewed local metadata. The provider evidence remains
provisional; the distinct source/unified records and their review flags are not
merged or promoted. The later December 2023 permanent-identifier evidence does
not independently certify that identifier on both sides of the 2021 transition.

The exception changes no feature, label, forecast, candidate rank or selected
row. The old and new origin lots remain separate in accounting, physical orders
net on the unchanged provider at the common execution mark, and the new monthly
episode resets its stop reference. It is not a general alias exception. All
unmatched runtime conflicts fail closed. A separate applied-handover table and
approval/evidence pins accompany the V2 phase, using a fresh output directory.
The actual independent V2 accounting receipt, `verification/backtest.json`,
reports `PASS` over all 64 scenarios, all 48 months and all 1,006 sessions.
It separately verifies the exact registry/approval bindings, applied handover
records and cash/units transitions. All source pins remained unchanged and all
43 files of the original failed V1 attempt remained preserved. The verifier
performed zero model or preprocessing fits and read no 2024+ financial values.
The separately completed backtest audit has raw SHA-256
`201dcf7f5622790d519b61a940a8017e107c5c54f00f2ada8f963cfcb4554f90`.

Peer review caught two verifier contract defects before the actual accounting
audit. The initial V2 verifier incorrectly required an HTTPS string where the
approved registry used exact pinned local JSON evidence references. Its
replacement authenticates every referenced payload and its pinned primary
findings. A separate replay correction handles an actual first failed fill
uniformly for monthly rebalances, nonmonthly opening stops, intraday stops and
terminal liquidation, including allocated fees and surviving stop/reference
values. Both intermediate versions are preserved. Thirty-one focused and
baseline synthetic methods passed before the actual V2 audit; no production
accounting, source, feature, model or selection changed for these verifier fixes.

The 28 complete proxy paths comprise both RF/Paper15 class 2/9 variants, both
XGBoost/Paper15 class 2/9 variants, both RF/expanded class 1/10 variants, and the
stopped XGBoost/expanded class 2/9 variant, each under four cost assumptions.
The other 36 paths remain withheld. Their first unresolved marks are associated
with the CIT, WBD, GRUB, FBIN or VYX issues below. This is a coverage distinction,
not a model-performance ranking: using the successful subset to choose a model
or stop rule would condition the comparison on later data availability.

The accounting-only implementation is `walkforward_accounting_v2.py`, invoked
through `scripts/run_walkforward_backtest_v2.py` only for a separately authorized
new phase with explicit parent, failed-attempt and registry pins. The actual
independent verifier is `scripts/verify_walkforward_backtest_v2.py`. Their V1
predecessors and the failed V1 output remain historical references. All fits,
forecasts and selected baskets remain the frozen originals.

Master notebook section 23 reads only pinned saved reports after the exact
feature, model and backtest phase audits and their three matching independent
receipts pass. Run All launches no fit, rebuild, model reload or source-price
scan. The displayed complete/withheld status must remain attached to every
comparison. Daily feature history and daily execution data remain separate from
monthly signal and target tables.

## Unresolved source and accounting leads

These findings preserve the frozen features, labels, forecasts and selected
exposures. They authorize no new exclusions, fabricated exits, substituted
share classes or automatic extra fits.

| Lead | Outstanding issue and required separation |
| --- | --- |
| CIT, 2021-05-05 | A frozen OHLC-quality refusal remains because a following-control dividend needs separately reviewed share units. This is a source-quality problem, not evidence of a merger on that date. |
| WBD, 2022-04-04 through 2022-04-08 | Missing regular legacy Discovery Series A observations remain unresolved. The [Nasdaq notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-63) distinguishes that history, Series C, and the separate when-issued market. The later legal conversion cannot manufacture the five earlier bars. |
| FBIN, 2022-12-14 | An interior gap overlaps the [MasterBrand distribution](https://www.sec.gov/Archives/edgar/data/1519751/000119312522306146/d420098dex991.htm) and associated unit review. The continuing parent remains outstanding and changes ticker; this parent-retaining spinoff cannot be represented by extinguishing the parent through the merger schema. |
| VYX, 2023-10-16 | An interior gap overlaps the [NCR Atleos distribution](https://www.sec.gov/Archives/edgar/data/70866/000119312523257806/d569776dex991.htm). Voyix remains the continuing parent. The distribution needs its own reviewed entitlement and price-unit treatment, rather than a full stock conversion. |
| GRUB, 2021-06-15 onward | [Primary evidence](https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2021-22) explicitly warns that the new Just Eat Takeaway ADS reuses the old Grubhub symbol. Legal terms and a permanent successor identifier do not yet establish a verified local successor quote history or its raw/adjusted units. |

There is also a separate Cypress source-impact limitation: primary evidence
places the halt on April 16, 2020, while a vendor key for that date is admitted
in the frozen historical views. The bounded post-fit diagnostic found that its
OHLC fields are flat at the preceding session's close and differ from the legal
cash consideration. This is not a tradable bar, and no execution uses it.
Metadata establishes a possible path into later daily cohort-context windows;
the numerical impact on those features has not been measured. The frozen source,
feature and forecast outputs remain unchanged, with no corrective refit. Any
source correction needs an explicit input-impact review and a separately
approved build.

Original feature-requested securities cannot be supplemented after fitting
under the successor-only execution supplement policy. A newly evidenced
successor can receive a small, separately pinned execution-only source where
that policy permits; it cannot silently repair old feature inputs. Broader
repairs require a new reviewed protocol and a decision on rebuild implications.

An unresolved held mark retains the selected exposure and withholds affected
full-period metrics. The declared 64-scenario computation and independent audit
are complete; historical data coverage remains incomplete for 36 paths.
Performance may be discussed only with the final registry, path-completeness
audit and independent accounting receipt attached. A PASS for preserved
withheld exposure is not economic-performance certification for that path.
Historical identity, knowledge-time and complete survivorship certification
remain limited by the source data and the declared provisional identity policy.
