# Workspace and research audit — 2026-11-06

## Corrected ranking baseline — 2026-11-10

The user authorized a new version correcting short-score tie selection and switching stopping to actual equal-weight N=2/N=3 gross portfolio returns. The shared selector preserves historical final-accounting behavior; only the old ranker validation selection was inconsistent. Full held-out candidate cohorts are predicted, with a fixed complete-outcome date mask for the conditional stopping statistic and exact row/date exclusion ledgers. This avoids silently dropping unknown stocks or changing the metric observation set across rounds.

Features, learner parameters, ordinal labels, original fitting row/date split fingerprints, model seeds, 100-member aggregation, 12/3 WFO and 30-round patience are unchanged. Full validation curves and date-level diagnostics persist. Unknown selected exits are not replaced or filled. No new ranker parameter search, loss change, one-minute work or RET modification is part of this experiment. The current RET job continues from its existing frozen sources. Compare all 4,000 new split/learner/forecast-row fingerprints to ranker v1 on completion. Output-free notebook 12 and a new private model release keep historical results immutable.

## Ranker objective and stopping audit — 2026-11-10

Evidence: `data/reports/11_ranker_objective_audit_20261110/AUDIT_REPORT.html`,
`SUMMARY.json`, `TIE_AUDIT.json`, and `CANDIDATE_COVERAGE.json` in that folder.
Completed ranker outputs and running RET sources/settings were not changed.

- All 4,000 member records show examined rounds = selected best rounds + 30.
  Median best is 13; median examined is 43; maximum best is 136. Four original
  member-zero fits reproduced exactly with identical seeds, parameters,
  stopping, selected iterations and saved best validation predictions.
- Confirmed validation-selector defect: ascending stable short selection takes
  different tied IDs from the final portfolio bottom tail. Replaying member
  zero in all forty quarters produced 2,015 member/date query occurrences;
  short sets differed in 33.7% at N=2 and 36.0% at N=3. These overlapping
  occurrences are not independent dates. A flat-score toy permits overlapping
  metric legs, but overlap was not observed in these forty saved members.
- The loss is whole-list rank:pairwise with eight sampled pairs per stock and
  no portfolio/NDCG swap weighting. Ordinal return labels discard magnitude.
  The symmetric stopping proxy considers only N=3, with intra-leg weights
  46.9%/29.6%/23.5% rather than equal thirds; N=2 is not explicitly scored.
- Correcting only short ties left the peak NDCG iteration unchanged in the
  four reproduced paths. Mean N=2/3 validation-return peaks differed in three
  paths, and coincided at round 3 in the 2026Q3 example. Alternative inner
  peaks are diagnostics, not new selected models or improved outer results.
- 11.8% of replayed validation query occurrences omit unknown-label forecast
  candidates. Labels are required for fitting, but a return-based validation
  metric needs the full forecast cohort and explicit unknown-exit accounting.
- Both LambdaRank normalizations are enabled. XGBoost documents possible
  convergence effects; no normalization ablation was run, so this remains
  a hypothesis. About fifty validation dates and per-member rather than
  ensemble scoring also limit conclusions about the economic optimum.

Recommendation: correct and predeclare the selector/metric before tuning, retain
30-round patience, log full curves, and isolate any later both-tail objective
change in a new version. A native one-sided top-k NDCG objective alone does not
implement the desired two-tail portfolio utility. This audit does not establish
that extra boosting rounds improve returns. All prior limitations still apply.

## Parallel learner experiments — 2026-11-09

The user authorized RET tuning on the current 42 inputs and an untuned ranking
baseline in parallel, with a complete 24-setting grid and three historical
whole-date validation splits per setting. No additional grouped-RET control
was requested. The two final runs each retain 100 members per quarter and
12/3 WFO. RET member row splits and class payoffs stay identical to completed
RET42; ranker queries and validation splits use whole dates. Reports must
identify that difference, not attribute every difference solely to the loss.

Native early stopping now supports an explicitly named, maximized metric for
ranking; the default RET mlogloss behavior retains native parity. Ranking
relevance preserves exact return ties, and symmetric validation measures both
long and short ends. Ranker percentile scores never receive uncalibrated return
EV/R² or fabricated class probabilities. Shared portfolio arithmetic is unchanged.
The process cleanup helper now imports subprocess for its existing timeout path.
These changes apply only to new pinned runs, not historical completed results.

Adversarial checks cover whole-query disjointness, future-label poisoning,
source/checkpoint hashes, tuning selection and future-outcome rejection,
maximize/minimize stopping, preserved RET member splits, resumable tuning,
and retained unknown selected exits. The independent 22-check baseline audit
passed after the shared-code updates. All remaining limitations below apply.

## New experiment — 2026-11-09

Notebook 10 adds twelve previous-close five-minute path features to joint30;
historical results and the original thirty input values are unchanged. The new
release retains the exact forecast cohort and labels, and records per-row
availability flags instead of dropping incomplete new inputs. New references
are source-matched, calendar-dense and exclude current observations. Flat
prices/zero volume do not become artificial zero ratios. Native NaN handling
is explicitly allowed for these twelve additions and the existing opening gap.

Adversarial tests cover future perturbations, prefixes, missing slots and whole
sessions, source changes, delivery timestamps, early closes, scale invariance
and the unchanged uncapped learner/WFO contract. The existing independent
22-check historical audit remains a baseline audit, not certification of new
results. Release checks and comparison fingerprints are recorded separately.
The controller now refreshes RUN_STATUS with PROGRESS during fitting; this is
an operational status correction, not a learner change. All remaining research
limitations below still apply. No one-minute work is authorized by this change.

Evidence: `data/reports/08_research_review/AUDIT.json`.
Cleanup/recovery ledger: `data/maintenance/20261106_workspace_review/`.
No new research model was fitted; frozen features and results were not rewritten.

## Corrected defects

- Consolidated the two source trees into the Finance Working Files project.
- Replaced three fitting controllers with one explicit experiment engine and
  shared accounting. Imports cannot overwrite the common feature list.
- Merged seven notebook runners into one. Three notebooks remain active; retired
  source notebooks, controllers and documents are preserved privately.
- Cleared thirty outputs from source notebook 01. Executed outputs stay private.
  Inspection does not start training or automatically rebuild missing releases.
- Replaced three dependency lists with one and seven incremental documents with
  two current documents. Original versions remain in the pre-cleanup archive.
- Reject duplicate stock/date forecasts, duplicate calendar dates, nonfinite
  ranking scores, future training outcomes and undefined supervised labels.
- Validate the named eight/fifteen input matrices explicitly; native NaN only
  for the gap. Production volume construction checks the exact dense calendar.
- Fixed incomplete-payoff coverage accounting: **89 fold-date occurrences** with
  no supervised rows were missing from old counters. Exported corrected counts;
  historical payoff vectors and P&L are unchanged.
- Completed controllers now write terminal progress as well as terminal status.
  Retired arbitrary cap/deadline code is not an active training path.

## Independent numerical checks

All **22 invariant checks passed**: row identities; exchange clocks; signal-day
membership; exact return horizons; independent SQL reconstruction of all six
volume features; tie-preserving labels; open-to-close targets; selected ranks and
weights; unknown exits; P&L/notional; input domains; forty complete 100-model
split/row fingerprints against the baseline; frozen RET arithmetic; uncapped
30-round validation stopping.

Adversarial tests poison future observations, remove calendar days, rescale
volume units, insert duplicate forecasts, introduce forbidden missing inputs and
attempt to execute before input delivery. Numerical passes do not certify PIT
or tradability.

The historical deep-momentum baseline was not a literal paper replication: it
omitted the paper's size input and used custom tree settings and a shorter RET
payoff calibration. The current fifteen inputs are an intentional hybrid, not
that paper's fifteen price inputs. Full historical deviations remain in the
archived methodology documents and `data/reports/05_paper_implementation_audit`.

## Remaining findings

| Finding | Evidence/consequence | Treatment |
| --- | --- | --- |
| Opening delivery after assumed fill | **1,106,428 panel rows** | Research-only; delayed fills remain for later one-minute work |
| Retrospective membership | No archived announcement timestamps; recent changes can precede effective dates | Do not certify historical as-known PIT |
| Adjusted-price vintages | Mixed retrieval dates; source masking is insufficient proof | Corporate-action/vintage sensitivity remains open |
| Eligibility selection | **261,510 / 1,370,296 member-days excluded**, 17 cash dates | Preserve ledger; change cohort only in a new declared experiment |
| Partial volume references | **7,199 panel rows** have fewer than ten valid prior reference days | Export exact rows; full-ten references are a stricter policy choice |
| Random inner validation | Shared dates and overlapping historical inputs | Later compare chronological/date-grouped validation with purging where labels overlap |
| Warm-up/volatility clock | Two-value normalizers; overnight increments treated as one bar | Explicit full-window and separate overnight-volatility sensitivities |
| Nonbinding clipping | Current-inclusive five-sigma threshold cannot bind at deployed decay | Do not claim robust clipping; new transform requires a new release |
| Unknown selected exits | N=2: 68 dates; N=3: 82; four-way comparison omits 84 | Keep unresolved; conditional metrics cannot certify a full path |
| Reused development history | Multiple feature/N choices viewed over 2017–2026 | Establish genuinely unseen future evaluation before final claims |
| Execution and shorts | Bar proxies; unmeasured quotes, slippage and borrow | Cost sensitivities do not establish tradability |

Exact exceptions are in `short_volume_reference_rows.parquet`. Feature tails are
in `feature_distribution.csv`, without ex-post deletion/winsorization. Corrected
training-date coverage is in `training_payoff_date_coverage.parquet`.

The cleanup does not select a new feature policy, change the authorized ensemble
or start a fit. Model building follows review of these choices; old results are
not silently promoted as validation of a changed implementation.


## Protective-stop overlay inspection — 2026-11-10

The three new exit policies inherit opening delivery/fill, adjusted-price vintage, retrospective membership, repeated-history, borrow and transaction-cost limitations. They introduce explicit prior-year calibration warm-up and a stronger, fixed within-model path/scale observation cohort. Gap-only thresholds are undefined for zero gaps; this is exported rather than silently replaced with ATR. Full-session five-minute paths can contain partial underlying-minute buckets; their count is retained. First-touch stops, gap-through opens and independent-leg exposure are research approximations. Baseline source audit and dedicated stop causality/accounting tests must pass; neither certifies execution.


## Standard cost inspection — 2026-11-10

Performance standard is now 10bp per executed entry/exit side, approximately 20bp for a flat round trip at 100% gross. Repricing saved stop exits leaves forecasts, fills, selected stocks, stop calibration and conditional date exclusions unchanged. Stops in release 13 remain calibrated under their original 1bp/side policy; they must not be described as optimized for 10bp. Charging both actual notionals does not resolve optimistic first-bar/opening fills, very tight barriers, adjustment/PIT vintages or unmeasured borrow. The original numerical audit remains separate from new cost-accounting and release checks.


## Cost-specific stop recalibration — 2026-11-10

Release 14 changes the past stop objective and forward execution cost together to 5/10/20bp per side. Original ML models, 12/3 WFO, saved entries, equal-N objective, empirical candidate/tie rules, warm-up and conditional date masks remain fixed. Each preserved trade still pays one entry and one exit; changing a stop affects dollar fees through exit price, not by eliminating an execution. Calibration improvements need not improve subsequent quarters. Compare with original 1bp-calibrated stops charged at the same cost. Dedicated tests and original-policy controls are separate from the unchanged baseline numerical audit. All prior timing, first-bar fill, tiny-barrier, adjustment/PIT, history reuse and borrow findings remain.
