# Workspace and research audit — 2026-11-06

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
