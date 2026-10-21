# Corrected-input refit: input decisions, 21 October 2026

This is an additive research rerun. Original panels, fits, forecasts, backtests,
failed attempts, and checkpoints remain reference artifacts. No new target family,
hyperparameter search, trading rule, or source-quality tolerance was introduced.

## Inputs admitted

The independently verified BK/XOM shadow catalog supplies the 2010–2023 learning
panels and the 48 causal RET payoff states. It adds 251 prediction-eligible rows
and 251 observed labels relative to the preceding inputs. All fourteen annual
files and all forty-eight payoff states are reused byte-for-byte. No further
annual feature or label rebuild was necessary after the two source checks below.

Historical membership, the provisional identity policy, Paper15, the expanded
715 predictor set, and the 80% feature context rule are retained. Daily source
prices remain distinct from monthly features and labels.

## Learning patches rejected

Both candidate patches were checked against the already frozen 2 basis point
volume compatibility tolerance. Neither contributed learning rows.

| Case | Decision | Consequence |
|---|---|---|
| RCM predecessor history | Reject complete learning patch. Price controls match, but one volume control differs by 0.025297%, above the 0.02% tolerance. | Do not partly admit prices or relax the tolerance after observing the check. |
| DISCA/WBD transition history | Reject complete learning patch because volume controls fail the same tolerance. | Retain the documented learning gap. |

These learning decisions do not invalidate the earlier independently checked,
execution-only price amendments. The daily backtest retains the existing forty
quote amendments: twenty-two RCM, five DISCA, and thirteen ADS bars, together with
the established corporate-action and identity-handover treatment.

## Label and fitting decisions

The target remains the provider-adjusted monthly endpoint return, with the
existing cross-sectional class and tie rules. A corporate-action cash or stock
settlement is not substituted as an invented predecessor quote. Twenty-nine
known event-overlap labels remain missing. Original rows and denominators are
retained; future endpoint availability never determines signal-time prediction
admission or portfolio selection.

The run has twenty estimator fits: five arms for each of 2020, 2021, 2022, and
2023. The four original RF/XGBoost arms are accompanied by RF expanded using the
previously selected annual CV settings. That fifth arm is not a newly retuned
winner. Each forecast year uses the preceding ten outcome years. Eight training-
only preprocessing states are shared across the appropriate estimators.

Independent annual and final checks must pass before scores or new backtests are
reported. The new forecasts, not saved historical portfolios, generate fresh RET,
class 1/10, and class 2/9 selections. Selection rules, weights, execution clock,
stop handling, and cost sensitivities remain unchanged. Full-period performance
is withheld for any incomplete path; missing holdings are not silently dropped.

## Limits on interpretation

The refit is conditional on the observed endpoint data and the provisional
historical identity policy. It does not establish comprehensive economic-return,
knowledge-time, or survivorship correctness. Missing terminal labels, rejected
learning histories, XOM's pre-2016 absence, and the retained source precision
warnings remain explicit limitations. A different economic-return or path-risk
label would be a separate experiment, after this comparison is complete.

The 2020–2023 outcomes have already been inspected during this research. This
corrected rerun is a research comparison on that previously reviewed period; it
does not create a new untouched holdout merely by rebuilding the inputs or
refitting the estimators.

The staged manifests, source receipts, independent checks, immutable plans,
resource guards, and unsuccessful attempts are preserved with the run. No cloud
mutation, external message, Git commit, push, or merge is part of this rerun.
