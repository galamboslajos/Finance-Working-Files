# Starting research workflow

Status: SETUP AND PAPER REVIEW COMPLETE; EXPERIMENT DESIGN OPEN.
New strategy research begins here; the previous attempt is a reference.

The locked paper is Han-Qin, *Bimodality Everywhere*, March 26, 2026. Read
[its precise reference](PAPER_REFERENCE_HAN_QIN_2026.md) and
[the U.S. decision ledger](US_REPLICATION_PLAN.md). It uses monthly ten-class
XGBoost probabilities and historical-payoff reclassification, not a required
neural architecture. Historical S&P 500/400/union data, additional features and
forest ensembles are our intended adaptations.

1. Audit dated U.S. membership and identity joins, return adjustments,
   availability, delisted names and corporate actions on a bounded approved
   pre-holdout sample. Preserve exclusion denominators and prior failure evidence.
2. Identify an approved historical market-capitalization source. Current
   downloaded schemas lack capitalization and shares outstanding. This blocks
   the full size-feature baseline and value weighting; a 15-feature experiment
   must be explicitly labeled an ablation.
3. Choose the primary historical S&P cohort and audit usable history. Freeze
   feature/label conventions, cleaning, signal/entry timing, completed-outcome
   cutoffs, annual retraining, class-payoff windows, validation, costs and trials.
   Resolve paper underspecification in the ledger instead of inventing author settings.
4. Implement and verify the paper mechanism with monthly momentum, mode-XGB
   and RET under a common cohort and economic ledger. Keep prediction quality
   distinct from portfolio/valuation completeness. The paper's random historical
   inner split and any temporal variant must have different named protocols.
5. Compare forest probability estimators on the same baseline inputs. Compare
   added feature groups separately, then combine using development-only
   calibration/weights if justified. Neural models or direct regressors are
   optional extensions, not prerequisites from this PDF.
6. Measure a small engineering sample, then prepare a bounded full experiment
   with resource estimates, cost limits, trial bounds, acceptance criteria,
   output provenance and cleanup controls under the project execution workflow.

The prior recovery used 1998 warmup, a 1999-2023 accepted signal panel,
2009-2023 primary inputs and 2015-2023 test folds. These are reference roles,
not the new design. Ten years of usable history are required before a
paper-style initial fit. Product-wide early dates do not establish that history.
The prior 2024+ holdout remains protected until a new protocol explicitly
determines its use.

The previous comparison's economics were inconclusive due to missing valuations
and event/identity limitations. Successful model fits, source-file verification
or feature construction are not evidence of complete economic performance.
