# Starting research workflow

Status: DATA FOUNDATIONS, MONTHLY RETURNS AND BOUNDED FEATURE WORKERS IMPLEMENTED; EXPERIMENT DESIGN OPEN.
New strategy research begins here; the previous attempt is a reference.

**USER DIRECTION, 2026-10-15:** initialize research in 2010 and use rolling ten-year
training / one-year outcome tests. The canonical settings and readable master
notebook are described in [the notebook workflow](NOTEBOOK_WORKFLOW.md).
Nominal tests run from 2020 through the existing pre-holdout boundary in 2023:
2010–2019 training precedes 2020 testing, advancing annually to 2013–2022
training / 2023 testing. There are four nominal folds; the rolling window and
monthly prediction frequency are unchanged. The [2010–2023 endpoint audit](INITIALIZATION_2010.md)
measures complete 15-input paper coverage under the accepted provisional identity
policy. Full 715-predictor/context coverage remains open. The feature worker
uses exact pre-signal warmup, including pre-2010 prices, described in [FEATURES.md](FEATURES.md).

The first source/date/identity/quote contracts and adversarial tests are in
[data foundations](DATA_FOUNDATIONS.md). One real 2010-01-29 combined snapshot
retains all 902 names. Its original strict-policy audit counted 130 clear
identity routes, 755 review-flagged routes, ten collisions and seven unlinked
names. The [active provisional policy](PROVISIONAL_IDENTITY_POLICY.md) accepts
unique resolved review-flagged routes while preserving the flags; missing/ambiguous
routes and collisions remain blocked. Dated identity evidence remains unverified,
without blocking provisional development or changing original denominators.

The [monthly worker and notebook stage](MONTHLY_RETURNS.md) now preserve daily
data independently and construct bounded provider-adjusted ratios from full
preceding-month cohorts. Missing/identity/unrequested states remain visible.
Economic event accounting and knowledge-time certification remain unresolved.

The locked paper is Han-Qin, *Bimodality Everywhere*, March 26, 2026. Read
[its precise reference](PAPER_REFERENCE_HAN_QIN_2026.md) and
[the U.S. decision ledger](US_REPLICATION_PLAN.md). It uses monthly ten-class
XGBoost probabilities and historical-payoff reclassification, not a required
neural architecture. Historical S&P 500/400/union data, additional features and
forest ensembles are our intended adaptations.

1. Audit dated U.S. membership and identity joins, return adjustments,
   availability, delisted names and corporate actions on a bounded approved
   pre-holdout sample. Preserve exclusion denominators and prior failure evidence.
2. Apply the user's decision to exclude market capitalization. Start from
   the 15 momentum/state inputs before extensions, omitting the paper's size
   feature, cap-based screens and cap-value weighting. Keep this departure
   explicit; obtaining capitalization is not a task or blocker. Portfolio
   weighting remains a separate OPEN decision.
3. Choose the primary historical S&P cohort and audit usable history within the
   selected 2010 initialization / rolling 10-1 chronology. Freeze
   feature/label conventions, cleaning, signal/entry timing, completed-outcome
   cutoffs, annual retraining, class-payoff windows, validation, costs and trials.
   Resolve paper underspecification in the ledger instead of inventing author settings.
4. Implement and verify the paper mechanism under the declared 15-input
   V2 adaptation, with monthly momentum, mode-XGB
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

The user-authorized [feature stage](FEATURES.md) registers 728 definitions with
715 eligible predictor names. It preserves original cohorts, signal-only history
and a separate completed-outcome X/y interface. Two-ID samples withhold complete
cohort statistics; no model fit or full-context scan is implicit.
