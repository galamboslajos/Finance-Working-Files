---
name: han-qin-deep-momentum
description: Use when designing, implementing, or reviewing the Han-Qin Bimodality Everywhere monthly return-classification and probability-to-payoff method in Deep Momentum V2, including U.S. S&P 500/400 adaptations, added features, and forest ensembles. Does not describe a generic LSTM or Sharpe-loss Deep Momentum Network.
---

# Han-Qin paper method for Deep Momentum V2

Read the V2 `AGENTS.md` first. Before changing features, labels, models,
portfolios or chronology, read the canonical
[paper reference](../../../docs/PAPER_REFERENCE_HAN_QIN_2026.md) and
[U.S. decision ledger](../../../docs/US_REPLICATION_PLAN.md). Load only the
section needed for a narrow factual question; load both completely for a new
experiment design. Read
[`source_manifest.json`](references/source_manifest.json) when checking source
identity. The original paper is local, ignored and not bundled with this skill.

## Preserve the paper mechanism

- Lock the supplied March 26, 2026, 115-page Han-Qin PDF. Check the recorded
  SHA-256 if a PDF is available. Do not substitute a newer SSRN version silently.
- Predict ten **next-calendar-month** relative-return classes, retaining all
  class probabilities. The paper uses multiclass XGBoost, 16 inputs and an
  annual 100-fit probability ensemble; it does not prescribe an LSTM.
- Follow the exact calendar indexing: `MOM1=r[t]`; longer `m=3,6,9,12`
  features compound `t-m+1` through `t-1`, with 2/5/8/11 returns. Include
  five normalized momenta, five cross-sectional means, five SDs and size decile.
- Expand classifier training history, but use trailing ten-year historical
  realized class payoffs for RET. Average monthly class means across time;
  do not pool stock rows or estimate payoffs from predicted-mode groups.
- Persist class-to-probability-column mapping. In the reference's notation,
  class 1 is best and 10 is worst. Align every fit before averaging probabilities.
  Calculate `RET=sum(p*mu)` using only completed outcomes available at the cutoff.
- Distinguish mode-selected XGB portfolios from top/bottom-decile RET portfolios.
  Read the reference before implementing DPR, SRP, bimodality or turnover:
  their formulas/units and several source ambiguities are easy to misread.

## Keep adaptations and evidence explicit

- Label facts as PAPER, INTERPRETATION, VERIFIED LOCAL, PROPOSED or OPEN.
  Never invent author settings for software defaults, seeds, early stopping,
  ties, size orientation, SD/variance estimators, winsorization or execution.
- Historical S&P membership plus Intrinio is a data/universe adaptation to
  broad CRSP. Do not use today's constituents or future quotes for eligibility.
  Keep missing/terminal outcomes in coverage and economic accounting.
- Historical market capitalization is absent from the currently downloaded
  source schemas. A price-only 15-input experiment is an ablation, not the
  complete paper baseline. Never manufacture size from price or dollar volume.
- Forests and added features are proposed extensions. Preserve the probability
  interface and RET layer when evaluating an estimator change. Separate
  feature changes from model changes; validate/calibrate/weight only on past
  development data. Preserve protected 2024+ values under the project rules.
- Sadig's daily five-feature scalar 20-session recovery and legacy Canadian
  code are references, not this paper's specification. Keep new work in V2.
- Compare reported values using the correct period, country, weighting and
  annualized-mean convention. RET did not improve U.S. Sharpe over XGB in
  2010-2023. Global and small-stock results are not S&P performance targets.

## Before implementation or reporting

1. Resolve required OPEN design decisions in the ledger/draft configuration;
   do not treat a draft file or paper as execution authorization.
2. Record causal cutoffs, complete calendar windows, class/payoff mapping,
   source pins, software/seed settings and portfolio accounting conventions.
3. Verify bounded pre-holdout identity, return and feature cases before scale-up.
   Follow the project's execution and cloud-policy workflow for heavy work.
4. Report both prediction and economic coverage. File integrity, a successful
   model fit and plausible accuracy do not establish investable performance.
5. Keep the canonical reference synchronized when correcting interpretation;
   preserve the original source lock and distinguish new evidence from proposals.
