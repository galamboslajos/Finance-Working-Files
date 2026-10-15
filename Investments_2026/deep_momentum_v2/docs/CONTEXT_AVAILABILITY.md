# Added-context availability: 80% primary, 90% sensitivity

**USER DIRECTION, 2026-10-15:** use the verified 80% panel as primary and retain
90% as the sensitivity check. This supersedes the earlier 90%-primary decision.
Both existing panels remain immutable; promotion changes their declared roles,
not their values. The threshold is a research assumption, unrelated to Sadig's
training-feature admission or per-stock missingness percentages. No model
performance was used to select it.

## Population and gates

The active named policy is `available_cohort_80pct`. The sensitivity policy is
`available_cohort_90pct`; `complete_cohort` remains available for historical
comparisons. The notebook, normalization and full-build configurations agree
on the active 80% policy. The paper's normalization policy is unchanged.

For each variable, date and declared context group:

1. Start with every original dated historical member before price joins.
   Members with missing, ambiguous or colliding identities remain in the
   denominator. Apply the provisional identity policy with all flags retained.
   Do not use today's constituents, future quotes or outcomes for eligibility.
2. Request every identity-admitted native ID in that group. An unrequested
   member is a request failure; the 20% allowance cannot absorb it. Withhold
   the group when requests are incomplete, irrespective of observed coverage.
3. Count finite observations satisfying the feature's exact lookback, quality,
   route and signal-cutoff rules. Primary coverage requires
   `10 * observed_members >= 8 * original_members`: exactly 8 of 10 passes,
   7 of 10 fails. Sensitivity requires `10 * observed >= 9 * original`.
4. Calculate only when the request and coverage gates pass and the statistic
   is mathematically defined. Preserve single-member, zero-MAD,
   zero-benchmark-variance and other undefined states. Do not impute, forward
   fill, replace missing values with zero or compress missing sessions.

Cross-sectional percentile ranks and median/MAD scores use each base variable's
valid members. Missing individuals retain missing ranks/scores. Average ties
and formulas are unchanged. The union and each sleeve qualify independently.
When both thresholds pass, they calculate on the same observed contributors;
changing the gate does not change those shared values.

Market-state statistics describe the observed historical subset after the gate;
they are not certified official full-index returns. Their shared values remain
broadcast as context. SP400-minus-SP500 spreads require each sleeve to pass
separately, so union coverage cannot hide a poorly covered sleeve.
Coverage/exclusion diagnostics remain outside X.

## Daily benchmarks and relative features

Use historical membership on **each daily session**, including warmup dates.
Apply exhaustive requests and the 80% gate separately to each day's union and
sleeve returns; the retained sensitivity uses 90%. A missing daily membership
slice cannot be filled from a month-end list. Save daily benchmark diagnostics
separately from monthly feature diagnostics.

Relative beta, residual, idiosyncratic-volatility and relative-return features
still require every stock/benchmark pair in their exact calendar windows. One
failed benchmark day can make the complete window unavailable. Identity-route
and same-sleeve continuity requirements remain. **80% of stocks is not 80% of a
feature's time window**; individual temporal windows stay complete.

## Unchanged inputs and model population

- Paper15 retain `signal_qualified_common_momentum_v1`, one common five-momentum
  population, sample-SD rules and exhaustive admitted-original requests.
- All 298 individual daily technical definitions and exact histories stay
  unchanged. All 728 definitions and 715 eligible names remain in the dictionary.
- Added-feature gaps do not remove otherwise paper-qualified prediction rows.
  Training-only imputation and constant/all-missing handling remain under
  `train_partition_median_missing_indicator_v1`. No preprocessing is fitted here.
- Monthly outcomes and daily execution/valuation quotes remain separate from
  signal-only predictors. Preserve raw/core daily, annual raw warmup, past-only
  histories, dated cohorts and calendars as independent artifacts.

## Verified role promotion

`configs/context_panel_roles.json` pins the already verified 80% primary and
90% sensitivity runs, archived recipes and comparison evidence. The master uses
`dmv2.context_panel_roles.read_context_panel_roles(catalog)` and exposes primary
features separately from the sensitivity and raw daily data. Original artifact
paths and hashes remain unchanged. This configuration is the active role selector;
`outputs/full_feature_panel/LATEST.json` and
`outputs/context_sensitivity/LATEST_80.json` remain immutable historical build
pointers and do not select the current primary/sensitivity roles. The historical comparison keeps its original
**90% → 80%** orientation; positive gains mean additional values released at 80%.
See [the comparison and promotion record](CONTEXT_SENSITIVITY_80.md).

Record original/observed counts, admitted/requested counts, exhaustive-request
status, group, date, policy and withheld reason. Coverage among paper-qualified
rows is 96.27% at 80% versus 71.06% at 90% over 2010–2023, and 92.09% versus 33.57%
in 2010–2015. Both panels have observations somewhere for all 402 eligible context
names. “Ever finite” does not admit a feature to every training fold. Do not use
full-history coverage to select or fit transformations for an earlier forecast.

Missingness may be systematic. Before 2016, newly released groups omit a higher
S&P400 share than their observed contributors; many omitted slots lack the
profiles needed to characterize them. The threshold does not certify
representativeness, identity, adjustment vintage, terminal returns or historical
knowledge-time PIT. Economic accounting and backtest readiness remain unverified.
Notebook Run All verifies saved evidence without starting a build, preprocessing
fit, model fit, backtest, protected-2024+ read or cloud mutation.
