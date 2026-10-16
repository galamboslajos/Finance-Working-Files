# Historical BBBY identity amendment and rebuild dependencies

**USER DIRECTION, 2026-10-16:** investigate and fix the historical BBBY identity
route, establish the affected data/model dependencies, preserve the original
runs, and keep later financial evaluation separate. This amendment disables a
verified wrong route in a new metadata view. **The replacement Intrinio security
ID and its historical price series remain unresolved.** It does not claim that
corrected features or models have already been rebuilt.

The private, hash-verified evidence/review package is:

`data/derived/identity_amendments/old-bbby-native-conflict-20261016-v1/`

Raw evidence, metadata panels and generated audit outputs remain outside Git.
The original source releases, configuration recipes, panel versions, pilot fits,
probabilities, selections, backtests and original output pointers are preserved.

## What is verified, and what remains unresolved

VERIFIED LOCAL METADATA: the pinned Siblis identity metadata identifies the
historical Bed Bath & Beyond security by **ISIN `US0758961009`**. Independent
SEC-hosted issuer/class evidence associates it with **CUSIP `075896100`**; that
CUSIP verification is separate from the Siblis field evidence. See the linked
SEC source and precise evidence limits in the
[historical lookup request](BBBY_LOOKUP_REQUEST.md). The old security is distinct
from ISIN `US6903701018`, the historical Overstock lineage associated with the
later reused BBBY ticker. Ticker agreement cannot establish continuity.

The two reviewed published routes are:

| Original source ID | Unified ID | Invalidated native route |
| --- | --- | --- |
| `siblis_ticker_company_00326c7d9ad13903e5e3` | `equity_us_883f1d58c4dcf36a3092bd1f` | `sec_ydbQnj` |
| `siblis_isin_c297a5f81a23c9f90503` | `equity_us_aacab20bc16f70ab265fd1e1` | `sec_ydbQnj` |

The lookup evidence exposes a concrete query-literal bug. The historical
ISIN-keyed route records `query_identifier_type="isin"`, but its actual
`query_identifier` is **`"BBBY"`**, and the retained source path contains
`query_identifier=BBBY-...`. The retained request evidence shows a ticker literal,
not an exact request for historical ISIN `US0758961009`. The old ticker route,
that mislabeled old-ISIN route, and the separate lookup using `US6903701018`
record the same native ID and retained response hash. Original response bodies
are unavailable locally, so matching declared hashes are not independent
verification of those payload bytes. A label saying “ISIN” does not establish
that the old ISIN was sent. This invalidates the historical native join without
supplying its replacement. The [forwardable lookup request](BBBY_LOOKUP_REQUEST.md)
remains a draft and has not been sent.

OPEN: independently verify the old issuer's actual Intrinio security ID,
identifier validity/continuity, dated provider lookup evidence and the required
historical prices. Do not substitute a modern ticker match, guessed native ID,
zero return, stale terminal quote or convenient price series. Legal identifier
validity and historical knowledge-time availability are not certified by this
review.

## The additive amendment

`src/dmv2/identity_amendment.py` reads the pinned declaration and exact local
lookup proof. `apply_identity_amendment` returns a detached metadata view which:

- Retains every original source/date constituent and its original route,
  statuses and admission evidence in audit fields.
- Preserves the reviewed historical ISIN while setting the known wrong
  effective native route to null and marking
  `historical_identity_conflict` / replacement unresolved.
- Reapplies the existing provisional identity qualifier to the complete dated
  original cohort. It does not select members using later prices or returns.
- Separates old `requested` and Paper15/prediction flags into original-reference
  fields. An identity amendment alone cannot certify new request completeness
  or recomputed feature eligibility.

`amend_universe_snapshot` supplies the corresponding detached snapshot adapter.
`require_corrected_price_route` deliberately refuses a corrected-price rebuild
while replacement evidence remains unresolved. `read_saved_identity_review`
verifies the saved small review and artifact hashes; it does not decode feature,
price, label or model panels.

The current new view covers the existing **2010–2019 outcome-year development
panels**. Its logical analysis bound is separate from any legal validity interval.
A broader metadata audit can list later dependencies without amending or reading
later financial panels.

## Exact development scope

The first four-arm pilot fits on 2010–2017 completed outcomes and validates on
2018–2019. Each outcome month uses its preceding month-end signal.

| Development segment | Monthly cohorts / original BBBY membership rows |
| --- | ---: |
| Fit-period outcomes 2010–2017 | 96; all 96 confirmed in the saved supervised fit mask |
| Validation outcomes 2018–2019 | 24 predictions; all 24 have available labels |
| Combined outcomes 2010–2019 | **120** |

The exact signal interval is **2009-12-31 through 2019-11-29**. Those panels
retain **108,652 original stock-month rows**, of which **98,626** were
Paper15-qualified in the reference version. Saved pilot metadata independently
confirms that all **96 BBBY training rows entered the supervised fit mask**, and
all **24 validation rows received predictions and have available labels**. These
were actual pilot exposures to the wrong route, not merely membership candidates.

Do not confuse this with a date predicate ending December 31, 2019 over every
saved partition. That wider predicate also includes the December 2019 signal
stored in the January 2020 outcome partition and therefore has **121** BBBY keys.
The current development amendment remains the exact 120-cohort view.

## Why the repair reaches beyond BBBY's own rows

The expanded predictor bank has **715 eligible columns** with unchanged formulas:
15 paper inputs, 298 native daily technicals and 402 context variables. A changed
route or admission can change another stock's normalized/context values even
when that other stock's raw prices remain identical.

| Eligible family | Columns | Potential dependency scope |
| --- | ---: | --- |
| Paper momentum/state | **15** | All common-qualified stocks at an affected signal: five z-scores and their shared means/SDs |
| Native daily technicals | **298** | BBBY's own source/signal attachment and its exact past native inputs |
| Cross-sectional context | **310** | 155 universe columns and 155 historical-sleeve columns at that signal |
| Shared market state | **62** | Shared original-cohort aggregates across all rows at affected signal dates |
| Relative context | **30** | BBBY directly and other stocks whose trailing daily benchmark windows contain affected days |

There are 728 registered definitions in total. Ten exact rank aliases and three
structural constants remain ineligible; this amendment does not alter that
allowlist, the 80% primary coverage rule, 90% sensitivity or missingness protocol.

### Paper15

`paper_features.normalize_common_momentum` uses a single identity-admitted
population having all five exact finite momenta. Changing only one momentum
with the same population can change the corresponding z-score/mean/SD for other
stocks. Changing common-population membership can change **all 15** inputs for
other qualified stocks. Preserve sample SD (`ddof=1`), exhaustive requests and
existing degeneracy checks. Recalculate complete affected-date populations,
not only the BBBY row.

MOM1 uses exact closes at month ends t-1 and t. For h=3/6/9/12, the worker checks
every endpoint t-h through t-1, producing h-1 monthly returns. Preserve those
exact indexing and completeness rules.

### Daily technicals and cross-sectional context

The 298 daily technicals are computed by native security on a complete session
grid and then attached to source/signal keys. They require one to 503 inclusive
sessions depending on the feature; the longest dependency is the abnormal-volume
construction. A future verified replacement must supply the correct native
history through each signal, including the initial pre-2010 warmup. Quarantine
currently invalidates BBBY's attachment; it does not manufacture a replacement.
Other securities' native technical calculations are not directly changed by
that isolated join.

The 310 cross-sectional predictors use 80 technical bases, universe/sleeve
percentile ranks and median/MAD transforms, after excluding the ten aliases.
Removing an invalid observation can change ranks, robust statistics or whether
a variable clears 80%/90% original-cohort coverage. Original denominators and
exhaustive admitted-ID requests remain required. Potential dependency does not
mean every statistic must numerically change.

### Shared market state and the daily benchmark tail

Shared market state uses the actual historical daily cohort at the signal date,
with horizons 1/5/20/63/126/252 sessions. It does not keep a removed constituent in
future current-member aggregates.

Relative context uses 20/42/63/126/252 complete daily-return pairs: four
market-relative and two sleeve-relative columns per window. An altered benchmark
return on session d can remain in another stock's 252-pair window through
**251 later sessions**. No daily membership is reconstructed from monthly
snapshots; route and sleeve-continuity checks remain in force.

The 503-session/~24-month requirement is a **historical technical/base lookback**,
not a 24-month forward market-memory rule. The metadata-only closure over all
saved wrong-route history finds the final observed BBBY daily membership on
2020-06-19; the 252-pair tail can extend through session 2021-06-18, with last saved
monthly signal 2021-05-28. Restricting the logical amendment to 2019-12-31 instead
bounds that tail at session 2020-12-29, last saved signal 2020-11-30.

## Labels, RET and later model work

`monthly.build_monthly_returns` must use the corrected version's dated route,
original cohort and exact signal/next-month endpoints. Missing outcomes remain
visible; neither target returns nor replacement prices are imputed.

`targets.build_monthly_targets` ranks accepted observed returns within the same
signal-time Paper15-qualified cohort, with exact-equality midranks, class 1 best
and class 10 worst. A changed BBBY return, eligibility or observed population size
can change **other stocks' classes**. Rebuild the whole affected monthly rank and
its coverage diagnostics, not only BBBY's label.

RET uses equal-stock realized-class monthly means, then an equal-month mean of
120 complete calendar months. A changed outcome month m can affect payoff
signals m through m+119. Keep the entire ten-class grid and its full-history
gate; RET remains structurally unavailable in this 2018–2019 development test.
Do not shorten the window to bypass this identity repair.

The reference training inputs, shared classes and population statistics can all
be affected. A genuinely corrected comparison therefore requires new versioned
feature/target panels, training-only preprocessing and **fresh fits of all four
arms**: RF/Paper15, XGBoost/Paper15, RF/expanded 715 and XGBoost/expanded 715.
Missing indicators, medians and training-only constant/all-missing schema rules
must be recomputed from their training partition. Preserve the existing split,
PIT safeguards, daily data separation and later rolling-ten-year design. No
model was fitted or refitted as part of this amendment.

The old runs remain reproducible historical evidence of their original inputs.
Reusing their predictions in a narrow accounting sensitivity would not turn
them into identity-corrected models or validate their economic results.

## Potential bounds versus measured changes

`src/dmv2/identity_impact.py` plans dependency scopes from identifiers, observed
membership dates, index labels and independent saved session metadata. It rejects
financial/feature/label columns, deduplicates identical annual warmup overlap,
and rejects conflicting repeated routes. Counts are **potential rebuild upper
bounds**, not observations that every cell changed.

For the exact 120 development cohorts, it records **36,905,386 potentially affected
predictor cells** across the original cohort scopes. This includes nonlocal
shared statistics; it is not 36.9 million measured errors or a replacement dataset.
Separate private reports preserve the 121-key date-bound scope and 126-key entire
saved BBBY history so neither is mistaken for the current 120-key amendment.

The initial two metadata scopes took 16.77 seconds with 353.5 MiB peak RSS; the exact
120-cohort scope took 6.41 seconds with 297.3 MiB. All inputs remained unchanged and
execution stayed within the 2 GiB / 120-second / 10 MiB output limits. The planner has
22 structural tests; static dictionary exports and a synthetic normalization
proof support its dependency rules.

**2020+ caveat:** forward closure uses only dates, source identities and index
labels after 2019. It does not read later prices, feature values, labels,
probabilities or payoff values, and does not rebuild or evaluate those years.
The private scope count named `metadata_rows_after_financial_boundary` classifies
stored signal dates; next-month target completion is a separate clock. A
December 2019 signal can therefore identify a January 2020 label dependency
without authorizing access to that label. No protected 2024+ financial data,
new model fitting or new backtest is part of this review.
