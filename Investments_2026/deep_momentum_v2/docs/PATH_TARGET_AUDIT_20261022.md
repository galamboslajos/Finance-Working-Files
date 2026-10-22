# Daily-path target audit — 22 October 2026

**COMPLETE BOUNDED ENGINEERING AUDIT; NO MODEL FITS.**
The separate [engineering contract](PATH_TARGET_ENGINEERING_20261022.md)
retains the original endpoint labels and implements directional executable
returns and `return - maximum_drawdown` utility, with unstopped and 10% asset-stop
variants. This is a V2 extension. No model/backtest performance improvement is
claimed. All old source inputs, runs, labels and selections remain unchanged.

## Measured population and coverage

The fixed audit uses full original cohorts for January and October 2020, chosen
before financial materialization: an initial execution window and a known
unresolved-source window. It is not a representative random sample or a full
2010–2023 training-history audit.

| Outcome month | Original members | Signal-time prediction eligible | Long unstopped complete | Long stopped complete | Short unstopped complete | Short stopped complete |
|---|---:|---:|---:|---:|---:|---:|
| January 2020 |909|876|874|874|874|875|
| October 2020 |910|878|875|875|875|875|

All 1,819 original stock-months remain, including 65 signal-time ineligible rows.
There are 7,016 requested directional/stop episodes: 6,997 complete and 19 unavailable
(**99.729% conditional path coverage**). The 19 are 15 missing-quote and 4
source-quality episodes, not 19 different stocks. Another 260 episode rows retain
signal-time ineligibility; these are not missing-price failures. Stopped cash
can explain higher short-stop coverage without filling later absent quotes.

Two target families per episode produce 14,552 original target rows, including
13,994 observed target values. Every monthly group passes exhaustive eligible
requests. Missing outcomes remain null; no stock was replaced and no future
path-completeness screen altered the prediction population.

## Endpoint versus executable-return alignment

For the long, unstopped control there are 1,749 stock-months with both the original
close-to-close label and the new next-open-to-next-open return:

- Mean absolute difference: **0.779 percentage points**.
- Maximum absolute difference: **21.670 percentage points**.
- **346 / 1,749 = 19.78%** change relative-return class.
- One stock-month has an original observed endpoint label and an unavailable
  executable path; no stock-month has only an executable return.

These are conditional label-definition comparisons, not portfolio returns,
prediction accuracy or evidence that the alternative target is better. The
execution interval differs from the original calendar endpoint interval.

## Stop ties expose a scoring design issue

October 2020 short stopped executable-return classes have no class 9. This is
intentional exact-midrank behavior, not a forced rebalance of class sizes.
The observed near-minus10% stop-return mass contains exact tie groups of 207
and 138 members, each occupying one class. Their numerical returns differ only
by floating-point roundoff in the inherited cash-ledger/unit transformation.
The existing exact-equality policy is preserved in this audit; no rounding,
jitter, class padding or alternative payoff mean was introduced after results.

Thus the strict 120-month/all-ten-class payoff rule can withhold a new stopped
return-class score for structural tie reasons even when source coverage is high.
Measure this across history and define a prospective treatment before fitting
that scoring variant. Do not weaken the original RET gate or fabricate an empty
class payoff. A continuous directional predictor is a possible later design,
not implemented or selected by this audit. Utility classes populate all ten
bins in these two months; that is not a full-history guarantee.

## Verification and limits

All 182 selected new/regression tests pass, including synthetic signed cash/stock
entitlements, missing successors, long/short stops, next-open completion,
negative short wealth, exact ties, value fingerprints and original denominators.
The independent checker recomputed returns, drawdowns, utility and exact ranks
for 14,552 target rows (13,994 observed values), all 16 class groups and their
scheduled endpoint grids. Its status is `PASS_BOUNDED_FORMULAS_AND_CLOCK`.
It did not certify source prices, historical identities or legal entitlements.

The audit consumed 9,275,913 projected compressed column bytes, completed in
257.91 seconds and peaked at 0.669 GiB process RSS. Source files were hash-checked
unchanged before/after. No model, preprocessing fit, inference, portfolio replay
or protected 2024+ financial read was performed. The one-asset accounting calls
are target engineering, not new strategy performance results.

Private evidence is under `outputs/path_target_audit/20261022-v1/`, including
original episode/target ledgers, the reviewed plan, audit and independent receipt.
Raw data, numerical ledgers and receipts remain outside Git. The master reads
saved summary metadata and verifies artifact bytes without decoding full panels.

## Next execution gate

The new core can represent 2010–2023 labels, but this initial accounting adapter
is deliberately restricted to 2020–2023. Extend/test its training-year support
and audit every original training stock-month before a full comparative refit.
Keep actual completion cutoffs, the protected December 2023 next-open endpoint,
missing/terminal reasons and original denominator reports. Then freeze the
small RF-expanded comparison, directional score/basket mapping, stop-tie policy,
common evaluation period and compute budget. The currently enabled fit budget
is **zero**. Universal identity certification and completion of every existing
comparator portfolio are not prerequisites for this engineering progression.
