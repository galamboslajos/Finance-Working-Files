# Executable return over drawdown: label engineering audit

**USER DIRECTION,22October2026:** implement a monthly target using executable daily cumulative return divided by within-holding-period drawdown. This is a V2 extension; it is not the Han–Qin paper's original calendar-return label.

## Frozen definition

For each historical member and each independent direction:

`target = cash_return / max(maximum_drawdown, 0.01)`

Both terms reuse the same independently audited **unstopped** unit-notional path: first trading open after the signal through the following rebalance open. Return is terminal cash P&L divided by initial capital. Drawdown is the positive maximum peak-to-trough decline divided by the running peak, with initial NAV and peak equal to one. The path observes opens and closes; it does not measure every intraday peak or trough. Corporate-action accounting remains the inherited audited research convention.

The1% denominator floor is a fixed engineering choice. It was frozen before reading the new ratios and was not selected using prediction or backtest performance. Long and short paths are calculated separately. Target costs are zero; trading costs and the10% asset stop remain execution alternatives. No stock or month is discarded to improve this candidate.

This is a **literal signed ratio**. A larger drawdown can make an otherwise identical negative return score less negative. Also, when a losing path ends at its maximum drawdown from initial capital, `DD = -R` and the ratio is exactly−1. Different cash losses then share one label. Neither behavior is repaired by the1% floor. The worker records these properties explicitly; this audit does not establish that the target is a desirable loss function.

## Classes, timing and payoff interface

Class1 is best and class10 worst, using the existing exact descending midrank rule. Exact ties stay together; no jitter or forced class balance is allowed. Classes require at least ten observed valid paths and exhaustive original requests. Missing outcomes remain present with null labels; historical membership and original prediction eligibility do not depend on future quote availability.

An outcome becomes observable only at its actual next-open exit. Annual training coverage uses the preceding ten calendar years and purges outcomes unfinished at the annual cutoff. In particular, the immediately preceding December outcome is not usable at a December month-end refit because its January exit has not happened.

The payoff interface retains120 consecutive completed calendar outcomes ending in the month before the signal month. All120×10 class cells are required; an empty class or missing month withholds the whole vector. It preserves **two separately named means** within the new ratio-defined classes:

- `expected_return_over_drawdown`: probabilities weighted by historical mean ratios.
- `expected_executable_cash_return`: probabilities weighted by historical mean executable cash returns.

Neither score is silently renamed as the archived original RET. Numeric monthly means are inspected only for the requested direction and completed120-month window. Future and opposite-side means remain untouched by a historical payoff request.

## Audit results

**VERIFIED LOCAL:** the complete168-month audit and independent reconstruction passed. The original152,307 stock-months become304,614 directional rows. Per direction,140,931 rows are prediction-eligible and139,414 have ratio labels (**98.92%**). The remaining1,517 remain visible:630 unavailable source paths and887 December2023 outcomes whose protected January2024 exit was not observed. No ratio adds an extra missing numerical label relative to the audited unstopped cash-return paths.

| Audit property | Long | Short |
|---|---:|---:|
| Observed ratio labels |139,414|139,414|
| Drawdown floor active |713|373|
| Negative cash returns |60,392|78,840|
| Exact−1 ratio labels |907|3,060|
| Largest exact tie in one month |125|264|
| Completed months with empty classes |0|2|
| Classes changed versus same-path cash-return classes |66,525 /139,414 (47.72%)|67,784 /139,414 (48.62%)|
| Complete120-month payoff vectors for48 nominal2020–2023 signals |47|0|

The long January2020 state is unavailable because its required history starts in December2009, before this audit's2010 start. The remaining47 long states are complete. This is the same lagged-payoff clock that gives the matched comparison a first tradable month of February2020.

For shorts, **September2010 has class10 empty**:162 observations tie exactly at−1 and share class9. **June2019 has classes8 and10 empty**:264 of868 observed short outcomes (30.41%) tie exactly at−1 and share class9. All2020–2023 short payoff windows include June2019, so all48 vectors are withheld under the unchanged120×10 rule. The underlying observed labels remain present. This is a structural class/payoff incompatibility of the literal ratio, distinct from the previously deferred quote cases. No class is populated artificially, no month skipped and no gate shortened.

Completed fit-row counts are identical for long and short:97,253 for the2020 window;98,732 for2021;99,693 for2022;100,681 for2023. Actual next-open completion purges874,876,873 and884 signal-eligible preceding-December rows respectively. Availability counts and incomplete boundary counts overlap; they must not be added as disjoint missingness categories.

The producer took21.63seconds and0.41GiB peak RSS. Independent verification took15.75seconds and0.53GiB peak RSS. Each decoded34.57MiB of declared source column blocks; independent derived reads were27.46MiB. Limits were64MiB source per pass,128MiB derived verification/output,4GiB RAM and600seconds. Numerical December2023 columns were never requested. **62 invented-data tests passed:**27 pure-worker,10 controller,19 independent-checker and6 passive-reader tests. The checks establish the new formulas, exact ranks, original metadata/denominators, clocks, summaries, dual payoff means and completion purges; they do not re-certify underlying provider data.


## Implementation and use

The pure worker is `src/dmv2/return_dd_targets.py`; the explicit metadata-plan/materialization controller is `src/dmv2/return_dd_audit.py` with `scripts/run_return_dd_audit.py`. The independent checker is `scripts/verify_return_dd_audit.py`, which reconstructs expected formulas, ranks, cohorts, clocks, sufficient statistics and payoff vectors without importing the production target worker.

The frozen contract is `configs/return_dd_target_protocol_20261022.json`. The private result is `outputs/return_dd_target_audit/2010-2023-20261022-v1`; the sealed source-read plan is in its sibling `plans/` directory. Monthly original identity/admission metadata is preserved and fingerprinted. Numerical target fields, floor flags and diagnostics are outcomes/audit data and must never enter predictors.

Master notebook section32 calls only `dmv2.return_dd_reporting.read_return_dd_audit`. It authenticates the independent proof and opens small JSON aggregates, with no Parquet/model import. Run All cannot rebuild labels, prepare matrices, fit estimators or replay portfolios. A fresh clone without private receipts prints that saved evidence is unavailable.

The explicit label build requires a fresh output and plan path; no automatic retry, overwrite or resume is implemented. The source projection is sealed from Parquet footers before decoding. Full projected column blocks are charged even if metadata filters later remove other target families. Integrity hashing is distinct from financial-value decoding. Raw/provider data, individual labels, numerical receipts and evidence plans remain ignored; Git receives original code, protocols, tests and this aggregate methodological note.

## Next research boundary

This phase creates and verifies labels only: **zero fits, zero preprocessing fits, zero predictions and zero portfolio replays**. Existing RF/XGBoost results remain historical comparators. The same715 predictors, cohorts, daily history, basket sizes,50/50 allocation, stop rules and costs remain fixed.

This candidate is label-engineered and independently verified, but it is **not ready for a two-sided ten-class probability-to-payoff run** under the unchanged gate. A concrete next option is separate long/short RF regressors predicting the continuous ratio directly, alongside the existing cash-return/classifier controls. This avoids empty class payoff cells and is a proposed next experiment, not a run launched here; the signed-loss ranking issue would still remain. Alternatively, any changed loss transform or class/payoff convention must be declared as a new target/protocol version before fitting. Do not silently split exact ties or impute nonexistent class payoffs.

Model settings, the comparison score and a bounded run must be frozen separately before fitting this target. The known execution gaps remain deferred under the existing user decision; they do not block independent label research. This formula audit does not certify source identity, corporate actions, investability, knowledge-time completeness or a bias-free price history.2024+ financial values remain protected; December2023's next-open outcome stays unavailable.
