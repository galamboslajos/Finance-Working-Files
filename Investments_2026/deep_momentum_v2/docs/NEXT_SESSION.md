# Deep Momentum V2 next session checklist

Prepared October 15, updated October 16, 2026. The aim is a complete, comparable
daily backtest across the four model arms, followed by the planned rolling
annual tests. This is a work plan; no overnight computation is scheduled.

## Periods and current state

| Stage | Training outcomes | Prediction and backtest months | Status |
| --- | --- | --- | --- |
| Development pilot | 2010–2017, one fit per arm | January 2018–December 2019, 24 months | Existing results require data correction |
| Planned rolling evaluation | Previous 10 completed years, annual refit | January 2020–December 2023, 48 months | Model and strategy evaluation not started |

The saved development rerun executed 64 scenarios: four model arms × two class
baskets × two stop variants × four cost assumptions. Twelve have complete price
accounting; 52 withhold full-period performance. The known old-BBBY → Overstock
mapping error also affects prior inputs and prevents interpreting the completed
paths as validated strategy performance. All 883 tests and the independent
64-scenario accounting replay passed; these checks do not repair security identity.

Start with [the current results and limitations](ACTION_BACKTEST_RESULTS.md),
[the action protocol](ACTION_BACKTEST.md), and
[the prepared owner data request](REMAINING_BACKTEST_DATA_REQUEST.md).

## Ordered tasks

1. **Correct the historical BBBY identity route.**

   - [x] Verify old Bed Bath's ISIN `US0758961009` and trace the published wrong
     route. The row labelled as an ISIN lookup actually queried `BBBY` and shares
     the native ID and response hash of the historical Overstock ISIN lookup.
   - [ ] Obtain the correct historical provider ID, original response and dated
     issuer/security evidence, plus matched history including the 2008–2015
     warmup. The verified old ISIN alone is not a repaired native price route.
     See [the prepared lookup request](BBBY_LOOKUP_REQUEST.md); not sent.
   - [ ] Check the same ticker-reuse failure pattern in the other admitted routes
     using identity metadata, without selecting names by later quote availability.
     The October 16 query-label diagnostic flagged five other source IDs for
     provenance review; a ticker literal under an ISIN tag alone does not prove
     a wrong issuer. No additional identity decision was changed.
   - [x] Save a versioned metadata amendment and affected-date/key inventory.
     [The October 16 amendment](BBBY_IDENTITY_AMENDMENT.md) disables the known wrong
     route, retains all 108,652 development cohort rows and source flags, and
     leaves the replacement explicitly unresolved. It does not rebuild financial
     inputs or certify historical knowledge time. This targeted repair does not
     reopen full identity certification as a prerequisite for all development.

2. **Resolve the remaining execution and corporate-action gaps.**

   - [ ] Obtain actual 2019 Ensco Rowan successor quotes for RDC entitlements,
     CUSIP `G3166L100`, on the correct post-consolidation share basis.
   - [ ] Resolve Dean Foods suspension/DFODQ identity, dated quote coverage and a
     stated halt valuation/exit convention.
     All eleven available November 8–December 2 bars are already cached; actual
     OTC bars remain missing on November 18–21. November 12 is a documented halt.
   - [x] Review ACC, K and SO on August 12, 2019 against independent raw SIP OHLC
     and valid adjustment controls. The [separate three-key amendment](DATA_REPAIRS_20261016.md)
     uses preceding valid units, retains original prices and does not widen
     tolerances. This prepares corrected inputs; no backtest has been rerun.
   - [ ] Consume that amendment in the separately versioned corrected workflow
     and audit affected daily-derived predictor snapshots before new predictions.
   - [ ] Pin source objects and review evidence. Preserve original bars and
     selected names; use separate corrections rather than stale fills, invented
     exits, replacements or dropping stocks. The [narrowed October 16 owner request](REMAINING_BACKTEST_DATA_REQUEST_20261016.md)
     has not been sent; the user can forward it if the data are not already available.

3. **Rebuild the affected development inputs and models under a new version.**

   - [x] Trace potential dependencies through raw history/warmup, daily features,
     monthly snapshots, shared normalization, target ranks and RET inputs. The
     metadata audit identifies 120 direct BBBY rows: 96 training examples and
     24 validation predictions. Shared calculations span 108,652 original rows,
     including 98,626 originally Paper15-qualified rows. These are recompute
     populations, not measured numerical changes. No financial rebuild occurred.
   - [ ] Reconcile monthly outcomes with the reviewed terminal-event treatment;
     any change in return semantics requires an explicit versioned contract.
     Preserve missing targets and ties; never impute labels.
   - [ ] After a bounded execution plan is agreed, rebuild affected development
     inputs through 2019, fit the four arms on corrected 2010–2017 outcomes and
     regenerate all 2018–2019 predictions. Fit preprocessing only on training rows.
   - [ ] Preserve the old runs. Recompute selections using identical rules in the
     corrected experiment and report changed keys and causes. Preserve selection
     rules; do not force stale predictions or secretly patch their holdings.

4. **Complete the 24 month development backtest before expanding.**

   - [ ] Rerun all four arms on both separate class 1/10 and 2/9 baskets, ten
     names per direction, 50% long / 50% short, with and without the 10% per-asset
     stop, at 0/5/10/25 bp per one-way traded notional.
   - [ ] Audit every selected/held valuation and encountered action. Aim for all
     64 complete paths; unresolved cases stay visible and their full-period
     performance remains withheld. Arithmetic completion and identity review
     are separate acceptance checks.
   - [ ] Independently reconcile cash, raw-share entitlements, successor units,
     fills, stop thresholds, net market-order costs and NAV. Keep daily source
     data separate from monthly prediction tables.

5. **Freeze the rolling evaluation before reading its performance.**

   | Test year | Training outcome years | Prediction months |
   | --- | --- | ---: |
   | 2020 | 2010–2019 | 12 |
   | 2021 | 2011–2020 | 12 |
   | 2022 | 2012–2021 | 12 |
   | 2023 | 2013–2022 | 12 |

   - [ ] Retain historical S&P 500+400 membership, RF/XGBoost × Paper15/expanded,
     the 80% primary context rule, 90% sensitivity role and the no-market-cap
     adaptation. Use annual classifier refits and monthly forecasts.
   - [ ] Freeze estimator settings, annual training-only preprocessing, completed
     label cutoffs, tie rules, execution timing, stop and cost conventions before
     evaluating 2020. Treat the repeatedly inspected 2018–2019 period as development.
   - [ ] Prepare the measured fit/read/output budget. The proposed primary model
     workload is 16 annual fits: four arms × four test years. A 90% model
     sensitivity, tuning or the paper's 100-fit ensemble is a separate workload.
   - [ ] Obtain the applicable execution authorization in the next session before
     new fits or 2020–2023 model/strategy evaluation. This checklist does not
     launch them. Keep 2024+ financial values protected.

6. **Add RET as the primary score when its corrected history qualifies.**

   - [ ] Re-audit all ten realized-class payoffs for each of the preceding 120
     completed calendar months at every signal. Use no future outcomes and do
     not shorten the gate. The prior panel's nominal January 2020 availability
     must be rechecked after data corrections.
   - [ ] Feed the same dated payoff vector and eligible cohort to both model
     families. Preserve the separate class 1/10 and 2/9 portfolios.
   - [ ] Specify RET basket breadth explicitly in the larger protocol; ten per
     direction is the proposed continuation of the practical portfolio size,
     distinct from the paper's decile breadth.

7. **Run and join the 48 monthly rolling forecasts into daily portfolios.**

   - [ ] Implement the larger period in a separately versioned worker/protocol;
     retain the current workers' explicit 2018–2019 guards and saved evidence.
   - [ ] Maintain continuous holdings, cash and cost/stop basis across annual
     model boundaries. Do not liquidate at every December close merely to join
     independently run years. Rebalance at the defined next monthly open and
     liquidate only at the final authorized terminal date.
   - [ ] Audit new selected holdings and actions under the same rules. Never
     filter them using future daily completeness. Missing exposures remain in
     coverage and withhold the affected continuous performance.

8. **Deliver a report with explicit periods and comparable coverage.**

   - [ ] Show calendar-year results and the compounded 2020–2023 result separately
     from the 2018–2019 development results. Label every date range and month count.
   - [ ] Report total return, CAGR, annualized arithmetic mean, annualized daily
     volatility/Sharpe, maximum drawdown, turnover, costs and stop statistics.
     Label return/DD period measures separately from annualized statistics.
   - [ ] Show gross and net results, equity/drawdown curves, long/short attribution,
     and prediction/identity/valuation coverage for every predefined arm. Keep
     zero cash interest/funding/borrow and idealized stop-fill assumptions explicit.
   - [ ] Report incomplete cases alongside completed ones; do not rank only the
     paths whose data happened to pass. Save source pins and independent checks.

## Starting point for the next session

Begin with tasks 1 and 2. While external data are pending, prepare the versioned
dependency inventory and larger execution plan. With a 2010 initialization and
ten training years, the currently planned full rolling test is **2020–2023,
48 months**.
